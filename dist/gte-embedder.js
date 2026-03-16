/**
 * GTE-base embedding module for high-quality semantic search.
 *
 * Uses Xenova/gte-base (768-dim) via @xenova/transformers ONNX runtime.
 * Independent from Smart Connections' bge-micro-v2 (384-dim) embeddings.
 *
 * Block splitting strategy (based on actual Vault note analysis):
 * 1. SYNC blocks → atomic, never split
 * 2. H2 sections → default split unit (SYNC within H2 is extracted, rest of H2 kept)
 * 3. H2 > 2000 chars + H3 ≥ 2 → split by H3
 * 4. YAML frontmatter → separate metadata block (excluded from note-level vector)
 * 5. < 30 chars sections → skip
 * 6. Note-level vector: length-weighted mean of content blocks (excluding YAML)
 */
import * as fs from 'fs';
import * as path from 'path';
let pipeline;
let extractor;
const GTE_MODEL = 'Xenova/gte-base';
const GTE_DIM = 768;
const INDEX_FILENAME = 'gte-index.json';
// --- Content Parsing ---
/**
 * Parse a note into blocks respecting SYNC markers, H2/H3 hierarchy, and YAML.
 */
function parseNoteIntoBlocks(content) {
    const blocks = [];
    const lines = content.split('\n');
    // 1. Extract YAML frontmatter
    let bodyStart = 0;
    if (lines[0] === '---') {
        const yamlEnd = lines.indexOf('---', 1);
        if (yamlEnd > 0) {
            const yamlContent = lines.slice(0, yamlEnd + 1).join('\n');
            if (yamlContent.length >= 30) {
                blocks.push({
                    key: '__yaml__',
                    type: 'yaml',
                    content: yamlContent,
                    heading: '__yaml__',
                });
            }
            bodyStart = yamlEnd + 1;
        }
    }
    // 2. Parse body: identify SYNC blocks and heading structure
    const bodyLines = lines.slice(bodyStart);
    const segments = extractSegments(bodyLines);
    // 3. Convert segments into blocks with adaptive splitting
    // Track heading counts for dedup
    const keyCount = new Map();
    for (const seg of segments) {
        if (seg.type === 'sync') {
            if (seg.content.length >= 30) {
                blocks.push({
                    key: `SYNC:${seg.syncId}`,
                    type: 'sync',
                    content: seg.content,
                    heading: seg.heading || `SYNC:${seg.syncId}`,
                });
            }
        }
        else if (seg.type === 'h2') {
            // Check if H2 section needs H3 splitting
            if (seg.content.length > 2000 && seg.h3Children && seg.h3Children.length >= 2) {
                // Emit pre-H3 content as H2 block
                if (seg.preH3Content && seg.preH3Content.length >= 30) {
                    const key = dedupKey(seg.heading, keyCount);
                    blocks.push({
                        key,
                        type: 'h2',
                        content: seg.preH3Content,
                        heading: seg.heading,
                    });
                }
                // Emit each H3 child
                for (const h3 of seg.h3Children) {
                    if (h3.content.length >= 30) {
                        const key = dedupKey(`${seg.heading} > ${h3.heading}`, keyCount);
                        blocks.push({
                            key,
                            type: 'h3',
                            content: h3.content,
                            heading: h3.heading,
                        });
                    }
                }
            }
            else {
                // Keep H2 section intact
                if (seg.content.length >= 30) {
                    const key = dedupKey(seg.heading, keyCount);
                    blocks.push({
                        key,
                        type: 'h2',
                        content: seg.content,
                        heading: seg.heading,
                    });
                }
            }
        }
        else if (seg.type === 'intro') {
            if (seg.content.length >= 30) {
                blocks.push({
                    key: '__intro__',
                    type: 'intro',
                    content: seg.content,
                    heading: '__intro__',
                });
            }
        }
    }
    return blocks;
}
/**
 * Generate unique key by appending suffix if duplicate.
 */
function dedupKey(key, counts) {
    const count = counts.get(key) || 0;
    counts.set(key, count + 1);
    return count === 0 ? key : `${key} (${count + 1})`;
}
/**
 * Extract segments from body lines, respecting SYNC markers.
 *
 * Key behavior for SYNC within H2:
 *   ## Project Status
 *   manual text before...
 *   <!-- SYNC_START:id -->   ← extracted as separate sync segment
 *   ...sync content...
 *   <!-- SYNC_END:id -->
 *   manual text after...     ← included in H2 segment (not lost)
 *
 * The H2 segment contains all non-SYNC content.
 * SYNC blocks are emitted as separate atomic segments.
 */
function extractSegments(lines) {
    const segments = [];
    let i = 0;
    // Collect intro (before first H2 or SYNC)
    const introLines = [];
    while (i < lines.length) {
        if (lines[i].match(/^## /))
            break;
        if (lines[i].match(/<!-- SYNC_START:/))
            break;
        introLines.push(lines[i]);
        i++;
    }
    if (introLines.length > 0) {
        const text = introLines.join('\n').trim();
        if (text) {
            segments.push({ type: 'intro', heading: '__intro__', content: text });
        }
    }
    // Parse remaining lines
    while (i < lines.length) {
        const line = lines[i];
        // H2 heading — collect entire H2 section, extracting SYNC blocks inline
        const h2Match = line.match(/^## (.+)/);
        if (h2Match) {
            const heading = h2Match[1].trim();
            const h2ContentLines = [line]; // non-SYNC lines of this H2
            i++;
            // Collect lines until next H2
            while (i < lines.length) {
                if (lines[i].match(/^## /))
                    break;
                // SYNC block within H2 — extract as separate segment
                const syncMatch = lines[i].match(/<!-- SYNC_START:(\S+) -->/);
                if (syncMatch) {
                    const syncId = syncMatch[1];
                    const syncLines = [];
                    // Collect until SYNC_END
                    while (i < lines.length) {
                        syncLines.push(lines[i]);
                        if (lines[i].match(new RegExp(`<!-- SYNC_END:${escapeRegex(syncId)} -->`))) {
                            i++;
                            break;
                        }
                        i++;
                    }
                    // Emit SYNC segment with parent heading context
                    segments.push({
                        type: 'sync',
                        heading,
                        content: syncLines.join('\n').trim(),
                        syncId,
                    });
                    continue;
                }
                h2ContentLines.push(lines[i]);
                i++;
            }
            const fullContent = h2ContentLines.join('\n').trim();
            // Parse H3 children within this H2 (from non-SYNC content)
            const h3Children = [];
            const preH3Lines = [];
            let currentH3 = null;
            for (const sLine of h2ContentLines.slice(1)) {
                const h3Match = sLine.match(/^### (.+)/);
                if (h3Match) {
                    if (currentH3) {
                        h3Children.push({
                            heading: currentH3.heading,
                            content: currentH3.lines.join('\n').trim(),
                        });
                    }
                    currentH3 = { heading: h3Match[1].trim(), lines: [sLine] };
                }
                else if (currentH3) {
                    currentH3.lines.push(sLine);
                }
                else {
                    preH3Lines.push(sLine);
                }
            }
            if (currentH3) {
                h3Children.push({
                    heading: currentH3.heading,
                    content: currentH3.lines.join('\n').trim(),
                });
            }
            segments.push({
                type: 'h2',
                heading,
                content: fullContent,
                h3Children: h3Children.length > 0 ? h3Children : undefined,
                preH3Content: preH3Lines.join('\n').trim() || undefined,
            });
            continue;
        }
        // Top-level SYNC block (not inside H2)
        const syncMatch = line.match(/<!-- SYNC_START:(\S+) -->/);
        if (syncMatch) {
            const syncId = syncMatch[1];
            const syncLines = [];
            while (i < lines.length) {
                syncLines.push(lines[i]);
                if (lines[i].match(new RegExp(`<!-- SYNC_END:${escapeRegex(syncId)} -->`))) {
                    i++;
                    break;
                }
                i++;
            }
            segments.push({
                type: 'sync',
                heading: `SYNC:${syncId}`,
                content: syncLines.join('\n').trim(),
                syncId,
            });
            continue;
        }
        // Skip orphan lines outside of sections
        i++;
    }
    return segments;
}
function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
// --- Utility ---
function contentHash(text) {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
        const char = text.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash |= 0;
    }
    return `${text.length}:${hash}`;
}
/**
 * Length-weighted mean of vectors.
 * Longer blocks contribute proportionally more to the note-level vector.
 * Normalizes result to unit length for cosine similarity.
 */
function lengthWeightedMean(vecs, lengths) {
    if (vecs.length === 0)
        return [];
    const dim = vecs[0].length;
    const totalLen = lengths.reduce((s, l) => s + l, 0);
    if (totalLen === 0)
        return vecs[0];
    const result = new Array(dim).fill(0);
    for (let v = 0; v < vecs.length; v++) {
        const weight = lengths[v] / totalLen;
        for (let d = 0; d < dim; d++) {
            result[d] += vecs[v][d] * weight;
        }
    }
    // Normalize to unit length
    let norm = 0;
    for (let d = 0; d < dim; d++)
        norm += result[d] * result[d];
    norm = Math.sqrt(norm);
    if (norm > 0) {
        for (let d = 0; d < dim; d++)
            result[d] /= norm;
    }
    return result;
}
// --- Main Class ---
export class GteEmbedder {
    vaultPath;
    indexPath;
    index = null;
    ready = false;
    constructor(vaultPath) {
        this.vaultPath = vaultPath;
        this.indexPath = path.join(vaultPath, '.smart-env', INDEX_FILENAME);
    }
    async initialize() {
        console.error('Loading GTE-base model (first run downloads ~250MB)...');
        const transformers = await import('@xenova/transformers');
        pipeline = transformers.pipeline;
        extractor = await pipeline('feature-extraction', GTE_MODEL, {
            quantized: true,
        });
        console.error('GTE-base model loaded successfully');
        this.loadIndex();
        this.ready = true;
    }
    loadIndex() {
        if (fs.existsSync(this.indexPath)) {
            try {
                const data = fs.readFileSync(this.indexPath, 'utf-8');
                const parsed = JSON.parse(data);
                if (!parsed.version || parsed.version < 3) {
                    console.error(`GTE index v${parsed.version || 1} detected, rebuilding with v3 (adaptive block splitting)`);
                    this.createEmptyIndex();
                    return;
                }
                this.index = parsed;
                const count = Object.keys(this.index.entries).length;
                console.error(`GTE index loaded: ${count} entries (v3, adaptive blocks)`);
            }
            catch (error) {
                console.error('Failed to load GTE index, creating new:', error);
                this.createEmptyIndex();
            }
        }
        else {
            this.createEmptyIndex();
        }
    }
    createEmptyIndex() {
        this.index = {
            model: GTE_MODEL,
            dimension: GTE_DIM,
            version: 3,
            created_at: Date.now(),
            updated_at: Date.now(),
            entries: {},
        };
    }
    saveIndex() {
        if (!this.index)
            return;
        this.index.updated_at = Date.now();
        fs.writeFileSync(this.indexPath, JSON.stringify(this.index), 'utf-8');
    }
    /**
     * Embed a single text string.
     * GTE-base max_seq_length = 512 tokens. Model truncates internally.
     */
    async embed(text) {
        if (!extractor)
            throw new Error('GTE model not initialized');
        const output = await extractor(text, { pooling: 'mean', normalize: true });
        return Array.from(output.data);
    }
    /**
     * Embed multiple texts in batches for ~1.5x speedup on M4 Pro.
     */
    async embedBatch(texts, batchSize = 10) {
        if (!extractor)
            throw new Error('GTE model not initialized');
        const allVecs = [];
        for (let start = 0; start < texts.length; start += batchSize) {
            const batch = texts.slice(start, start + batchSize);
            const output = await extractor(batch, { pooling: 'mean', normalize: true });
            for (let i = 0; i < batch.length; i++) {
                allVecs.push(Array.from(output.data.slice(i * GTE_DIM, (i + 1) * GTE_DIM)));
            }
        }
        return allVecs;
    }
    /**
     * Build/update index with adaptive block splitting.
     * Hash-based change detection at note level: if note unchanged, skip all blocks.
     */
    async buildIndex(notePaths, readContent, onProgress) {
        if (!this.index)
            this.createEmptyIndex();
        const stats = {
            notes_processed: 0,
            notes_unchanged: 0,
            blocks_total: 0,
            blocks_by_type: { yaml: 0, sync: 0, h2: 0, h3: 0, intro: 0, full: 0 },
        };
        const currentNotePaths = new Set(notePaths);
        // Remove entries for deleted notes
        for (const entryKey of Object.keys(this.index.entries)) {
            if (!currentNotePaths.has(this.index.entries[entryKey].note_path)) {
                delete this.index.entries[entryKey];
            }
        }
        for (let i = 0; i < notePaths.length; i++) {
            const notePath = notePaths[i];
            try {
                const content = readContent(notePath);
                if (!content || content.length < 30)
                    continue;
                const noteHash = contentHash(content);
                // Check if note is unchanged via __full__ entry hash
                const existingFull = this.index.entries[notePath];
                if (existingFull && existingFull.hash === noteHash) {
                    stats.notes_unchanged++;
                    stats.notes_processed++;
                    continue;
                }
                // Note changed — remove all old entries for this note
                for (const entryKey of Object.keys(this.index.entries)) {
                    if (this.index.entries[entryKey].note_path === notePath) {
                        delete this.index.entries[entryKey];
                    }
                }
                // Parse into blocks
                const blocks = parseNoteIntoBlocks(content);
                const contentVecs = []; // for note-level vector (excluding YAML)
                const contentLens = [];
                // Batch embed all blocks for this note (~1.5x vs sequential)
                const blockTexts = blocks.map(b => b.content);
                const blockVecs = await this.embedBatch(blockTexts);
                for (let b = 0; b < blocks.length; b++) {
                    const block = blocks[b];
                    const vec = blockVecs[b];
                    const entryKey = `${notePath}#${block.key}`;
                    this.index.entries[entryKey] = {
                        vec,
                        note_path: notePath,
                        block_key: block.key,
                        block_type: block.type,
                        char_length: block.content.length,
                        hash: noteHash,
                        updated_at: Date.now(),
                    };
                    stats.blocks_by_type[block.type]++;
                    stats.blocks_total++;
                    if (block.type !== 'yaml') {
                        contentVecs.push(vec);
                        contentLens.push(block.content.length);
                    }
                }
                // Note-level vector: length-weighted mean of content blocks
                if (contentVecs.length > 0) {
                    const noteVec = lengthWeightedMean(contentVecs, contentLens);
                    this.index.entries[notePath] = {
                        vec: noteVec,
                        note_path: notePath,
                        block_key: '__full__',
                        block_type: 'full',
                        char_length: content.length,
                        hash: noteHash,
                        updated_at: Date.now(),
                    };
                    stats.blocks_by_type.full++;
                    stats.blocks_total++;
                }
                stats.notes_processed++;
                if (onProgress) {
                    onProgress(i + 1, notePaths.length, notePath);
                }
            }
            catch (error) {
                continue;
            }
        }
        this.saveIndex();
        return stats;
    }
    /**
     * Semantic search at block level.
     * Excludes __full__ and __yaml__ entries (use searchNotes() for note-level).
     */
    async search(queryText, limit = 10, threshold = 0.3) {
        if (!this.ready || !this.index) {
            throw new Error('GTE embedder not initialized');
        }
        const queryVec = await this.embed(queryText);
        const results = [];
        for (const entry of Object.values(this.index.entries)) {
            // Skip note-level and yaml entries in block search
            if (entry.block_type === 'full' || entry.block_type === 'yaml')
                continue;
            const sim = this.cosineSim(queryVec, entry.vec);
            if (sim >= threshold) {
                results.push({
                    path: entry.note_path,
                    block: entry.block_key,
                    blockType: entry.block_type,
                    similarity: sim,
                });
            }
        }
        return results
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }
    /**
     * Search at note level only (using __full__ vectors).
     */
    async searchNotes(queryText, limit = 10, threshold = 0.3) {
        if (!this.ready || !this.index) {
            throw new Error('GTE embedder not initialized');
        }
        const queryVec = await this.embed(queryText);
        const results = [];
        for (const entry of Object.values(this.index.entries)) {
            if (entry.block_type !== 'full')
                continue;
            const sim = this.cosineSim(queryVec, entry.vec);
            if (sim >= threshold) {
                results.push({ path: entry.note_path, similarity: sim });
            }
        }
        return results
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }
    cosineSim(a, b) {
        if (a.length !== b.length)
            return 0;
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        const mag = Math.sqrt(normA) * Math.sqrt(normB);
        return mag === 0 ? 0 : dot / mag;
    }
    getStats() {
        if (!this.index)
            return null;
        const entries = Object.values(this.index.entries);
        const uniqueNotes = new Set(entries.map(e => e.note_path));
        const blockTypes = {};
        for (const e of entries) {
            blockTypes[e.block_type] = (blockTypes[e.block_type] || 0) + 1;
        }
        return {
            model: this.index.model,
            dimension: this.index.dimension,
            entries: entries.length,
            notes: uniqueNotes.size,
            blockTypes,
            updated_at: this.index.updated_at,
        };
    }
}
//# sourceMappingURL=gte-embedder.js.map