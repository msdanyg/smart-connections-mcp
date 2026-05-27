/**
 * Embedding module for high-quality semantic search.
 *
 * Uses onnx-community/embeddinggemma-300m-ONNX (768-dim) via @huggingface/transformers v4 ONNX runtime.
 * Migrated 2026-04-17: replaces Xenova/gte-base after A/B benchmark showed 2.3x better
 * top-5 relevance (3.9x on Korean queries) due to GTE-base anisotropic collapse on Korean.
 *
 * EmbeddingGemma requires task-specific prefixes (asymmetric):
 *   query:    "task: search result | query: " + text
 *   document: "title: none | text: " + text
 *
 * Class name GteEmbedder is kept for backwards compatibility with downstream code.
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
let AutoTokenizer;
let AutoModel;
let tokenizer;
let model;
const GTE_MODEL = 'onnx-community/embeddinggemma-300m-ONNX';
const GTE_DIM = 768;
const INDEX_FILENAME = 'embedding-index.json';
const QUERY_PREFIX = 'task: search result | query: ';
const DOC_PREFIX = 'title: none | text: ';
function l2normalize(v) {
    let s = 0;
    for (const x of v)
        s += x * x;
    const n = Math.sqrt(s);
    if (n === 0)
        return v;
    const out = new Array(v.length);
    for (let i = 0; i < v.length; i++)
        out[i] = v[i] / n;
    return out;
}
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
        // Fast path: load index only. Model is lazy-loaded on first embed() call
        // to avoid MCP init timeout (EmbeddingGemma download/warmup can exceed the harness timeout).
        this.loadIndex();
        this.ready = true;
    }
    modelLoadPromise = null;
    async ensureModelLoaded() {
        if (model && tokenizer)
            return;
        if (this.modelLoadPromise)
            return this.modelLoadPromise;
        this.modelLoadPromise = (async () => {
            console.error('Loading EmbeddingGemma-300m model (first run downloads ~150MB)...');
            const transformers = await import('@huggingface/transformers');
            AutoTokenizer = transformers.AutoTokenizer;
            AutoModel = transformers.AutoModel;
            tokenizer = await AutoTokenizer.from_pretrained(GTE_MODEL);
            model = await AutoModel.from_pretrained(GTE_MODEL, { dtype: 'q8' });
            console.error('EmbeddingGemma-300m model loaded successfully');
        })();
        try {
            await this.modelLoadPromise;
        }
        catch (e) {
            this.modelLoadPromise = null;
            throw e;
        }
    }
    loadIndex() {
        if (fs.existsSync(this.indexPath)) {
            try {
                const data = fs.readFileSync(this.indexPath, 'utf-8');
                const parsed = JSON.parse(data);
                // Version 4+ = EmbeddingGemma. Version 3 = old GTE-base (incompatible vectors).
                if (!parsed.version || parsed.version < 4 || parsed.model !== GTE_MODEL) {
                    console.error(`Index v${parsed.version || 1} (model=${parsed.model || 'unknown'}) detected, rebuilding with v4 EmbeddingGemma`);
                    this.createEmptyIndex();
                    return;
                }
                this.index = parsed;
                const count = Object.keys(this.index.entries).length;
                console.error(`Embedding index loaded: ${count} entries (v4, EmbeddingGemma-300m)`);
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
            version: 4,
            created_at: Date.now(),
            updated_at: Date.now(),
            entries: {},
        };
    }
    /**
     * Clear in-memory index for force-rebuild. Persisted file is overwritten
     * on next saveIndex(). In-place mutation preserves references held by
     * SearchEngine / other consumers.
     */
    clearIndex() {
        this.createEmptyIndex();
    }
    saveIndex() {
        if (!this.index)
            return;
        this.index.updated_at = Date.now();
        fs.writeFileSync(this.indexPath, JSON.stringify(this.index), 'utf-8');
    }
    /**
     * Embed a single text string.
     * EmbeddingGemma max_seq_length = 2048 tokens. Model truncates internally.
     * isQuery=true uses asymmetric query prefix; false (default) uses document prefix.
     */
    async embed(text, isQuery = false) {
        await this.ensureModelLoaded();
        const prefix = isQuery ? QUERY_PREFIX : DOC_PREFIX;
        const inputs = await tokenizer([prefix + text], { padding: true, truncation: true, max_length: 2048 });
        const { sentence_embedding } = await model(inputs);
        const vec = Array.from(sentence_embedding.data.slice(0, GTE_DIM));
        return l2normalize(vec);
    }
    /**
     * Embed multiple document texts in batches.
     * Batch size 8 is a memory/throughput sweet spot for 300M model on M4 Pro (q8).
     */
    async embedBatch(texts, batchSize = 8) {
        await this.ensureModelLoaded();
        const allVecs = [];
        for (let start = 0; start < texts.length; start += batchSize) {
            const batch = texts.slice(start, start + batchSize).map(t => DOC_PREFIX + t);
            const inputs = await tokenizer(batch, { padding: true, truncation: true, max_length: 2048 });
            const { sentence_embedding } = await model(inputs);
            for (let i = 0; i < batch.length; i++) {
                const raw = Array.from(sentence_embedding.data.slice(i * GTE_DIM, (i + 1) * GTE_DIM));
                allVecs.push(l2normalize(raw));
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
                // Note changed — remove all old entries for this note.
                // Entry keys are either `notePath` (__full__) or `notePath#blockKey`, so
                // prefix matching avoids reading the note_path property on every entry.
                const notePrefix = notePath + '#';
                for (const entryKey of Object.keys(this.index.entries)) {
                    if (entryKey === notePath || entryKey.startsWith(notePrefix)) {
                        delete this.index.entries[entryKey];
                    }
                }
                // Parse into blocks
                const blocks = parseNoteIntoBlocks(content);
                const contentVecs = []; // for note-level vector (excluding YAML)
                const contentLens = [];
                if (blocks.length > 0) {
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
                }
                // Note-level vector: length-weighted mean of content blocks.
                // If parsing yields no usable content blocks, fall back to embedding the entire note
                // so the __full__ entry still exists for unchanged-note detection on subsequent runs.
                let noteVec;
                if (contentVecs.length > 0) {
                    noteVec = lengthWeightedMean(contentVecs, contentLens);
                }
                else {
                    const fallbackVecs = await this.embedBatch([content]);
                    noteVec = fallbackVecs[0];
                }
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
            throw new Error('EmbeddingGemma embedder not initialized');
        }
        const queryVec = await this.embed(queryText, true);
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
            throw new Error('EmbeddingGemma embedder not initialized');
        }
        const queryVec = await this.embed(queryText, true);
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
    /**
     * Find notes similar to a given note using the gte note-level (__full__) vectors.
     * Consistent with search_notes (same 768d EmbeddingGemma space), and covers notes
     * that only exist in this gte index (e.g. disk-walk discovered notes the Obsidian
     * plugin never embedded into .ajson legacy vectors).
     * Returns null if the note has no __full__ entry, so callers can fall back to legacy.
     * Synchronous: reuses the stored note vector, no query embedding needed.
     */
    similarByPath(notePath, limit = 10, threshold = 0.5) {
        if (!this.index)
            return null;
        const self = this.index.entries[notePath];
        if (!self || self.block_type !== 'full')
            return null;
        const results = [];
        for (const entry of Object.values(this.index.entries)) {
            if (entry.block_type !== 'full')
                continue;
            if (entry.note_path === notePath)
                continue;
            const sim = this.cosineSim(self.vec, entry.vec);
            if (sim >= threshold) {
                results.push({ path: entry.note_path, similarity: sim });
            }
        }
        return results.sort((a, b) => b.similarity - a.similarity).slice(0, limit);
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