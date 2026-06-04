/**
 * Semantic search engine for Smart Connections
 */
import { findNearestNeighbors } from './embedding-utils.js';
export class SearchEngine {
    loader;
    embeddingModelKey;
    constructor(loader) {
        this.loader = loader;
        this.embeddingModelKey = loader.getEmbeddingModelKey();
    }
    /**
     * Find similar notes to a given note path
     */
    getSimilarNotes(notePath, threshold = 0.5, limit = 10) {
        const source = this.loader.getSource(notePath);
        if (!source) {
            throw new Error(`Note not found: ${notePath}`);
        }
        const embeddings = source.embeddings[this.embeddingModelKey];
        if (!embeddings || !embeddings.vec) {
            throw new Error(`No embeddings found for note: ${notePath}`);
        }
        // Build vector dataset from all sources
        const vectors = Array.from(this.loader.getSources().entries())
            .filter(([path]) => path !== notePath) // Exclude the query note itself
            .map(([path, src]) => {
            const emb = src.embeddings[this.embeddingModelKey];
            return {
                id: path,
                vec: emb?.vec || [],
                metadata: {
                    blocks: Object.keys(src.blocks || {}),
                    lastModified: src.last_import?.mtime || 0
                }
            };
        })
            .filter(item => item.vec.length > 0);
        // Find nearest neighbors
        const neighbors = findNearestNeighbors(embeddings.vec, vectors, limit, threshold);
        // Convert to SimilarNote format
        return neighbors.map(neighbor => ({
            path: neighbor.id,
            similarity: neighbor.similarity,
            blocks: neighbor.metadata.blocks
        }));
    }
    /**
     * Get embedding neighbors for a given embedding vector
     */
    getEmbeddingNeighbors(embeddingVector, k = 10, threshold = 0.5) {
        // Build vector dataset from all sources
        const vectors = Array.from(this.loader.getSources().entries())
            .map(([path, src]) => {
            const emb = src.embeddings[this.embeddingModelKey];
            return {
                id: path,
                vec: emb?.vec || [],
                metadata: {
                    blocks: Object.keys(src.blocks || {}),
                    lastModified: src.last_import?.mtime || 0
                }
            };
        })
            .filter(item => item.vec.length > 0);
        // Find nearest neighbors
        const neighbors = findNearestNeighbors(embeddingVector, vectors, k, threshold);
        // Convert to SimilarNote format
        return neighbors.map(neighbor => ({
            path: neighbor.id,
            similarity: neighbor.similarity,
            blocks: neighbor.metadata.blocks
        }));
    }
    /**
     * Build a connection graph starting from a note
     */
    getConnectionGraph(notePath, depth = 2, threshold = 0.6, maxPerLevel = 5) {
        const visited = new Set();
        const flatConnections = [];
        const buildGraph = (currentPath, currentDepth, parentSimilarity = 1.0) => {
            visited.add(currentPath);
            // Add to flat list (skip root at depth 0)
            if (currentDepth > 0) {
                flatConnections.push({
                    path: currentPath,
                    depth: currentDepth,
                    similarity: parentSimilarity
                });
            }
            // Stop if we've reached max depth
            if (currentDepth >= depth) {
                return;
            }
            // Find similar notes
            try {
                const similar = this.getSimilarNotes(currentPath, threshold, maxPerLevel);
                // Recursively build connections
                for (const sim of similar) {
                    // Skip already visited nodes to prevent cycles
                    if (!visited.has(sim.path)) {
                        buildGraph(sim.path, currentDepth + 1, sim.similarity);
                    }
                }
            }
            catch (error) {
                console.error(`Error building graph for ${currentPath}:`, error);
            }
        };
        buildGraph(notePath, 0);
        return {
            root: notePath,
            connections: flatConnections
        };
    }
    /**
     * Search notes by content similarity
     */
    searchByQuery(queryText, limit = 10, threshold = 0.5) {
        // For now, we'll do a simple keyword match since we don't have
        // a way to generate embeddings for arbitrary text without the model.
        // In a full implementation, you'd call the embedding model here.
        const results = [];
        const queryLower = queryText.toLowerCase();
        for (const [path, source] of this.loader.getSources()) {
            try {
                const content = this.loader.readNoteContent(path).toLowerCase();
                // Simple relevance scoring based on keyword matches
                const matches = (content.match(new RegExp(queryLower, 'gi')) || []).length;
                if (matches > 0) {
                    // Normalize score (this is a crude approximation)
                    const score = Math.min(matches / 10, 1.0);
                    if (score >= threshold) {
                        results.push({
                            path,
                            similarity: score,
                            blocks: Object.keys(source.blocks || {})
                        });
                    }
                }
            }
            catch (error) {
                // Skip notes that can't be read
                continue;
            }
        }
        // Sort by similarity and limit
        return results
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }
    /**
     * Get note content with matched blocks highlighted
     */
    getNoteWithContext(notePath, includeBlocks = []) {
        const content = this.loader.readNoteContent(notePath);
        const source = this.loader.getSource(notePath);
        const availableBlocks = source ? Object.keys(source.blocks || {}) : [];
        return {
            path: notePath,
            content,
            blocks: availableBlocks
        };
    }
    /**
     * Return an enriched semantic neighbourhood for a query.
     * Uses multi-keyword scoring over file content, then reads each result's
     * YAML frontmatter to attach summary, confidence, domains, and linked_to.
     */
    getSessionSubgraph(query, n = 20, minConfidence) {
        const pool = this.searchByKeywords(query, Math.min(n * 4, 160));
        const nodes = [];
        for (const { path: notePath, similarity } of pool) {
            try {
                const content = this.loader.readNoteContent(notePath);
                const fm = this.parseFrontmatter(content);
                if (minConfidence !== undefined) {
                    if (fm.confidence === null || fm.confidence < minConfidence)
                        continue;
                }
                const basename = notePath.split('/').pop()?.replace(/\.md$/, '') ?? notePath;
                nodes.push({
                    path: notePath,
                    name: basename,
                    similarity,
                    summary_1line: fm.summary_1line,
                    confidence: fm.confidence,
                    domains: fm.domains,
                    linked_to: fm.linked_to,
                    page_type: this.inferPageType(basename),
                });
            }
            catch {
                // skip files that can't be read
            }
        }
        return nodes.sort((a, b) => b.similarity - a.similarity).slice(0, n);
    }
    // --- private helpers ---
    /**
     * Tokenize a query and score each vault note by how many keyword hits it has.
     * Unlike searchByQuery (which treats the whole string as a single regex), this
     * splits on whitespace so multi-word queries work correctly.
     */
    searchByKeywords(query, limit) {
        const stopWords = new Set([
            'a', 'an', 'the', 'and', 'or', 'of', 'for', 'in', 'to', 'with',
            'by', 'at', 'on', 'is', 'are', 'was', 'be', 'its', 'that', 'this',
        ]);
        const keywords = query
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length > 2 && !stopWords.has(w))
            .map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
        if (keywords.length === 0)
            return [];
        const scored = [];
        for (const [notePath] of this.loader.getSources()) {
            try {
                const content = this.loader.readNoteContent(notePath).toLowerCase();
                let total = 0;
                for (const kw of keywords) {
                    total += (content.match(new RegExp(kw, 'g')) || []).length;
                }
                if (total > 0) {
                    scored.push({ path: notePath, similarity: Math.min(total / (keywords.length * 3), 1.0) });
                }
            }
            catch {
                // skip unreadable notes
            }
        }
        return scored.sort((a, b) => b.similarity - a.similarity).slice(0, limit);
    }
    parseFrontmatter(content) {
        const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
        if (!match)
            return { summary_1line: '', confidence: null, domains: [], linked_to: [] };
        const yaml = match[1];
        const summaryMatch = yaml.match(/^summary_1line:\s*(.+)$/m);
        const summary_1line = summaryMatch
            ? summaryMatch[1].trim().replace(/^["']|["']$/g, '')
            : '';
        const confidenceMatch = yaml.match(/^confidence:\s*([\d.]+)/m);
        const confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) : null;
        const domainsMatch = yaml.match(/^domains:\s*\[([^\]]*)\]/m);
        const domains = domainsMatch
            ? domainsMatch[1].split(',').map(d => d.trim().replace(/^["']|["']$/g, '')).filter(Boolean)
            : [];
        const linkedToMatch = yaml.match(/^linked_to:\s*\[(.+)\]/m);
        const linked_to = [];
        if (linkedToMatch) {
            const raw = linkedToMatch[1];
            const wikiRe = /\[\[([^\]]+)\]\]/g;
            let m;
            while ((m = wikiRe.exec(raw)) !== null) {
                linked_to.push(m[1]);
            }
        }
        return { summary_1line, confidence, domains, linked_to };
    }
    inferPageType(name) {
        if (name.startsWith('entity_'))
            return 'entity';
        if (name.startsWith('synthesis_'))
            return 'synthesis';
        if (name.startsWith('summary_'))
            return 'summary';
        if (name.startsWith('query_'))
            return 'query';
        return 'other';
    }
    /**
     * Get statistics about the knowledge base
     */
    getStats() {
        const sources = this.loader.getSources();
        let totalBlocks = 0;
        let embeddingDim = 0;
        for (const source of sources.values()) {
            totalBlocks += Object.keys(source.blocks || {}).length;
            if (embeddingDim === 0) {
                const emb = source.embeddings[this.embeddingModelKey];
                if (emb?.vec) {
                    embeddingDim = emb.vec.length;
                }
            }
        }
        return {
            totalNotes: sources.size,
            totalBlocks,
            embeddingDimension: embeddingDim,
            modelKey: this.embeddingModelKey
        };
    }
}
//# sourceMappingURL=search-engine.js.map