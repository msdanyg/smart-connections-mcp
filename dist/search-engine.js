/**
 * Semantic search engine for Smart Connections
 */
import { findNearestNeighbors } from './embedding-utils.js';
export class SearchEngine {
    loader;
    embeddingModelKey;
    gteEmbedder = null;
    constructor(loader, gteEmbedder) {
        this.loader = loader;
        this.embeddingModelKey = loader.getEmbeddingModelKey();
        this.gteEmbedder = gteEmbedder || null;
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
     * Search notes by semantic similarity using GTE-base embeddings.
     * Falls back to keyword search if GTE embedder is not available.
     */
    async searchByQuery(queryText, limit = 10, threshold = 0.3) {
        // Use GTE-base semantic search if available (block-level)
        if (this.gteEmbedder) {
            const gteResults = await this.gteEmbedder.search(queryText, limit * 3, threshold);
            // Group by note, keep best block per note, but include matched block info
            const noteMap = new Map();
            for (const r of gteResults) {
                const existing = noteMap.get(r.path);
                if (!existing || r.similarity > existing.similarity) {
                    const source = this.loader.getSource(r.path);
                    noteMap.set(r.path, {
                        path: r.path,
                        similarity: r.similarity,
                        matchedBlock: r.block,
                        matchedBlockType: r.blockType,
                        blocks: source ? Object.keys(source.blocks || {}) : [],
                    });
                }
            }
            return Array.from(noteMap.values())
                .sort((a, b) => b.similarity - a.similarity)
                .slice(0, limit)
                .map(r => ({
                path: r.path,
                similarity: r.similarity,
                blocks: r.blocks,
                matchedContent: `[${r.matchedBlockType}] ${r.matchedBlock}`,
            }));
        }
        // Fallback: keyword search
        const results = [];
        const queryLower = queryText.toLowerCase();
        for (const [path, source] of this.loader.getSources()) {
            try {
                const content = this.loader.readNoteContent(path).toLowerCase();
                const matches = (content.match(new RegExp(queryLower, 'gi')) || []).length;
                if (matches > 0) {
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
                continue;
            }
        }
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
     * Get statistics about the knowledge base.
     *
     * The MCP hosts two independent embedding indexes:
     * - **legacy**: original Obsidian Smart Connections plugin index
     *   (read from `.smart-env/multi/*.ajson`, typically bge-micro-v2 384d).
     *   Reported via `totalBlocks` / `embeddingDimension` / `modelKey`.
     * - **gte**: this MCP's custom block-level semantic index
     *   (read from `.smart-env/embedding-index.json`, currently
     *   EmbeddingGemma-300m 768d). Reported under `gte`.
     *
     * `search_notes` prefers the gte index when available (see `searchByQuery`),
     * falling back to keyword search if absent. Callers that care about the
     * actual query backend should consult `primary`.
     *
     * Top-level fields are preserved for backward compatibility with earlier
     * clients that only expected the legacy fields.
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
        const gteStats = this.gteEmbedder?.getStats() ?? null;
        return {
            totalNotes: sources.size,
            totalBlocks,
            embeddingDimension: embeddingDim,
            modelKey: this.embeddingModelKey,
            legacy: {
                modelKey: this.embeddingModelKey,
                embeddingDimension: embeddingDim,
                totalBlocks,
            },
            gte: gteStats,
            primary: gteStats ? 'gte' : 'legacy',
        };
    }
}
//# sourceMappingURL=search-engine.js.map