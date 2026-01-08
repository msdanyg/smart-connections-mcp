/**
 * Semantic search engine for Smart Connections
 */
import { findNearestNeighbors } from './embedding-utils.js';
import { QueryEmbedder } from './query-embedder.js';
export class SearchEngine {
    loader;
    embeddingModelKey;
    queryEmbedder;
    constructor(loader) {
        this.loader = loader;
        this.embeddingModelKey = loader.getEmbeddingModelKey();
        // Initialize query embedder with the same model Smart Connections uses
        this.queryEmbedder = new QueryEmbedder('TaylorAI/bge-micro-v2');
    }
    /**
     * Initialize the query embedder (call this before using searchByQuery)
     */
    async initializeEmbedder() {
        await this.queryEmbedder.initialize();
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
     * Search notes by semantic similarity using embeddings
     * Searches through both note-level (sources) and block-level embeddings
     */
    async searchByQuery(queryText, limit = 10, threshold = 0.5) {
        // Generate embedding for the query text
        const queryEmbedding = await this.queryEmbedder.embed(queryText);
        // Build vector dataset from all blocks (more granular search)
        const blockVectors = Array.from(this.loader.getBlocks().entries())
            .map(([key, block]) => {
            const emb = block.embeddings[this.embeddingModelKey];
            // Extract the file path from the block key (e.g., "file.md##Heading" -> "file.md")
            const filePath = key.split('#')[0];
            return {
                id: key,
                vec: emb?.vec || [],
                metadata: {
                    filePath,
                    lines: block.lines,
                    isBlock: true
                }
            };
        })
            .filter(item => item.vec.length > 0);
        // Also include source-level vectors for notes without blocks
        const sourceVectors = Array.from(this.loader.getSources().entries())
            .map(([path, src]) => {
            const emb = src.embeddings[this.embeddingModelKey];
            return {
                id: path,
                vec: emb?.vec || [],
                metadata: {
                    filePath: path,
                    lines: undefined,
                    isBlock: false,
                    blocks: Object.keys(src.blocks || {})
                }
            };
        })
            .filter(item => item.vec.length > 0);
        // Combine both vectors, prioritizing blocks
        const allVectors = [...blockVectors, ...sourceVectors];
        // Find nearest neighbors using cosine similarity
        const neighbors = findNearestNeighbors(queryEmbedding, allVectors, limit * 2, // Get more results to allow deduplication
        threshold);
        // Convert to SimilarNote format
        const results = neighbors.map(neighbor => {
            const result = {
                path: neighbor.metadata.isBlock ? neighbor.id : neighbor.id,
                similarity: Math.round(neighbor.similarity * 100) / 100,
                isBlock: neighbor.metadata.isBlock
            };
            if (neighbor.metadata.lines) {
                result.lines = neighbor.metadata.lines;
            }
            if (neighbor.metadata.blocks) {
                result.blocks = neighbor.metadata.blocks;
            }
            return result;
        });
        // Return top results (limit)
        return results.slice(0, limit);
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
     * Get statistics about the knowledge base
     */
    getStats() {
        const sources = this.loader.getSources();
        const blocks = this.loader.getBlocks();
        let totalBlocksInSources = 0;
        let embeddingDim = 0;
        for (const source of sources.values()) {
            totalBlocksInSources += Object.keys(source.blocks || {}).length;
            if (embeddingDim === 0) {
                const emb = source.embeddings[this.embeddingModelKey];
                if (emb?.vec) {
                    embeddingDim = emb.vec.length;
                }
            }
        }
        return {
            totalNotes: sources.size,
            totalBlocks: totalBlocksInSources,
            totalEmbeddedBlocks: blocks.size,
            embeddingDimension: embeddingDim,
            modelKey: this.embeddingModelKey
        };
    }
}
//# sourceMappingURL=search-engine.js.map