/**
 * Semantic search engine for Smart Connections
 */
import type { SimilarNote, ConnectionGraph, NoteContent } from './types.js';
import type { SmartConnectionsLoader } from './smart-connections-loader.js';
export declare class SearchEngine {
    private loader;
    private embeddingModelKey;
    private queryEmbedder;
    constructor(loader: SmartConnectionsLoader);
    /**
     * Initialize the query embedder (call this before using searchByQuery)
     */
    initializeEmbedder(): Promise<void>;
    /**
     * Find similar notes to a given note path
     */
    getSimilarNotes(notePath: string, threshold?: number, limit?: number): SimilarNote[];
    /**
     * Get embedding neighbors for a given embedding vector
     */
    getEmbeddingNeighbors(embeddingVector: number[], k?: number, threshold?: number): SimilarNote[];
    /**
     * Build a connection graph starting from a note
     */
    getConnectionGraph(notePath: string, depth?: number, threshold?: number, maxPerLevel?: number): ConnectionGraph;
    /**
     * Search notes by semantic similarity using embeddings
     * Searches through both note-level (sources) and block-level embeddings
     */
    searchByQuery(queryText: string, limit?: number, threshold?: number): Promise<SimilarNote[]>;
    /**
     * Get note content with matched blocks highlighted
     */
    getNoteWithContext(notePath: string, includeBlocks?: string[]): NoteContent;
    /**
     * Get statistics about the knowledge base
     */
    getStats(): {
        totalNotes: number;
        totalBlocks: number;
        totalEmbeddedBlocks: number;
        embeddingDimension: number;
        modelKey: string;
    };
}
//# sourceMappingURL=search-engine.d.ts.map