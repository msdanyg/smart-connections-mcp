/**
 * Semantic search engine for Smart Connections
 */
import type { SimilarNote, ConnectionGraph, NoteContent, SubgraphNode } from './types.js';
import type { SmartConnectionsLoader } from './smart-connections-loader.js';
export declare class SearchEngine {
    private loader;
    private embeddingModelKey;
    constructor(loader: SmartConnectionsLoader);
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
     * Search notes by content similarity
     */
    searchByQuery(queryText: string, limit?: number, threshold?: number): SimilarNote[];
    /**
     * Get note content with matched blocks highlighted
     */
    getNoteWithContext(notePath: string, includeBlocks?: string[]): NoteContent;
    /**
     * Return an enriched semantic neighbourhood for a query.
     * Uses multi-keyword scoring over file content, then reads each result's
     * YAML frontmatter to attach summary, confidence, domains, and linked_to.
     */
    getSessionSubgraph(query: string, n?: number, minConfidence?: number): SubgraphNode[];
    /**
     * Tokenize a query and score each vault note by how many keyword hits it has.
     * Unlike searchByQuery (which treats the whole string as a single regex), this
     * splits on whitespace so multi-word queries work correctly.
     */
    private searchByKeywords;
    private parseFrontmatter;
    private inferPageType;
    /**
     * Get statistics about the knowledge base
     */
    getStats(): {
        totalNotes: number;
        totalBlocks: number;
        embeddingDimension: number;
        modelKey: string;
    };
}
//# sourceMappingURL=search-engine.d.ts.map