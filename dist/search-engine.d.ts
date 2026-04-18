/**
 * Semantic search engine for Smart Connections
 */
import type { SimilarNote, ConnectionGraph, NoteContent } from './types.js';
import type { SmartConnectionsLoader } from './smart-connections-loader.js';
import type { GteEmbedder } from './gte-embedder.js';
export declare class SearchEngine {
    private loader;
    private embeddingModelKey;
    private gteEmbedder;
    constructor(loader: SmartConnectionsLoader, gteEmbedder?: GteEmbedder);
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
     * Search notes by semantic similarity using EmbeddingGemma embeddings.
     * Falls back to keyword search if the semantic index is unavailable or empty.
     */
    searchByQuery(queryText: string, limit?: number, threshold?: number): Promise<SimilarNote[]>;
    /**
     * Get note content with matched blocks highlighted
     */
    getNoteWithContext(notePath: string, includeBlocks?: string[]): NoteContent;
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
    getStats(): {
        totalNotes: number;
        totalBlocks: number;
        embeddingDimension: number;
        modelKey: string;
        legacy: {
            modelKey: string;
            embeddingDimension: number;
            totalBlocks: number;
        };
        gte: {
            model: string;
            dimension: number;
            entries: number;
            notes: number;
            blockTypes: Record<string, number>;
            updated_at: number;
        } | null;
        primary: 'gte' | 'legacy';
    };
}
//# sourceMappingURL=search-engine.d.ts.map