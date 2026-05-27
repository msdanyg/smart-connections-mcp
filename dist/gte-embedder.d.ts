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
export declare class GteEmbedder {
    private vaultPath;
    private indexPath;
    private index;
    private ready;
    constructor(vaultPath: string);
    initialize(): Promise<void>;
    private modelLoadPromise;
    private ensureModelLoaded;
    private loadIndex;
    private createEmptyIndex;
    /**
     * Clear in-memory index for force-rebuild. Persisted file is overwritten
     * on next saveIndex(). In-place mutation preserves references held by
     * SearchEngine / other consumers.
     */
    clearIndex(): void;
    private saveIndex;
    /**
     * Embed a single text string.
     * EmbeddingGemma max_seq_length = 2048 tokens. Model truncates internally.
     * isQuery=true uses asymmetric query prefix; false (default) uses document prefix.
     */
    embed(text: string, isQuery?: boolean): Promise<number[]>;
    /**
     * Embed multiple document texts in batches.
     * Batch size 8 is a memory/throughput sweet spot for 300M model on M4 Pro (q8).
     */
    embedBatch(texts: string[], batchSize?: number): Promise<number[][]>;
    /**
     * Build/update index with adaptive block splitting.
     * Hash-based change detection at note level: if note unchanged, skip all blocks.
     */
    buildIndex(notePaths: string[], readContent: (path: string) => string, onProgress?: (current: number, total: number, path: string) => void): Promise<{
        notes_processed: number;
        notes_unchanged: number;
        blocks_total: number;
        blocks_by_type: {
            yaml: number;
            sync: number;
            h2: number;
            h3: number;
            intro: number;
            full: number;
        };
    }>;
    /**
     * Semantic search at block level.
     * Excludes __full__ and __yaml__ entries (use searchNotes() for note-level).
     */
    search(queryText: string, limit?: number, threshold?: number): Promise<Array<{
        path: string;
        block: string;
        blockType: string;
        similarity: number;
    }>>;
    /**
     * Search at note level only (using __full__ vectors).
     */
    searchNotes(queryText: string, limit?: number, threshold?: number): Promise<Array<{
        path: string;
        similarity: number;
    }>>;
    /**
     * Find notes similar to a given note using the gte note-level (__full__) vectors.
     * Consistent with search_notes (same 768d EmbeddingGemma space), and covers notes
     * that only exist in this gte index (e.g. disk-walk discovered notes the Obsidian
     * plugin never embedded into .ajson legacy vectors).
     * Returns null if the note has no __full__ entry, so callers can fall back to legacy.
     * Synchronous: reuses the stored note vector, no query embedding needed.
     */
    similarByPath(notePath: string, limit?: number, threshold?: number): Array<{
        path: string;
        similarity: number;
    }> | null;
    private cosineSim;
    getStats(): {
        model: string;
        dimension: number;
        entries: number;
        notes: number;
        blockTypes: Record<string, number>;
        updated_at: number;
    } | null;
}
//# sourceMappingURL=gte-embedder.d.ts.map