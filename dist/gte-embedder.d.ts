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
export declare class GteEmbedder {
    private vaultPath;
    private indexPath;
    private index;
    private ready;
    constructor(vaultPath: string);
    initialize(): Promise<void>;
    private loadIndex;
    private createEmptyIndex;
    private saveIndex;
    /**
     * Embed a single text string.
     * GTE-base max_seq_length = 512 tokens. Model truncates internally.
     */
    embed(text: string): Promise<number[]>;
    /**
     * Embed multiple texts in batches for ~1.5x speedup on M4 Pro.
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