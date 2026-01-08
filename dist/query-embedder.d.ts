/**
 * Query embedding generator using transformers.js
 * Uses the same model as Smart Connections for compatibility
 */
export declare class QueryEmbedder {
    private model;
    private modelName;
    private initPromise;
    constructor(modelName?: string);
    /**
     * Initialize the embedding model (lazy loading)
     */
    initialize(): Promise<void>;
    /**
     * Generate embedding for a query string
     * Matches Smart Connections' exact configuration
     */
    embed(text: string): Promise<number[]>;
    /**
     * Check if model is loaded
     */
    isReady(): boolean;
}
export declare function getQueryEmbedder(modelName?: string): QueryEmbedder;
//# sourceMappingURL=query-embedder.d.ts.map