/**
 * Embedding generator using the same model as Smart Connections (bge-micro-v2).
 * Lazy-loads the model on first use and caches it in memory.
 */
import { FeatureExtractionPipeline } from '@xenova/transformers';
/**
 * Generate an embedding vector from text.
 * Returns a 384-dimensional vector matching bge-micro-v2's output.
 */
export declare function generateEmbedding(text: string, modelName?: string): Promise<number[]>;
/**
 * Preload the embedding model.
 * Call this during server initialization to avoid first-query latency.
 */
export declare function preloadModel(modelName?: string): Promise<void>;
/**
 * Check if the model is already loaded.
 */
export declare function isModelLoaded(): boolean;
export type { FeatureExtractionPipeline };
//# sourceMappingURL=embedding-generator.d.ts.map