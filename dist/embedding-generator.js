/**
 * Embedding generator using the same model as Smart Connections (bge-micro-v2).
 * Lazy-loads the model on first use and caches it in memory.
 */
import { pipeline } from '@xenova/transformers';
let embedder = null;
let loadingPromise = null;
const DEFAULT_MODEL = 'TaylorAI/bge-micro-v2';
/**
 * Get or initialize the embedding pipeline.
 * Lazy-loads on first call, then caches in memory.
 */
async function getEmbedder(modelName = DEFAULT_MODEL) {
    if (embedder)
        return embedder;
    if (loadingPromise)
        return loadingPromise;
    loadingPromise = pipeline('feature-extraction', modelName, {
        quantized: true,
    });
    embedder = await loadingPromise;
    return embedder;
}
/**
 * Generate an embedding vector from text.
 * Returns a 384-dimensional vector matching bge-micro-v2's output.
 */
export async function generateEmbedding(text, modelName) {
    const pipe = await getEmbedder(modelName);
    const output = await pipe(text, {
        pooling: 'mean',
        normalize: true,
    });
    // Convert tensor to plain array
    const vec = Array.from(output.data);
    return vec;
}
/**
 * Preload the embedding model.
 * Call this during server initialization to avoid first-query latency.
 */
export async function preloadModel(modelName) {
    await getEmbedder(modelName);
    console.error('Embedding model preloaded successfully');
}
/**
 * Check if the model is already loaded.
 */
export function isModelLoaded() {
    return embedder !== null;
}
//# sourceMappingURL=embedding-generator.js.map