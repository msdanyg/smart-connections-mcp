/**
 * Embedding generator using the same model as Smart Connections (bge-micro-v2).
 * Lazy-loads the model on first use and caches it in memory.
 */

import { pipeline, FeatureExtractionPipeline } from '@xenova/transformers';

let embedder: FeatureExtractionPipeline | null = null;
let loadingPromise: Promise<FeatureExtractionPipeline> | null = null;

const DEFAULT_MODEL = 'TaylorAI/bge-micro-v2';

/**
 * Get or initialize the embedding pipeline.
 * Lazy-loads on first call, then caches in memory.
 */
async function getEmbedder(modelName: string = DEFAULT_MODEL): Promise<FeatureExtractionPipeline> {
  if (embedder) return embedder;

  if (loadingPromise) return loadingPromise;

  loadingPromise = pipeline('feature-extraction', modelName, {
    quantized: true,
  }) as Promise<FeatureExtractionPipeline>;

  embedder = await loadingPromise;
  return embedder;
}

/**
 * Generate an embedding vector from text.
 * Returns a 384-dimensional vector matching bge-micro-v2's output.
 */
export async function generateEmbedding(text: string, modelName?: string): Promise<number[]> {
  const pipe = await getEmbedder(modelName);

  const output = await pipe(text, {
    pooling: 'mean',
    normalize: true,
  });

  // Convert tensor to plain array
  const vec = Array.from(output.data) as number[];
  return vec;
}

/**
 * Preload the embedding model.
 * Call this during server initialization to avoid first-query latency.
 */
export async function preloadModel(modelName?: string): Promise<void> {
  await getEmbedder(modelName);
  console.error('Embedding model preloaded successfully');
}

/**
 * Check if the model is already loaded.
 */
export function isModelLoaded(): boolean {
  return embedder !== null;
}

// Re-export for type compatibility
export type { FeatureExtractionPipeline };