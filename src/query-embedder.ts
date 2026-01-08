/**
 * Query embedding generator using transformers.js
 * Uses the same model as Smart Connections for compatibility
 */

import { pipeline, type FeatureExtractionPipeline } from '@xenova/transformers';

export class QueryEmbedder {
  private model: FeatureExtractionPipeline | null = null;
  private modelName: string;
  private initPromise: Promise<void> | null = null;

  constructor(modelName: string = 'TaylorAI/bge-micro-v2') {
    this.modelName = modelName;
  }

  /**
   * Initialize the embedding model (lazy loading)
   */
  async initialize(): Promise<void> {
    if (this.model) return;

    if (this.initPromise) {
      await this.initPromise;
      return;
    }

    this.initPromise = (async () => {
      console.error(`Loading embedding model: ${this.modelName}...`);
      this.model = await pipeline('feature-extraction', this.modelName, {
        quantized: true, // Use quantized model for faster loading
      });
      console.error(`Embedding model loaded successfully`);
    })();

    await this.initPromise;
  }

  /**
   * Generate embedding for a query string
   * Matches Smart Connections' exact configuration
   */
  async embed(text: string): Promise<number[]> {
    await this.initialize();

    if (!this.model) {
      throw new Error('Model not initialized');
    }

    // Generate embedding with same settings as Smart Connections
    const output = await this.model(text, {
      pooling: 'mean',
      normalize: true,
    });

    // Convert to array and apply same rounding as Smart Connections
    // for precision matching: Math.round(val * 1e8) / 1e8
    const rawArray = Array.from(output.data as Float32Array);
    return rawArray.map(val => Math.round(val * 1e8) / 1e8);
  }

  /**
   * Check if model is loaded
   */
  isReady(): boolean {
    return this.model !== null;
  }
}

// Singleton instance
let embedderInstance: QueryEmbedder | null = null;

export function getQueryEmbedder(modelName?: string): QueryEmbedder {
  if (!embedderInstance) {
    embedderInstance = new QueryEmbedder(modelName);
  }
  return embedderInstance;
}
