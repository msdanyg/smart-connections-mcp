/**
 * Embeds query text locally with the same model a vault's Smart Connections data used.
 * Model weights download once (transformers.js cache) and run fully offline after that.
 */

import { EmbedUnavailableError } from './errors.js';
import { cosineSimilarity } from './vector-index.js';

export type EmbedFn = (text: string) => Promise<number[]>;

export type RawExtractor = (
  text: string,
  opts: { pooling: 'mean'; normalize: boolean },
) => Promise<{ data: ArrayLike<number> }>;

export type PipelineFactory = (
  modelId: string,
  opts: { dtype: 'fp32' | 'q8' },
) => Promise<RawExtractor>;

/** The parts of a transformers.js pipeline that truncation depends on. */
export interface PipelineLike {
  tokenizer: { readonly model_max_length: number };
  model: { config?: { max_position_embeddings?: unknown } };
}

/**
 * Make the tokenizer truncate at the model's real position limit.
 *
 * transformers.js truncates at `tokenizer.model_max_length`, but some models ship a
 * placeholder there (TaylorAI/bge-micro-v2: 1e30) while `config.json` carries the
 * true `max_position_embeddings` (512). Without this, any input past 512 tokens
 * crashes onnxruntime (position-embedding broadcast error) — and 512 tokens is well
 * under 1500 chars for German, CJK, or code. Returns the effective limit.
 */
export function clampTokenizerToModel(p: PipelineLike): number | undefined {
  const limit = p.model?.config?.max_position_embeddings;
  if (typeof limit !== 'number' || !Number.isFinite(limit) || limit <= 0) return undefined;
  const current = p.tokenizer.model_max_length;
  if (typeof current === 'number' && current <= limit) return current;
  // model_max_length is a prototype getter; an own property shadows it.
  Object.defineProperty(p.tokenizer, 'model_max_length', {
    value: limit,
    configurable: true,
    enumerable: true,
    writable: true,
  });
  return limit;
}

const defaultFactory: PipelineFactory = async (modelId, opts) => {
  const { pipeline } = await import('@huggingface/transformers');
  const p = await pipeline('feature-extraction', modelId, { dtype: opts.dtype });
  clampTokenizerToModel(p as unknown as PipelineLike);
  return p as unknown as RawExtractor;
};

const PARITY_WARN_THRESHOLD = 0.99;

/**
 * Coarse guard on input size so a pathological note can't stall tokenization.
 * Real truncation happens at the token level inside the pipeline (see
 * clampTokenizerToModel); this is far above any model's token window.
 */
export const EMBED_MAX_CHARS = 20_000;

export class Embedder {
  private factory: PipelineFactory;
  private cache = new Map<string, Promise<EmbedFn>>();

  constructor(factory: PipelineFactory = defaultFactory) {
    this.factory = factory;
  }

  getEmbedFn(
    modelKey: string,
    parity?: { text: string; vec: number[] },
    warn: (msg: string) => void = () => {},
  ): Promise<EmbedFn> {
    const cached = this.cache.get(modelKey);
    if (cached) return cached;
    const built = this.build(modelKey, parity, warn);
    this.cache.set(modelKey, built);
    built.catch(() => this.cache.delete(modelKey));
    return built;
  }

  private async build(
    modelKey: string,
    parity: { text: string; vec: number[] } | undefined,
    warn: (msg: string) => void,
  ): Promise<EmbedFn> {
    const basename = modelKey.split('/').pop() ?? modelKey;
    const modelIds = [...new Set([modelKey, `Xenova/${basename}`])];
    const dtypes: Array<'fp32' | 'q8'> = ['fp32', 'q8'];
    let lastError: unknown = new Error('no variants attempted');

    for (const modelId of modelIds) {
      for (const dtype of dtypes) {
        let extractor: RawExtractor;
        try {
          extractor = await this.factory(modelId, { dtype });
        } catch (e) {
          lastError = e;
          continue;
        }
        const embed: EmbedFn = async (text) => {
          const input = text.length > EMBED_MAX_CHARS ? text.slice(0, EMBED_MAX_CHARS) : text;
          const out = await extractor(input, { pooling: 'mean', normalize: true });
          return Array.from(out.data as Float32Array);
        };
        if (parity) {
          let computed: number[];
          try {
            computed = await embed(parity.text);
          } catch (e) {
            lastError = e;
            continue;
          }
          if (computed.length !== parity.vec.length) {
            lastError = new Error(
              `${modelId} (${dtype}) produced ${computed.length}-dim vectors, vault has ${parity.vec.length}-dim`,
            );
            continue;
          }
          const cos = cosineSimilarity(computed, parity.vec);
          if (cos < PARITY_WARN_THRESHOLD) {
            warn(
              `embedding parity for ${modelId} (${dtype}) is ${cos.toFixed(4)} (< ${PARITY_WARN_THRESHOLD}); ` +
                `rankings may differ slightly from Smart Connections`,
            );
          }
        }
        return embed;
      }
    }
    throw new EmbedUnavailableError(
      `Could not load embedding model "${modelKey}": ${lastError instanceof Error ? lastError.message : String(lastError)}`,
    );
  }
}
