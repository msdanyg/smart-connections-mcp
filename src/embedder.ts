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

const defaultFactory: PipelineFactory = async (modelId, opts) => {
  const { pipeline } = await import('@huggingface/transformers');
  const p = await pipeline('feature-extraction', modelId, { dtype: opts.dtype });
  return p as unknown as RawExtractor;
};

const PARITY_WARN_THRESHOLD = 0.99;

// Some tokenizer configs (e.g. TaylorAI/bge-micro-v2's tokenizer_config.json) leave
// model_max_length unset, so transformers.js's `truncation: true` never actually engages —
// it falls back to that missing value, i.e. no cap. Text longer than the model's real,
// fixed position-embedding table (e.g. 512 tokens for small BERT-family models) then
// overflows and onnxruntime throws. This bites the parity check in particular, since it
// samples a whole raw note (Vault.paritySample), which can run to thousands of tokens.
// Retry once with a conservative character-based truncation rather than failing the
// entire model load over a single oversized input.
const TRUNCATE_RETRY_CHARS = 1500;

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
          try {
            const out = await extractor(text, { pooling: 'mean', normalize: true });
            return Array.from(out.data as Float32Array);
          } catch (e) {
            if (text.length <= TRUNCATE_RETRY_CHARS) throw e;
            const out = await extractor(text.slice(0, TRUNCATE_RETRY_CHARS), {
              pooling: 'mean',
              normalize: true,
            });
            return Array.from(out.data as Float32Array);
          }
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
