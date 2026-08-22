import { describe, it, expect } from 'vitest';
import { Embedder, EMBED_MAX_CHARS, clampTokenizerToModel, type PipelineFactory } from '../src/embedder.js';
import { EmbedUnavailableError } from '../src/errors.js';

const okExtractor = (vec: number[]) => async () => ({ data: Float32Array.from(vec) });

describe('Embedder', () => {
  it('returns an embed fn from the first working variant and caches it', async () => {
    const calls: string[] = [];
    const factory: PipelineFactory = async (modelId, { dtype }) => {
      calls.push(`${modelId}:${dtype}`);
      return okExtractor([1, 0, 0, 0]);
    };
    const embedder = new Embedder(factory);
    const embed = await embedder.getEmbedFn('org/model-x');
    expect(await embed('hello')).toEqual([1, 0, 0, 0]);
    await embedder.getEmbedFn('org/model-x');
    expect(calls).toEqual(['org/model-x:fp32']); // cached, single pipeline load
  });

  it('falls through failing variants to the Xenova mirror', async () => {
    const calls: string[] = [];
    const factory: PipelineFactory = async (modelId, { dtype }) => {
      calls.push(`${modelId}:${dtype}`);
      if (modelId !== 'Xenova/model-x') throw new Error('404');
      return okExtractor([0, 1]);
    };
    const embed = await new Embedder(factory).getEmbedFn('org/model-x');
    expect(await embed('q')).toEqual([0, 1]);
    expect(calls).toEqual(['org/model-x:fp32', 'org/model-x:q8', 'Xenova/model-x:fp32']);
  });

  it('warns on low parity but still works; rejects wrong-dim variants', async () => {
    const factory: PipelineFactory = async (modelId, { dtype }) =>
      dtype === 'fp32' ? okExtractor([1, 0]) : okExtractor([0.5, 0.5, 0, 0]); // fp32 wrong dim
    const warnings: string[] = [];
    const embed = await new Embedder(factory).getEmbedFn(
      'org/m',
      { text: 'sample', vec: [1, 0, 0, 0] }, // 4-dim stored vec; fp32 (2-dim) must be skipped
      (m) => warnings.push(m),
    );
    expect((await embed('q')).length).toBe(4);
    expect(warnings.length).toBe(1); // cos([0.5,0.5,0,0],[1,0,0,0]) ≈ 0.707 < 0.99
    expect(warnings[0]).toContain('parity');
  });

  it('guards against pathological input sizes before tokenization', async () => {
    const received: string[] = [];
    const factory: PipelineFactory = async () => async (text: string) => {
      received.push(text);
      return { data: Float32Array.from([1, 0]) };
    };
    const embed = await new Embedder(factory).getEmbedFn('org/m');
    await embed('x'.repeat(EMBED_MAX_CHARS * 3));
    expect(received[0].length).toBe(EMBED_MAX_CHARS);
  });

  it('passes short inputs through unmodified', async () => {
    const received: string[] = [];
    const factory: PipelineFactory = async () => async (text: string) => {
      received.push(text);
      return { data: Float32Array.from([1, 0]) };
    };
    const embed = await new Embedder(factory).getEmbedFn('org/m');
    await embed('short query');
    expect(received[0]).toBe('short query');
  });

  it('throws EmbedUnavailableError when nothing loads, and retries next call', async () => {
    let attempts = 0;
    const factory: PipelineFactory = async () => {
      attempts++;
      throw new Error('offline');
    };
    const embedder = new Embedder(factory);
    await expect(embedder.getEmbedFn('org/m')).rejects.toThrow(EmbedUnavailableError);
    const firstRound = attempts;
    await expect(embedder.getEmbedFn('org/m')).rejects.toThrow(EmbedUnavailableError);
    expect(attempts).toBe(firstRound * 2); // cache evicted on failure
  });
});

describe('clampTokenizerToModel', () => {
  // Issue #10: TaylorAI/bge-micro-v2 ships model_max_length = 1e30 in its tokenizer
  // config, so transformers.js never truncates and onnxruntime crashes past 512
  // positions. The model's own config.json knows the real limit.
  const fake = (tokenizerMax: number | undefined, modelMax: unknown) => {
    class Tok {
      get model_max_length() {
        return tokenizerMax ?? Infinity;
      }
    }
    return { tokenizer: new Tok(), model: { config: { max_position_embeddings: modelMax } } };
  };

  it('caps an unusable tokenizer limit at the model position limit', () => {
    const p = fake(1e30, 512);
    expect(clampTokenizerToModel(p)).toBe(512);
    expect(p.tokenizer.model_max_length).toBe(512);
  });

  it('leaves a tokenizer limit that is already tighter than the model alone', () => {
    const p = fake(512, 514);
    expect(clampTokenizerToModel(p)).toBe(512);
    expect(p.tokenizer.model_max_length).toBe(512);
  });

  it('does nothing when the model config has no position limit', () => {
    const p = fake(1e30, undefined);
    expect(clampTokenizerToModel(p)).toBeUndefined();
    expect(p.tokenizer.model_max_length).toBe(1e30);
  });
});
