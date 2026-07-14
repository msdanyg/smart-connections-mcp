import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { Embedder } from '../../src/embedder.js';
import { SearchEngine } from '../../src/search-engine.js';
import { VaultRegistry } from '../../src/vault-registry.js';
import { cosineSimilarity } from '../../src/vector-index.js';

const MODEL = 'TaylorAI/bge-micro-v2';

describe('live embedder (downloads real model)', () => {
  it('embeds 384-dim normalized vectors with sane semantics', async () => {
    const embed = await new Embedder().getEmbedFn(MODEL);
    const a = await embed('The king rules his kingdom.');
    const b = await embed('A monarch governs the realm.');
    const c = await embed('Recipe for chocolate chip cookies.');
    expect(a.length).toBe(384);
    expect(Math.sqrt(a.reduce((s, x) => s + x * x, 0))).toBeCloseTo(1, 3);
    expect(cosineSimilarity(a, b)).toBeGreaterThan(cosineSimilarity(a, c));
  }, 300_000);

  it('semantic search end-to-end over a real-model vault', async () => {
    const embedder = new Embedder();
    const embed = await embedder.getEmbedFn(MODEL);
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'scmcp-live-'));
    try {
      const notes = {
        'Cooking.md': '# Cooking\nHow to bake sourdough bread at home with a starter.',
        'Space.md': '# Space\nThe James Webb telescope observes distant galaxies.',
      };
      fs.mkdirSync(path.join(tmp, '.smart-env', 'multi'), { recursive: true });
      fs.writeFileSync(
        path.join(tmp, '.smart-env', 'smart_env.json'),
        JSON.stringify({
          smart_sources: {
            embed_model: { adapter: 'transformers', transformers: { model_key: MODEL }, [MODEL]: {} },
          },
        }),
      );
      for (const [file, content] of Object.entries(notes)) {
        fs.writeFileSync(path.join(tmp, file), content);
        const vec = await embed(content);
        fs.writeFileSync(
          path.join(tmp, '.smart-env', 'multi', `${file.replace(/\./g, '_')}.ajson`),
          `"smart_sources:${file}": ${JSON.stringify({ path: file, class_name: 'SmartSource', embeddings: { [MODEL]: { vec } }, blocks: {} })},\n`,
        );
      }
      const engine = new SearchEngine(VaultRegistry.fromPaths([tmp]), embedder);
      const res = await engine.search('astronomy and telescopes');
      expect(res.mode).toBe('semantic');
      expect(res.results[0].path).toBe('Space.md');
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  }, 300_000);
});
