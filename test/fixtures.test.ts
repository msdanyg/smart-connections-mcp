import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { createVaultData, parseAjson } from '../src/ajson-loader.js';

const FIXTURE_A = path.resolve(import.meta.dirname, 'fixtures/vault-a');
const FIXTURE_B = path.resolve(import.meta.dirname, 'fixtures/vault-b');

describe('fixture vault-a', () => {
  it('loads via the parser with expected contents', () => {
    const data = createVaultData();
    const multi = path.join(FIXTURE_A, '.smart-env', 'multi');
    for (const f of fs.readdirSync(multi).filter((f) => f.endsWith('.ajson'))) {
      parseAjson(fs.readFileSync(path.join(multi, f), 'utf-8'), data);
    }
    expect([...data.sources.keys()].sort()).toEqual(['Alpha.md', 'Gamma.md', 'Plain.md', 'Sub/Beta.md']);
    expect(data.sources.get('Gamma.md')?.embeddings?.['test-model-8d']?.vec).toEqual([0, 0, 1, 0, 0, 0, 0, 0]);
    expect(data.blocks.size).toBe(2);
    // block line ranges match the actual markdown
    const alpha = fs.readFileSync(path.join(FIXTURE_A, 'Alpha.md'), 'utf-8').split('\n');
    expect(alpha[5]).toBe('## Intro'); // line 6, 1-indexed
    expect(alpha[7]).toBe('More intro.'); // line 8
  });
});