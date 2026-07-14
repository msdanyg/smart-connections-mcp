import { describe, it, expect } from 'vitest';
import * as path from 'node:path';
import * as fs from 'node:fs';
import * as os from 'node:os';
import { SearchEngine, type QueryEmbedder } from '../src/search-engine.js';
import { VaultRegistry } from '../src/vault-registry.js';
import { EmbedUnavailableError, AmbiguousNoteError, NoteNotFoundError } from '../src/errors.js';

const FIXTURE_A = path.resolve(import.meta.dirname, 'fixtures/vault-a');
const FIXTURE_B = path.resolve(import.meta.dirname, 'fixtures/vault-b');

const E1 = [1, 0, 0, 0, 0, 0, 0, 0];
const E3 = [0, 0, 1, 0, 0, 0, 0, 0];

/** Fake: queries containing "alpha" → e1, otherwise → e3. */
const fakeEmbedder: QueryEmbedder = {
  getEmbedFn: async () => async (text: string) => (text.includes('alpha') ? E1 : E3),
};

const offlineEmbedder: QueryEmbedder = {
  getEmbedFn: async () => {
    throw new EmbedUnavailableError('offline');
  },
};

function engine(embedder: QueryEmbedder = fakeEmbedder, paths = [FIXTURE_A, FIXTURE_B]) {
  return new SearchEngine(VaultRegistry.fromPaths(paths), embedder);
}

describe('search — semantic', () => {
  it('ranks notes and blocks across vaults with snippets', async () => {
    const res = await engine().search('alpha ideas', { threshold: 0.4 });
    expect(res.mode).toBe('semantic');
    const ids = res.results.map((r) => `${r.vault}:${r.path}${r.block ?? ''}`);
    expect(ids).toEqual([
      'vault-a:Alpha.md',           // 1.0
      'vault-a:Alpha.md##Intro',    // 0.8 (block)
      'vault-a:Sub/Beta.md',        // 0.6
      'vault-b:Alpha.md',           // 0.6
    ]);
    expect(res.results[0].similarity).toBeCloseTo(1);
    expect(res.results[0].snippet.startsWith('# Alpha')).toBe(true);
    expect(res.results[1].scope).toBe('block');
    expect(res.results[1].snippet).toContain('Alpha intro text');
    // Gamma (cos 0) and the frontmatter block are absent
    expect(ids.join()).not.toContain('Gamma');
    expect(ids.join()).not.toContain('frontmatter');
  });

  it('honors scope, vault, and limit', async () => {
    const notesOnly = await engine().search('alpha', { scope: 'notes', threshold: 0.4 });
    expect(notesOnly.results.every((r) => r.scope === 'note')).toBe(true);

    const aOnly = await engine().search('alpha', { vault: 'vault-a', threshold: 0.4 });
    expect(aOnly.results.every((r) => r.vault === 'vault-a')).toBe(true);

    const limited = await engine().search('alpha', { limit: 2, threshold: 0.4 });
    expect(limited.results.length).toBe(2);
  });
});

describe('search — keyword fallback', () => {
  it('falls back with an explicit mode and finds literal matches', async () => {
    const res = await engine(offlineEmbedder).search('apples');
    expect(res.mode).toBe('keyword-fallback');
    expect(res.warning).toMatch(/vault-a/);
    expect(res.results[0].path).toBe('Alpha.md');
    expect(res.results[0].snippet).toContain('apples');
  });

  it('does not crash on regex metacharacters', async () => {
    const res = await engine(offlineEmbedder).search('alpha (topics)? *');
    expect(res.mode).toBe('keyword-fallback');
    expect(Array.isArray(res.results)).toBe(true);
  });
});

describe('getSimilarNotes / graph', () => {
  it('finds similar notes within the note vault', () => {
    const sims = engine().getSimilarNotes('Gamma.md', { threshold: 0.1 });
    // Gamma = e3; every other note vector is orthogonal (cos 0 < 0.1) → empty.
    const simsA = engine().getSimilarNotes('Alpha.md', { vault: 'vault-a', threshold: 0.5 });
    expect(simsA[0].path).toBe('Sub/Beta.md'); // cos 0.6
    expect(simsA[0].vault).toBe('vault-a');
    expect(sims.length).toBe(0);
  });

  it('requires vault for ambiguous notes', () => {
    expect(() => engine().getSimilarNotes('Alpha.md')).toThrow(AmbiguousNoteError);
  });

  it('builds a connection graph', () => {
    const g = engine().getConnectionGraph('Alpha.md', { vault: 'vault-a', depth: 2, threshold: 0.5 });
    expect(g.root).toBe('Alpha.md');
    expect(g.vault).toBe('vault-a');
    expect(g.connections.some((c) => c.path === 'Sub/Beta.md' && c.depth === 1)).toBe(true);
  });

  it('propagates an error when the root note has no stored embedding', () => {
    expect(() => engine().getConnectionGraph('Plain.md', { vault: 'vault-a' })).toThrow(EmbedUnavailableError);
  });
});

describe('getNoteContent', () => {
  it('returns full content by default', () => {
    const r = engine().getNoteContent('Gamma.md') as { content: string; blocks: string[] };
    expect(r.content).toContain('gadgets');
  });

  it('extracts requested blocks', () => {
    const r = engine().getNoteContent('Alpha.md', { vault: 'vault-a', includeBlocks: ['##Intro', '##Nope'] }) as {
      extracted: Record<string, string>;
      missing: string[];
      content?: string;
    };
    expect(r.extracted['##Intro']).toContain('Alpha intro text');
    expect(r.missing).toEqual(['##Nope']);
    expect(r.content).toBeUndefined();
  });

  it('rethrows when the backing file is gone instead of reporting a missing heading', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'scmcp-engine-'));
    try {
      fs.cpSync(FIXTURE_A, tmp, { recursive: true });
      fs.rmSync(path.join(tmp, 'Alpha.md'));
      const eng = new SearchEngine(VaultRegistry.fromPaths([tmp]), fakeEmbedder);
      expect(() => eng.getNoteContent('Alpha.md', { includeBlocks: ['##Intro'] })).toThrow(NoteNotFoundError);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });
});

describe('stats and vault listing', () => {
  it('lists vaults including failures', () => {
    const reg = VaultRegistry.fromPaths([FIXTURE_A, '/nonexistent']);
    const infos = new SearchEngine(reg, fakeEmbedder).listVaults();
    expect(infos.length).toBe(2);
    expect(infos[0].status).toBe('ok');
    expect(infos[0].notes).toBe(4);
    expect(infos[1].status).toBe('error');
  });

  it('aggregates stats with totals', () => {
    const s = engine().getStats() as { vaults: unknown[]; totals: { notes: number } };
    expect(s.vaults.length).toBe(2);
    expect(s.totals.notes).toBe(5); // 4 + 1
  });
});
