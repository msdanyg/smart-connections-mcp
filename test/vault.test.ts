import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { Vault } from '../src/vault.js';
import { NoteNotFoundError, BlockNotFoundError, PathEscapeError } from '../src/errors.js';

const FIXTURE_A = path.resolve(import.meta.dirname, 'fixtures/vault-a');

describe('Vault.load', () => {
  it('loads fixture vault-a and builds the index', () => {
    const v = Vault.load(FIXTURE_A, 'vault-a');
    expect(v.modelKey).toBe('test-model-8d');
    const s = v.stats();
    expect(s.notes).toBe(4); // Alpha, Sub/Beta, Gamma, Plain
    expect(s.blocks).toBe(2); // frontmatter + Intro
    expect(s.indexed).toBe(4); // 3 note vecs + 1 non-frontmatter block vec
    expect(s.embeddingDim).toBe(8);
  });

  it('throws a clear error when .smart-env is missing', () => {
    expect(() => Vault.load(os.tmpdir(), 'x')).toThrow(/smart-env/i);
  });
});

describe('Vault reads', () => {
  const v = Vault.load(FIXTURE_A, 'vault-a');

  it('reads notes, sandboxed', () => {
    expect(v.readNote('Alpha.md')).toContain('about apples');
    expect(() => v.readNote('Missing.md')).toThrow(NoteNotFoundError);
    expect(() => v.readNote('../../etc/passwd')).toThrow(PathEscapeError);
  });

  it('extracts blocks by heading', () => {
    expect(v.extractBlockByHeading('Alpha.md', '##Intro')).toBe('## Intro\nAlpha intro text about apples.\nMore intro.');
    expect(() => v.extractBlockByHeading('Alpha.md', '##Nope')).toThrow(BlockNotFoundError);
  });

  it('builds snippets', () => {
    expect(v.blockSnippet('Alpha.md##Intro')).toContain('Alpha intro text');
    const ns = v.noteSnippet('Alpha.md');
    expect(ns.startsWith('# Alpha')).toBe(true); // frontmatter stripped
    expect(v.noteSnippet('Missing.md')).toBe('');
  });

  it('provides a parity sample', () => {
    const p = v.paritySample();
    expect(p?.vec.length).toBe(8);
    expect(p?.text.length).toBeGreaterThan(0);
  });
});

describe('Vault.maybeReload', () => {
  let tmp: string;
  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'scmcp-'));
    fs.cpSync(FIXTURE_A, tmp, { recursive: true });
  });
  afterEach(() => fs.rmSync(tmp, { recursive: true, force: true }));

  it('picks up appended entries, honoring the throttle', () => {
    const v = Vault.load(tmp, 't');
    const t0 = 1_000_000;
    expect(v.stats().notes).toBe(4);

    fs.writeFileSync(path.join(tmp, 'Delta.md'), '# Delta\nDelta note.\n');
    fs.appendFileSync(
      path.join(tmp, '.smart-env/multi/Gamma_md.ajson'),
      '"smart_sources:Delta.md": {"path":"Delta.md","class_name":"SmartSource","embeddings":{"test-model-8d":{"vec":[0,0,0,1,0,0,0,0]}},"blocks":{}},\n',
    );

    v.maybeReload(t0); // first check: records change
    expect(v.stats().notes).toBe(5);
    expect(v.index.topK([0, 0, 0, 1, 0, 0, 0, 0], 1, 0.9)[0].entry.id).toBe('Delta.md');

    // throttled: a change within 2s of the last check is not seen yet
    fs.appendFileSync(
      path.join(tmp, '.smart-env/multi/Gamma_md.ajson'),
      '"smart_sources:Delta.md": null,\n',
    );
    v.maybeReload(t0 + 500);
    expect(v.stats().notes).toBe(5);
    v.maybeReload(t0 + 2500);
    expect(v.stats().notes).toBe(4);
  });

  it('handles a removed ajson file with a full reload', () => {
    const v = Vault.load(tmp, 't');
    fs.rmSync(path.join(tmp, '.smart-env/multi/Sub_Beta_md.ajson'));
    v.maybeReload(5_000_000);
    expect(v.stats().notes).toBe(3);
    expect(v.data.sources.has('Sub/Beta.md')).toBe(false);
  });
});

describe('Vault model key reconciliation (issue #9)', () => {
  let tmp: string;
  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'scmcp-key-'));
    fs.cpSync(FIXTURE_A, tmp, { recursive: true });
  });
  afterEach(() => fs.rmSync(tmp, { recursive: true, force: true }));

  const declare = (modelKey: string) =>
    fs.writeFileSync(
      path.join(tmp, '.smart-env/smart_env.json'),
      JSON.stringify({
        smart_sources: { embed_model: { adapter: 'transformers', transformers: { model_key: modelKey } } },
      }),
    );

  it('trusts the embeddings over a stale smart_env.json model_key', () => {
    declare('stale/model');
    const v = Vault.load(tmp, 't');
    expect(v.declaredModelKey).toBe('stale/model');
    expect(v.modelKey).toBe('test-model-8d');
    const s = v.stats();
    expect(s.indexed).toBe(4);
    expect(s.embeddingDim).toBe(8);
    expect(s.modelKey).toBe('test-model-8d');
    expect(s.declaredModelKey).toBe('stale/model');
  });

  it('keeps the declared key when it has vectors, and omits declaredModelKey from stats', () => {
    const v = Vault.load(tmp, 't');
    expect(v.modelKey).toBe('test-model-8d');
    expect(v.declaredModelKey).toBe('test-model-8d');
    expect(v.stats().declaredModelKey).toBeUndefined();
  });

  it('leaves the declared key alone when nothing is embedded at all', () => {
    declare('stale/model');
    stripEmbeddings(tmp);
    const v = Vault.load(tmp, 't');
    expect(v.modelKey).toBe('stale/model');
    expect(v.stats().indexed).toBe(0);
  });

  it('re-reconciles after a reload that re-embeds under a new key', () => {
    const v = Vault.load(tmp, 't');
    expect(v.modelKey).toBe('test-model-8d');
    const multi = path.join(tmp, '.smart-env/multi');
    for (const f of fs.readdirSync(multi)) {
      const rewritten = fs.readFileSync(path.join(multi, f), 'utf-8').replaceAll('"test-model-8d"', '"new-model-8d"');
      fs.writeFileSync(path.join(multi, f), rewritten);
    }
    v.maybeReload(9_000_000);
    expect(v.modelKey).toBe('new-model-8d');
    expect(v.stats().indexed).toBe(4);
  });
});

/** Drop every `embeddings` object from a fixture copy's .ajson files. */
function stripEmbeddings(vaultDir: string): void {
  const multi = path.join(vaultDir, '.smart-env/multi');
  for (const f of fs.readdirSync(multi)) {
    const lines = fs.readFileSync(path.join(multi, f), 'utf-8').split('\n');
    const out = lines.map((line) => {
      const m = /^("[^"]+"): (\{.*\}),?$/.exec(line.trim());
      if (!m) return line;
      const obj = JSON.parse(m[2]) as { embeddings?: unknown };
      delete obj.embeddings;
      return `${m[1]}: ${JSON.stringify(obj)},`;
    });
    fs.writeFileSync(path.join(multi, f), out.join('\n'));
  }
}
