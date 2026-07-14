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
