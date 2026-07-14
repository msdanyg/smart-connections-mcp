import { describe, it, expect } from 'vitest';
import { VectorIndex, cosineSimilarity } from '../src/vector-index.js';

const e = (i: number) => { const v = new Array(4).fill(0); v[i] = 1; return v; };

describe('cosineSimilarity', () => {
  it('computes cosine', () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
    expect(cosineSimilarity([1, 0], [0.8, 0.6])).toBeCloseTo(0.8);
    expect(cosineSimilarity([0, 0], [1, 0])).toBe(0);
  });
  it('throws on length mismatch', () => {
    expect(() => cosineSimilarity([1], [1, 2])).toThrow();
  });
});

describe('VectorIndex', () => {
  const note = (id: string): { id: string; kind: 'note'; notePath: string } => ({ id, kind: 'note', notePath: id });

  it('normalizes on insert and ranks by dot product', () => {
    const idx = new VectorIndex(4);
    idx.set(note('a'), [10, 0, 0, 0]); // non-unit input, must be normalized
    idx.set(note('b'), [0.8, 0.6, 0, 0]);
    idx.set(note('c'), e(2));
    const top = idx.topK(e(0), 10, 0.4);
    expect(top.map((m) => m.entry.id)).toEqual(['a', 'b']);
    expect(top[0].similarity).toBeCloseTo(1);
    expect(top[1].similarity).toBeCloseTo(0.8);
  });

  it('applies k, threshold, and filter', () => {
    const idx = new VectorIndex(4);
    idx.set(note('a'), e(0));
    idx.set(note('b'), [0.8, 0.6, 0, 0]);
    idx.set({ id: 'a#B', kind: 'block', notePath: 'a' }, [0.9, 0.435889894, 0, 0]);
    expect(idx.topK(e(0), 1, 0).length).toBe(1);
    expect(idx.topK(e(0), 10, 0.85).map((m) => m.entry.id)).toEqual(['a', 'a#B']);
    expect(idx.topK(e(0), 10, 0, (en) => en.kind === 'note').map((m) => m.entry.id)).toEqual(['a', 'b']);
  });

  it('rejects wrong-dim and zero vectors; supports delete', () => {
    const idx = new VectorIndex(4);
    expect(idx.set(note('bad'), [1, 2])).toBe(false);
    expect(idx.set(note('zero'), [0, 0, 0, 0])).toBe(false);
    idx.set(note('a'), e(0));
    idx.set({ id: 'a#B', kind: 'block', notePath: 'a' }, e(1));
    idx.set(note('b'), e(2));
    expect(idx.size).toBe(3);
    idx.deleteByNotePath('a');
    expect(idx.size).toBe(1);
    idx.delete('b');
    expect(idx.size).toBe(0);
  });

  it('replaces entries with the same id', () => {
    const idx = new VectorIndex(4);
    idx.set(note('a'), e(0));
    idx.set(note('a'), e(1));
    expect(idx.size).toBe(1);
    expect(idx.topK(e(1), 1, 0.9)[0].entry.id).toBe('a');
  });
});
