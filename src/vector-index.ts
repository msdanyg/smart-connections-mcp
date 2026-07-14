/** In-memory vector store. Vectors are L2-normalized on insert; similarity = dot product. */

export interface IndexEntry {
  id: string;
  kind: 'note' | 'block';
  notePath: string;
}

export interface Match {
  entry: IndexEntry;
  similarity: number;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error('Vectors must have the same length');
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const mag = Math.sqrt(na) * Math.sqrt(nb);
  return mag === 0 ? 0 : dot / mag;
}

function normalize(vec: number[]): Float32Array | null {
  const f = Float32Array.from(vec);
  let norm = 0;
  for (let i = 0; i < f.length; i++) norm += f[i] * f[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return null;
  for (let i = 0; i < f.length; i++) f[i] /= norm;
  return f;
}

export class VectorIndex {
  readonly dim: number;
  private items = new Map<string, { entry: IndexEntry; vec: Float32Array }>();

  constructor(dim: number) {
    this.dim = dim;
  }

  get size(): number {
    return this.items.size;
  }

  set(entry: IndexEntry, vec: number[]): boolean {
    if (vec.length !== this.dim) return false;
    const f = normalize(vec);
    if (!f) return false;
    this.items.set(entry.id, { entry, vec: f });
    return true;
  }

  delete(id: string): void {
    this.items.delete(id);
  }

  deleteByNotePath(notePath: string): void {
    for (const [id, item] of this.items) {
      if (item.entry.notePath === notePath) this.items.delete(id);
    }
  }

  topK(query: number[], k: number, threshold: number, filter?: (e: IndexEntry) => boolean): Match[] {
    if (query.length !== this.dim) return [];
    const q = normalize(query);
    if (!q) return [];
    const matches: Match[] = [];
    for (const { entry, vec } of this.items.values()) {
      if (filter && !filter(entry)) continue;
      let dot = 0;
      for (let i = 0; i < vec.length; i++) dot += vec[i] * q[i];
      if (dot >= threshold) matches.push({ entry, similarity: dot });
    }
    matches.sort((a, b) => b.similarity - a.similarity);
    return matches.slice(0, k);
  }
}
