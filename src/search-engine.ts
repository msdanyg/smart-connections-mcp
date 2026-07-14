/** Orchestrates search, similarity, graphs, content, and stats across vaults. */

import type { EmbedFn } from './embedder.js';
import { BlockNotFoundError, EmbedUnavailableError } from './errors.js';
import type {
  ConnectionGraph,
  SearchResponse,
  SearchResult,
  SimilarNote,
  VaultInfo,
} from './types.js';
import type { Vault } from './vault.js';
import type { VaultRegistry } from './vault-registry.js';
import type { IndexEntry } from './vector-index.js';

export interface QueryEmbedder {
  getEmbedFn(
    modelKey: string,
    parity?: { text: string; vec: number[] },
    warn?: (msg: string) => void,
  ): Promise<EmbedFn>;
}

const SNIPPET_MAX = 700;
const round = (n: number) => Math.round(n * 10_000) / 10_000;

export class SearchEngine {
  constructor(
    private registry: VaultRegistry,
    private embedder: QueryEmbedder,
  ) {}

  async search(
    query: string,
    opts: { vault?: string; scope?: 'notes' | 'blocks' | 'both'; limit?: number; threshold?: number } = {},
  ): Promise<SearchResponse> {
    const { vault, scope = 'both', limit = 10, threshold = 0.4 } = opts;
    const vaults = this.registry.byName(vault);
    const warnings: string[] = [];
    const fallbackVaults: string[] = [];
    const results: SearchResult[] = [];

    const scopeFilter =
      scope === 'both' ? undefined : (e: IndexEntry) => e.kind === (scope === 'notes' ? 'note' : 'block');

    for (const v of vaults) {
      v.maybeReload();
      try {
        const embed = await this.embedder.getEmbedFn(v.modelKey, v.paritySample(), (m) =>
          warnings.push(`${v.name}: ${m}`),
        );
        const qvec = await embed(query);
        for (const m of v.index.topK(qvec, limit, threshold, scopeFilter)) {
          results.push(this.toResult(v, m.entry, m.similarity));
        }
      } catch (e) {
        if (!(e instanceof EmbedUnavailableError)) throw e;
        fallbackVaults.push(v.name);
        results.push(...this.keywordSearch(v, query, limit));
      }
    }

    results.sort((a, b) => b.similarity - a.similarity);
    const allFellBack = fallbackVaults.length === vaults.length && vaults.length > 0;
    if (fallbackVaults.length > 0) {
      warnings.push(
        `semantic model unavailable for ${fallbackVaults.join(', ')} — used literal keyword matching there ` +
          `(scores are match counts, not cosine similarity)`,
      );
    }
    return {
      mode: allFellBack ? 'keyword-fallback' : 'semantic',
      ...(warnings.length ? { warning: warnings.join(' | ') } : {}),
      results: results.slice(0, limit),
    };
  }

  private toResult(v: Vault, entry: IndexEntry, similarity: number): SearchResult {
    if (entry.kind === 'block') {
      return {
        path: entry.notePath,
        vault: v.name,
        similarity: round(similarity),
        scope: 'block',
        block: entry.id.slice(entry.notePath.length),
        snippet: v.blockSnippet(entry.id),
      };
    }
    return {
      path: entry.notePath,
      vault: v.name,
      similarity: round(similarity),
      scope: 'note',
      snippet: v.noteSnippet(entry.notePath),
    };
  }

  /** Literal keyword scoring — no RegExp built from user input, ever. */
  private keywordSearch(v: Vault, query: string, limit: number): SearchResult[] {
    const tokens = query.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
    if (tokens.length === 0) return [];
    const out: SearchResult[] = [];
    for (const notePath of v.data.sources.keys()) {
      let raw: string;
      try {
        raw = v.readNote(notePath);
      } catch {
        continue;
      }
      const content = raw.toLowerCase();
      let total = 0;
      let firstIdx = -1;
      for (const t of tokens) {
        let idx = content.indexOf(t);
        if (idx !== -1 && (firstIdx === -1 || idx < firstIdx)) firstIdx = idx;
        while (idx !== -1) {
          total++;
          idx = content.indexOf(t, idx + t.length);
        }
      }
      if (total === 0) continue;
      const start = Math.max(0, firstIdx - 200);
      out.push({
        path: notePath,
        vault: v.name,
        similarity: round(Math.min(total / 10, 1)),
        scope: 'note',
        snippet: raw.slice(start, start + SNIPPET_MAX),
      });
    }
    return out.sort((a, b) => b.similarity - a.similarity).slice(0, limit);
  }

  getSimilarNotes(
    notePath: string,
    opts: { vault?: string; threshold?: number; limit?: number } = {},
  ): SimilarNote[] {
    const { vault, threshold = 0.5, limit = 10 } = opts;
    const v = this.registry.resolveNote(notePath, vault);
    v.maybeReload();
    const vec = v.data.sources.get(notePath)?.embeddings?.[v.modelKey]?.vec;
    if (!vec) throw new EmbedUnavailableError(`No stored embedding for note: ${notePath}`);
    return v.index
      .topK(vec, limit, threshold, (e) => e.kind === 'note' && e.notePath !== notePath)
      .map((m) => ({
        path: m.entry.notePath,
        vault: v.name,
        similarity: round(m.similarity),
        blocks: Object.keys(v.data.sources.get(m.entry.notePath)?.blocks ?? {}),
      }));
  }

  getConnectionGraph(
    notePath: string,
    opts: { vault?: string; depth?: number; threshold?: number; maxPerLevel?: number } = {},
  ): ConnectionGraph {
    const { vault, depth = 2, threshold = 0.6, maxPerLevel = 5 } = opts;
    const v = this.registry.resolveNote(notePath, vault);
    const visited = new Set<string>();
    const connections: ConnectionGraph['connections'] = [];

    const walk = (current: string, level: number, similarity: number): void => {
      visited.add(current);
      if (level > 0) connections.push({ path: current, depth: level, similarity: round(similarity) });
      if (level >= depth) return;
      let similar: SimilarNote[];
      try {
        similar = this.getSimilarNotes(current, { vault: v.name, threshold, limit: maxPerLevel });
      } catch (e) {
        // A node beyond the root may lack a stored embedding — skip expanding it.
        if (level > 0 && e instanceof EmbedUnavailableError) return;
        throw e;
      }
      for (const s of similar) {
        if (!visited.has(s.path)) walk(s.path, level + 1, s.similarity);
      }
    };

    walk(notePath, 0, 1);
    return { root: notePath, vault: v.name, connections };
  }

  getNoteContent(
    notePath: string,
    opts: { vault?: string; includeBlocks?: string[] } = {},
  ): object {
    const v = this.registry.resolveNote(notePath, opts.vault);
    v.maybeReload();
    const blocks = Object.keys(v.data.sources.get(notePath)?.blocks ?? {});
    if (opts.includeBlocks && opts.includeBlocks.length > 0) {
      const extracted: Record<string, string> = {};
      const missing: string[] = [];
      for (const heading of opts.includeBlocks) {
        try {
          extracted[heading] = v.extractBlockByHeading(notePath, heading);
        } catch (e) {
          if (e instanceof BlockNotFoundError) {
            missing.push(heading);
            continue;
          }
          throw e;
        }
      }
      return { path: notePath, vault: v.name, blocks, extracted, missing };
    }
    return { path: notePath, vault: v.name, blocks, content: v.readNote(notePath) };
  }

  listVaults(): VaultInfo[] {
    const ok: VaultInfo[] = this.registry.vaults.map((v) => ({
      name: v.name,
      path: v.path,
      status: 'ok' as const,
      ...v.stats(),
    }));
    const failed: VaultInfo[] = this.registry.failures.map((f) => ({
      name: f.name,
      path: f.path,
      status: 'error' as const,
      error: f.error,
    }));
    return [...ok, ...failed];
  }

  getStats(vaultName?: string): object {
    const vaults = this.registry.byName(vaultName);
    const perVault = vaults.map((v) => ({ name: v.name, ...v.stats() }));
    return {
      vaults: perVault,
      totals: {
        notes: perVault.reduce((sum, s) => sum + s.notes, 0),
        blocks: perVault.reduce((sum, s) => sum + s.blocks, 0),
        indexed: perVault.reduce((sum, s) => sum + s.indexed, 0),
      },
    };
  }
}
