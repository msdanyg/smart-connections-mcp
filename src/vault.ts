/** One Obsidian vault: its Smart Connections data, vector index, and file access. */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { blockNotePath, createVaultData, isFrontmatterBlock, parseAjson } from './ajson-loader.js';
import { BlockNotFoundError, NoteNotFoundError } from './errors.js';
import { resolveInsideVault } from './paths.js';
import type { SmartEnvConfig, VaultData } from './types.js';
import { VectorIndex } from './vector-index.js';

const RELOAD_THROTTLE_MS = 2000;
const SNIPPET_MAX = 700;
const FRONTMATTER_RE = /^---\r?\n[\s\S]*?\r?\n---\r?\n/;

function extractModelKey(config: SmartEnvConfig): string {
  const embedModel = config.smart_sources?.embed_model;
  if (!embedModel) throw new Error('No embed_model in smart_env.json');
  const adapter = embedModel.adapter;
  if (adapter && typeof embedModel[adapter] === 'object' && embedModel[adapter] !== null) {
    const modelKey = (embedModel[adapter] as Record<string, unknown>).model_key;
    if (typeof modelKey === 'string') return modelKey;
  }
  const fallback = Object.keys(embedModel).find(
    (k) => k !== 'adapter' && typeof embedModel[k] === 'object' && embedModel[k] !== null,
  );
  if (!fallback) throw new Error('No embedding model key found in smart_env.json');
  return fallback;
}

export class Vault {
  readonly name: string;
  readonly path: string;
  readonly modelKey: string;
  data: VaultData;
  index: VectorIndex;
  private fileStates = new Map<string, string>(); // filename -> "mtimeMs:size"
  private lastCheck = 0;

  private constructor(vaultPath: string, name: string, modelKey: string) {
    this.path = vaultPath;
    this.name = name;
    this.modelKey = modelKey;
    this.data = createVaultData();
    this.index = new VectorIndex(1);
  }

  static load(vaultPath: string, name: string): Vault {
    const smartEnv = path.join(vaultPath, '.smart-env');
    if (!fs.existsSync(smartEnv)) {
      throw new Error(`No .smart-env directory in ${vaultPath} — is Smart Connections installed and indexed?`);
    }
    const configPath = path.join(smartEnv, 'smart_env.json');
    if (!fs.existsSync(configPath)) throw new Error(`Missing ${configPath}`);
    const config = JSON.parse(fs.readFileSync(configPath, 'utf-8')) as SmartEnvConfig;
    const multi = path.join(smartEnv, 'multi');
    if (!fs.existsSync(multi)) throw new Error(`Missing ${multi} — no embeddings generated yet?`);
    const vault = new Vault(vaultPath, name, extractModelKey(config));
    vault.loadAll();
    return vault;
  }

  private multiDir(): string {
    return path.join(this.path, '.smart-env', 'multi');
  }

  private listAjsonFiles(): Map<string, string> {
    const dir = this.multiDir();
    const states = new Map<string, string>();
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith('.ajson')) continue;
      const st = fs.statSync(path.join(dir, f));
      if (st.isFile()) states.set(f, `${st.mtimeMs}:${st.size}`);
    }
    return states;
  }

  private parseFile(filename: string): void {
    const content = fs.readFileSync(path.join(this.multiDir(), filename), 'utf-8');
    parseAjson(content, this.data, (msg) => console.error(`[${this.name}] ${filename}: ${msg}`));
  }

  private loadAll(): void {
    this.data = createVaultData();
    this.fileStates = this.listAjsonFiles();
    for (const filename of this.fileStates.keys()) this.parseFile(filename);
    this.rebuildIndex();
  }

  private rebuildIndex(): void {
    let dim = 0;
    for (const s of this.data.sources.values()) {
      const v = s.embeddings?.[this.modelKey]?.vec;
      if (v?.length) { dim = v.length; break; }
    }
    if (!dim) {
      for (const b of this.data.blocks.values()) {
        const v = b.embeddings?.[this.modelKey]?.vec;
        if (v?.length) { dim = v.length; break; }
      }
    }
    this.index = new VectorIndex(dim || 1);
    for (const [key, s] of this.data.sources) {
      const v = s.embeddings?.[this.modelKey]?.vec;
      if (v) this.index.set({ id: key, kind: 'note', notePath: key }, v);
    }
    for (const [key, b] of this.data.blocks) {
      if (isFrontmatterBlock(key)) continue;
      const v = b.embeddings?.[this.modelKey]?.vec;
      if (v) this.index.set({ id: key, kind: 'block', notePath: blockNotePath(key) }, v);
    }
  }

  maybeReload(now: number = Date.now()): void {
    if (now - this.lastCheck < RELOAD_THROTTLE_MS) return;
    this.lastCheck = now;
    const current = this.listAjsonFiles();
    const removed = [...this.fileStates.keys()].some((f) => !current.has(f));
    if (removed) {
      this.loadAll();
      return;
    }
    const changed = [...current.entries()].filter(([f, state]) => this.fileStates.get(f) !== state);
    if (changed.length === 0) return;
    for (const [f, state] of changed) {
      this.parseFile(f);
      this.fileStates.set(f, state);
    }
    this.rebuildIndex();
  }

  readNote(notePath: string): string {
    const full = resolveInsideVault(this.path, notePath);
    if (!fs.existsSync(full)) throw new NoteNotFoundError(`Note file not found: ${notePath} (vault ${this.name})`);
    return fs.readFileSync(full, 'utf-8');
  }

  extractBlockByHeading(notePath: string, heading: string): string {
    const source = this.data.sources.get(notePath);
    const range = source?.blocks?.[heading];
    if (!range) {
      const available = Object.keys(source?.blocks ?? {}).join(', ') || '(none)';
      throw new BlockNotFoundError(`Block "${heading}" not found in ${notePath}. Available: ${available}`);
    }
    const lines = this.readNote(notePath).split('\n');
    return lines.slice(range[0] - 1, range[1]).join('\n');
  }

  blockSnippet(blockKey: string, maxLen: number = SNIPPET_MAX): string {
    const notePath = blockNotePath(blockKey);
    const block = this.data.blocks.get(blockKey);
    const range = block?.lines ?? this.data.sources.get(notePath)?.blocks?.[blockKey.slice(notePath.length)];
    if (!range) return '';
    try {
      const lines = this.readNote(notePath).split('\n');
      return lines.slice(range[0] - 1, range[1]).join('\n').slice(0, maxLen);
    } catch {
      return '';
    }
  }

  noteSnippet(notePath: string, maxLen: number = SNIPPET_MAX): string {
    try {
      return this.readNote(notePath).replace(FRONTMATTER_RE, '').trimStart().slice(0, maxLen);
    } catch {
      return '';
    }
  }

  paritySample(): { text: string; vec: number[] } | undefined {
    for (const [key, s] of this.data.sources) {
      const vec = s.embeddings?.[this.modelKey]?.vec;
      if (!vec) continue;
      try {
        return { text: this.readNote(key), vec };
      } catch {
        continue;
      }
    }
    return undefined;
  }

  stats(): { notes: number; blocks: number; indexed: number; embeddingDim: number; modelKey: string } {
    return {
      notes: this.data.sources.size,
      blocks: this.data.blocks.size,
      indexed: this.index.size,
      embeddingDim: this.index.dim,
      modelKey: this.modelKey,
    };
  }
}
