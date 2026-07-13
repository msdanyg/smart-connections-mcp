# Smart Connections MCP V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild smart-connections-mcp as v2.0.0: true local semantic search, multi-vault support, block-level retrieval, freshness, security fixes, and a real test suite, per the approved spec at `docs/superpowers/specs/2026-07-13-smart-connections-mcp-v2-design.md`.

**Architecture:** A vault registry loads each Obsidian vault's `.smart-env` data (sources + blocks) into a per-vault in-memory vector index of pre-normalized Float32Arrays. A lazy embedder runs each vault's own embedding model locally via transformers.js to embed query text. A thin MCP layer (current SDK, `McpServer` + zod) exposes six tools. Everything below the MCP layer is plain testable classes; the embedder is injected so CI never downloads a model.

**Tech Stack:** TypeScript (ESM, NodeNext), `@modelcontextprotocol/sdk` ^1.29.0, `zod` ^3.25.0, `@huggingface/transformers` ^4.2.0, `vitest` ^4.1.10, Node >= 20.

## Global Constraints

- Package stays `smart-connections-mcp`, bin stays `smart-connections-mcp`, version becomes `2.0.0`.
- ESM only (`"type": "module"`), `module`/`moduleResolution` NodeNext, `target` ES2022, `strict: true`. All relative imports end in `.js`.
- Node engine floor: `">=20"`.
- Tool names (exactly): `search_notes`, `get_similar_notes`, `get_connection_graph`, `get_note_content`, `get_stats`, `list_vaults`. `get_embedding_neighbors` is removed.
- Defaults: search threshold `0.4`, similar-notes threshold `0.5`, graph threshold `0.6`, limit `10`, graph depth `2`, graph max_per_level `5`, snippet max `700` chars, reload throttle `2000` ms, parity warning threshold cosine `0.99` (warn-only, never fatal).
- Never interpolate user input into a `RegExp`. Never read files outside a vault root (`resolveInsideVault` is the only path join for note reads).
- Env config: `SMART_VAULT_PATHS` (alias, wins) or `SMART_VAULT_PATH`, comma-separated absolute paths.
- All work on branch `v2`. Every task ends with a commit; commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- `npm test` must pass at the end of every task (it runs `tsc` then vitest; live tests excluded unless `LIVE_TESTS=1`).
- Frontmatter blocks (key contains `#---frontmatter---`) are never indexed for search.
- stderr is the only logging channel (stdout belongs to the MCP stdio transport).

## File Map

| File | Responsibility | Task |
|---|---|---|
| `src/types.ts` | Data shapes for `.smart-env` entries and results | 1 |
| `src/errors.ts` | Typed error classes | 1 |
| `src/ajson-loader.ts` | Parse `.ajson` (sources + blocks, overrides, null-deletes) | 2 |
| `test/fixtures/vault-a`, `vault-b` | Synthetic vaults (8-dim vectors) | 3 |
| `src/paths.ts` | Vault path sandbox | 4 |
| `src/vector-index.ts` | Normalized vector store + top-k + cosine | 5 |
| `src/embedder.ts` | Lazy transformers.js pipelines, parity check, fallback error | 6 |
| `src/vault.ts` | One vault: load, index, freshness, note/block reads | 7 |
| `src/vault-registry.ts` | Env parsing, multi-vault, name/note resolution | 8 |
| `src/search-engine.ts` | Search/similar/graph/content/stats/list-vaults | 9 |
| `src/server.ts` | McpServer + 6 tool registrations | 10 |
| `src/index.ts` | bin entry (env → registry → server → stdio) | 11 |
| `test/live/embedder.live.test.ts`, `scripts/smoke.mjs` | Opt-in real-model tests, manual smoke | 11 |
| `README.md`, `CHANGELOG.md`, `server.json`, docs | Release prep | 12 |

Deleted in Task 1: `src/smart-connections-loader.ts`, `src/embedding-utils.ts`, `src/search-engine.ts` (old), `src/index.ts` (old), `src/types.ts` (old, replaced), `test-*.mjs` (4 files), `RESTART-INSTRUCTIONS.md`, tracked `dist/`.

---

### Task 1: Tooling reset (branch, deps, vitest, fresh types/errors)

**Files:**
- Modify: `package.json`, `tsconfig.json`, `.gitignore`
- Create: `vitest.config.ts`, `src/types.ts` (rewrite), `src/errors.ts`, `test/smoke.test.ts`
- Delete: `src/smart-connections-loader.ts`, `src/embedding-utils.ts`, `src/search-engine.ts`, `src/index.ts`, `test-all-tools.mjs`, `test-embeddings.mjs`, `test-mcp-server.mjs`, `test-search.mjs`, `RESTART-INSTRUCTIONS.md`, tracked `dist/`

**Interfaces:**
- Produces: all types in `src/types.ts` and error classes in `src/errors.ts` below — later tasks import them verbatim.

- [ ] **Step 1: Branch and delete legacy files**

```bash
git checkout -b v2
git rm -q test-all-tools.mjs test-embeddings.mjs test-mcp-server.mjs test-search.mjs RESTART-INSTRUCTIONS.md
git rm -q src/smart-connections-loader.ts src/embedding-utils.ts src/search-engine.ts src/index.ts src/types.ts
git rm -rq --cached dist
echo "dist/" >> .gitignore
rm -rf dist
```

- [ ] **Step 2: Update package.json**

Replace the `version`, `description`, `engines`, `scripts`, `dependencies`, `devDependencies` fields (keep everything else — name, mcpName, bin, files, keywords, author, license, repo fields — as-is):

```json
{
  "version": "2.0.0",
  "description": "MCP server for true local semantic search over Obsidian vaults using Smart Connections embeddings — multi-vault, block-level retrieval, no cloud calls",
  "engines": { "node": ">=20" },
  "scripts": {
    "build": "tsc",
    "watch": "tsc --watch",
    "start": "node dist/index.js",
    "test": "npm run build && vitest run",
    "test:live": "LIVE_TESTS=1 vitest run",
    "smoke": "node scripts/smoke.mjs",
    "prepublishOnly": "npm run build"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.29.0",
    "@huggingface/transformers": "^4.2.0",
    "zod": "^3.25.0"
  },
  "devDependencies": {
    "@types/node": "^22.10.5",
    "typescript": "^5.7.2",
    "vitest": "^4.1.10"
  }
}
```

Then run: `npm install` (expect success; transformers is large, ~1–2 min).

- [ ] **Step 3: Write tsconfig.json (full replacement)**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true
  },
  "include": ["src"]
}
```

- [ ] **Step 4: Write vitest.config.ts**

```ts
import { defineConfig, configDefaults } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
    exclude: [
      ...configDefaults.exclude,
      ...(process.env.LIVE_TESTS ? [] : ['test/live/**']),
    ],
    testTimeout: 30_000,
  },
});
```

- [ ] **Step 5: Write src/types.ts (complete replacement)**

```ts
/** Shapes of Smart Connections `.smart-env` data (v3.x) and server results. */

export interface EmbeddingData {
  vec?: number[];
}

export interface SmartSource {
  path: string;
  class_name?: string;
  embeddings?: Record<string, EmbeddingData>;
  /** heading key -> [startLine, endLine], 1-indexed inclusive */
  blocks?: Record<string, [number, number]>;
  last_import?: { mtime?: number; size?: number };
  metadata?: Record<string, unknown>;
}

export interface SmartBlock {
  /** e.g. "Note.md##Heading#{2}" — always set (from the AJSON key if absent) */
  key: string;
  path?: string | null;
  class_name?: string;
  /** [startLine, endLine], 1-indexed inclusive */
  lines?: [number, number];
  embeddings?: Record<string, EmbeddingData>;
  size?: number;
}

export interface VaultData {
  /** note path -> source */
  sources: Map<string, SmartSource>;
  /** block key -> block */
  blocks: Map<string, SmartBlock>;
}

export interface SmartEnvConfig {
  smart_sources?: {
    embed_model?: { adapter?: string; [key: string]: unknown };
  };
}

export interface SearchResult {
  path: string;
  vault: string;
  similarity: number;
  scope: 'note' | 'block';
  /** heading portion of the block key, e.g. "##Intro" — block hits only */
  block?: string;
  snippet: string;
}

export interface SearchResponse {
  mode: 'semantic' | 'keyword-fallback';
  warning?: string;
  results: SearchResult[];
}

export interface SimilarNote {
  path: string;
  vault: string;
  similarity: number;
  blocks: string[];
}

export interface ConnectionGraph {
  root: string;
  vault: string;
  connections: Array<{ path: string; depth: number; similarity: number }>;
}

export interface VaultInfo {
  name: string;
  path: string;
  status: 'ok' | 'error';
  error?: string;
  notes?: number;
  blocks?: number;
  indexed?: number;
  embeddingDim?: number;
  modelKey?: string;
}
```

- [ ] **Step 6: Write src/errors.ts**

```ts
export class NoteNotFoundError extends Error {}
export class BlockNotFoundError extends Error {}
export class AmbiguousNoteError extends Error {}
export class VaultNotFoundError extends Error {}
export class PathEscapeError extends Error {}
export class EmbedUnavailableError extends Error {}
```

- [ ] **Step 7: Write test/smoke.test.ts**

```ts
import { describe, it, expect } from 'vitest';
import { NoteNotFoundError } from '../src/errors.js';

describe('toolchain smoke', () => {
  it('compiles and imports src modules', () => {
    expect(new NoteNotFoundError('x')).toBeInstanceOf(Error);
  });
});
```

- [ ] **Step 8: Run the suite**

Run: `npm test`
Expected: tsc succeeds, vitest reports `1 passed`.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore!: v2 tooling reset — modern deps, vitest, fresh types; drop legacy scripts and tracked dist

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: AJSON parser

**Files:**
- Create: `src/ajson-loader.ts`
- Test: `test/ajson-loader.test.ts`

**Interfaces:**
- Consumes: `VaultData`, `SmartSource`, `SmartBlock` from `src/types.ts`.
- Produces: `createVaultData(): VaultData`; `parseAjson(content: string, data: VaultData, warn?: (msg: string) => void): void`; `blockNotePath(blockKey: string): string`; `isFrontmatterBlock(blockKey: string): boolean`.

- [ ] **Step 1: Write the failing test — test/ajson-loader.test.ts**

```ts
import { describe, it, expect } from 'vitest';
import { createVaultData, parseAjson, blockNotePath, isFrontmatterBlock } from '../src/ajson-loader.js';

const SAMPLE = `
"smart_sources:Alpha.md": {"path":"Alpha.md","class_name":"SmartSource","embeddings":{"m":{"vec":[1,0]}},"blocks":{"##Intro":[6,8]}},
"smart_blocks:Alpha.md##Intro": {"path":null,"class_name":"SmartBlock","lines":[6,8],"embeddings":{"m":{"vec":[0.8,0.6]}}},
"smart_sources:Old.md": {"path":"Old.md","embeddings":{"m":{"vec":[0,1]}}},
"smart_sources:Old.md": {"path":"Old.md","embeddings":{"m":{"vec":[1,1]}}},
"smart_sources:Gone.md": {"path":"Gone.md"},
"smart_sources:Gone.md": null,
this line is not json at all
"smart_sources:NoPath.md": {"class_name":"SmartSource"},
`;

describe('parseAjson', () => {
  it('parses sources and blocks, applies overrides and null deletions, skips junk', () => {
    const data = createVaultData();
    const warnings: string[] = [];
    parseAjson(SAMPLE, data, (m) => warnings.push(m));

    expect([...data.sources.keys()].sort()).toEqual(['Alpha.md', 'Old.md']);
    expect(data.sources.get('Old.md')?.embeddings?.m?.vec).toEqual([1, 1]); // later line wins
    expect(data.sources.has('Gone.md')).toBe(false); // null deletes
    expect(data.sources.has('NoPath.md')).toBe(false); // entries without path are dropped

    expect(data.blocks.size).toBe(1);
    const block = data.blocks.get('Alpha.md##Intro');
    expect(block?.key).toBe('Alpha.md##Intro'); // key backfilled from AJSON key
    expect(block?.lines).toEqual([6, 8]);

    expect(warnings.length).toBe(1); // the junk line
  });

  it('block helpers', () => {
    expect(blockNotePath('Sub/Note.md##A#{2}')).toBe('Sub/Note.md');
    expect(blockNotePath('Note.md')).toBe('Note.md');
    expect(isFrontmatterBlock('Note.md#---frontmatter---')).toBe(true);
    expect(isFrontmatterBlock('Note.md##Intro')).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/ajson-loader.test.ts`
Expected: FAIL — cannot find module `../src/ajson-loader.js`.

- [ ] **Step 3: Write src/ajson-loader.ts**

```ts
/**
 * Parser for Smart Connections AJSON files (`.smart-env/multi/*.ajson`).
 * Format: one entry per line, `"collection:key": {json},` — append semantics:
 * later lines override earlier ones, a null value deletes the key.
 */

import type { SmartBlock, SmartSource, VaultData } from './types.js';

export function createVaultData(): VaultData {
  return { sources: new Map(), blocks: new Map() };
}

export function parseAjson(
  content: string,
  data: VaultData,
  warn: (msg: string) => void = () => {},
): void {
  const lines = content.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim().replace(/,\s*$/, '');
    if (!line) continue;
    let obj: Record<string, unknown>;
    try {
      obj = JSON.parse(`{${line}}`);
    } catch {
      warn(`skipping unparseable line ${i + 1}`);
      continue;
    }
    for (const [rawKey, value] of Object.entries(obj)) {
      const sep = rawKey.indexOf(':');
      if (sep === -1) continue;
      const collection = rawKey.slice(0, sep);
      const key = rawKey.slice(sep + 1);
      if (collection === 'smart_sources') {
        if (value === null) {
          data.sources.delete(key);
        } else if (typeof value === 'object' && (value as SmartSource).path) {
          data.sources.set(key, value as SmartSource);
        }
      } else if (collection === 'smart_blocks') {
        if (value === null) {
          data.blocks.delete(key);
        } else if (typeof value === 'object') {
          const block = value as SmartBlock;
          block.key = block.key ?? key;
          data.blocks.set(key, block);
        }
      }
    }
  }
}

/** Note path a block belongs to: portion of its key before the first '#'. */
export function blockNotePath(blockKey: string): string {
  const i = blockKey.indexOf('#');
  return i === -1 ? blockKey : blockKey.slice(0, i);
}

export function isFrontmatterBlock(blockKey: string): boolean {
  return blockKey.includes('#---frontmatter---');
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run test/ajson-loader.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ajson-loader.ts test/ajson-loader.test.ts
git commit -m "feat: AJSON parser for smart_sources and smart_blocks with override/delete semantics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Fixture vaults

**Files:**
- Create: `test/fixtures/vault-a/` (smart-env + 4 md files), `test/fixtures/vault-b/` (smart-env + 1 md file)
- Test: `test/fixtures.test.ts`

**Interfaces:**
- Produces: fixture paths used by every later test: `test/fixtures/vault-a`, `test/fixtures/vault-b`. Model key in both fixtures: `test-model-8d` (8-dim vectors).
- Fixture vector geometry (all unit vectors, dims 8): `Alpha.md` = e1 `[1,0,0,0,0,0,0,0]`; block `Alpha.md##Intro` = `[0.8,0.6,0,...]` (cos 0.8 vs e1); `Sub/Beta.md` = `[0.6,0.8,0,...]` (cos 0.6 vs e1); `Gamma.md` = `[0,0,1,0,...]` (cos 0 vs e1); frontmatter block = `[0,1,0,...]` (must never appear in results); vault-b `Alpha.md` = `[0.6,0.8,0,...]`.

- [ ] **Step 1: Write vault-a markdown files**

`test/fixtures/vault-a/Alpha.md` (exactly these 11 lines):

```markdown
---
title: Alpha
---

# Alpha
## Intro
Alpha intro text about apples.
More intro.
## Detail
Alpha detail text about orchards.
The end.
```

`test/fixtures/vault-a/Sub/Beta.md`:

```markdown
# Beta
Beta is adjacent to alpha topics.
```

`test/fixtures/vault-a/Gamma.md`:

```markdown
# Gamma
Gamma is entirely about gadgets.
```

`test/fixtures/vault-a/Plain.md`:

```markdown
# Plain
Plain note with no embeddings yet.
```

- [ ] **Step 2: Write vault-a smart-env files**

`test/fixtures/vault-a/.smart-env/smart_env.json`:

```json
{
  "is_obsidian_vault": true,
  "smart_blocks": { "embed_blocks": true, "min_chars": 10 },
  "smart_sources": {
    "min_chars": 10,
    "embed_model": {
      "adapter": "transformers",
      "transformers": { "model_key": "test-model-8d" },
      "test-model-8d": {}
    },
    "excluded_headings": "",
    "file_exclusions": "",
    "folder_exclusions": ""
  }
}
```

`test/fixtures/vault-a/.smart-env/multi/Alpha_md.ajson` (note the leading blank line, one entry per line, trailing commas — mirrors real Smart Connections output):

```
 
"smart_sources:Alpha.md": {"path":"Alpha.md","class_name":"SmartSource","embeddings":{"test-model-8d":{"vec":[1,0,0,0,0,0,0,0],"last_embed":{"hash":"h1","tokens":10}}},"last_import":{"mtime":1700000000000,"size":200},"blocks":{"#---frontmatter---":[1,3],"##Intro":[6,8],"##Detail":[9,11]}},
"smart_blocks:Alpha.md#---frontmatter---": {"path":null,"key":"Alpha.md#---frontmatter---","class_name":"SmartBlock","lines":[1,3],"embeddings":{"test-model-8d":{"vec":[0,1,0,0,0,0,0,0]}}},
"smart_blocks:Alpha.md##Intro": {"path":null,"key":"Alpha.md##Intro","class_name":"SmartBlock","lines":[6,8],"embeddings":{"test-model-8d":{"vec":[0.8,0.6,0,0,0,0,0,0]}}},
```

`test/fixtures/vault-a/.smart-env/multi/Sub_Beta_md.ajson`:

```
"smart_sources:Sub/Beta.md": {"path":"Sub/Beta.md","class_name":"SmartSource","embeddings":{"test-model-8d":{"vec":[0.6,0.8,0,0,0,0,0,0]}},"blocks":{}},
```

`test/fixtures/vault-a/.smart-env/multi/Gamma_md.ajson` (exercises override, delete, junk, no-embedding source):

```
"smart_sources:Gamma.md": {"path":"Gamma.md","class_name":"SmartSource","embeddings":{"test-model-8d":{"vec":[0,1,0,0,0,0,0,0]}},"blocks":{}},
"smart_sources:Gamma.md": {"path":"Gamma.md","class_name":"SmartSource","embeddings":{"test-model-8d":{"vec":[0,0,1,0,0,0,0,0]}},"blocks":{}},
"smart_sources:Deleted.md": {"path":"Deleted.md","class_name":"SmartSource"},
"smart_sources:Deleted.md": null,
this line is deliberately not valid json
"smart_sources:Plain.md": {"path":"Plain.md","class_name":"SmartSource","blocks":{}},
```

- [ ] **Step 3: Write vault-b files**

`test/fixtures/vault-b/Alpha.md`:

```markdown
# Alpha (vault b)
A different alpha note living in vault b.
```

`test/fixtures/vault-b/.smart-env/smart_env.json`: identical content to vault-a's `smart_env.json`.

`test/fixtures/vault-b/.smart-env/multi/Alpha_md.ajson`:

```
"smart_sources:Alpha.md": {"path":"Alpha.md","class_name":"SmartSource","embeddings":{"test-model-8d":{"vec":[0.6,0.8,0,0,0,0,0,0]}},"blocks":{}},
```

- [ ] **Step 4: Write the fixture-validating test — test/fixtures.test.ts**

```ts
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
```

- [ ] **Step 5: Run test**

Run: `npx vitest run test/fixtures.test.ts`
Expected: PASS. If the line-number assertions fail, fix the fixture markdown (not the test).

- [ ] **Step 6: Commit**

```bash
git add test/fixtures test/fixtures.test.ts
git commit -m "test: synthetic fixture vaults with 8-dim embeddings, blocks, overrides, deletions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Path sandbox

**Files:**
- Create: `src/paths.ts`
- Test: `test/paths.test.ts`

**Interfaces:**
- Consumes: `PathEscapeError` from `src/errors.ts`.
- Produces: `resolveInsideVault(vaultRoot: string, notePath: string): string` — returns absolute path or throws `PathEscapeError`.

- [ ] **Step 1: Write the failing test — test/paths.test.ts**

```ts
import { describe, it, expect } from 'vitest';
import * as path from 'node:path';
import { resolveInsideVault } from '../src/paths.js';
import { PathEscapeError } from '../src/errors.js';

const ROOT = path.resolve('/tmp/vault');

describe('resolveInsideVault', () => {
  it('resolves normal and nested note paths', () => {
    expect(resolveInsideVault(ROOT, 'Note.md')).toBe(path.join(ROOT, 'Note.md'));
    expect(resolveInsideVault(ROOT, 'Sub/Deep/Note.md')).toBe(path.join(ROOT, 'Sub', 'Deep', 'Note.md'));
  });

  it('rejects traversal and absolute escapes', () => {
    expect(() => resolveInsideVault(ROOT, '../secrets.txt')).toThrow(PathEscapeError);
    expect(() => resolveInsideVault(ROOT, 'a/../../etc/passwd')).toThrow(PathEscapeError);
    expect(() => resolveInsideVault(ROOT, '/etc/passwd')).toThrow(PathEscapeError);
  });

  it('allows internal ".." that stays inside', () => {
    expect(resolveInsideVault(ROOT, 'a/../Note.md')).toBe(path.join(ROOT, 'Note.md'));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/paths.test.ts`
Expected: FAIL — cannot find module `../src/paths.js`.

- [ ] **Step 3: Write src/paths.ts**

```ts
import * as path from 'node:path';
import { PathEscapeError } from './errors.js';

/** Resolve a vault-relative note path, refusing anything that escapes the vault root. */
export function resolveInsideVault(vaultRoot: string, notePath: string): string {
  const root = path.resolve(vaultRoot);
  const resolved = path.resolve(root, notePath);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    throw new PathEscapeError(`Path escapes vault: ${notePath}`);
  }
  return resolved;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run test/paths.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/paths.ts test/paths.test.ts
git commit -m "feat: vault path sandbox blocking traversal escapes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Vector index

**Files:**
- Create: `src/vector-index.ts`
- Test: `test/vector-index.test.ts`

**Interfaces:**
- Produces:
  - `interface IndexEntry { id: string; kind: 'note' | 'block'; notePath: string }`
  - `interface Match { entry: IndexEntry; similarity: number }`
  - `class VectorIndex { constructor(dim: number); readonly dim: number; get size(): number; set(entry: IndexEntry, vec: number[]): boolean; delete(id: string): void; deleteByNotePath(notePath: string): void; topK(query: number[], k: number, threshold: number, filter?: (e: IndexEntry) => boolean): Match[] }`
  - `cosineSimilarity(a: number[], b: number[]): number`

- [ ] **Step 1: Write the failing test — test/vector-index.test.ts**

```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/vector-index.test.ts`
Expected: FAIL — cannot find module `../src/vector-index.js`.

- [ ] **Step 3: Write src/vector-index.ts**

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run test/vector-index.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vector-index.ts test/vector-index.test.ts
git commit -m "feat: normalized in-memory vector index with top-k search

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Embedder

**Files:**
- Create: `src/embedder.ts`
- Test: `test/embedder.test.ts`

**Interfaces:**
- Consumes: `cosineSimilarity` from `src/vector-index.ts`, `EmbedUnavailableError` from `src/errors.ts`.
- Produces:
  - `type EmbedFn = (text: string) => Promise<number[]>`
  - `type PipelineFactory = (modelId: string, opts: { dtype: 'fp32' | 'q8' }) => Promise<RawExtractor>` where `RawExtractor = (text: string, opts: { pooling: 'mean'; normalize: boolean }) => Promise<{ data: ArrayLike<number> }>`
  - `class Embedder { constructor(factory?: PipelineFactory); getEmbedFn(modelKey: string, parity?: { text: string; vec: number[] }, warn?: (msg: string) => void): Promise<EmbedFn> }`
- Behavior: tries model ids `[modelKey, 'Xenova/<basename>']` × dtypes `['fp32','q8']` in order; first variant that loads and (if a parity sample is given) produces a matching-dimension embedding wins. Parity cosine < 0.99 → `warn(...)` but still used. Nothing loads → `EmbedUnavailableError`. Results cached per modelKey; a rejected build is evicted so the next call retries. Queries are embedded with no prefix (matches Smart Connections' own lookup behavior; the parity warning surfaces any drift).

- [ ] **Step 1: Write the failing test — test/embedder.test.ts**

```ts
import { describe, it, expect } from 'vitest';
import { Embedder, type PipelineFactory } from '../src/embedder.js';
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/embedder.test.ts`
Expected: FAIL — cannot find module `../src/embedder.js`.

- [ ] **Step 3: Write src/embedder.ts**

```ts
/**
 * Embeds query text locally with the same model a vault's Smart Connections data used.
 * Model weights download once (transformers.js cache) and run fully offline after that.
 */

import { EmbedUnavailableError } from './errors.js';
import { cosineSimilarity } from './vector-index.js';

export type EmbedFn = (text: string) => Promise<number[]>;

export type RawExtractor = (
  text: string,
  opts: { pooling: 'mean'; normalize: boolean },
) => Promise<{ data: ArrayLike<number> }>;

export type PipelineFactory = (
  modelId: string,
  opts: { dtype: 'fp32' | 'q8' },
) => Promise<RawExtractor>;

const defaultFactory: PipelineFactory = async (modelId, opts) => {
  const { pipeline } = await import('@huggingface/transformers');
  const p = await pipeline('feature-extraction', modelId, { dtype: opts.dtype });
  return p as unknown as RawExtractor;
};

const PARITY_WARN_THRESHOLD = 0.99;

export class Embedder {
  private factory: PipelineFactory;
  private cache = new Map<string, Promise<EmbedFn>>();

  constructor(factory: PipelineFactory = defaultFactory) {
    this.factory = factory;
  }

  getEmbedFn(
    modelKey: string,
    parity?: { text: string; vec: number[] },
    warn: (msg: string) => void = () => {},
  ): Promise<EmbedFn> {
    const cached = this.cache.get(modelKey);
    if (cached) return cached;
    const built = this.build(modelKey, parity, warn);
    this.cache.set(modelKey, built);
    built.catch(() => this.cache.delete(modelKey));
    return built;
  }

  private async build(
    modelKey: string,
    parity: { text: string; vec: number[] } | undefined,
    warn: (msg: string) => void,
  ): Promise<EmbedFn> {
    const basename = modelKey.split('/').pop() ?? modelKey;
    const modelIds = [...new Set([modelKey, `Xenova/${basename}`])];
    const dtypes: Array<'fp32' | 'q8'> = ['fp32', 'q8'];
    let lastError: unknown = new Error('no variants attempted');

    for (const modelId of modelIds) {
      for (const dtype of dtypes) {
        let extractor: RawExtractor;
        try {
          extractor = await this.factory(modelId, { dtype });
        } catch (e) {
          lastError = e;
          continue;
        }
        const embed: EmbedFn = async (text) => {
          const out = await extractor(text, { pooling: 'mean', normalize: true });
          return Array.from(out.data as Float32Array);
        };
        if (parity) {
          let computed: number[];
          try {
            computed = await embed(parity.text);
          } catch (e) {
            lastError = e;
            continue;
          }
          if (computed.length !== parity.vec.length) {
            lastError = new Error(
              `${modelId} (${dtype}) produced ${computed.length}-dim vectors, vault has ${parity.vec.length}-dim`,
            );
            continue;
          }
          const cos = cosineSimilarity(computed, parity.vec);
          if (cos < PARITY_WARN_THRESHOLD) {
            warn(
              `embedding parity for ${modelId} (${dtype}) is ${cos.toFixed(4)} (< ${PARITY_WARN_THRESHOLD}); ` +
                `rankings may differ slightly from Smart Connections`,
            );
          }
        }
        return embed;
      }
    }
    throw new EmbedUnavailableError(
      `Could not load embedding model "${modelKey}": ${lastError instanceof Error ? lastError.message : String(lastError)}`,
    );
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run test/embedder.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/embedder.ts test/embedder.test.ts
git commit -m "feat: lazy local embedder with variant fallback and parity warning

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Vault

**Files:**
- Create: `src/vault.ts`
- Test: `test/vault.test.ts`

**Interfaces:**
- Consumes: `parseAjson`/`createVaultData`/`blockNotePath`/`isFrontmatterBlock`, `VectorIndex`, `resolveInsideVault`, types, errors.
- Produces `class Vault`:
  - `static load(vaultPath: string, name: string): Vault` — throws `Error` with a clear message if `.smart-env`, `smart_env.json`, or `multi/` is missing.
  - `readonly name: string; readonly path: string; readonly modelKey: string; data: VaultData; index: VectorIndex`
  - `maybeReload(now?: number): void` — throttled (2000 ms between checks, tracked from the passed/derived `now`); re-parses only `.ajson` files whose `mtimeMs:size` changed, then rebuilds the index; a removed `.ajson` file triggers a full data reload.
  - `readNote(notePath: string): string` — sandboxed; `NoteNotFoundError` if missing.
  - `extractBlockByHeading(notePath: string, heading: string): string` — via `source.blocks[heading]`; `BlockNotFoundError` if unknown.
  - `blockSnippet(blockKey: string, maxLen?: number): string` — via `block.lines` (fallback: source.blocks by heading part); `''` if unresolvable.
  - `noteSnippet(notePath: string, maxLen?: number): string` — content with leading frontmatter stripped, trimmed, truncated; `''` if unreadable.
  - `paritySample(): { text: string; vec: number[] } | undefined` — first source with a vector whose file reads successfully.
  - `stats(): { notes: number; blocks: number; indexed: number; embeddingDim: number; modelKey: string }`

- [ ] **Step 1: Write the failing test — test/vault.test.ts**

```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/vault.test.ts`
Expected: FAIL — cannot find module `../src/vault.js`.

- [ ] **Step 3: Write src/vault.ts**

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run test/vault.test.ts`
Expected: PASS (7 tests). Note: the maybeReload test relies on `Vault.load` setting `lastCheck` to 0, so `maybeReload(1_000_000)` is past the throttle window.

- [ ] **Step 5: Commit**

```bash
git add src/vault.ts test/vault.test.ts
git commit -m "feat: Vault — load smart-env data, build index, throttled incremental reload, sandboxed reads

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Vault registry + env config

**Files:**
- Create: `src/vault-registry.ts`
- Test: `test/vault-registry.test.ts`

**Interfaces:**
- Consumes: `Vault`, errors.
- Produces:
  - `parseVaultPaths(env?: Record<string, string | undefined>): string[]` — `SMART_VAULT_PATHS` wins over `SMART_VAULT_PATH`; comma-separated; trimmed; empties dropped.
  - `interface VaultFailure { name: string; path: string; error: string }`
  - `class VaultRegistry { vaults: Vault[]; failures: VaultFailure[]; static fromPaths(paths: string[]): VaultRegistry; byName(name?: string): Vault[]; resolveNote(notePath: string, vaultName?: string): Vault }`
- Behavior: vault names are directory basenames, de-duplicated with `-2`, `-3` suffixes; `byName(undefined)` returns all vaults; `byName('x')` throws `VaultNotFoundError` listing available names; `resolveNote` throws `NoteNotFoundError` (in none) or `AmbiguousNoteError` (in several, message lists vault names).

- [ ] **Step 1: Write the failing test — test/vault-registry.test.ts**

```ts
import { describe, it, expect } from 'vitest';
import * as path from 'node:path';
import { VaultRegistry, parseVaultPaths } from '../src/vault-registry.js';
import { AmbiguousNoteError, NoteNotFoundError, VaultNotFoundError } from '../src/errors.js';

const FIXTURE_A = path.resolve(import.meta.dirname, 'fixtures/vault-a');
const FIXTURE_B = path.resolve(import.meta.dirname, 'fixtures/vault-b');

describe('parseVaultPaths', () => {
  it('parses single, multiple, and alias env vars', () => {
    expect(parseVaultPaths({ SMART_VAULT_PATH: '/a' })).toEqual(['/a']);
    expect(parseVaultPaths({ SMART_VAULT_PATH: '/a, /b ,' })).toEqual(['/a', '/b']);
    expect(parseVaultPaths({ SMART_VAULT_PATHS: '/c', SMART_VAULT_PATH: '/a' })).toEqual(['/c']);
    expect(parseVaultPaths({})).toEqual([]);
  });
});

describe('VaultRegistry', () => {
  it('loads multiple vaults with basename names', () => {
    const reg = VaultRegistry.fromPaths([FIXTURE_A, FIXTURE_B]);
    expect(reg.vaults.map((v) => v.name)).toEqual(['vault-a', 'vault-b']);
    expect(reg.failures).toEqual([]);
    expect(reg.byName().length).toBe(2);
    expect(reg.byName('vault-b')[0].name).toBe('vault-b');
    expect(() => reg.byName('nope')).toThrow(VaultNotFoundError);
  });

  it('disambiguates duplicate basenames', () => {
    const reg = VaultRegistry.fromPaths([FIXTURE_A, FIXTURE_A]);
    expect(reg.vaults.map((v) => v.name)).toEqual(['vault-a', 'vault-a-2']);
  });

  it('captures failures without dying', () => {
    const reg = VaultRegistry.fromPaths([FIXTURE_A, '/nonexistent/vault']);
    expect(reg.vaults.length).toBe(1);
    expect(reg.failures.length).toBe(1);
    expect(reg.failures[0].error).toMatch(/smart-env/i);
  });

  it('resolves notes across vaults', () => {
    const reg = VaultRegistry.fromPaths([FIXTURE_A, FIXTURE_B]);
    expect(reg.resolveNote('Gamma.md').name).toBe('vault-a'); // unique
    expect(reg.resolveNote('Alpha.md', 'vault-b').name).toBe('vault-b'); // explicit
    expect(() => reg.resolveNote('Alpha.md')).toThrow(AmbiguousNoteError); // in both
    expect(() => reg.resolveNote('Missing.md')).toThrow(NoteNotFoundError);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/vault-registry.test.ts`
Expected: FAIL — cannot find module `../src/vault-registry.js`.

- [ ] **Step 3: Write src/vault-registry.ts**

```ts
/** Owns all configured vaults; resolves vault names and note paths. */

import * as path from 'node:path';
import { AmbiguousNoteError, NoteNotFoundError, VaultNotFoundError } from './errors.js';
import { Vault } from './vault.js';

export interface VaultFailure {
  name: string;
  path: string;
  error: string;
}

export function parseVaultPaths(
  env: Record<string, string | undefined> = process.env,
): string[] {
  const raw = env.SMART_VAULT_PATHS ?? env.SMART_VAULT_PATH ?? '';
  return raw.split(',').map((s) => s.trim()).filter(Boolean);
}

export class VaultRegistry {
  vaults: Vault[] = [];
  failures: VaultFailure[] = [];

  static fromPaths(paths: string[]): VaultRegistry {
    const reg = new VaultRegistry();
    const used = new Set<string>();
    for (const p of paths) {
      const base = path.basename(p.replace(/[\\/]+$/, '')) || p;
      let name = base;
      for (let n = 2; used.has(name); n++) name = `${base}-${n}`;
      used.add(name);
      try {
        reg.vaults.push(Vault.load(p, name));
      } catch (e) {
        reg.failures.push({ name, path: p, error: e instanceof Error ? e.message : String(e) });
      }
    }
    return reg;
  }

  byName(name?: string): Vault[] {
    if (name === undefined) return this.vaults;
    const vault = this.vaults.find((v) => v.name === name);
    if (!vault) {
      throw new VaultNotFoundError(
        `Unknown vault "${name}". Available: ${this.vaults.map((v) => v.name).join(', ')}`,
      );
    }
    return [vault];
  }

  resolveNote(notePath: string, vaultName?: string): Vault {
    const candidates = this.byName(vaultName).filter((v) => v.data.sources.has(notePath));
    if (candidates.length === 0) {
      throw new NoteNotFoundError(`Note not found in any vault: ${notePath}`);
    }
    if (candidates.length > 1) {
      throw new AmbiguousNoteError(
        `Note "${notePath}" exists in vaults: ${candidates.map((v) => v.name).join(', ')} — pass the "vault" parameter`,
      );
    }
    return candidates[0];
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run test/vault-registry.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vault-registry.ts test/vault-registry.test.ts
git commit -m "feat: multi-vault registry with env parsing, name disambiguation, note resolution

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Search engine

**Files:**
- Create: `src/search-engine.ts`
- Test: `test/search-engine.test.ts`

**Interfaces:**
- Consumes: `VaultRegistry`, `Vault`, `EmbedFn`, errors, result types from `src/types.ts`.
- Produces:
  - `interface QueryEmbedder { getEmbedFn(modelKey: string, parity?: { text: string; vec: number[] }, warn?: (msg: string) => void): Promise<EmbedFn> }` (the `Embedder` class satisfies this; tests inject fakes)
  - `class SearchEngine { constructor(registry: VaultRegistry, embedder: QueryEmbedder); search(query: string, opts?: { vault?: string; scope?: 'notes' | 'blocks' | 'both'; limit?: number; threshold?: number }): Promise<SearchResponse>; getSimilarNotes(notePath: string, opts?: { vault?: string; threshold?: number; limit?: number }): SimilarNote[]; getConnectionGraph(notePath: string, opts?: { vault?: string; depth?: number; threshold?: number; maxPerLevel?: number }): ConnectionGraph; getNoteContent(notePath: string, opts?: { vault?: string; includeBlocks?: string[] }): object; getStats(vaultName?: string): object; listVaults(): VaultInfo[] }`
- Behavior notes:
  - `search` calls `maybeReload()` on each target vault, embeds the query per vault model, merges matches across vaults, sorts by similarity desc, slices to `limit`. Similarities rounded to 4 decimals. Block hits carry `block` (heading part of key) and a block snippet; note hits carry a note snippet.
  - `EmbedUnavailableError` for a vault → keyword fallback for that vault; response `mode` is `keyword-fallback` only if ALL target vaults fell back, otherwise `semantic` with a `warning` naming fallback vaults. Keyword scoring: lowercase tokens (length > 1), non-overlapping `indexOf` counts, score `min(total/10, 1)`, threshold NOT applied (different scale), snippet centered near the first match.
  - `getSimilarNotes`/`getConnectionGraph` operate within the resolved note's own vault, notes scope only, excluding the query note. Graph = v1's visited-set recursion.
  - `getNoteContent` with `includeBlocks` returns `{ path, vault, blocks, extracted, missing }` (no full content); without it returns `{ path, vault, blocks, content }`.

- [ ] **Step 1: Write the failing test — test/search-engine.test.ts**

```ts
import { describe, it, expect } from 'vitest';
import * as path from 'node:path';
import { SearchEngine, type QueryEmbedder } from '../src/search-engine.js';
import { VaultRegistry } from '../src/vault-registry.js';
import { EmbedUnavailableError, AmbiguousNoteError } from '../src/errors.js';

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/search-engine.test.ts`
Expected: FAIL — cannot find module `../src/search-engine.js`.

- [ ] **Step 3: Write src/search-engine.ts**

```ts
/** Orchestrates search, similarity, graphs, content, and stats across vaults. */

import { blockNotePath } from './ajson-loader.js';
import type { EmbedFn } from './embedder.js';
import { EmbedUnavailableError } from './errors.js';
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
      } catch {
        return;
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
        } catch {
          missing.push(heading);
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run test/search-engine.test.ts`
Expected: PASS (11 tests).

- [ ] **Step 5: Run the full suite**

Run: `npm test`
Expected: all tests pass, tsc clean.

- [ ] **Step 6: Commit**

```bash
git add src/search-engine.ts test/search-engine.test.ts
git commit -m "feat: search engine — semantic multi-vault search, explicit keyword fallback, graphs, block extraction

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: MCP server layer

**Files:**
- Create: `src/server.ts`
- Test: `test/server.test.ts`

**Interfaces:**
- Consumes: `SearchEngine`.
- Produces: `buildServer(engine: SearchEngine): McpServer` — registers the six tools; every handler returns JSON text content; thrown errors become `{ error: message }` with `isError: true`.

- [ ] **Step 1: Write the failing test — test/server.test.ts**

```ts
import { describe, it, expect, beforeAll } from 'vitest';
import * as path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { buildServer } from '../src/server.js';
import { SearchEngine, type QueryEmbedder } from '../src/search-engine.js';
import { VaultRegistry } from '../src/vault-registry.js';

const FIXTURE_A = path.resolve(import.meta.dirname, 'fixtures/vault-a');
const E1 = [1, 0, 0, 0, 0, 0, 0, 0];
const fakeEmbedder: QueryEmbedder = { getEmbedFn: async () => async () => E1 };

let client: Client;

function textOf(res: unknown): string {
  return (res as { content: Array<{ type: string; text: string }> }).content[0].text;
}

beforeAll(async () => {
  const engine = new SearchEngine(VaultRegistry.fromPaths([FIXTURE_A]), fakeEmbedder);
  const server = buildServer(engine);
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  client = new Client({ name: 'test-client', version: '1.0.0' });
  await client.connect(clientTransport);
});

describe('MCP server', () => {
  it('lists exactly the six v2 tools', async () => {
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name).sort()).toEqual([
      'get_connection_graph',
      'get_note_content',
      'get_similar_notes',
      'get_stats',
      'list_vaults',
      'search_notes',
    ]);
  });

  it('search_notes round-trips', async () => {
    const res = await client.callTool({ name: 'search_notes', arguments: { query: 'alpha' } });
    const body = JSON.parse(textOf(res));
    expect(body.mode).toBe('semantic');
    expect(body.results[0].path).toBe('Alpha.md');
  });

  it('get_note_content extracts blocks', async () => {
    const res = await client.callTool({
      name: 'get_note_content',
      arguments: { note_path: 'Alpha.md', include_blocks: ['##Intro'] },
    });
    const body = JSON.parse(textOf(res));
    expect(body.extracted['##Intro']).toContain('Alpha intro text');
  });

  it('returns isError for unknown notes without crashing', async () => {
    const res = await client.callTool({ name: 'get_similar_notes', arguments: { note_path: 'Nope.md' } });
    expect((res as { isError?: boolean }).isError).toBe(true);
    expect(JSON.parse(textOf(res)).error).toMatch(/not found/i);
  });

  it('blocks path traversal via the tool boundary', async () => {
    const res = await client.callTool({
      name: 'get_note_content',
      arguments: { note_path: '../../../etc/passwd' },
    });
    expect((res as { isError?: boolean }).isError).toBe(true);
  });

  it('list_vaults and get_stats respond', async () => {
    const vaults = JSON.parse(textOf(await client.callTool({ name: 'list_vaults', arguments: {} })));
    expect(vaults[0].name).toBe('vault-a');
    const stats = JSON.parse(textOf(await client.callTool({ name: 'get_stats', arguments: {} })));
    expect(stats.totals.notes).toBe(4);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/server.test.ts`
Expected: FAIL — cannot find module `../src/server.js`.

- [ ] **Step 3: Write src/server.ts**

```ts
/** MCP layer: registers the six v2 tools on an McpServer. */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import type { SearchEngine } from './search-engine.js';

export function buildServer(engine: SearchEngine): McpServer {
  const server = new McpServer({ name: 'smart-connections-mcp', version: '2.0.0' });

  const json = (value: unknown) => ({
    content: [{ type: 'text' as const, text: JSON.stringify(value, null, 2) }],
  });

  const guard = async (fn: () => unknown | Promise<unknown>) => {
    try {
      return json(await fn());
    } catch (e) {
      return { ...json({ error: e instanceof Error ? e.message : String(e) }), isError: true };
    }
  };

  const vaultParam = z
    .string()
    .optional()
    .describe('Vault name (see list_vaults). Omit to use all vaults / auto-resolve.');

  server.registerTool(
    'search_notes',
    {
      title: 'Semantic note search',
      description:
        'Search notes by meaning using the vault’s own Smart Connections embedding model, run locally. ' +
        'Returns ranked note- and block-level matches with content snippets. ' +
        'Falls back to literal keyword matching (mode: "keyword-fallback") only if the embedding model cannot load.',
      inputSchema: {
        query: z.string().min(1).describe('Natural-language search query'),
        vault: vaultParam,
        scope: z.enum(['notes', 'blocks', 'both']).default('both').describe('Match whole notes, blocks, or both'),
        limit: z.number().int().positive().max(100).default(10).describe('Maximum results'),
        threshold: z.number().min(0).max(1).default(0.4).describe('Minimum cosine similarity'),
      },
    },
    async ({ query, vault, scope, limit, threshold }) =>
      guard(() => engine.search(query, { vault, scope, limit, threshold })),
  );

  server.registerTool(
    'get_similar_notes',
    {
      title: 'Find similar notes',
      description:
        'Find notes semantically similar to a given note using its stored embedding (no model needed). ' +
        'Searches within the note’s own vault.',
      inputSchema: {
        note_path: z.string().describe('Vault-relative note path, e.g. "Folder/Note.md"'),
        vault: vaultParam,
        threshold: z.number().min(0).max(1).default(0.5).describe('Minimum cosine similarity'),
        limit: z.number().int().positive().max(100).default(10).describe('Maximum results'),
      },
    },
    async ({ note_path, vault, threshold, limit }) =>
      guard(() => engine.getSimilarNotes(note_path, { vault, threshold, limit })),
  );

  server.registerTool(
    'get_connection_graph',
    {
      title: 'Build connection graph',
      description: 'Walk semantic similarity links outward from a note to map how ideas connect.',
      inputSchema: {
        note_path: z.string().describe('Vault-relative note path to start from'),
        vault: vaultParam,
        depth: z.number().int().positive().max(5).default(2).describe('Levels to traverse'),
        threshold: z.number().min(0).max(1).default(0.6).describe('Minimum similarity per hop'),
        max_per_level: z.number().int().positive().max(20).default(5).describe('Connections per node'),
      },
    },
    async ({ note_path, vault, depth, threshold, max_per_level }) =>
      guard(() => engine.getConnectionGraph(note_path, { vault, depth, threshold, maxPerLevel: max_per_level })),
  );

  server.registerTool(
    'get_note_content',
    {
      title: 'Read note content',
      description:
        'Read a note’s full markdown, or pass include_blocks (heading keys from search/similar results) to extract only those sections.',
      inputSchema: {
        note_path: z.string().describe('Vault-relative note path'),
        vault: vaultParam,
        include_blocks: z.array(z.string()).optional().describe('Block heading keys to extract, e.g. ["##Intro"]'),
      },
    },
    async ({ note_path, vault, include_blocks }) =>
      guard(() => engine.getNoteContent(note_path, { vault, includeBlocks: include_blocks })),
  );

  server.registerTool(
    'get_stats',
    {
      title: 'Knowledge base statistics',
      description: 'Note/block/index counts and embedding model per vault, with totals.',
      inputSchema: { vault: vaultParam },
    },
    async ({ vault }) => guard(() => engine.getStats(vault)),
  );

  server.registerTool(
    'list_vaults',
    {
      title: 'List configured vaults',
      description: 'All configured vaults with load status, counts, and embedding model. Failed vaults include the error.',
      inputSchema: {},
    },
    async () => guard(() => engine.listVaults()),
  );

  return server;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run test/server.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/server.ts test/server.test.ts
git commit -m "feat: MCP server layer — six v2 tools on McpServer with zod schemas

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Bin entry, stdio E2E, live tests, smoke script

**Files:**
- Create: `src/index.ts`, `test/e2e-stdio.test.ts`, `test/live/embedder.live.test.ts`, `scripts/smoke.mjs`

**Interfaces:**
- Consumes: everything.
- Produces: the `dist/index.js` bin. Exit code 1 with a stderr message when no paths configured or no vault loads.

- [ ] **Step 1: Write src/index.ts**

```ts
#!/usr/bin/env node

/** smart-connections-mcp v2 — stdio entry point. */

import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { Embedder } from './embedder.js';
import { SearchEngine } from './search-engine.js';
import { buildServer } from './server.js';
import { parseVaultPaths, VaultRegistry } from './vault-registry.js';

const paths = parseVaultPaths();
if (paths.length === 0) {
  console.error('Error: SMART_VAULT_PATH is required.');
  console.error('Set it to one or more comma-separated Obsidian vault paths, e.g.:');
  console.error('  SMART_VAULT_PATH="/Users/me/Vault A,/Users/me/Vault B"');
  process.exit(1);
}

const registry = VaultRegistry.fromPaths(paths);
for (const f of registry.failures) {
  console.error(`Vault failed to load: ${f.path} — ${f.error}`);
}
if (registry.vaults.length === 0) {
  console.error('Error: no vaults loaded successfully.');
  process.exit(1);
}

const engine = new SearchEngine(registry, new Embedder());
const server = buildServer(engine);
await server.connect(new StdioServerTransport());
console.error(
  `smart-connections-mcp v2 ready — ${registry.vaults
    .map((v) => `${v.name} (${v.stats().notes} notes)`)
    .join(', ')}`,
);
```

- [ ] **Step 2: Write the E2E test — test/e2e-stdio.test.ts**

Runs the actual built bin over stdio, exactly as Claude Desktop would. `test-model-8d` is not a real HF repo, so `search_notes` exercises the keyword fallback end-to-end without network.

```ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const FIXTURE_A = path.resolve(import.meta.dirname, 'fixtures/vault-a');
let client: Client;

function textOf(res: unknown): string {
  return (res as { content: Array<{ type: string; text: string }> }).content[0].text;
}

beforeAll(async () => {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [path.resolve(import.meta.dirname, '../dist/index.js')],
    env: { ...process.env, SMART_VAULT_PATH: FIXTURE_A, HF_HUB_OFFLINE: '1' },
    stderr: 'ignore',
  });
  client = new Client({ name: 'e2e', version: '1.0.0' });
  await client.connect(transport);
}, 60_000);

afterAll(async () => {
  await client?.close();
});

describe('stdio E2E (built dist)', () => {
  it('serves tools over stdio', async () => {
    const { tools } = await client.listTools();
    expect(tools.length).toBe(6);
  });

  it('search falls back to keyword mode for the fake model and flags it', async () => {
    const res = await client.callTool({ name: 'search_notes', arguments: { query: 'apples' } });
    const body = JSON.parse(textOf(res));
    expect(body.mode).toBe('keyword-fallback');
    expect(body.results[0].path).toBe('Alpha.md');
  }, 60_000);
});
```

- [ ] **Step 3: Run tests**

Run: `npm test` (builds dist first, then runs all CI-tier tests)
Expected: all pass, including the two E2E tests.

- [ ] **Step 4: Write test/live/embedder.live.test.ts (opt-in, real model)**

```ts
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
```

- [ ] **Step 5: Run live tests once (requires network on first run)**

Run: `npm run test:live -- test/live`
Expected: PASS (2 tests). First run downloads ~25MB of model weights. If `TaylorAI/bge-micro-v2` fails to load and the Xenova fallback also fails, record the failure output — the Embedder variant order may need a third candidate; fix in `src/embedder.ts` `modelIds` list.

- [ ] **Step 6: Write scripts/smoke.mjs (manual pre-release check against a real vault)**

```js
#!/usr/bin/env node
// Usage: node scripts/smoke.mjs "/path/to/real/vault" "a search query"
import { Embedder } from '../dist/embedder.js';
import { SearchEngine } from '../dist/search-engine.js';
import { VaultRegistry } from '../dist/vault-registry.js';

const [vaultPath, query = 'what is this vault about'] = process.argv.slice(2);
if (!vaultPath) {
  console.error('Usage: node scripts/smoke.mjs "/path/to/vault" "query"');
  process.exit(1);
}

const registry = VaultRegistry.fromPaths([vaultPath]);
if (registry.failures.length) console.error('Failures:', registry.failures);
const engine = new SearchEngine(registry, new Embedder());

console.log('--- list_vaults');
console.log(JSON.stringify(engine.listVaults(), null, 2));

console.log(`--- search: "${query}"`);
const res = await engine.search(query, { limit: 5 });
console.log(`mode: ${res.mode}${res.warning ? ` | warning: ${res.warning}` : ''}`);
for (const r of res.results) {
  console.log(`${r.similarity.toFixed(3)}  [${r.scope}] ${r.path}${r.block ?? ''}`);
}

const first = registry.vaults[0]?.data.sources.keys().next().value;
if (first) {
  console.log(`--- similar to: ${first}`);
  console.log(JSON.stringify(engine.getSimilarNotes(first, { limit: 5 }), null, 2));
}
```

- [ ] **Step 7: Run the smoke script against the real vault**

Run: `npm run build && node scripts/smoke.mjs "/Users/danielglickman/Library/Mobile Documents/com~apple~CloudDocs/Liberalism" "individual freedom and the state"`
Expected: `mode: semantic`, non-empty ranked results including Mill/Tocqueville/Berlin-Rawls-Nozick clippings, block-level hits present. THIS is the moment v1's headline bug is visibly fixed — record the output in the task report.

- [ ] **Step 8: Commit**

```bash
git add src/index.ts test/e2e-stdio.test.ts test/live scripts/smoke.mjs
git commit -m "feat: v2 bin entry, stdio E2E test, live model tests, real-vault smoke script

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Docs and release prep

**Files:**
- Modify: `README.md`, `QUICKSTART.md`, `TROUBLESHOOTING.md`, `server.json`
- Create: `CHANGELOG.md`

**Interfaces:**
- Consumes: final tool behavior from Tasks 9–11.
- Produces: release-ready docs. No code changes in this task.

- [ ] **Step 1: Write CHANGELOG.md**

```markdown
# Changelog

## 2.0.0 — 2026-07-13

### Added
- **True semantic search**: `search_notes` now embeds your query locally with the
  same model your vault's Smart Connections index used (via transformers.js).
  Conceptual queries work; nothing leaves your machine.
- **Multi-vault**: `SMART_VAULT_PATH` accepts comma-separated paths; tools take an
  optional `vault` parameter; new `list_vaults` tool.
- **Block-level retrieval**: search matches individual sections (blocks) and
  returns content snippets inline.
- **Freshness**: `.smart-env` changes are picked up automatically (throttled
  incremental reload) — no server restart after editing notes.
- `get_note_content` `include_blocks` now actually extracts the named blocks.
- Explicit `mode: "keyword-fallback"` + warning when the embedding model cannot
  load, instead of silently degraded results.

### Fixed
- Path traversal in `get_note_content` (reads outside the vault are rejected).
- Crash on regex metacharacters in search queries.
- `.ajson` deletion entries (`null`) are now honored.

### Removed (breaking)
- `get_embedding_neighbors` tool.
- `search_notes` response shape changed (adds `vault`, `scope`, `block`,
  `snippet`, `mode`).

### Changed
- Requires Node >= 20. MCP SDK updated to the current 1.x line.
- First run downloads the embedding model (~25MB, cached locally forever).
```

- [ ] **Step 2: Rewrite README.md**

Full replacement. Keep the badges block and MIT license reference from v1; the body becomes:

```markdown
# Smart Connections MCP Server

**Give Claude true semantic memory of your Obsidian vault.** An MCP server that
searches your notes by *meaning* — reusing the embeddings the
[Smart Connections](https://github.com/brianpetro/obsidian-smart-connections)
Obsidian plugin already generated, and running the same embedding model locally
to understand your queries. No cloud calls; your vault never leaves your machine.

[badges — keep from v1]

## What it does

- **`search_notes`** — semantic search across one or many vaults. Matches whole
  notes *and* individual sections (blocks), returns similarity-ranked results
  with content snippets.
- **`get_similar_notes`** — notes similar to a given note (stored embeddings).
- **`get_connection_graph`** — walk similarity links outward to map related ideas.
- **`get_note_content`** — read a note, or extract specific blocks.
- **`list_vaults` / `get_stats`** — what's loaded, counts, models, load errors.

## Requirements

- Node.js 20+
- An Obsidian vault with the Smart Connections plugin installed and embeddings
  generated (v2 tested against Smart Connections 3.x data)
- An MCP client (Claude Desktop, Claude Code, …)

## Setup (Claude Desktop)

Add to `claude_desktop_config.json` and restart Claude Desktop:

    {
      "mcpServers": {
        "smart-connections": {
          "command": "npx",
          "args": ["-y", "smart-connections-mcp"],
          "env": {
            "SMART_VAULT_PATH": "/path/to/Vault One,/path/to/Vault Two"
          }
        }
      }
    }

One vault or several — separate paths with commas.

### Claude Code

    claude mcp add smart-connections -e SMART_VAULT_PATH="/path/to/vault" -- npx -y smart-connections-mcp

## How it works

Smart Connections stores an embedding vector for every note and block in
`.smart-env/`. This server loads those vectors into memory and, when you search,
embeds your query with the *same model* your vault used (downloaded once,
~25MB, runs locally via transformers.js). Results are ranked by cosine
similarity. Edits you make in Obsidian are picked up automatically.

If the embedding model can't load (e.g. no network on very first run), search
degrades to literal keyword matching and says so explicitly
(`"mode": "keyword-fallback"`).

## Migrating from v1

- `get_embedding_neighbors` was removed.
- `search_notes` is now genuinely semantic and its response includes `vault`,
  `scope`, `block`, `snippet`, and `mode` fields.
- Everything else is backward compatible; single-vault `SMART_VAULT_PATH`
  configs work unchanged.

## Development

    npm install
    npm test              # build + CI-tier tests (no network)
    npm run test:live     # + real-model tests (downloads ~25MB once)
    npm run smoke -- "/path/to/vault" "your query"

MIT — see [LICENSE](LICENSE).
```

- [ ] **Step 3: Update QUICKSTART.md and TROUBLESHOOTING.md**

QUICKSTART: replace any v1 install/config snippets with the README's config block (multi-vault example included); remove references to `get_embedding_neighbors` and to cloning/building (npx path is primary).

TROUBLESHOOTING: keep the generic Claude Desktop restart guidance; add two new entries: (1) `"mode": "keyword-fallback"` in results → the model download failed; check network once, it caches afterward; (2) a vault shows `status: "error"` in `list_vaults` → the `.smart-env` folder is missing or Smart Connections hasn't finished indexing.

- [ ] **Step 4: Update server.json**

Set `"version": "2.0.0"` (and the packages entry version if present). Description stays under 100 chars, e.g.: `Local semantic search over Obsidian vaults via Smart Connections embeddings. Multi-vault, private.`

- [ ] **Step 5: Full verification**

```bash
npm test
node scripts/smoke.mjs "/Users/danielglickman/Library/Mobile Documents/com~apple~CloudDocs/Liberalism" "what did Mill think about liberty"
```

Expected: suite green; smoke shows `mode: semantic` with sensible ranked results.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: v2 README/CHANGELOG/quickstart/troubleshooting; bump server.json to 2.0.0

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Post-plan (not tasks): merge `v2` to `main`, tag `v2.0.0` to trigger the existing npm + MCP-registry publish workflow — only after Daniel reviews and approves the release.
