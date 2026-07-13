# Smart Connections MCP Server — V2 Design

**Date:** 2026-07-13
**Status:** Approved by Daniel (design review in session)
**Target version:** `smart-connections-mcp@2.0.0`

## Background and motivation

V1 (1.0.0) reads the `.smart-env` data that the Obsidian Smart Connections plugin
generates and exposes it over MCP. A live review against a real vault
(Smart Connections v3.0.80, "Liberalism" vault) found:

**Working:** `.ajson` parsing, note-level embeddings (384-dim, TaylorAI/bge-micro-v2),
`get_similar_notes`, `get_note_content`, `get_stats`, `get_connection_graph`.

**Broken or false:**

1. `search_notes` is a literal substring counter, not semantic search. Multi-word
   conceptual queries return zero results. The README's headline claim is false.
2. Block-level embeddings (`smart_blocks:` entries, present when
   `embed_blocks: true`) are ignored entirely.
3. `search_notes` crashes on regex metacharacters in the query (raw string is
   passed to `new RegExp`).
4. Path traversal: `get_note_content` with `../../…` reads files outside the vault.
5. Data loads once at startup; vault edits are invisible until restart.
6. `get_embedding_neighbors` requires the caller to supply a raw 384-dim vector —
   unusable by LLM clients.
7. `get_note_content`'s `include_blocks` parameter is accepted and silently ignored.
8. Bundled test scripts hardcode `CLAUDE.md` as the test note and fail on any vault
   without one; there is no real test suite.
9. MCP SDK pinned at ^1.0.4 using the deprecated low-level `Server` API; every tool
   schema is defined twice (zod + hand-written JSON Schema).

## Goals

1. **True semantic search**: embed query text locally with the same model the vault
   used, so `search_notes` does what the README promises. No cloud calls.
2. **Multi-vault**: one server instance serves several vaults (the user has two
   with Smart Connections data).
3. **Block-level retrieval**: use `smart_blocks` embeddings for precise results.
4. **Freshness**: pick up `.smart-env` changes without a restart.
5. **Correctness and safety**: fix path traversal, regex injection, silent
   parameter no-ops.
6. **Real tests**: CI-runnable suite with a fixture vault; no model download in CI.
7. **Modern SDK**: current `@modelcontextprotocol/sdk`, `McpServer` high-level API,
   single source of truth for schemas (zod).

## Non-goals

- Persistent/external vector store (SQLite, etc.). Brute-force dot product over
  pre-normalized Float32Arrays handles 10k+ notes in milliseconds; a native
  dependency would complicate `npx` installs for nothing. (Considered, rejected.)
- Generating embeddings for notes Smart Connections has not embedded. The plugin
  remains the indexer; this server remains a reader plus query-embedder.
- MCP resources/prompts surface. Tools only, as in v1.

## Tool surface (V2)

| Tool | Status | Behavior |
|---|---|---|
| `search_notes` | changed | Semantic search. Params: `query` (string), `vault?` (name), `scope?` (`notes` \| `blocks` \| `both`, default `both`), `limit?` (default 10), `threshold?` (default 0.4). Embeds the query with the vault's model; returns `{path, vault, similarity, scope, block?, snippet}` ranked across the selected vaults. Snippet = the matched block's text (or note head for note-level hits), truncated (~700 chars). |
| `get_similar_notes` | kept | v1 behavior + optional `vault`. Uses stored note vectors only (no model needed). |
| `get_connection_graph` | kept | v1 behavior + optional `vault`. |
| `get_note_content` | fixed | `include_blocks` now actually extracts the named blocks. Paths resolved against the vault root and rejected if they escape it. Optional `vault`. |
| `list_vaults` | new | Name, path, note/block counts, embedding model key, load status, and load error if any. |
| `get_stats` | changed | Per-vault breakdown + totals. Optional `vault`. |
| `get_embedding_neighbors` | **removed** | Breaking change; no LLM client can supply a 384-number vector by hand. |

Vault addressing: tools accept an optional `vault` parameter (the vault's directory
basename, disambiguated with a numeric suffix on collision). Omitted = operate on
all vaults (search merges and re-ranks across vaults; note-addressed tools resolve
the note in whichever vault contains it, erroring if ambiguous).

## Configuration

- `SMART_VAULT_PATH` — one or more absolute vault paths, comma-separated.
  Backward compatible: a single path behaves like v1.
- `SMART_VAULT_PATHS` — accepted as an alias.
- A vault that fails to load (missing `.smart-env`, parse failure) is reported via
  `list_vaults` and stderr; the server still starts and serves healthy vaults.
  The server exits with an error only if zero vaults load.

## Architecture

Modules under `src/`:

- **`vault-registry.ts`** — parses config, owns one `Vault` per path, exposes
  lookup by name and resolution of note paths across vaults. Vault load failures
  are captured per-vault, not thrown.
- **`ajson-loader.ts`** — parses `.smart-env/multi/*.ajson`. Line-oriented format:
  each line is `"collection:key": {…},`. Handles `smart_sources:` **and**
  `smart_blocks:` entries, later-line overrides (append semantics), and `null`
  values as deletions (removes the key). Skips unparseable lines with a stderr
  warning, never aborts the file.
- **`vector-index.ts`** — per vault: pre-normalized `Float32Array` matrix over
  note and block vectors plus id/metadata arrays. Top-k = dot product + partial
  sort. Supports incremental rebuild of a single source file's entries.
- **`embedder.ts`** — lazy per-model-key cache of transformers.js
  (`@huggingface/transformers`) feature-extraction pipelines. Model key read from
  each vault's `smart_env.json` (`smart_sources.embed_model`). **Parity check**:
  on first use, embed one stored source's text and require cosine ≈ 1.0 (> 0.99)
  against its stored vector; if parity fails, try known variants (query prefix
  on/off, `Xenova/` mirror repo) and log which configuration matched. Model files
  cache to the standard transformers.js cache dir; everything runs offline after
  first download.
- **`search-engine.ts`** — orchestrates: query → embedder → vector-index top-k →
  snippet extraction (block line ranges from loader) → results. Also similar-notes
  and graph logic (unchanged math from v1).
- **`freshness.ts`** — throttled staleness check (≥2s between checks): stat
  `.ajson` files; reload changed/new/removed files only and patch the index.
  Runs at the start of a tool call, so no background timers or watchers.
- **`server.ts`** — `McpServer` from the current SDK; tools registered with
  `registerTool` and zod schemas (single definition). stdio transport.
- **`index.ts`** — bin entry: env parsing, registry init, server start.

## Data flow (search)

`search_notes("individual freedom and the state")` →
freshness check → embedder embeds query (per-vault model, cached pipeline) →
dot product against that vault's block+note matrix → merge across vaults →
top-k over threshold → read snippet lines from the `.md` files → JSON result.

## Error handling

- Unknown tool/note/vault → structured MCP error result (`isError: true`), server
  keeps running.
- Embedding model unavailable (e.g., offline before first model download) →
  `search_notes` degrades to sanitized keyword scoring and sets
  `"mode": "keyword-fallback"` plus a human-readable `warning` in the response.
  Semantic mode reports `"mode": "semantic"`. Never a silent downgrade.
- All note reads resolve through a sandbox helper: `path.resolve` against the
  vault root, reject if the result is outside it (fixes traversal).
- Query text is never interpolated into a `RegExp` (fixes crash); keyword fallback
  uses literal-substring scoring.
- `.ajson` parse errors: skip line, log to stderr with file and line number.

## Testing

Vitest. Two tiers:

1. **CI tier (default, no network):**
   - Fixture vault in `test/fixtures/vault/` with real markdown files and a
     synthetic `.smart-env` (tiny 8-dim vectors, fake model key) exercising:
     multi-entry lines, override lines, `null` deletion, blocks, frontmatter
     block, missing-embedding sources.
   - Unit tests: ajson parsing, vector math/top-k, path sandbox (traversal
     attempts), block extraction, vault-name disambiguation, keyword fallback.
   - Integration: server over the SDK's `InMemoryTransport`; `listTools` snapshot
     and a `callTool` round-trip for every tool, including error cases. Embedder
     is injected/mocked at the engine boundary.
2. **Live tier (opt-in, `LIVE_TESTS=1`):** downloads the real model, runs the
   parity check and a semantic query against the fixture (embeddings regenerated
   at fixture-build time) — plus a manual smoke run against the real Liberalism
   vault before release.

## Release

- Version `2.0.0`; breaking changes: `get_embedding_neighbors` removed,
  `search_notes` response shape changed (snippets, `mode`, `vault` fields).
- README rewritten to match reality (semantic search now true), with a short
  migration section; CHANGELOG added; `server.json` bumped.
- Existing GitHub Actions publish flow (npm + MCP registry via OIDC on tag) kept.
- Known trade-off: `@huggingface/transformers` adds onnxruntime (~50–75MB extra
  on first `npx` install, then cached). First semantic query also downloads the
  embedding model (~25MB, cached). Documented in README.
