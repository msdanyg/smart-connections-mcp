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
