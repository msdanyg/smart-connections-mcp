# Smart Connections MCP Server

**Give Claude true semantic memory of your Obsidian vault.** An MCP server that
searches your notes by *meaning* — reusing the embeddings the
[Smart Connections](https://github.com/brianpetro/obsidian-smart-connections)
Obsidian plugin already generated, and running the same embedding model locally
to understand your queries. No cloud calls; your vault never leaves your machine.

[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-1f6feb)](https://modelcontextprotocol.io/)
[![Obsidian](https://img.shields.io/badge/Obsidian-Smart_Connections-7c3aed)](https://github.com/brianpetro/obsidian-smart-connections)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/msdanyg/smart-connections-mcp?style=social)](https://github.com/msdanyg/smart-connections-mcp/stargazers)

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

```json
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
```

One vault or several — separate paths with commas.

### Claude Code

```bash
claude mcp add smart-connections -e SMART_VAULT_PATH="/path/to/vault" -- npx -y smart-connections-mcp
```

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

```bash
npm install
npm test              # build + CI-tier tests (no network)
npm run test:live     # + real-model tests (downloads ~25MB once)
npm run smoke -- "/path/to/vault" "your query"
```

MIT — see [LICENSE](LICENSE).
