# Troubleshooting Smart Connections MCP Server

## `"mode": "keyword-fallback"` in search results

`search_notes` returned results, but the response says `"mode": "keyword-fallback"`
instead of `"mode": "semantic"`. This means the local embedding model couldn't be
loaded, so the server fell back to literal keyword matching and told you so
explicitly (rather than silently degrading).

**Fix**: the model (~25MB) downloads from Hugging Face on first use and is then
cached locally forever. Check your network connection once — if the machine had
no internet on the very first run, the download never completed. Restart the
server (or Claude Desktop) with network access and it will retry the download;
subsequent runs work fully offline.

## A vault shows `"status": "error"` in `list_vaults`

Call `list_vaults` (or check `get_stats`) — if an entry reports `status: "error"`,
that vault's Smart Connections data couldn't be loaded. Usually one of:

- The `.smart-env` folder doesn't exist yet at that vault path — check
  `SMART_VAULT_PATH` for typos and confirm the folder is really there.
- Smart Connections hasn't finished indexing that vault in Obsidian yet — open
  the vault in Obsidian, let Smart Connections finish generating embeddings,
  then restart the MCP server (or wait for the next automatic reload).

## Long queries get truncated

Queries longer than ~1500 characters are truncated before being embedded (small
embedding models cap out near 512 tokens); the server logs a notice to stderr
when this happens. If you're pasting a long passage as a "query," keep it short
and specific instead — semantic search works better on a focused question than
a wall of text anyway.

## Server not appearing in Claude Desktop

1. Verify the configuration file syntax (JSON must be valid — no trailing commas, proper quotes)
2. Check that `SMART_VAULT_PATH` is set to absolute path(s), not relative
3. Restart Claude Desktop completely (see below)
4. Check Claude Desktop logs for error messages

### "Smart Connections directory not found"

- Ensure your vault has the Smart Connections plugin installed
- Verify embeddings have been generated (check the `.smart-env/multi/` directory)
- Check that `SMART_VAULT_PATH` points to the correct vault

## Doing a complete restart

If the server seems to be running stale code or a stale config, do a **complete**
restart, not just a refresh:

1. **Quit Claude Desktop completely**:
   - Press **Cmd+Q** (not just closing the window!)
   - Or: Right-click Claude icon in Dock → Quit

2. **Verify it's completely stopped**:
   ```bash
   ps aux | grep -i claude
   ```
   Should show nothing (or only the grep command itself)

3. **Kill any orphaned MCP server processes**:
   ```bash
   pkill -f "smart-connections-mcp"
   ```

4. **Wait a few seconds**, then reopen Claude Desktop and wait for it to fully
   initialize (may take 10-15 seconds) before testing again.

## Still not working?

1. Test the server directly, the same way Claude Desktop launches it:
   ```bash
   SMART_VAULT_PATH="/path/to/vault" npx -y smart-connections-mcp
   ```
   Look for log lines confirming the vault(s) loaded, then press Ctrl+C to stop.
2. Temporarily remove the `"smart-connections"` entry from
   `claude_desktop_config.json`, restart, and confirm the tools disappear —
   then add it back and restart again.
3. Confirm you're on Node.js 20 or newer: `node --version`.
