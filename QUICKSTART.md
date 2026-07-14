# Quick Start Guide

## Step 1: Configure Claude Desktop

1. Open your Claude Desktop configuration file:
   ```bash
   open ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. Add the Smart Connections server configuration. If the file is empty or only has `{}`, replace it with:

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

   `npx` fetches and runs the published package — no clone, no build. Point
   `SMART_VAULT_PATH` at the vault folder that contains your `.smart-env`
   directory. One vault or several — separate paths with commas.

   If you already have other MCP servers configured, just add the `"smart-connections"` entry to your existing `mcpServers` object.

## Step 2: Restart Claude Desktop

1. Quit Claude Desktop completely (Cmd+Q)
2. Reopen Claude Desktop
3. Look for the 🔌 icon in the interface (indicates MCP servers are connected)

## Step 3: Test the Connection

Try these prompts in Claude:

1. **Get stats about your knowledge base:**
   ```
   Use the Smart Connections server to show me statistics about my Obsidian vault
   ```

2. **Search for content (true semantic search):**
   ```
   Search my notes for information about [your topic]
   ```

3. **Find similar notes:**
   ```
   Find notes similar to "YourNoteName.md"
   ```

4. **Build a connection graph:**
   ```
   Show me a connection graph starting from [your note name]
   ```

5. **List configured vaults (if you set up more than one):**
   ```
   List my configured vaults
   ```

## Troubleshooting

### Server not showing up?

1. Check Claude Desktop logs (if available in the app)
2. Verify the configuration file is valid JSON (no trailing commas, proper quotes)
3. Make sure `SMART_VAULT_PATH` points to a vault that contains a `.smart-env` directory
4. Try running the server manually:
   ```bash
   SMART_VAULT_PATH="/path/to/vault" npx -y smart-connections-mcp
   ```
   You should see log lines confirming the vault(s) loaded successfully.
   Press Ctrl+C to stop.

### "Smart Connections directory not found"

Make sure your Obsidian vault has:
- Smart Connections plugin installed
- Embeddings generated (check the `.smart-env/multi/` directory exists and has files)

### Need help?

Check the full `README.md` and `TROUBLESHOOTING.md` for detailed documentation and troubleshooting steps.

## What's Next?

Once connected, Claude can:
- 🔍 Search your Obsidian notes semantically (whole notes and individual blocks)
- 🕸️ Discover hidden connections between notes
- 📊 Analyze your knowledge graph
- 📝 Read and extract specific sections from notes
- 💡 Answer questions using your entire knowledge base — across multiple vaults if configured

All of this happens **locally** — embeddings and the query model both run on your machine, nothing is sent to the cloud.
