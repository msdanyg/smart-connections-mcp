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
