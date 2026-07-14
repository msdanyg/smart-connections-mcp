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
