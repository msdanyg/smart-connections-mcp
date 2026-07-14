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
