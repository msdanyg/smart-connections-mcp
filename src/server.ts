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
        'Search notes by meaning using the vault\'s own Smart Connections embedding model, run locally. ' +
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
        'Searches within the note\'s own vault.',
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
        'Read a note\'s full markdown, or pass include_blocks (heading keys from search/similar results) to extract only those sections.',
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
