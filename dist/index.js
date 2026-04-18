#!/usr/bin/env node
/**
 * Smart Connections MCP Server
 *
 * Provides semantic search and knowledge graph capabilities for Obsidian Smart Connections
 * via the Model Context Protocol (MCP).
 */
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema, } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { SmartConnectionsLoader } from './smart-connections-loader.js';
import { SearchEngine } from './search-engine.js';
import { GteEmbedder } from './gte-embedder.js';
// Environment variable for vault path
const VAULT_PATH = process.env.SMART_VAULT_PATH;
if (!VAULT_PATH) {
    console.error('Error: SMART_VAULT_PATH environment variable is required');
    console.error('Please set it to your Obsidian vault path, e.g.:');
    console.error('  export SMART_VAULT_PATH="/Users/username/My Vault"');
    process.exit(1);
}
// Initialize loader
const loader = new SmartConnectionsLoader(VAULT_PATH);
await loader.initialize();
// Initialize EmbeddingGemma embedder (index only — model is lazy-loaded on first embed()
// to keep MCP init under the harness timeout; first tool call pays the ~150MB cold-start cost).
const gteEmbedder = new GteEmbedder(VAULT_PATH);
await gteEmbedder.initialize();
// Create search engine with EmbeddingGemma embedder
const searchEngine = new SearchEngine(loader, gteEmbedder);
console.error('Smart Connections MCP Server initialized successfully (model: lazy)');
console.error(`Vault: ${VAULT_PATH}`);
console.error(`Loaded ${loader.getSources().size} notes`);
const gteStats = gteEmbedder.getStats();
if (gteStats) {
    console.error(`GTE index: ${gteStats.entries} blocks from ${gteStats.notes} notes (${gteStats.model}, ${gteStats.dimension}d)`);
}
// Create MCP server
const server = new Server({
    name: 'smart-connections-mcp',
    version: '1.0.0',
}, {
    capabilities: {
        tools: {},
    },
});
// Define tool schemas
const GetSimilarNotesSchema = z.object({
    note_path: z.string().describe('Path to the note (e.g., "Note.md" or "Folder/Note.md")'),
    threshold: z.number().min(0).max(1).default(0.5).describe('Similarity threshold (0-1)'),
    limit: z.number().int().positive().default(10).describe('Maximum number of results'),
});
const GetConnectionGraphSchema = z.object({
    note_path: z.string().describe('Path to the note to start from'),
    depth: z.number().int().positive().default(2).describe('Depth of the connection graph'),
    threshold: z.number().min(0).max(1).default(0.6).describe('Similarity threshold (0-1)'),
    max_per_level: z.number().int().positive().default(5).describe('Max connections per level'),
});
const SearchNotesSchema = z.object({
    query: z.string().describe('Search query text'),
    limit: z.number().int().positive().default(10).describe('Maximum number of results'),
    threshold: z.number().min(0).max(1).default(0.5).describe('Similarity threshold (0-1)'),
});
const GetEmbeddingNeighborsSchema = z.object({
    embedding_vector: z.array(z.number()).describe('384-dimensional embedding vector'),
    k: z.number().int().positive().default(10).describe('Number of neighbors to return'),
    threshold: z.number().min(0).max(1).default(0.5).describe('Similarity threshold (0-1)'),
});
const GetNoteContentSchema = z.object({
    note_path: z.string().describe('Path to the note'),
    include_blocks: z.array(z.string()).optional().describe('Specific block headings to include'),
});
const GetStatsSchema = z.object({});
const RebuildGteIndexSchema = z.object({
    force: z.boolean().default(false).describe('Force re-embed all notes (ignore cache)'),
});
// Define available tools
const tools = [
    {
        name: 'get_similar_notes',
        description: 'Find notes semantically similar to a given note using embeddings. Returns paths, similarity scores, and available blocks.',
        inputSchema: {
            type: 'object',
            properties: {
                note_path: {
                    type: 'string',
                    description: 'Path to the note (e.g., "Note.md" or "Folder/Note.md")',
                },
                threshold: {
                    type: 'number',
                    description: 'Similarity threshold (0-1), default 0.5',
                    minimum: 0,
                    maximum: 1,
                    default: 0.5,
                },
                limit: {
                    type: 'number',
                    description: 'Maximum number of results, default 10',
                    minimum: 1,
                    default: 10,
                },
            },
            required: ['note_path'],
        },
    },
    {
        name: 'get_connection_graph',
        description: 'Build a multi-level connection graph starting from a note, showing how notes are semantically connected.',
        inputSchema: {
            type: 'object',
            properties: {
                note_path: {
                    type: 'string',
                    description: 'Path to the note to start from',
                },
                depth: {
                    type: 'number',
                    description: 'Depth of the connection graph (levels), default 2',
                    minimum: 1,
                    default: 2,
                },
                threshold: {
                    type: 'number',
                    description: 'Similarity threshold (0-1), default 0.6',
                    minimum: 0,
                    maximum: 1,
                    default: 0.6,
                },
                max_per_level: {
                    type: 'number',
                    description: 'Max connections per level, default 5',
                    minimum: 1,
                    default: 5,
                },
            },
            required: ['note_path'],
        },
    },
    {
        name: 'search_notes',
        description: 'Semantic search for notes using EmbeddingGemma-300m embeddings. Returns notes ranked by cosine similarity. Falls back to keyword search if the semantic index is unavailable.',
        inputSchema: {
            type: 'object',
            properties: {
                query: {
                    type: 'string',
                    description: 'Search query text',
                },
                limit: {
                    type: 'number',
                    description: 'Maximum number of results, default 10',
                    minimum: 1,
                    default: 10,
                },
                threshold: {
                    type: 'number',
                    description: 'Similarity threshold (0-1), default 0.5',
                    minimum: 0,
                    maximum: 1,
                    default: 0.5,
                },
            },
            required: ['query'],
        },
    },
    {
        name: 'get_embedding_neighbors',
        description: 'Find nearest neighbors for a given embedding vector. Useful for custom similarity searches.',
        inputSchema: {
            type: 'object',
            properties: {
                embedding_vector: {
                    type: 'array',
                    items: { type: 'number' },
                    description: '384-dimensional embedding vector',
                },
                k: {
                    type: 'number',
                    description: 'Number of neighbors to return, default 10',
                    minimum: 1,
                    default: 10,
                },
                threshold: {
                    type: 'number',
                    description: 'Similarity threshold (0-1), default 0.5',
                    minimum: 0,
                    maximum: 1,
                    default: 0.5,
                },
            },
            required: ['embedding_vector'],
        },
    },
    {
        name: 'get_note_content',
        description: 'Retrieve the full content of a note, optionally with specific blocks/sections extracted.',
        inputSchema: {
            type: 'object',
            properties: {
                note_path: {
                    type: 'string',
                    description: 'Path to the note',
                },
                include_blocks: {
                    type: 'array',
                    items: { type: 'string' },
                    description: 'Specific block headings to include (optional)',
                },
            },
            required: ['note_path'],
        },
    },
    {
        name: 'get_stats',
        description: 'Get statistics about the Smart Connections knowledge base (total notes, blocks, embedding model, etc.).',
        inputSchema: {
            type: 'object',
            properties: {},
        },
    },
    {
        name: 'rebuild_gte_index',
        description: 'Rebuild the semantic search index (EmbeddingGemma-300m, 768d). Only re-embeds notes whose content has changed. Use force=true to re-embed everything.',
        inputSchema: {
            type: 'object',
            properties: {
                force: {
                    type: 'boolean',
                    description: 'Force re-embed all notes (ignore cache), default false',
                    default: false,
                },
            },
        },
    },
];
// Handle tool list requests
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return { tools };
});
// Handle tool execution requests
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    try {
        switch (name) {
            case 'get_similar_notes': {
                const { note_path, threshold, limit } = GetSimilarNotesSchema.parse(args);
                const results = searchEngine.getSimilarNotes(note_path, threshold, limit);
                return {
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify(results, null, 2),
                        },
                    ],
                };
            }
            case 'get_connection_graph': {
                const { note_path, depth, threshold, max_per_level } = GetConnectionGraphSchema.parse(args);
                const graph = searchEngine.getConnectionGraph(note_path, depth, threshold, max_per_level);
                return {
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify(graph, null, 2),
                        },
                    ],
                };
            }
            case 'search_notes': {
                const { query, limit, threshold } = SearchNotesSchema.parse(args);
                const results = await searchEngine.searchByQuery(query, limit, threshold);
                return {
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify(results, null, 2),
                        },
                    ],
                };
            }
            case 'get_embedding_neighbors': {
                const { embedding_vector, k, threshold } = GetEmbeddingNeighborsSchema.parse(args);
                const results = searchEngine.getEmbeddingNeighbors(embedding_vector, k, threshold);
                return {
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify(results, null, 2),
                        },
                    ],
                };
            }
            case 'get_note_content': {
                const { note_path, include_blocks } = GetNoteContentSchema.parse(args);
                const result = searchEngine.getNoteWithContext(note_path, include_blocks);
                return {
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify(result, null, 2),
                        },
                    ],
                };
            }
            case 'get_stats': {
                GetStatsSchema.parse(args);
                const stats = searchEngine.getStats();
                return {
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify(stats, null, 2),
                        },
                    ],
                };
            }
            case 'rebuild_gte_index': {
                const { force } = RebuildGteIndexSchema.parse(args);
                // Get all note paths from Smart Connections loader
                const notePaths = Array.from(loader.getSources().keys());
                // If force, clear in-memory index so buildIndex() re-embeds every note.
                // In-place mutation preserves the reference held by searchEngine.
                if (force) {
                    gteEmbedder.clearIndex();
                }
                const stats = await gteEmbedder.buildIndex(notePaths, (p) => loader.readNoteContent(p), (current, total, p) => {
                    if (current % 20 === 0 || current === total) {
                        console.error(`GTE indexing: ${current}/${total} - ${p}`);
                    }
                });
                return {
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify({
                                message: 'GTE index rebuilt successfully',
                                ...stats,
                                total_indexed: gteEmbedder.getStats()?.entries ?? 0,
                                gte_stats: gteEmbedder.getStats(),
                            }, null, 2),
                        },
                    ],
                };
            }
            default:
                throw new Error(`Unknown tool: ${name}`);
        }
    }
    catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify({ error: errorMessage }, null, 2),
                },
            ],
            isError: true,
        };
    }
});
// Start the server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error('Smart Connections MCP Server running on stdio');
//# sourceMappingURL=index.js.map