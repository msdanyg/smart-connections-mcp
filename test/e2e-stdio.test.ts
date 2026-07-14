import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const FIXTURE_A = path.resolve(import.meta.dirname, 'fixtures/vault-a');
let client: Client;

function textOf(res: unknown): string {
  return (res as { content: Array<{ type: string; text: string }> }).content[0].text;
}

beforeAll(async () => {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [path.resolve(import.meta.dirname, '../dist/index.js')],
    env: { ...process.env, SMART_VAULT_PATH: FIXTURE_A, HF_HUB_OFFLINE: '1' },
    stderr: 'ignore',
  });
  client = new Client({ name: 'e2e', version: '1.0.0' });
  await client.connect(transport);
}, 60_000);

afterAll(async () => {
  await client?.close();
});

describe('stdio E2E (built dist)', () => {
  it('serves tools over stdio', async () => {
    const { tools } = await client.listTools();
    expect(tools.length).toBe(6);
  });

  it('search falls back to keyword mode for the fake model and flags it', async () => {
    const res = await client.callTool({ name: 'search_notes', arguments: { query: 'apples' } });
    const body = JSON.parse(textOf(res));
    expect(body.mode).toBe('keyword-fallback');
    expect(body.results[0].path).toBe('Alpha.md');
  }, 60_000);
});
