/**
 * Parser for Smart Connections AJSON files (`.smart-env/multi/*.ajson`).
 * Format: one entry per line, `"collection:key": {json},` — append semantics:
 * later lines override earlier ones, a null value deletes the key.
 */

import type { SmartBlock, SmartSource, VaultData } from './types.js';

export function createVaultData(): VaultData {
  return { sources: new Map(), blocks: new Map() };
}

export function parseAjson(
  content: string,
  data: VaultData,
  warn: (msg: string) => void = () => {},
): void {
  const lines = content.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim().replace(/,\s*$/, '');
    if (!line) continue;
    let obj: Record<string, unknown>;
    try {
      obj = JSON.parse(`{${line}}`);
    } catch {
      warn(`skipping unparseable line ${i + 1}`);
      continue;
    }
    for (const [rawKey, value] of Object.entries(obj)) {
      const sep = rawKey.indexOf(':');
      if (sep === -1) continue;
      const collection = rawKey.slice(0, sep);
      const key = rawKey.slice(sep + 1);
      if (collection === 'smart_sources') {
        if (value === null) {
          data.sources.delete(key);
        } else if (typeof value === 'object' && (value as SmartSource).path) {
          data.sources.set(key, value as SmartSource);
        }
      } else if (collection === 'smart_blocks') {
        if (value === null) {
          data.blocks.delete(key);
        } else if (typeof value === 'object') {
          const block = value as SmartBlock;
          block.key = block.key ?? key;
          data.blocks.set(key, block);
        }
      }
    }
  }
}

/** Note path a block belongs to: portion of its key before the first '#'. */
export function blockNotePath(blockKey: string): string {
  const i = blockKey.indexOf('#');
  return i === -1 ? blockKey : blockKey.slice(0, i);
}

export function isFrontmatterBlock(blockKey: string): boolean {
  return blockKey.includes('#---frontmatter---');
}
