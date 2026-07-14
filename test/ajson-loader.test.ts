import { describe, it, expect } from 'vitest';
import { createVaultData, parseAjson, blockNotePath, isFrontmatterBlock } from '../src/ajson-loader.js';

const SAMPLE = `
"smart_sources:Alpha.md": {"path":"Alpha.md","class_name":"SmartSource","embeddings":{"m":{"vec":[1,0]}},"blocks":{"##Intro":[6,8]}},
"smart_blocks:Alpha.md##Intro": {"path":null,"class_name":"SmartBlock","lines":[6,8],"embeddings":{"m":{"vec":[0.8,0.6]}}},
"smart_sources:Old.md": {"path":"Old.md","embeddings":{"m":{"vec":[0,1]}}},
"smart_sources:Old.md": {"path":"Old.md","embeddings":{"m":{"vec":[1,1]}}},
"smart_sources:Gone.md": {"path":"Gone.md"},
"smart_sources:Gone.md": null,
this line is not json at all
"smart_sources:NoPath.md": {"class_name":"SmartSource"},
`;

describe('parseAjson', () => {
  it('parses sources and blocks, applies overrides and null deletions, skips junk', () => {
    const data = createVaultData();
    const warnings: string[] = [];
    parseAjson(SAMPLE, data, (m) => warnings.push(m));

    expect([...data.sources.keys()].sort()).toEqual(['Alpha.md', 'Old.md']);
    expect(data.sources.get('Old.md')?.embeddings?.m?.vec).toEqual([1, 1]); // later line wins
    expect(data.sources.has('Gone.md')).toBe(false); // null deletes
    expect(data.sources.has('NoPath.md')).toBe(false); // entries without path are dropped

    expect(data.blocks.size).toBe(1);
    const block = data.blocks.get('Alpha.md##Intro');
    expect(block?.key).toBe('Alpha.md##Intro'); // key backfilled from AJSON key
    expect(block?.lines).toEqual([6, 8]);

    expect(warnings.length).toBe(1); // the junk line
  });

  it('block helpers', () => {
    expect(blockNotePath('Sub/Note.md##A#{2}')).toBe('Sub/Note.md');
    expect(blockNotePath('Note.md')).toBe('Note.md');
    expect(isFrontmatterBlock('Note.md#---frontmatter---')).toBe(true);
    expect(isFrontmatterBlock('Note.md##Intro')).toBe(false);
  });
});
