import { describe, it, expect } from 'vitest';
import * as path from 'node:path';
import { resolveInsideVault } from '../src/paths.js';
import { PathEscapeError } from '../src/errors.js';

const ROOT = path.resolve('/tmp/vault');

describe('resolveInsideVault', () => {
  it('resolves normal and nested note paths', () => {
    expect(resolveInsideVault(ROOT, 'Note.md')).toBe(path.join(ROOT, 'Note.md'));
    expect(resolveInsideVault(ROOT, 'Sub/Deep/Note.md')).toBe(path.join(ROOT, 'Sub', 'Deep', 'Note.md'));
  });

  it('rejects traversal and absolute escapes', () => {
    expect(() => resolveInsideVault(ROOT, '../secrets.txt')).toThrow(PathEscapeError);
    expect(() => resolveInsideVault(ROOT, 'a/../../etc/passwd')).toThrow(PathEscapeError);
    expect(() => resolveInsideVault(ROOT, '/etc/passwd')).toThrow(PathEscapeError);
  });

  it('allows internal ".." that stays inside', () => {
    expect(resolveInsideVault(ROOT, 'a/../Note.md')).toBe(path.join(ROOT, 'Note.md'));
  });
});
