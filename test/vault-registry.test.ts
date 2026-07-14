import { describe, it, expect } from 'vitest';
import * as path from 'node:path';
import { VaultRegistry, parseVaultPaths } from '../src/vault-registry.js';
import { AmbiguousNoteError, NoteNotFoundError, VaultNotFoundError } from '../src/errors.js';

const FIXTURE_A = path.resolve(import.meta.dirname, 'fixtures/vault-a');
const FIXTURE_B = path.resolve(import.meta.dirname, 'fixtures/vault-b');

describe('parseVaultPaths', () => {
  it('parses single, multiple, and alias env vars', () => {
    expect(parseVaultPaths({ SMART_VAULT_PATH: '/a' })).toEqual(['/a']);
    expect(parseVaultPaths({ SMART_VAULT_PATH: '/a, /b ,' })).toEqual(['/a', '/b']);
    expect(parseVaultPaths({ SMART_VAULT_PATHS: '/c', SMART_VAULT_PATH: '/a' })).toEqual(['/c']);
    expect(parseVaultPaths({})).toEqual([]);
  });
});

describe('VaultRegistry', () => {
  it('loads multiple vaults with basename names', () => {
    const reg = VaultRegistry.fromPaths([FIXTURE_A, FIXTURE_B]);
    expect(reg.vaults.map((v) => v.name)).toEqual(['vault-a', 'vault-b']);
    expect(reg.failures).toEqual([]);
    expect(reg.byName().length).toBe(2);
    expect(reg.byName('vault-b')[0].name).toBe('vault-b');
    expect(() => reg.byName('nope')).toThrow(VaultNotFoundError);
  });

  it('disambiguates duplicate basenames', () => {
    const reg = VaultRegistry.fromPaths([FIXTURE_A, FIXTURE_A]);
    expect(reg.vaults.map((v) => v.name)).toEqual(['vault-a', 'vault-a-2']);
  });

  it('captures failures without dying', () => {
    const reg = VaultRegistry.fromPaths([FIXTURE_A, '/nonexistent/vault']);
    expect(reg.vaults.length).toBe(1);
    expect(reg.failures.length).toBe(1);
    expect(reg.failures[0].error).toMatch(/smart-env/i);
  });

  it('resolves notes across vaults', () => {
    const reg = VaultRegistry.fromPaths([FIXTURE_A, FIXTURE_B]);
    expect(reg.resolveNote('Gamma.md').name).toBe('vault-a'); // unique
    expect(reg.resolveNote('Alpha.md', 'vault-b').name).toBe('vault-b'); // explicit
    expect(() => reg.resolveNote('Alpha.md')).toThrow(AmbiguousNoteError); // in both
    expect(() => reg.resolveNote('Missing.md')).toThrow(NoteNotFoundError);
  });
});
