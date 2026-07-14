/** Owns all configured vaults; resolves vault names and note paths. */

import * as path from 'node:path';
import { AmbiguousNoteError, NoteNotFoundError, VaultNotFoundError } from './errors.js';
import { Vault } from './vault.js';

export interface VaultFailure {
  name: string;
  path: string;
  error: string;
}

export function parseVaultPaths(
  env: Record<string, string | undefined> = process.env,
): string[] {
  const raw = env.SMART_VAULT_PATHS ?? env.SMART_VAULT_PATH ?? '';
  return raw.split(',').map((s) => s.trim()).filter(Boolean);
}

export class VaultRegistry {
  vaults: Vault[] = [];
  failures: VaultFailure[] = [];

  static fromPaths(paths: string[]): VaultRegistry {
    const reg = new VaultRegistry();
    const used = new Set<string>();
    for (const p of paths) {
      const base = path.basename(p.replace(/[\\/]+$/, '')) || p;
      let name = base;
      for (let n = 2; used.has(name); n++) name = `${base}-${n}`;
      used.add(name);
      try {
        reg.vaults.push(Vault.load(p, name));
      } catch (e) {
        reg.failures.push({ name, path: p, error: e instanceof Error ? e.message : String(e) });
      }
    }
    return reg;
  }

  byName(name?: string): Vault[] {
    if (name === undefined) return this.vaults;
    const vault = this.vaults.find((v) => v.name === name);
    if (!vault) {
      throw new VaultNotFoundError(
        `Unknown vault "${name}". Available: ${this.vaults.map((v) => v.name).join(', ')}`,
      );
    }
    return [vault];
  }

  resolveNote(notePath: string, vaultName?: string): Vault {
    const candidates = this.byName(vaultName).filter((v) => v.data.sources.has(notePath));
    if (candidates.length === 0) {
      throw new NoteNotFoundError(`Note not found in any vault: ${notePath}`);
    }
    if (candidates.length > 1) {
      throw new AmbiguousNoteError(
        `Note "${notePath}" exists in vaults: ${candidates.map((v) => v.name).join(', ')} — pass the "vault" parameter`,
      );
    }
    return candidates[0];
  }
}
