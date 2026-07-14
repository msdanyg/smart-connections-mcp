import * as path from 'node:path';
import { PathEscapeError } from './errors.js';

/** Resolve a vault-relative note path, refusing anything that escapes the vault root. */
export function resolveInsideVault(vaultRoot: string, notePath: string): string {
  const root = path.resolve(vaultRoot);
  const resolved = path.resolve(root, notePath);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    throw new PathEscapeError(`Path escapes vault: ${notePath}`);
  }
  return resolved;
}
