import { describe, it, expect } from 'vitest';
import { NoteNotFoundError } from '../src/errors.js';

describe('toolchain smoke', () => {
  it('compiles and imports src modules', () => {
    expect(new NoteNotFoundError('x')).toBeInstanceOf(Error);
  });
});
