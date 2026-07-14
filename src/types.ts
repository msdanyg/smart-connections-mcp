/** Shapes of Smart Connections `.smart-env` data (v3.x) and server results. */

export interface EmbeddingData {
  vec?: number[];
}

export interface SmartSource {
  path: string;
  class_name?: string;
  embeddings?: Record<string, EmbeddingData>;
  /** heading key -> [startLine, endLine], 1-indexed inclusive */
  blocks?: Record<string, [number, number]>;
  last_import?: { mtime?: number; size?: number };
  metadata?: Record<string, unknown>;
}

export interface SmartBlock {
  /** e.g. "Note.md##Heading#{2}" — always set (from the AJSON key if absent) */
  key: string;
  path?: string | null;
  class_name?: string;
  /** [startLine, endLine], 1-indexed inclusive */
  lines?: [number, number];
  embeddings?: Record<string, EmbeddingData>;
  size?: number;
}

export interface VaultData {
  /** note path -> source */
  sources: Map<string, SmartSource>;
  /** block key -> block */
  blocks: Map<string, SmartBlock>;
}

export interface SmartEnvConfig {
  smart_sources?: {
    embed_model?: { adapter?: string; [key: string]: unknown };
  };
}

export interface SearchResult {
  path: string;
  vault: string;
  similarity: number;
  scope: 'note' | 'block';
  /** heading portion of the block key, e.g. "##Intro" — block hits only */
  block?: string;
  snippet: string;
  /** set on rows produced by keyword fallback; semantic rows omit it */
  match?: 'keyword';
}

export interface SearchResponse {
  mode: 'semantic' | 'keyword-fallback';
  warning?: string;
  results: SearchResult[];
}

export interface SimilarNote {
  path: string;
  vault: string;
  similarity: number;
  blocks: string[];
}

export interface ConnectionGraph {
  root: string;
  vault: string;
  connections: Array<{ path: string; depth: number; similarity: number }>;
}

export interface VaultInfo {
  name: string;
  path: string;
  status: 'ok' | 'error';
  error?: string;
  notes?: number;
  blocks?: number;
  indexed?: number;
  embeddingDim?: number;
  modelKey?: string;
}
