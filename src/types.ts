/**
 * Type definitions for Smart Connections MCP Server
 */

export interface SmartSource {
  path: string;
  embeddings: {
    [modelKey: string]: {
      vec: number[];
      last_embed: {
        hash: string;
        tokens: number;
      };
    };
  };
  last_read: {
    hash: string;
    at: number;
  };
  class_name: string;
  last_import: {
    mtime: number;
    size: number;
    at: number;
    hash: string;
  };
  blocks: {
    [heading: string]: [number, number]; // [start_line, end_line]
  };
}

export interface SmartBlock {
  key: string; // e.g., "file.md##Heading" or "file.md##{1}"
  path: string | null;
  embeddings: {
    [modelKey: string]: {
      vec: number[];
      last_embed: {
        hash: string;
        tokens: number;
      };
    };
  };
  lines: [number, number]; // [start_line, end_line]
  size: number;
  class_name: string;
  outlinks?: Array<{
    title: string;
    target: string;
    line: number;
  }>;
}

export interface SmartEnvConfig {
  is_obsidian_vault: boolean;
  smart_blocks: {
    embed_blocks: boolean;
    min_chars: number;
  };
  smart_sources: {
    single_file_data_path: string;
    min_chars: number;
    embed_model: {
      adapter: string;
      [key: string]: any;
    };
    excluded_headings: string;
    file_exclusions: string;
    folder_exclusions: string;
  };
  smart_chat_threads?: {
    chat_model: {
      adapter: string;
      [key: string]: any;
    };
    active_thread_key?: string;
  };
}

export interface SimilarNote {
  path: string;
  similarity: number;
  blocks?: string[];
  matchedContent?: string;
  lines?: [number, number]; // [start_line, end_line] for block-level results
  isBlock?: boolean; // true if this is a block result, false/undefined for note-level
}

export interface ConnectionNode {
  root: string;
  path: string;
  depth: number;
  connections: ConnectionNode[];
  similarity: number;
}

export interface ConnectionGraph {
  root: string;
  connections: Array<{
    path: string;
    depth: number;
    similarity: number;
  }>;
}

export interface NoteContent {
  path: string;
  content: string;
  blocks: string[];
}
