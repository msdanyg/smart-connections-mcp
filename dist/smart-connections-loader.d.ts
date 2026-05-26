/**
 * Loader for Smart Connections data from .smart-env directory
 */
import type { SmartSource, SmartEnvConfig } from './types.js';
export declare class SmartConnectionsLoader {
    private vaultPath;
    private smartEnvPath;
    private config;
    private sources;
    constructor(vaultPath: string);
    /**
     * Initialize and load all Smart Connections data
     */
    initialize(): Promise<void>;
    /**
     * Walk the vault for .md files absent from the plugin's .ajson sources and
     * register minimal SmartSource entries (path-only). Closes the gap where the
     * MCP only saw plugin-embedded notes (e.g. 586 of 624). Note-level GTE
     * embedding works from disk; block-level extraction degrades gracefully.
     */
    private discoverUnindexedNotes;
    /**
     * Load smart_env.json configuration
     */
    private loadConfig;
    /**
     * Load all .ajson files from the multi directory
     */
    private loadSources;
    /**
     * Re-read .smart-env from disk (config + sources) without a process restart.
     * The Obsidian Smart Connections plugin flushes new/changed notes to
     * .smart-env/multi/*.ajson; this picks those up so the MCP no longer serves
     * a stale source list cached at startup. Returns sources count before/after.
     */
    reload(): Promise<{
        before: number;
        after: number;
    }>;
    /**
     * Get all sources
     */
    getSources(): Map<string, SmartSource>;
    /**
     * Get a specific source by path
     */
    getSource(notePath: string): SmartSource | undefined;
    /**
     * Get configuration
     */
    getConfig(): SmartEnvConfig | null;
    /**
     * Get the embedding model key from config
     */
    getEmbeddingModelKey(): string;
    /**
     * Get vault path
     */
    getVaultPath(): string;
    /**
     * Read the actual markdown content of a note
     */
    readNoteContent(notePath: string): string;
    /**
     * Extract content for specific blocks/sections
     */
    extractBlockContent(notePath: string, blockHeading: string): string;
}
//# sourceMappingURL=smart-connections-loader.d.ts.map