/**
 * Loader for Smart Connections data from .smart-env directory
 */
import type { SmartSource, SmartBlock, SmartEnvConfig } from './types.js';
export declare class SmartConnectionsLoader {
    private vaultPath;
    private smartEnvPath;
    private config;
    private sources;
    private blocks;
    constructor(vaultPath: string);
    /**
     * Initialize and load all Smart Connections data
     */
    initialize(): Promise<void>;
    /**
     * Load smart_env.json configuration
     */
    private loadConfig;
    /**
     * Load all .ajson files from the multi directory
     */
    private loadSources;
    /**
     * Get all sources
     */
    getSources(): Map<string, SmartSource>;
    /**
     * Get a specific source by path
     */
    getSource(notePath: string): SmartSource | undefined;
    /**
     * Get all blocks
     */
    getBlocks(): Map<string, SmartBlock>;
    /**
     * Get a specific block by key
     */
    getBlock(blockKey: string): SmartBlock | undefined;
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