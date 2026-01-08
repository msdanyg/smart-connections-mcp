/**
 * Loader for Smart Connections data from .smart-env directory
 */
import * as fs from 'fs';
import * as path from 'path';
export class SmartConnectionsLoader {
    vaultPath;
    smartEnvPath;
    config = null;
    sources = new Map();
    blocks = new Map();
    constructor(vaultPath) {
        this.vaultPath = vaultPath;
        this.smartEnvPath = path.join(vaultPath, '.smart-env');
    }
    /**
     * Initialize and load all Smart Connections data
     */
    async initialize() {
        // Check if .smart-env exists
        if (!fs.existsSync(this.smartEnvPath)) {
            throw new Error(`Smart Connections directory not found at: ${this.smartEnvPath}`);
        }
        // Load configuration
        await this.loadConfig();
        // Load all sources
        await this.loadSources();
    }
    /**
     * Load smart_env.json configuration
     */
    async loadConfig() {
        const configPath = path.join(this.smartEnvPath, 'smart_env.json');
        if (!fs.existsSync(configPath)) {
            throw new Error(`Configuration file not found at: ${configPath}`);
        }
        const configData = fs.readFileSync(configPath, 'utf-8');
        this.config = JSON.parse(configData);
    }
    /**
     * Load all .ajson files from the multi directory
     */
    async loadSources() {
        const multiPath = path.join(this.smartEnvPath, 'multi');
        if (!fs.existsSync(multiPath)) {
            throw new Error(`Multi directory not found at: ${multiPath}`);
        }
        const files = fs.readdirSync(multiPath);
        const ajsonFiles = files.filter(f => f.endsWith('.ajson'));
        console.error(`Loading ${ajsonFiles.length} source files...`);
        for (const file of ajsonFiles) {
            try {
                const filePath = path.join(multiPath, file);
                const content = fs.readFileSync(filePath, 'utf-8');
                // Parse the AJSON format (JSONL - one JSON object per line)
                // Each line is a single object like: "key": {...}
                const lines = content.trim().split('\n');
                for (const line of lines) {
                    if (!line.trim())
                        continue;
                    try {
                        // Each line is formatted as: "key1": {...}, "key2": {...}, "key3": {...},
                        // Remove trailing comma and wrap with curly braces to make valid JSON
                        const cleanedLine = line.replace(/,\s*$/, '');
                        const obj = JSON.parse(`{${cleanedLine}}`);
                        // Process all key-value pairs in the object
                        for (const key of Object.keys(obj)) {
                            // Process smart_sources entries
                            if (key.startsWith('smart_sources:')) {
                                const sourceData = obj[key];
                                // Skip entries with null/undefined paths
                                if (sourceData && sourceData.path) {
                                    this.sources.set(sourceData.path, sourceData);
                                }
                            }
                            // Process smart_blocks entries
                            else if (key.startsWith('smart_blocks:')) {
                                const blockData = obj[key];
                                // Use the key after "smart_blocks:" as the block key
                                const blockKey = key.replace('smart_blocks:', '');
                                if (blockData && blockData.embeddings) {
                                    // Store with the full key (path#heading)
                                    blockData.key = blockKey;
                                    this.blocks.set(blockKey, blockData);
                                }
                            }
                        }
                    }
                    catch (parseError) {
                        // Skip lines that can't be parsed
                        console.error(`Parse error in ${file}:`, parseError);
                    }
                }
            }
            catch (error) {
                console.error(`Error loading ${file}:`, error);
            }
        }
        console.error(`Loaded ${this.sources.size} sources and ${this.blocks.size} blocks successfully`);
    }
    /**
     * Get all sources
     */
    getSources() {
        return this.sources;
    }
    /**
     * Get a specific source by path
     */
    getSource(notePath) {
        return this.sources.get(notePath);
    }
    /**
     * Get all blocks
     */
    getBlocks() {
        return this.blocks;
    }
    /**
     * Get a specific block by key
     */
    getBlock(blockKey) {
        return this.blocks.get(blockKey);
    }
    /**
     * Get configuration
     */
    getConfig() {
        return this.config;
    }
    /**
     * Get the embedding model key from config
     */
    getEmbeddingModelKey() {
        if (!this.config) {
            throw new Error('Configuration not loaded');
        }
        // Extract the model key from the embed_model configuration
        const embedModel = this.config.smart_sources.embed_model;
        const adapter = embedModel.adapter;
        // The actual model key is nested in the adapter configuration
        // e.g., embed_model.transformers.model_key = "TaylorAI/bge-micro-v2"
        if (adapter && embedModel[adapter] && typeof embedModel[adapter] === 'object') {
            const adapterConfig = embedModel[adapter];
            if (adapterConfig.model_key) {
                return adapterConfig.model_key;
            }
        }
        // Fallback: find first object key that's not 'adapter'
        const modelKeys = Object.keys(embedModel).filter(k => k !== 'adapter' && typeof embedModel[k] === 'object');
        if (modelKeys.length === 0) {
            throw new Error('No embedding model found in configuration');
        }
        return modelKeys[0];
    }
    /**
     * Get vault path
     */
    getVaultPath() {
        return this.vaultPath;
    }
    /**
     * Read the actual markdown content of a note
     */
    readNoteContent(notePath) {
        const fullPath = path.join(this.vaultPath, notePath);
        if (!fs.existsSync(fullPath)) {
            throw new Error(`Note not found at: ${fullPath}`);
        }
        return fs.readFileSync(fullPath, 'utf-8');
    }
    /**
     * Extract content for specific blocks/sections
     */
    extractBlockContent(notePath, blockHeading) {
        const content = this.readNoteContent(notePath);
        const source = this.getSource(notePath);
        if (!source || !source.blocks[blockHeading]) {
            return '';
        }
        const [startLine, endLine] = source.blocks[blockHeading];
        const lines = content.split('\n');
        return lines.slice(startLine - 1, endLine).join('\n');
    }
}
//# sourceMappingURL=smart-connections-loader.js.map