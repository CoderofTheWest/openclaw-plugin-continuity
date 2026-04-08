/**
 * EmbeddingProvider — Shared, cached embedding pipeline with tensor disposal.
 *
 * Replaces both @chroma-core/default-embed (which re-creates a pipeline on
 * every generate() call) and the direct @huggingface/transformers fallback
 * (which never disposed output tensors).
 *
 * Key properties:
 * - Pipeline created ONCE, cached for all subsequent calls
 * - ONNX tensors explicitly disposed after data extraction (native memory!)
 * - Single instance shared between Indexer and Searcher per agent
 * - dispose() releases pipeline for GC when agent session ends
 *
 * Fixes: CoderofTheWest/openclaw-metacognitive-suite#1 (memory leak)
 */

class EmbeddingProvider {
    /**
     * @param {object} [config] - embedding config section
     * @param {string} [config.model] - HuggingFace model name
     * @param {number} [config.dimensions] - expected embedding dimensions
     */
    constructor(config = {}) {
        this.model = config.model || 'Xenova/all-MiniLM-L6-v2';
        this.dimensions = config.dimensions || 384;
        this._pipeline = null;
        this._initPromise = null;
        this._initialized = false;
    }

    /**
     * Lazy-initialize the pipeline. Safe to call multiple times.
     * Uses a promise guard to prevent concurrent initialization.
     */
    async initialize() {
        if (this._initialized) return;
        if (this._initPromise) {
            await this._initPromise;
            return;
        }

        this._initPromise = this._doInit();
        await this._initPromise;
        this._initPromise = null;
    }

    async _doInit() {
        // Try @huggingface/transformers directly — one pipeline, cached.
        // We skip @chroma-core/default-embed because its generate() calls
        // pipeline() on every invocation (no caching, no tensor disposal).
        try {
            const { pipeline } = require('@huggingface/transformers');
            this._pipeline = await pipeline('feature-extraction', this.model);

            // Warm up + verify dimensions
            const test = await this._pipeline('test', { pooling: 'mean', normalize: true });
            const dims = test.data?.length || 0;

            // Dispose the warmup tensor
            if (typeof test.dispose === 'function') {
                test.dispose();
            }

            if (dims !== this.dimensions) {
                console.warn(`[EmbeddingProvider] Dimension mismatch: expected ${this.dimensions}, got ${dims}`);
            }

            this._initialized = true;
            console.log(`[EmbeddingProvider] Pipeline ready — ${this.model} (${dims}d, cached)`);
        } catch (err) {
            console.error(`[EmbeddingProvider] Failed to initialize pipeline: ${err.message}`);
            throw err;
        }
    }

    /**
     * Generate embeddings for one or more texts.
     * Pipeline is reused across calls. Tensors are disposed after extraction.
     *
     * @param {string[]} texts - array of texts to embed
     * @returns {Promise<number[][]>} array of embedding vectors
     */
    async generate(texts) {
        if (!this._initialized) {
            await this.initialize();
        }

        if (!this._pipeline) {
            throw new Error('EmbeddingProvider not initialized');
        }

        const results = [];
        for (const text of texts) {
            const output = await this._pipeline(text, { pooling: 'mean', normalize: true });
            results.push(Array.from(output.data));

            // CRITICAL: Dispose the ONNX tensor to free native memory.
            // Without this, each embedding call leaks ~1-2KB of WASM/native memory
            // that V8's GC cannot see or reclaim.
            if (typeof output.dispose === 'function') {
                output.dispose();
            }
        }
        return results;
    }

    /**
     * Embed a single text string. Convenience wrapper around generate().
     *
     * @param {string} text - single text to embed
     * @returns {Promise<number[]>} embedding vector
     */
    async embed(text) {
        const [embedding] = await this.generate([text]);
        return embedding;
    }

    /**
     * Release the pipeline and all associated resources.
     * Call this when the agent session ends.
     */
    dispose() {
        if (this._pipeline) {
            // Some pipelines have a dispose method
            if (typeof this._pipeline.dispose === 'function') {
                try {
                    this._pipeline.dispose();
                } catch {
                    // Best effort
                }
            }
            this._pipeline = null;
        }
        this._initialized = false;
        this._initPromise = null;
    }

    /**
     * Whether the provider is ready to generate embeddings.
     * @returns {boolean}
     */
    get ready() {
        return this._initialized && this._pipeline !== null;
    }
}

module.exports = EmbeddingProvider;
