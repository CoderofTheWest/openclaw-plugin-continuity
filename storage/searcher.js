/**
 * Searcher — Cross-session hybrid retrieval via SQLite-vec + FTS5.
 *
 * Extracted from Clint's archiveIndexer.js (search logic) +
 * knowledgeSystem.js (vec MATCH query pattern).
 *
 * Shares the same continuity.db as Indexer. Runs two parallel searches:
 *   1. Semantic — vec_exchanges MATCH (embedding similarity)
 *   2. Keyword  — fts_exchanges MATCH (BM25 full-text search via FTS5)
 *
 * Results are fused using Reciprocal Rank Fusion (RRF), then re-ranked
 * with temporal decay so newer exchanges about the same topic outrank
 * older ones.
 *
 * Falls back to semantic-only if FTS5 table is not available.
 *
 * Temporal ranking pattern adapted from Clint's intelligentRetrieval.js:
 * recencyBoost = exp(-ageInDays / halfLife) * weight
 *
 * RRF pattern from Hindsight/Cormack et al.:
 * score(doc) = SUM(1 / (k + rank)) across ranked lists
 */

class Searcher {
    /**
     * @param {object} config - full plugin config
     * @param {string} dataDir - plugin data directory
     * @param {object} db - shared better-sqlite3 database instance (from Indexer)
     * @param {object} [embeddingProvider] - shared EmbeddingProvider instance (from embedding.js)
     */
    constructor(config = {}, dataDir, db, embeddingProvider = null) {
        this.db = db;
        this._embeddingFn = embeddingProvider; // shared provider (preferred)
        this._initialized = !!embeddingProvider; // ready if provider injected
        this.model = config.embedding?.model || 'Xenova/all-MiniLM-L6-v2';

        // Temporal ranking config
        this._recencyHalfLifeDays = config.search?.recencyHalfLifeDays || 14;
        this._recencyWeight = config.search?.recencyWeight || 0.15;

        // Formational memory config (knowledge.db backfill)
        this._formationalBoost = config.search?.formationalBoost || 0.08;
        this._formationalWeight = config.search?.formationalWeight || 0.85;
        this._arcWeight = config.search?.arcWeight || 0.5;

        // RRF config
        this._rrfK = config.search?.rrfK || 60;

        // FTS5 availability (checked on first search)
        this._fts5Checked = false;
        this._fts5Available = false;

        // Summary search availability (checked on first search)
        this._summarySearchChecked = false;
        this._summarySearchAvailable = false;
    }

    /**
     * Initialize embedding function for query generation.
     * Skipped if a shared EmbeddingProvider was injected via constructor.
     * Fallback: creates own pipeline with tensor disposal.
     */
    async initialize() {
        if (this._initialized) return;

        // Fallback: direct transformers.js with tensor disposal
        // (DefaultEmbeddingFunction is NOT used — it re-creates the pipeline
        //  on every generate() call, which is the primary memory leak source)
        try {
            const { pipeline } = require('@huggingface/transformers');
            const pipe = await pipeline('feature-extraction', this.model);
            this._embeddingFn = {
                generate: async (texts) => {
                    const results = [];
                    for (const text of texts) {
                        const output = await pipe(text, { pooling: 'mean', normalize: true });
                        results.push(Array.from(output.data));
                        // Dispose ONNX tensor to free native memory
                        if (typeof output.dispose === 'function') {
                            output.dispose();
                        }
                    }
                    return results;
                }
            };
            this._initialized = true;
            console.log('[Searcher] Fallback embedding pipeline ready (with tensor disposal)');
        } catch (err) {
            console.error('[Searcher] No embedding model available:', err.message);
        }
    }

    /**
     * Hybrid search: semantic + keyword retrieval fused with RRF.
     *
     * 1. Run semantic search (vec_exchanges MATCH)
     * 2. Run keyword search (fts_exchanges MATCH) — if available
     * 3. Fuse ranked lists with Reciprocal Rank Fusion
     * 4. Apply temporal decay boost
     * 5. Return top results sorted by composite score (higher = better)
     *
     * API is unchanged from the semantic-only version: callers don't
     * need to know about the dual-retrieval under the hood.
     *
     * @param {string} query - natural language search query
     * @param {number} [limit=5] - max results to return
     * @param {string} [agentId] - agent ID for graph results
     * @param {object} [scope] - optional spatial scope filter
     * @param {string[]} [scope.topics] - topic tags to filter by (OR logic)
     * @returns {{ exchanges: Array, distances: number[] }}
     */
    async search(query, limit = 5, agentId, scope = null) {
        if (!this.db) {
            return { exchanges: [], distances: [], error: 'Database not available' };
        }

        if (!this._initialized) {
            await this.initialize();
        }

        if (!this._embeddingFn) {
            return { exchanges: [], distances: [], error: 'Embedding model not available' };
        }

        // Check FTS5 availability once
        if (!this._fts5Checked) {
            this._checkFts5();
        }

        try {
            // Fetch more candidates than needed — re-ranking may reorder them.
            const fetchLimit = Math.min(limit * 2, 60);

            // Run semantic and keyword searches
            const semanticResults = await this._semanticSearch(query, fetchLimit);
            const keywordResults = this._fts5Available
                ? this._ftsSearch(query, fetchLimit)
                : [];

            // Pick up graph results from the graph plugin (if running)
            const graphResults = agentId ? this._getGraphResults(agentId) : [];

            // Search summary DAG (if available)
            if (!this._summarySearchChecked) this._checkSummarySearch();
            const summaryResults = this._summarySearchAvailable
                ? await this._summarySearch(query, Math.ceil(fetchLimit / 2))
                : [];

            // Build exchange lookup (all candidates from all lists)
            const exchangeMap = new Map();
            for (const r of semanticResults) {
                exchangeMap.set(r.id, r);
            }
            for (const r of keywordResults) {
                if (!exchangeMap.has(r.id)) {
                    exchangeMap.set(r.id, r);
                }
            }
            for (const r of graphResults) {
                if (!exchangeMap.has(r.id)) {
                    exchangeMap.set(r.id, r);
                }
            }
            for (const r of summaryResults) {
                if (!exchangeMap.has(r.id)) {
                    exchangeMap.set(r.id, r);
                }
            }

            // Fuse ranked lists with RRF (4-way when summary + graph results are available)
            const rrfScores = this._reciprocalRankFusion(
                [semanticResults, keywordResults, graphResults, summaryResults],
                this._rrfK
            );

            // Build scope tag set for spatial filtering boost
            const scopeTopics = scope?.topics?.length
                ? new Set(scope.topics.map(t => t.toLowerCase()))
                : null;

            // Apply temporal decay and build final results
            const now = Date.now();
            const fused = [];

            for (const [id, rrfScore] of rrfScores) {
                const ex = exchangeMap.get(id);
                if (!ex) continue;

                const ageMs = now - this._parseTimestamp(ex.date, ex.exchangeIndex, ex.createdAt);
                const ageDays = ageMs / (1000 * 60 * 60 * 24);
                const recencyBoost = Math.exp(-ageDays / this._recencyHalfLifeDays) * this._recencyWeight;

                // Source-aware scoring for formational memories (knowledge.db backfill)
                //
                // Formational memories are old (100+ days) so natural recency decay
                // reduces their boost to near-zero (~0.0008). To prevent them being
                // permanently buried, we replace decayed recency with a fixed floor
                // boost, then apply a type-specific weight multiplier.
                //
                // ARC-AGI puzzle solves get a lower weight (0.5x) than identity/
                // philosophical memories (0.85x) because they're task-specific.
                const meta = typeof ex.metadata === 'object' ? ex.metadata : this._parseMetadata(ex.metadata);
                let compositeScore;

                if (meta?.source === 'formational') {
                    const typeWeight = meta?.arcAgi
                        ? this._arcWeight
                        : this._formationalWeight;
                    compositeScore = rrfScore * (1 + this._formationalBoost) * typeWeight;
                } else {
                    // Standard scoring for ongoing (new) memories
                    compositeScore = rrfScore * (1 + recencyBoost);
                }

                // Spatial scope boost: if the exchange's topic tags overlap with
                // the requested scope, boost its score. This is a soft filter —
                // unscoped results still appear but rank lower when scope is active.
                if (scopeTopics) {
                    const tags = (ex.topicTags || ex.topic_tags || '').toLowerCase().split(',').filter(Boolean);
                    const hasOverlap = tags.some(t => scopeTopics.has(t));
                    if (hasOverlap) {
                        compositeScore *= 1.5; // 50% boost for scope match
                    }
                }

                fused.push({
                    id: ex.id,
                    date: ex.date,
                    exchangeIndex: ex.exchangeIndex,
                    userText: ex.userText,
                    agentText: ex.agentText,
                    combined: ex.combined,
                    metadata: ex.metadata,
                    distance: ex.distance ?? 1.0, // preserve for backward compat
                    rrfScore,
                    recencyBoost,
                    compositeScore
                });
            }

            // Sort by composite score (higher = more relevant + more recent)
            fused.sort((a, b) => b.compositeScore - a.compositeScore);

            // Take the top results
            const top = fused.slice(0, limit);

            return {
                exchanges: top,
                distances: top.map(r => r.distance)
            };
        } catch (error) {
            console.error('[Searcher] Search error:', error.message);
            return { exchanges: [], distances: [], error: error.message };
        }
    }

    /**
     * Format search results into a prompt-injectable block.
     * Results are sorted chronologically (oldest first) so the model
     * sees the natural progression of events — corrections appear
     * AFTER the original statements they correct.
     *
     * @param {{ exchanges: Array }} results - from search()
     * @param {number} [maxResults=3] - limit for prompt injection
     * @returns {string} formatted block or empty string
     */
    formatRetrieval(results, maxResults = 3) {
        if (!results?.exchanges || results.exchanges.length === 0) return '';

        const lines = ['[ARCHIVE RETRIEVAL]'];
        const items = results.exchanges.slice(0, maxResults);

        // Sort chronologically for display (oldest → newest)
        items.sort((a, b) => {
            if (a.date !== b.date) return a.date.localeCompare(b.date);
            return (a.exchangeIndex || 0) - (b.exchangeIndex || 0);
        });

        for (const ex of items) {
            lines.push(`[${ex.date}]`);
            if (ex.userText) lines.push(`  User: ${this._truncate(ex.userText, 800)}`);
            if (ex.agentText) lines.push(`  Agent: ${this._truncate(ex.agentText, 800)}`);
            lines.push('');
        }

        return lines.join('\n');
    }

    // ---------------------------------------------------------------
    // Graph results (from openclaw-plugin-graph via global bus)
    // ---------------------------------------------------------------

    /**
     * Pick up graph-based retrieval results exposed by the graph plugin.
     *
     * The graph plugin runs during before_agent_start (priority 8, before
     * continuity at priority 10) and posts its ranked exchange IDs to
     * global.__ocGraph.lastResults[agentId].
     *
     * We only use results that are fresh (<5s old) to avoid stale data
     * from a previous turn bleeding in.
     *
     * Graph results have exchange IDs but may not have full text — we
     * look them up in our exchanges table to get the full row.
     *
     * @param {string} agentId - agent ID to look up results for
     * @returns {Array} ranked results compatible with RRF input
     */
    _getGraphResults(agentId) {
        if (!global.__ocGraph?.lastResults?.[agentId]) return [];
        const cached = global.__ocGraph.lastResults[agentId];

        // Only use results from the current turn (within 5 seconds)
        if (Date.now() - cached.timestamp > 5000) return [];

        const exchanges = cached.exchanges || [];
        if (exchanges.length === 0) return [];

        // Look up full exchange data from our continuity DB
        const results = [];
        for (const gex of exchanges) {
            if (!gex.id) continue;
            try {
                const row = this.db.prepare(
                    'SELECT * FROM exchanges WHERE id = ?'
                ).get(gex.id);
                if (row) {
                    results.push({
                        id: row.id,
                        date: row.date,
                        exchangeIndex: row.exchange_index,
                        userText: row.user_text,
                        agentText: row.agent_text,
                        combined: row.combined,
                        metadata: this._parseMetadata(row.metadata),
                        createdAt: row.created_at,
                        distance: null, // no vector distance for graph results
                        graphScore: gex.score
                    });
                }
            } catch {
                // Exchange not found in continuity DB — skip
            }
        }
        return results;
    }

    // ---------------------------------------------------------------
    // Summary search (4th lane)
    // ---------------------------------------------------------------

    /**
     * Check if vec_summaries table exists.
     */
    _checkSummarySearch() {
        this._summarySearchChecked = true;
        try {
            const row = this.db.prepare(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_summaries'"
            ).get();
            this._summarySearchAvailable = !!row;
            if (this._summarySearchAvailable) {
                console.log('[Searcher] Summary search enabled (4-way hybrid mode)');
            }
        } catch {
            this._summarySearchAvailable = false;
        }
    }

    /**
     * Semantic search through summary DAG via vec_summaries.
     * Returns summary results formatted as pseudo-exchanges for RRF fusion.
     * Summaries get lower natural weight because they're contextual, not exact recall.
     *
     * @param {string} query
     * @param {number} limit
     * @returns {Array} ranked results compatible with RRF input
     */
    async _summarySearch(query, limit) {
        try {
            const embeddings = await this._embeddingFn.generate([query]);
            const queryEmbedding = embeddings?.[0];
            if (!queryEmbedding) return [];

            const results = this.db.prepare(`
                SELECT
                    s.id,
                    s.summary_text,
                    s.date_range_start,
                    s.date_range_end,
                    s.topics,
                    s.message_count,
                    s.level,
                    s.metadata,
                    s.created_at,
                    v.distance
                FROM vec_summaries v
                JOIN summaries s ON s.id = v.id
                WHERE v.embedding MATCH ?
                AND k = ?
                ORDER BY v.distance ASC
            `).all(new Float32Array(queryEmbedding), limit);

            return results.map(r => ({
                id: r.id,
                date: r.date_range_start,
                exchangeIndex: 0,
                userText: null,
                agentText: r.summary_text,
                combined: r.summary_text,
                metadata: { ...this._parseMetadata(r.metadata), isSummary: true, level: r.level },
                createdAt: r.created_at,
                distance: r.distance
            }));
        } catch (err) {
            console.warn('[Searcher] Summary search failed:', err.message);
            return [];
        }
    }

    // ---------------------------------------------------------------
    // Search strategies
    // ---------------------------------------------------------------

    /**
     * Semantic search via SQLite-vec embedding similarity.
     * Returns results ranked by vector distance (lower = more similar).
     *
     * @param {string} query
     * @param {number} limit
     * @returns {Array} ranked results with exchange data
     */
    async _semanticSearch(query, limit) {
        const embeddings = await this._embeddingFn.generate([query]);
        const queryEmbedding = embeddings?.[0];
        if (!queryEmbedding) return [];

        const results = this.db.prepare(`
            SELECT
                e.id,
                e.combined,
                e.user_text,
                e.agent_text,
                e.date,
                e.exchange_index,
                e.metadata,
                e.topic_tags,
                e.created_at,
                v.distance
            FROM vec_exchanges v
            JOIN exchanges e ON e.id = v.id
            WHERE v.embedding MATCH ?
            AND k = ?
            ORDER BY v.distance ASC
        `).all(new Float32Array(queryEmbedding), limit);

        return results.map(r => ({
            id: r.id,
            date: r.date,
            exchangeIndex: r.exchange_index,
            userText: r.user_text,
            agentText: r.agent_text,
            combined: r.combined,
            metadata: this._parseMetadata(r.metadata),
            topicTags: r.topic_tags || '',
            createdAt: r.created_at,
            distance: r.distance
        }));
    }

    /**
     * Keyword search via FTS5 full-text index.
     * Returns results ranked by BM25 relevance (lower rank = more relevant).
     *
     * FTS5's bm25() returns negative scores where more negative = more relevant,
     * so we negate and sort descending. The rank order is what matters for RRF
     * though, not the raw scores.
     *
     * @param {string} query
     * @param {number} limit
     * @returns {Array} ranked results with exchange data
     */
    _ftsSearch(query, limit) {
        const ftsQuery = this._sanitizeFtsQuery(query);
        if (!ftsQuery) return [];

        try {
            // FTS5 MATCH with BM25 ranking
            // bm25(fts_exchanges) returns negative floats: more negative = better match
            const results = this.db.prepare(`
                SELECT
                    f.id,
                    e.combined,
                    e.user_text,
                    e.agent_text,
                    e.date,
                    e.exchange_index,
                    e.metadata,
                    e.topic_tags,
                    e.created_at,
                    bm25(fts_exchanges) AS bm25_score
                FROM fts_exchanges f
                JOIN exchanges e ON e.id = f.id
                WHERE fts_exchanges MATCH ?
                ORDER BY bm25(fts_exchanges) ASC
                LIMIT ?
            `).all(ftsQuery, limit);

            return results.map(r => ({
                id: r.id,
                date: r.date,
                exchangeIndex: r.exchange_index,
                userText: r.user_text,
                agentText: r.agent_text,
                combined: r.combined,
                metadata: this._parseMetadata(r.metadata),
                topicTags: r.topic_tags || '',
                createdAt: r.created_at,
                distance: null, // no vector distance for keyword results
                bm25Score: r.bm25_score
            }));
        } catch (err) {
            console.warn('[Searcher] FTS5 search failed:', err.message);
            return [];
        }
    }

    // ---------------------------------------------------------------
    // Reciprocal Rank Fusion
    // ---------------------------------------------------------------

    /**
     * Merge multiple ranked result lists using Reciprocal Rank Fusion.
     *
     * For each document appearing in any list:
     *   score(doc) = SUM( 1 / (k + rank) ) across all lists
     *
     * k=60 is the standard constant that prevents top-ranked documents
     * from dominating — it smooths the contribution curve.
     *
     * @param {Array<Array>} rankedLists - arrays of results (each with .id)
     * @param {number} k - RRF constant (default 60)
     * @returns {Map<string, number>} id → fused score (higher = better)
     */
    _reciprocalRankFusion(rankedLists, k = 60) {
        const scores = new Map();

        for (const list of rankedLists) {
            if (!list || list.length === 0) continue;
            for (let rank = 0; rank < list.length; rank++) {
                const id = list[rank].id;
                const prev = scores.get(id) || 0;
                scores.set(id, prev + 1 / (k + rank + 1));
            }
        }

        return scores;
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    /**
     * Check if the FTS5 table exists in the database.
     * Called once, result cached.
     */
    _checkFts5() {
        this._fts5Checked = true;
        try {
            // sqlite_master query to check for the FTS5 table
            const row = this.db.prepare(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_exchanges'"
            ).get();
            this._fts5Available = !!row;
            if (this._fts5Available) {
                console.log('[Searcher] FTS5 keyword search enabled (hybrid mode)');
            } else {
                console.log('[Searcher] FTS5 not available — semantic-only mode');
            }
        } catch {
            this._fts5Available = false;
        }
    }

    /**
     * Sanitize a natural language query for FTS5 MATCH syntax.
     *
     * FTS5 has special characters (*, ", ^, NEAR, AND, OR, NOT) that
     * can cause parse errors if passed raw. We extract meaningful words
     * and join them with implicit AND (FTS5 default).
     *
     * Also applies porter stemming awareness: "running" and "run" match
     * because the tokenizer handles it, so we just need clean words.
     *
     * @param {string} query - raw user query
     * @returns {string} sanitized FTS5 query, or empty string if nothing useful
     */
    _sanitizeFtsQuery(query) {
        if (!query) return '';

        // Remove FTS5 operators and special chars
        let cleaned = query
            .replace(/[*"^(){}[\]:]/g, '')   // FTS5 special chars
            .replace(/\b(AND|OR|NOT|NEAR)\b/gi, '')  // FTS5 operators
            .replace(/[.,!?;]/g, ' ')         // punctuation to spaces
            .replace(/\s+/g, ' ')             // collapse whitespace
            .trim();

        // Split into words and filter out very short ones (noise)
        const words = cleaned.split(' ').filter(w => w.length >= 2);

        if (words.length === 0) return '';

        // Join with spaces — FTS5 implicit AND between terms.
        // Wrap each word in quotes to prevent them being parsed as operators.
        return words.map(w => `"${w}"`).join(' ');
    }

    /**
     * Parse a timestamp from the exchange data.
     * Tries created_at first, then falls back to date + exchange_index.
     */
    _parseTimestamp(date, exchangeIndex, createdAt) {
        if (createdAt) {
            const ts = new Date(createdAt).getTime();
            if (!isNaN(ts)) return ts;
        }
        // Fallback: date string + exchange_index as fractional day position
        const ts = new Date(date + 'T12:00:00Z').getTime();
        if (!isNaN(ts)) return ts + (exchangeIndex || 0) * 60000;
        return Date.now(); // last resort — no boost
    }

    _parseMetadata(metadataStr) {
        try {
            return JSON.parse(metadataStr || '{}');
        } catch {
            return {};
        }
    }

    _truncate(text, maxLen) {
        if (!text || text.length <= maxLen) return text;
        // Sentence-boundary aware: find last sentence end before maxLen
        const region = text.substring(0, maxLen);
        const lastSentenceEnd = Math.max(
            region.lastIndexOf('. '),
            region.lastIndexOf('? '),
            region.lastIndexOf('! '),
            region.lastIndexOf('.\n'),
            region.lastIndexOf('?\n'),
            region.lastIndexOf('!\n')
        );
        if (lastSentenceEnd > maxLen * 0.5) {
            return text.substring(0, lastSentenceEnd + 1) + '...';
        }
        return text.substring(0, maxLen - 3) + '...';
    }
}

module.exports = Searcher;
