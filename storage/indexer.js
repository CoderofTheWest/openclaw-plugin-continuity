/**
 * Indexer — Exchange pairing + SQLite-vec semantic embedding.
 *
 * Extracted from Clint's archiveIndexer.js (251 lines).
 * Replaces ChromaDB with SQLite-vec (same pattern as knowledgeSystem.js).
 *
 * Pairs user/agent exchanges from daily archives, generates 384-dim
 * embeddings via Xenova/all-MiniLM-L6-v2, and stores them in a SQLite
 * database for semantic retrieval.
 *
 * Requires: better-sqlite3, sqlite-vec, @chroma-core/default-embed
 */

const Database = require('better-sqlite3');
const sqliteVec = require('sqlite-vec');
const fs = require('fs');
const path = require('path');

class Indexer {
    /**
     * @param {object} config - full plugin config (reads embedding section)
     * @param {string} dataDir - plugin data directory
     * @param {object} [embeddingProvider] - shared EmbeddingProvider instance (from embedding.js)
     */
    constructor(config = {}, dataDir, embeddingProvider = null) {
        const ec = config.embedding || {};
        this.dbPath = path.join(dataDir, ec.dbFile || 'continuity.db');
        this.dimensions = ec.dimensions || 384;
        this.model = ec.model || 'Xenova/all-MiniLM-L6-v2';
        this.indexLogPath = path.join(dataDir, 'index-log.json');

        this.db = null;
        this._embeddingFn = embeddingProvider; // shared provider (preferred)
        this._embeddingPipeline = null;        // legacy fallback only
        this._initialized = false;
        this._fts5Available = false;
    }

    /**
     * Initialize: open DB, load sqlite-vec, create tables, init embedding model.
     * @returns {boolean} success
     */
    async initialize() {
        if (this._initialized) return true;

        try {
            // Ensure parent directory
            const dir = path.dirname(this.dbPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }

            // Open database and load sqlite-vec
            this.db = new Database(this.dbPath);
            sqliteVec.load(this.db);

            const vecVersion = this.db.prepare('SELECT vec_version()').get();
            console.log(`[Indexer] sqlite-vec loaded: ${vecVersion['vec_version()']}`);

            // WAL mode for concurrent read performance
            this.db.pragma('journal_mode = WAL');

            // Create tables
            this._createTables();

            // Initialize embeddings (skip if shared provider was injected)
            if (!this._embeddingFn) {
                await this._initEmbeddings();
            }

            this._initialized = true;
            console.log('[Indexer] Initialized — SQLite-vec ready');
            return true;
        } catch (error) {
            console.error('[Indexer] Initialization failed:', error.message);
            return false;
        }
    }

    /**
     * Index a day's conversations from the archive.
     *
     * @param {string} date - YYYY-MM-DD
     * @param {Array} messages - messages from the archiver
     * @param {object} [options] - optional indexing metadata
     * @param {string[]} [options.topicTags] - topic labels from TopicTracker for spatial scoping
     * @returns {{ indexed: number, date: string }}
     */
    async indexDay(date, messages, options = {}) {
        if (!this._initialized) {
            throw new Error('Indexer not initialized. Call initialize() first.');
        }

        if (!messages || messages.length === 0) {
            return { indexed: 0, date };
        }

        // Pair exchanges
        const exchanges = this._pairExchanges(messages);

        const insertExchange = this.db.prepare(`
            INSERT OR REPLACE INTO exchanges
            (id, date, exchange_index, user_text, agent_text, combined, metadata, topic_tags, thread_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        `);

        // Topic tags for spatial scoping (from TopicTracker)
        const topicTagsStr = (options.topicTags || []).join(',');
        const threadId = options.threadId || null;

        // sqlite-vec virtual tables don't support INSERT OR REPLACE,
        // so delete first then insert
        const deleteVec = this.db.prepare(`DELETE FROM vec_exchanges WHERE id = ?`);
        const insertVec = this.db.prepare(`
            INSERT INTO vec_exchanges (id, embedding)
            VALUES (?, ?)
        `);

        // FTS5 keyword index (parallel to vec)
        const deleteFts = this._fts5Available
            ? this.db.prepare(`DELETE FROM fts_exchanges WHERE id = ?`)
            : null;
        const insertFts = this._fts5Available
            ? this.db.prepare(`INSERT INTO fts_exchanges (id, user_text, agent_text) VALUES (?, ?, ?)`)
            : null;

        let indexed = 0;

        for (let i = 0; i < exchanges.length; i++) {
            const exchange = exchanges[i];
            const combined = this._formatExchange(exchange, date);
            const id = `exchange_${date}_${i}`;

            try {
                const embedding = await this._embed(combined);
                if (!embedding) continue;

                const metadata = JSON.stringify({
                    timestamp: exchange.user?.timestamp || exchange.agent?.timestamp,
                    hasUser: !!exchange.user,
                    hasAgent: !!exchange.agent
                });

                const userText = this._sanitizeForIndex(exchange.user?.text || '');
                const agentText = exchange.agent?.text || '';

                const transaction = this.db.transaction(() => {
                    insertExchange.run(
                        id, date, i,
                        userText,
                        agentText,
                        combined,
                        metadata,
                        topicTagsStr,
                        threadId
                    );
                    deleteVec.run(id);
                    insertVec.run(id, new Float32Array(embedding));

                    // FTS5 index (keyword search)
                    if (deleteFts && insertFts) {
                        deleteFts.run(id);
                        insertFts.run(id, userText, agentText);
                    }
                });
                transaction();

                indexed++;
            } catch (err) {
                console.warn(`[Indexer] Failed to index exchange ${id}:`, err.message);
            }
        }

        // Mark date as indexed
        this.markIndexed(date);

        console.log(`[Indexer] Indexed ${indexed} exchanges for ${date}`);
        return { indexed, date };
    }

    /**
     * Get the set of dates already indexed.
     * @returns {Set<string>}
     */
    getIndexedDates() {
        try {
            if (fs.existsSync(this.indexLogPath)) {
                const log = JSON.parse(fs.readFileSync(this.indexLogPath, 'utf8'));
                return new Set(log.dates || []);
            }
        } catch (err) {
            console.warn('[Indexer] Failed to read index log:', err.message);
        }
        return new Set();
    }

    /**
     * Record a date as indexed.
     * @param {string} date - YYYY-MM-DD
     */
    markIndexed(date) {
        try {
            let log = { dates: [], lastIndexed: null };
            if (fs.existsSync(this.indexLogPath)) {
                log = JSON.parse(fs.readFileSync(this.indexLogPath, 'utf8'));
            }
            if (!log.dates.includes(date)) {
                log.dates.push(date);
                log.dates.sort();
            }
            log.lastIndexed = new Date().toISOString();
            fs.writeFileSync(this.indexLogPath, JSON.stringify(log, null, 2), 'utf8');
        } catch (err) {
            console.warn('[Indexer] Failed to update index log:', err.message);
        }
    }

    /**
     * Get total indexed exchange count.
     * @returns {number}
     */
    getExchangeCount() {
        if (!this.db) return 0;
        try {
            const row = this.db.prepare('SELECT COUNT(*) as count FROM exchanges').get();
            return row?.count || 0;
        } catch {
            return 0;
        }
    }

    /**
     * Close the database connection.
     */
    close() {
        if (this.db) {
            this.db.close();
            this.db = null;
            this._initialized = false;
        }
    }

    // ---------------------------------------------------------------
    // Internal
    // ---------------------------------------------------------------

    _createTables() {
        // Main exchanges table
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS exchanges (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                exchange_index INTEGER,
                user_text TEXT,
                agent_text TEXT,
                combined TEXT,
                metadata TEXT,
                topic_tags TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        `);

        this.db.exec(`
            CREATE INDEX IF NOT EXISTS idx_exchanges_date ON exchanges(date);
        `);

        // Add topic_tags column if upgrading from older schema
        try {
            this.db.exec(`ALTER TABLE exchanges ADD COLUMN topic_tags TEXT DEFAULT ''`);
        } catch (e) {
            // Column already exists — expected for existing databases
            if (!e.message.includes('duplicate column')) throw e;
        }

        // Add thread_id column for infinite thread scoping
        try {
            this.db.exec(`ALTER TABLE exchanges ADD COLUMN thread_id TEXT DEFAULT NULL`);
        } catch (e) {
            if (!e.message.includes('duplicate column')) throw e;
        }
        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_exchanges_thread ON exchanges(thread_id)`);

        // Vector table
        try {
            this.db.exec(`
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_exchanges USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[${this.dimensions}]
                )
            `);
        } catch (e) {
            if (!e.message.includes('already exists')) {
                throw e;
            }
        }

        // FTS5 full-text search table (keyword search alongside semantic)
        // Uses porter stemmer so "running" matches "run", plus unicode61 for broad char support.
        this._createFts5Table();

        console.log('[Indexer] Database tables ready');
    }

    /**
     * Create the FTS5 virtual table for keyword search.
     * Backfills from existing exchanges if the table is new.
     */
    _createFts5Table() {
        try {
            // content='' would make it contentless (smaller) but we need DELETE support
            // for INSERT OR REPLACE semantics, so we use a regular FTS5 table.
            this.db.exec(`
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_exchanges USING fts5(
                    id,
                    user_text,
                    agent_text,
                    tokenize='porter unicode61'
                )
            `);

            // Check if backfill is needed: if fts_exchanges is empty but exchanges has rows
            const ftsCount = this.db.prepare('SELECT COUNT(*) as count FROM fts_exchanges').get();
            const mainCount = this.db.prepare('SELECT COUNT(*) as count FROM exchanges').get();

            if (ftsCount.count === 0 && mainCount.count > 0) {
                console.log(`[Indexer] Backfilling FTS5 index for ${mainCount.count} existing exchanges...`);

                const insertFts = this.db.prepare(
                    `INSERT INTO fts_exchanges (id, user_text, agent_text) VALUES (?, ?, ?)`
                );

                // Use iterate() instead of all() to avoid loading all exchanges
                // into memory at once (fixes memory spike with 13K+ exchanges)
                const iter = this.db.prepare(
                    'SELECT id, user_text, agent_text FROM exchanges'
                ).iterate();

                let backfillCount = 0;
                const backfill = this.db.transaction(() => {
                    for (const row of iter) {
                        insertFts.run(row.id, row.user_text || '', row.agent_text || '');
                        backfillCount++;
                    }
                });
                backfill();

                console.log(`[Indexer] FTS5 backfill complete: ${backfillCount} exchanges indexed`);
            }
        } catch (e) {
            // FTS5 is an enhancement — don't block startup if it fails
            console.warn('[Indexer] FTS5 table creation failed (non-fatal):', e.message);
            this._fts5Available = false;
            return;
        }
        this._fts5Available = true;
    }

    /**
     * Initialize embedding model.
     * Primary: @chroma-core/default-embed (same as knowledgeSystem.js)
     * Fallback: @huggingface/transformers pipeline
     */
    async _initEmbeddings() {
        try {
            const { DefaultEmbeddingFunction } = require('@chroma-core/default-embed');
            this._embeddingFn = new DefaultEmbeddingFunction();

            // Warm up
            const test = await this._embeddingFn.generate(['test']);
            if (test && test[0] && test[0].length === this.dimensions) {
                console.log(`[Indexer] Embedding model ready (${this.dimensions} dimensions)`);
                return;
            }
            console.warn(`[Indexer] Dimension mismatch: expected ${this.dimensions}, got ${test?.[0]?.length}`);
        } catch (err) {
            console.warn('[Indexer] @chroma-core/default-embed failed:', err.message);
        }

        // Fallback: direct transformers.js (with tensor disposal)
        try {
            const { pipeline } = require('@huggingface/transformers');
            this._embeddingPipeline = await pipeline('feature-extraction', this.model);
            this._embeddingFn = {
                generate: async (texts) => {
                    const results = [];
                    for (const text of texts) {
                        const output = await this._embeddingPipeline(text, { pooling: 'mean', normalize: true });
                        results.push(Array.from(output.data));
                        // Dispose ONNX tensor to free native memory
                        if (typeof output.dispose === 'function') {
                            output.dispose();
                        }
                    }
                    return results;
                }
            };
            console.log('[Indexer] Fallback embedding model ready');
        } catch (fallbackErr) {
            console.error('[Indexer] All embedding models failed:', fallbackErr.message);
            throw new Error('No embedding model available. Install @chroma-core/default-embed or @huggingface/transformers.');
        }
    }

    /**
     * Generate embedding for a single text.
     * @param {string} text
     * @returns {Float32Array|null}
     */
    async _embed(text) {
        if (!this._embeddingFn) return null;
        try {
            const results = await this._embeddingFn.generate([text]);
            return results?.[0] || null;
        } catch (err) {
            console.warn('[Indexer] Embedding generation failed:', err.message);
            return null;
        }
    }

    /**
     * Pair messages into user→agent exchanges.
     * @param {Array} messages - sorted by timestamp
     * @returns {Array<{ user: object|null, agent: object|null }>}
     */
    _pairExchanges(messages) {
        const exchanges = [];
        let currentExchange = { user: null, agent: null };

        for (const msg of messages) {
            if (msg.sender === 'user') {
                // If we already have a user message, push current and start new
                if (currentExchange.user) {
                    exchanges.push(currentExchange);
                    currentExchange = { user: null, agent: null };
                }
                currentExchange.user = msg;
            } else if (msg.sender === 'agent') {
                currentExchange.agent = msg;
                exchanges.push(currentExchange);
                currentExchange = { user: null, agent: null };
            }
        }

        // Push any remaining
        if (currentExchange.user || currentExchange.agent) {
            exchanges.push(currentExchange);
        }

        return exchanges;
    }

    /**
     * Lightweight re-strip for index-time safety net.
     * Catches context block content that leaked through the primary strip
     * in agent_end (e.g., [PROJECT CONTEXT] body, workspace file references).
     * Not a full replacement for _stripContextBlocks — just catches obvious leaks.
     */
    _sanitizeForIndex(text) {
        if (!text) return '';

        // If text starts with a known context block header, it's entirely polluted
        const BLOCK_HEADERS = [
            '[CONTINUITY CONTEXT]', '[STABILITY CONTEXT]', '[PROJECT CONTEXT',
            '[CONTEMPLATION STATE]', '[NIGHTSHIFT REPORT', '[ARCHIVE RETRIEVAL]',
            '[GRAPH CONTEXT]', '[GRAPH NOTE]', '[TOPIC NOTE]',
        ];
        if (BLOCK_HEADERS.some(h => text.startsWith(h))) {
            // Try to find user text after a timestamp
            const tsMatch = text.match(/\n\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s\d{4}-\d{2}-\d{2}\s[^\]]*\]\s*/g);
            if (tsMatch) {
                const lastTs = tsMatch[tsMatch.length - 1];
                const idx = text.lastIndexOf(lastTs);
                return text.substring(idx + lastTs.length).trim();
            }
            // No timestamp — this is entirely context (e.g., heartbeat)
            return '';
        }

        // If text contains workspace file references as primary content, it's polluted
        if (/^# Project:/.test(text) || /^## Stack\b/.test(text) || /^## Architecture\b/.test(text)) {
            return '';
        }

        return text;
    }

    /**
     * Format an exchange for embedding.
     * @param {object} exchange - { user, agent }
     * @param {string} date
     * @returns {string}
     */
    _formatExchange(exchange, date) {
        const time = exchange.user?.timestamp?.substring(11, 16) ||
                     exchange.agent?.timestamp?.substring(11, 16) || '00:00';
        const parts = [`[${date} ${time}]`];

        if (exchange.user?.text) {
            parts.push(`User: ${this._sanitizeForIndex(exchange.user.text)}`);
        }
        if (exchange.agent?.text) {
            parts.push(`Agent: ${exchange.agent.text}`);
        }

        return parts.join('\n');
    }
}

module.exports = Indexer;
