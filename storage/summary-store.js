/**
 * SummaryStore — DAG-based summary storage + async summarization queue.
 *
 * Stores hierarchical summaries in the shared continuity.db:
 *   - Level 0 (leaf): One summary per compaction event
 *   - Level 1 (branch): Condensed from ~6 leaf summaries (~1-2 days)
 *   - Level 2 (root): Condensed from ~6 branch summaries (~1-2 weeks)
 *
 * Uses the existing SQLite connection from Indexer (WAL mode, shared).
 * All tables are additive — CREATE IF NOT EXISTS, no migration needed.
 *
 * Inspired by lossless-claw's DAG architecture, adapted for Clint's
 * continuity plugin (hook-based, not a full ContextEngine replacement).
 */

const sqliteVec = require('sqlite-vec');

class SummaryStore {
    /**
     * @param {object} db - shared better-sqlite3 Database from Indexer
     * @param {object} config - full plugin config (reads summarization section)
     * @param {object} embeddingProvider - shared EmbeddingProvider instance
     */
    constructor(db, config = {}, embeddingProvider = null) {
        this.db = db;
        this.config = config.summarization || {};
        this.dimensions = config.embedding?.dimensions || 384;
        this._embeddingFn = embeddingProvider;
    }

    /**
     * Create summary tables (idempotent).
     * Call after Indexer._createTables() has already loaded sqlite-vec.
     */
    createTables() {
        if (!this.db) return;

        // Summary DAG nodes
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                level INTEGER NOT NULL,
                parent_id TEXT,
                agent_id TEXT NOT NULL,
                date_range_start TEXT,
                date_range_end TEXT,
                message_count INTEGER,
                summary_text TEXT NOT NULL,
                topics TEXT,
                anchors TEXT,
                entropy_avg REAL,
                metadata TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (parent_id) REFERENCES summaries(id)
            )
        `);

        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_summaries_agent ON summaries(agent_id)`);
        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_summaries_level ON summaries(level)`);
        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_summaries_date ON summaries(date_range_start, date_range_end)`);
        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_summaries_parent ON summaries(parent_id)`);

        // Summary vector table for semantic search
        try {
            this.db.exec(`
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_summaries USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[${this.dimensions}]
                )
            `);
        } catch (e) {
            if (!e.message.includes('already exists')) {
                console.warn('[SummaryStore] vec_summaries creation failed (non-fatal):', e.message);
            }
        }

        // Async summarization queue
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS summary_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                compaction_date TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                anchor_state TEXT,
                topic_state TEXT,
                entropy_score REAL,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                completed_at TEXT
            )
        `);

        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_queue_status ON summary_queue(status)`);

        // Add thread_id columns for infinite thread scoping
        try {
            this.db.exec(`ALTER TABLE summaries ADD COLUMN thread_id TEXT DEFAULT NULL`);
        } catch (e) {
            if (!e.message.includes('duplicate column')) throw e;
        }
        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_summaries_thread ON summaries(thread_id)`);

        try {
            this.db.exec(`ALTER TABLE summary_queue ADD COLUMN thread_id TEXT DEFAULT NULL`);
        } catch (e) {
            if (!e.message.includes('duplicate column')) throw e;
        }

        // Topic hierarchy persistence
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS topic_hierarchy (
                topic TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                parent_topic TEXT,
                mentions INTEGER DEFAULT 0,
                co_occurrence_count INTEGER DEFAULT 0,
                first_seen TEXT,
                last_seen TEXT,
                confidence REAL DEFAULT 0,
                PRIMARY KEY (topic, agent_id)
            )
        `);

        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_topic_parent ON topic_hierarchy(parent_topic, agent_id)`);

        console.log('[SummaryStore] Tables ready');
    }

    // -------------------------------------------------------------------
    // Queue management
    // -------------------------------------------------------------------

    /**
     * Enqueue messages for async LLM summarization.
     */
    enqueue(agentId, messages, anchorState, topicState, entropyScore, threadId = null) {
        if (!this.db) return;
        const stmt = this.db.prepare(`
            INSERT INTO summary_queue (agent_id, compaction_date, messages_json, anchor_state, topic_state, entropy_score, thread_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        `);
        stmt.run(
            agentId,
            new Date().toISOString().substring(0, 10),
            JSON.stringify(messages),
            JSON.stringify(anchorState || []),
            JSON.stringify(topicState || []),
            entropyScore || 0,
            threadId
        );
    }

    /**
     * Dequeue pending items for processing.
     * @param {number} limit
     * @returns {Array}
     */
    dequeue(limit = 3) {
        if (!this.db) return [];
        return this.db.prepare(`
            SELECT * FROM summary_queue WHERE status = 'pending'
            ORDER BY created_at ASC LIMIT ?
        `).all(limit);
    }

    /**
     * Mark a queue item as completed.
     */
    markCompleted(queueId) {
        if (!this.db) return;
        this.db.prepare(`
            UPDATE summary_queue SET status = 'completed', completed_at = datetime('now')
            WHERE id = ?
        `).run(queueId);
    }

    /**
     * Mark a queue item as failed.
     */
    markFailed(queueId, error) {
        if (!this.db) return;
        this.db.prepare(`
            UPDATE summary_queue SET status = 'failed', completed_at = datetime('now')
            WHERE id = ?
        `).run(queueId);
    }

    // -------------------------------------------------------------------
    // Summary CRUD
    // -------------------------------------------------------------------

    /**
     * Store a summary node in the DAG.
     * @param {object} summary
     */
    async storeSummary(summary) {
        if (!this.db) return;

        const stmt = this.db.prepare(`
            INSERT OR REPLACE INTO summaries
            (id, level, parent_id, agent_id, date_range_start, date_range_end,
             message_count, summary_text, topics, anchors, entropy_avg, metadata, thread_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `);

        stmt.run(
            summary.id,
            summary.level,
            summary.parentId || null,
            summary.agentId,
            summary.dateRangeStart,
            summary.dateRangeEnd,
            summary.messageCount || 0,
            summary.summaryText,
            JSON.stringify(summary.topics || []),
            JSON.stringify(summary.anchors || []),
            summary.entropyAvg || 0,
            JSON.stringify(summary.metadata || {}),
            summary.threadId || null
        );

        // Embed the summary for semantic search
        if (this._embeddingFn && summary.summaryText) {
            try {
                const embeddings = await this._embeddingFn.generate([summary.summaryText]);
                const embedding = embeddings?.[0];
                if (embedding) {
                    // sqlite-vec doesn't support INSERT OR REPLACE
                    this.db.prepare(`DELETE FROM vec_summaries WHERE id = ?`).run(summary.id);
                    this.db.prepare(`INSERT INTO vec_summaries (id, embedding) VALUES (?, ?)`)
                        .run(summary.id, new Float32Array(embedding));
                }
            } catch (err) {
                console.warn(`[SummaryStore] Embedding failed for ${summary.id}: ${err.message}`);
            }
        }
    }

    /**
     * Get a single summary by ID.
     */
    getSummary(id) {
        if (!this.db) return null;
        const row = this.db.prepare(`SELECT * FROM summaries WHERE id = ?`).get(id);
        return row ? this._deserializeSummary(row) : null;
    }

    /**
     * Get child summaries of a parent.
     */
    getChildren(parentId) {
        if (!this.db) return [];
        return this.db.prepare(`
            SELECT * FROM summaries WHERE parent_id = ?
            ORDER BY date_range_start ASC
        `).all(parentId).map(r => this._deserializeSummary(r));
    }

    // -------------------------------------------------------------------
    // DAG queries
    // -------------------------------------------------------------------

    /**
     * Get leaf summaries (level 0) without a parent, for condensation.
     */
    getUncondensedLeaves(agentId, threshold) {
        if (!this.db) return [];
        const t = threshold || this.config.leafCondenseThreshold || 6;
        const leaves = this.db.prepare(`
            SELECT * FROM summaries
            WHERE agent_id = ? AND level = 0 AND parent_id IS NULL
            ORDER BY date_range_start ASC
        `).all(agentId).map(r => this._deserializeSummary(r));

        return leaves.length >= t ? leaves : [];
    }

    /**
     * Get branch summaries (level 1) without a parent, for condensation.
     */
    getUncondensedBranches(agentId, threshold) {
        if (!this.db) return [];
        const t = threshold || this.config.branchCondenseThreshold || 6;
        const branches = this.db.prepare(`
            SELECT * FROM summaries
            WHERE agent_id = ? AND level = 1 AND parent_id IS NULL
            ORDER BY date_range_start ASC
        `).all(agentId).map(r => this._deserializeSummary(r));

        return branches.length >= t ? branches : [];
    }

    // -------------------------------------------------------------------
    // Search
    // -------------------------------------------------------------------

    /**
     * Semantic search over summaries using vector similarity.
     * @param {string} agentId
     * @param {Float32Array} embedding - query embedding
     * @param {number} limit
     * @returns {Array}
     */
    searchSummaries(agentId, embedding, limit = 5) {
        if (!this.db || !embedding) return [];

        try {
            const rows = this.db.prepare(`
                SELECT s.*, v.distance
                FROM vec_summaries v
                JOIN summaries s ON s.id = v.id
                WHERE s.agent_id = ?
                AND v.embedding MATCH ?
                ORDER BY v.distance ASC
                LIMIT ?
            `).all(agentId, new Float32Array(embedding), limit);

            return rows.map(r => ({
                ...this._deserializeSummary(r),
                distance: r.distance
            }));
        } catch (err) {
            console.warn('[SummaryStore] Search failed:', err.message);
            return [];
        }
    }

    /**
     * Get timeline summaries matching a topic (text search in topics JSON + summary_text).
     */
    getTimelineSummaries(agentId, topic, dateRange) {
        if (!this.db) return [];

        let sql = `
            SELECT * FROM summaries
            WHERE agent_id = ?
            AND (topics LIKE ? OR summary_text LIKE ?)
        `;
        const params = [agentId, `%${topic}%`, `%${topic}%`];

        if (dateRange?.start) {
            sql += ` AND date_range_end >= ?`;
            params.push(dateRange.start);
        }
        if (dateRange?.end) {
            sql += ` AND date_range_start <= ?`;
            params.push(dateRange.end);
        }

        sql += ` ORDER BY date_range_start ASC`;

        return this.db.prepare(sql).all(...params).map(r => this._deserializeSummary(r));
    }

    // -------------------------------------------------------------------
    // Topic hierarchy persistence
    // -------------------------------------------------------------------

    /**
     * Persist topic hierarchy to SQLite.
     * @param {string} agentId
     * @param {Array} topics - from topicTracker.getAllTopicsWithHierarchy()
     */
    persistTopicHierarchy(agentId, topics) {
        if (!this.db || !topics?.length) return;

        const upsert = this.db.prepare(`
            INSERT OR REPLACE INTO topic_hierarchy
            (topic, agent_id, parent_topic, mentions, co_occurrence_count, first_seen, last_seen, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `);

        const transaction = this.db.transaction(() => {
            for (const t of topics) {
                upsert.run(
                    t.topic, agentId, t.parentTopic || null,
                    t.mentions || 0, t.coOccurrenceCount || 0,
                    t.firstSeen || null, t.lastSeen || null,
                    t.confidence || 0
                );
            }
        });
        transaction();
    }

    /**
     * Load persisted topic hierarchy from SQLite.
     * @param {string} agentId
     * @returns {Array}
     */
    loadTopicHierarchy(agentId) {
        if (!this.db) return [];
        try {
            return this.db.prepare(`
                SELECT * FROM topic_hierarchy WHERE agent_id = ?
            `).all(agentId);
        } catch {
            return [];
        }
    }

    // -------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------

    /**
     * Get summary statistics for an agent.
     */
    getStats(agentId) {
        if (!this.db) return { total: 0, byLevel: {}, queuePending: 0 };

        const total = this.db.prepare(
            `SELECT COUNT(*) as count FROM summaries WHERE agent_id = ?`
        ).get(agentId)?.count || 0;

        const levels = this.db.prepare(
            `SELECT level, COUNT(*) as count FROM summaries WHERE agent_id = ? GROUP BY level`
        ).all(agentId);

        const byLevel = {};
        for (const row of levels) {
            byLevel[row.level] = row.count;
        }

        const queuePending = this.db.prepare(
            `SELECT COUNT(*) as count FROM summary_queue WHERE agent_id = ? AND status = 'pending'`
        ).get(agentId)?.count || 0;

        return { total, byLevel, queuePending };
    }

    // -------------------------------------------------------------------
    // Internal
    // -------------------------------------------------------------------

    _deserializeSummary(row) {
        return {
            id: row.id,
            level: row.level,
            parentId: row.parent_id,
            agentId: row.agent_id,
            dateRangeStart: row.date_range_start,
            dateRangeEnd: row.date_range_end,
            messageCount: row.message_count,
            summaryText: row.summary_text,
            topics: _safeJsonParse(row.topics, []),
            anchors: _safeJsonParse(row.anchors, []),
            entropyAvg: row.entropy_avg,
            metadata: _safeJsonParse(row.metadata, {}),
            createdAt: row.created_at
        };
    }
}

function _safeJsonParse(str, fallback) {
    if (!str) return fallback;
    try { return JSON.parse(str); } catch { return fallback; }
}

module.exports = SummaryStore;
