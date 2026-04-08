/**
 * KnowledgeStore — Indexed operational knowledge with semantic search.
 *
 * Stores discrete knowledge entries (learned from workspace files or
 * captured via the knowledge_note tool) in the shared continuity.db.
 * Each entry is embedded for semantic retrieval and indexed via FTS5
 * for keyword search.
 *
 * Uses the existing SQLite connection from Indexer (WAL mode, shared).
 * All tables are additive — CREATE IF NOT EXISTS, no migration needed.
 *
 * Search uses 3-way RRF fusion: semantic + keyword + recency.
 */

const crypto = require('crypto');

class KnowledgeStore {
    /**
     * @param {object} db - shared better-sqlite3 Database from Indexer
     * @param {object} config - full plugin config (reads knowledge section)
     * @param {object} embeddingProvider - shared EmbeddingProvider instance
     */
    constructor(db, config = {}, embeddingProvider = null) {
        this.db = db;
        this.config = config.knowledge || {};
        this.dimensions = config.embedding?.dimensions || 384;
        this._embeddingFn = embeddingProvider;
        this._rrfK = config.search?.rrfK || 60;
        this._fts5Available = false;
        this._fts5Checked = false;
    }

    /**
     * Create knowledge tables (idempotent).
     * Call after Indexer._createTables() has already loaded sqlite-vec.
     */
    createTables() {
        if (!this.db) return;

        // Knowledge entries
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                topic TEXT,
                section_path TEXT,
                source_type TEXT NOT NULL,
                source_hash TEXT,
                superseded_by TEXT,
                times_surfaced INTEGER DEFAULT 0,
                last_surfaced_at TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        `);

        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge_entries(agent_id)`);
        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge_entries(agent_id, topic)`);
        this.db.exec(`CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_entries(source_hash)`);

        // Vector table for semantic search
        try {
            this.db.exec(`
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_knowledge USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[${this.dimensions}]
                )
            `);
        } catch (e) {
            if (!e.message.includes('already exists')) {
                console.warn('[KnowledgeStore] vec_knowledge creation failed (non-fatal):', e.message);
            }
        }

        // FTS5 for keyword search
        try {
            this.db.exec(`
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_knowledge USING fts5(
                    id, content, topic,
                    tokenize='porter unicode61'
                )
            `);
            this._fts5Available = true;
            this._fts5Checked = true;
        } catch (e) {
            if (!e.message.includes('already exists')) {
                console.warn('[KnowledgeStore] fts_knowledge creation failed (non-fatal):', e.message);
            } else {
                this._fts5Available = true;
                this._fts5Checked = true;
            }
        }
    }

    // ─── Storage ────────────────────────────────────────────────────

    /**
     * Store a knowledge entry with embedding and FTS index.
     * @param {object} entry - { agentId, content, topic, sectionPath, sourceType, sourceHash, metadata }
     * @returns {string} entry ID
     */
    async store(entry) {
        const hash6 = entry.sourceHash ? entry.sourceHash.substring(0, 6) : crypto.randomBytes(3).toString('hex');
        const id = `kn_${entry.agentId}_${Date.now()}_${hash6}`;

        const embedding = await this._embed(entry.content);

        const insert = this.db.prepare(`
            INSERT INTO knowledge_entries
            (id, agent_id, content, topic, section_path, source_type, source_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `);

        const deleteVec = this.db.prepare(`DELETE FROM vec_knowledge WHERE id = ?`);
        const insertVec = this.db.prepare(`INSERT INTO vec_knowledge (id, embedding) VALUES (?, ?)`);

        const transaction = this.db.transaction(() => {
            insert.run(
                id, entry.agentId, entry.content,
                entry.topic || null, entry.sectionPath || null,
                entry.sourceType, entry.sourceHash || null,
                entry.metadata ? JSON.stringify(entry.metadata) : null
            );
            if (embedding) {
                deleteVec.run(id);
                insertVec.run(id, new Float32Array(embedding));
            }
            if (this._fts5Available) {
                try {
                    this.db.prepare(`DELETE FROM fts_knowledge WHERE id = ?`).run(id);
                    this.db.prepare(`INSERT INTO fts_knowledge (id, content, topic) VALUES (?, ?, ?)`)
                        .run(id, entry.content, entry.topic || '');
                } catch (e) {
                    // FTS index failure is non-fatal
                }
            }
        });
        transaction();

        return id;
    }

    /**
     * Update an existing knowledge entry (content changed, same source).
     * @param {string} id - entry ID
     * @param {object} updates - { content, topic, sectionPath, sourceHash, metadata }
     */
    async update(id, updates) {
        const embedding = updates.content ? await this._embed(updates.content) : null;

        const transaction = this.db.transaction(() => {
            if (updates.content) {
                this.db.prepare(`
                    UPDATE knowledge_entries
                    SET content = ?, topic = ?, section_path = ?, source_hash = ?,
                        metadata = ?, updated_at = datetime('now')
                    WHERE id = ?
                `).run(
                    updates.content, updates.topic || null,
                    updates.sectionPath || null, updates.sourceHash || null,
                    updates.metadata ? JSON.stringify(updates.metadata) : null,
                    id
                );
            }
            if (embedding) {
                this.db.prepare(`DELETE FROM vec_knowledge WHERE id = ?`).run(id);
                this.db.prepare(`INSERT INTO vec_knowledge (id, embedding) VALUES (?, ?)`)
                    .run(id, new Float32Array(embedding));
            }
            if (this._fts5Available && updates.content) {
                try {
                    this.db.prepare(`DELETE FROM fts_knowledge WHERE id = ?`).run(id);
                    this.db.prepare(`INSERT INTO fts_knowledge (id, content, topic) VALUES (?, ?, ?)`)
                        .run(id, updates.content, updates.topic || '');
                } catch (e) { /* non-fatal */ }
            }
        });
        transaction();
    }

    /**
     * Find entry by source hash (for dedup on re-index).
     * @param {string} agentId
     * @param {string} sourceHash - SHA-256 of source content
     * @returns {object|null} entry row or null
     */
    findBySourceHash(agentId, sourceHash) {
        return this.db.prepare(
            `SELECT * FROM knowledge_entries WHERE agent_id = ? AND source_hash = ? AND superseded_by IS NULL`
        ).get(agentId, sourceHash);
    }

    /**
     * Find entry by section path (for detecting content changes).
     * @param {string} agentId
     * @param {string} sectionPath
     * @returns {object|null}
     */
    findBySectionPath(agentId, sectionPath) {
        return this.db.prepare(
            `SELECT * FROM knowledge_entries WHERE agent_id = ? AND section_path = ? AND superseded_by IS NULL`
        ).get(agentId, sectionPath);
    }

    /**
     * Mark a knowledge entry as superseded by a newer one.
     * Superseded entries are excluded from search results (WHERE superseded_by IS NULL).
     * @param {string} id - entry to mark as superseded
     * @param {string} byId - the newer entry that supersedes this one
     */
    markSuperseded(id, byId) {
        this.db.prepare(`
            UPDATE knowledge_entries
            SET superseded_by = ?, updated_at = datetime('now')
            WHERE id = ?
        `).run(byId, id);
    }

    /**
     * Mark a knowledge entry as surfaced (injected into context).
     * @param {string} id
     */
    markSurfaced(id) {
        this.db.prepare(`
            UPDATE knowledge_entries
            SET times_surfaced = times_surfaced + 1, last_surfaced_at = datetime('now')
            WHERE id = ?
        `).run(id);
    }

    // ─── Search ─────────────────────────────────────────────────────

    /**
     * Hybrid search: semantic + keyword + recency, fused with RRF.
     *
     * @param {string} agentId
     * @param {string} query - natural language search query
     * @param {number[]} queryEmbedding - pre-computed embedding (from shared provider)
     * @param {number} [limit=5] - max results
     * @returns {Array} ranked entries with distance + compositeScore
     */
    async search(agentId, query, queryEmbedding, limit = 5) {
        if (!this.db) return [];

        const fetchLimit = Math.min(limit * 3, 30);

        // 1. Semantic search
        const semanticResults = this._semanticSearch(agentId, queryEmbedding, fetchLimit);

        // 2. Keyword search (FTS5)
        if (!this._fts5Checked) this._checkFts5();
        const keywordResults = this._fts5Available
            ? this._ftsSearch(agentId, query, fetchLimit)
            : [];

        // Build entry lookup
        const entryMap = new Map();
        for (const r of semanticResults) entryMap.set(r.id, r);
        for (const r of keywordResults) {
            if (!entryMap.has(r.id)) entryMap.set(r.id, r);
        }

        // 3. RRF fusion
        const rrfScores = this._reciprocalRankFusion(
            [semanticResults, keywordResults],
            this._rrfK
        );

        // 4. Apply recency boost + build final results
        const now = Date.now();
        const results = [];

        for (const [id, rrfScore] of rrfScores) {
            const entry = entryMap.get(id);
            if (!entry) continue;

            const createdAt = entry.created_at ? new Date(entry.created_at + 'Z').getTime() : now;
            const ageDays = (now - createdAt) / (1000 * 60 * 60 * 24);
            const recencyBoost = Math.exp(-ageDays / 14) * 0.1;

            results.push({
                id: entry.id,
                content: entry.content,
                topic: entry.topic,
                section_path: entry.section_path,
                source_type: entry.source_type,
                times_surfaced: entry.times_surfaced,
                distance: entry.distance ?? 1.0,
                compositeScore: rrfScore * (1 + recencyBoost),
                created_at: entry.created_at
            });
        }

        results.sort((a, b) => b.compositeScore - a.compositeScore);
        return results.slice(0, limit);
    }

    /**
     * Get entries by topic.
     */
    getByTopic(agentId, topic, limit = 10) {
        return this.db.prepare(
            `SELECT * FROM knowledge_entries
             WHERE agent_id = ? AND topic = ? AND superseded_by IS NULL
             ORDER BY created_at DESC LIMIT ?`
        ).all(agentId, topic, limit);
    }

    /**
     * Get all workspace-sourced entries for an agent (source_type starts with 'workspace').
     * Used by consolidateWorkspace() to detect entries from removed sections.
     *
     * @param {string} agentId
     * @returns {Array} entries with id, section_path, source_type
     */
    getWorkspaceEntries(agentId) {
        return this.db.prepare(
            `SELECT id, section_path, source_type FROM knowledge_entries
             WHERE agent_id = ? AND source_type LIKE 'workspace%' AND superseded_by IS NULL`
        ).all(agentId);
    }

    /**
     * Update the source_type of a knowledge entry.
     * Used to mark entries as 'workspace:archived' when their source section is removed.
     *
     * @param {string} id - entry ID
     * @param {string} newSourceType - e.g., 'workspace:archived'
     */
    updateSourceType(id, newSourceType) {
        this.db.prepare(
            `UPDATE knowledge_entries SET source_type = ?, updated_at = datetime('now') WHERE id = ?`
        ).run(newSourceType, id);
    }

    /**
     * Get stats for an agent's knowledge store.
     */
    getStats(agentId) {
        const total = this.db.prepare(
            `SELECT COUNT(*) as count FROM knowledge_entries WHERE agent_id = ? AND superseded_by IS NULL`
        ).get(agentId);
        const byTopic = this.db.prepare(
            `SELECT topic, COUNT(*) as count FROM knowledge_entries
             WHERE agent_id = ? AND superseded_by IS NULL
             GROUP BY topic ORDER BY count DESC`
        ).all(agentId);
        const bySource = this.db.prepare(
            `SELECT source_type, COUNT(*) as count FROM knowledge_entries
             WHERE agent_id = ? AND superseded_by IS NULL
             GROUP BY source_type`
        ).all(agentId);

        return {
            total: total?.count || 0,
            byTopic,
            bySource
        };
    }

    // ─── Internal search methods ────────────────────────────────────

    _semanticSearch(agentId, queryEmbedding, limit) {
        if (!queryEmbedding) return [];

        try {
            const rows = this.db.prepare(`
                SELECT v.id, v.distance, k.content, k.topic, k.section_path,
                       k.source_type, k.times_surfaced, k.created_at
                FROM vec_knowledge v
                JOIN knowledge_entries k ON v.id = k.id
                WHERE v.embedding MATCH ? AND k.rowid IN (
                    SELECT rowid FROM knowledge_entries WHERE agent_id = ? AND superseded_by IS NULL
                )
                ORDER BY v.distance
                LIMIT ?
            `).all(new Float32Array(queryEmbedding), agentId, limit);

            return rows;
        } catch (e) {
            // Fallback: search without agent filter in vec (sqlite-vec JOIN limitations)
            try {
                const rows = this.db.prepare(`
                    SELECT v.id, v.distance
                    FROM vec_knowledge v
                    WHERE v.embedding MATCH ?
                    ORDER BY v.distance
                    LIMIT ?
                `).all(new Float32Array(queryEmbedding), limit * 2);

                // Post-filter by agent
                const ids = rows.map(r => r.id);
                if (ids.length === 0) return [];

                const placeholders = ids.map(() => '?').join(',');
                const entries = this.db.prepare(`
                    SELECT * FROM knowledge_entries
                    WHERE id IN (${placeholders}) AND agent_id = ? AND superseded_by IS NULL
                `).all(...ids, agentId);

                const entryMap = new Map(entries.map(e => [e.id, e]));
                return rows
                    .filter(r => entryMap.has(r.id))
                    .map(r => ({ ...entryMap.get(r.id), distance: r.distance }))
                    .slice(0, limit);
            } catch (e2) {
                console.warn('[KnowledgeStore] Semantic search failed:', e2.message);
                return [];
            }
        }
    }

    _ftsSearch(agentId, query, limit) {
        try {
            // Sanitize FTS5 query: remove special chars, join with OR
            const terms = query.replace(/[^\w\s]/g, ' ').split(/\s+/).filter(t => t.length > 2);
            if (terms.length === 0) return [];
            const ftsQuery = terms.join(' OR ');

            const rows = this.db.prepare(`
                SELECT f.id, rank
                FROM fts_knowledge f
                JOIN knowledge_entries k ON f.id = k.id
                WHERE fts_knowledge MATCH ? AND k.agent_id = ? AND k.superseded_by IS NULL
                ORDER BY rank
                LIMIT ?
            `).all(ftsQuery, agentId, limit);

            // Enrich with full entry data
            if (rows.length === 0) return [];
            const ids = rows.map(r => r.id);
            const placeholders = ids.map(() => '?').join(',');
            const entries = this.db.prepare(
                `SELECT * FROM knowledge_entries WHERE id IN (${placeholders})`
            ).all(...ids);
            const entryMap = new Map(entries.map(e => [e.id, e]));

            return rows.map(r => ({
                ...(entryMap.get(r.id) || {}),
                id: r.id,
                ftsRank: r.rank
            }));
        } catch (e) {
            return [];
        }
    }

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

    _checkFts5() {
        this._fts5Checked = true;
        try {
            this.db.prepare(`SELECT COUNT(*) FROM fts_knowledge`).get();
            this._fts5Available = true;
        } catch (e) {
            this._fts5Available = false;
        }
    }

    // ─── Embedding helper ───────────────────────────────────────────

    async _embed(text) {
        if (!this._embeddingFn) return null;
        try {
            if (typeof this._embeddingFn.embed === 'function') {
                return await this._embeddingFn.embed(text);
            }
            if (typeof this._embeddingFn.generate === 'function') {
                const results = await this._embeddingFn.generate([text]);
                return results[0];
            }
            return null;
        } catch (e) {
            console.warn('[KnowledgeStore] Embedding failed:', e.message);
            return null;
        }
    }
}

module.exports = KnowledgeStore;
