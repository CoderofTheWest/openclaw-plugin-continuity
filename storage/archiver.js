/**
 * Archiver — Daily conversation storage with deduplication.
 *
 * Extracted from Clint's conversationArchiver.js (268 lines).
 * 100% portable per PDR — no Clint-coupled dependencies.
 *
 * Stores conversations as daily JSON files:
 *   {dataDir}/archive/{YYYY-MM-DD}.json
 *
 * Each file contains timestamped, deduplicated message exchanges.
 * Deduplication key: `${timestamp}_${sender}` prevents double-archiving.
 */

const fs = require('fs');
const path = require('path');

class Archiver {
    /**
     * @param {object} config - full plugin config (reads archive section)
     * @param {string} dataDir - plugin data directory
     */
    constructor(config = {}, dataDir) {
        const ac = config.archive || {};
        this.archiveDir = path.join(dataDir, ac.archiveDir || 'archive');
        this.retentionDays = ac.retentionDays || 90;

        // Ensure archive directory
        if (!fs.existsSync(this.archiveDir)) {
            fs.mkdirSync(this.archiveDir, { recursive: true });
        }
    }

    /**
     * Archive messages — group by date, deduplicate, write.
     *
     * @param {Array} messages - conversation messages
     * @param {Object} opts - options (sessionId, etc.)
     * @returns {{ archived: number, dates: string[] }}
     */
    archive(messages, opts = {}) {
        if (!messages || messages.length === 0) {
            return { archived: 0, dates: [] };
        }

        // Normalize messages to archivable format
        const normalized = messages
            .filter(m => m.role === 'user' || m.role === 'assistant')
            .map(m => ({
                timestamp: this._normalizeTimestamp(m.timestamp),
                sender: m.role === 'user' ? 'user' : 'agent',
                text: this._extractText(m),
                sessionId: opts.sessionId || null
            }));

        // Group by date
        const byDate = new Map();
        for (const msg of normalized) {
            const date = msg.timestamp.substring(0, 10); // YYYY-MM-DD
            if (!byDate.has(date)) byDate.set(date, []);
            byDate.get(date).push(msg);
        }

        let totalArchived = 0;
        const dates = [];

        for (const [date, dayMessages] of byDate) {
            const filePath = path.join(this.archiveDir, `${date}.json`);
            let existing = { date, messageCount: 0, messages: [] };

            // Load existing archive for this date
            if (fs.existsSync(filePath)) {
                try {
                    existing = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                } catch (err) {
                    console.warn(`[Archiver] Failed to read ${filePath}:`, err.message);
                }
            }

            // Build dedup set from existing messages
            const dedupKeys = new Set(
                existing.messages.map(m => `${m.timestamp}_${m.sender}`)
            );

            // Add new messages that don't already exist
            let added = 0;
            for (const msg of dayMessages) {
                const key = `${msg.timestamp}_${msg.sender}`;
                if (!dedupKeys.has(key)) {
                    existing.messages.push(msg);
                    dedupKeys.add(key);
                    added++;
                }
            }

            if (added > 0) {
                // Sort by timestamp
                existing.messages.sort((a, b) =>
                    new Date(a.timestamp) - new Date(b.timestamp)
                );
                existing.messageCount = existing.messages.length;

                // Write
                try {
                    fs.writeFileSync(filePath, JSON.stringify(existing, null, 2), 'utf8');
                    totalArchived += added;
                    dates.push(date);
                } catch (err) {
                    console.error(`[Archiver] Failed to write ${filePath}:`, err.message);
                }
            }
        }

        return { archived: totalArchived, dates };
    }

    /**
     * Search archived conversations by text content.
     *
     * @param {string} query - text to search for
     * @param {object} [opts]
     * @param {string} [opts.startDate] - YYYY-MM-DD
     * @param {string} [opts.endDate] - YYYY-MM-DD
     * @param {string} [opts.sender] - 'user' or 'agent'
     * @param {number} [opts.limit] - max results
     * @returns {Array<{ date: string, sender: string, text: string, timestamp: string }>}
     */
    search(query, opts = {}) {
        const dates = this.getDates();
        const results = [];
        const limit = opts.limit || 20;
        const lowerQuery = query.toLowerCase();

        for (const date of dates) {
            if (opts.startDate && date < opts.startDate) continue;
            if (opts.endDate && date > opts.endDate) continue;

            const conversation = this.getConversation(date);
            if (!conversation) continue;

            for (const msg of conversation.messages) {
                if (opts.sender && msg.sender !== opts.sender) continue;
                if (msg.text.toLowerCase().includes(lowerQuery)) {
                    results.push({ date, ...msg });
                    if (results.length >= limit) return results;
                }
            }
        }

        return results;
    }

    /**
     * Load a specific day's conversation.
     * @param {string} date - YYYY-MM-DD
     * @returns {object|null}
     */
    getConversation(date) {
        const filePath = path.join(this.archiveDir, `${date}.json`);
        if (!fs.existsSync(filePath)) return null;
        try {
            return JSON.parse(fs.readFileSync(filePath, 'utf8'));
        } catch (err) {
            console.warn(`[Archiver] Failed to read ${filePath}:`, err.message);
            return null;
        }
    }

    /**
     * List all available archive dates (sorted ascending).
     * @returns {string[]}
     */
    getDates() {
        try {
            return fs.readdirSync(this.archiveDir)
                .filter(f => f.endsWith('.json'))
                .map(f => f.replace('.json', ''))
                .sort();
        } catch (err) {
            return [];
        }
    }

    /**
     * Get archive statistics.
     * @returns {{ totalSessions: number, totalMessages: number, dateRange: { first: string, last: string } }}
     */
    getStats() {
        const dates = this.getDates();
        if (dates.length === 0) {
            return { totalSessions: 0, totalMessages: 0, dateRange: { first: null, last: null } };
        }

        let totalMessages = 0;
        for (const date of dates) {
            const conversation = this.getConversation(date);
            if (conversation) {
                totalMessages += conversation.messageCount || 0;
            }
        }

        return {
            totalSessions: dates.length,
            totalMessages,
            dateRange: {
                first: dates[0],
                last: dates[dates.length - 1]
            }
        };
    }

    /**
     * Get dates that haven't been indexed yet.
     * @param {Set|Array} indexedDates - dates already indexed
     * @returns {string[]}
     */
    getUnindexedDates(indexedDates) {
        const indexed = indexedDates instanceof Set
            ? indexedDates
            : new Set(indexedDates || []);
        return this.getDates().filter(d => !indexed.has(d));
    }

    /**
     * Remove archive files older than retention period.
     * @returns {{ removed: number }}
     */
    pruneOld() {
        const cutoff = new Date();
        cutoff.setDate(cutoff.getDate() - this.retentionDays);
        const cutoffStr = cutoff.toISOString().substring(0, 10);

        const dates = this.getDates();
        let removed = 0;

        for (const date of dates) {
            if (date < cutoffStr) {
                const filePath = path.join(this.archiveDir, `${date}.json`);
                try {
                    fs.unlinkSync(filePath);
                    removed++;
                } catch (err) {
                    console.warn(`[Archiver] Failed to remove ${filePath}:`, err.message);
                }
            }
        }

        return { removed };
    }

    // ---------------------------------------------------------------
    // Internal
    // ---------------------------------------------------------------

    _normalizeTimestamp(ts) {
        if (!ts) return new Date().toISOString();
        if (typeof ts === 'string') return ts;
        if (typeof ts === 'number') return new Date(ts).toISOString();
        if (ts instanceof Date) return ts.toISOString();
        return new Date().toISOString();
    }

    _extractText(msg) {
        if (!msg) return '';
        if (typeof msg.content === 'string') return msg.content;
        if (Array.isArray(msg.content)) {
            return msg.content.map(c => c.text || c.content || '').join(' ');
        }
        return String(msg.content || '');
    }
}

module.exports = Archiver;
