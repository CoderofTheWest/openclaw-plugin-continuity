/**
 * MaintenanceService — Background batch indexing, pruning, and health reporting.
 *
 * Uses shared Archiver and Indexer instances from the main plugin register()
 * closure to avoid duplicate SQLite connections.
 *
 * Provides:
 * - Batch-index un-indexed archive dates into SQLite-vec
 * - Prune archives older than retention period
 * - Report continuity health metrics
 * - Periodic re-indexing via startInterval()
 */

class MaintenanceService {
    /**
     * @param {object} config - full plugin config
     * @param {object} archiver - shared Archiver instance
     * @param {object} indexer - shared Indexer instance (already initialized)
     * @param {object} [summarizer] - optional Summarizer instance for queue processing + condensation
     */
    constructor(config = {}, archiver, indexer, summarizer = null, agentId = 'main') {
        this.config = config;
        this.archiver = archiver;
        this.indexer = indexer;
        this.summarizer = summarizer;
        this._agentId = agentId;
        this.batchDelay = config.archive?.batchIndexDelay || 100;

        this._lastRun = null;
        this._runCount = 0;
        this._interval = null;
    }

    /**
     * Execute one maintenance cycle.
     * @returns {object} health report
     */
    async execute() {
        this._runCount++;
        this._lastRun = new Date();

        const report = {
            timestamp: this._lastRun.toISOString(),
            runNumber: this._runCount,
            indexed: 0,
            pruned: 0,
            archiveStats: null,
            errors: []
        };

        // 1. Batch-index un-indexed dates
        try {
            const indexedDates = this.indexer.getIndexedDates();
            const unindexed = this.archiver.getUnindexedDates(indexedDates);

            for (const date of unindexed) {
                const conversation = this.archiver.getConversation(date);
                if (conversation && conversation.messages) {
                    const result = await this.indexer.indexDay(date, conversation.messages);
                    report.indexed += result.indexed;

                    if (this.batchDelay > 0) {
                        await _sleep(this.batchDelay);
                    }
                }
            }
        } catch (err) {
            report.errors.push(`Batch index: ${err.message}`);
        }

        // 2. Prune old archives
        try {
            const pruneResult = this.archiver.pruneOld();
            report.pruned = pruneResult.removed;
        } catch (err) {
            report.errors.push(`Prune: ${err.message}`);
        }

        // 3. Process summarization queue + condensation
        if (this.summarizer) {
            try {
                const queueResult = await this.summarizer.processQueue(3);
                report.summaryQueue = queueResult;
            } catch (err) {
                report.errors.push(`Summary queue: ${err.message}`);
            }

            try {
                // Condense accumulated leaf/branch summaries into higher-level summaries
                // Uses 'main' as default agentId — maintenance runs per-agent from index.js
                await this.summarizer.maybeCondense(this._agentId || 'main');
            } catch (err) {
                report.errors.push(`Summary condensation: ${err.message}`);
            }
        }

        // 4. Archive stats
        try {
            report.archiveStats = this.archiver.getStats();
        } catch (err) {
            report.errors.push(`Stats: ${err.message}`);
        }

        // 5. Log health
        const exchangeCount = this.indexer.getExchangeCount();
        if (report.errors.length === 0) {
            console.log(
                `[Continuity Maintenance] Run #${this._runCount} — ` +
                `indexed: ${report.indexed}, pruned: ${report.pruned}, ` +
                `total exchanges: ${exchangeCount}, ` +
                `archive sessions: ${report.archiveStats?.totalSessions || 0}`
            );
        } else {
            console.warn(
                `[Continuity Maintenance] Run #${this._runCount} — ` +
                `errors: ${report.errors.join('; ')}`
            );
        }

        return report;
    }

    /**
     * Start periodic re-indexing.
     * @param {number} ms - interval in milliseconds
     */
    startInterval(ms) {
        this.stopInterval();
        this._interval = setInterval(() => {
            this.execute().catch(err => {
                console.warn(`[Continuity Maintenance] Periodic run failed: ${err.message}`);
            });
        }, ms);
        // Don't let the interval keep the process alive
        if (this._interval.unref) this._interval.unref();
    }

    /**
     * Stop periodic re-indexing.
     */
    stopInterval() {
        if (this._interval) {
            clearInterval(this._interval);
            this._interval = null;
        }
    }

    /**
     * Get service status for gateway inspection.
     * @returns {object}
     */
    getStatus() {
        return {
            lastRun: this._lastRun?.toISOString() || null,
            runCount: this._runCount,
            exchangeCount: this.indexer ? this.indexer.getExchangeCount() : 0,
            intervalActive: !!this._interval
        };
    }
}

function _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = MaintenanceService;
