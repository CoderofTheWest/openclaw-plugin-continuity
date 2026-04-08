#!/usr/bin/env node
/**
 * Bootstrap DAG Summaries from Historical Exchanges
 *
 * Generates Level 0 (leaf) extractive summaries from all exchange dates
 * that don't yet have summaries. Then runs maybeCondense() to build
 * Level 1 (branch) summaries where enough leaves exist.
 *
 * No LLM required — extractiveSummary() is purely algorithmic.
 * LLM-based condensation (Level 1+) only fires if enough leaves accumulate.
 *
 * Usage:
 *   node scripts/bootstrap-summaries.js --dry-run    # Preview without writing
 *   node scripts/bootstrap-summaries.js              # Execute bootstrap
 *
 * Requires: Gateway to NOT be running (exclusive DB access).
 */

const path = require('path');
const fs = require('fs');

// ---------------------------------------------------------------
// Config
// ---------------------------------------------------------------

const CONTINUITY_DIR = path.join(__dirname, '..', 'data', 'agents', 'clint');
const AGENT_ID = 'clint';
const DRY_RUN = process.argv.includes('--dry-run');
const MIN_EXCHANGES_PER_SUMMARY = 3; // skip dates with very few exchanges

// ---------------------------------------------------------------
// Main
// ---------------------------------------------------------------

async function main() {
    console.log(`\n=== Bootstrap DAG Summaries ===`);
    console.log(`Mode: ${DRY_RUN ? 'DRY RUN (no writes)' : 'LIVE'}\n`);

    // 1. Initialize DB + components
    const Indexer = require('../storage/indexer');
    const SummaryStore = require('../storage/summary-store');
    const Summarizer = require('../lib/summarizer');
    const EmbeddingProvider = require('../storage/embedding');
    const config = JSON.parse(fs.readFileSync(
        path.join(__dirname, '..', 'config.default.json'), 'utf8'
    ));

    const indexer = new Indexer(config, CONTINUITY_DIR);
    await indexer.initialize();

    const embeddingProvider = new EmbeddingProvider(config.embedding || {});
    await embeddingProvider.initialize();

    const summaryStore = new SummaryStore(indexer.db, config.summaries || {}, embeddingProvider);
    summaryStore.createTables();

    const summarizer = new Summarizer(config.summaries || {}, summaryStore);

    // 2. Get all unique dates with exchanges
    const dates = indexer.db.prepare(
        `SELECT DISTINCT date FROM exchanges ORDER BY date ASC`
    ).all().map(r => r.date);
    console.log(`Total dates with exchanges: ${dates.length}`);

    // 3. Get dates that already have Level 0 summaries
    const existingSummaryDates = new Set();
    const existingSummaries = indexer.db.prepare(
        `SELECT date_range_start FROM summaries WHERE agent_id = ? AND level = 0`
    ).all(AGENT_ID);
    for (const s of existingSummaries) {
        existingSummaryDates.add(s.date_range_start);
    }
    console.log(`Dates with existing summaries: ${existingSummaryDates.size}`);

    const datesToProcess = dates.filter(d => !existingSummaryDates.has(d));
    console.log(`Dates to bootstrap: ${datesToProcess.length}\n`);

    if (datesToProcess.length === 0) {
        console.log('Nothing to do — all dates already have summaries.');
        indexer.close();
        return;
    }

    // 4. Generate Level 0 summaries per date
    let created = 0;
    let skipped = 0;
    let errors = 0;

    for (const date of datesToProcess) {
        // Get exchanges for this date
        const exchanges = indexer.db.prepare(
            `SELECT combined FROM exchanges WHERE date = ? ORDER BY exchange_index ASC`
        ).all(date);

        if (exchanges.length < MIN_EXCHANGES_PER_SUMMARY) {
            skipped++;
            continue;
        }

        // Convert to message format for extractiveSummary
        const messages = exchanges.map(e => e.combined);

        try {
            // Generate extractive summary (no LLM, purely algorithmic)
            const result = summarizer.extractiveSummary(messages, [], []);

            if (!result.text || result.text.trim().length < 20) {
                skipped++;
                continue;
            }

            const summaryId = `summary_${date}_bootstrap`;

            if (DRY_RUN) {
                console.log(`  ${date}: ${exchanges.length} exchanges → ${result.text.length} char summary (${result.topics.length} topics)`);
                created++;
                continue;
            }

            await summaryStore.storeSummary({
                id: summaryId,
                level: 0,
                parentId: null,
                agentId: AGENT_ID,
                dateRangeStart: date,
                dateRangeEnd: date,
                messageCount: exchanges.length,
                summaryText: result.text,
                topics: result.topics,
                anchors: result.anchors,
                entropyAvg: 0,
                metadata: {
                    strategy: 'extractive',
                    source: 'bootstrap',
                    createdAt: new Date().toISOString(),
                }
            });

            created++;

            if (created % 20 === 0) {
                process.stdout.write(`  ${created} summaries created...\r`);
            }
        } catch (err) {
            console.error(`  ${date}: ERROR — ${err.message}`);
            errors++;
        }
    }

    console.log(`\nLevel 0 summaries: ${created} created, ${skipped} skipped, ${errors} errors`);

    if (DRY_RUN) {
        console.log('\n=== DRY RUN COMPLETE — No summaries written ===');
        indexer.close();
        return;
    }

    // 5. Run condensation to build Level 1+ branches
    console.log('\nRunning condensation (Level 1+ branches)...');
    console.log('(Condensation requires enough leaves — threshold is ~6 per branch)');

    try {
        // Run maybeCondense which checks for enough uncondensed leaves/branches
        const condenseResult = await summarizer.maybeCondense(AGENT_ID);
        if (condenseResult) {
            console.log(`Condensation result: ${JSON.stringify(condenseResult)}`);
        } else {
            console.log('No condensation needed yet (not enough uncondensed leaves).');
        }
    } catch (err) {
        console.warn(`Condensation skipped: ${err.message}`);
        console.log('(This is expected if no LLM provider is configured for abstractive summaries)');
    }

    // 6. Final stats
    const stats = indexer.db.prepare(
        `SELECT level, count(*) as count FROM summaries WHERE agent_id = ? GROUP BY level`
    ).all(AGENT_ID);

    console.log('\n=== BOOTSTRAP COMPLETE ===');
    console.log('Summary DAG:');
    for (const s of stats) {
        const label = s.level === 0 ? 'leaves' : s.level === 1 ? 'branches' : 'roots';
        console.log(`  Level ${s.level} (${label}): ${s.count}`);
    }

    indexer.close();
}

// ---------------------------------------------------------------
// Run
// ---------------------------------------------------------------

main().catch(err => {
    console.error('\nFATAL:', err.message);
    console.error(err.stack);
    process.exit(1);
});
