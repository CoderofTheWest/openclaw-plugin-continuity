#!/usr/bin/env node
/**
 * Run one maintenance cycle — backfills unindexed archive dates.
 *
 * Usage:
 *   node scripts/run-maintenance.js
 *
 * Requires: Gateway to NOT be running (exclusive DB access).
 */

const path = require('path');
const fs = require('fs');

const CONTINUITY_DIR = path.join(__dirname, '..', 'data', 'agents', 'clint');
const ARCHIVE_DIR = path.join(CONTINUITY_DIR, 'archive');
const AGENT_ID = 'clint';

async function main() {
    console.log('\n=== Manual Maintenance Cycle ===\n');

    const Indexer = require('../storage/indexer');
    const Archiver = require('../storage/archiver');
    const MaintenanceService = require('../services/maintenance');
    const config = JSON.parse(fs.readFileSync(
        path.join(__dirname, '..', 'config.default.json'), 'utf8'
    ));

    // Override batch delay for faster backfill
    config.archive = config.archive || {};
    config.archive.batchIndexDelay = 50;

    const indexer = new Indexer(config, CONTINUITY_DIR);
    await indexer.initialize();

    const archiver = new Archiver(config, CONTINUITY_DIR);

    const maintenance = new MaintenanceService(config, archiver, indexer, null, AGENT_ID);

    const indexedBefore = indexer.getExchangeCount();
    console.log(`Exchanges before: ${indexedBefore}`);

    const report = await maintenance.execute();

    const indexedAfter = indexer.getExchangeCount();
    console.log(`Exchanges after: ${indexedAfter}`);
    console.log(`New exchanges indexed: ${report.indexed}`);
    console.log(`Archives pruned: ${report.pruned}`);

    if (report.errors.length > 0) {
        console.log(`Errors: ${report.errors.join(', ')}`);
    }

    if (report.archiveStats) {
        console.log(`\nArchive stats: ${report.archiveStats.totalDates} dates, ${report.archiveStats.dateRange?.first} → ${report.archiveStats.dateRange?.last}`);
    }

    indexer.close();
    console.log('\n=== Done ===');
}

main().catch(err => {
    console.error('\nFATAL:', err.message);
    console.error(err.stack);
    process.exit(1);
});
