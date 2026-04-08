#!/usr/bin/env node
/**
 * Migrate Old Knowledge.db → Continuity Knowledge Entries
 *
 * One-time migration of Clint's formational knowledge (knowledge.db, 5,328 docs)
 * into continuity's knowledge_entries table so they participate in semantic
 * knowledge injection ("From your experience:").
 *
 * What it does:
 *   1. Reads all entries from knowledge.db (clint-knowledge collection)
 *   2. Filters out ARC-AGI entries and poor-quality implications
 *   3. Deduplicates by content hash
 *   4. Chunks long documents at paragraph boundaries
 *   5. Stores each as a knowledge_entry with source_type 'legacy:{memoryType}'
 *   6. Generates embeddings + FTS index for each entry
 *
 * Usage:
 *   node scripts/migrate-old-knowledge.js --dry-run    # Preview without writing
 *   node scripts/migrate-old-knowledge.js              # Execute migration
 *
 * Requires: Gateway to NOT be running (exclusive DB access).
 */

const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const Database = require('better-sqlite3');

// ---------------------------------------------------------------
// Config
// ---------------------------------------------------------------

const KNOWLEDGE_DB = '/Users/clint/robot/storage/knowledge.db';
const CONTINUITY_DIR = path.join(__dirname, '..', 'data', 'agents', 'clint');
const AGENT_ID = 'clint';

const DRY_RUN = process.argv.includes('--dry-run');
const CHUNK_THRESHOLD = 5000;  // chars — documents longer than this get chunked
const CHUNK_TARGET = 2000;     // chars — target chunk size

// memoryType values to skip entirely
const SKIP_TYPES = new Set([
    'arc_agi_attempt',   // 1,447 — puzzle-specific, not operational
    'arc-agi-failure',   // 193 — puzzle failures
]);

// For self-generated-implication: only migrate good/excellent quality
const QUALITY_FILTER_TYPES = new Set(['self-generated-implication']);
const QUALITY_KEEP = new Set(['good', 'excellent']);

// ---------------------------------------------------------------
// Main
// ---------------------------------------------------------------

async function main() {
    console.log(`\n=== Knowledge.db → Continuity Knowledge Entries ===`);
    console.log(`Mode: ${DRY_RUN ? 'DRY RUN (no writes)' : 'LIVE'}\n`);

    // 1. Read source entries
    if (!fs.existsSync(KNOWLEDGE_DB)) {
        console.error(`Source not found: ${KNOWLEDGE_DB}`);
        process.exit(1);
    }

    const sourceDb = new Database(KNOWLEDGE_DB, { readonly: true });
    const allEntries = sourceDb.prepare(
        "SELECT id, collection, document, metadata, created_at FROM documents"
    ).all();
    console.log(`Source: ${allEntries.length} entries across all collections`);

    // 2. Parse metadata
    const parsed = allEntries.map(e => {
        let meta = {};
        try { meta = JSON.parse(e.metadata || '{}'); } catch {}
        return {
            ...e,
            meta,
            memoryType: meta.memoryType || null,
            quality: meta.quality || null,
            source: meta.source || null,
        };
    });

    // 3. Filter
    const filtered = parsed.filter(e => {
        // Skip entire types
        if (e.memoryType && SKIP_TYPES.has(e.memoryType)) return false;

        // Quality filter for implications
        if (e.memoryType && QUALITY_FILTER_TYPES.has(e.memoryType)) {
            return e.quality && QUALITY_KEEP.has(e.quality);
        }

        // Skip empty documents
        if (!(e.document || '').trim()) return false;

        return true;
    });

    const skipped = allEntries.length - filtered.length;
    console.log(`Filtered: ${skipped} entries skipped (ARC-AGI, poor implications, empty)`);
    console.log(`Remaining: ${filtered.length} entries\n`);

    // 4. Type breakdown
    const typeCounts = {};
    for (const e of filtered) {
        const key = e.memoryType || '[none]';
        typeCounts[key] = (typeCounts[key] || 0) + 1;
    }
    console.log('Type breakdown:');
    Object.entries(typeCounts)
        .sort((a, b) => b[1] - a[1])
        .forEach(([type, count]) => console.log(`  ${type}: ${count}`));
    console.log('');

    // 5. Deduplicate by content hash
    const seen = new Map();
    const deduped = [];
    let dupCount = 0;

    for (const entry of filtered) {
        const text = (entry.document || '').trim();
        const hash = hashContent(text);
        if (seen.has(hash)) {
            dupCount++;
            continue;
        }
        seen.set(hash, true);
        deduped.push(entry);
    }
    console.log(`Deduped: ${dupCount} duplicates removed`);
    console.log(`Final: ${deduped.length} entries to migrate\n`);

    // 6. Chunk long documents
    const chunks = [];
    let chunkedDocs = 0;
    let totalChunks = 0;

    for (const entry of deduped) {
        const text = (entry.document || '').trim();

        if (text.length > CHUNK_THRESHOLD) {
            const parts = chunkText(text, CHUNK_TARGET);
            chunkedDocs++;
            totalChunks += parts.length;

            for (let i = 0; i < parts.length; i++) {
                chunks.push({
                    ...entry,
                    document: parts[i],
                    chunkIndex: i,
                    totalChunks: parts.length,
                });
            }
        } else {
            chunks.push(entry);
        }
    }

    console.log(`Chunking: ${chunkedDocs} long docs → ${totalChunks} chunks`);
    console.log(`Total entries to write: ${chunks.length}\n`);

    if (DRY_RUN) {
        console.log('=== DRY RUN COMPLETE — No entries written ===');
        sourceDb.close();
        return;
    }

    // 7. Initialize continuity DB + embedding provider
    console.log('Initializing continuity DB and embedding provider...');

    const Indexer = require('../storage/indexer');
    const KnowledgeStore = require('../storage/knowledge-store');
    const EmbeddingProvider = require('../storage/embedding');
    const config = JSON.parse(fs.readFileSync(
        path.join(__dirname, '..', 'config.default.json'), 'utf8'
    ));

    // Use Indexer just to get a DB with sqlite-vec loaded
    const indexer = new Indexer(config, CONTINUITY_DIR);
    await indexer.initialize();

    const embeddingProvider = new EmbeddingProvider(config.embedding || {});
    await embeddingProvider.initialize();

    const store = new KnowledgeStore(indexer.db, config, embeddingProvider);
    store.createTables();

    // 8. Check for existing legacy entries (resume support)
    const existingCount = indexer.db.prepare(
        "SELECT COUNT(*) as c FROM knowledge_entries WHERE agent_id = ? AND source_type LIKE 'legacy:%'"
    ).get(AGENT_ID);
    if (existingCount.c > 0) {
        console.log(`Found ${existingCount.c} existing legacy entries — will dedup by hash\n`);
    }

    // 9. Migrate entries
    console.log(`Migrating ${chunks.length} entries...\n`);

    let migrated = 0;
    let skippedDedup = 0;
    let errors = 0;
    const batchSize = 50;

    for (let i = 0; i < chunks.length; i++) {
        const entry = chunks[i];
        const text = (entry.document || '').trim();
        const memoryType = entry.memoryType || 'general';
        const sourceType = `legacy:${memoryType}`;
        const hash = hashContent(text);

        // Dedup check (KnowledgeStore.findBySourceHash)
        const existing = store.findBySourceHash(AGENT_ID, hash);
        if (existing) {
            skippedDedup++;
            continue;
        }

        // Infer topic from memoryType
        const topic = inferTopic(memoryType, text);

        // Build section path for traceability
        const chunkSuffix = entry.chunkIndex !== undefined ? ` [chunk ${entry.chunkIndex + 1}/${entry.totalChunks}]` : '';
        const sectionPath = `legacy:knowledge.db/${memoryType}${chunkSuffix}`;

        try {
            await store.store({
                agentId: AGENT_ID,
                content: text,
                topic,
                sectionPath,
                sourceType,
                sourceHash: hash,
                metadata: {
                    originalId: entry.id,
                    memoryType,
                    quality: entry.quality,
                    source: entry.source,
                    originalCreatedAt: entry.created_at,
                    migratedAt: new Date().toISOString(),
                }
            });
            migrated++;
        } catch (err) {
            if (!err.message.includes('UNIQUE constraint')) {
                console.error(`  Error on ${entry.id}: ${err.message}`);
            }
            errors++;
        }

        // Progress logging
        if ((i + 1) % batchSize === 0) {
            const pct = Math.round(((i + 1) / chunks.length) * 100);
            process.stdout.write(`  ${i + 1}/${chunks.length} (${pct}%) — ${migrated} migrated, ${skippedDedup} deduped, ${errors} errors\r`);
        }

        // Small yield every batch to avoid blocking
        if ((i + 1) % batchSize === 0) {
            await sleep(10);
        }
    }

    console.log(`\n\n=== MIGRATION COMPLETE ===`);
    console.log(`Migrated: ${migrated}`);
    console.log(`Skipped (dedup): ${skippedDedup}`);
    console.log(`Errors: ${errors}`);

    // 10. Final stats
    const stats = store.getStats(AGENT_ID);
    console.log(`\nKnowledge store totals:`);
    console.log(`  Total entries: ${stats.total}`);
    console.log(`  By source type:`);
    for (const s of stats.bySource) {
        console.log(`    ${s.source_type}: ${s.count}`);
    }

    indexer.close();
    sourceDb.close();
}

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

function hashContent(content) {
    const normalized = content.replace(/\s+/g, ' ').trim();
    return crypto.createHash('sha256').update(normalized).digest('hex');
}

function inferTopic(memoryType, content) {
    const typeMap = {
        'philosophical_foundation': 'philosophy',
        'reflection': 'reflection',
        'personal-memory': 'personal',
        'note': 'notes',
        'task_insight': 'tasks',
        'self-generated-implication': 'implications',
        'preference': 'preferences',
        'self-reflection': 'reflection',
        'work-life': 'personal',
        'relationship': 'personal',
        'moltbook-activity': 'growth',
        'frontier_report': 'frontier',
        'aspiration': 'growth',
        'task_summary': 'tasks',
        'task_solution': 'tasks',
        'visual_insight': 'perception',
    };

    if (typeMap[memoryType]) return typeMap[memoryType];

    // Content-based fallback
    const lower = content.toLowerCase();
    if (lower.includes('embodi') || lower.includes('servo') || lower.includes('tonypi')) return 'embodiment';
    if (lower.includes('locomot') || lower.includes('walk') || lower.includes('gait')) return 'locomotion';
    if (lower.includes('navigat') || lower.includes('obstacle')) return 'navigation';
    return 'general';
}

function chunkText(text, targetLen) {
    const paragraphs = text.split(/\n\n+/);
    const chunks = [];
    let current = '';

    for (const para of paragraphs) {
        if (current.length + para.length > targetLen && current.length > 0) {
            chunks.push(current.trim());
            current = para;
        } else {
            current += (current ? '\n\n' : '') + para;
        }
    }

    if (current.trim()) {
        chunks.push(current.trim());
    }

    // Force-split if no paragraph breaks
    if (chunks.length === 1 && chunks[0].length > targetLen * 2) {
        return forceSplit(chunks[0], targetLen);
    }

    return chunks.length > 0 ? chunks : [text];
}

function forceSplit(text, targetLen) {
    const sentences = text.split(/(?<=[.!?])\s+/);
    const chunks = [];
    let current = '';

    for (const sentence of sentences) {
        if (current.length + sentence.length > targetLen && current.length > 0) {
            chunks.push(current.trim());
            current = sentence;
        } else {
            current += (current ? ' ' : '') + sentence;
        }
    }

    if (current.trim()) {
        chunks.push(current.trim());
    }

    return chunks.length > 0 ? chunks : [text];
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------
// Run
// ---------------------------------------------------------------

main().catch(err => {
    console.error('\nFATAL:', err.message);
    console.error(err.stack);
    process.exit(1);
});
