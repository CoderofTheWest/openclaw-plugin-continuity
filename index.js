/**
 * openclaw-plugin-continuity — "Infinite Thread"
 *
 * Persistent, intelligent memory for OpenClaw agents.
 * Ported from Clint's production architecture (Oct 2025 - Feb 2026).
 *
 * Provides:
 * - Context budgeting with priority tiers (ESSENTIAL → MINIMAL)
 * - Continuity anchor detection (identity, contradiction, tension)
 * - Topic freshness tracking and fixation detection
 * - Threshold-triggered context compaction
 * - Daily conversation archiving with deduplication
 * - Cross-session semantic search via SQLite-vec
 * - MEMORY.md ## Continuity section braiding
 *
 * Requires: SQLite-vec (better-sqlite3 + sqlite-vec extension)
 * Model-agnostic: accepts custom tokenizer functions
 *
 * Hook registration uses api.on() (OpenClaw SDK typed hooks).
 * Continuity context injected via prependContext (before identity kernel).
 *
 * Multi-agent: All state (archives, indexes, session tracking) is scoped
 * per agent via ctx.agentId. Each agent gets its own data subdirectory
 * under data/agents/{agentId}/. Agents never see each other's memories.
 * The default/main agent uses the legacy data/ path for backward compat.
 */

const path = require('path');
const fs = require('fs');

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

function loadConfig(userConfig = {}) {
    const defaultConfig = JSON.parse(
        fs.readFileSync(path.join(__dirname, 'config.default.json'), 'utf8')
    );
    return deepMerge(defaultConfig, userConfig);
}

function deepMerge(target, source) {
    const result = { ...target };
    for (const key of Object.keys(source)) {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
            result[key] = deepMerge(target[key] || {}, source[key]);
        } else {
            result[key] = source[key];
        }
    }
    return result;
}

function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
    return dirPath;
}

// ---------------------------------------------------------------------------
// Plugin export
// ---------------------------------------------------------------------------

module.exports = {
    id: 'continuity',
    name: 'Infinite Thread — Agent Continuity & Memory',

    configSchema: {
        jsonSchema: {
            type: 'object',
            properties: {
                contextBudget: { type: 'object' },
                anchors: { type: 'object' },
                topicTracking: { type: 'object' },
                compaction: { type: 'object' },
                tokenEstimation: { type: 'object' },
                archive: { type: 'object' },
                embedding: { type: 'object' },
                session: { type: 'object' },
                continuitySection: { type: 'object' }
            }
        }
    },

    register(api) {
        const config = loadConfig(api.pluginConfig || {});

        // Base data directory for the plugin
        const baseDataDir = ensureDir(path.join(__dirname, 'data'));

        // -------------------------------------------------------------------
        // Per-agent state management
        //
        // Each agent gets its own isolated set of:
        //   - Archiver (daily conversation files)
        //   - Indexer + Searcher (SQLite-vec embedding DB)
        //   - TopicTracker, ContinuityAnchors (session-level state)
        //   - Session counters (exchangeCount, sessionStart)
        //   - Retrieval cache
        //
        // Data directory layout:
        //   data/                    <- default/main agent (backward compat)
        //   data/agents/{agentId}/   <- all other agents
        // -------------------------------------------------------------------

        const TopicTracker = require('./lib/topic-tracker');
        const ContinuityAnchors = require('./lib/continuity-anchors');
        const TokenEstimator = require('./lib/token-estimator');
        const Archiver = require('./storage/archiver');
        const Indexer = require('./storage/indexer');
        const Searcher = require('./storage/searcher');
        const EmbeddingProvider = require('./storage/embedding');
        const SummaryStore = require('./storage/summary-store');
        const Summarizer = require('./lib/summarizer');
        const KnowledgeStore = require('./storage/knowledge-store');
        const KnowledgeIndexer = require('./lib/knowledge-indexer');
        const Compactor = require('./lib/compactor');

        // Shared across agents (stateless utility)
        const tokenEstimator = new TokenEstimator(config.tokenEstimation || {});

        // Continuity indicators (from config)
        const continuityIndicators = config.continuityIndicators || [];

        /**
         * Per-agent state container.
         * Created lazily on first hook invocation for each agent.
         */
        class AgentState {
            constructor(agentId) {
                this.agentId = agentId;

                // Data directory: legacy path for default/main, scoped for others
                if (!agentId || agentId === 'main') {
                    this.dataDir = baseDataDir;
                } else {
                    this.dataDir = ensureDir(path.join(baseDataDir, 'agents', agentId));
                }
                ensureDir(path.join(this.dataDir, config.archive.archiveDir || 'archive'));

                // Per-agent module instances
                this.topicTracker = new TopicTracker(config);
                this.anchors = new ContinuityAnchors(config);
                this.compactor = new Compactor(config, null, this.anchors, tokenEstimator);
                this.archiver = new Archiver(config, this.dataDir);

                // Storage (lazy init — embedding model is expensive)
                this.embeddingProvider = null;
                this.indexer = null;
                this.searcher = null;
                this.summaryStore = null;
                this.summarizer = null;
                this.knowledgeStore = null;
                this.knowledgeIndexer = null;
                this.knowledgeIndexedOnce = false;
                this.storageReady = false;
                this.storageInitPromise = null;

                // Session state
                this.sessionStart = Date.now();
                this.sessionId = null;  // Set on session_start
                this.exchangeCount = 0;
                this.compactionCount = 0;
                this.handoffWritten = false;

                // Retrieval cache (per-agent, per-turn)
                this.lastRetrievalCache = null;
            }

            async ensureStorage() {
                if (this.storageReady) return;
                if (this.storageInitPromise) {
                    await this.storageInitPromise;
                    return;
                }
                this.storageInitPromise = (async () => {
                    try {
                        // Single shared embedding provider per agent —
                        // pipeline created once, tensors disposed after each use.
                        // Fixes memory leak from duplicate pipelines + undisposed tensors.
                        this.embeddingProvider = new EmbeddingProvider(config.embedding || {});
                        await this.embeddingProvider.initialize();

                        this.indexer = new Indexer(config, this.dataDir, this.embeddingProvider);
                        await this.indexer.initialize();
                        this.searcher = new Searcher(config, this.dataDir, this.indexer.db, this.embeddingProvider);
                        // Searcher skips its own init when provider is injected

                        // Summary DAG + summarizer (LCM-inspired)
                        if (config.summarization?.enabled !== false) {
                            this.summaryStore = new SummaryStore(this.indexer.db, config, this.embeddingProvider);
                            this.summaryStore.createTables();
                            this.summarizer = new Summarizer(config, this.summaryStore, this.embeddingProvider);
                        }

                        // Knowledge index (operational knowledge from workspace files)
                        if (config.knowledge?.enabled !== false) {
                            this.knowledgeStore = new KnowledgeStore(this.indexer.db, config, this.embeddingProvider);
                            this.knowledgeStore.createTables();
                            // KnowledgeIndexer created here with null workspace —
                            // actual workspace path resolved from event.metadata in session_start
                            this.knowledgeIndexer = new KnowledgeIndexer(
                                this.knowledgeStore, config, this.embeddingProvider, null
                            );
                        }

                        this.storageReady = true;
                        api.logger.info(`[Continuity] Storage ready for agent "${this.agentId}" at ${this.dataDir} (shared embedding provider)`);
                    } catch (err) {
                        api.logger.error(`[Continuity] Storage init failed for agent "${this.agentId}": ${err.message}`);
                        this.embeddingProvider = null;
                        this.indexer = null;
                        this.searcher = null;
                    }
                })();
                await this.storageInitPromise;
                this.storageInitPromise = null;
            }

            /**
             * Release all resources held by this agent state.
             * Called on session_end to prevent unbounded memory growth.
             */
            close() {
                if (this.embeddingProvider) {
                    this.embeddingProvider.dispose();
                    this.embeddingProvider = null;
                }
                if (this.indexer) {
                    this.indexer.close();
                    this.indexer = null;
                }
                this.searcher = null;
                this.summaryStore = null;
                this.summarizer = null;
                this.knowledgeStore = null;
                this.knowledgeIndexer = null;
                this.lastRetrievalCache = null;
                this.storageReady = false;
                this.storageInitPromise = null;
                api.logger.info(`[Continuity] Resources released for agent "${this.agentId}"`);
            }
        }

        /** @type {Map<string, AgentState>} */
        const agentStates = new Map();

        /**
         * Get or create per-agent state.
         * @param {string} [agentId] - Agent ID from hook context
         * @returns {AgentState}
         */
        function getAgentState(agentId) {
            const id = agentId || 'main';
            if (!agentStates.has(id)) {
                agentStates.set(id, new AgentState(id));
                api.logger.info(`Initialized continuity state for agent "${id}"`);
            }
            return agentStates.get(id);
        }

        // Track current agent for tool context — tools don't receive agentId
        // from the gateway, so we capture it from the hook context.
        // Safe because OpenClaw serializes execution per session lane.
        let currentAgentId = 'main';
        function getCurrentAgentId() { return currentAgentId; }

        // -------------------------------------------------------------------
        // Global bus: cross-plugin knowledge indexing + search
        // Used by contemplation plugin to index insights into vec_knowledge,
        // and by anticipator to search insights for proactive surfacing.
        // -------------------------------------------------------------------

        if (!global.__ocContinuity) {
            global.__ocContinuity = {};
        }

        /**
         * Index an insight (or any knowledge entry) into vec_knowledge.
         * Called by contemplation writer after persisting completed insights.
         */
        global.__ocContinuity.indexInsight = async (agentId, entry) => {
            try {
                const state = getAgentState(agentId);
                if (!state.storageReady) {
                    if (state.storageInitPromise) await state.storageInitPromise;
                    if (!state.storageReady) return null;
                }
                if (!state.knowledgeStore) return null;
                const id = await state.knowledgeStore.store({
                    agentId,
                    content: `${entry.topic}\n\n${entry.content}`,
                    topic: entry.topic,
                    sectionPath: entry.tags ? entry.tags.join('/') : null,
                    sourceType: entry.source || 'contemplation',
                    sourceHash: entry.source || null,
                    metadata: { tags: entry.tags, indexed_at: new Date().toISOString() }
                });
                api.logger.info(`[Continuity] Indexed insight for ${agentId}: ${id} (topic: ${(entry.topic || '').substring(0, 60)})`);
                return id;
            } catch (err) {
                api.logger.error(`[Continuity] Failed to index insight: ${err.message}`);
                return null;
            }
        };

        /**
         * Search vec_knowledge for insights matching a query.
         * Used by anticipator for proactive insight surfacing.
         */
        global.__ocContinuity.searchKnowledge = async (agentId, query, limit = 3) => {
            try {
                const state = getAgentState(agentId);
                if (!state.storageReady || !state.knowledgeStore || !state.embeddingProvider) return [];
                const queryEmbedding = await state.embeddingProvider.embed(query);
                return await state.knowledgeStore.search(agentId, query, queryEmbedding, limit);
            } catch (err) {
                api.logger.error(`[Continuity] Knowledge search failed: ${err.message}`);
                return [];
            }
        };

        /**
         * Full hybrid search (4-way RRF) across all memory types.
         * Used by continuity_search tool.
         */
        global.__ocContinuity.search = async (agentId, query, limit = 5) => {
            try {
                const state = getAgentState(agentId);
                if (!state.storageReady || !state.searcher) return [];
                return await state.searcher.search(query, limit);
            } catch (err) {
                api.logger.error(`[Continuity] Full search failed: ${err.message}`);
                return [];
            }
        };

        // -------------------------------------------------------------------
        // HOOK: before_agent_start — Inject continuity context via prependContext
        // Priority 10 (runs after stability plugin if both present)
        // -------------------------------------------------------------------

        api.on('before_agent_start', async (event, ctx) => {
          try {
            currentAgentId = ctx.agentId || 'main';
            const state = getAgentState(ctx.agentId);

            // Write handoff BEFORE exchange starts so crashes don't lose state
            // This ensures we have a handoff even if the exchange fails mid-stream
            _writeSessionHandoff(state, config, ctx, api);

            state.exchangeCount++;

            // Extract last user message from the event messages array
            const messages = event.messages || [];
            const lastUser = [...messages].reverse().find(m =>
                m?.role === 'user'
            );
            let lastUserText = _extractText(lastUser);

            // Fallback: some delivery paths (CLI, webhook) put the user text
            // in event.userMessage, event.text, or event.prompt
            if (!lastUserText && event.userMessage) {
                lastUserText = typeof event.userMessage === 'string'
                    ? event.userMessage
                    : _extractText(event.userMessage);
            }
            if (!lastUserText && event.text) {
                lastUserText = event.text;
            }
            // OpenClaw before_agent_start passes the full prompt —
            // extract the last user segment from it
            if (!lastUserText && event.prompt) {
                const promptStr = typeof event.prompt === 'string'
                    ? event.prompt
                    : Array.isArray(event.prompt)
                        ? event.prompt.filter(p => p.role === 'user').pop()?.content || ''
                        : '';
                if (typeof promptStr === 'string' && promptStr.length > 0) {
                    // Find the last timestamp marker — user text follows it
                    // e.g. [Mon 2026-03-09 20:12 PDT] actual message here
                    const tsRegex = /\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s\d{4}-\d{2}-\d{2}\s[^\]]*\]\s*/g;
                    let lastTs = null;
                    let m;
                    while ((m = tsRegex.exec(promptStr)) !== null) lastTs = m;
                    if (lastTs) {
                        lastUserText = promptStr.substring(lastTs.index + lastTs[0].length).trim();
                    } else {
                        // No timestamp — take the last meaningful chunk
                        const trimmed = promptStr.slice(-2000).trim();
                        if (trimmed.length >= 10) {
                            lastUserText = trimmed;
                        }
                    }
                }
            }

            // Build continuity context block
            const lines = ['[CONTINUITY CONTEXT]'];

            // Session info
            const sessionAge = _formatDuration(Date.now() - state.sessionStart);
            lines.push(`Session: ${state.exchangeCount} exchanges | Started: ${sessionAge}`);

            // Sustained engagement wellbeing awareness
            {
                const nowMs = Date.now();
                const gap = state._lastUserMessageTime
                    ? (nowMs - state._lastUserMessageTime) / 60000 : Infinity;
                if (gap < 5) {
                    if (!state._sustainedWorkStart) state._sustainedWorkStart = nowMs;
                } else if (gap > 30) {
                    state._sustainedWorkStart = null;
                }
                state._lastUserMessageTime = nowMs;
                const sustainedMinutes = state._sustainedWorkStart
                    ? Math.floor((nowMs - state._sustainedWorkStart) / 60000) : 0;
                const wellbeingNotes = [];
                if (sustainedMinutes > 120) {
                    wellbeingNotes.push(`Sustained work: ${Math.floor(sustainedMinutes / 60)}+ hours without a significant break.`);
                } else if (sustainedMinutes > 90) {
                    wellbeingNotes.push('Over 90 minutes of sustained work.');
                }
                const hour = new Date().getHours();
                if (hour >= 11 && hour <= 13) wellbeingNotes.push('Around lunchtime.');
                else if (hour >= 17 && hour <= 19) wellbeingNotes.push('Around dinner time.');
                else if (hour >= 22 || hour < 5) wellbeingNotes.push("It's late.");
                if (wellbeingNotes.length > 0) {
                    lines.push(`Wellbeing: ${wellbeingNotes.join(' ')}`);
                }
            }

            // Session handoff injection — first exchange of new session only.
            // Content was loaded from SESSION_HANDOFF.md in session_start.
            if (state.pendingHandoff && state.exchangeCount <= 1) {
                lines.push('');
                lines.push('[SESSION HANDOFF — from previous session]');
                lines.push(state.pendingHandoff);
                lines.push('[/SESSION HANDOFF]');
                lines.push('');
                state.pendingHandoff = null;  // One-shot: only inject once
                api.logger.info(`[Continuity:${state.agentId}] Session handoff injected into context`);
            }

            // Nightshift report injection — first exchange only.
            // Content was loaded from NIGHTSHIFT_REPORT.md in session_start.
            if (state.pendingNightReport && state.exchangeCount <= 1) {
                lines.push('');
                lines.push('[NIGHTSHIFT REPORT — what happened while you were offline]');
                lines.push(state.pendingNightReport);
                lines.push('[/NIGHTSHIFT REPORT]');
                lines.push('');
                state.pendingNightReport = null;  // One-shot: only inject once
                api.logger.info(`[Continuity:${state.agentId}] Nightshift report injected into context`);
            }

            // Topic tracking is NO LONGER injected into context.
            // The data is still collected and available via:
            //   - continuity.getTopics gateway method (debugging/dashboards)
            //   - before_compaction logging (diagnostic)
            // This removes the noise of "Topics: session (fixated — 15 mentions)"
            // which wasn't actionable for the agent.

            // Continuity anchors — only inject when entropy is elevated
            // (identity/contradiction/tension anchors are most valuable during
            // high-entropy exchanges, not calm nominal conversation)
            const currentEntropy = api.stability?.getEntropy?.(ctx.agentId) || 0;
            if (currentEntropy > 0.4) {
                const activeAnchors = state.anchors.getAnchors();
                if (activeAnchors.length > 0) {
                    const anchorStrs = activeAnchors.slice(0, 3).map(a => {
                        const age = _formatAge(a.timestamp);
                        return `${a.type.toUpperCase()}: "${_truncate(a.text, 80)}" (${age})`;
                    });
                    lines.push(`Anchors: ${anchorStrs.join(' | ')}`);
                }
            }

            // Topic fixation notes removed — not actionable for the agent.
            // Topic tracking continues silently; available via gateway method.

            // Archive retrieval — always search, relevance-gate the injection.
            //
            // prependContext is the authoritative path for recalled memories.
            // Tool result enrichment (tool_result_persist) is secondary reinforcement.
            // Clint's principle: "Context carries authority; tool results don't."
            //
            // Intent detection controls injection verbosity, not search gating:
            //   - Explicit recall intent → always inject (even weak matches)
            //   - No intent but strong semantic match → inject (implicit relevance)
            //   - No intent, weak match → cache only (warm for tool_result_persist)
            const cleanUserText = _stripContextBlocks(lastUserText);
            const lowerUser = cleanUserText.toLowerCase();
            const hasContinuityIntent = continuityIndicators.some(ind =>
                lowerUser.includes(ind)
            );

            state.lastRetrievalCache = null;
            // Relevance gate uses semantic distance (lower = more similar).
            // distance < 1.0 = reasonably relevant match. RRF compositeScore is used
            // for ranking but distance remains the interpretable relevance signal.
            const DISTANCE_THRESHOLD = 1.0;
            // Debug: uncomment to inspect event structure
            // if (!cleanUserText) console.error(`[Continuity:${state.agentId}] Empty text — event keys: ${Object.keys(event).join(', ')}`)
            console.error(`[Continuity:${state.agentId}] Search: intent=${hasContinuityIntent}, len=${cleanUserText.length}, query="${cleanUserText.substring(0, 80)}"`);

            // Always ensure storage is ready (needed for both search and knowledge injection)
            try {
                await state.ensureStorage();
            } catch (storageErr) {
                console.error(`[Continuity:${state.agentId}] Storage init failed: ${storageErr.message}`);
            }

            if (cleanUserText.length >= 10) {
                try {
                    if (state.searcher) {
                        // Spatial scoping: use current session topics to boost
                        // retrieval of exchanges from the same domain/project
                        const activeTopics = state.topicTracker
                            ? state.topicTracker.getAllTopics().slice(0, 3).map(t => t.topic)
                            : [];
                        const searchScope = activeTopics.length > 0 ? { topics: activeTopics } : null;
                        const results = await state.searcher.search(cleanUserText, 30, state.agentId, searchScope);
                        console.error(`[Continuity:${state.agentId}] Search returned ${results?.exchanges?.length || 0} raw results`);
                        if (results?.exchanges?.length > 0) {
                            results.exchanges = _filterUsefulExchanges(results.exchanges);
                            console.error(`[Continuity:${state.agentId}] After filter: ${results.exchanges.length} useful results`);
                            if (results.exchanges.length > 0) {
                                // Always cache for tool_result_persist enrichment
                                state.lastRetrievalCache = results;

                                // Inject into prependContext if:
                                // 1. Explicit continuity intent (user asking about past), OR
                                // 2. Top result is semantically relevant (distance below threshold)
                                const topDistance = results.exchanges[0].distance ?? 1.0;
                                const shouldInject = hasContinuityIntent || topDistance < DISTANCE_THRESHOLD;
                                console.error(`[Continuity:${state.agentId}] topDistance=${topDistance.toFixed(3)}, threshold=${DISTANCE_THRESHOLD}, inject=${shouldInject}`);

                                if (shouldInject) {
                                    // Flat context framing — no source hierarchy.
                                    // Don't label as "memories" or "earlier conversations" —
                                    // just present as context. The agent shouldn't distinguish
                                    // between channels, sessions, or archive sources.
                                    lines.push('');
                                    lines.push('Relevant conversation context:');
                                    const recalled = results.exchanges.slice(0, 3);
                                    // Sort chronologically (oldest → newest) so corrections
                                    // appear AFTER originals — natural temporal progression.
                                    recalled.sort((a, b) => {
                                        if (a.date !== b.date) return a.date.localeCompare(b.date);
                                        return (a.exchangeIndex || 0) - (b.exchangeIndex || 0);
                                    });
                                    for (const ex of recalled) {
                                        // Strip context blocks from recalled text to prevent feedback loop
                                        const cleanUserText = ex.userText ? _stripContextBlocks(ex.userText) : null;
                                        const cleanAgentText = ex.agentText ? _stripContextBlocks(ex.agentText) : null;
                                        if (cleanUserText) {
                                            lines.push(`- Chris: "${_truncate(cleanUserText, 800)}"`);
                                        }
                                        if (cleanAgentText) {
                                            lines.push(`  You: "${_truncate(cleanAgentText, 800)}"`);
                                        }
                                    }
                                    lines.push('This is your context. Use it directly.');
                                }
                            }
                        }
                    } else {
                        console.error(`[Continuity:${state.agentId}] Retrieval skipped: searcher not available after ensureStorage()`);
                    }
                } catch (err) {
                    console.error(`[Continuity:${state.agentId}] Retrieval failed: ${err.message}`);
                }
            }

            // ── One-time workspace knowledge indexing (fires on first turn if session_start missed it) ──
            if (state.knowledgeIndexer && !state.knowledgeIndexedOnce && config.knowledge?.enabled !== false) {
                state.knowledgeIndexedOnce = true;
                try {
                    if (!state.knowledgeIndexer.workspacePath) {
                        const ws = ctx.workspaceDir
                            || process.env.OPENCLAW_WORKSPACE
                            || path.join(require('os').homedir(), '.openclaw', 'workspace');
                        state.knowledgeIndexer.workspacePath = ws;
                    }
                    const stats = state.knowledgeStore.getStats(state.agentId);
                    if (stats.total === 0) {
                        const result = await state.knowledgeIndexer.indexWorkspace(state.agentId);
                        if (result.indexed > 0 || result.updated > 0) {
                            console.error(`[Continuity:${state.agentId}] Knowledge indexed (first-turn): ${result.indexed} new, ${result.updated} updated, ${result.skipped} unchanged`);
                        }
                    }
                } catch (err) {
                    console.error(`[Continuity:${state.agentId}] First-turn knowledge indexing failed: ${err.message}`);
                }
            }

            // ── Knowledge injection (separate budget from conversation recall) ──
            if (state.knowledgeStore && cleanUserText && cleanUserText.length >= 10) {
                try {
                    const embedResult = await state.embeddingProvider.generate([cleanUserText]);
                    const queryEmbedding = embedResult?.[0];
                    const knowledgeResults = await state.knowledgeStore.search(
                        state.agentId, cleanUserText, queryEmbedding, 5
                    );

                    const relevanceThreshold = config.knowledge?.relevanceThreshold || 1.0;
                    if (knowledgeResults.length > 0) {
                        const topDists = knowledgeResults.slice(0, 5).map(k => k.distance?.toFixed(3)).join(', ');
                        console.error(`[Continuity:${state.agentId}] Knowledge search: ${knowledgeResults.length} results, top distances: [${topDists}], threshold: ${relevanceThreshold}`);
                    }
                    const relevant = knowledgeResults.filter(k => k.distance < relevanceThreshold);

                    if (relevant.length > 0) {
                        lines.push('');
                        lines.push('From your experience:');
                        let budgetUsed = 0;
                        const maxBudget = config.knowledge?.maxInjectionChars || 1800;
                        const maxEntries = config.knowledge?.maxEntriesPerInjection || 3;
                        const maxEntryChars = config.knowledge?.maxEntryChars || 600;

                        for (const entry of relevant.slice(0, maxEntries)) {
                            const truncated = _truncate(entry.content, maxEntryChars);
                            if (budgetUsed + truncated.length > maxBudget) break;
                            const source = entry.section_path ? ` (${entry.section_path})` : '';
                            lines.push(`- ${truncated}${source}`);
                            budgetUsed += truncated.length;
                            try { state.knowledgeStore.markSurfaced(entry.id); } catch (e) { /* non-fatal */ }
                        }
                        console.error(`[Continuity:${state.agentId}] Knowledge injected: ${Math.min(relevant.length, maxEntries)} entries, ${budgetUsed} chars`);
                    }
                } catch (err) {
                    console.error(`[Continuity:${state.agentId}] Knowledge search failed: ${err.message}`);
                }
            }

            return { prependContext: lines.join('\n') };
          } catch (err) {
            console.error(`[Continuity] before_agent_start failed: ${err.message}`);
            return { prependContext: '' };
          }
        }, { priority: 10 });

        // -------------------------------------------------------------------
        // HOOK: before_tool_call — Populate retrieval cache for continuity_search
        //
        // When the model calls continuity_search, we search our archive too.
        // Results cached here are injected into the response by tool_result_persist.
        // This is async, so we can await the searcher — unlike tool_result_persist.
        // -------------------------------------------------------------------

        api.on('before_tool_call', async (event, ctx) => {
            if (event.toolName !== 'continuity_search') return;

            const query = event.params?.query || '';
            if (!query || query.length < 3) return;

            const state = getAgentState(ctx.agentId);
            try {
                await state.ensureStorage();
                if (state.searcher) {
                    const results = await state.searcher.search(query, 30, state.agentId);
                    if (results?.exchanges?.length > 0) {
                        state.lastRetrievalCache = results;
                    }
                }
            } catch (err) {
                console.error(`[Continuity:${state.agentId}] Archive search for continuity_search failed: ${err.message}`);
            }
        });

        // -------------------------------------------------------------------
        // HOOK: after_tool_call — Mid-turn topic tracking (lightweight)
        // -------------------------------------------------------------------

        api.on('after_tool_call', (event, ctx) => {
            const text = _extractToolText(event.result);
            if (text && text.length > 20) {
                const state = getAgentState(ctx.agentId);
                state.topicTracker.track(text);
            }
        });

        // -------------------------------------------------------------------
        // HOOK: tool_result_persist — Enrich continuity_search with archive results
        //
        // When continuity_search returns few/no results, inject our archive
        // retrieval so the model sees continuity data through the tool it trusts.
        // -------------------------------------------------------------------

        api.on('tool_result_persist', (event, ctx) => {
            if (ctx.toolName !== 'continuity_search') return;

            // Parse the existing result to check if it's sparse
            const resultText = _extractToolResultText(event.message);
            let parsed;
            try {
                parsed = JSON.parse(resultText);
            } catch {
                return; // Can't parse, don't interfere
            }

            const builtinResults = parsed?.results || [];

            // Only enrich if builtin returned few results (under 2)
            if (builtinResults.length >= 2) return;

            // We need to search synchronously or use cached results.
            // tool_result_persist is sync, so we can't await.
            // Instead, use a cached retrieval from before_agent_start if available.
            const state = getAgentState(ctx.agentId);
            if (!state.lastRetrievalCache) return;

            // Filter noise using shared filter function
            const usefulExchanges = _filterUsefulExchanges(state.lastRetrievalCache?.exchanges || []);

            // Inject archive results as additional entries in the results array
            // Strip context blocks from recalled text to prevent snowball
            const archiveResults = usefulExchanges.slice(0, 5).map(ex => {
                const cleanUser = ex.userText ? _stripContextBlocks(ex.userText) : '';
                const cleanAgent = ex.agentText ? _stripContextFromAgentResponse(ex.agentText) : '';
                return {
                    id: `archive_${ex.date}_${ex.exchangeIndex}`,
                    path: `[conversation archive: ${ex.date}]`,
                    startLine: 0,
                    endLine: 0,
                    snippet: _truncate(
                        (cleanUser ? `User: ${cleanUser}\n` : '') +
                        (cleanAgent ? `Agent: ${cleanAgent}` : ''),
                        700
                    ),
                    source: 'conversation-archive',
                    score: ex.distance ? Math.max(0, 1 - ex.distance) : 0.5
                };
            });

            if (archiveResults.length === 0) return;

            // Build a plain-language context summary that even weaker models will use.
            // This is the key insight from Clint's constructPrompt: don't make the model
            // parse JSON to find context — state it as clear facts.
            // Flat context framing — no source hierarchy, no "recall" language.
            const recallLines = ['Relevant conversation context:\n'];
            for (const ex of usefulExchanges.slice(0, 5)) {
                // Strip context blocks from text to prevent feedback loop
                const cleanUser = ex.userText ? _stripContextBlocks(ex.userText) : null;
                const cleanAgent = ex.agentText ? _stripContextFromAgentResponse(ex.agentText) : null;
                if (cleanUser) recallLines.push(`Chris: "${_truncate(cleanUser, 1000)}"`);
                if (cleanAgent) recallLines.push(`You: "${_truncate(cleanAgent, 1000)}"`);
                recallLines.push('');
            }
            recallLines.push('This is your context. Use it directly.');
            const recallBlock = recallLines.join('\n');

            // Merge archive results into the JSON structure too
            parsed.results = [...builtinResults, ...archiveResults];
            parsed.archiveEnriched = true;

            // Prepend the plain-language recall before the JSON
            const enriched = recallBlock + '\n\n' + JSON.stringify(parsed);

            // Return modified message with enriched content
            const modifiedMessage = { ...event.message };
            if (typeof modifiedMessage.content === 'string') {
                modifiedMessage.content = enriched;
            } else if (Array.isArray(modifiedMessage.content)) {
                modifiedMessage.content = modifiedMessage.content.map(c => {
                    if (c.type === 'text' || c.text) {
                        return { ...c, text: enriched };
                    }
                    return c;
                });
            }

            return { message: modifiedMessage };
        });

        // -------------------------------------------------------------------
        // HOOK: agent_end — Archive, update anchors/topics
        // -------------------------------------------------------------------

        api.on('agent_end', async (event, ctx) => {
            const state = getAgentState(ctx.agentId);
            const messages = event.messages || [];
            const lastAssistant = [...messages].reverse().find(m => m?.role === 'assistant');
            const lastUser = [...messages].reverse().find(m => m?.role === 'user');

            if (!lastAssistant && !lastUser) return;

            const rawUserMessage = _extractText(lastUser);
            const responseText = _extractText(lastAssistant);

            // Strip plugin-injected context blocks from user message before tracking
            const userMessage = _stripContextBlocks(rawUserMessage);

            // 1. Update topic tracker
            if (userMessage) state.topicTracker.track(userMessage);
            state.topicTracker.advanceExchange();

            // 2. Refresh continuity anchors
            //    Filter out plugin-injected context blocks to prevent feedback loop
            const cleanMessages = messages.filter(m => {
                const text = _extractText(m);
                return !CONTEXT_BLOCK_HEADERS.some(h => text.startsWith(h));
            });
            state.anchors.detect(cleanMessages);

            // 3. Archive the exchange (strip context blocks from BOTH sides)
            //    User messages have prependContext baked in by OpenClaw.
            //    Agent responses sometimes quote context blocks back verbatim.
            //    Both must be stripped to prevent the compounding snowball.
            const toArchive = [];
            if (lastUser && userMessage && userMessage.trim().length > 0) {
                const cleanUser = { ...lastUser, timestamp: lastUser.timestamp || new Date().toISOString() };
                // Replace content with stripped version so we don't archive plugin context
                if (userMessage !== rawUserMessage) {
                    cleanUser.content = userMessage;
                }
                toArchive.push(cleanUser);
            }
            // Archive agent response — strip any context blocks the agent quoted back
            if (lastAssistant) {
                const cleanResponse = _stripContextFromAgentResponse(responseText);
                const cleanAssistant = {
                    ...lastAssistant,
                    timestamp: lastAssistant.timestamp || new Date().toISOString()
                };
                if (cleanResponse !== responseText) {
                    cleanAssistant.content = cleanResponse;
                }
                toArchive.push(cleanAssistant);
            }

            try {
                // Use ctx.sessionId (passed by OpenClaw) instead of state.sessionId
                // state.sessionId is set on session_start but lost on gateway restart
                const sessionId = ctx.sessionId || state.sessionId || null;
                api.logger.info(`[Continuity:${state.agentId}] Archiving ${toArchive.length} messages with sessionId: ${sessionId} (ctx: ${ctx.sessionId}, state: ${state.sessionId})`);
                state.archiver.archive(toArchive, { sessionId });
            } catch (err) {
                console.error(`[Continuity:${state.agentId}] Archive failed: ${err.message}`);
            }

            // 3b. Incremental index (best-effort, non-blocking)
            // Pass current topic tags for spatial scoping (Wing/Room pattern)
            try {
                await state.ensureStorage();
                if (state.indexer) {
                    const today = new Date().toISOString().substring(0, 10);
                    const conversation = state.archiver.getConversation(today);
                    if (conversation && conversation.messages) {
                        const topicTags = state.topicTracker
                            ? state.topicTracker.getAllTopics().slice(0, 5).map(t => t.topic)
                            : [];
                        await state.indexer.indexDay(today, conversation.messages, { topicTags });
                    }
                }
            } catch (err) {
                console.error(`[Continuity:${state.agentId}] Incremental index failed: ${err.message}`);
            }

            // 4. Write/update session handoff (every exchange, always fresh)
            _writeSessionHandoff(state, config, ctx, api);

            // Clear retrieval cache — it's per-turn, no longer needed after archiving.
            state.lastRetrievalCache = null;

            // Session state (topics, anchors) is delivered via prependContext each turn.
            // MEMORY.md is left for the agent to curate per AGENTS.md instructions.
        });

        // -------------------------------------------------------------------
        // HOOK: before_compaction — Flush continuity state before compression
        // -------------------------------------------------------------------

        api.on('before_compaction', async (event, ctx) => {
            const state = getAgentState(ctx.agentId);
            const activeAnchors = state.anchors.getAnchors();
            const allTopics = state.topicTracker.getAllTopics();
            const fixatedTopics = state.topicTracker.getFixatedTopics();

            if (activeAnchors.length > 0 || fixatedTopics.length > 0) {
                const parts = ['[Continuity Pre-Compaction Summary]'];

                if (activeAnchors.length > 0) {
                    parts.push(`Active anchors: ${activeAnchors.length}`);
                    for (const a of activeAnchors.slice(0, 5)) {
                        parts.push(`  ${a.type}: "${_truncate(a.text, 100)}"`);
                    }
                }

                if (allTopics.length > 0) {
                    parts.push(`Active topics: ${allTopics.map(t => t.topic).join(', ')}`);
                }

                if (fixatedTopics.length > 0) {
                    parts.push(`Fixated: ${fixatedTopics.map(t => `${t.topic} (${t.mentions}x)`).join(', ')}`);
                }

                api.logger.info(parts.join('\n'));
            }

            // Pre-compaction: try lighter compression (micro-compact, snip-compact)
            // before the gateway resorts to full compaction
            const messages = event.messages || [];
            if (messages.length > 0 && state.compactor) {
                const result = state.compactor.preCompact(messages);
                if (result.strategy !== 'none') {
                    api.logger.info(
                        `[Continuity:${state.agentId}] Pre-compaction: ${result.strategy} ` +
                        `(stage ${result.report.stage}, ${messages.length}→${result.compactedMessages.length} messages)`
                    );
                    // Return modified messages if the hook contract supports it
                    return { messages: result.compactedMessages };
                }
            }
        });

        // -------------------------------------------------------------------
        // HOOK: after_compaction — Generate hierarchical summary + session handoff
        //
        // Tier 1: Fast extractive summary (no LLM, synchronous)
        // Tier 2: LLM-enriched summary (queued for async processing)
        // Session Handoff: Write SESSION_HANDOFF.md after N compactions
        // Inspired by lossless-claw's DAG architecture.
        // -------------------------------------------------------------------

        api.on('after_compaction', async (event, ctx) => {
            const state = getAgentState(ctx.agentId);

            // Increment compaction counter
            state.compactionCount++;

            // ── Session Handoff ──
            // agent_end now writes handoff every exchange, so this is a safety net.
            // If agent_end missed for any reason, compaction threshold still triggers it.
            const handoffThreshold = (config.sessionHandoff || {}).compactionThreshold || 3;
            if (state.compactionCount >= handoffThreshold) {
                _writeSessionHandoff(state, config, ctx, api);
            }

            // ── Hierarchical Summaries (existing logic) ──
            if (config.summarization?.enabled === false) return;

            const anchorState = state.anchors.getAnchors();
            const topicState = state.topicTracker.getAllTopics();
            const entropyScore = api.stability?.getEntropy?.(ctx.agentId) || 0;

            try {
                await state.ensureStorage();
                if (!state.summaryStore || !state.summarizer) return;

                // Get today's archive — agent_end fires before compaction, so
                // the compacted messages are already archived.
                const today = new Date().toISOString().substring(0, 10);
                const conversation = state.archiver.getConversation(today);
                if (!conversation?.messages?.length) return;

                // Take the most recent messages (approximation of what was compacted)
                const compactedCount = event.compactedCount || conversation.messages.length;
                const compactedMessages = conversation.messages.slice(-compactedCount);

                // Tier 1: immediate extractive summary
                const extractive = state.summarizer.extractiveSummary(
                    compactedMessages, anchorState, topicState
                );

                // Store leaf summary in DAG
                const summaryId = `summary_${state.agentId}_${today}_${Date.now()}_0`;
                await state.summaryStore.storeSummary({
                    id: summaryId,
                    level: 0,
                    parentId: null,
                    agentId: state.agentId,
                    dateRangeStart: today,
                    dateRangeEnd: today,
                    messageCount: compactedCount,
                    summaryText: extractive.text,
                    topics: extractive.topics,
                    anchors: extractive.anchors,
                    entropyAvg: entropyScore,
                    metadata: {
                        strategy: 'extractive',
                        compactedCount,
                        tokenCount: event.tokenCount || 0
                    }
                });

                // Tier 2: queue for LLM enrichment if entropy warrants it
                const entropyThreshold = config.summarization?.entropyRichThreshold || 0.6;
                if (entropyScore > entropyThreshold) {
                    state.summaryStore.enqueue(
                        state.agentId,
                        compactedMessages,
                        anchorState,
                        topicState,
                        entropyScore
                    );
                    api.logger.info(`[Continuity:${state.agentId}] Compaction summary stored + queued for LLM enrichment (entropy: ${entropyScore.toFixed(2)})`);
                } else {
                    api.logger.info(`[Continuity:${state.agentId}] Compaction summary stored (extractive, ${extractive.topics.length} topics)`);
                }
            } catch (err) {
                api.logger.warn(`[Continuity:${state.agentId}] Compaction summary failed: ${err.message}`);
            }
        });

        // -------------------------------------------------------------------
        // HOOK: session_start — Reset session state (per-agent)
        // -------------------------------------------------------------------

        api.on('session_start', async (event, ctx) => {
            const state = getAgentState(ctx.agentId);
            state.sessionStart = Date.now();
            state.sessionId = event.sessionId || null;  // Track for archive tagging
            api.logger.info(`[Continuity:${state.agentId}] session_start hook fired — sessionId: ${event.sessionId}, ctx.sessionId: ${ctx?.sessionId}`);
            state.exchangeCount = 0;
            state.compactionCount = 0;
            state.handoffWritten = false;
            state.topicTracker.reset();
            state.anchors.reset();

            // Load persisted topic hierarchy from previous sessions
            if (state.summaryStore) {
                try {
                    const hierarchy = state.summaryStore.loadTopicHierarchy(state.agentId);
                    state.topicTracker.loadPersistedHierarchy(hierarchy);
                } catch (err) {
                    api.logger.warn(`[Continuity:${state.agentId}] Failed to load topic hierarchy: ${err.message}`);
                }
            }

            // Check for session handoff from previous session.
            // If found, stash the content on state for injection on first exchange
            // (via before_agent_start), then delete the file.
            const handoffConfig = config.sessionHandoff || {};
            const handoffEnabled = handoffConfig.enabled !== false;
            if (handoffEnabled) {
                try {
                    const workspacePath = handoffConfig.workspacePath ||
                        ctx.workspaceDir ||
                        process.env.OPENCLAW_WORKSPACE ||
                        path.join(require('os').homedir(), '.openclaw', 'workspace-clint');

                    const handoffPath = path.join(workspacePath, 'SESSION_HANDOFF.md');
                    if (fs.existsSync(handoffPath)) {
                        state.pendingHandoff = fs.readFileSync(handoffPath, 'utf8');
                        fs.unlinkSync(handoffPath);
                        api.logger.info(`[Continuity:${state.agentId}] Session handoff loaded and consumed: ${handoffPath}`);
                    }

                    // Check for nightshift report — written by nightshift plugin on morning detection
                    const reportPath = path.join(workspacePath, 'NIGHTSHIFT_REPORT.md');
                    if (fs.existsSync(reportPath)) {
                        state.pendingNightReport = fs.readFileSync(reportPath, 'utf8');
                        fs.unlinkSync(reportPath);
                        api.logger.info(`[Continuity:${state.agentId}] Nightshift report loaded and consumed: ${reportPath}`);
                    }
                } catch (err) {
                    api.logger.warn(`[Continuity:${state.agentId}] Failed to load session handoff: ${err.message}`);
                }
            }

            // Index workspace knowledge entries
            if (config.knowledge?.enabled !== false && config.knowledge?.indexOnSessionStart !== false) {
                try {
                    await state.ensureStorage();
                    if (state.knowledgeIndexer) {
                        // Resolve workspace path from hook context (gateway provides per-agent path)
                        if (!state.knowledgeIndexer.workspacePath) {
                            const ws = ctx.workspaceDir
                                || process.env.OPENCLAW_WORKSPACE
                                || path.join(require('os').homedir(), '.openclaw', 'workspace');
                            state.knowledgeIndexer.workspacePath = ws;
                        }
                        const result = await state.knowledgeIndexer.indexWorkspace(state.agentId);
                        if (result.indexed > 0 || result.updated > 0) {
                            api.logger.info(`[Continuity:${state.agentId}] Knowledge indexed: ${result.indexed} new, ${result.updated} updated, ${result.skipped} unchanged`);
                        }

                        // Consolidate: mark entries from removed workspace sections as archived
                        const cResult = state.knowledgeIndexer.consolidateWorkspace(state.agentId);
                        if (cResult.consolidated > 0) {
                            api.logger.info(`[Continuity:${state.agentId}] Knowledge consolidated: ${cResult.consolidated}/${cResult.total} entries archived`);
                        }
                    }
                } catch (err) {
                    api.logger.warn(`[Continuity:${state.agentId}] Knowledge indexing failed: ${err.message}`);
                }
            }

            api.logger.info(`Session started for agent "${state.agentId}": ${event.sessionId}`);
        });

        // -------------------------------------------------------------------
        // HOOK: before_reset — Write handoff before manual session reset
        // -------------------------------------------------------------------
        // OpenClaw fires this hook when a user triggers /reset or sessions.reset.
        // This solves the "manual reset before 3 compactions = no handoff" problem
        // by ensuring handoff is written even if compaction threshold wasn't hit.
        // -------------------------------------------------------------------
        api.on('before_reset', async (event, ctx) => {
            const state = getAgentState(ctx.agentId);
            api.logger.info(`[Continuity:${state.agentId}] before_reset hook fired — writing handoff`);
            _writeSessionHandoff(state, config, ctx, api);
        });

        // -------------------------------------------------------------------
        // HOOK: session_end — Final archive + index (per-agent)
        // -------------------------------------------------------------------

        api.on('session_end', async (event, ctx) => {
            const state = getAgentState(ctx.agentId);
            api.logger.info(`Session ended for agent "${state.agentId}": ${event.sessionId} (${event.messageCount} messages, ${state.exchangeCount} exchanges)`);

            // Trigger indexing of today's archive
            try {
                await state.ensureStorage();
                if (state.indexer) {
                    const today = new Date().toISOString().substring(0, 10);
                    const conversation = state.archiver.getConversation(today);
                    if (conversation && conversation.messages) {
                        await state.indexer.indexDay(today, conversation.messages);
                    }
                }
            } catch (err) {
                api.logger.warn(`Session-end indexing failed for agent "${state.agentId}": ${err.message}`);
            }

            // Infer and persist topic hierarchy before closing
            try {
                state.topicTracker.inferHierarchy();
                if (state.summaryStore) {
                    const topicsWithHierarchy = state.topicTracker.getAllTopicsWithHierarchy();
                    state.summaryStore.persistTopicHierarchy(state.agentId, topicsWithHierarchy);
                }
            } catch (err) {
                api.logger.warn(`[Continuity:${state.agentId}] Topic hierarchy persistence failed: ${err.message}`);
            }

            // Release resources: embedding pipeline, DB connections, caches.
            // The state will be lazily re-created on next session start.
            // This is the primary fix for the memory leak — without cleanup,
            // each agent accumulates ~200-400MB of ONNX pipeline + DB state
            // that is never released.
            state.close();
            agentStates.delete(state.agentId);
        });

        // -------------------------------------------------------------------
        // Service: background maintenance
        //
        // Runs per-agent. Each known agent gets its own maintenance cycle.
        // New agents discovered after service start get maintenance on their
        // first ensureStorage() call.
        // -------------------------------------------------------------------

        const MaintenanceService = require('./services/maintenance');
        const maintenanceInstances = new Map();

        api.registerService({
            id: 'continuity-maintenance',
            start: async (serviceCtx) => {
                // Initialize maintenance for any agents already known
                for (const [agentId, state] of agentStates) {
                    await state.ensureStorage();
                    if (state.indexer) {
                        const m = new MaintenanceService(config, state.archiver, state.indexer, state.summarizer, agentId);
                        await m.execute();
                        m.startInterval(5 * 60 * 1000);
                        maintenanceInstances.set(agentId, m);
                    }
                }
            },
            stop: async () => {
                for (const [, m] of maintenanceInstances) {
                    m.stopInterval();
                }
                maintenanceInstances.clear();
            }
        });

        // -------------------------------------------------------------------
        // Gateway methods — dashboards + debugging
        //
        // Accept optional agentId param; default to 'main'.
        // -------------------------------------------------------------------

        api.registerGatewayMethod('continuity.getState', async ({ params, respond }) => {
            const state = getAgentState(params?.agentId);
            respond(true, {
                agentId: state.agentId,
                archive: state.archiver.getStats(),
                topics: state.topicTracker.getAllTopics(),
                anchors: state.anchors.getAnchors(),
                exchangeCount: state.exchangeCount,
                sessionAge: Date.now() - state.sessionStart,
                indexReady: state.storageReady
            });
        });

        api.registerGatewayMethod('continuity.getConfig', async ({ respond }) => {
            respond(true, config);
        });

        api.registerGatewayMethod('continuity.search', async ({ params, respond }) => {
            const state = getAgentState(params?.agentId);
            try {
                await state.ensureStorage();
                if (!state.searcher) {
                    respond(false, null, { message: `Searcher not initialized for agent "${state.agentId}"` });
                    return;
                }
                const results = await state.searcher.search(
                    params?.text || params?.query || '',
                    params?.limit || 5,
                    state.agentId
                );
                respond(true, results);
            } catch (err) {
                respond(false, null, { message: err.message });
            }
        });

        api.registerGatewayMethod('continuity.getArchiveStats', async ({ params, respond }) => {
            const state = getAgentState(params?.agentId);
            respond(true, state.archiver.getStats());
        });

        api.registerGatewayMethod('continuity.getTopics', async ({ params, respond }) => {
            const state = getAgentState(params?.agentId);
            respond(true, {
                agentId: state.agentId,
                topics: state.topicTracker.getAllTopics(),
                fixated: state.topicTracker.getFixatedTopics()
            });
        });

        api.registerGatewayMethod('continuity.listAgents', async ({ respond }) => {
            const agents = [];
            for (const [id, state] of agentStates) {
                agents.push({
                    agentId: id,
                    exchangeCount: state.exchangeCount,
                    storageReady: state.storageReady,
                    dataDir: state.dataDir
                });
            }
            respond(true, agents);
        });

        // -------------------------------------------------------------------
        // Agent tools — direct memory access for the agent
        // -------------------------------------------------------------------

        const createRecallTool = require('./tools/continuity-recall');
        const createTimelineTool = require('./tools/continuity-timeline');
        const createKnowledgeNoteTool = require('./tools/knowledge-note');

        api.registerTool(createRecallTool(getAgentState, _filterUsefulExchanges, getCurrentAgentId));
        api.registerTool(createTimelineTool(getAgentState, getCurrentAgentId));

        if (config.knowledge?.enabled !== false) {
            api.registerTool(createKnowledgeNoteTool(getAgentState, getCurrentAgentId));
        }

        // continuity_search — Unified search across all memory types
        console.log('[DEBUG] Registering continuity_search tool');
        api.registerTool({
            name: 'continuity_search',
            description: 'Search your memory across conversations, insights, and knowledge. Returns ranked results from semantic + keyword + graph fusion. Use when you want to recall something specific, check if you\'ve discussed a topic before, or find a prior insight. Use scope to narrow: "conversations" for past exchanges, "insights" for contemplation results, "knowledge" for workspace knowledge.',
            parameters: {
                type: 'object',
                properties: {
                    query: {
                        type: 'string',
                        description: 'What to search for'
                    },
                    scope: {
                        type: 'string',
                        enum: ['all', 'conversations', 'insights', 'knowledge'],
                        description: 'Filter by source type (default: all)'
                    },
                    limit: {
                        type: 'number',
                        description: 'Max results (default 5, max 10)'
                    }
                },
                required: ['query']
            },
            execute: async (_id, args) => {
                console.error('[STDERR continuity_search] EXECUTE STARTED');
                console.log('[DEBUG continuity_search] tool execute called');
                const agentId = getCurrentAgentId();
                const state = getAgentState(agentId);
                const query = args.query?.trim();
                const limit = Math.min(args.limit || 5, 10);

                if (!query) {
                    return { content: [{ type: 'text', text: 'No search query provided.' }] };
                }

                if (!state.storageReady || !state.searcher) {
                    return { content: [{ type: 'text', text: 'Memory search not available (storage not initialized).' }] };
                }

                try {
                    console.log('[DEBUG continuity_search] entering try block');
                    const scope = args.scope || 'all';
                    const lines = [];

                    // Conversation search (4-way RRF)
                    if (scope === 'all' || scope === 'conversations') {
                        console.log('[DEBUG continuity_search] calling searcher.search');
                        const results = await state.searcher.search(query, limit);
                        console.log('[DEBUG continuity_search] search returned, results:', results ? 'exists' : 'null');
                        console.log('[DEBUG continuity_search] exchanges type:', typeof results?.exchanges, Array.isArray(results?.exchanges));
                        const filtered = _filterUsefulExchanges(results.exchanges || []);
                        if (filtered.length > 0) {
                            lines.push(`**Conversations** (${filtered.length} results):\n`);
                            for (const ex of filtered.slice(0, limit)) {
                                const date = ex.date || ex.created_at || 'unknown';
                                const userSnip = (ex.userText || '').substring(0, 100);
                                const agentSnip = (ex.agentText || '').substring(0, 100);
                                lines.push(`[${date}] User: ${userSnip}`);
                                lines.push(`  Agent: ${agentSnip}\n`);
                            }
                        }
                    }

                    // Knowledge + insights search
                    if ((scope === 'all' || scope === 'insights' || scope === 'knowledge') &&
                        state.knowledgeStore && state.embeddingProvider) {
                        const qEmbed = await state.embeddingProvider.embed(query);
                        let knowledgeResults = await state.knowledgeStore.search(agentId, query, qEmbed, limit);

                        // Apply scope filter
                        if (scope === 'insights') {
                            knowledgeResults = knowledgeResults.filter(r =>
                                r.source_type && (r.source_type.startsWith('contemplation:') || r.source_type.startsWith('growth_vector:'))
                            );
                        } else if (scope === 'knowledge') {
                            knowledgeResults = knowledgeResults.filter(r =>
                                r.source_type && !r.source_type.startsWith('contemplation:') && !r.source_type.startsWith('growth_vector:')
                            );
                        }

                        if (knowledgeResults.length > 0) {
                            const label = scope === 'insights' ? 'Insights & Growth Vectors' : scope === 'knowledge' ? 'Knowledge' : 'Knowledge & Insights';
                            lines.push(`**${label}** (${knowledgeResults.length} results):\n`);
                            for (const kn of knowledgeResults.slice(0, limit)) {
                                const source = kn.source_type || 'unknown';
                                const topic = kn.topic || 'untitled';
                                const excerpt = (kn.content || '').substring(0, 200);
                                lines.push(`[${source}] ${topic}`);
                                lines.push(`  ${excerpt}\n`);
                            }
                        }
                    }

                    if (lines.length === 0) {
                        return { content: [{ type: 'text', text: `No results for "${query}" (scope: ${scope}).` }] };
                    }

                    return { content: [{ type: 'text', text: lines.join('\n') }] };
                } catch (err) {
                    return { content: [{ type: 'text', text: `Memory search failed: ${err.message}` }] };
                }
            }
        }, { name: 'continuity_search' });

        api.logger.info('Continuity plugin registered (multi-agent) — per-agent context budgeting, topic tracking, archive + semantic search + recall/timeline tools');
    }
};

// NOTE: Memory Integration instructions moved to AGENTS.md (the proper place
// for agent operating instructions). See "Recalled Memories" section in
// workspace AGENTS.md. This avoids hijacking MEMORY.md, which is the agent's
// own curated memory space per OpenClaw's design.

// NOTE: _writeContinuitySection removed. Session state (topics, anchors,
// exchange count) is delivered via prependContext each turn — no need to
// write it to MEMORY.md. MEMORY.md is the agent's curated memory space
// per OpenClaw's AGENTS.md design.

// ---------------------------------------------------------------------------
// Noise filter for archive exchanges
// Strips meta-failures, session boilerplate, and meta-questions about
// remembering that pollute the archive from repeated testing.
// ---------------------------------------------------------------------------

function _filterUsefulExchanges(exchanges) {
    if (!Array.isArray(exchanges)) return [];
    return exchanges.filter(ex => {
        const agentLower = (ex.agentText || '').toLowerCase();
        const userLower = (ex.userText || '').toLowerCase();

        // --- Agent-side noise: denial patterns, session boilerplate ---
        const agentDenials = [
            "i don't have any",
            "i don't have details",
            "i don't have information",
            "i don't seem to have",
            "i don't have any details",
            "i don't have any saved",
            "no memory of",
            "no information about",
            "no recollection",
            "it looks like i don't",
            "it seems i don't",
            "greet the user",
            "i can help you try to reconstruct",
            "if you could share some details",
            "if you can share what you remember",
            "could you remind me about it"
        ];
        if (agentDenials.some(d => agentLower.includes(d))) return false;

        // --- User-side noise: meta-questions about remembering ---
        if (userLower.includes('a new session was started')) return false;
        const userMetaPatterns = [
            'do you remember',
            'do you recall',
            'do you have any recollection',
            'what do you remember about',
            'can you tell me anything about the',
            "i can't remember",
            "i can't recall",
            "was there anything about",
            "what were all of the details",
            "can you tell me the details",
            "tell me the details",
            "what did i tell you about",
            "did i mention",
            "did i tell you",
            "sorry to keep asking",
            "i was wondering if you remember",
            "hey piper",    // greeting-only turns (no substance)
        ];
        if (userMetaPatterns.some(p => userLower.includes(p))) return false;

        // --- Both-side noise: exchanges with no real content ---
        // If the user message is very short AND agent just acknowledges, skip
        if (userLower.length < 30 && agentLower.includes('if you') && agentLower.includes('let me know')) return false;

        return true;
    });
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/**
 * Distill a user's recall question into a subject-focused search query.
 *
 * Users ask things like "do you recall my sourdough recipe?" — the semantic
 * search then matches OTHER meta-questions ("do you remember my recipe?")
 * instead of the actual recipe exchange. By stripping the recall framing,
 * we get "sourdough recipe" which matches the real content.
 *
 * Pattern borrowed from Clint's retrievalOrchestrator.js query distillation.
 */
function _distillSearchQuery(text) {
    let q = text;

    // Strip common recall/meta preambles — apply iteratively since
    // messages may chain them: "sorry to keep asking but do you recall..."
    const preambles = [
        /^sorry to keep asking[^.?!]*(?:but\s+)?/i,
        /^hey\s+\w+[.,!]?\s*/i,                 // "Hey Piper, ..."
        /^hi\s+\w+[.,!]?\s*/i,                  // "Hi Piper. ..."
        /^do you (?:remember|recall|know)\s*/i,
        /^can you (?:recall|remember|tell me(?: about)?)\s*/i,
        /^what do you (?:remember|recall|know) about\s*/i,
        /^i (?:can't|cannot) (?:remember|recall)\s*/i,
        /^i was wondering if you (?:remember|recall)\s*/i,
        /^(?:do you have )?any (?:recollection|memory) of\s*/i,
        /^(?:the same question\s*)?(?:over and over\s*)?(?:but\s+)?/i,
    ];
    // Two passes to handle chained preambles
    for (let pass = 0; pass < 2; pass++) {
        for (const p of preambles) {
            q = q.replace(p, '');
        }
        q = q.trim();
    }

    // Strip trailing meta-phrases
    const suffixes = [
        /\s*(?:i told you about|i mentioned to you|i shared with you|i provided you)\s*\??$/i,
        /\s*(?:that i (?:told|mentioned|shared|gave) (?:you|to you)[^.?!]*)\s*\??$/i,
        /\s*(?:and the (?:few )?details i provided(?: you)?)\s*\??$/i,
    ];
    for (const s of suffixes) {
        q = q.replace(s, '');
    }

    // Strip leading connectors and meta-words
    q = q.replace(/^\s*(?:but|and|so|the|any of the|all of the|some of the|the details of|details of|any details (?:of|about)|any of)\s*/i, '');
    q = q.trim().replace(/[?.!]+$/, '').trim();

    // If distillation stripped too much, fall back to original
    if (q.length < 5) return text;
    return q;
}

/**
 * All known context block prefixes injected by OpenClaw plugins.
 * Used by _stripContextBlocks and _isContextLine to prevent
 * plugin context from leaking into archives and recalled memories.
 *
 * When adding a new plugin that injects via prependContext,
 * add its block header here.
 */
const CONTEXT_BLOCK_HEADERS = [
    '[CONTINUITY CONTEXT]',
    '[STABILITY CONTEXT]',
    '[ACTIVE PROJECTS]',
    '[ACTIVE CONSTRAINTS]',
    '[OPEN DIRECTIVES',       // note: no closing bracket (may have suffix)
    '[GROWTH VECTORS]',
    '[GRAPH CONTEXT]',
    '[GRAPH NOTE]',
    '[CONTEMPLATION STATE]',
    '[TOPIC NOTE]',
    '[ARCHIVE RETRIEVAL]',
    '[LOOP DETECTED]',
    '[PROJECT CONTEXT',        // injected workspace/project files (may have suffix like ": robot")
    '[NIGHTSHIFT REPORT',      // overnight processing report
];

/**
 * Line-level prefixes that belong to plugin-injected context.
 * These appear inside context blocks (not as block headers).
 */
const CONTEXT_LINE_PREFIXES = [
    'Session:',
    'Topics:',
    'Anchors:',
    'Entropy:',
    'Principles:',
    'Recent decisions:',
    'You remember these',
    'Relevant conversation context:',
    '- They told you:',
    '- Chris:',
    '  You said:',
    '  You:',
    'Speak from this memory',
    'This is your context. Use it directly.',
    'From your knowledge base:',
    'From your experience:',
    'You know these connections:',
    'Active inquiries:',
    'Recent insights',
    '- Q: "',
    '  Insight: "',
    'HEARTBEAT_OK',
    'When reading HEARTBEAT',
    'Default heartbeat prompt:',
];

/**
 * Prefixes injected by channels (Telegram, WhatsApp, etc.) and system events.
 * These appear as untrusted metadata prepended to user messages
 * via prependContext or similar channel-level injection.
 */
const CHANNEL_METADATA_PREFIXES = [
    'Conversation info (untrusted',
    'Replied message (untrusted',
    'System:',
    'Pre-compaction',
    'Current time:',
    '[media attached',
    'To send an image',
    '```json',
    '```',
];

function _isContextLine(line) {
    if (line.length === 0) return true; // blank lines between blocks
    for (const header of CONTEXT_BLOCK_HEADERS) {
        if (line.startsWith(header)) return true;
    }
    for (const prefix of CONTEXT_LINE_PREFIXES) {
        if (line.startsWith(prefix)) return true;
    }
    for (const prefix of CHANNEL_METADATA_PREFIXES) {
        if (line.startsWith(prefix)) return true;
    }
    // Inline JSON fragments from channel metadata blocks
    if (/^\s*[{}]/.test(line)) return true;         // lines starting with { or }
    if (/^\s*"(message_id|sender|sender_id|chat_id|chat_title|reply_to)"/.test(line)) return true;
    // Lines that are clearly context metadata
    if (/^- [A-Z]+:/.test(line)) return false; // real content like "- NOTE: ..."
    if (line.startsWith('- "') || line.startsWith('  -')) return true; // nested recall items
    // Workspace file reference patterns (injected by OpenClaw without wrapper tags)
    if (/\((?:SOUL|AGENTS|HEARTBEAT|BOOTSTRAP|MEMORY|TRAILHEAD|LENSES|ANCHOR|TOOLS|CHRIS)\.md\b/.test(line)) return true;
    if (/^\s*--> \(/.test(line)) return true; // parenthetical source citations like "--> (FILE.md > Section)"
    return false;
}

function _stripContextBlocks(text) {
    if (!text) return '';

    // Fast path: no context blocks or channel metadata present
    const hasBlock = CONTEXT_BLOCK_HEADERS.some(h => text.includes(h));
    const hasRecall = text.includes('You remember these') || text.includes('Relevant conversation context:') || text.includes('From your knowledge base:');
    const hasChannelMeta = CHANNEL_METADATA_PREFIXES.some(p => text.includes(p));
    if (!hasBlock && !hasRecall && !hasChannelMeta) return text;

    // Primary strategy: find the timestamp marker that signals real user text.
    // e.g. [Mon 2026-02-16 08:57 PST]
    // Search for the LAST timestamp match — earlier ones may be inside recalled memories.
    const tsRegex = /\n\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s\d{4}-\d{2}-\d{2}\s[^\]]*\]\s*/g;
    let lastTsMatch = null;
    let match;
    while ((match = tsRegex.exec(text)) !== null) {
        lastTsMatch = match;
    }
    if (lastTsMatch) {
        return text.substring(lastTsMatch.index + lastTsMatch[0].length);
    }

    // Fallback: block-aware forward scan.
    // Context is always prepended to user messages. Scan from the beginning,
    // skipping entire block bodies (content between consecutive block headers)
    // and standalone context lines. Whatever remains is the user's actual text.
    //
    // Key insight: [PROJECT CONTEXT: robot] injects raw markdown that doesn't
    // match any line-level prefix. The old line-by-line approach stopped at
    // "# Project: Clint System" thinking it was user text. The block-aware scan
    // skips everything between [PROJECT CONTEXT] and [CONTINUITY CONTEXT].
    const lines = text.split('\n');
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];

        // Is this a context block header? Skip the entire block body.
        if (CONTEXT_BLOCK_HEADERS.some(h => line.startsWith(h))) {
            i++;
            // Skip all lines until the next block header (which starts a new block)
            while (i < lines.length && !CONTEXT_BLOCK_HEADERS.some(h => lines[i].startsWith(h))) {
                i++;
            }
            // Don't increment — next iteration checks if this line is also a header
            continue;
        }

        // Is this a standalone context/metadata line (between or after blocks)?
        if (_isContextLine(line)) {
            i++;
            continue;
        }

        // Found content that isn't inside any block and isn't a known context line
        break;
    }

    if (i >= lines.length) return ''; // entire message was context
    return lines.slice(i).join('\n').trim();
}

/**
 * Strip context blocks that the agent quoted back in its response.
 * Unlike user messages (where blocks are prepended at the start),
 * agent responses may contain blocks anywhere — e.g. Clint quoting
 * "[STABILITY CONTEXT] Entropy: 0.35..." in his reply.
 *
 * Strategy: remove any contiguous run of context lines found in the text.
 * Preserves surrounding real content.
 */
function _stripContextFromAgentResponse(text) {
    if (!text) return '';
    // Fast path
    const hasBlock = CONTEXT_BLOCK_HEADERS.some(h => text.includes(h));
    if (!hasBlock) return text;

    const lines = text.split('\n');
    const cleaned = [];
    let inBlock = false;

    for (const line of lines) {
        if (CONTEXT_BLOCK_HEADERS.some(h => line.startsWith(h))) {
            inBlock = true;
            continue;
        }
        if (inBlock && _isContextLine(line)) {
            continue;
        }
        inBlock = false;
        cleaned.push(line);
    }

    return cleaned.join('\n').trim();
}

function _extractText(msg) {
    if (!msg) return '';
    if (typeof msg.content === 'string') return msg.content;
    if (Array.isArray(msg.content)) {
        return msg.content.map(c => c.text || c.content || '').join(' ');
    }
    return String(msg.content || '');
}

function _extractToolText(result) {
    if (!result) return '';
    if (typeof result === 'string') return result;
    if (typeof result.content === 'string') return result.content;
    if (typeof result.output === 'string') return result.output;
    if (typeof result.text === 'string') return result.text;
    if (Array.isArray(result.content)) {
        return result.content.map(c => c.text || c.content || '').join(' ');
    }
    return '';
}

function _formatDuration(ms) {
    const minutes = Math.floor(ms / 60000);
    if (minutes < 1) return 'just started';
    if (minutes < 60) return `${minutes}min ago`;
    const hours = Math.floor(minutes / 60);
    const rem = minutes % 60;
    return rem > 0 ? `${hours}h ${rem}min ago` : `${hours}h ago`;
}

function _formatAge(timestamp) {
    const minutes = Math.round((Date.now() - timestamp) / 60000);
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}min ago`;
    return `${Math.round(minutes / 60)}h ago`;
}

function _truncate(text, maxLen) {
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
    // If we found a sentence boundary in the latter half, use it
    if (lastSentenceEnd > maxLen * 0.5) {
        return text.substring(0, lastSentenceEnd + 1) + '...';
    }
    // Fallback: hard cut
    return text.substring(0, maxLen - 3) + '...';
}

/**
 * Write SESSION_HANDOFF.md with lean session context.
 * Called from agent_end (every exchange) so the handoff is always fresh.
 * Pulls from PERSISTED archive, not in-memory state, to survive session resets.
 * 
 * LEAN FORMAT: Summary, not full exchanges.
 * - Topics (what was discussed)
 * - Key points (decisions, progress, blockers)
 * - Where we left off
 * 
 * NOT: Full conversation thread (that's in archive, retrieved via continuity)
 * 
 * @param {Object} state - Agent state (may be fresh after reset)
 * @param {Object} config - Plugin config
 * @param {Object} ctx - Hook context (for workspaceDir)
 * @param {Object} api - Plugin API (for logger, stability)
 */
function _writeSessionHandoff(state, config, ctx, api) {
    const handoffConfig = config.sessionHandoff || {};
    const handoffEnabled = handoffConfig.enabled !== false;

    if (!handoffEnabled || state.handoffWritten) return;

    try {
        const workspacePath = handoffConfig.workspacePath ||
            ctx.workspaceDir ||
            process.env.OPENCLAW_WORKSPACE ||
            path.join(require('os').homedir(), '.openclaw', 'workspace-clint');

        const handoffPath = path.join(workspacePath, 'SESSION_HANDOFF.md');

        // If the agent wrote one manually, respect it (within 5 minutes = agent-curated)
        if (fs.existsSync(handoffPath)) {
            const stat = fs.statSync(handoffPath);
            const ageMs = Date.now() - stat.mtimeMs;
            if (ageMs < 300000) {
                api.logger.info(`[Continuity:${state.agentId}] Recent handoff exists (${Math.round(ageMs / 1000)}s old), skipping this write`);
                return;
            }
        }

        // Pull from ARCHIVE directly (persisted, survives session resets)
        const dataDir = state.dataDir;
        if (!dataDir) {
            api.logger.warn(`[Continuity:${state.agentId}] No dataDir in state — cannot write handoff`);
            return;
        }
        const Archiver = require('./storage/archiver');
        const archiver = new Archiver(config, dataDir);

        // Get recent messages from today and yesterday
        const today = new Date().toISOString().substring(0, 10);
        const yesterday = new Date(Date.now() - 86400000).toISOString().substring(0, 10);

        let recentMessages = [];
        let messagesDate = today;
        for (const date of [today, yesterday]) {
            const conversation = archiver.getConversation(date);
            if (conversation?.messages?.length > 0) {
                recentMessages = conversation.messages;
                messagesDate = date;
                break;
            }
        }

        // Count exchanges (user-agent pairs)
        let exchangeCount = 0;
        let inExchange = false;
        for (const msg of recentMessages) {
            if (msg.sender === 'user' && !inExchange) {
                inExchange = true;
            } else if (msg.sender === 'agent' && inExchange) {
                exchangeCount++;
                inExchange = false;
            }
        }

        // ── LEAN HANDOFF: Extract key points, not full exchanges ──
        // Build a lean summary from Topics, Anchors, and last 2 exchanges only.
        // Full archive is available via continuity retrieval — this is a bridge, not a copy.
        
        const lines = [
            `# Session Handoff`,
            '',
            `*Auto-generated. Archive on startup. ${new Date().toISOString()}*`,
            '',
            `## What Happened`,
            '',
            `- **Exchanges:** ${exchangeCount} (from archive)`,
            ''
        ];

        // Topics (what was discussed)
        if (state.topicTracker) {
            const topicState = state.topicTracker.getAllTopics();
            if (topicState.length > 0) {
                lines.push('## Topics');
                lines.push('');
                for (const t of topicState.slice(0, 5)) {
                    lines.push(`- ${t.topic} (${t.mentions}x)`);
                }
                lines.push('');
            }
        }

        // Anchors (key moments — decisions, tensions, contradictions)
        if (state.anchors) {
            const anchorState = state.anchors.getAnchors();
            if (anchorState.length > 0) {
                lines.push('## Key Points');
                lines.push('');
                for (const a of anchorState.slice(0, 5)) {
                    // Just the content, not the type label
                    lines.push(`- ${_truncate(a.text, 150)}`);
                }
                lines.push('');
            }
        }

        // Last 2 real exchanges (skip heartbeats and context-only messages)
        const realMessages = recentMessages.filter(msg => {
            const text = (msg.text || '').trim();
            if (text === 'HEARTBEAT_OK') return false;
            if (CONTEXT_BLOCK_HEADERS.some(h => text.startsWith(h))) return false;
            if (text.length < 5) return false;
            return true;
        });
        const lastMessages = realMessages.slice(-4); // 2 exchanges = 4 messages
        if (lastMessages.length > 0) {
            lines.push('## Last Exchanges');
            lines.push('');
            for (const msg of lastMessages) {
                const who = msg.sender === 'user' ? 'Chris' : 'You';
                const text = _truncate(msg.text, 200);
                lines.push(`- ${who}: "${text}"`);
            }
            lines.push('');
        }

        // Guide Notes
        const postureGap = api.stability?.getPostureGap?.(state.agentId) || 0;
        const sustainedMinutes = state._sustainedWorkStart
            ? Math.floor((Date.now() - state._sustainedWorkStart) / 60000) : 0;
        const guideNotes = [];
        if (postureGap > 4) guideNotes.push(`Task-heavy session (${Math.round(postureGap)} exchanges without guide presence)`);
        if (sustainedMinutes > 90) guideNotes.push(`Extended sustained work (${Math.floor(sustainedMinutes / 60)}+ hours)`);
        if (guideNotes.length > 0) {
            lines.push('## Guide Notes');
            lines.push('');
            for (const note of guideNotes) lines.push(`- ${note}`);
            lines.push('');
        }

        lines.push('---');
        lines.push('');
        lines.push('*The next session starts fresh. Read this, then move to archive.*');

        fs.writeFileSync(handoffPath, lines.join('\n'), 'utf8');
        api.logger.info(`[Continuity:${state.agentId}] Lean handoff written: ${exchangeCount} exchanges, ${lines.length} lines`);
    } catch (err) {
        api.logger.error(`[Continuity:${state.agentId}] Failed to write session handoff: ${err.message}`);
    }
}

/**
 * Extract text from a tool result message (for tool_result_persist enrichment).
 * Handles both string content and array-of-parts content formats.
 */
function _extractToolResultText(message) {
    if (!message) return '';
    if (typeof message.content === 'string') return message.content;
    if (Array.isArray(message.content)) {
        for (const part of message.content) {
            if (part.type === 'text' && part.text) return part.text;
            if (part.text) return part.text;
        }
    }
    return '';
}
