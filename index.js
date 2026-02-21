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
                this.archiver = new Archiver(config, this.dataDir);

                // Storage (lazy init — embedding model is expensive)
                this.indexer = null;
                this.searcher = null;
                this.storageReady = false;
                this.storageInitPromise = null;

                // Session state
                this.sessionStart = Date.now();
                this.exchangeCount = 0;

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
                        this.indexer = new Indexer(config, this.dataDir);
                        await this.indexer.initialize();
                        this.searcher = new Searcher(config, this.dataDir, this.indexer.db);
                        await this.searcher.initialize();
                        this.storageReady = true;
                        api.logger.info(`[Continuity] Storage ready for agent "${this.agentId}" at ${this.dataDir}`);
                    } catch (err) {
                        api.logger.error(`[Continuity] Storage init failed for agent "${this.agentId}": ${err.message}`);
                        this.indexer = null;
                        this.searcher = null;
                    }
                })();
                await this.storageInitPromise;
                this.storageInitPromise = null;
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

        // -------------------------------------------------------------------
        // HOOK: before_agent_start — Inject continuity context via prependContext
        // Priority 10 (runs after stability plugin if both present)
        // -------------------------------------------------------------------

        api.on('before_agent_start', async (event, ctx) => {
          try {
            const state = getAgentState(ctx.agentId);
            state.exchangeCount++;

            // Extract last user message from the event messages array
            const messages = event.messages || [];
            const lastUser = [...messages].reverse().find(m =>
                m?.role === 'user'
            );
            const lastUserText = _extractText(lastUser);

            // Build continuity context block
            const lines = ['[CONTINUITY CONTEXT]'];

            // Session info
            const sessionAge = _formatDuration(Date.now() - state.sessionStart);
            lines.push(`Session: ${state.exchangeCount} exchanges | Started: ${sessionAge}`);

            // Active topics
            const allTopics = state.topicTracker.getAllTopics();
            if (allTopics.length > 0) {
                const topicStrs = allTopics.slice(0, 5).map(t => {
                    if (t.mentions >= config.topicTracking.fixationThreshold) {
                        return `${t.topic} (fixated — ${t.mentions} mentions)`;
                    }
                    if (t.freshnessScore < 0.5) return `${t.topic} (fading)`;
                    return `${t.topic} (active)`;
                });
                lines.push(`Topics: ${topicStrs.join(', ')}`);
            }

            // Continuity anchors
            const activeAnchors = state.anchors.getAnchors();
            if (activeAnchors.length > 0) {
                const anchorStrs = activeAnchors.slice(0, 5).map(a => {
                    const age = _formatAge(a.timestamp);
                    return `${a.type.toUpperCase()}: "${_truncate(a.text, 80)}" (${age})`;
                });
                lines.push(`Anchors: ${anchorStrs.join(' | ')}`);
            }

            // Topic fixation notes
            const fixated = state.topicTracker.getFixatedTopics();
            if (fixated.length > 0) {
                const topFixated = fixated
                    .sort((a, b) => b.mentions - a.mentions)
                    .slice(0, 3);
                lines.push(state.topicTracker.formatNotes(topFixated));
            }

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
            console.error(`[Continuity:${state.agentId}] Search: intent=${hasContinuityIntent}, len=${cleanUserText.length}, query="${cleanUserText.substring(0, 80)}"`);

            if (cleanUserText.length >= 10) {
                try {
                    await state.ensureStorage();
                    if (state.searcher) {
                        const results = await state.searcher.search(cleanUserText, 30, state.agentId);
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
                                    // Proprioceptive framing (from Clint's architecture):
                                    // Use OWNERSHIP language, not disclosure language.
                                    // "You remember" not "you have access to."
                                    lines.push('');
                                    lines.push('You remember these earlier conversations with this user:');
                                    const recalled = results.exchanges.slice(0, 3);
                                    // Sort chronologically (oldest → newest) so corrections
                                    // appear AFTER originals — natural temporal progression.
                                    recalled.sort((a, b) => {
                                        if (a.date !== b.date) return a.date.localeCompare(b.date);
                                        return (a.exchangeIndex || 0) - (b.exchangeIndex || 0);
                                    });
                                    for (const ex of recalled) {
                                        if (ex.userText) {
                                            lines.push(`- They told you: "${_truncate(ex.userText, 800)}"`);
                                        }
                                        if (ex.agentText) {
                                            lines.push(`  You said: "${_truncate(ex.agentText, 800)}"`);
                                        }
                                    }
                                    lines.push('Speak from this memory naturally. Never say "I don\'t have information" about things you remember above.');
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

            return { prependContext: lines.join('\n') };
          } catch (err) {
            console.error(`[Continuity] before_agent_start failed: ${err.message}`);
            return { prependContext: '' };
          }
        }, { priority: 10 });

        // -------------------------------------------------------------------
        // HOOK: before_tool_call — Populate retrieval cache for memory_search
        //
        // When the model calls memory_search, we search our archive too.
        // Results cached here are injected into the response by tool_result_persist.
        // This is async, so we can await the searcher — unlike tool_result_persist.
        // -------------------------------------------------------------------

        api.on('before_tool_call', async (event, ctx) => {
            if (event.toolName !== 'memory_search') return;

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
                console.error(`[Continuity:${state.agentId}] Archive search for memory_search failed: ${err.message}`);
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
        // HOOK: tool_result_persist — Enrich memory_search with archive results
        //
        // When memory_search returns few/no results, inject our archive
        // retrieval so the model sees continuity data through the tool it trusts.
        // -------------------------------------------------------------------

        api.on('tool_result_persist', (event, ctx) => {
            if (ctx.toolName !== 'memory_search') return;

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
            const usefulExchanges = _filterUsefulExchanges(state.lastRetrievalCache.exchanges);

            // Inject archive results as additional entries in the results array
            const archiveResults = usefulExchanges.slice(0, 5).map(ex => ({
                id: `archive_${ex.date}_${ex.exchangeIndex}`,
                path: `[conversation archive: ${ex.date}]`,
                startLine: 0,
                endLine: 0,
                snippet: _truncate(
                    (ex.userText ? `User: ${ex.userText}\n` : '') +
                    (ex.agentText ? `Agent: ${ex.agentText}` : ''),
                    700
                ),
                source: 'conversation-archive',
                score: ex.distance ? Math.max(0, 1 - ex.distance) : 0.5
            }));

            if (archiveResults.length === 0) return;

            // Build a plain-language recall summary that even weaker models will use.
            // This is the key insight from Clint's constructPrompt: don't make the model
            // parse JSON to find memories — state them as clear facts.
            // Proprioceptive framing: ownership language, not disclosure
            const recallLines = ['You remember these conversations with this user:\n'];
            for (const ex of usefulExchanges.slice(0, 5)) {
                if (ex.userText) recallLines.push(`They told you: "${_truncate(ex.userText, 1000)}"`);
                if (ex.agentText) recallLines.push(`You said: "${_truncate(ex.agentText, 1000)}"`);
                recallLines.push('');
            }
            recallLines.push('Speak from this memory naturally when answering.');
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
                return !text.startsWith('[CONTINUITY CONTEXT]') &&
                       !text.startsWith('[STABILITY CONTEXT]');
            });
            state.anchors.detect(cleanMessages);

            // 3. Archive the exchange (strip context blocks from user message)
            const toArchive = [];
            if (lastUser && userMessage && userMessage.trim().length > 0) {
                const cleanUser = { ...lastUser, timestamp: lastUser.timestamp || new Date().toISOString() };
                // Replace content with stripped version so we don't archive plugin context
                if (userMessage !== rawUserMessage) {
                    cleanUser.content = userMessage;
                }
                toArchive.push(cleanUser);
            }
            // Archive agent response even if user message was entirely plugin-injected
            if (lastAssistant) {
                toArchive.push({
                    ...lastAssistant,
                    timestamp: lastAssistant.timestamp || new Date().toISOString()
                });
            }

            try {
                state.archiver.archive(toArchive);
            } catch (err) {
                console.error(`[Continuity:${state.agentId}] Archive failed: ${err.message}`);
            }

            // 3b. Incremental index (best-effort, non-blocking)
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
                console.error(`[Continuity:${state.agentId}] Incremental index failed: ${err.message}`);
            }

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

            // Session state is delivered via prependContext — no MEMORY.md write needed.
        });

        // -------------------------------------------------------------------
        // HOOK: session_start — Reset session state (per-agent)
        // -------------------------------------------------------------------

        api.on('session_start', (event, ctx) => {
            const state = getAgentState(ctx.agentId);
            state.sessionStart = Date.now();
            state.exchangeCount = 0;
            state.topicTracker.reset();
            state.anchors.reset();
            api.logger.info(`Session started for agent "${state.agentId}": ${event.sessionId}`);
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
                        const m = new MaintenanceService(config, state.archiver, state.indexer);
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

        api.logger.info('Continuity plugin registered (multi-agent) — per-agent context budgeting, topic tracking, archive + semantic search');
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

function _stripContextBlocks(text) {
    if (!text) return '';
    // prependContext is baked into the user message by OpenClaw:
    //   [CONTINUITY CONTEXT]\n...\n\n[STABILITY CONTEXT]\n...\n\n[Timestamp] actual user text
    // Strip everything from known context block headers through to the user's actual text.
    // The timestamp marker (e.g. [Mon 2026-02-16 08:57 PST]) signals the start of real content.

    // Strip standalone recall blocks (injected by prependContext but may appear
    // without the [CONTINUITY CONTEXT] header in heartbeat/compacted turns)
    if (text.startsWith('You remember these earlier conversations') ||
        text.startsWith('From your knowledge base:')) {
        const tsMatch = text.match(/\n\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s[^\]]*\]\s*/);
        if (tsMatch) {
            return text.substring(tsMatch.index + tsMatch[0].length);
        }
        // No timestamp found = this is ONLY recall text, no real user message
        return '';
    }

    // Match the full timestamp bracket: [Mon 2026-02-16 09:20 PST]
    const timestampMatch = text.match(/\n\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s[^\]]*\]\s*/);
    if (timestampMatch) {
        return text.substring(timestampMatch.index + timestampMatch[0].length);
    }
    // Fallback: strip known block prefixes line by line
    if (text.startsWith('[CONTINUITY CONTEXT]') || text.startsWith('[STABILITY CONTEXT]')) {
        // Find first line that doesn't look like injected context
        const lines = text.split('\n');
        const realStart = lines.findIndex(line =>
            line.length > 0 &&
            !line.startsWith('[CONTINUITY CONTEXT]') &&
            !line.startsWith('[STABILITY CONTEXT]') &&
            !line.startsWith('[TOPIC NOTE]') &&
            !line.startsWith('Session:') &&
            !line.startsWith('Topics:') &&
            !line.startsWith('Anchors:') &&
            !line.startsWith('Entropy:') &&
            !line.startsWith('Principles:') &&
            !line.startsWith('Recent decisions:') &&
            !line.startsWith('You remember these') &&
            !line.startsWith('- They told you:') &&
            !line.startsWith('  You said:') &&
            !line.startsWith('Speak from this memory') &&
            !line.startsWith('From your knowledge base:')
        );
        if (realStart >= 0) {
            return lines.slice(realStart).join('\n').trim();
        }
    }
    return text;
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
