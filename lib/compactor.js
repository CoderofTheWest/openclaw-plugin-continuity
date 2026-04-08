/**
 * Compactor — Threshold-triggered context compression.
 *
 * Extracted from Clint's contextCompactor.js (244 lines).
 * Stripped of emergence-aware compression. Two strategies:
 *
 * 1. Conversational: Budget-aware selection + anchor preservation.
 *    Used for general conversation — compresses by tier priority,
 *    keeping recent turns and continuity anchors intact.
 *
 * 2. Task-aware: Prioritizes original user request, task state,
 *    tool results, and agent reasoning. Used when a task context
 *    message is present in the conversation.
 *
 * Triggers when estimated tokens exceed threshold (default 80%
 * of the token ceiling).
 */

class Compactor {
    /**
     * @param {object} config - full plugin config (reads compaction section)
     * @param {ContextBudget} contextBudget
     * @param {ContinuityAnchors} continuityAnchors
     * @param {TokenEstimator} tokenEstimator
     */
    constructor(config = {}, contextBudget, continuityAnchors, tokenEstimator) {
        const cc = config.compaction || config;
        this.threshold = cc.threshold || 0.80;
        this.fallbackMessages = cc.fallbackMessages || 20;
        this.taskAwareCompaction = cc.taskAwareCompaction !== false;

        this.contextBudget = contextBudget;
        this.continuityAnchors = continuityAnchors;
        this.tokenEstimator = tokenEstimator;
    }

    /**
     * Check whether compaction should trigger.
     *
     * @param {Array} messages
     * @param {number} [maxTokens] - override ceiling
     * @returns {boolean}
     */
    shouldCompact(messages, maxTokens) {
        const ceiling = maxTokens || this.tokenEstimator.getMaxTokens();
        return this.tokenEstimator.isOverBudget(messages, this.threshold);
    }

    /**
     * DANGEROUS: Compact messages to fit within the token budget.
     * Destructively removes conversation history — compacted messages cannot be recovered
     * from this function alone (archive separately if needed).
     *
     * @param {Array} messages
     * @param {number} [maxTokens] - override ceiling
     * @param {string} [_reason] - why compaction is needed
     * @returns {{ compactedMessages: Array, strategy: string, report: object }}
     */
    DANGEROUS_compact(messages, maxTokens, _reason = 'token budget exceeded') {
        if (!messages || messages.length === 0) {
            return { compactedMessages: [], strategy: 'none', report: {} };
        }

        const ceiling = maxTokens || this.tokenEstimator.getMaxTokens();

        // Detect task context
        const hasTaskContext = this.taskAwareCompaction && this._hasTaskContext(messages);

        let result;
        if (hasTaskContext) {
            result = this._taskAwareStrategy(messages, ceiling);
        } else {
            result = this._conversationalStrategy(messages, ceiling);
        }

        // Verify we're within budget; if not, use fallback
        if (this.tokenEstimator.isOverBudget(result.compactedMessages, 0.95)) {
            result = this._fallbackStrategy(messages);
        }

        return result;
    }

    /**
     * Conversational strategy: Use ContextBudget for intelligent selection
     * and preserve continuity anchors as a summary block.
     */
    _conversationalStrategy(messages, maxTokens) {
        // 1. Run context budget optimization
        const { optimizedMessages, tokenCount, budgetReport } =
            this.contextBudget.optimize(messages, maxTokens);

        // 2. Extract continuity anchors as a summary
        const anchors = this.continuityAnchors.detect(messages);
        const anchorBlock = this.continuityAnchors.format(anchors);

        // 3. Inject anchor summary as a system message if there are anchors
        const compacted = [...optimizedMessages];
        if (anchorBlock) {
            // Find system message and append, or create one
            const systemIdx = compacted.findIndex(m => m.role === 'system');
            if (systemIdx >= 0) {
                const systemMsg = compacted[systemIdx];
                compacted[systemIdx] = {
                    ...systemMsg,
                    content: this._extractText(systemMsg) + '\n\n' + anchorBlock
                };
            } else {
                compacted.unshift({
                    role: 'system',
                    content: anchorBlock
                });
            }
        }

        return {
            compactedMessages: compacted,
            strategy: 'conversational',
            report: {
                ...budgetReport,
                anchorsPreserved: anchors.length,
                originalMessages: messages.length,
                compactedMessages: compacted.length
            }
        };
    }

    /**
     * Task-aware strategy: Prioritize task-relevant content.
     *
     * Priority order:
     * 1. System message (always kept)
     * 2. Original user request (first user message — always kept)
     * 3. Task state / plan messages (if present)
     * 4. Tool results (last 15)
     * 5. Agent reasoning (last 5)
     */
    _taskAwareStrategy(messages, maxTokens) {
        const budget = Math.floor(maxTokens * this.contextBudget.budgetRatio);
        const compacted = [];
        let tokensUsed = 0;

        // 1. System message
        const systemMsg = messages.find(m => m.role === 'system');
        if (systemMsg) {
            compacted.push(systemMsg);
            tokensUsed += this.tokenEstimator.estimate(this._extractText(systemMsg));
        }

        // 2. First user message (the original request)
        const firstUser = messages.find(m => m.role === 'user');
        if (firstUser) {
            compacted.push(firstUser);
            tokensUsed += this.tokenEstimator.estimate(this._extractText(firstUser));
        }

        // 3. Messages with tool results (last 15)
        const toolMessages = messages.filter(m =>
            m.role === 'tool' || m.role === 'function' ||
            (m.content && typeof m.content === 'string' && m.content.includes('[tool_result]'))
        );
        const recentTools = toolMessages.slice(-15);
        for (const msg of recentTools) {
            const text = this._extractText(msg);
            const tokens = this.tokenEstimator.estimate(text);
            if (tokensUsed + tokens < budget * 0.7) {
                compacted.push(msg);
                tokensUsed += tokens;
            }
        }

        // 4. Recent agent reasoning (last 5 assistant messages)
        const assistantMessages = messages.filter(m => m.role === 'assistant');
        const recentAssistant = assistantMessages.slice(-5);
        for (const msg of recentAssistant) {
            if (compacted.includes(msg)) continue;
            const text = this._extractText(msg);
            const truncated = text.length > 1500 ? text.substring(0, 1500) + ' [...]' : text;
            const tokens = this.tokenEstimator.estimate(truncated);
            if (tokensUsed + tokens < budget * 0.9) {
                compacted.push({ ...msg, content: truncated });
                tokensUsed += tokens;
            }
        }

        // 5. Recent user messages (last 5)
        const userMessages = messages.filter(m => m.role === 'user');
        const recentUser = userMessages.slice(-5);
        for (const msg of recentUser) {
            if (compacted.includes(msg)) continue;
            const tokens = this.tokenEstimator.estimate(this._extractText(msg));
            if (tokensUsed + tokens < budget) {
                compacted.push(msg);
                tokensUsed += tokens;
            }
        }

        // Sort by original position
        const indexMap = new Map(messages.map((m, i) => [m, i]));
        compacted.sort((a, b) => (indexMap.get(a) || 0) - (indexMap.get(b) || 0));

        return {
            compactedMessages: compacted,
            strategy: 'task-aware',
            report: {
                budget,
                tokensUsed,
                originalMessages: messages.length,
                compactedMessages: compacted.length,
                toolMessagesKept: recentTools.length,
                assistantMessagesKept: recentAssistant.length
            }
        };
    }

    /**
     * Fallback: Keep system message + last N messages.
     */
    _fallbackStrategy(messages) {
        const systemMsg = messages.find(m => m.role === 'system');
        const recent = messages.slice(-this.fallbackMessages);
        const compacted = systemMsg ? [systemMsg, ...recent] : recent;

        return {
            compactedMessages: compacted,
            strategy: 'fallback',
            report: {
                originalMessages: messages.length,
                compactedMessages: compacted.length,
                keptLast: this.fallbackMessages
            }
        };
    }

    // ---------------------------------------------------------------
    // Progressive compression stages (March 31, 2026)
    // Inspired by Claude Code's multi-tier compaction architecture.
    // ---------------------------------------------------------------

    /**
     * Micro-compact: Strip old tool results, keeping only the last N.
     * Cheapest compression — no LLM call, just removes stale tool output.
     * Stage 2 in the cascade (80-90% budget).
     *
     * @param {Array} messages
     * @param {number} keepLastN - number of recent tool results to keep (default 5)
     * @returns {{ compactedMessages: Array, strategy: string, report: object }}
     */
    _microCompact(messages, keepLastN = 5) {
        const toolIndices = [];
        messages.forEach((m, i) => {
            if (m.role === 'tool' || m.role === 'function' ||
                (m.content && typeof m.content === 'string' && m.content.includes('[tool_result]'))) {
                toolIndices.push(i);
            }
        });

        // Keep the last N tool results, remove older ones
        const removeIndices = new Set(toolIndices.slice(0, -keepLastN));
        const compacted = messages.filter((_, i) => !removeIndices.has(i));

        return {
            compactedMessages: compacted,
            strategy: 'micro-compact',
            report: {
                originalMessages: messages.length,
                compactedMessages: compacted.length,
                toolResultsRemoved: removeIndices.size,
                toolResultsKept: Math.min(toolIndices.length, keepLastN)
            }
        };
    }

    /**
     * Snip-compact: Remove middle conversation segments, keeping first/last.
     * Stage 3 in the cascade (90-95% budget).
     * Preserves: system message, first 3 user messages, last 10 messages,
     * and any messages flagged as continuity anchors.
     *
     * @param {Array} messages
     * @param {number} maxTokens
     * @returns {{ compactedMessages: Array, strategy: string, report: object }}
     */
    _snipCompact(messages, maxTokens) {
        if (messages.length <= 15) {
            return { compactedMessages: [...messages], strategy: 'snip-compact', report: { noOp: true } };
        }

        const keep = new Set();

        // Always keep system message
        messages.forEach((m, i) => { if (m.role === 'system') keep.add(i); });

        // Keep first 3 user messages (original context)
        let userCount = 0;
        for (let i = 0; i < messages.length && userCount < 3; i++) {
            if (messages[i].role === 'user') { keep.add(i); userCount++; }
        }

        // Keep last 10 messages (recent context at full fidelity)
        for (let i = Math.max(0, messages.length - 10); i < messages.length; i++) {
            keep.add(i);
        }

        // Keep continuity anchors
        if (this.continuityAnchors) {
            const anchors = this.continuityAnchors.detect(messages);
            for (const anchor of anchors) {
                if (anchor.messageIndex !== undefined) keep.add(anchor.messageIndex);
            }
        }

        const compacted = messages.filter((_, i) => keep.has(i));

        // Insert a snip marker where the gap is
        const snipMarker = {
            role: 'system',
            content: `[CONTEXT SNIPPED — ${messages.length - compacted.length} messages compressed. Continuity anchors preserved. Ask if you need details from the snipped section.]`
        };

        // Find insertion point (after the first kept block, before the last block)
        const firstKeptInLastBlock = Math.max(0, messages.length - 10);
        const insertIdx = compacted.findIndex((_, i) => {
            const origIdx = messages.indexOf(compacted[i]);
            return origIdx >= firstKeptInLastBlock;
        });

        if (insertIdx > 0) {
            compacted.splice(insertIdx, 0, snipMarker);
        }

        return {
            compactedMessages: compacted,
            strategy: 'snip-compact',
            report: {
                originalMessages: messages.length,
                compactedMessages: compacted.length,
                messagesSnipped: messages.length - compacted.length + 1, // +1 for marker
                anchorsPreserved: keep.size
            }
        };
    }

    /**
     * Pre-compaction cascade — lighter compression before full gateway compaction.
     * Runs micro-compact then snip-compact. Does NOT duplicate the after_compaction
     * LLM summarization (Tier 2) or session handoff — those handle post-compaction.
     *
     * @param {Array} messages
     * @param {number} maxTokens
     * @returns {{ compactedMessages: Array, strategy: string, report: object }}
     */
    preCompact(messages, maxTokens) {
        const ceiling = maxTokens || this.tokenEstimator.getMaxTokens();

        // Stage 1: Check if any pre-compaction is needed
        if (!this.tokenEstimator.isOverBudget(messages, 0.80)) {
            return { compactedMessages: [...messages], strategy: 'none', report: { stage: 1 } };
        }

        // Stage 2: Micro-compact (strip old tool results — cheapest)
        let result = this._microCompact(messages);
        if (!this.tokenEstimator.isOverBudget(result.compactedMessages, 0.90)) {
            result.report.stage = 2;
            return result;
        }

        // Stage 3: Snip-compact (remove middle segments, keep anchors)
        result = this._snipCompact(result.compactedMessages, ceiling);
        result.report.stage = 3;
        return result;
    }

    // ---------------------------------------------------------------
    // Internal
    // ---------------------------------------------------------------

    /**
     * Detect if the conversation contains task context.
     * Looks for tool messages, function calls, or task-related content.
     */
    _hasTaskContext(messages) {
        return messages.some(m =>
            m.role === 'tool' ||
            m.role === 'function' ||
            (m.tool_calls && m.tool_calls.length > 0) ||
            (m.function_call)
        );
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

module.exports = Compactor;
