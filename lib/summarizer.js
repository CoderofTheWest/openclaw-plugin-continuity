/**
 * Summarizer — Two-tier summarization for compacted conversations.
 *
 * Tier 1 (Extractive): Fast, deterministic, no LLM. Runs synchronously in
 *   after_compaction. Extracts key sentences, preserves anchors and topics.
 *
 * Tier 2 (Abstractive): LLM-based, async. Queued for processing by
 *   MaintenanceService. Only triggered for high-entropy conversations.
 *
 * DAG condensation: When uncondensed leaf summaries accumulate past a
 * threshold, condenses them into branch summaries. Same at branch→root.
 *
 * LLM calling pattern adapted from openclaw-plugin-contemplation/lib/reflect.js.
 */

/**
 * Detect API format from endpoint URL.
 */
function detectFormat(endpoint) {
    if (/\/api\/(generate|chat)\b/.test(endpoint)) return 'ollama';
    return 'openai';
}

/**
 * Call an LLM endpoint (OpenAI-compatible or Ollama native).
 */
async function callLLM({ endpoint, model, prompt, temperature, maxTokens, timeoutMs, apiKey, format }) {
    const resolvedFormat = format || detectFormat(endpoint);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs || 30000);

    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
    }

    try {
        let body;
        if (resolvedFormat === 'ollama') {
            body = JSON.stringify({
                model,
                prompt,
                stream: false,
                options: { temperature, num_predict: maxTokens }
            });
        } else {
            body = JSON.stringify({
                model,
                messages: [{ role: 'user', content: prompt }],
                temperature,
                max_tokens: maxTokens,
                stream: false
            });
        }

        const res = await fetch(endpoint, {
            method: 'POST',
            headers,
            body,
            signal: controller.signal
        });

        if (!res.ok) {
            const errText = await res.text().catch(() => '');
            throw new Error(`LLM request failed (${res.status}): ${errText.substring(0, 200)}`);
        }

        const payload = await res.json();

        if (resolvedFormat === 'ollama') {
            if (typeof payload?.response === 'string' && payload.response.trim()) {
                return payload.response.trim();
            }
            throw new Error('Ollama response missing "response" text');
        } else {
            const text = payload?.choices?.[0]?.message?.content;
            if (typeof text === 'string' && text.trim()) {
                return text.trim();
            }
            throw new Error('LLM response missing choices[0].message.content');
        }
    } finally {
        clearTimeout(timer);
    }
}

class Summarizer {
    /**
     * @param {object} config - full plugin config
     * @param {object} summaryStore - SummaryStore instance
     * @param {object} embeddingProvider - shared EmbeddingProvider
     */
    constructor(config = {}, summaryStore, embeddingProvider = null) {
        this.config = config.summarization || {};
        this.summaryStore = summaryStore;
        this._embeddingFn = embeddingProvider;
    }

    // -------------------------------------------------------------------
    // Tier 1: Extractive summary (fast, no LLM)
    // -------------------------------------------------------------------

    /**
     * Generate a fast extractive summary from compacted messages.
     * Extracts key sentences by scoring words against topic and anchor relevance.
     *
     * @param {Array} messages - the compacted messages
     * @param {Array} anchorState - active anchors at compaction time
     * @param {Array} topicState - active topics at compaction time
     * @returns {{ text: string, topics: string[], anchors: string[], messageCount: number }}
     */
    extractiveSummary(messages, anchorState = [], topicState = []) {
        const maxSentences = this.config.extractiveMaxSentences || 8;

        // Collect all text from messages
        const texts = messages
            .map(m => _extractText(m))
            .filter(t => t && t.length > 10);

        if (texts.length === 0) {
            return { text: '(empty compaction)', topics: [], anchors: [], messageCount: 0 };
        }

        // Build term importance from topics
        const topicTerms = new Set(
            (topicState || []).map(t => (t.topic || t).toLowerCase())
        );

        // Score sentences
        const allSentences = [];
        for (const text of texts) {
            const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.length > 15);
            for (const sentence of sentences) {
                const words = sentence.toLowerCase().split(/\s+/);
                let score = 0;

                // Topic term hits
                for (const w of words) {
                    if (topicTerms.has(w)) score += 2;
                }

                // Length bonus (prefer substantive sentences)
                if (words.length > 8 && words.length < 40) score += 1;

                // Question/decision indicators
                if (/\b(decided|agreed|should|because|therefore|fixed|changed|discovered)\b/i.test(sentence)) {
                    score += 2;
                }

                allSentences.push({ sentence: sentence.trim(), score });
            }
        }

        // Sort by score, take top N
        allSentences.sort((a, b) => b.score - a.score);
        const keySentences = allSentences
            .slice(0, maxSentences)
            .map(s => s.sentence);

        // Extract preserved anchors
        const anchorTexts = (anchorState || [])
            .slice(0, 3)
            .map(a => `[${a.type || 'anchor'}] ${_truncate(a.text || '', 150)}`);

        // Extract topic list
        const topics = [...topicTerms].slice(0, 10);

        // Build summary text
        const parts = [];
        if (topics.length > 0) {
            parts.push(`Topics: ${topics.join(', ')}`);
        }
        if (anchorTexts.length > 0) {
            parts.push(`Anchors: ${anchorTexts.join(' | ')}`);
        }
        parts.push(`Key points: ${keySentences.join(' ')}`);

        return {
            text: parts.join('\n'),
            topics,
            anchors: anchorTexts,
            messageCount: messages.length
        };
    }

    // -------------------------------------------------------------------
    // Tier 2: Abstractive summary (async, LLM-based)
    // -------------------------------------------------------------------

    /**
     * Generate an LLM-based abstractive summary.
     * Called asynchronously from the maintenance cycle.
     *
     * @param {Array} messages - raw messages to summarize
     * @param {string} extractiveText - the Tier 1 extractive summary for reference
     * @param {number} entropyScore - entropy at compaction time
     * @returns {string} LLM-generated summary text
     */
    async abstractiveSummary(messages, extractiveText, entropyScore = 0) {
        const llmConfig = this.config.llm || {};
        const endpoint = llmConfig.endpoint || 'http://127.0.0.1:11434/v1/chat/completions';

        // Build conversation text (truncated for token budget)
        const conversationText = messages
            .map(m => {
                const text = _extractText(m);
                const sender = m.sender || m.role || 'unknown';
                return `${sender}: ${_truncate(text, 500)}`;
            })
            .join('\n');

        const entropyNote = entropyScore > 0.6
            ? '\nThis period contained significant cognitive tension. Preserve any moments of contradiction resolution, identity assertion, or key decisions.'
            : '';

        const prompt = [
            'Summarize this conversation segment for long-term memory. Preserve:',
            '- Key decisions and their reasoning',
            '- Identity moments (assertions, corrections, tensions)',
            '- Technical details needed to resume work',
            '- Emotional tone and relational dynamics',
            '',
            `Entropy during this period: ${entropyScore.toFixed(2)}${entropyNote}`,
            '',
            `Extractive summary for reference:\n${extractiveText}`,
            '',
            `Conversation (${messages.length} messages):`,
            conversationText,
            '',
            'Write a concise summary (2-4 paragraphs). Use first-person perspective where appropriate ("We discussed...", "The user asked me to...").'
        ].join('\n');

        return callLLM({
            endpoint,
            model: llmConfig.model || null,
            prompt,
            temperature: llmConfig.temperature ?? 0.3,
            maxTokens: llmConfig.maxTokens ?? 500,
            timeoutMs: llmConfig.timeoutMs ?? 30000,
            apiKey: llmConfig.apiKey || null,
            format: llmConfig.format || null
        });
    }

    // -------------------------------------------------------------------
    // DAG condensation
    // -------------------------------------------------------------------

    /**
     * Condense child summaries into a parent summary.
     * @param {Array} children - child summary objects
     * @param {number} level - target level for the new parent
     * @returns {string} condensed summary text
     */
    async condenseSummaries(children, level) {
        const llmConfig = this.config.llm || {};
        const endpoint = llmConfig.endpoint || 'http://127.0.0.1:11434/v1/chat/completions';

        const childTexts = children.map((c, i) => {
            const period = c.dateRangeStart === c.dateRangeEnd
                ? c.dateRangeStart
                : `${c.dateRangeStart} to ${c.dateRangeEnd}`;
            return `[${period}] (${c.messageCount} messages)\n${c.summaryText}`;
        }).join('\n\n---\n\n');

        const levelName = level === 1 ? 'daily' : level === 2 ? 'weekly' : `level-${level}`;

        const prompt = [
            `Create a ${levelName} summary from these ${children.length} conversation summaries.`,
            'Preserve the most important themes, decisions, identity moments, and technical context.',
            'Be concise — this summary represents a longer period and should highlight only what matters for continuity.',
            '',
            'Source summaries:',
            childTexts,
            '',
            `Write a concise ${levelName} summary (1-2 paragraphs).`
        ].join('\n');

        return callLLM({
            endpoint,
            model: llmConfig.model || null,
            prompt,
            temperature: llmConfig.temperature ?? 0.3,
            maxTokens: llmConfig.maxTokens ?? (level >= 2 ? 800 : 500),
            timeoutMs: llmConfig.timeoutMs ?? 30000,
            apiKey: llmConfig.apiKey || null,
            format: llmConfig.format || null
        });
    }

    // -------------------------------------------------------------------
    // Queue processing (called by MaintenanceService)
    // -------------------------------------------------------------------

    /**
     * Process pending items from the summarization queue.
     * @param {number} limit - max items to process per cycle
     * @returns {{ processed: number, errors: number }}
     */
    async processQueue(limit = 3) {
        const items = this.summaryStore.dequeue(limit);
        let processed = 0;
        let errors = 0;

        for (const item of items) {
            try {
                const messages = JSON.parse(item.messages_json);

                // Find the existing extractive summary for this compaction
                const existingSummaries = this.summaryStore.getTimelineSummaries(
                    item.agent_id,
                    '', // no topic filter
                    { start: item.compaction_date, end: item.compaction_date }
                ).filter(s => s.level === 0);

                const extractiveText = existingSummaries.length > 0
                    ? existingSummaries[existingSummaries.length - 1].summaryText
                    : '';

                // Generate abstractive summary
                const richSummary = await this.abstractiveSummary(
                    messages,
                    extractiveText,
                    item.entropy_score || 0
                );

                // Update the most recent leaf summary for this date with the richer text
                if (existingSummaries.length > 0) {
                    const target = existingSummaries[existingSummaries.length - 1];
                    target.summaryText = richSummary;
                    target.metadata = { ...target.metadata, strategy: 'abstractive', enrichedAt: new Date().toISOString() };
                    await this.summaryStore.storeSummary(target);
                }

                this.summaryStore.markCompleted(item.id);
                processed++;
            } catch (err) {
                console.warn(`[Summarizer] Queue item ${item.id} failed: ${err.message}`);
                this.summaryStore.markFailed(item.id, err.message);
                errors++;
            }
        }

        return { processed, errors };
    }

    /**
     * Check if condensation is needed and perform it.
     * Runs after queue processing in the maintenance cycle.
     *
     * @param {string} agentId
     */
    async maybeCondense(agentId) {
        const maxDepth = this.config.maxDepth || 3;

        // Level 0 → 1 condensation (leaves → branches)
        if (maxDepth >= 2) {
            const leaves = this.summaryStore.getUncondensedLeaves(agentId);
            if (leaves.length > 0) {
                await this._condenseGroup(agentId, leaves, 1);
            }
        }

        // Level 1 → 2 condensation (branches → roots)
        if (maxDepth >= 3) {
            const branches = this.summaryStore.getUncondensedBranches(agentId);
            if (branches.length > 0) {
                await this._condenseGroup(agentId, branches, 2);
            }
        }
    }

    /**
     * Condense a group of summaries into a parent.
     * @param {string} agentId
     * @param {Array} children
     * @param {number} targetLevel
     */
    async _condenseGroup(agentId, children, targetLevel) {
        try {
            const condensedText = await this.condenseSummaries(children, targetLevel);

            const dateStart = children[0].dateRangeStart;
            const dateEnd = children[children.length - 1].dateRangeEnd;
            const totalMessages = children.reduce((sum, c) => sum + (c.messageCount || 0), 0);
            const avgEntropy = children.reduce((sum, c) => sum + (c.entropyAvg || 0), 0) / children.length;

            // Merge all topics from children
            const allTopics = [...new Set(children.flatMap(c => c.topics || []))];

            const parentId = `summary_${agentId}_${dateStart}_${Date.now()}_${targetLevel}`;

            await this.summaryStore.storeSummary({
                id: parentId,
                level: targetLevel,
                parentId: null,
                agentId,
                dateRangeStart: dateStart,
                dateRangeEnd: dateEnd,
                messageCount: totalMessages,
                summaryText: condensedText,
                topics: allTopics,
                anchors: [],
                entropyAvg: avgEntropy,
                metadata: { strategy: 'condensed', childCount: children.length }
            });

            // Link children to parent
            for (const child of children) {
                child.parentId = parentId;
                await this.summaryStore.storeSummary(child);
            }

            console.log(`[Summarizer] Condensed ${children.length} level-${targetLevel - 1} summaries into ${parentId}`);
        } catch (err) {
            console.warn(`[Summarizer] Condensation failed for ${agentId} level ${targetLevel}: ${err.message}`);
        }
    }
}

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

function _extractText(msg) {
    if (!msg) return '';
    if (typeof msg === 'string') return msg;
    if (typeof msg.text === 'string') return msg.text;
    if (typeof msg.content === 'string') return msg.content;
    if (Array.isArray(msg.content)) {
        return msg.content
            .filter(c => c.type === 'text')
            .map(c => c.text)
            .join('\n');
    }
    return '';
}

function _truncate(text, maxLen) {
    if (!text || text.length <= maxLen) return text || '';
    return text.substring(0, maxLen - 3) + '...';
}

module.exports = Summarizer;
