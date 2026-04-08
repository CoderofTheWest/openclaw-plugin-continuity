/**
 * continuity_recall — Direct semantic search through archived conversations.
 *
 * Gives the agent active control over memory recall instead of relying
 * on passive injection via before_agent_start. The agent decides when
 * to search and what to look for.
 *
 * Uses the same searcher + noise filter as the automatic injection path.
 */

module.exports = function createRecallTool(getAgentState, filterFn, getCurrentAgentId) {
    return {
        name: 'continuity_recall',
        description: 'Search through your archived conversation history. Use this to recall specific past conversations, decisions, or details you discussed with the user. Returns relevant exchanges sorted by relevance and recency.',
        parameters: {
            type: 'object',
            properties: {
                query: {
                    type: 'string',
                    description: 'What to search for in conversation history (e.g., "sourdough recipe", "TonyPI walk calibration", "our discussion about consciousness")'
                },
                limit: {
                    type: 'integer',
                    description: 'Maximum results to return (default 5, max 15)',
                    default: 5
                },
                date_start: {
                    type: 'string',
                    description: 'Optional start date for search range (YYYY-MM-DD)'
                },
                date_end: {
                    type: 'string',
                    description: 'Optional end date for search range (YYYY-MM-DD)'
                }
            },
            required: ['query']
        },
        execute: async (toolCallId, args) => {
            const agentId = getCurrentAgentId();
            const state = getAgentState(agentId);

            try {
                await state.ensureStorage();
            } catch (err) {
                return { content: [{ type: 'text', text: `Memory system not available: ${err.message}` }] };
            }

            if (!state.searcher) {
                return { content: [{ type: 'text', text: 'Memory search not available (storage not initialized)' }] };
            }

            const limit = Math.min(Math.max(args.limit || 5, 1), 15);

            try {
                const results = await state.searcher.search(args.query, limit * 2, agentId);

                if (!results?.exchanges?.length) {
                    return { content: [{ type: 'text', text: `No archived conversations found matching "${args.query}"` }] };
                }

                // Apply noise filter
                const filtered = filterFn ? filterFn(results.exchanges) : results.exchanges;

                if (filtered.length === 0) {
                    return { content: [{ type: 'text', text: `Found matches for "${args.query}" but they were all meta-conversation noise. Try a more specific query.` }] };
                }

                // Format with proprioceptive framing
                const formatted = filtered.slice(0, limit).map(ex => {
                    const parts = [`[${ex.date || 'unknown date'}]`];
                    if (ex.userText) {
                        parts.push(`  They told you: ${_truncate(ex.userText, 600)}`);
                    }
                    if (ex.agentText) {
                        parts.push(`  You said: ${_truncate(ex.agentText, 600)}`);
                    }
                    return parts.join('\n');
                });

                return {
                    content: [{
                        type: 'text',
                        text: `You remember ${filtered.length} relevant conversation(s):\n\n${formatted.join('\n\n---\n\n')}`
                    }]
                };
            } catch (err) {
                return { content: [{ type: 'text', text: `Memory search failed: ${err.message}` }] };
            }
        }
    };
};

function _truncate(text, maxLen) {
    if (!text || text.length <= maxLen) return text || '';
    return text.substring(0, maxLen - 3) + '...';
}
