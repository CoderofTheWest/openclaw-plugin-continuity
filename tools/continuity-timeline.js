/**
 * continuity_timeline — Summary timeline with drill-down.
 *
 * Shows hierarchical summaries about a topic over time.
 * The agent can drill into any summary to see its children,
 * tracing from high-level overview down to specific exchanges.
 */

module.exports = function createTimelineTool(getAgentState, getCurrentAgentId) {
    return {
        name: 'continuity_timeline',
        description: 'View a timeline of conversation summaries about a topic, with the ability to drill down into specific periods. Use this to understand the evolution of a topic over time without loading full conversation history.',
        parameters: {
            type: 'object',
            properties: {
                topic: {
                    type: 'string',
                    description: 'Topic to trace through conversation history (e.g., "embodiment", "TonyPI", "identity")'
                },
                level: {
                    type: 'string',
                    enum: ['overview', 'detailed'],
                    description: 'Level of detail: "overview" shows branch/root summaries, "detailed" shows leaf summaries',
                    default: 'overview'
                },
                expand_summary_id: {
                    type: 'string',
                    description: 'Drill into a specific summary by ID to see its children. Get IDs from previous timeline results.'
                },
                date_start: {
                    type: 'string',
                    description: 'Optional start date (YYYY-MM-DD)'
                },
                date_end: {
                    type: 'string',
                    description: 'Optional end date (YYYY-MM-DD)'
                }
            },
            required: ['topic']
        },
        execute: async (toolCallId, args) => {
            const agentId = getCurrentAgentId();
            const state = getAgentState(agentId);

            try {
                await state.ensureStorage();
            } catch (err) {
                return { content: [{ type: 'text', text: `Summary system not available: ${err.message}` }] };
            }

            if (!state.summaryStore) {
                return { content: [{ type: 'text', text: 'Summary timeline not available. Summaries are generated during conversation compaction — keep talking and they will appear.' }] };
            }

            // Drill-down mode: expand a specific summary
            if (args.expand_summary_id) {
                const children = state.summaryStore.getChildren(args.expand_summary_id);
                if (children.length === 0) {
                    // Try to show the summary itself
                    const self = state.summaryStore.getSummary(args.expand_summary_id);
                    if (self) {
                        return {
                            content: [{
                                type: 'text',
                                text: _formatSummary(self) + '\n\n(This is a leaf summary — no further drill-down available. Use continuity_recall to search the original messages.)'
                            }]
                        };
                    }
                    return { content: [{ type: 'text', text: `No summary found with ID "${args.expand_summary_id}"` }] };
                }

                const formatted = children.map(_formatSummary).join('\n\n---\n\n');
                return {
                    content: [{
                        type: 'text',
                        text: `Expanded ${args.expand_summary_id} — ${children.length} sub-summaries:\n\n${formatted}`
                    }]
                };
            }

            // Timeline mode: search summaries by topic
            const dateRange = {};
            if (args.date_start) dateRange.start = args.date_start;
            if (args.date_end) dateRange.end = args.date_end;

            const summaries = state.summaryStore.getTimelineSummaries(
                agentId,
                args.topic,
                Object.keys(dateRange).length > 0 ? dateRange : null
            );

            if (summaries.length === 0) {
                return {
                    content: [{
                        type: 'text',
                        text: `No conversation summaries found for topic "${args.topic}". Summaries are generated when conversations are compacted. If this is a recent topic, it may not have been summarized yet.`
                    }]
                };
            }

            // Filter by level if requested
            let filtered = summaries;
            if (args.level === 'overview') {
                // Prefer higher-level summaries; fall back to leaves if no branches exist
                const higher = summaries.filter(s => s.level >= 1);
                filtered = higher.length > 0 ? higher : summaries;
            }

            const formatted = filtered.map(_formatSummary).join('\n\n---\n\n');

            return {
                content: [{
                    type: 'text',
                    text: `Timeline for "${args.topic}" — ${filtered.length} period(s):\n\n${formatted}\n\nUse expand_summary_id to drill into any summary above.`
                }]
            };
        }
    };
};

function _formatSummary(s) {
    const period = s.dateRangeStart === s.dateRangeEnd
        ? s.dateRangeStart
        : `${s.dateRangeStart} to ${s.dateRangeEnd}`;
    const levelLabel = s.level === 0 ? 'leaf' : s.level === 1 ? 'branch' : 'root';
    const topicsStr = s.topics?.length > 0 ? `\nTopics: ${s.topics.join(', ')}` : '';

    return `**${period}** (${s.messageCount} messages, ${levelLabel})\nID: ${s.id}${topicsStr}\n${s.summaryText}`;
}
