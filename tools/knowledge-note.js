/**
 * knowledge_note — Capture a learning mid-session for future reference.
 *
 * Gives the agent an explicit path to record operational knowledge.
 * Entries are immediately indexed (embedded + FTS) and will surface
 * in future sessions via passive injection when relevant.
 *
 * Complements the automatic workspace-scan path: workspace files are
 * indexed at session_start, but knowledge_note captures real-time
 * learnings that haven't been written to a file yet.
 */

module.exports = function createKnowledgeNoteTool(getAgentState, getCurrentAgentId) {
    return {
        name: 'knowledge_note',
        description: 'Record something you\'ve learned for future reference. This gets indexed and will surface in future sessions when the topic is relevant. Use this to capture operational knowledge, debugging insights, tool patterns, or anything worth remembering.',
        parameters: {
            type: 'object',
            properties: {
                content: {
                    type: 'string',
                    description: 'What you learned — be concise and specific (e.g., "climb_stairs action chains into high-step gait for obstacle clearance")'
                },
                topic: {
                    type: 'string',
                    description: 'Primary topic (e.g., "embodiment", "navigation", "debugging", "architecture"). Optional — will be inferred if not provided.'
                }
            },
            required: ['content']
        },
        execute: async (toolCallId, args) => {
            const agentId = getCurrentAgentId();
            const state = getAgentState(agentId);

            try {
                await state.ensureStorage();
            } catch (err) {
                return { content: [{ type: 'text', text: `Knowledge system not available: ${err.message}` }] };
            }

            if (!state.knowledgeIndexer) {
                return { content: [{ type: 'text', text: 'Knowledge indexer not available (knowledge system not enabled)' }] };
            }

            if (!args.content || args.content.trim().length < 10) {
                return { content: [{ type: 'text', text: 'Knowledge note too short — be specific about what you learned.' }] };
            }

            try {
                const id = await state.knowledgeIndexer.indexEntry(
                    agentId,
                    args.content.trim(),
                    args.topic || null,
                    'tool:knowledge_note'
                );

                const topicLabel = args.topic ? ` [${args.topic}]` : '';
                return {
                    content: [{
                        type: 'text',
                        text: `Knowledge noted${topicLabel}. This will surface in future sessions when relevant. (id: ${id})`
                    }]
                };
            } catch (err) {
                return { content: [{ type: 'text', text: `Failed to store knowledge note: ${err.message}` }] };
            }
        }
    };
};
