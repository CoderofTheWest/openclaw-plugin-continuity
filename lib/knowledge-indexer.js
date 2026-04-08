/**
 * KnowledgeIndexer — Parses workspace .md files into discrete knowledge entries.
 *
 * Scans all .md files in the agent's workspace directory, splits them into
 * sections and bullets, deduplicates by content hash, and stores entries
 * in the KnowledgeStore.
 *
 * Chunking strategy (bullet-level granularity):
 *   1. Split by ## and ### markdown headers
 *   2. Within each section, split at bullet/paragraph boundaries
 *   3. Short sections (<minSectionCharsForSplit) stay atomic
 *   4. Section path preserved for traceability
 *
 * Deduplication:
 *   - SHA-256 hash of content (normalized whitespace)
 *   - Unchanged sections skipped on re-index
 *   - Changed content updates the existing entry + re-embeds
 *   - Deleted source sections are NOT auto-removed (knowledge persists)
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

class KnowledgeIndexer {
    /**
     * @param {object} knowledgeStore - KnowledgeStore instance
     * @param {object} config - full plugin config
     * @param {object} embeddingProvider - shared EmbeddingProvider
     * @param {string} workspacePath - absolute path to workspace directory
     */
    constructor(knowledgeStore, config = {}, embeddingProvider = null, workspacePath = null) {
        this.store = knowledgeStore;
        this.config = config.knowledge || {};
        this._embeddingFn = embeddingProvider;
        this.workspacePath = workspacePath;

        // Config
        this.skipFiles = new Set((this.config.skipFiles || [
            'SOUL.md', 'BOOTSTRAP.md', 'IDENTITY.md', 'CHRIS.md', 'USER.md', 'MEMORY.md'
        ]).map(f => f.toLowerCase()));

        this.skipSections = new Set((this.config.skipSections || [
            'First Run', 'Every Session', 'Memory', 'Recovery Protocol',
            'Your Workspace', 'Safety', 'Task Transparency',
            'Default Tool Usage', 'Shell Access', 'Agent Mode Rules',
            'Make It Yours', 'Office Hours'
        ]).map(s => s.toLowerCase()));

        this.bulletLevelChunking = this.config.bulletLevelChunking !== false;
        this.minSectionCharsForSplit = this.config.minSectionCharsForSplit || 200;
    }

    /**
     * Index all workspace .md files for an agent.
     *
     * @param {string} agentId
     * @returns {{ indexed: number, updated: number, skipped: number, errors: number }}
     */
    async indexWorkspace(agentId) {
        if (!this.workspacePath) {
            return { indexed: 0, updated: 0, skipped: 0, errors: 0, reason: 'no workspace path' };
        }

        const stats = { indexed: 0, updated: 0, skipped: 0, errors: 0 };

        let files;
        try {
            files = fs.readdirSync(this.workspacePath)
                .filter(f => f.endsWith('.md') && !this.skipFiles.has(f.toLowerCase()));
        } catch (e) {
            return { ...stats, errors: 1, reason: `readdir failed: ${e.message}` };
        }

        for (const file of files) {
            try {
                const filePath = path.join(this.workspacePath, file);
                const content = fs.readFileSync(filePath, 'utf-8');
                const chunks = this._parseMarkdownSections(content, file);

                for (const chunk of chunks) {
                    if (this._shouldSkip(chunk.sectionPath, file)) {
                        stats.skipped++;
                        continue;
                    }

                    // Skip very short content (headers only, empty sections)
                    if (chunk.content.trim().length < 20) {
                        stats.skipped++;
                        continue;
                    }

                    const hash = this._hashContent(chunk.content);

                    // Check for existing entry with same source hash (content unchanged)
                    const existing = this.store.findBySourceHash(agentId, hash);
                    if (existing) {
                        stats.skipped++;
                        continue;
                    }

                    // Check for existing entry at same section path (content changed)
                    const atPath = this.store.findBySectionPath(agentId, chunk.sectionPath);
                    if (atPath) {
                        await this.store.update(atPath.id, {
                            content: chunk.content,
                            topic: chunk.topic,
                            sectionPath: chunk.sectionPath,
                            sourceHash: hash,
                            metadata: chunk.metadata
                        });
                        stats.updated++;
                        continue;
                    }

                    // New entry
                    await this.store.store({
                        agentId,
                        content: chunk.content,
                        topic: chunk.topic,
                        sectionPath: chunk.sectionPath,
                        sourceType: 'workspace',
                        sourceHash: hash,
                        metadata: chunk.metadata
                    });
                    stats.indexed++;
                }
            } catch (e) {
                stats.errors++;
                console.warn(`[KnowledgeIndexer] Error indexing ${file}: ${e.message}`);
            }
        }

        return stats;
    }

    /**
     * Directly index a knowledge entry (for knowledge_note tool).
     *
     * @param {string} agentId
     * @param {string} content
     * @param {string} [topic]
     * @param {string} [source] - e.g., 'tool:knowledge_note'
     * @returns {string} entry ID
     */
    async indexEntry(agentId, content, topic = null, source = 'tool:knowledge_note') {
        const hash = this._hashContent(content);

        // Dedup: skip if identical content already exists
        const existing = this.store.findBySourceHash(agentId, hash);
        if (existing) {
            return existing.id;
        }

        return await this.store.store({
            agentId,
            content,
            topic: topic || this._inferTopic(null, content),
            sectionPath: null,
            sourceType: source.startsWith('tool:') ? 'tool' : source,
            sourceHash: hash,
            metadata: { capturedAt: new Date().toISOString() }
        });
    }

    // ─── Markdown Parsing ───────────────────────────────────────────

    /**
     * Parse a markdown file into knowledge chunks.
     *
     * Strategy:
     *   - Split by ## and ### headers
     *   - Within sections, split at bullet boundaries if section is long enough
     *   - Each chunk gets a section_path for traceability
     *
     * @param {string} content - file content
     * @param {string} filename - e.g., 'AGENTS.md'
     * @returns {Array<{ content, topic, sectionPath, metadata }>}
     */
    _parseMarkdownSections(content, filename) {
        const lines = content.split('\n');
        const chunks = [];
        let currentH2 = null;
        let currentH3 = null;
        let sectionLines = [];
        let sectionStartLine = 0;

        const flushSection = () => {
            if (sectionLines.length === 0) return;

            const sectionContent = sectionLines.join('\n').trim();
            if (!sectionContent) {
                sectionLines = [];
                return;
            }

            const sectionName = currentH3 || currentH2 || filename;
            const sectionPath = this._buildSectionPath(filename, currentH2, currentH3);
            const topic = this._inferTopic(sectionPath, sectionContent);

            // Bullet-level chunking for longer sections
            if (this.bulletLevelChunking && sectionContent.length > this.minSectionCharsForSplit) {
                const bulletChunks = this._splitIntoBullets(sectionContent, sectionPath, topic);
                if (bulletChunks.length > 1) {
                    chunks.push(...bulletChunks);
                    sectionLines = [];
                    return;
                }
            }

            // Atomic section
            chunks.push({
                content: sectionContent,
                topic,
                sectionPath,
                metadata: { filename, h2: currentH2, h3: currentH3 }
            });

            sectionLines = [];
        };

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            // ## Header — start new top-level section
            if (/^##\s+/.test(line) && !/^###/.test(line)) {
                flushSection();
                currentH2 = line.replace(/^##\s+/, '').replace(/\s*\(.*\)\s*$/, '').trim();
                currentH3 = null;
                sectionStartLine = i;
                continue;
            }

            // ### Header — start new subsection
            if (/^###\s+/.test(line)) {
                flushSection();
                currentH3 = line.replace(/^###\s+/, '').replace(/\s*[:—]\s*$/, '').trim();
                sectionStartLine = i;
                continue;
            }

            // # Top-level header — skip (file title, not knowledge)
            if (/^#\s+/.test(line) && !/^##/.test(line)) {
                flushSection();
                currentH2 = null;
                currentH3 = null;
                continue;
            }

            sectionLines.push(line);
        }

        flushSection();
        return chunks;
    }

    /**
     * Split a section into bullet-level chunks.
     * Groups of related bullets (same indentation block) stay together.
     *
     * @param {string} content
     * @param {string} sectionPath
     * @param {string} topic
     * @returns {Array<{ content, topic, sectionPath, metadata }>}
     */
    _splitIntoBullets(content, sectionPath, topic) {
        const lines = content.split('\n');
        const groups = [];
        let currentGroup = [];
        let currentLabel = null;

        for (const line of lines) {
            const trimmed = line.trim();

            // New top-level bullet or bold label
            if (/^[-*]\s+\*\*/.test(trimmed) || /^[-*]\s+\S/.test(trimmed) && !trimmed.startsWith('  ')) {
                // Flush previous group if substantial
                if (currentGroup.length > 0 && currentGroup.join('\n').trim().length > 30) {
                    groups.push({
                        content: currentGroup.join('\n').trim(),
                        label: currentLabel
                    });
                }
                currentGroup = [line];
                // Extract label from bold text
                const boldMatch = trimmed.match(/\*\*([^*]+)\*\*/);
                currentLabel = boldMatch ? boldMatch[1] : null;
            } else if (trimmed === '' && currentGroup.length > 0) {
                // Empty line: flush if group is substantial
                if (currentGroup.join('\n').trim().length > 30) {
                    groups.push({
                        content: currentGroup.join('\n').trim(),
                        label: currentLabel
                    });
                    currentGroup = [];
                    currentLabel = null;
                }
            } else {
                // Continuation line (indented sub-bullets, text)
                currentGroup.push(line);
            }
        }

        // Flush final group
        if (currentGroup.length > 0 && currentGroup.join('\n').trim().length > 30) {
            groups.push({
                content: currentGroup.join('\n').trim(),
                label: currentLabel
            });
        }

        // Only return bullet chunks if we got multiple meaningful groups
        if (groups.length <= 1) return [];

        return groups.map((g, i) => ({
            content: g.content,
            topic,
            sectionPath: g.label ? `${sectionPath} > ${g.label}` : `${sectionPath} [${i + 1}]`,
            metadata: { bulletGroup: i, label: g.label }
        }));
    }

    // ─── Helpers ─────────────────────────────────────────────────────

    _buildSectionPath(filename, h2, h3) {
        const parts = [filename];
        if (h2) parts.push(h2);
        if (h3) parts.push(h3);
        return parts.join(' > ');
    }

    _inferTopic(sectionPath, content) {
        if (!sectionPath) {
            // Infer from content keywords
            const lower = content.toLowerCase();
            if (lower.includes('embodi') || lower.includes('servo') || lower.includes('tonypi')) return 'embodiment';
            if (lower.includes('locomot') || lower.includes('walk') || lower.includes('gait')) return 'locomotion';
            if (lower.includes('navigat') || lower.includes('obstacle')) return 'navigation';
            if (lower.includes('sensor') || lower.includes('imu') || lower.includes('ultrasonic')) return 'sensors';
            if (lower.includes('debug') || lower.includes('error') || lower.includes('fix')) return 'debugging';
            if (lower.includes('podcast') || lower.includes('newsletter')) return 'content';
            if (lower.includes('shopify') || lower.includes('store')) return 'commerce';
            if (lower.includes('ellis')) return 'ellis';
            return 'general';
        }

        // Infer from section path
        const lower = sectionPath.toLowerCase();
        if (lower.includes('embodiment') || lower.includes('body code')) return 'embodiment';
        if (lower.includes('locomotion') || lower.includes('trust hierarchy')) return 'locomotion';
        if (lower.includes('embodied habits') || lower.includes('navigation')) return 'navigation';
        if (lower.includes('distilled principles') || lower.includes('relational')) return 'principles';
        if (lower.includes('heartbeat') || lower.includes('active task')) return 'tasks';
        if (lower.includes('completed')) return 'completed-tasks';
        if (lower.includes('deployment') || lower.includes('architecture')) return 'architecture';
        return 'general';
    }

    _shouldSkip(sectionPath, filename) {
        if (!sectionPath) return false;

        // Check file-level skip
        if (this.skipFiles.has(filename.toLowerCase())) return true;

        // Check section-level skip
        const pathLower = sectionPath.toLowerCase();
        for (const skip of this.skipSections) {
            if (pathLower.includes(skip)) return true;
        }

        return false;
    }

    /**
     * Consolidate workspace knowledge entries.
     *
     * Detects entries whose section_path no longer exists in the current
     * workspace files and marks them as 'workspace:archived'. This distinguishes
     * "content that's in the file AND the DB" from "content that's only in the DB"
     * (e.g., after AGENTS.md slimming).
     *
     * Does NOT delete or supersede — archived entries are still valuable knowledge
     * and continue to participate in semantic search.
     *
     * @param {string} agentId
     * @returns {{ consolidated: number, total: number }}
     */
    consolidateWorkspace(agentId) {
        if (!this.workspacePath) {
            return { consolidated: 0, total: 0, reason: 'no workspace path' };
        }

        // 1. Get all workspace-sourced entries
        const entries = this.store.getWorkspaceEntries(agentId);
        if (entries.length === 0) {
            return { consolidated: 0, total: 0 };
        }

        // 2. Build set of current section paths from workspace files
        const currentPaths = new Set();
        let files;
        try {
            files = fs.readdirSync(this.workspacePath)
                .filter(f => f.endsWith('.md') && !this.skipFiles.has(f.toLowerCase()));
        } catch (e) {
            return { consolidated: 0, total: entries.length, reason: `readdir failed: ${e.message}` };
        }

        for (const file of files) {
            try {
                const filePath = path.join(this.workspacePath, file);
                const content = fs.readFileSync(filePath, 'utf-8');
                const chunks = this._parseMarkdownSections(content, file);

                for (const chunk of chunks) {
                    if (!this._shouldSkip(chunk.sectionPath, file)) {
                        currentPaths.add(chunk.sectionPath);
                    }
                }
            } catch (e) {
                // Skip files that can't be read
            }
        }

        // 3. Mark entries whose section_path no longer exists in workspace
        let consolidated = 0;
        for (const entry of entries) {
            if (entry.section_path && !currentPaths.has(entry.section_path)) {
                this.store.updateSourceType(entry.id, 'workspace:archived');
                consolidated++;
            }
        }

        return { consolidated, total: entries.length };
    }

    _hashContent(content) {
        const normalized = content.replace(/\s+/g, ' ').trim();
        return crypto.createHash('sha256').update(normalized).digest('hex');
    }
}

module.exports = KnowledgeIndexer;
