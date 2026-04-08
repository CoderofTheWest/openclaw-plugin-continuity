/**
 * TopicTracker — Topic freshness scoring and fixation detection.
 *
 * Extracted from Clint's topicFreshnessTracker.js (194 lines).
 * Stripped of hardcoded Clint-specific topic patterns.
 *
 * Tracks topic mentions within a sliding window of exchanges.
 * When a topic exceeds the fixation threshold, a note is generated
 * for the agent — awareness, not a hard redirect.
 */

// Common English words (5+ chars) that should never be tracked as topics.
// Merged with any user-provided stopWords from config.
const DEFAULT_STOPWORDS = new Set([
    'about', 'above', 'after', 'again', 'against', 'along', 'almost',
    'already', 'among', 'another', 'around', 'based', 'because', 'before',
    'being', 'below', 'between', 'both', 'built', 'called', 'comes',
    'could', 'currently', 'doesn', 'doing', 'during', 'each', 'either',
    'enough', 'every', 'first', 'following', 'found', 'given', 'going',
    'hasn', 'having', 'here', 'however', 'include', 'instead', 'isn',
    'itself', 'just', 'keep', 'known', 'later', 'least', 'level',
    'likely', 'looks', 'makes', 'maybe', 'might', 'much', 'needs',
    'never', 'note', 'often', 'only', 'other', 'ought', 'quite',
    'rather', 'really', 'recently', 'seems', 'shall', 'should', 'since',
    'something', 'started', 'still', 'takes', 'their', 'them', 'then',
    'there', 'these', 'thing', 'things', 'think', 'those', 'though',
    'through', 'times', 'under', 'until', 'using', 'very', 'wants',
    'wasn', 'well', 'were', 'what', 'whatever', 'when', 'where',
    'which', 'while', 'whose', 'will', 'with', 'within', 'without',
    'would', 'your'
]);

class TopicTracker {
    /**
     * @param {object} config - full plugin config (reads topicTracking section)
     */
    constructor(config = {}) {
        const tc = config.topicTracking || config;
        this.windowSize = tc.windowSize || 6;
        this.fixationThreshold = tc.fixationThreshold || 3;
        this.decayFactor = tc.decayFactor || 0.5;
        this.customPatterns = (tc.customPatterns || []).map(p => new RegExp(p, 'gi'));
        this.stopWords = new Set([...DEFAULT_STOPWORDS, ...(tc.stopWords || [])]);
        this.minWordLength = tc.minWordLength || 5;
        this.pruneAge = tc.pruneAge || 86400000; // 24h default

        // topic → { mentions, lastSeen, firstSeen, lastTimestamp, coOccurrences: Map, parentTopic }
        this._topics = new Map();
        this._currentExchange = 0;

        // Hierarchy config
        const hc = tc.hierarchy || {};
        this.hierarchyEnabled = hc.enabled !== false;
        this.hierarchyCoOccurrenceThreshold = hc.coOccurrenceThreshold || 0.6;
        this.hierarchyMinMentions = hc.minMentions || 3;
    }

    /**
     * Process a message and update topic state.
     *
     * @param {string} messageText
     * @param {number} [exchangeIndex] - optional; auto-increments if omitted
     * @returns {{ topics: string[], fixatedTopics: string[], freshnessScores: object }}
     */
    track(messageText, exchangeIndex) {
        if (exchangeIndex !== undefined) {
            this._currentExchange = exchangeIndex;
        } else {
            this._currentExchange++;
        }

        // Prune topics outside the window
        this._pruneWindow(this._currentExchange);

        // Extract topics from this message
        const topics = this.extractTopics(messageText);

        // Update mention counts
        for (const topic of topics) {
            const existing = this._topics.get(topic);
            if (existing) {
                existing.mentions++;
                existing.lastSeen = this._currentExchange;
                existing.lastTimestamp = Date.now();
            } else {
                this._topics.set(topic, {
                    mentions: 1,
                    firstSeen: this._currentExchange,
                    lastSeen: this._currentExchange,
                    lastTimestamp: Date.now()
                });
            }
        }

        // Track co-occurrences between topics in this message
        if (this.hierarchyEnabled && topics.length > 1) {
            this._trackCoOccurrences(topics);
        }

        // Build results
        const freshnessScores = {};
        const fixatedTopics = [];
        const allTopics = [];

        for (const [topic, data] of this._topics) {
            const score = this.getFreshnessScore(topic);
            freshnessScores[topic] = score;
            allTopics.push(topic);

            if (data.mentions >= this.fixationThreshold) {
                fixatedTopics.push(topic);
            }
        }

        return {
            topics: allTopics,
            fixatedTopics,
            freshnessScores
        };
    }

    /**
     * Extract topics from text using:
     * A) Custom regex patterns (if configured)
     * B) Frequency counting (words appearing 3+ times, length > minWordLength, not stopword)
     *
     * For single-pass messages, frequency counting uses cross-exchange
     * state — a word that appears once per exchange across 3 exchanges
     * is still tracked.
     *
     * @param {string} text
     * @returns {string[]} extracted topic strings (lowercase)
     */
    extractTopics(text) {
        if (!text) return [];
        const topics = new Set();

        // A) Custom patterns
        for (const pattern of this.customPatterns) {
            pattern.lastIndex = 0;
            let match;
            while ((match = pattern.exec(text)) !== null) {
                topics.add(match[0].toLowerCase());
            }
        }

        // B) Frequency counting within this message
        const words = text.toLowerCase().split(/\s+/).filter(w =>
            w.length >= this.minWordLength &&
            !this.stopWords.has(w) &&
            /^[a-z]/.test(w) // starts with a letter
        );

        // Clean punctuation from words
        const cleaned = words.map(w => w.replace(/[^a-z0-9-]/g, ''));

        // Count frequency within this single message
        const freq = new Map();
        for (const word of cleaned) {
            if (!word) continue;
            freq.set(word, (freq.get(word) || 0) + 1);
        }

        // Words appearing 2+ times in a single message are likely topics
        for (const [word, count] of freq) {
            if (count >= 2) {
                topics.add(word);
            }
        }

        // Also add any word that already exists in the topic map
        // (cross-exchange tracking — even a single mention counts as a revisit)
        for (const word of cleaned) {
            if (this._topics.has(word)) {
                topics.add(word);
            }
        }

        return [...topics];
    }

    /**
     * Calculate freshness score for a topic.
     * 1.0 = fresh (never or rarely mentioned), 0.0 = completely stale.
     *
     * @param {string} topic
     * @returns {number} 0.0 - 1.0
     */
    getFreshnessScore(topic) {
        const data = this._topics.get(topic);
        if (!data) return 1.0;
        return Math.max(0.0, 1.0 - (data.mentions / this.fixationThreshold) * this.decayFactor);
    }

    /**
     * Get topics that have exceeded the fixation threshold.
     * @returns {Array<{ topic: string, mentions: number, freshnessScore: number }>}
     */
    getFixatedTopics() {
        const fixated = [];
        for (const [topic, data] of this._topics) {
            if (data.mentions >= this.fixationThreshold) {
                fixated.push({
                    topic,
                    mentions: data.mentions,
                    freshnessScore: this.getFreshnessScore(topic)
                });
            }
        }
        return fixated;
    }

    /**
     * Get all tracked topics with metadata.
     * @returns {Array<{ topic: string, mentions: number, freshnessScore: number, lastSeen: number }>}
     */
    getAllTopics() {
        const all = [];
        for (const [topic, data] of this._topics) {
            all.push({
                topic,
                mentions: data.mentions,
                freshnessScore: this.getFreshnessScore(topic),
                lastSeen: data.lastSeen
            });
        }
        // Sort by mentions descending
        all.sort((a, b) => b.mentions - a.mentions);
        return all;
    }

    /**
     * Format fixation notes for prompt injection.
     * These are awareness signals — the model decides what to do with them.
     *
     * @param {Array} [fixatedTopics] - optional override
     * @returns {string} formatted notes or empty string
     */
    formatNotes(fixatedTopics) {
        const list = fixatedTopics || this.getFixatedTopics();
        if (!list || list.length === 0) return '';

        return list.map(t =>
            `[TOPIC NOTE] The topic '${t.topic}' has come up ${t.mentions} times recently.`
        ).join('\n');
    }

    /**
     * Advance the exchange counter without tracking a message.
     * Used to mark agent responses as an exchange boundary.
     */
    advanceExchange() {
        this._currentExchange++;
    }

    /**
     * Clear all topic state.
     */
    reset() {
        this._topics.clear();
        this._currentExchange = 0;
    }

    // ---------------------------------------------------------------
    // Hierarchy — co-occurrence tracking and parent-child inference
    // ---------------------------------------------------------------

    /**
     * Track co-occurrences between topics in the same message.
     * @param {string[]} topics
     */
    _trackCoOccurrences(topics) {
        for (let i = 0; i < topics.length; i++) {
            for (let j = i + 1; j < topics.length; j++) {
                const a = this._topics.get(topics[i]);
                const b = this._topics.get(topics[j]);
                if (a && b) {
                    if (!a.coOccurrences) a.coOccurrences = new Map();
                    if (!b.coOccurrences) b.coOccurrences = new Map();
                    a.coOccurrences.set(topics[j], (a.coOccurrences.get(topics[j]) || 0) + 1);
                    b.coOccurrences.set(topics[i], (b.coOccurrences.get(topics[i]) || 0) + 1);
                }
            }
        }
    }

    /**
     * Infer parent-child relationships from co-occurrence patterns.
     * A topic B is a child of topic A if:
     *   - B co-occurs with A > threshold of the time
     *   - A has more total mentions than B (broader topic)
     *   - Both have enough mentions to be meaningful
     * Called at session_end to update hierarchy before persistence.
     */
    inferHierarchy() {
        if (!this.hierarchyEnabled) return;

        for (const [topic, data] of this._topics) {
            if (!data.coOccurrences || data.coOccurrences.size === 0) continue;
            if (data.mentions < this.hierarchyMinMentions) continue;

            let bestParent = null;
            let bestRatio = 0;

            for (const [other, count] of data.coOccurrences) {
                const otherData = this._topics.get(other);
                if (!otherData) continue;

                const coOccurrenceRatio = count / data.mentions;

                // Parent should be broader (more mentions) and co-occur frequently
                if (otherData.mentions > data.mentions && coOccurrenceRatio > this.hierarchyCoOccurrenceThreshold) {
                    if (coOccurrenceRatio > bestRatio) {
                        bestRatio = coOccurrenceRatio;
                        bestParent = other;
                    }
                }
            }

            if (bestParent) {
                data.parentTopic = bestParent;
                data.hierarchyConfidence = bestRatio;
            }
        }
    }

    /**
     * Get the topic hierarchy as a tree.
     * @returns {Array<{ topic, mentions, children: Array, parentTopic }>}
     */
    getHierarchy() {
        const childMap = new Map();

        for (const [topic, data] of this._topics) {
            if (data.parentTopic) {
                if (!childMap.has(data.parentTopic)) childMap.set(data.parentTopic, []);
                childMap.get(data.parentTopic).push({
                    topic,
                    mentions: data.mentions,
                    parentTopic: data.parentTopic
                });
            }
        }

        const roots = [];
        for (const [topic, data] of this._topics) {
            if (!data.parentTopic) {
                roots.push({
                    topic,
                    mentions: data.mentions,
                    children: childMap.get(topic) || [],
                    parentTopic: null
                });
            }
        }

        return roots.sort((a, b) => b.mentions - a.mentions);
    }

    /**
     * Get a topic and all its subtopics.
     * @param {string} parentTopic
     * @returns {string[]}
     */
    getSubtopics(parentTopic) {
        const result = [parentTopic];
        for (const [topic, data] of this._topics) {
            if (data.parentTopic === parentTopic) {
                result.push(topic);
            }
        }
        return result;
    }

    /**
     * Get all topics with hierarchy data for SQLite persistence.
     * @returns {Array}
     */
    getAllTopicsWithHierarchy() {
        const result = [];
        for (const [topic, data] of this._topics) {
            const coOccCount = data.coOccurrences
                ? [...data.coOccurrences.values()].reduce((a, b) => a + b, 0)
                : 0;
            result.push({
                topic,
                mentions: data.mentions,
                parentTopic: data.parentTopic || null,
                coOccurrenceCount: coOccCount,
                firstSeen: data.firstSeen,
                lastSeen: data.lastSeen,
                confidence: data.hierarchyConfidence || 0
            });
        }
        return result;
    }

    /**
     * Load persisted hierarchy from SQLite data.
     * Merges parent-child relationships into current session state.
     * @param {Array} rows - from SummaryStore.loadTopicHierarchy()
     */
    loadPersistedHierarchy(rows) {
        if (!rows || !rows.length) return;
        for (const row of rows) {
            const existing = this._topics.get(row.topic);
            if (existing && row.parent_topic) {
                existing.parentTopic = row.parent_topic;
                existing.hierarchyConfidence = row.confidence || 0;
            }
            // Don't create topics that aren't active in this session —
            // hierarchy is only meaningful for topics the tracker already knows about
        }
    }

    // ---------------------------------------------------------------
    // Internal
    // ---------------------------------------------------------------

    /**
     * Remove topics whose last-seen exchange is outside the window.
     * @param {number} currentExchange
     */
    _pruneWindow(currentExchange) {
        const cutoff = currentExchange - this.windowSize;
        for (const [topic, data] of this._topics) {
            if (data.lastSeen < cutoff) {
                this._topics.delete(topic);
            }
        }
    }
}

module.exports = TopicTracker;
