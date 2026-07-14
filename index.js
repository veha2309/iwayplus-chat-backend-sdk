const { Ollama } = require('ollama');
const { ChromaClient } = require('chromadb');
const { OllamaEmbeddingFunction } = require('@chroma-core/ollama');
const { LRUCache } = require('lru-cache');
const fs = require('fs');
const path = require('path');
const https = require('https');

// ─────────────────────────────────────────────
// SSE Streaming Token filter
// ─────────────────────────────────────────────
class ThinkingFilter {
    constructor() {
        this.inThinking = false;
        this.buffer = '';
    }

    filter(token) {
        this.buffer += token;
        let output = '';

        while (true) {
            if (this.inThinking) {
                const endTagIndex = this.buffer.indexOf('</think>');
                if (endTagIndex !== -1) {
                    this.inThinking = false;
                    this.buffer = this.buffer.substring(endTagIndex + 8);
                } else {
                    this.buffer = '';
                    break;
                }
            } else {
                const startTagIndex = this.buffer.indexOf('<think>');
                if (startTagIndex !== -1) {
                    output += this.buffer.substring(0, startTagIndex);
                    this.inThinking = true;
                    this.buffer = this.buffer.substring(startTagIndex + 7);
                } else {
                    output += this.buffer;
                    this.buffer = '';
                    break;
                }
            }
        }
        return output;
    }
}

// ─────────────────────────────────────────────
// Emoji-preserving Sentence Truncator
// ─────────────────────────────────────────────
function truncateToTwoSentences(text, maxSentences = 2, maxChars = 280) {
    if (!text) return text;

    // Extract any trailing emojis/pictographs from the end of the original text
    const emojiMatch = text.match(/[\p{Emoji_Presentation}\p{Extended_Pictographic}\u200d\uFE0F\s]+$/u);
    const trailingEmojis = emojiMatch ? emojiMatch[0].trim() : '';

    // Dynamically increase char limit if this response describes multiple entities
    const isMultiSubject = text.toLowerCase().includes('and') || text.includes(';') || text.includes(',');
    const activeMaxChars = isMultiSubject ? maxChars * 2 : maxChars;

    // Split on sentence-ending punctuation only when followed by whitespace or end-of-string.
    const parts = text.split(/(?<=[.!?])(?=\s)/);
    let result = '';
    let count = 0;
    for (const part of parts) {
        const trimmed = part.trim();
        if (trimmed.length < 15) continue; // skip ultra-short fragments like "dhaan)."
        result += (result ? ' ' : '') + trimmed;
        count++;
        if (count >= maxSentences) break;
        if (result.length >= activeMaxChars) break;
    }
    if (!result) result = text; // fallback

    // Absolute character cap
    if (result.length > activeMaxChars + 50) {
        result = result.substring(0, activeMaxChars + 50).replace(/\s+\S*$/, '').trim() + '…';
    }

    // Append trailing emojis back if they were truncated/skipped
    if (trailingEmojis && !result.includes(trailingEmojis)) {
        result = result.trim() + ' ' + trailingEmojis;
    }

    return result;
}

// Levenshtein distance helper
function levenshtein(a, b) {
    const tmp = [];
    for (let i = 0; i <= a.length; i++) {
        tmp[i] = [i];
    }
    for (let j = 0; j <= b.length; j++) {
        tmp[0][j] = j;
    }
    for (let i = 1; i <= a.length; i++) {
        for (let j = 1; j <= b.length; j++) {
            tmp[i][j] = Math.min(
                tmp[i - 1][j] + 1,
                tmp[i][j - 1] + 1,
                tmp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
            );
        }
    }
    return tmp[a.length][b.length];
}

// ─────────────────────────────────────────────
// Navigation query detector helper
// ─────────────────────────────────────────────
function isNavigationQuery(query) {
    const qLower = query.toLowerCase();

    // Exclude informational/biological queries from navigation routing
    if (qLower.includes('habitat') || qLower.includes('diet') || qLower.includes('conservation') || qLower.includes('lifespan') || qLower.includes('आवास')) {
        return false;
    }

    const navKeywords = [
        'navigate', 'navigation', 'direction', 'directions', 'route', 'routes',
        'take me to', 'take me', 'bring me to', 'show me the way',
        'how to get to', 'show path to', 'path to', 'where is', 'map to',
        'lead me', 'guide me', 'find route', 'go to', 'get to', 'how to reach', 'where',
        'rasta', 'raste', 'marg', 'disha', 'kahan hai', 'kaha hai',
        'kidhar hai', 'jaun', 'jaane ka', 'le jao', 'le chalo', 'chalao',
        'pahuncha', 'pahunchao', 'batao kahan', 'dikhao rasta', 'kahan', 'kaha', 'kidhar',
        'रास्ता', 'मार्ग', 'दिशा', 'नक्शा', 'जाएं', 'किधर', 'कहाँ है', 'कहां है',
        'ले जाओ', 'पहुँचाओ', 'रास्ता बताओ', 'ले चलो', 'कहाँ', 'कहां'
    ];
    return navKeywords.some(keyword => qLower.includes(keyword));
}

// ─────────────────────────────────────────────
// Core Multi-Tenant SDK Class
// ─────────────────────────────────────────────
class ChatEngine {
    constructor(config = {}) {
        this.ollamaUrl = config.ollamaUrl || 'http://localhost:11434';
        this.chromaUrl = config.chromaUrl || 'http://localhost:8000';
        this.chatModel = config.chatModel || 'gemma4:e4b';
        this.embedModel = config.embedModel || 'mxbai-embed-large';
        this.extractorModel = config.extractorModel || 'iwayplus-zoo:latest';

        this.ollama = new Ollama({ host: this.ollamaUrl });
        this.chromaClient = new ChromaClient({ path: this.chromaUrl });
        this.embedFunction = new OllamaEmbeddingFunction({
            url: this.ollamaUrl,
            model: this.embedModel
        });

        // LRU Cache Pools
        this.graphCache = new LRUCache({ max: 200, ttl: 1000 * 60 * 30 });
        this.responseCache = new LRUCache({ max: 1000, ttl: 1000 * 60 * 15 });
        this.chromaSearchCache = new LRUCache({ max: 500, ttl: 1000 * 60 * 15 });

        // Registered venues configuration maps
        this.venues = new Map();
    }

    /**
     * Register a new venue (hospital, museum, mall, etc.) with its configurations and data
     */
    async registerVenue({
        venueId,
        collectionName,
        personaName = 'Guide',
        defaultEmoji = '💬',
        venueName = '',
        venueType = '',
        exampleEntityMissing = '',
        exampleEntityPartial = '',
        hindiVenueType = '',
        hindiVenueName = '',
        closedDay = null,
        hindiClosedDay = null,
        registry,
        facilitySynonyms = {},
        hindiDict = {},
        hindiGlossaryReplacements = {},
        graphPath = null,
        geojsonPath = null
    }) {
        const activeCollectionName = collectionName || `${venueId}_collection`;
        const venueData = {
            personaName,
            defaultEmoji,
            venueName,
            venueType,
            exampleEntityMissing,
            exampleEntityPartial,
            hindiVenueType,
            hindiVenueName,
            closedDay,
            hindiClosedDay,
            registry,
            facilitySynonyms,
            hindiDict,
            hindiGlossaryReplacements,
            graph: { nodes: [], edges: [] },
            geojsonData: [],
            trieIndex: new Map(),
            collection: null
        };

        // 1. Connect to ChromaDB collection
        try {
            venueData.collection = await this.chromaClient.getOrCreateCollection({
                name: activeCollectionName,
                embeddingFunction: this.embedFunction
            });
            console.log(`[SDK] Connected to ChromaDB collection: ${activeCollectionName}`);
        } catch (err) {
            console.error(`[SDK] Failed to initialize ChromaDB collection ${activeCollectionName} for ${venueId}:`, err.message);
        }

        // 2. Load Graph Data
        if (graphPath && fs.existsSync(graphPath)) {
            try {
                venueData.graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
                console.log(`[SDK] Loaded ${venueData.graph.nodes?.length || 0} nodes from graph: ${graphPath}`);
            } catch (err) {
                console.error(`[SDK] Failed to load graph for ${venueId}:`, err.message);
            }
        }

        // 3. Load GeoJSON Data
        if (geojsonPath && fs.existsSync(geojsonPath)) {
            try {
                const rawGeo = JSON.parse(fs.readFileSync(geojsonPath, 'utf8'));
                venueData.geojsonData = rawGeo.features || [];
                console.log(`[SDK] Loaded ${venueData.geojsonData.length} features from GeoJSON: ${geojsonPath}`);
            } catch (err) {
                console.error(`[SDK] Failed to load GeoJSON for ${venueId}:`, err.message);
            }
        }

        // 4. Build Trie Index
        const entries = [];
        const TRIE_BLACKLIST = new Set([
            'bird', 'birds', 'animal', 'animals', 'reptile', 'reptiles',
            'mammal', 'mammals', 'fish', 'insect', 'insects', 'plant', 'plants',
            'cat', 'cats', 'dog', 'dogs', 'pet', 'pets', 'zebra', 'jebra', 'ज़ेब्रा'
        ]);

        if (registry) {
            for (const [phrase, name] of Object.entries(registry.lookup || {})) {
                if (TRIE_BLACKLIST.has(phrase.toLowerCase())) continue;
                entries.push([phrase.toLowerCase(), name]);
            }
            for (const name of (registry.canonicalNames || [])) {
                const lower = name.toLowerCase();
                if (TRIE_BLACKLIST.has(lower)) continue;
                if (!entries.some(e => e[0] === lower)) {
                    entries.push([lower, name]);
                }
            }
            entries.sort((a, b) => b[0].length - a[0].length);
            venueData.trieIndex = new Map(entries.map(([phrase, name]) => [
                phrase,
                { name, regex: new RegExp(`\\b${phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i') }
            ]));
            console.log(`[SDK] [TRIE] Built index with ${venueData.trieIndex.size} entries for ${venueId}.`);
        }

        this.venues.set(venueId, venueData);
    }

    /**
     * Traverse Graph to get contextual nodes
     */
    graphTraversal(venueId, startNodeId, maxDepth = 1) {
        const venue = this.venues.get(venueId);
        if (!venue || !venue.graph) return [];

        const cacheKey = `${venueId}:${startNodeId}:${maxDepth}`;
        if (this.graphCache.has(cacheKey)) {
            return this.graphCache.get(cacheKey);
        }

        const visited = new Set();
        const results = [];
        const queue = [{ id: startNodeId, depth: 0 }];

        while (queue.length > 0) {
            const { id, depth } = queue.shift();
            if (visited.has(id)) continue;
            visited.add(id);

            const node = venue.graph.nodes.find(n => n.id === id);
            if (node) results.push(node);

            if (depth < maxDepth) {
                const edges = venue.graph.edges.filter(e => e.source === id || e.target === id);
                for (const edge of edges) {
                    const neighbor = edge.source === id ? edge.target : edge.source;
                    if (!visited.has(neighbor)) {
                        queue.push({ id: neighbor, depth: depth + 1 });
                    }
                }
            }
        }

        this.graphCache.set(cacheKey, results);
        return results;
    }

    /**
     * Perform Semantic Search + RAG Context Extraction
     */
    async searchRAG(venueId, query, subject, isFacilityMatch, limit = 3, language = 'en', isEventQuery = false) {
        const venue = this.venues.get(venueId);
        if (!venue || !venue.collection) {
            return { context: '', references: [], topScore: 0, subject: 'general', sortedContext: [] };
        }

        const cacheKey = `${venueId}:${query}:${subject}:${isFacilityMatch}:${limit}:${language}:${isEventQuery}`;
        if (this.chromaSearchCache.has(cacheKey)) {
            return this.chromaSearchCache.get(cacheKey);
        }

        console.log('[SDK] [DEBUG] searchRAG: Querying ChromaDB collection...');
        const results = await venue.collection.query({
            queryTexts: [query],
            nResults: limit,
            where: { language }
        });
        console.log('[SDK] [DEBUG] searchRAG: ChromaDB query complete. Found ' + (results?.documents?.[0]?.length || 0) + ' docs.');

        const sortedContext = [];
        let topScore = 0;

        if (results && results.documents && results.documents[0]) {
            for (let i = 0; i < results.documents[0].length; i++) {
                const docText = results.documents[0][i];
                const meta = results.metadatas[0][i];
                const distance = results.distances ? results.distances[0][i] : 1.0;
                const score = 1.0 - distance;

                if (i === 0) topScore = score;

                sortedContext.push({
                    text: docText,
                    metadata: meta,
                    score
                });
            }
        }

        // Format raw text context
        let contextText = sortedContext.map(c => c.text).join('\n\n');

        // Extract reference cards
        let refs = [];
        if (subject && subject !== 'general') {
            refs.push(subject);
        }
        for (const c of sortedContext) {
            if (c.metadata && c.metadata.doc_id && !refs.includes(c.metadata.doc_id)) {
                // Strip en/hi suffixes if present
                const cleanId = c.metadata.doc_id.replace(/_(en|hi)$/, '');
                if (!refs.includes(cleanId)) refs.push(cleanId);
            }
        }

        const responseObj = {
            context: contextText,
            references: refs,
            topScore,
            subject: subject || 'general',
            sortedContext
        };

        this.chromaSearchCache.set(cacheKey, responseObj);
        return responseObj;
    }

    /**
     * Trie-based Fast Subject Extractor
     */
    fastExtract(venueId, query, QUERY_STOP_WORDS = new Set()) {
        const venue = this.venues.get(venueId);
        if (!venue || !venue.trieIndex) return null;

        const q = query.toLowerCase();
        const cleanQ = q.replace(/[?!.,;()'"]/g, '').trim();
        const queryWords = cleanQ.split(/\s+/).filter(w => w.length > 0 && !QUERY_STOP_WORDS.has(w));
        const foundMatches = [];

        for (const [phrase, { name, regex }] of venue.trieIndex) {
            if (regex.test(q)) {
                const phraseWords = phrase.split(/\s+/).filter(w => w.length > 0);
                const extraWords = queryWords.filter(qw => !phraseWords.includes(qw));
                if (extraWords.length > 0) continue; // skip if there are other key words

                if (!foundMatches.some(m => m.phrase.includes(phrase))) {
                    foundMatches.push({ phrase, name });
                }
            }
        }

        const uniqueNames = [...new Set(foundMatches.map(m => m.name))];
        if (uniqueNames.length === 1) {
            const wordCount = q.split(/\s+/).length;
            if (wordCount <= 7) return uniqueNames[0];
        }
        return null;
    }

    /**
     * Check coordinates mapping from GeoJSON
     */
    getCoordinatesForPlace(venueId, name) {
        const venue = this.venues.get(venueId);
        if (!venue || !venue.geojsonData) return null;

        const cleanName = name.toLowerCase().replace(/\s+\d+$/, '').trim();
        for (const feature of venue.geojsonData) {
            const fName = feature.properties?.name || '';
            const matchName = fName.toLowerCase().replace(/\s+\d+$/, '').trim();
            if (matchName === cleanName) {
                const geom = feature.geometry;
                if (geom.type === 'Point') {
                    return {
                        name: fName,
                        latitude: geom.coordinates[1],
                        longitude: geom.coordinates[0]
                    };
                } else if (geom.type === 'Polygon' || geom.type === 'MultiPolygon') {
                    // Simple centroid calculation
                    let coords = [];
                    if (geom.type === 'Polygon') {
                        coords = geom.coordinates[0];
                    } else {
                        coords = geom.coordinates[0][0];
                    }
                    let sumLat = 0, sumLng = 0;
                    for (const pt of coords) {
                        sumLng += pt[0];
                        sumLat += pt[1];
                    }
                    return {
                        name: fName,
                        latitude: sumLat / coords.length,
                        longitude: sumLng / coords.length
                    };
                }
            }
        }
        return null;
    }

    /**
     * Map query keywords to related cards
     */
    findRelatedEntities(venueId, subject, queryText, QUERY_STOP_WORDS = new Set(), ADJECTIVE_BLACKLIST = new Set()) {
        const venue = this.venues.get(venueId);
        if (!venue || !venue.registry) return [];

        const seeds = new Set();
        if (queryText) {
            const queryWords = queryText.toLowerCase().replace(/[?!.,()]/g, '').split(/\s+/);
            for (const w of queryWords) {
                if (w.length >= 3 && !QUERY_STOP_WORDS.has(w) && !ADJECTIVE_BLACKLIST.has(w)) {
                    seeds.add(w);
                    if (w.endsWith('s') && w.length > 3) seeds.add(w.slice(0, -1));

                    // Map Hindi keywords to English seeds
                    for (const [engKey, hindiVal] of Object.entries(venue.hindiDict || {})) {
                        if (hindiVal === w || engKey === w) {
                            const engWords = engKey.split(/\s+/);
                            for (const ew of engWords) {
                                if (ew.length >= 3 && !QUERY_STOP_WORDS.has(ew)) {
                                    seeds.add(ew);
                                    if (ew.endsWith('s') && ew.length > 3) seeds.add(ew.slice(0, -1));
                                }
                            }
                        }
                    }
                }
            }
        }

        const related = new Set();
        for (const seed of seeds) {
            for (const name of (venue.registry.rawNames || [])) {
                const cleanName = name.replace(/\s+\d+$/, '').trim();
                if (venue.registry.eventNames?.has(cleanName)) continue; // skip events
                const nameLower = name.toLowerCase().replace(/[0-9]/g, '');
                if (new RegExp(`\\b${seed}\\b`, 'i').test(nameLower)) {
                    related.add(name);
                }
            }
        }

        return Array.from(related);
    }

    /**
     * Main RAG Query Processor Lifecycle
     */
    async processQuery({
        venueId,
        question,
        language = 'en',
        history = [],
        userLocation = null,
        stream = false,
        res = null,
        callNavigationAPI = null,
        buildNavigationAnswer = null,
        QUERY_STOP_WORDS = new Set(),
        ADJECTIVE_BLACKLIST = new Set(),
        facilitySynonyms = {},
        deepSearch = false
    }) {
        const venue = this.venues.get(venueId);
        if (!venue) {
            const errText = "Venue configuration not found.";
            if (stream && res) {
                res.write(`data: ${JSON.stringify({ token: errText })}\n\n`);
                return res.end();
            }
            return { answer: errText, keyword: 'general', references: [] };
        }

        const isHindi = language === 'hi';
        const qLower = question.toLowerCase().trim();

        // 1. Check LRU cache for exact hits
        const cacheKey = `${venueId}:${qLower}:${language}:${stream}`;
        if (!stream && this.responseCache.has(cacheKey)) {
            console.log(`[SDK] [CACHE] HIT (enriched) for "${qLower}"`);
            return this.responseCache.get(cacheKey);
        }

        // 2. Perform subject extraction using LLM-Bypass fastExtract or Holistic Heuristics
        let extractedSubject = this.fastExtract(venueId, question, QUERY_STOP_WORDS);
        if (extractedSubject) {
            console.log(`[SDK] [LLM-BYPASS] Subject extracted via Trie: "${extractedSubject}"`);
        } else {
            // Run Holistic Matcher before falling back to LLM
            const cleanQLower = qLower.replace(/[?!.,;()'"]/g, '').trim();
            const words = cleanQLower.split(/\s+/).filter(w => w.length > 0);

            // Check Conjunction / Multi-Entity Pre-Pass
            let extracted = new Set();
            const registry = venue.registry;
            if (registry) {
                // Check lookup keys
                for (const key in (registry.lookup || {})) {
                    const escapedKey = key.replace(/[.*+?^${}()|[\]\s\\]/g, '\\$&');
                    const regex = new RegExp(`\\b${escapedKey}\\b`, 'i');
                    if (regex.test(qLower)) {
                        extracted.add(registry.lookup[key]);
                    }
                }

                if (extracted.size > 1) {
                    extractedSubject = Array.from(extracted).join(', ');
                    console.log(`[SDK] [HOLISTIC-BYPASS] Resolved multiple subjects: "${extractedSubject}"`);
                }
            }

            if (!extractedSubject && registry && registry.canonicalNames) {
                let bestEntity = null;
                let highestScore = 0;
                for (const canonical of registry.canonicalNames) {
                    if (registry.eventNames?.has(canonical) && !/\bday\b/.test(qLower)) continue;
                    const cWords = canonical.toLowerCase().split(/[^a-z0-9]+/).filter(w => w.length > 2 && !QUERY_STOP_WORDS.has(w));
                    if (cWords.length === 0) continue;

                    let matchedTokens = 0;
                    for (const cw of cWords) {
                        let bestWordScore = 0;
                        for (const qw of words) {
                            if (QUERY_STOP_WORDS.has(qw)) continue;
                            if (qw === cw) {
                                bestWordScore = 1.0;
                            } else if (qw.length >= 4 && cw.length >= 4 && (qw.includes(cw) || cw.includes(qw))) {
                                bestWordScore = 0.7;
                            } else if (qw.length >= 4 && cw.length >= 4) {
                                const dist = levenshtein(qw, cw);
                                const maxAllowed = cw.length >= 6 ? 2 : 1;
                                if (dist <= maxAllowed) {
                                    bestWordScore = 1.0 - (dist * 0.2);
                                }
                            }
                        }
                        if (bestWordScore > 0) matchedTokens += bestWordScore;
                    }

                    const overlapRatio = matchedTokens / cWords.length;
                    let score = matchedTokens * 10 + overlapRatio * 5;
                    if (canonical.toLowerCase().includes(qLower) || qLower.includes(canonical.toLowerCase())) score += 3;

                    if (score > highestScore) {
                        highestScore = score;
                        bestEntity = canonical;
                    }
                }

                if (highestScore >= 13 && bestEntity) {
                    extractedSubject = bestEntity;
                    if (registry.lookup && registry.lookup[extractedSubject.toLowerCase()]) {
                        extractedSubject = registry.lookup[extractedSubject.toLowerCase()];
                    }
                    console.log(`[SDK] [HOLISTIC-BYPASS] Selected Subject: "${extractedSubject}" (Score: ${highestScore.toFixed(2)})`);
                }
            }

            // Fallback to LLM if still general
            if (!extractedSubject || extractedSubject === 'general') {
                // Wake up Ollama extractor model
                try {
                    const prompt = `Task: Extract the main subject or species name from the user question. Return ONLY the canonical name. If none is found, return "general".
Question: "${question}"
Subject:`;
                    const resp = await this.ollama.chat({
                        model: this.extractorModel,
                        messages: [{ role: 'user', content: prompt }],
                        options: { temperature: 0.0, num_predict: 20 }
                    });
                    let ext = (resp.message?.content || '').trim().toLowerCase();
                    ext = ext.split(/[.!?\n]/)[0].trim();
                    ext = ext.replace(/[^a-zA-Z0-9\s\u0900-\u097F]/g, '').trim();

                    if (ext.split(/\s+/).length > 3 || ext.length < 2) {
                        extractedSubject = 'general';
                    } else {
                        let candidate = null;
                        const registry = venue.registry;
                        if (registry) {
                            if (registry.lookup && registry.lookup[ext]) {
                                candidate = registry.lookup[ext];
                            } else if (registry.canonicalNames) {
                                const exactHit = registry.canonicalNames.find(n => n.toLowerCase() === ext);
                                if (exactHit) {
                                    candidate = exactHit;
                                } else {
                                    const fuzzyHit = registry.canonicalNames.find(n => {
                                        const cws = n.toLowerCase().split(/[^a-zA-Z0-9]+/);
                                        return cws.some(cw => cw.length >= 4 && Math.abs(ext.length - cw.length) <= 1 && ext.includes(cw));
                                    });
                                    if (fuzzyHit) candidate = fuzzyHit;
                                }
                            }
                        }
                        extractedSubject = candidate || ext;
                    }
                    console.log(`[SDK] [LLM-EXTRACT] Extracted & Sanitized: "${extractedSubject}"`);
                } catch (err) {
                    console.error('[SDK] Extractor failed, falling back to general:', err.message);
                    extractedSubject = 'general';
                }
            }
        }

        // 3. Resolve matched facilities or events
        let isFacilityMatch = false;
        let matchedFacility = null;
        for (const [facility, syns] of Object.entries(facilitySynonyms)) {
            const isMatch = syns.some(syn => qLower.includes(syn));
            if (isMatch) {
                isFacilityMatch = true;
                matchedFacility = facility;
                break;
            }
        }

        const isEventQuery = (venue.registry?.eventNames || new Set()).has(extractedSubject);
        const isLocationIntent = isNavigationQuery(question);

        // 4. Perform Search (ChromaDB + GraphRAG)
        let searchResult = null;
        let context = '';
        let references = [];
        let topScore = 0;
        let sortedContext = [];
        let finalSubject = extractedSubject;

        const isTraitOrCategory = /\b(eat|diet|live|habitat|likes|conservation)\b/i.test(qLower);
        const searchK = isTraitOrCategory ? 8 : 3;

        searchResult = await this.searchRAG(venueId, question, extractedSubject, isFacilityMatch, searchK, language, isEventQuery);
        context = searchResult.context;
        references = searchResult.references;
        topScore = searchResult.topScore;
        sortedContext = searchResult.sortedContext;
        finalSubject = extractedSubject !== 'general' ? extractedSubject : searchResult.subject;

        // Augment with related cards
        const related = this.findRelatedEntities(venueId, finalSubject, question, QUERY_STOP_WORDS, ADJECTIVE_BLACKLIST);
        if (related.length > 0) {
            references = [...new Set([...references, ...related])]
                .filter(r => r.toLowerCase().replace(/\s+\d+$/, '').trim() !== finalSubject.toLowerCase().replace(/\s+\d+$/, '').trim())
                .slice(0, 5);
        }

        // Check Graph Traversal augmentation (DeepSearch ONLY)
        let graphAugmented = false;
        if (deepSearch && venue.graph && finalSubject && finalSubject !== 'general') {
            const cleanSub = finalSubject.toLowerCase();
            const matchedNode = venue.graph.nodes.find(n => n.id && n.id.toLowerCase().includes(cleanSub));
            if (matchedNode) {
                const nodes = this.graphTraversal(venueId, matchedNode.id, 1).slice(0, 3);
                const graphCtx = nodes
                    .filter(n => n.description)
                    .map(n => `${n.id} (${n.type}): ${n.description}`)
                    .join('\n');
                if (graphCtx) {
                    console.log(`[SDK] [GRAPH] Augmenting with ${nodes.length} nodes`);
                    context = graphCtx + '\n\n' + context;
                    graphAugmented = true;
                }
            }
        }

        // 5. Navigation Interceptor
        const isNavQuery = isNavigationQuery(question);
        if (isNavQuery && finalSubject && finalSubject !== 'general' && callNavigationAPI && buildNavigationAnswer) {
            let destCoords = this.getCoordinatesForPlace(venueId, finalSubject);
            let resolvedSubject = finalSubject;

            if (!destCoords && references.length > 0) {
                for (const ref of references) {
                    const coords = this.getCoordinatesForPlace(venueId, ref);
                    if (coords) {
                        destCoords = coords;
                        resolvedSubject = ref;
                        break;
                    }
                }
            }

            if (destCoords) {
                const sourceCoords = userLocation || { latitude: 28.60638, longitude: 77.24377 }; // default entrance
                const apiResult = await callNavigationAPI(sourceCoords, destCoords);
                const humanAnswer = buildNavigationAnswer(apiResult, destCoords.name, isHindi);
                const routeSegments = Array.isArray(apiResult) ? apiResult : (apiResult ? [apiResult] : []);

                const navResponse = {
                    answer: humanAnswer,
                    navigation: {
                        source: sourceCoords,
                        destination: destCoords,
                        routeData: routeSegments
                    },
                    keyword: resolvedSubject,
                    references: []
                };

                if (stream && res) {
                    res.setHeader('Content-Type', 'text/event-stream');
                    res.write(`data: ${JSON.stringify({ token: humanAnswer })}\n\n`);
                    res.write(`data: ${JSON.stringify({ done: true, keyword: navResponse.keyword, navigation: navResponse.navigation, references: [] })}\n\n`);
                    return res.end();
                } else {
                    return navResponse;
                }
            }
        }

        // 6. Build Prompt Template
        let rawContext = context;
        if (references.length > 0) {
            const refList = references.join(', ');
            rawContext += isHindi
                ? `\n\nसंबंधित स्थान और जानकारी: ${refList}`
                : `\n\nRelated exhibits and locations: ${refList}`;
        }
        const trimmedContext = rawContext.substring(0, isTraitOrCategory ? 600 : 1200);

        let systemPrompt = '';
        const NO_THOUGHT_INSTRUCTION_EN = "You may reason inside <think>...</think> tags. Keep your thinking block extremely short (at most 1-2 sentences/30 tokens) to save time before providing your final response in English.";
        const NO_THOUGHT_INSTRUCTION_HI = "You may reason inside <think>...</think> tags. Keep your thinking block extremely short (at most 1-2 sentences/30 tokens) to save time. IMPORTANT: You must write your final response ENTIRELY in Hindi.";

        // Set up location instructions (excluding registry canonical names so it doesn't hallucinate layout)
        let hiLocRule = '';
        let enLocRule = '';
        const registryCanonicalNames = (venue.registry?.canonicalNames || venue.zooRegistry?.canonicalNames || []);
        if (isLocationIntent && references.length > 0) {
            const locationRefs = references.filter(ref => !registryCanonicalNames.includes(ref));
            if (locationRefs.length > 0) {
                const refNames = locationRefs.join(', ');
                hiLocRule = `\n5. महत्वपूर्ण: उपयोगकर्ता पूछ रहा है कि वे कहाँ मिल सकते हैं। उन्हें सीधे बताएं कि वे ${venue.hindiVenueType} में निम्नलिखित स्थानों पर मिल सकते हैं: ${refNames}।`;
                enLocRule = `\n5. IMPORTANT: The user is asking where to find them. Explicitly state they can be found at the following locations: ${refNames}.`;
            }
        }

        if (matchedFacility === 'Timings & Hours') {
            const closedRuleHi = venue.closedDay ? `2. ${venue.hindiVenueType} ${venue.hindiClosedDay} को बंद रहता है।` : '';
            const closedRuleEn = venue.closedDay ? `2. The ${venue.venueType} is strictly CLOSED on ${venue.closedDay}s.` : '';

            systemPrompt = isHindi
                ? `आप ${venue.personaName} हैं, जो ${venue.hindiVenueName} के मार्गदर्शक हैं।
${NO_THOUGHT_INSTRUCTION_HI}
संदर्भ (Context): ${trimmedContext}
नियम:
1. केवल संदर्भ का उपयोग करके समय संबंधी प्रश्न का उत्तर दें।
${closedRuleHi}
3. जवाब केवल 1 या 2 वाक्यों में दें। अंत में एक इमोजी लगाएं ${venue.defaultEmoji}।`
                : `You are ${venue.personaName}, the guide at ${venue.venueName}.
${NO_THOUGHT_INSTRUCTION_EN}
Context: ${trimmedContext}
Rules:
1. Answer timing questions using ONLY the context.
${closedRuleEn}
3. Limit response strictly to 1 or 2 sentences. End with exactly one emoji ${venue.defaultEmoji}.`;
        } else {
            systemPrompt = isHindi
                ? `आप ${venue.hindiVenueName} के गाइड '${venue.personaName}' हैं।
${NO_THOUGHT_INSTRUCTION_HI}
संदर्भ (Context): ${trimmedContext}
नियम:
1. दिए गए संदर्भ का उपयोग करके सीधे उपयोगकर्ता के सवाल का जवाब दें।
2. जवाब केवल 1 या 2 वाक्यों में (अधिकतम 25 शब्द) रखें। अंत में एक इमोजी लगाएं ${venue.defaultEmoji}।${hiLocRule}`
                : `You are ${venue.personaName}, the friendly guide at the ${venue.venueName}.
${NO_THOUGHT_INSTRUCTION_EN}
Context: ${trimmedContext}
Rules:
1. Answer factually using ONLY the provided context.
2. Limit response strictly to at most 1 or 2 sentences maximum (strictly under 30 words). End with exactly one relevant emoji ${venue.defaultEmoji}.${enLocRule}`;
        }

        // 7. Call LLM (Ollama)
        let userMessageContent = question;
        if (finalSubject && finalSubject !== 'general') {
            const isMultiSubject = finalSubject.includes(',');
            if (isMultiSubject) {
                const subjects = finalSubject.split(',').map(s => s.trim()).filter(Boolean);
                const subjectList = subjects.map(s => `"${s}"`).join(' and ');
                userMessageContent = `[Topics: ${finalSubject}] The user asked: "${userMessageContent}".\nInstruction: Briefly mention ONE key fact about EACH of ${subjectList} in your reply. Keep it to 1 short sentence per item.`;
            } else {
                userMessageContent = `[Topic: ${finalSubject}] The user just said: "${userMessageContent}". \nInstruction: Reply to the user.`;
            }
        }

        const chatMessages = [
            { role: 'system', content: systemPrompt }
        ];
        if (Array.isArray(history) && history.length > 0) {
            chatMessages.push(...history.slice(-4));
        }
        chatMessages.push({ role: 'user', content: userMessageContent });

        if (stream && res) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);

            let fullAnswer = '';
            try {
                const streamResp = await this.ollama.chat({
                    model: this.chatModel,
                    messages: chatMessages,
                    stream: true,
                    options: { temperature: 0.3, num_predict: 128 }
                });

                const filter = new ThinkingFilter();
                for await (const chunk of streamResp) {
                    const token = chunk.message?.content || '';
                    fullAnswer += token;
                    const filteredToken = filter.filter(token);
                    if (filteredToken) {
                        res.write(`data: ${JSON.stringify({ token: filteredToken })}\n\n`);
                    }
                }

                // Final post-process and terminate stream
                let finalAnswer = fullAnswer.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
                // Apply Hindi Glossary fixes
                if (isHindi) {
                    for (const [glKey, glVal] of Object.entries(venue.hindiGlossaryReplacements || {})) {
                        finalAnswer = finalAnswer.replace(new RegExp(glKey, 'gi'), glVal);
                    }
                }
                finalAnswer = truncateToTwoSentences(finalAnswer, finalSubject.includes(',') ? 2 : 1);

                res.write(`data: ${JSON.stringify({ done: true, keyword: finalSubject, references })}\n\n`);
                return res.end();
            } catch (err) {
                console.error('[SDK] Streaming failed:', err.message);
                res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
                return res.end();
            }
        } else {
            try {
                console.log('[SDK] [DEBUG] Calling Ollama Chat model...');
                const resp = await this.ollama.chat({
                    model: this.chatModel,
                    messages: chatMessages,
                    stream: false,
                    options: { temperature: 0.3, num_predict: 128 }
                });
                console.log('[SDK] [DEBUG] Ollama Chat response received.');

                let answer = (resp.message?.content || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim();

                // Apply Glossary
                if (isHindi) {
                    for (const [glKey, glVal] of Object.entries(venue.hindiGlossaryReplacements || {})) {
                        answer = answer.replace(new RegExp(glKey, 'gi'), glVal);
                    }
                }
                answer = truncateToTwoSentences(answer, finalSubject.includes(',') ? 2 : 1);

                const finalObj = { answer, keyword: finalSubject, references };
                this.responseCache.set(cacheKey, finalObj);
                return finalObj;
            } catch (err) {
                console.error('[SDK] Chat generation failed:', err.message);
                return { answer: `Failed to generate response: ${err.message}`, keyword: 'general', references: [] };
            }
        }
    }
}

module.exports = ChatEngine;
