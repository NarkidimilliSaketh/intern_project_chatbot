// server/services/HybridRagService.js
const serviceManager = require('./serviceManager');
const User = require('../models/User');
const File = require('../models/File'); // Import the File model
const DocumentProcessor = require('./documentProcessor'); // Import DocumentProcessor

const RAG_CONFIDENCE_THRESHOLD = 0.65;

class HybridRagService {
    // --- NEW: Query Router ---
    async processQuery(query, userId, fileId = null) {
        const { geminiAI } = serviceManager.getServices();
        
        // Step 1: Determine the user's intent
        const queryAnalysis = await geminiAI.determineQueryType(query);
        console.log(`[RAG Router] Query classified as: ${queryAnalysis.type}. Reason: ${queryAnalysis.reason}`);

        // Step 2: Route to the appropriate handler
        if (queryAnalysis.type === 'broad') {
            return this._handleBroadQuery(query, userId, fileId);
        } else {
            return this._handleSpecificQuery(query, userId, fileId);
        }
    }

    // --- NEW: Handler for broad queries like "summarize this" ---
    async _handleBroadQuery(query, userId, fileId) {
        console.log('[RAG Service] Handling broad query with summarization strategy.');
        const { geminiAI, documentProcessor } = serviceManager.getServices();

        if (!fileId) {
            return {
                message: "To get a summary, please first select a specific file to chat with from the 'My Files' menu.",
                metadata: { searchType: 'summary_requires_file', sources: [] }
            };
        }

        try {
            const file = await File.findOne({ _id: fileId, user: userId });
            if (!file) {
                throw new Error('File not found for summarization.');
            }

            // Use the document processor to get the full text content
            const fileContent = await documentProcessor.parseFile(file.path);

            if (!fileContent || fileContent.trim().length < 100) {
                 return {
                    message: "The selected document doesn't have enough content to generate a summary.",
                    metadata: { searchType: 'summary_insufficient_content', sources: [{ title: file.originalname, type: 'document' }] }
                };
            }

            const summary = await geminiAI.summarizeDocumentContent(fileContent, query);

            return {
                message: summary,
                metadata: {
                    searchType: 'summary',
                    sources: [{ title: file.originalname, type: 'document' }]
                }
            };
        } catch (error) {
            console.error('Error in _handleBroadQuery:', error);
            return {
                message: "I encountered an error while trying to summarize the document. Please try again.",
                metadata: { searchType: 'summary_error', sources: [] }
            };
        }
    }

    // --- REFACTORED: The original logic is now the specific query handler ---
    async _handleSpecificQuery(query, userId, fileId = null) {
        console.log('[RAG Service] Handling specific query with vector search strategy.');
        const { vectorStore, geminiAI } = serviceManager.getServices();

        const correctedQuery = await this.correctAndClarifyQuery(query);

        const filters = { userId };
        if (fileId) {
            filters.fileId = fileId;
            console.log(`[RAG Service] Scoping search to fileId: ${fileId}`);
        } else {
            console.log(`[RAG Service] Searching across all documents for user: ${userId}`);
        }

        const relevantChunks = await vectorStore.searchDocuments(correctedQuery, {
            limit: 5,
            filters: filters
        });

        console.log(`[RAG Service] Found ${relevantChunks.length} relevant chunks.`);
        if (relevantChunks.length > 0) {
            console.log('Top chunks and scores:');
            relevantChunks.forEach((chunk, i) => {
                console.log(`  ${i + 1}. Score: ${chunk.score.toFixed(4)} | Content: "${chunk.content.substring(0, 80)}..."`);
            });
        }

        const isContextSufficient = relevantChunks.length > 0 && relevantChunks[0].score > RAG_CONFIDENCE_THRESHOLD;

        if (isContextSufficient) {
            console.log(`[RAG Service] Context sufficient (top score: ${relevantChunks[0].score.toFixed(4)}). Answering from document(s).`);
            const context = relevantChunks.map(chunk => chunk.content).join('\n\n');
            
            const user = await User.findById(userId);
            const personalizationProfile = user ? user.personalizationProfile : '';
            
            const prompt = geminiAI.buildRagPrompt(correctedQuery, context, personalizationProfile);
            
            const answer = await geminiAI.generateText(prompt);
            return {
                message: answer,
                metadata: { 
                    searchType: 'rag', 
                    sources: this.formatSources(relevantChunks),
                    source_count: relevantChunks.length // Add source count
                }
            };
        } else {
            const reason = relevantChunks.length === 0 ? 'No relevant chunks found' : `Top score (${relevantChunks.length > 0 ? relevantChunks[0].score.toFixed(4) : 'N/A'}) is below threshold of ${RAG_CONFIDENCE_THRESHOLD}`;
            console.log(`[RAG Service] Context insufficient. Reason: ${reason}. Returning fallback message.`);
            
            const fallbackMessage = fileId
                ? "I couldn't find a confident answer for that in the selected document. Please try rephrasing your question with more specific keywords."
                : "I couldn't find a confident answer for that in your uploaded documents. Please try rephrasing your question or uploading more relevant files.";
            return {
                message: fallbackMessage,
                metadata: { searchType: 'rag_fallback', sources: [] }
            };
        }
    }

    async correctAndClarifyQuery(query) {
        const { geminiAI } = serviceManager.getServices();
        try {
            const prompt = `Correct any spelling mistakes in the following user query. Return ONLY the corrected query, nothing else. Do not answer the question. Original Query: "${query}" Corrected Query:`;
            const correctedQuery = await geminiAI.generateText(prompt);
            const finalQuery = correctedQuery.trim().replace(/["*]/g, '');
            console.log(`[Query Correction] Original: "${query}" -> Corrected: "${finalQuery}"`);
            return finalQuery;
        } catch (error) {
            console.error("Error during query correction, using original query.", error);
            return query;
        }
    }
    
    formatSources(chunks) {
        const uniqueSources = new Map();
        chunks.forEach(chunk => {
            const fileName = chunk.metadata.fileName || 'Unknown Document';
            if (!uniqueSources.has(fileName)) {
                uniqueSources.set(fileName, { title: fileName, type: 'document' });
            }
        });
        return Array.from(uniqueSources.values());
    }
}

module.exports = new HybridRagService();