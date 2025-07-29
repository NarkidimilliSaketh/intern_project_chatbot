// server/services/documentProcessor.js

const fs = require('fs');
const path = require('path');
const pdf = require('pdf-parse');
const mammoth = require('mammoth');

const MAX_PDF_PAGES = 20; 

class DocumentProcessor {
    constructor(vectorStore) {
        if (!vectorStore) {
            throw new Error("DocumentProcessor requires a VectorStore instance.");
        }
        this.chunkSize = 512;
        this.chunkOverlap = 100;
        this.vectorStore = vectorStore;
    }

    async parseFile(filePath) {
        // ... (this function remains the same)
        const ext = path.extname(filePath).toLowerCase();
        try {
            let text = '';
            switch (ext) {
                case '.txt':
                    text = fs.readFileSync(filePath, 'utf-8');
                    break;
                case '.pdf':
                    console.log(`📄 Parsing PDF: ${path.basename(filePath)}`);
                    const dataBuffer = fs.readFileSync(filePath);
                    const options = { max: MAX_PDF_PAGES };
                    const data = await pdf(dataBuffer, options);
                    text = data.text;
                    console.log(`✅ Parsed first ${data.numpages > 0 ? data.numpages : 'many'} pages of PDF.`);
                    break;
                case '.docx':
                    const result = await mammoth.extractRawText({ path: filePath });
                    text = result.value;
                    break;
                default:
                    console.warn(`Unsupported file type for parsing: ${ext}. Skipping content extraction.`);
                    return '';
            }
            return text || '';
        } catch (error) {
            console.error(`Error parsing file ${filePath}:`, error.message);
            return '';
        }
    }

    // --- MODIFICATION: Add filename context to each chunk ---
    chunkText(text, filename) {
        if (typeof text !== 'string' || !text.trim()) {
            return [];
        }

        const chunks = [];
        let startIndex = 0;
        let chunkIndex = 0;

        while (startIndex < text.length) {
            const endIndex = Math.min(startIndex + this.chunkSize, text.length);
            let chunkTextSlice = text.slice(startIndex, endIndex);

            if (endIndex < text.length) {
                const lastSpace = chunkTextSlice.lastIndexOf(' ');
                if (lastSpace > 0) {
                    chunkTextSlice = chunkTextSlice.substring(0, lastSpace);
                }
            }
            
            // Prepend the document title to each chunk for better context
            const chunkWithContext = `Source Document: "${filename}"\n\nContent: ${chunkTextSlice.trim()}`;

            chunks.push({
                pageContent: chunkWithContext, // Use the content with the title
                metadata: {
                    source: filename,
                    chunkId: `${filename}_chunk_${chunkIndex}`
                }
            });

            const actualEndIndex = startIndex + chunkTextSlice.length;
            startIndex = actualEndIndex - this.chunkOverlap;
            if (startIndex <= actualEndIndex - chunkTextSlice.length) {
                startIndex = actualEndIndex;
            }
            chunkIndex++;
        }
        return chunks.filter(chunk => chunk.pageContent.length > 0);
    }

    async processFile(filePath, options = {}) {
        try {
            console.log(`📄 Processing file: ${options.originalName}`);
            const text = await this.parseFile(filePath);
            
            if (!text) {
                console.warn(`⚠️ No text content extracted from file: ${options.originalName}`);
                return { success: true, chunksAdded: 0, message: 'File had no readable content.' };
            }

            const chunks = this.chunkText(text, options.originalName);

            if (chunks.length === 0) {
                return { success: true, chunksAdded: 0, message: 'File had no content to chunk.' };
            }

            // Add user and file ID to metadata for filtering
            const documentsToStore = chunks.map(chunk => ({
                pageContent: chunk.pageContent,
                metadata: {
                    ...chunk.metadata,
                    userId: options.userId,
                    fileId: options.fileId,
                    fileName: options.originalName // Explicitly add fileName
                }
            }));

            const result = await this.vectorStore.addDocuments(documentsToStore);
            return { success: true, chunksAdded: result.count };
            
        } catch (error) {
            console.error(`❌ Error during processFile for ${options.originalName}:`, error.message);
            throw error;
        }
    }
}

module.exports = DocumentProcessor;