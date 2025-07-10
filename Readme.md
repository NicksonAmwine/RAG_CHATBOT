# RAG Chatbot - Document Querying System

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and ChromaDB that allows users to upload documents and ask questions about their content using Google's Gemini AI.

## Project Overview

This RAG system enables users to:
- Upload text documents for analysis
- Ask questions about document content in natural language
- Receive contextually relevant answers based on document content
- Maintain conversation history for context-aware responses
- Process large documents efficiently through intelligent chunking

### Chosen Document
The system comes pre-configured with Leo Tolstoy's "War and Peace" as the default document for demonstration purposes.

### Component Interactions:

1. **Streamlit**: Provides the web interface and handles user interactions
2. **LangChain**: Manages document processing, embedding generation, and LLM interactions
3. **ChromaDB**: Stores document embeddings and performs similarity searches
4. **Google Gemini**: Generates embeddings and provides AI responses

## Technical Specifications

### Chunking Strategy
- **Method**: RecursiveTokenChunker from chunking_evaluation library
- **Chunk Size**: 3000 tokens
- **Overlap**: 50 tokens
- **Separators**: `["\n\n", "\n", " ", ""]`

**Rationale**: This chunking approach balances context preservation with processing efficiency. The recursive nature ensures natural breaks at paragraph and sentence boundaries, while the overlap maintains context continuity between chunks.

### Embedding Model
- **Model**: Google's `text-embedding-3-small` (via `models/embedding-001`)
- **Dimensions**: 768
- **Provider**: Google Generative AI

**Choice Justification**: Google's embedding model provides high-quality semantic understanding while being cost-effective and well-integrated with the Gemini ecosystem.

### Language Model
- **Model**: Google Gemini 2.0 Flash
- **Context Window**: 32,768 tokens
- **Temperature**: 0.1 (for consistent, factual responses)

## Setup Instructions

### Prerequisites
- Docker and Docker Compose installed
- Google AI API key
- Git (for cloning the repository)

### 1. Clone the Repository
```bash
git clone https://github.com/NicksonAmwine/RAG_CHATBOT.git
cd AFTA_chatbot
```

2. Set Up Environment Variables
Create a chatbot.env file in the project root:

```bash
# chatbot.env
GOOGLE_API_KEY=your_google_ai_api_key_here
```

4. Build the Docker Image

```bash
docker compose build
```
This command will:

Create a Docker image based on Python 3.11
Install all required dependencies
Set up the application environment

5. Run the Application Stack
```bash
docker compose up
```
6. Access the Application
Open your web browser and navigate to:

```bash
http://localhost:8501
```

Usage Instructions
1. Process a Document: Click "Process War and Peace" in the sidebar to load the default document
2. Upload Custom Document: Use the file uploader to process your own text files
3. Ask Questions: Type questions in the chat interface
4. View History: Scroll through the conversation history
5. Clear Chat: Use the clear button to reset the conversation