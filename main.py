import streamlit as st
from dotenv import dotenv_values
from langchain_core.messages import HumanMessage
from app import render_chat_interface
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from chunking_evaluation.chunking import RecursiveTokenChunker
import chromadb
import uuid
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pydantic import SecretStr
import os, io, PyPDF2
from typing import List, Optional
from langchain.memory import ConversationBufferWindowMemory

config = dotenv_values("chatbot.env")
gemini_api_key = config.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not gemini_api_key:
    st.error("GOOGLE_API_KEY not found! Please check your chatbot.env file.")
    st.stop()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=SecretStr(gemini_api_key) if gemini_api_key is not None else None
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=SecretStr(gemini_api_key) if gemini_api_key is not None else None
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="/app/chroma_db")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5,
        return_messages=True,
        memory_key="chat_history"
        )


st.markdown("""
<div class="custom-header">
    <h1>🤖 Intelligent Document Assistant</h1>
    <p>Upload documents and chat with AI-powered insights • Powered by Google Gemini</p>
</div>
""", 
unsafe_allow_html=True)

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF file."""
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Create PDF reader
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        
        # Extract text from all pages
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text + "\n"
            except Exception as e:
                st.warning(f"Could not extract text from page {page_num + 1}: {e}")
                continue
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF")
            
        return text.strip()
        
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def load_document(file_path: str) -> str:
    """Load document from file path."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        elif os.path.exists(f"/app/{file_path}"):
            with open(f"/app/{file_path}", "r", encoding="utf-8") as file:
                return file.read()
        else:
            st.error(f"File not found: {file_path}")
            return ""
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return ""

def chunk_document(text: str) -> List[Document]:
    """Split document into chunks."""
    text_splitter = RecursiveTokenChunker(
        chunk_size=3000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "?"],
    )
    
    chunks = text_splitter.split_text(text)
    
    chunk_documents = []
    for i, chunk in enumerate(chunks):
        chunk_doc = Document(
            page_content=chunk,
            metadata={
                'chunk_id': str(uuid.uuid4()),
                'chunk_index': i,
            }
        )
        chunk_documents.append(chunk_doc)
    return chunk_documents

def create_or_load_vector_store(chunk_documents=None, document_name=None) -> Optional[chromadb.Collection]:
    """Create or load ChromaDB vector store, considering the document name."""
    try:
        # If no document name provided, return None
        if document_name is None:
            st.error("Document name is required")
            return None
            
        # Check if collection exists and matches document_name
        try:
            collection = chroma_client.get_collection(name="chunked_documents")
            collection_metadata = collection.metadata or {}
            stored_doc_name = collection_metadata.get("document_name")
            
            if collection.count() > 0 and stored_doc_name == document_name:
                # st.sidebar.info(f"✅ Reusing existing collection for: {document_name}")
                return collection 
            elif collection.count() > 0 and stored_doc_name != document_name:
                # st.sidebar.info(f"🔄 Replacing collection (was: {stored_doc_name}, now: {document_name})")
                chroma_client.delete_collection(name="chunked_documents")
        except:
            st.sidebar.info(f"📝 Creating new collection for: {document_name}")
            pass  
        
        if chunk_documents is None:
            return None
        
        collection = chroma_client.create_collection(
            name="chunked_documents",
            metadata={"document_name": document_name}
        )
        
        # Generate embeddings for all chunks
        texts = [chunk.page_content for chunk in chunk_documents]
        chunk_embeddings = embeddings.embed_documents(texts)
        chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
        
        # Prepare data for ChromaDB
        ids = [chunk.metadata['chunk_id'] for chunk in chunk_documents]
        metadatas = [chunk.metadata for chunk in chunk_documents]
        
        # Add to collection
        collection.add(
            embeddings=chunk_embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        st.sidebar.success(f"✅ Created collection with {len(texts)} chunks")
        return collection
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None
    
def similarity_search(collection: chromadb.Collection, query: str, k: int = 8) -> List[str]:
    """Search for similar documents in the vector store."""
    try:
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results['documents'][0] if results['documents'] else []
        
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def generate_answer(context: str, question: str,) -> str:
    """Generate answer using LLM with retrieved context."""

    # Get conversation history
    try:
        memory_vars = st.session_state.memory.load_memory_variables({})
        history = memory_vars.get("chat_history", [])
        
        # Format history for the prompt
        history_str = ""
        if history:
            formatted_messages = []
            for msg in history:
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        formatted_messages.append(f"Human: {msg.content}")
                    elif msg.type == 'ai':
                        formatted_messages.append(f"Assistant: {msg.content}")
                elif hasattr(msg, 'content'):
                    formatted_messages.append(f"Message: {msg.content}")
            
            history_str = "\n".join(formatted_messages)
    except:
        history_str = ""
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert assistant answering questions about uploaded documents."
    Use the following context from the uploaded documents to provide a detailed and accurate answer. Try to be concise and relevant.
    You can derive deductions from the provided context to answer questions but be sure to stay within the context of the document. Make the conversation flow naturally and try not to sound like a robot.

    Conversation History:
    {history}

    Document context:
    {context}

    Question:
    {question}

    Answer based on the context above
    """)
    
    try:
        # Get chat history from memory
        prompt = prompt_template.format(
            context=context, 
            question=question, 
            history=history_str
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Save to memory
        st.session_state.memory.save_context(
            {"input": question},
            {"output": response.content}
        )
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list):
            return "\n".join(str(item) for item in response.content)
        else:
            return str(response.content)
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def process_uploaded_file(uploaded_file) -> bool:
    """Process uploaded text file and update vector store."""
    try:
        if uploaded_file.name.endswith('.pdf'):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("Could not extract text from PDF file")
                return False
        elif uploaded_file.name.endswith('.txt'):
            # Read text file
            uploaded_file.seek(0)
            text = str(uploaded_file.read(), "utf-8")
        else:
            st.error("Unsupported file type")
            return False
        
        if not text.strip():
            st.error("Document appears to be empty")
            return False
        
        # Save to temporary file
        os.makedirs("/app/uploaded_documents", exist_ok=True)
        base_name = os.path.splitext(uploaded_file.name)[0]
        save_path = f"/app/uploaded_documents/{base_name}.txt"
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Process the document
        chunk_documents = chunk_document(text)
        
        # Create new vector store
        collection = create_or_load_vector_store(chunk_documents, document_name=uploaded_file.name)
        
        if collection:
            st.session_state.collection = collection
            st.session_state.document_processed = True
            st.session_state.current_document = uploaded_file.name
            return True
        
        return False
        
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return False

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "collection" not in st.session_state:
    st.session_state.collection = None
    st.session_state.document_processed = False
    st.session_state.current_document = "war_and_peace.txt"

if not st.session_state.document_processed:
    try:
        collection = chroma_client.get_collection(name="chunked_documents")
        collection_metadata = collection.metadata or {}
        stored_doc_name = collection_metadata.get("document_name", "war_and_peace.txt")
        if collection.count() > 0:
            st.session_state.collection = collection
            st.session_state.document_processed = True
            st.session_state.current_document = stored_doc_name
            st.sidebar.success(f"Auto-loaded: {stored_doc_name}")
    except:
        pass

# Sidebar for document processing
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h1 style="color: white; margin: 0;">Document Manager</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Default document processing
    st.markdown("### 📖 Default Document")
    if st.button("✨ Process War and Peace", use_container_width=True):
        with st.spinner("Processing War and Peace..."):
            collection = create_or_load_vector_store(document_name="war_and_peace.txt")
            if collection and collection.count() > 0:
                st.session_state.collection = collection
                st.session_state.document_processed = True
                st.session_state.current_document = "war_and_peace.txt"
                st.success("Existing document loaded successfully!")
            else:
                text = load_document("war_and_peace.txt")
                if text:
                    chunk_documents = chunk_document(text)
                    st.info(f"Created {len(chunk_documents)} chunks")
                    
                    collection = create_or_load_vector_store(chunk_documents, "war_and_peace.txt")
                    if collection:
                        st.session_state.collection = collection
                        st.session_state.document_processed = True
                        st.session_state.current_document = "war_and_peace.txt"
                        st.success("Document processed successfully!")
                    else:
                        st.error("Failed to create vector store.")
                else:
                    st.error("Failed to load War and Peace document.")

    # File upload option
    st.markdown("### 📤 Upload Custom Document")
    uploaded_file = st.file_uploader("Choose a text or PDF file", type=["txt", "pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        uploaded_file.seek(0)
        if st.button("✨ Process Uploaded Document", use_container_width=True):
            with st.spinner("Processing uploaded document..."):
                if process_uploaded_file(uploaded_file):
                    st.success("Uploaded document processed successfully!")
                else:
                    st.error("Failed to process uploaded document")

    if st.session_state.document_processed:
        st.markdown(f"""
        <div class="metric-card">
            <p>📄 Current Document: <span style="color: green;">{st.session_state.current_document}</span></p>
            <p>📊 {st.session_state.collection.count() if st.session_state.collection else 0:,} chunks available</p>
            <p>Status: <span style="color: green;">Ready</span></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <h4>⚠️ No Document Loaded</h4>
            <p>Process a document below to get started</p>
        </div>
        """, unsafe_allow_html=True)

def handle_user_query(user_query):
    """Handle user query submission."""
    with st.spinner("🔍 Processing your question..."):
        if st.session_state.collection is not None:
            retrieved_docs = similarity_search(st.session_state.collection, user_query, k=8)
            
            if retrieved_docs:
                context = "\n\n".join(retrieved_docs)
                answer = generate_answer(context, user_query)
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "human": user_query,
                    "ai": answer,
                    "context_chunks": len(retrieved_docs)
                })
                
                st.rerun()
            else:
                st.error("No relevant information found in the document. Try rephrasing your question.")
        else:
            st.error("No document collection found. Please process a document first.")

# Main interface logic
user_action = render_chat_interface()

if user_action:
    if user_action["action"] == "submit":
        handle_user_query(user_action["query"])
    elif user_action["action"] == "clear":
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.success("Chat history cleared!")
        st.rerun()