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
import os
from typing import List
from langchain.memory import ConversationBufferMemory

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
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

st.title("MY RAG CHATBOT")
st.subheader("Document Querying System")

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
        separators=["\n\n", "\n", " ", ""],
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

def create_or_load_vector_store(chunk_documents=None, document_name=None) -> chromadb.Collection:
    """Create or load ChromaDB vector store"""
    try:
        try:
            collection = chroma_client.get_collection(name="chunked_documents")
            # Check if the existing collection matches the document name
            if collection.count() > 0 and st.session_state.get('current_document') == document_name:
                return collection 
            else:
                chroma_client.delete_collection(name="chunked_documents")
        except:
            pass 
        
        if chunk_documents is None:
            raise ValueError("No chunk documents provided for new collection")
        
        collection = chroma_client.create_collection(name="chunked_documents")
        
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
        
        return collection
        
    except Exception as e:
        raise RuntimeError(f"Error creating vector store: {str(e)}")

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
        history = memory_vars.get("history", [])
        
        # Format history for the prompt
        history_str = ""
        if history:
            history_str = "\n".join([
                f"Human: {msg.content}" if hasattr(msg, 'type') and msg.type == 'human' 
                else f"Assistant: {msg.content}" if hasattr(msg, 'content') 
                else str(msg) for msg in history[-6:]
            ])
    except:
        history_str = ""
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert assistant answering questions about uploaded documents."
    Use the following context from the uploaded documents to provide a detailed and accurate answer. Try to be concise and relevant.
    You can derive deductions from the provided context to answer questions but try to stay within the context for the document. Make the conversation to flow naturally and try not to sound like a robot.

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
            {"user": question},
            {"bot": response.content}
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
        # Read uploaded file
        text = str(uploaded_file.read(), "utf-8")
        
        # Save to temporary file
        os.makedirs("/app/uploaded_documents", exist_ok=True)
        with open(f"/app/uploaded_documents/{uploaded_file.name}", "w", encoding="utf-8") as f:
            f.write(text)
        
        # Process the document
        chunk_documents = chunk_document(text)
        
        # Create new vector store
        collection = create_or_load_vector_store(chunk_documents)
        
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

# Sidebar for document processing
with st.sidebar:
    st.header("Document Management")
    
    # Default document processing
    if st.button("Process War and Peace"):
        with st.spinner("Processing War and Peace..."):
            try:
                collection = create_or_load_vector_store()
                if collection.count() > 0:
                    st.session_state.collection = collection
                    st.session_state.document_processed = True
                    st.session_state.current_document = "war_and_peace.txt"
                    st.success("Existing document loaded successfully!")
                else:
                    text = load_document("war_and_peace.txt")
                    if text:
                        chunk_documents = chunk_document(text)
                        st.info(f"Created {len(chunk_documents)} chunks")
                        
                        collection = create_or_load_vector_store(chunk_documents)
                        if collection:
                            st.session_state.collection = collection
                            st.session_state.document_processed = True
                            st.session_state.current_document = "war_and_peace.txt"
                            st.success("Document processed successfully!")
            except:
                text = load_document("war_and_peace.txt")
                if text:
                    chunk_documents = chunk_document(text)
                    st.info(f"Created {len(chunk_documents)} chunks")
                    
                    collection = create_or_load_vector_store(chunk_documents)
                    if collection:
                        st.session_state.collection = collection
                        st.session_state.document_processed = True
                        st.session_state.current_document = "war_and_peace.txt"
                        st.success("Document processed successfully!")
    
    # File upload option
    st.subheader("Upload Your Own Document")
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    
    if uploaded_file is not None:
        if st.button("Process Uploaded Document"):
            with st.spinner("Processing uploaded document..."):
                if process_uploaded_file(uploaded_file):
                    st.success("Uploaded document processed successfully!")
    
    # Display current document
    if st.session_state.document_processed:
        st.info(f"Current document: {st.session_state.current_document}")
        st.info(f"Status: Ready for queries")

render_chat_interface(similarity_search, generate_answer)