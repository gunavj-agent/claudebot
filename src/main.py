import os
import time
import uuid
import json
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from anthropic import Anthropic

# RAG imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropicMessages
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("notif_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("notif_chatbot")

# Load environment variables
load_dotenv()

# Get token limits from environment variables or use defaults
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", 2000))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 1000))

# Initialize FastAPI app
app = FastAPI(title="Notif Chatbot")

# Get the current directory and set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
templates_dir = os.path.join(current_dir, "templates")
uploads_dir = os.path.join(os.path.dirname(current_dir), "uploads")

# Create uploads directory if it doesn't exist
os.makedirs(uploads_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize templates
templates = Jinja2Templates(directory=templates_dir)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
vectorstore = None
vectorstore_path = os.path.join(os.path.dirname(current_dir), "vectorstore")

# Create vectorstore directory if it doesn't exist
os.makedirs(vectorstore_path, exist_ok=True)

# Try to load existing vectorstore if it exists
try:
    if os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
        logger.info(f"Loading existing vector store from {vectorstore_path}")
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Vector store loaded with {vectorstore.index.ntotal} embeddings")
    else:
        logger.info("No existing vector store found")
except Exception as e:
    logger.error(f"Error initializing vector store: {str(e)}")
    vectorstore = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str = Field(..., max_length=8000)  # Approx 2000 tokens
    max_tokens: Optional[int] = Field(default=1000, le=MAX_OUTPUT_TOKENS)  # Limit output tokens
    use_rag: Optional[bool] = Field(default=False)  # Whether to use RAG

class ChatResponse(BaseModel):
    response: str

class UploadResponse(BaseModel):
    uploaded_files: List[dict]

class UploadedFile(BaseModel):
    filename: str
    file_id: str
    file_type: str

# Helper functions for document processing
def process_document(file_path, file_type):
    """Process a document and add it to the vector store."""
    global vectorstore
    
    try:
        logger.info(f"Processing document: {file_path} (type: {file_type})")
        
        # Load the document based on file type
        if file_type == 'pdf':
            logger.info(f"Loading PDF document")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_type == 'txt':
            logger.info(f"Loading TXT document")
            loader = TextLoader(file_path)
            documents = loader.load()
        else:
            # For unsupported file types, create a simple document
            logger.info(f"Loading unsupported file type as raw text")
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
                documents = [Document(page_content=content, metadata={"source": file_path})]
        
        logger.info(f"Loaded {len(documents)} document(s)")
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Add metadata to each chunk
        for chunk in chunks:
            chunk.metadata["file_id"] = str(uuid.uuid4())
            chunk.metadata["source_file"] = os.path.basename(file_path)
        
        # Initialize embeddings
        logger.info(f"Initializing HuggingFaceEmbeddings")
        embeddings = HuggingFaceEmbeddings()
        
        # Create or update the vector store
        if vectorstore is None:
            logger.info(f"Creating new FAISS vector store")
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            logger.info(f"Updating existing FAISS vector store")
            vectorstore.add_documents(chunks)
        
        # Save the vector store
        logger.info(f"Saving vector store to {vectorstore_path}")
        vectorstore.save_local(vectorstore_path, allow_dangerous_deserialization=True)
        
        logger.info(f"Document processing complete")
        return True
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/test-upload")
async def test_upload(request: Request):
    return templates.TemplateResponse("test_upload.html", {"request": request})

@app.get("/vector-view")
async def vector_view(request: Request):
    """Serve the vector info HTML page"""
    return templates.TemplateResponse("vector_info.html", {"request": request})

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    logger.info(f"Upload endpoint called with {len(files)} files")
    
    uploaded_files = []
    
    for file in files:
        try:
            # Get file info
            filename = file.filename
            file_type = filename.split('.')[-1].lower()
            logger.info(f"Processing file: {filename}")
            logger.info(f"File type: {file_type}")
            
            # Generate a unique ID for the file
            file_id = str(uuid.uuid4())
            
            # Create the uploads directory if it doesn't exist
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Define the file path
            file_path = os.path.join(uploads_dir, f"{file_id}.{file_type}")
            logger.info(f"Saving to: {file_path}")
            
            # Save the file
            contents = await file.read()
            logger.info(f"Read {len(contents)} bytes from file")
            
            with open(file_path, "wb") as f:
                f.write(contents)
            
            # Process the document for RAG
            logger.info("Processing document for RAG")
            success = process_document(file_path, file_type)
            
            if success:
                logger.info(f"Successfully processed {filename}")
                uploaded_files.append({
                    "filename": filename,
                    "file_id": file_id,
                    "file_type": file_type
                })
            else:
                logger.info(f"Failed to process {filename} for RAG")
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Returning {len(uploaded_files)} successfully processed files")
    return {"uploaded_files": uploaded_files}

@app.get("/vector-info")
async def vector_info():
    """Return information about the vector store"""
    global vectorstore
    
    if vectorstore is None:
        return {
            "status": "error",
            "message": "No vector store available",
            "document_count": 0,
            "documents": []
        }
    
    try:
        # Get the number of documents in the vector store
        doc_count = vectorstore.index.ntotal if hasattr(vectorstore, 'index') else 0
        
        # Get document information
        documents = []
        if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
            for doc_id, doc in vectorstore.docstore._dict.items():
                if hasattr(doc, 'metadata'):
                    doc_info = {
                        "id": str(doc_id),
                        "metadata": doc.metadata,
                        "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    }
                    documents.append(doc_info)
        
        return {
            "status": "success",
            "message": "Vector store information retrieved successfully",
            "document_count": doc_count,
            "documents": documents
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving vector store information: {str(e)}",
            "document_count": 0,
            "documents": []
        }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Check if API key is set (without logging it)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY environment variable is not set")
            raise HTTPException(status_code=500, detail="API key not configured")
        logger.info("API key is configured")

        # Validate message length
        if len(request.message) > 8000:  # Approx 2000 tokens
            raise HTTPException(
                status_code=400,
                detail=f"Message too long. Please limit to {MAX_INPUT_TOKENS} tokens (approximately 8000 characters)"
            )

        # Create a base system message
        system_message = "You are a helpful AI assistant."
        context = ""
        use_rag = False
        
        # Debug: Print information about the request and vector store
        logger.info("========== RAG DEBUG INFORMATION ==========")
        logger.info(f"User query: {request.message}")
        logger.info(f"Vector store exists: {vectorstore is not None}")
        if vectorstore is not None:
            logger.info(f"Documents in vector store: {vectorstore.index.ntotal}")
        logger.info("==========================================")
        
        # Intelligently decide whether to use RAG based on vector store availability and query relevance
        if vectorstore is not None and vectorstore.index.ntotal > 0:
            logger.info(f"Checking document relevance for query: {request.message}")
            try:
                # Detailed debug information for semantic search
                print("\n\n===== DEBUG: Starting semantic search =====")
                print(f"Query: {request.message}")
                print(f"Vector store has {vectorstore.index.ntotal} documents")
                logger.info(f"Vector store path: {vectorstore_path}")
                logger.info(f"Vector store dimensions: {vectorstore.index.d}")
                
                # Perform semantic search on the entire query
                logger.info("Executing similarity_search_with_score with k=5 (retrieving more potential matches)")
                search_results = vectorstore.similarity_search_with_score(
                    request.message,
                    k=5  # Retrieve top 5 most relevant documents for more options
                )
                
                # Check if we have any results with good relevance scores
                relevant_docs = []
                logger.info("========== DOCUMENT RELEVANCE SCORES ==========")
                logger.info(f"Total documents retrieved: {len(search_results)}")
                logger.info(f"Relevance threshold: 0.6 (lower is better)")
                
                for i, (doc, score) in enumerate(search_results):
                    # Lower score is better in FAISS (cosine distance)
                    source = doc.metadata.get("source_file", "Unknown source")
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    doc_id = doc.metadata.get("file_id", "Unknown ID")
                    
                    logger.info(f"Document {i+1} (Source: {source}, ID: {doc_id[:8]}...)")
                    logger.info(f"  Score: {score:.6f}")
                    logger.info(f"  Preview: {preview}")
                    
                    # More detailed debug print for document evaluation
                    print(f"\n----- Document {i+1} Evaluation -----")
                    print(f"Source: {source}")
                    print(f"Score: {score:.6f}")
                    print(f"Doc ID: {doc_id[:8]}...")
                    print(f"Content Length: {len(doc.page_content)} characters")
                    print(f"Preview: {preview}")
                    print(f"Threshold: 0.6 (lower is better)")
                    
                    # Use a threshold to determine relevance - tuned based on testing
                    if score < 0.6:  # Threshold of 0.6 based on our testing
                        logger.info(f"  RELEVANT: Yes (score {score:.6f} < threshold 0.6)")
                        relevant_docs.append(doc)
                        print(f"  DECISION: RELEVANT (score {score:.6f} < 0.6)")
                    else:
                        logger.info(f"  RELEVANT: No (score {score:.6f} >= threshold 0.6)")
                        print(f"  DECISION: NOT RELEVANT (score {score:.6f} >= 0.6)")
                
                logger.info("===============================================")
                
                # Enhanced debug information for RAG decision
                print("\n===== DEBUG: RAG Decision =====")
                print(f"Found {len(relevant_docs)} relevant documents out of {len(search_results)} retrieved")
                logger.info(f"Relevant documents found: {len(relevant_docs)} out of {len(search_results)}")
                
                if relevant_docs:
                    use_rag = True
                    logger.info("========== RAG DECISION ==========")
                    logger.info(f"Using RAG: YES (Found {len(relevant_docs)} relevant documents)")
                    logger.info("===================================")
                    
                    # Format the context from relevant documents with more structure
                    context = "\n\n---\n\nRelevant information from documents:\n\n"
                    
                    # Sort documents by relevance score (if available in metadata)
                    for i, doc in enumerate(relevant_docs):
                        # Add document content with detailed source information
                        source = doc.metadata.get("source_file", "Unknown source")
                        doc_id = doc.metadata.get("file_id", "Unknown ID")
                        
                        # Add section header with document metadata
                        context += f"Document {i+1} (Source: {source}, ID: {doc_id[:8]}...):\n"
                        
                        # Add the document content with proper formatting
                        content = doc.page_content.strip()
                        context += f"{content}\n\n"
                        
                        # Log the document being used
                        logger.info(f"Using document {i+1}: {source} (first 50 chars: {content[:50]}...)")
                    
                    # Update system message to include context with better instructions
                    system_message += "\n\nYou have access to the following relevant information from documents. "
                    system_message += "Use this information to answer the user's question if relevant. "
                    system_message += "If the information directly addresses the user's question, prioritize it over your general knowledge. "
                    system_message += "If the information partially addresses the user's question, combine it with your general knowledge. "
                    system_message += "If the information doesn't address the user's question at all, rely on your general knowledge."
                else:
                    logger.info("========== RAG DECISION ==========")
                    logger.info("Using RAG: NO (No sufficiently relevant documents found)")
                    logger.info("Falling back to Claude's general knowledge")
                    logger.info("===================================")
                    use_rag = False
            except Exception as e:
                logger.error(f"Error during semantic search: {str(e)}")
                use_rag = False
        else:
            logger.info("Using regular chat without RAG")
            use_rag = False
        
        # Prepare messages for Claude
        messages = [{"role": "user", "content": request.message}]
        
        # If we have context from RAG, include it in the system message
        if use_rag and context:
            system_message += context
            logger.info("========== FINAL SYSTEM MESSAGE WITH CONTEXT ==========")
            logger.info(f"System message length: {len(system_message)} characters")
            logger.info("First 200 characters of system message:")
            logger.info(system_message[:200] + "...")
            logger.info("======================================================")
        else:
            logger.info("========== FINAL SYSTEM MESSAGE ==========")
            logger.info("Using default system message without RAG context")
            logger.info(f"System message: {system_message}")
            logger.info("==========================================")
        
        # Call Claude API
        completion = anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=min(request.max_tokens, MAX_OUTPUT_TOKENS),
            system=system_message,
            messages=messages
        )
        response_text = completion.content[0].text if completion.content else "No response generated"
        
        return ChatResponse(
            response=response_text
        )

    except Exception as e:
        # Print the full error for debugging
        logger.error("Error in chat endpoint:", str(e))
        error_msg = f"Chat error: {str(e)}"
        if "invalid x-api-key" in str(e).lower():
            error_msg = "Invalid API key. Please check your ANTHROPIC_API_KEY in the .env file."
        raise HTTPException(status_code=500, detail=error_msg)
