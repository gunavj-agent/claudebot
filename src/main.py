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

        response_text = ""
        
        # Check if vectorstore exists
        if vectorstore is None:
            logger.info("No vectorstore available")
            has_documents = False
        else:
            logger.info(f"Vectorstore available with {vectorstore.index.ntotal} embeddings")
            has_documents = vectorstore.index.ntotal > 0
        
        # Extract keywords from the question
        def extract_keywords(text):
            # Remove common words and punctuation
            common_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'and', 'or', 'not', 'but', 'if', 'then', 'than', 'so', 'as', 'of', 'from', 'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'do', 'does', 'did', 'has', 'have', 'had', 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must', 'be', 'been', 'being', 'am', 'was', 'were', 'i', 'you', 'he', 'she', 'we', 'they'}
            words = text.lower().split()
            keywords = [word.strip('.,?!()[]{}":;') for word in words if word.strip('.,?!()[]{}":;').lower() not in common_words and len(word) > 2]
            return keywords
        
        keywords = extract_keywords(request.message)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Check if keywords exist in vector store
        keywords_in_vectorstore = False
        relevant_docs = []
        
        if has_documents and keywords:
            try:
                # Search for each keyword in the vector store
                for keyword in keywords:
                    if len(keyword) < 3:  # Skip very short keywords
                        continue
                    
                    # Search the vector store for the keyword
                    docs = vectorstore.similarity_search(keyword, k=1)
                    
                    if docs:
                        logger.info(f"Keyword '{keyword}' found in vector store")
                        keywords_in_vectorstore = True
                        for doc in docs:
                            if doc not in relevant_docs:
                                relevant_docs.append(doc)
                
                logger.info(f"Keywords found in vector store: {keywords_in_vectorstore}")
                logger.info(f"Found {len(relevant_docs)} relevant documents")
            except Exception as e:
                logger.error(f"Error searching keywords in vector store: {str(e)}")
        
        # Use RAG if requested, documents exist, and keywords were found in the vector store
        if request.use_rag and has_documents and keywords_in_vectorstore:
            try:
                logger.info("Using RAG for response generation")
                
                # Create a retriever
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                
                # If we already found relevant documents, we can use them directly
                if relevant_docs:
                    logger.info(f"Using {len(relevant_docs)} pre-fetched relevant documents")
                    
                    # Create a custom system prompt for RAG
                    system_prompt = """You are a helpful AI assistant that answers questions based on the provided documents.
                    When answering, use only the information from the documents provided. 
                    If the documents don't contain the answer, say that you don't know based on the available information.
                    Always cite your sources by mentioning which document(s) you used to answer the question."""
                    
                    # Create LLM
                    llm = ChatAnthropicMessages(
                        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                        model_name="claude-3-haiku-20240307",
                        max_tokens=min(request.max_tokens, MAX_OUTPUT_TOKENS),
                    )
                    
                    # Prepare document content for the prompt
                    doc_content = ""
                    source_references = "\n\nSources:\n"
                    
                    for i, doc in enumerate(relevant_docs):
                        doc_content += f"\nDocument {i+1}:\n{doc.page_content}\n"
                        source_name = doc.metadata.get("source", f"Document {i+1}")
                        source_references += f"- {source_name}\n"
                    
                    # Create the prompt with document content
                    prompt = f"""Based on the following documents, please answer this question: {request.message}
                    
                    {doc_content}
                    """
                    
                    # Get response from Claude
                    completion = anthropic.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=min(request.max_tokens, MAX_OUTPUT_TOKENS),
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    # Extract answer
                    answer = completion.content[0].text if completion.content else "No response generated"
                    
                    # Combine answer with source references
                    response_text = answer + source_references
                else:
                    # Create memory
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer"  # Specify the output key
                    )
                    
                    # Create the chain
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        memory=memory,
                        return_source_documents=True,
                    )
                    
                    # Run the chain
                    result = qa_chain.invoke({"question": request.message})
                    
                    # Extract answer and source documents
                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])
                    
                    # Format source document references
                    source_references = ""
                    if source_docs:
                        source_references = "\n\nSources:\n"
                        for i, doc in enumerate(source_docs):
                            source_name = doc.metadata.get("source", f"Document {i+1}")
                            source_references += f"- {source_name}\n"
                    
                    # Combine answer with source references
                    response_text = answer + source_references
            except Exception as e:
                logger.error(f"RAG error: {str(e)}")
                # Fall back to regular chat if RAG fails
                completion = anthropic.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=min(request.max_tokens, MAX_OUTPUT_TOKENS),
                    messages=[{"role": "user", "content": request.message}]
                )
                response_text = completion.content[0].text if completion.content else "No response generated"
        else:
            # Regular chat without RAG
            logger.info("Using regular chat without RAG")
            
            # Create a system message
            system_message = "You are a helpful AI assistant."
            
            # Check if RAG was requested but no documents are available
            if request.use_rag and not has_documents:
                logger.info("RAG was requested but no documents are available")
                system_message += " The user has requested to use documents for answering, but no documents have been uploaded yet or the documents couldn't be processed."
            # Check if it's not a document-specific question but documents are available
            elif request.use_rag and has_documents and not keywords_in_vectorstore:
                logger.info("Documents available but question doesn't appear to be document-specific")
                system_message += " The user has uploaded documents, but your current question doesn't appear to be about those documents. I'm answering based on my general knowledge."
            
            completion = anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=min(request.max_tokens, MAX_OUTPUT_TOKENS),
                system=system_message,
                messages=[{"role": "user", "content": request.message}]
            )
            response_text = completion.content[0].text if completion.content else "No response generated"
            
            # Add a note about missing documents if applicable
            if "document" in request.message.lower() or "upload" in request.message.lower():
                if not has_documents:
                    response_text += "\n\n(Note: I don't have access to any uploaded documents yet. Please upload documents first if you'd like me to reference them.)"
        
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
