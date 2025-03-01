graph TD
    subgraph "Frontend"
        UI[HTML/CSS/JS Interface]
        Upload[Document Upload]
        Chat[Chat Interface]
    end
    
    subgraph "Backend (FastAPI)"
        API[API Endpoints]
        DocProc[Document Processor]
        KeywordExtract[Keyword Extractor]
        VectorSearch[Vector Search]
        RAG[RAG System]
        Chat_LLM[Chat LLM]
    end
    
    subgraph "Storage"
        VectorDB[FAISS Vector Database]
        DocStorage[Document Storage]
    end
    
    subgraph "External Services"
        Claude[Claude 3 Haiku API]
        HFEmbeddings[HuggingFace Embeddings]
    end
    
    %% Frontend to Backend connections
    UI --> API
    Upload --> API
    Chat --> API
    
    %% Backend internal connections
    API --> DocProc
    API --> KeywordExtract
    API --> VectorSearch
    API --> RAG
    API --> Chat_LLM
    
    DocProc --> VectorDB
    KeywordExtract --> VectorSearch
    VectorSearch --> VectorDB
    VectorSearch --> RAG
    RAG --> Chat_LLM
    
    %% Backend to Storage connections
    DocProc --> DocStorage
    
    %% Backend to External Services connections
    Chat_LLM --> Claude
    DocProc --> HFEmbeddings
    VectorSearch --> HFEmbeddings
