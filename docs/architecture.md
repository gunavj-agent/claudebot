# Notif Chatbot Architecture

## System Architecture Diagram

```mermaid
graph TD
    subgraph "Frontend"
        UI[HTML/CSS/JS Interface]
        Upload[Document Upload]
        Chat[Chat Interface]
    end
    
    subgraph "Backend (FastAPI)"
        API[API Endpoints]
        DocProc[Document Processor]
        SemanticSearch[Semantic Search]
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
    API --> SemanticSearch
    API --> VectorSearch
    API --> RAG
    API --> Chat_LLM
    
    DocProc --> VectorDB
    SemanticSearch --> VectorSearch
    VectorSearch --> VectorDB
    VectorSearch --> RAG
    RAG --> Chat_LLM
    
    %% Backend to Storage connections
    DocProc --> DocStorage
    
    %% Backend to External Services connections
    Chat_LLM --> Claude
    DocProc --> HFEmbeddings
    VectorSearch --> HFEmbeddings
    
    %% Define styles
    classDef frontend fill:#f9f,stroke:#333,stroke-width:2px;
    classDef backend fill:#bbf,stroke:#333,stroke-width:2px;
    classDef storage fill:#bfb,stroke:#333,stroke-width:2px;
    classDef external fill:#fbb,stroke:#333,stroke-width:2px;
    
    class UI,Upload,Chat frontend;
    class API,DocProc,SemanticSearch,VectorSearch,RAG,Chat_LLM backend;
    class VectorDB,DocStorage storage;
    class Claude,HFEmbeddings external;
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant UI as Web Interface
    participant API as FastAPI Backend
    participant SS as Semantic Search
    participant VS as Vector Search
    participant RAG as RAG System
    participant VDB as Vector Database
    participant Claude as Claude 3 Haiku
    
    %% Document Upload Flow
    User->>UI: Upload Document
    UI->>API: POST /upload
    API->>API: Process Document
    API->>VDB: Store Document Embeddings
    API->>UI: Return Upload Status
    UI->>User: Display Upload Confirmation
    
    %% Chat Flow
    User->>UI: Send Message
    UI->>API: POST /chat
    API->>SS: Process Full Query
    SS->>VS: Semantic Search in Vector DB
    VS->>VDB: Query Vector Database
    VDB->>VS: Return Documents with Relevance Scores
    
    alt Relevant Documents Found (Score < 0.6)
        VS->>RAG: Use RAG with Relevant Documents
        RAG->>Claude: Generate Response with Document Context
        Claude->>RAG: Return Document-Based Response
        RAG->>API: Return Response with Source Citations
    else No Relevant Documents Found
        API->>Claude: Generate Response from General Knowledge
        Claude->>API: Return General Knowledge Response
    end
    
    API->>UI: Return Response
    UI->>User: Display Response
```

## Component Architecture

```mermaid
flowchart TD
    subgraph "User Interface"
        direction TB
        UI_Container[Chat Container]
        UI_Input[Message Input]
        UI_Upload[Document Upload]
        UI_Messages[Message Display]
        UI_Status[Status Indicators]
        
        UI_Container --> UI_Input
        UI_Container --> UI_Upload
        UI_Container --> UI_Messages
        UI_Container --> UI_Status
    end
    
    subgraph "API Layer"
        direction TB
        API_Chat[/chat Endpoint]
        API_Upload[/upload Endpoint]
        API_Vector[/vector-info Endpoint]
        API_VectorView[/vector-view Endpoint]
        
        API_Chat --> API_Upload
        API_Chat --> API_Vector
    end
    
    subgraph "Core Processing"
        direction TB
        CP_DocProc[Document Processor]
        CP_Embeddings[Embedding Generator]
        CP_Chunking[Text Chunker]
        CP_SemanticSearch[Semantic Search]
        CP_VectorSearch[Vector Search]
        
        CP_DocProc --> CP_Chunking
        CP_Chunking --> CP_Embeddings
        CP_SemanticSearch --> CP_VectorSearch
    end
    
    subgraph "RAG System"
        direction TB
        RAG_Retriever[Document Retriever]
        RAG_Context[Context Builder]
        RAG_Generator[Response Generator]
        
        RAG_Retriever --> RAG_Context
        RAG_Context --> RAG_Generator
    end
    
    subgraph "Storage"
        direction TB
        ST_FAISS[FAISS Vector Store]
        ST_Uploads[Document Storage]
        
        ST_FAISS --> ST_Uploads
    end
    
    subgraph "External APIs"
        direction TB
        EXT_Claude[Claude API]
        EXT_HF[HuggingFace API]
    end
    
    %% Cross-component connections
    UI_Input --> API_Chat
    UI_Upload --> API_Upload
    API_Upload --> CP_DocProc
    CP_DocProc --> ST_Uploads
    CP_Embeddings --> ST_FAISS
    API_Chat --> CP_SemanticSearch
    CP_VectorSearch --> ST_FAISS
    CP_VectorSearch --> RAG_Retriever
    RAG_Generator --> EXT_Claude
    CP_Embeddings --> EXT_HF
    
    %% Define styles
    classDef ui fill:#f9f,stroke:#333,stroke-width:1px;
    classDef api fill:#bbf,stroke:#333,stroke-width:1px;
    classDef core fill:#bfb,stroke:#333,stroke-width:1px;
    classDef rag fill:#fbf,stroke:#333,stroke-width:1px;
    classDef storage fill:#fbb,stroke:#333,stroke-width:1px;
    classDef external fill:#bff,stroke:#333,stroke-width:1px;
    
    class UI_Container,UI_Input,UI_Upload,UI_Messages,UI_Status ui;
    class API_Chat,API_Upload,API_Vector,API_VectorView api;
    class CP_DocProc,CP_Embeddings,CP_Chunking,CP_SemanticSearch,CP_VectorSearch core;
    class RAG_Retriever,RAG_Context,RAG_Generator rag;
    class ST_FAISS,ST_Uploads storage;
    class EXT_Claude,EXT_HF external;
```

## Keyword-Based RAG System

```mermaid
flowchart LR
    subgraph "User Input"
        UI[User Question]
    end
    
    subgraph "Keyword Processing"
        KE[Keyword Extractor]
        KF[Keyword Filter]
        
        KE --> KF
    end
    
    subgraph "Vector Search"
        VS[Vector Search]
        VR[Vector Results]
        
        VS --> VR
    end
    
    subgraph "Decision Logic"
        DL{Keywords Found?}
    end
    
    subgraph "Response Generation"
        RG_RAG[RAG Response]
        RG_General[General Response]
    end
    
    %% Flow connections
    UI --> KE
    KF --> VS
    VR --> DL
    DL -->|Yes| RG_RAG
    DL -->|No| RG_General
    
    %% Define styles
    classDef input fill:#f9f,stroke:#333,stroke-width:1px;
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef decision fill:#fbb,stroke:#333,stroke-width:1px;
    classDef output fill:#bfb,stroke:#333,stroke-width:1px;
    
    class UI input;
    class KE,KF,VS,VR process;
    class DL decision;
    class RG_RAG,RG_General output;
```
