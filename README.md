# Notif Chatbot - Keyword-Based RAG System

A modern AI-powered chatbot with intelligent document processing capabilities, built with FastAPI and Claude 3 Haiku.

## 🚀 Features

- **Intelligent RAG System**: Keyword-based document search that only uses RAG when relevant
- **Document Processing**: Upload and process PDF, TXT, and other text-based documents
- **Vector Store Management**: FAISS-based vector database for efficient document retrieval
- **Modern UI**: Clean, responsive interface with animated buttons and status indicators
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📋 Architecture

### System Overview

```
+-------------------+        +----------------------+        +-------------------+
|     Frontend      |        |       Backend        |        | External Services |
+-------------------+        +----------------------+        +-------------------+
| - HTML/CSS/JS UI  | -----> | - FastAPI Endpoints  | -----> | - Claude 3 Haiku  |
| - Document Upload |        | - Document Processor |        | - HuggingFace     |
| - Chat Interface  |        | - Keyword Extractor  |        |   Embeddings      |
+-------------------+        | - Vector Search      |        +-------------------+
                             | - RAG System         |
                             +----------------------+
                                       |
                                       v
                             +----------------------+
                             |       Storage        |
                             +----------------------+
                             | - FAISS Vector DB    |
                             | - Document Storage   |
                             +----------------------+
```

### Keyword-Based RAG Flow

```
+-------------------+        +----------------------+        +-------------------+
|    User Request   |        |   Keyword Search     |        |    Response       |
+-------------------+        +----------------------+        +-------------------+
| Question          | -----> | Extract Keywords     | -----> | If keywords found |
|                   |        | Search Vector DB     |        | - Use RAG         |
|                   |        |                      |        | If not found      |
|                   |        |                      |        | - Use Claude      |
+-------------------+        +----------------------+        +-------------------+
```

For more detailed architecture diagrams, see the [architecture documentation](docs/architecture.md).

### Backend
- **FastAPI**: High-performance web framework
- **Claude 3 Haiku**: Anthropic's powerful AI model for chat responses
- **LangChain**: Framework for RAG (Retrieval Augmented Generation)
- **FAISS**: Vector database for efficient similarity search
- **HuggingFace Embeddings**: For converting text to vector embeddings

### Frontend
- **HTML/CSS/JavaScript**: Modern, responsive UI
- **Fetch API**: For asynchronous communication with the backend
- **SVG Icons**: For intuitive visual elements

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- Anthropic API Key (for Claude 3 Haiku)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd notif_chatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
MAX_INPUT_TOKENS=2000  # Optional
MAX_OUTPUT_TOKENS=1000  # Optional
```

### Running the Application

Start the application with:
```bash
python -m uvicorn src.main:app --reload
```

The application will be available at: http://127.0.0.1:8000

## 📊 Key Endpoints

- **Main Chat Interface**: http://127.0.0.1:8000
- **Vector Store Info (JSON)**: http://127.0.0.1:8000/vector-info
- **Vector Store View (HTML)**: http://127.0.0.1:8000/vector-view

## 🔍 How It Works

### Keyword-Based RAG System

1. **Keyword Extraction**:
   - The system extracts meaningful keywords from user questions
   - Common words and short terms are filtered out

2. **Vector Database Search**:
   - Each keyword is searched in the vector database
   - If any keyword is found, the system identifies relevant documents

3. **Smart Response Generation**:
   - If keywords are found in the vector database:
     - The system uses the pre-fetched relevant documents
     - The response is generated based specifically on these documents
   - If no keywords are found in the vector database:
     - The system falls back to the general Anthropic model
     - It provides a response based on its general knowledge

### Document Processing

1. Documents are uploaded through the web interface
2. The system processes the document based on its type (PDF, TXT, etc.)
3. Documents are split into chunks and embedded using HuggingFace embeddings
4. The embeddings are stored in a FAISS vector database for efficient retrieval

## 📁 Project Structure

```
notif_chatbot/
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
├── notif_chatbot.log     # Application logs
├── src/
│   ├── main.py           # FastAPI application and endpoints
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css # CSS styles
│   │   ├── js/
│   │   │   └── chat.js   # Frontend JavaScript
│   │   └── img/          # Images and icons
│   └── templates/
│       ├── chat.html     # Main chat interface
│       └── vector.html   # Vector store view
├── uploads/              # Uploaded documents
└── vector_store/         # FAISS vector database
```

## ⚙️ Configuration Options

The application can be configured through environment variables:

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)
- `MAX_INPUT_TOKENS`: Maximum tokens for user input (default: 2000)
- `MAX_OUTPUT_TOKENS`: Maximum tokens for AI response (default: 1000)

## 🔒 Security Considerations

- The application uses CORS middleware to allow cross-origin requests
- File uploads are limited to text-based documents
- API keys should be kept secure and not committed to version control
- Use the provided `.env.example` as a template and create your own `.env` file
- Never commit your actual API keys or sensitive credentials to version control
- Consider using a secrets management solution for production deployments
- Regularly rotate your API keys as a security best practice

## 📝 Logging

The application uses Python's built-in logging module with two handlers:
- File handler: Writes logs to `notif_chatbot.log`
- Stream handler: Outputs logs to the console

Log levels include:
- INFO: General operational information
- ERROR: Errors that might still allow the application to continue running
- WARNING: Warning messages

## 🚀 Future Enhancements

- Multi-user support with authentication
- Document management interface (delete, rename)
- Advanced RAG strategies (hybrid search, re-ranking)
- Support for more document types (DOCX, HTML, etc.)
- Conversation history persistence

## 📄 License

[MIT License](LICENSE)
