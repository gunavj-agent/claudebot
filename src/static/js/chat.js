document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const documentUploadInput = document.getElementById('document-upload-input');
    const uploadStatus = document.getElementById('upload-status');
    
    // Track uploaded documents
    let uploadedDocuments = [];
    
    // Check if we have documents in the vector store on page load
    checkVectorStore();
    
    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
    });

    // Handle enter key
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendButton.addEventListener('click', sendMessage);
    
    // RAG is handled intelligently on the backend
    
    // Handle document upload
    documentUploadInput.addEventListener('change', async (event) => {
        const files = event.target.files;
        if (!files || files.length === 0) {
            return;
        }
        
        const uploadStatus = document.getElementById('upload-status');
        uploadStatus.textContent = 'Uploading...';
        uploadStatus.className = 'info';
        
        console.log(`Selected ${files.length} files for upload`);
        
        try {
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
                console.log(`Adding file: ${files[i].name} (${files[i].size} bytes)`);
            }
            
            console.log('Sending upload request...');
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed with status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Upload response:', result);
            
            if (result.uploaded_files && result.uploaded_files.length > 0) {
                uploadStatus.textContent = `Successfully uploaded ${result.uploaded_files.length} file(s)`;
                uploadStatus.className = 'success';
                
                // Add a system message about the upload
                const fileNames = result.uploaded_files.map(f => f.filename).join(', ');
                
                // Create a bot message in the chat window
                const messageContent = `I've successfully processed the following document(s): ${fileNames}.\n\nThese documents are now available for me to reference when answering your questions. I'll automatically use them when relevant to your questions.\n\nTry asking me something specific about the content, such as "What are the key points in the document?" or "Summarize the main ideas from the uploaded files."`;
                
                const messageElement = createMessageElement(messageContent, false);
                chatMessages.appendChild(messageElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // RAG is handled intelligently on the backend
                
                // Update our tracking of uploaded documents
                checkVectorStore();
            } else {
                uploadStatus.textContent = 'Upload completed, but no files were processed successfully';
                uploadStatus.className = 'warning';
            }
        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.className = 'error';
        }
    });

    function createMessageElement(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        return messageDiv;
    }

    function createTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message bot';
        indicator.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-circle"></div>
                <div class="typing-circle"></div>
                <div class="typing-circle"></div>
            </div>
        `;
        return indicator;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        userInput.disabled = true;
        sendButton.disabled = true;

        // Add user message to chat
        chatMessages.appendChild(createMessageElement(message, true));
        userInput.value = '';
        userInput.style.height = 'auto';

        // Add typing indicator
        const typingIndicator = createTypingIndicator();
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    max_tokens: 1000,
                    use_rag: true  // Always enable RAG, backend will decide intelligently
                }),
            });

            const data = await response.json();
            
            // Remove typing indicator
            typingIndicator.remove();

            // Add bot response
            chatMessages.appendChild(createMessageElement(data.response));
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } catch (error) {
            // Remove typing indicator
            typingIndicator.remove();

            // Add error message
            const errorMessage = createMessageElement('Sorry, I encountered an error. Please try again.');
            errorMessage.classList.add('error');
            chatMessages.appendChild(errorMessage);
        } finally {
            // Re-enable input and button
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        }
    }

    function addMessage(type, content) {
        const messageElement = createMessageElement(content, type === 'user');
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // RAG functionality has been removed

    function checkVectorStore() {
        fetch('/vector-info')
            .then(response => response.json())
            .then(data => {
                console.log('Vector store info:', data);
                if (data.status === 'success' && data.documents && data.documents.length > 0) {
                    uploadedDocuments = data.documents;
                    console.log(`Found ${uploadedDocuments.length} documents in vector store`);
                    
                    // Add a message if documents are available
                    if (uploadedDocuments.length > 0 && chatMessages.children.length === 1) {
                        const messageContent = `I have access to ${uploadedDocuments.length} document chunks that I can reference when answering your questions. I'll automatically use them when relevant to your questions. Ask me anything about the uploaded documents!`;
                        const messageElement = createMessageElement(messageContent, false);
                        chatMessages.appendChild(messageElement);
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                        
                        // RAG is handled intelligently on the backend
                    }
                }
            })
            .catch(error => {
                console.error('Error checking vector store:', error);
            });
    }
});
