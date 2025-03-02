// Debug script to test file upload
console.log('Debug script loaded');

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM content loaded');
    
    // Check if the upload elements exist
    const documentUploadInput = document.getElementById('document-upload-input');
    const uploadStatus = document.getElementById('upload-status');
    const uploadButton = document.querySelector('.upload-button');
    
    console.log('Upload input element:', documentUploadInput);
    console.log('Upload status element:', uploadStatus);
    console.log('Upload button element:', uploadButton);
    
    // Add click event to the upload button for debugging
    if (uploadButton) {
        uploadButton.addEventListener('click', () => {
            console.log('Upload button clicked');
        });
    }
    
    // Add change event to the file input for debugging
    if (documentUploadInput) {
        documentUploadInput.addEventListener('change', (e) => {
            console.log('File input change event triggered');
            console.log('Files selected:', e.target.files);
        });
    }
});
