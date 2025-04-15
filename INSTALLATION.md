# Installation Guide for RAG Colab System

This guide explains how to set up the RAG Colab system with Google Drive integration.

## Prerequisites

- Google account with access to Google Colab
- Google Drive account for storing vector stores
- Git (optional, for local development)

## Installation Options

### Option 1: Direct Colab Usage (Recommended)

1. Open one of the provided notebooks directly in Google Colab:
   - [RAG with Google Drive Integration](https://colab.research.google.com/github/yourusername/new_rag_colab/blob/main/colab_notebooks/rag_with_drive.ipynb)
   - [Quick Start Guide](https://colab.research.google.com/github/yourusername/new_rag_colab/blob/main/colab_notebooks/quick_start.ipynb)

2. Follow the instructions in the notebook to install dependencies and set up the system

### Option 2: GitHub Installation

1. In Google Colab, run the following commands:
   ```python
   !git clone https://github.com/yourusername/new_rag_colab.git
   %cd new_rag_colab
   !pip install -e .
   ```

2. Import and use the modules as shown in the example notebooks

### Option 3: Manual Installation (For Development)

1. Clone the repository locally:
   ```bash
   git clone https://github.com/yourusername/new_rag_colab.git
   cd new_rag_colab
   ```

2. Copy the necessary modules from the original RAG system:
   ```bash
   python copy_modules.py
   ```

3. Upload the notebooks to Google Colab and run them

## Setting Up Google Drive Integration

The system automatically handles Google Drive integration when running in Colab:

1. When you run the `mount_drive()` method, you'll be prompted to authorize access to your Google Drive
2. Follow the authorization link and copy the provided code
3. Paste the code in the input field in Colab
4. The system will create a folder named `RAG_vector_stores` in your Google Drive to store vector stores

## Verifying Installation

To verify that the installation is working correctly:

1. Run the quick start notebook
2. Upload a sample document (PDF, text, JSON, or CSV)
3. Process the document and save the vector store to Google Drive
4. Query the system with a simple question
5. Check that the vector store folder appears in your Google Drive

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'new_rag_colab'**
   - Make sure you've installed the package with `pip install -e .`
   - Check that you're running the code in the correct directory

2. **Google Drive authorization fails**
   - Try clearing your browser cache and cookies
   - Ensure you're using the same Google account for Colab and Drive

3. **CUDA out of memory error**
   - Reduce batch sizes in the embedding provider
   - Use a smaller embedding model
   - Switch to CPU-only mode

4. **File not found errors**
   - Check that the file paths are correct
   - Ensure files are uploaded to the Colab runtime

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub issues](https://github.com/yourusername/new_rag_colab/issues) for similar problems
2. Create a new issue with details about your problem
3. Include error messages and steps to reproduce the issue
