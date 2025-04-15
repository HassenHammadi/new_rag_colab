# RAG System for Google Colab with Drive Integration

This repository contains a Retrieval-Augmented Generation (RAG) system specifically designed to run in Google Colab with Google Drive integration for persistent storage of vector stores.

## Features

- **Google Colab Integration**: Designed to run seamlessly in Google Colab notebooks
- **Google Drive Storage**: Store and load vector stores from Google Drive for persistence
- **Multi-Format Support**: Process PDF, text, JSON, and CSV files
- **Interactive Interface**: User-friendly interface for querying the RAG system
- **Comprehensive Debugging**: Advanced debugging capabilities for troubleshooting
- **Performance Monitoring**: Track execution times and resource usage
- **Robust Error Handling**: Standardized error handling with fallbacks
- **Caching System**: Multi-level caching for embeddings and queries
- **Well-Documented Interfaces**: Clear documentation of component interfaces
- **Version Compatibility**: Compatibility with the original RAG system

## Getting Started

### Option 1: Open the Notebook in Google Colab

1. Open the notebook directly in Google Colab:
   - [RAG with Google Drive Integration](https://colab.research.google.com/github/yourusername/new_rag_colab/blob/main/colab_notebooks/rag_with_drive.ipynb)
   - [Quick Start Guide](https://colab.research.google.com/github/yourusername/new_rag_colab/blob/main/colab_notebooks/quick_start.ipynb)
   - [Debug Mode](https://colab.research.google.com/github/yourusername/new_rag_colab/blob/main/colab_notebooks/debug_mode.ipynb)

### Option 2: Manual Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/new_rag_colab.git
   ```

2. Upload the notebook to Google Colab:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click on "File" > "Upload notebook"
   - Select the notebook from `colab_notebooks/rag_with_drive.ipynb`

3. Follow the instructions in the notebook to set up and use the RAG system

## System Components

- **Drive Handler**: Manages Google Drive operations for storing and loading vector stores
- **Drive Vector Store**: FAISS vector store with Google Drive integration
- **Colab RAG Pipeline**: RAG pipeline designed for Google Colab with Drive integration
- **Processors**: Process different file formats (PDF, text, JSON, CSV)
- **Chunkers**: Split documents into chunks for embedding
- **Retrievers**: Retrieve relevant chunks based on queries
- **Debug Utilities**: Tools for debugging and performance monitoring
- **Error Handling**: Standardized error handling system
- **Caching System**: Multi-level caching for improved performance

## Usage

1. **Mount Google Drive**: Connect to Google Drive for persistent storage
2. **Upload Files**: Upload PDF, text, JSON, or CSV files to process
3. **Process Files**: Extract text, create chunks, and generate embeddings
4. **Save Vector Store**: Store the vector store in Google Drive
5. **Query the System**: Ask questions and get answers based on your documents

## Example

```python
# Create the RAG pipeline
rag_pipeline = ColabRAGPipeline(
    chunker=chunker,
    vector_store=vector_store,
    retriever=retriever,
    processors=processors,
    drive_handler=drive_handler
)

# Process files
rag_pipeline.process_file("document.pdf")

# Save to Google Drive
rag_pipeline.save_vector_store("vector_store", drive_subfolder="my_vector_store")

# Load from Google Drive
rag_pipeline.load_vector_store("vector_store", from_drive=True, drive_path="my_vector_store")

# Query the system
results = rag_pipeline.query("What is machine learning?")
```

## Debugging

The system includes comprehensive debugging capabilities:

```python
# Enable debug mode
from new_rag_colab.utils.debug_utils import debug_logger
import logging

# Set logging level
debug_logger.logger.setLevel(logging.DEBUG)

# Inspect vector store
from new_rag_colab.utils.debug_utils import DebugInspector
DebugInspector.print_vector_store_info(vector_store)

# Track performance
debug_logger.start_timer("operation")
# ... perform operation ...
duration = debug_logger.end_timer("operation")
print(f"Operation completed in {duration:.4f} seconds")
```

## Documentation

Detailed documentation is available in the following files:

- [HOW_TO_USE.txt](HOW_TO_USE.txt): Comprehensive usage guide
- [INTERFACE_DOCUMENTATION.md](INTERFACE_DOCUMENTATION.md): Detailed interface documentation
- [VERSION_COMPATIBILITY.md](VERSION_COMPATIBILITY.md): Compatibility information
- [INSTALLATION.md](INSTALLATION.md): Installation guide

## Testing

Unit tests are available in the `tests` directory. Run them with:

```python
python -m unittest discover -s new_rag_colab/tests
```

## Requirements

- Google Colab environment
- Google Drive account
- Python 3.7+
- See `requirements.txt` for full dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details
