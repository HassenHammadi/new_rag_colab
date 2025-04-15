# Version Compatibility

This document outlines the compatibility of the RAG Colab system with the original RAG system and external dependencies.

## Compatibility with Original RAG System

The RAG Colab system is designed to be compatible with the original RAG system while extending it for the Google Colab environment. Here's how the compatibility is maintained:

### Compatible Components

- **Processors**: The processors for different file formats (PDF, text, JSON, CSV) maintain the same interface and functionality as the original system.
- **Chunkers**: The chunking strategies are compatible with the original system, with the same interface for splitting documents.
- **Retrievers**: The retrieval mechanisms maintain compatibility with the original system, with the same interface for retrieving documents.
- **Vector Stores**: The vector store implementation is compatible with the original system, with extensions for Google Drive integration.

### Extended Components

- **DriveHandler**: New component for Google Drive integration.
- **DriveVectorStore**: Extended version of the vector store with Google Drive integration.
- **ColabRAGPipeline**: Extended version of the RAG pipeline with Google Drive integration.
- **Debug Utilities**: New components for debugging and performance monitoring.
- **Caching Utilities**: Enhanced caching mechanisms for improved performance.

### Migration from Original System

To migrate from the original RAG system to the RAG Colab system:

1. Copy the necessary modules from the original system using the `copy_modules.py` script.
2. Update import paths to use the new package structure.
3. Replace the original RAG pipeline with the Colab RAG pipeline.
4. Update vector store usage to use the Drive vector store.

## Dependency Compatibility

The RAG Colab system has the following dependency requirements:

### Python Version

- Python 3.7 or higher is required.

### Core Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| faiss-cpu | >=1.7.4 | Vector storage and similarity search |
| sentence-transformers | >=2.2.2 | Text embeddings |
| transformers | >=4.35.0 | Transformer models |
| torch | >=2.0.0 | Deep learning framework |
| requests | >=2.31.0 | HTTP requests |
| python-dotenv | >=1.0.0 | Environment variable management |
| numpy | >=1.24.3 | Numerical operations |
| tqdm | >=4.66.1 | Progress bars |

### File Format Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| pypdf | >=4.0.0 | PDF processing |
| unstructured | >=0.11.0 | Unstructured document processing |
| pdf2image | >=1.16.3 | PDF to image conversion for OCR |
| pytesseract | >=0.3.10 | OCR for PDF files |
| pandas | >=2.0.3 | CSV and JSON processing |

### Google Colab Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| google-colab | latest | Google Colab integration |
| ipywidgets | >=7.7.1 | Interactive widgets for notebooks |
| ipython | >=7.0.0 | IPython for notebook integration |

## Version History

### Version 0.1.0 (Current)

- Initial release of the RAG Colab system.
- Google Drive integration for vector store persistence.
- Enhanced debugging capabilities.
- Improved caching mechanisms.
- Comprehensive documentation.

## Future Compatibility

The RAG Colab system is designed to maintain compatibility with future versions of the original RAG system. As the original system evolves, the RAG Colab system will be updated to incorporate new features and improvements while maintaining the Google Colab integration.

### Planned Compatibility Improvements

- **Automatic Synchronization**: Automatically synchronize changes between the original system and the RAG Colab system.
- **Plugin Architecture**: Develop a plugin architecture to make it easier to extend the system with new components.
- **Version Checking**: Add version checking to ensure compatibility between components.
- **Dependency Management**: Improve dependency management to handle different versions of dependencies.

## Reporting Compatibility Issues

If you encounter compatibility issues between the RAG Colab system and the original RAG system, please report them by creating an issue on the GitHub repository with the following information:

1. Description of the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Versions of both systems
6. Relevant error messages or logs
