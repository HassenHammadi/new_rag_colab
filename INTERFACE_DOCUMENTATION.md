# RAG Colab System Interface Documentation

This document provides detailed information about the interfaces of key components in the RAG Colab system.

## Table of Contents

1. [DriveHandler Interface](#drivehandler-interface)
2. [DriveVectorStore Interface](#drivevectorstore-interface)
3. [ColabRAGPipeline Interface](#colabragpipeline-interface)
4. [EmbeddingProvider Interface](#embeddingprovider-interface)
5. [Chunker Interface](#chunker-interface)
6. [Retriever Interface](#retriever-interface)
7. [Processor Interface](#processor-interface)
8. [LLMFormatter Interface](#llmformatter-interface)
9. [Caching Interface](#caching-interface)
10. [Debug Utilities Interface](#debug-utilities-interface)

## DriveHandler Interface

The `DriveHandler` class provides an interface for Google Drive operations.

### Methods

#### `mount_drive() -> bool`

Mounts Google Drive in Colab.

- **Returns**: `True` if successful, `False` otherwise

#### `save_to_drive(local_path: Union[str, Path], drive_subfolder: Optional[str] = None) -> bool`

Saves files from a local path to Google Drive.

- **Parameters**:
  - `local_path`: Local path to save from
  - `drive_subfolder`: Optional subfolder within the base folder
- **Returns**: `True` if successful, `False` otherwise

#### `load_from_drive(drive_path: str, local_path: Optional[Union[str, Path]] = None) -> Optional[Path]`

Loads files from Google Drive to a local path.

- **Parameters**:
  - `drive_path`: Path within the base folder in Google Drive
  - `local_path`: Optional local destination path
- **Returns**: Path to the local files if successful, `None` otherwise

#### `list_vector_stores() -> list`

Lists available vector stores in Google Drive.

- **Returns**: List of vector store names

## DriveVectorStore Interface

The `DriveVectorStore` class provides an interface for vector storage with Google Drive integration.

### Methods

#### `add_documents(documents: List[Dict[str, Any]]) -> List[str]`

Adds documents to the vector store.

- **Parameters**:
  - `documents`: List of document dictionaries with content and metadata
- **Returns**: List of document IDs

#### `similarity_search(query: str, k: int = 4) -> List[Dict[str, Any]]`

Searches for documents similar to the query.

- **Parameters**:
  - `query`: Query string
  - `k`: Number of results to return
- **Returns**: List of document dictionaries with content, metadata, and similarity score

#### `save(directory: Union[str, Path], drive_subfolder: Optional[str] = None) -> None`

Saves the vector store to disk and optionally to Google Drive.

- **Parameters**:
  - `directory`: Directory to save to
  - `drive_subfolder`: Optional subfolder in Google Drive to save to

#### `load(directory: Union[str, Path], from_drive: bool = False, drive_path: Optional[str] = None) -> None`

Loads the vector store from disk or Google Drive.

- **Parameters**:
  - `directory`: Directory to load from
  - `from_drive`: Whether to load from Google Drive
  - `drive_path`: Path within the base folder in Google Drive

#### `get_debug_info() -> Dict[str, Any]`

Gets debug information about the vector store.

- **Returns**: Dictionary with debug information

## ColabRAGPipeline Interface

The `ColabRAGPipeline` class provides an interface for the RAG pipeline with Google Drive integration.

### Methods

#### `process_file(file_path: Union[str, Path], file_type: Optional[str] = None) -> List[str]`

Processes a file and adds it to the vector store.

- **Parameters**:
  - `file_path`: Path to the file
  - `file_type`: Type of file (pdf, text, json, csv) or None to infer from extension
- **Returns**: List of document IDs

#### `process_directory(directory: Union[str, Path], file_types: Optional[List[str]] = None, recursive: bool = False) -> Dict[str, List[str]]`

Processes all supported files in a directory.

- **Parameters**:
  - `directory`: Directory containing files
  - `file_types`: List of file types to process (pdf, text, json, csv) or None for all
  - `recursive`: Whether to process subdirectories
- **Returns**: Dictionary mapping filenames to document IDs

#### `query(query: str, use_cache: bool = True, **kwargs) -> List[Dict[str, Any]]`

Queries the RAG pipeline.

- **Parameters**:
  - `query`: Query string
  - `use_cache`: Whether to use query cache (if enabled)
  - `**kwargs`: Additional arguments to pass to the retriever
- **Returns**: List of retrieved documents

#### `query_with_markdown(query: str, **kwargs) -> str`

Queries the RAG pipeline and returns a Markdown-formatted response.

- **Parameters**:
  - `query`: Query string
  - `**kwargs`: Additional arguments to pass to the retriever or formatter
- **Returns**: Markdown-formatted response string

#### `save_vector_store(directory: Union[str, Path], drive_subfolder: Optional[str] = None) -> None`

Saves the vector store to disk and optionally to Google Drive.

- **Parameters**:
  - `directory`: Directory to save to
  - `drive_subfolder`: Optional subfolder in Google Drive to save to

#### `load_vector_store(directory: Union[str, Path], from_drive: bool = False, drive_path: Optional[str] = None) -> None`

Loads the vector store from disk or Google Drive.

- **Parameters**:
  - `directory`: Directory to load from
  - `from_drive`: Whether to load from Google Drive
  - `drive_path`: Path within the base folder in Google Drive

#### `list_drive_vector_stores() -> list`

Lists available vector stores in Google Drive.

- **Returns**: List of vector store names

## EmbeddingProvider Interface

The `EmbeddingProvider` class provides an interface for embedding providers.

### Methods

#### `get_embeddings(texts: List[str]) -> List[List[float]]`

Gets embeddings for a list of texts.

- **Parameters**:
  - `texts`: List of texts to embed
- **Returns**: List of embeddings

#### `get_embedding(text: str) -> List[float]`

Gets embedding for a single text with caching.

- **Parameters**:
  - `text`: Text to embed
- **Returns**: Embedding vector

## Chunker Interface

The `BaseChunker` class provides an interface for document chunkers.

### Methods

#### `chunk_document(document: Dict[str, Any]) -> List[Dict[str, Any]]`

Splits a document into chunks.

- **Parameters**:
  - `document`: Document to split
- **Returns**: List of chunks

## Retriever Interface

The `BaseRetriever` class provides an interface for document retrievers.

### Methods

#### `retrieve(query: str, **kwargs) -> List[Dict[str, Any]]`

Retrieves documents relevant to a query.

- **Parameters**:
  - `query`: Query string
  - `**kwargs`: Additional arguments
- **Returns**: List of retrieved documents

## Processor Interface

The processor classes provide interfaces for processing different file types.

### PDFProcessor Methods

#### `load_pdf(file_path: Union[str, Path]) -> Dict[str, Any]`

Loads a PDF file and extracts text.

- **Parameters**:
  - `file_path`: Path to the PDF file
- **Returns**: Dictionary with metadata and pages

### TextProcessor Methods

#### `load_text(file_path: Union[str, Path]) -> Dict[str, Any]`

Loads a text file and extracts text.

- **Parameters**:
  - `file_path`: Path to the text file
- **Returns**: Dictionary with metadata and content

### JSONProcessor Methods

#### `load_json(file_path: Union[str, Path]) -> Dict[str, Any]`

Loads a JSON file and extracts text.

- **Parameters**:
  - `file_path`: Path to the JSON file
- **Returns**: Dictionary with metadata and items

### CSVProcessor Methods

#### `load_csv(file_path: Union[str, Path]) -> Dict[str, Any]`

Loads a CSV file and extracts text.

- **Parameters**:
  - `file_path`: Path to the CSV file
- **Returns**: Dictionary with metadata and rows

## LLMFormatter Interface

The `LLMFormatter` class provides an interface for LLM-based response formatting.

### Methods

#### `format_response(query: str, context: List[Dict[str, Any]], **kwargs) -> str`

Formats RAG results into a structured response.

- **Parameters**:
  - `query`: The original query
  - `context`: List of retrieved documents with content and metadata
  - `**kwargs`: Additional arguments
- **Returns**: Formatted response as a string

## Caching Interface

The caching classes provide interfaces for caching different types of data.

### EmbeddingCache Methods

#### `get(text: str) -> Optional[List[float]]`

Gets an embedding from the cache.

- **Parameters**:
  - `text`: Text to get embedding for
- **Returns**: Cached embedding or None if not found

#### `put(text: str, embedding: List[float]) -> None`

Puts an embedding in the cache.

- **Parameters**:
  - `text`: Text to cache embedding for
  - `embedding`: Embedding to cache

#### `clear() -> None`

Clears the cache.

### QueryCache Methods

#### `get(query: str, **kwargs) -> Optional[List[Dict[str, Any]]]`

Gets query results from the cache.

- **Parameters**:
  - `query`: Query string
  - `**kwargs`: Additional query parameters
- **Returns**: Cached query results or None if not found

#### `put(query: str, results: List[Dict[str, Any]], **kwargs) -> None`

Puts query results in the cache.

- **Parameters**:
  - `query`: Query string
  - `results`: Query results to cache
  - `**kwargs`: Additional query parameters

#### `clear() -> None`

Clears the cache.

## Debug Utilities Interface

The debug utilities provide interfaces for debugging the RAG system.

### DebugLogger Methods

#### `start_timer(operation: str) -> None`

Starts a timer for performance tracking.

- **Parameters**:
  - `operation`: Operation name

#### `end_timer(operation: str) -> Optional[float]`

Ends a timer and logs the duration.

- **Parameters**:
  - `operation`: Operation name
- **Returns**: Duration in seconds or None if timer not found

#### `log_function_call(func_name: str, args: tuple, kwargs: dict) -> None`

Logs a function call with its arguments.

- **Parameters**:
  - `func_name`: Function name
  - `args`: Function arguments
  - `kwargs`: Function keyword arguments

#### `log_object_state(obj: Any, name: str = "object") -> None`

Logs the state of an object.

- **Parameters**:
  - `obj`: Object to log
  - `name`: Name to use for the object

#### `log_exception(e: Exception, context: str = "") -> None`

Logs an exception with stack trace.

- **Parameters**:
  - `e`: Exception to log
  - `context`: Context for the exception

### DebugInspector Methods

#### `inspect_object(obj: Any) -> Dict[str, Any]`

Inspects an object and returns its attributes and methods.

- **Parameters**:
  - `obj`: Object to inspect
- **Returns**: Dictionary with object information

#### `print_object_info(obj: Any, name: str = "object") -> None`

Prints information about an object.

- **Parameters**:
  - `obj`: Object to inspect
  - `name`: Name to use for the object

#### `inspect_vector_store(vector_store: Any) -> Dict[str, Any]`

Inspects a vector store and returns information about it.

- **Parameters**:
  - `vector_store`: Vector store to inspect
- **Returns**: Dictionary with vector store information

#### `print_vector_store_info(vector_store: Any) -> None`

Prints information about a vector store.

- **Parameters**:
  - `vector_store`: Vector store to inspect
