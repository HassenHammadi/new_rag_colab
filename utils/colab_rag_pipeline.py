"""
RAG pipeline for Google Colab with Google Drive integration.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import os
import time
import traceback
from pathlib import Path

from ..chunkers.base_chunker import BaseChunker
from ..vector_stores.drive_vector_store import DriveVectorStore
from ..retrievers.base_retriever import BaseRetriever
from ..utils.drive_utils import DriveHandler
from ..utils.llm_integration import LLMFormatter, MarkdownFormatter

class ColabRAGPipeline:
    """RAG pipeline for Google Colab with Google Drive integration."""
    
    def __init__(self,
                 chunker: BaseChunker,
                 vector_store: DriveVectorStore,
                 retriever: BaseRetriever,
                 processors: Dict[str, Any] = None,
                 response_formatter: Optional[LLMFormatter] = None,
                 drive_handler: Optional[DriveHandler] = None,
                 use_query_cache: bool = True,
                 query_cache_size: int = 100):
        """
        Initialize the Colab RAG pipeline.
        
        Args:
            chunker: Chunker for splitting documents
            vector_store: Vector store for storing embeddings
            retriever: Retriever for retrieving documents
            processors: Dictionary of processors for different file types
            response_formatter: Formatter for enhancing responses
            drive_handler: Handler for Google Drive operations
            use_query_cache: Whether to cache query results
            query_cache_size: Maximum number of queries to cache
        """
        self.chunker = chunker
        self.vector_store = vector_store
        self.retriever = retriever
        
        # Initialize processors
        self.processors = processors or {}
        
        # Initialize response formatter
        self.response_formatter = response_formatter
        
        # Initialize Drive handler if not provided
        if drive_handler is None:
            self.drive_handler = DriveHandler()
        else:
            self.drive_handler = drive_handler
        
        # Initialize query cache
        self.use_query_cache = use_query_cache
        if use_query_cache:
            self.query_cache = {}
            self.query_cache_size = query_cache_size
        else:
            self.query_cache = None
        
        # Mount Google Drive if we're in Colab
        try:
            import google.colab
            self.in_colab = True
            self.drive_handler.mount_drive()
        except ImportError:
            self.in_colab = False
            print("Not running in Google Colab. Drive integration will be limited.")
    
    def process_file(self, file_path: Union[str, Path], file_type: Optional[str] = None) -> List[str]:
        """
        Process a file and add it to the vector store.
        
        Args:
            file_path: Path to the file
            file_type: Type of file (pdf, text, json, csv) or None to infer from extension
            
        Returns:
            List of document IDs
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return []
            
            # Infer file type from extension if not provided
            if file_type is None:
                extension = file_path.suffix.lower()
                if extension == ".pdf":
                    file_type = "pdf"
                elif extension in [".txt", ".md", ".rst"]:
                    file_type = "text"
                elif extension == ".json":
                    file_type = "json"
                elif extension == ".csv":
                    file_type = "csv"
                else:
                    print(f"Unsupported file type: {extension}")
                    return []
            
            # Check if we have a processor for this file type
            if file_type not in self.processors:
                print(f"No processor available for file type: {file_type}")
                return []
            
            # Process the file based on its type
            processor = self.processors[file_type]
            
            if file_type == "pdf":
                document = processor.load_pdf(file_path)
            elif file_type == "text":
                document = processor.load_text(file_path)
                # Convert to format expected by chunker
                document = {
                    "metadata": document["metadata"],
                    "pages": [{
                        "page_number": 1,
                        "content": document["content"],
                        "metadata": {}
                    }]
                }
            elif file_type == "json":
                document = processor.load_json(file_path)
                # Process each item as a separate document
                chunks = []
                for i, item in enumerate(document["items"]):
                    chunk = {
                        "content": item["content"],
                        "metadata": {
                            "source_file": file_path.name,
                            "source_type": "json",
                            "item_index": i,
                            "json_path": item["metadata"].get("path", ""),
                            "field": item["metadata"].get("field", "")
                        }
                    }
                    chunks.append(chunk)
                return self.vector_store.add_documents(chunks)
            elif file_type == "csv":
                document = processor.load_csv(file_path)
                # Process each row as a separate document
                chunks = []
                for row in document["rows"]:
                    chunk = {
                        "content": row["content"],
                        "metadata": {
                            "source_file": file_path.name,
                            "source_type": "csv",
                            "row_number": row["metadata"]["row_number"],
                            "headers": document["metadata"]["headers"]
                        }
                    }
                    # Add row data to metadata
                    for key, value in row["metadata"]["row_data"].items():
                        chunk["metadata"][f"row_{key}"] = value
                    chunks.append(chunk)
                return self.vector_store.add_documents(chunks)
            
            # Chunk the document
            chunks = self.chunker.chunk_document(document)
            
            # Add source file information to metadata
            for chunk in chunks:
                chunk["metadata"]["source_file"] = file_path.name
                chunk["metadata"]["source_type"] = file_type
            
            # Add chunks to the vector store
            doc_ids = self.vector_store.add_documents(chunks)
            
            return doc_ids
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            traceback.print_exc()
            return []
    
    def process_directory(self, directory: Union[str, Path], 
                          file_types: Optional[List[str]] = None,
                          recursive: bool = False) -> Dict[str, List[str]]:
        """
        Process all supported files in a directory.
        
        Args:
            directory: Directory containing files
            file_types: List of file types to process (pdf, text, json, csv) or None for all
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary mapping filenames to document IDs
        """
        try:
            directory = Path(directory)
            
            # Check if directory exists
            if not directory.exists():
                print(f"Directory not found: {directory}")
                return {}
            
            results = {}
            
            # Determine which file types to process
            supported_extensions = []
            if file_types is None:
                file_types = []
                if "pdf" in self.processors:
                    file_types.append("pdf")
                if "text" in self.processors:
                    file_types.append("text")
                if "json" in self.processors:
                    file_types.append("json")
                if "csv" in self.processors:
                    file_types.append("csv")
            
            # Map file types to extensions
            for file_type in file_types:
                if file_type == "pdf":
                    supported_extensions.append(".pdf")
                elif file_type == "text":
                    supported_extensions.extend([".txt", ".md", ".rst"])
                elif file_type == "json":
                    supported_extensions.append(".json")
                elif file_type == "csv":
                    supported_extensions.append(".csv")
            
            # Process files
            for extension in supported_extensions:
                # Use rglob for recursive search, glob for non-recursive
                glob_func = directory.rglob if recursive else directory.glob
                for file_path in glob_func(f"*{extension}"):
                    try:
                        doc_ids = self.process_file(file_path)
                        if doc_ids:
                            results[file_path.name] = doc_ids
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
            
            return results
        except Exception as e:
            print(f"Error processing directory {directory}: {str(e)}")
            return {}
    
    def query(self, query: str, use_cache: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """
        Query the RAG pipeline.
        
        Args:
            query: Query string
            use_cache: Whether to use query cache (if enabled)
            **kwargs: Additional arguments to pass to the retriever
            
        Returns:
            List of retrieved documents
        """
        # Check if query is empty or too short
        if not query or len(query.strip()) < 2:
            print("Query is empty or too short")
            return []
            
        # Normalize query
        normalized_query = query.strip()
        
        # Check cache if enabled
        if self.use_query_cache and self.query_cache is not None and use_cache:
            cache_key = f"{normalized_query}_{kwargs.get('top_k', 4)}"
            if cache_key in self.query_cache:
                print(f"Using cached results for query: '{normalized_query[:50]}...'")
                return self.query_cache[cache_key]
        
        # Execute the query
        try:
            results = self.retriever.retrieve(normalized_query, **kwargs)
            
            # Cache the results if caching is enabled
            if self.use_query_cache and self.query_cache is not None and use_cache:
                cache_key = f"{normalized_query}_{kwargs.get('top_k', 4)}"
                self.query_cache[cache_key] = results
                
                # Limit cache size
                if len(self.query_cache) > self.query_cache_size:
                    # Remove oldest entry (first key)
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
            
            return results
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            traceback.print_exc()
            return []
    
    def query_with_markdown(self, query: str, **kwargs) -> str:
        """
        Query the RAG pipeline and return a Markdown-formatted response.
        
        Args:
            query: Query string
            **kwargs: Additional arguments to pass to the retriever or formatter
            
        Returns:
            Markdown-formatted response string
        """
        # Get the raw results
        results = self.query(query, **kwargs)
        
        # If we have a response formatter, use it
        if self.response_formatter is not None:
            # Extract formatter-specific parameters
            formatter_kwargs = {}
            for key in list(kwargs.keys()):
                if key.startswith('formatter_'):
                    formatter_kwargs[key[10:]] = kwargs.pop(key)
            
            # Format the response using the LLM
            return self.response_formatter.format_response(
                query=query,
                context=results,
                response_format="markdown",
                **formatter_kwargs
            )
        else:
            # Use the simple Markdown formatter
            return MarkdownFormatter.format_response(query, results)
    
    def save_vector_store(self, directory: Union[str, Path], drive_subfolder: Optional[str] = None) -> None:
        """
        Save the vector store to disk and optionally to Google Drive.
        
        Args:
            directory: Directory to save to
            drive_subfolder: Optional subfolder in Google Drive to save to
        """
        self.vector_store.save(directory, drive_subfolder)
    
    def load_vector_store(self, directory: Union[str, Path], from_drive: bool = False, drive_path: Optional[str] = None) -> None:
        """
        Load the vector store from disk or Google Drive.
        
        Args:
            directory: Directory to load from
            from_drive: Whether to load from Google Drive
            drive_path: Path within the base folder in Google Drive
        """
        self.vector_store.load(directory, from_drive, drive_path)
    
    def list_drive_vector_stores(self) -> list:
        """
        List available vector stores in Google Drive.
        
        Returns:
            List of vector store names
        """
        return self.drive_handler.list_vector_stores()
