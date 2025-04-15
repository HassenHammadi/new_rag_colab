"""
FAISS vector store with Google Drive integration and debugging capabilities.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import os
import json
import pickle
import numpy as np
import tempfile
import time
import traceback
from pathlib import Path

try:
    import faiss
except ImportError:
    raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")

from ..utils.drive_utils import DriveHandler
from ..utils.debug_utils import debug_logger, debug_function, DebugInspector

class DriveVectorStore:
    """Vector store implementation using FAISS with Google Drive integration and debugging capabilities."""

    def __init__(self, embedding_function: Callable, dimension: int = 768, drive_handler: Optional[DriveHandler] = None,
                 debug: bool = True):
        """
        Initialize the Drive vector store.

        Args:
            embedding_function: Function to convert text to embeddings
            dimension: Dimension of the embeddings
            drive_handler: Handler for Google Drive operations
            debug: Whether to enable debugging
        """
        self.embedding_function = embedding_function
        self.dimension = dimension
        self.documents = []
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance index
        self.debug = debug

        # Initialize Drive handler if not provided
        if drive_handler is None:
            self.drive_handler = DriveHandler()
        else:
            self.drive_handler = drive_handler

        if self.debug:
            debug_logger.logger.info(f"Initialized DriveVectorStore with dimension {dimension}")
            debug_logger.logger.debug(f"Using Drive handler: {type(self.drive_handler).__name__}")

    @debug_function(log_args=False)
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document dictionaries with content and metadata

        Returns:
            List of document IDs
        """
        if not documents:
            if self.debug:
                debug_logger.logger.warning("add_documents called with empty document list")
            return []

        if self.debug:
            debug_logger.logger.info(f"Adding {len(documents)} documents to vector store")
            debug_logger.start_timer("add_documents")

        document_ids = []
        embeddings = []

        try:
            for i, doc in enumerate(documents):
                try:
                    # Generate a document ID
                    doc_id = f"doc_{len(self.documents) + i}"

                    # Create a copy of the document with the ID
                    document = doc.copy()
                    document["id"] = doc_id

                    # Check if document has content
                    if "content" not in document or not document["content"]:
                        if self.debug:
                            debug_logger.logger.warning(f"Document {doc_id} has no content, skipping")
                        continue

                    # Generate the embedding
                    if self.debug:
                        debug_logger.start_timer(f"embedding_{doc_id}")

                    embedding = self.embedding_function(document["content"])

                    if self.debug:
                        debug_logger.end_timer(f"embedding_{doc_id}")

                    # Validate embedding dimension
                    if len(embedding) != self.dimension:
                        if self.debug:
                            debug_logger.logger.error(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}")
                        raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}")

                    # Store the document
                    self.documents.append(document)
                    embeddings.append(embedding)
                    document_ids.append(doc_id)

                except Exception as e:
                    if self.debug:
                        debug_logger.logger.error(f"Error processing document {i}: {str(e)}")
                        debug_logger.logger.debug(f"Document content preview: {doc.get('content', '')[:100]}...")

            # Check if we have any valid embeddings
            if not embeddings:
                if self.debug:
                    debug_logger.logger.warning("No valid embeddings generated")
                return []

            # Convert embeddings to numpy array and add to FAISS index
            if self.debug:
                debug_logger.start_timer("faiss_add")

            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)

            if self.debug:
                debug_logger.end_timer("faiss_add")
                debug_logger.logger.info(f"Added {len(document_ids)} documents to vector store")
                debug_logger.end_timer("add_documents")

            return document_ids

        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error in add_documents: {str(e)}")
                debug_logger.logger.debug(traceback.format_exc())
            raise

    @debug_function(log_args=True)
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of document dictionaries with content, metadata, and similarity score
        """
        if not self.documents:
            if self.debug:
                debug_logger.logger.warning("similarity_search called with empty document store")
            return []

        if self.debug:
            debug_logger.logger.info(f"Searching for query: '{query[:50]}...' with k={k}")
            debug_logger.start_timer("similarity_search")

        try:
            # Generate the query embedding
            if self.debug:
                debug_logger.start_timer("query_embedding")

            query_embedding = self.embedding_function(query)

            if self.debug:
                debug_logger.end_timer("query_embedding")
                debug_logger.logger.debug(f"Query embedding dimension: {len(query_embedding)}")

            # Validate embedding dimension
            if len(query_embedding) != self.dimension:
                if self.debug:
                    debug_logger.logger.error(f"Query embedding dimension mismatch: expected {self.dimension}, got {len(query_embedding)}")
                raise ValueError(f"Query embedding dimension mismatch: expected {self.dimension}, got {len(query_embedding)}")

            query_array = np.array([query_embedding]).astype('float32')

            # Search the FAISS index
            k = min(k, len(self.documents))

            if self.debug:
                debug_logger.start_timer("faiss_search")

            distances, indices = self.index.search(query_array, k)

            if self.debug:
                debug_logger.end_timer("faiss_search")
                debug_logger.logger.debug(f"FAISS search returned {len(indices[0])} results")

            # Convert distances to similarity scores (smaller distance = higher similarity)
            max_distance = np.max(distances) + 1e-6  # Avoid division by zero

            # Return the results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):  # Ensure index is valid
                    document = self.documents[idx].copy()
                    # Convert distance to similarity score (1 = most similar, 0 = least similar)
                    document["score"] = 1.0 - (distances[0][i] / max_distance)
                    results.append(document)

            if self.debug:
                debug_logger.logger.info(f"Returning {len(results)} results")
                for i, result in enumerate(results[:3]):  # Log first 3 results
                    debug_logger.logger.debug(f"Result {i+1}: score={result['score']:.4f}, source={result.get('metadata', {}).get('source_file', 'unknown')}")
                debug_logger.end_timer("similarity_search")

            return results

        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error in similarity_search: {str(e)}")
                debug_logger.logger.debug(traceback.format_exc())
            raise

    @debug_function()
    def save(self, directory: Union[str, Path], drive_subfolder: Optional[str] = None) -> None:
        """
        Save the vector store to disk and optionally to Google Drive.

        Args:
            directory: Directory to save to
            drive_subfolder: Optional subfolder in Google Drive to save to
        """
        try:
            directory = Path(directory)
            os.makedirs(directory, exist_ok=True)

            if self.debug:
                debug_logger.logger.info(f"Saving vector store to {directory}")
                debug_logger.start_timer("save_vector_store")
                debug_logger.logger.debug(f"Vector store contains {len(self.documents)} documents")

            # Save documents as JSON
            if self.debug:
                debug_logger.start_timer("save_documents")

            with open(directory / "documents.json", "w") as f:
                json.dump(self.documents, f)

            if self.debug:
                debug_logger.end_timer("save_documents")

            # Save FAISS index
            if self.debug:
                debug_logger.start_timer("save_faiss_index")

            faiss.write_index(self.index, str(directory / "faiss.index"))

            if self.debug:
                debug_logger.end_timer("save_faiss_index")

            # Save dimension and other metadata
            with open(directory / "metadata.json", "w") as f:
                metadata = {
                    "dimension": self.dimension,
                    "document_count": len(self.documents),
                    "timestamp": time.time()
                }
                json.dump(metadata, f)

            # Save to Google Drive if requested
            if drive_subfolder is not None:
                if self.debug:
                    debug_logger.logger.info(f"Saving vector store to Google Drive: {drive_subfolder}")
                    debug_logger.start_timer("save_to_drive")

                success = self.drive_handler.save_to_drive(directory, drive_subfolder)

                if self.debug:
                    debug_logger.end_timer("save_to_drive")
                    if success:
                        debug_logger.logger.info(f"Successfully saved to Google Drive: {drive_subfolder}")
                    else:
                        debug_logger.logger.error(f"Failed to save to Google Drive: {drive_subfolder}")

            if self.debug:
                debug_logger.end_timer("save_vector_store")
                debug_logger.logger.info(f"Vector store saved successfully to {directory}")

        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error saving vector store: {str(e)}")
                debug_logger.logger.debug(traceback.format_exc())
            raise

    @debug_function()
    def load(self, directory: Union[str, Path], from_drive: bool = False, drive_path: Optional[str] = None) -> None:
        """
        Load the vector store from disk or Google Drive.

        Args:
            directory: Directory to load from
            from_drive: Whether to load from Google Drive
            drive_path: Path within the base folder in Google Drive
        """
        try:
            if self.debug:
                debug_logger.logger.info(f"Loading vector store from {'Google Drive' if from_drive else 'local directory'}")
                debug_logger.start_timer("load_vector_store")

            # If loading from Drive, first copy to local directory
            if from_drive and drive_path is not None:
                if self.debug:
                    debug_logger.logger.info(f"Loading from Google Drive: {drive_path}")
                    debug_logger.start_timer("load_from_drive")

                temp_dir = self.drive_handler.load_from_drive(drive_path, directory)

                if self.debug:
                    debug_logger.end_timer("load_from_drive")

                if temp_dir is None:
                    error_msg = f"Failed to load vector store from Google Drive: {drive_path}"
                    if self.debug:
                        debug_logger.logger.error(error_msg)
                    raise ValueError(error_msg)

                directory = temp_dir

                if self.debug:
                    debug_logger.logger.info(f"Files loaded from Google Drive to {directory}")

            directory = Path(directory)

            # Check if required files exist
            required_files = ["documents.json", "faiss.index", "metadata.json"]
            for file in required_files:
                if not (directory / file).exists():
                    error_msg = f"Required file {file} not found in {directory}"
                    if self.debug:
                        debug_logger.logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

            # Load documents
            if self.debug:
                debug_logger.start_timer("load_documents")

            with open(directory / "documents.json", "r") as f:
                self.documents = json.load(f)

            if self.debug:
                debug_logger.end_timer("load_documents")
                debug_logger.logger.info(f"Loaded {len(self.documents)} documents")

            # Load metadata
            with open(directory / "metadata.json", "r") as f:
                metadata = json.load(f)
                self.dimension = metadata["dimension"]

                if self.debug and "timestamp" in metadata:
                    age = time.time() - metadata["timestamp"]
                    debug_logger.logger.debug(f"Vector store age: {age/86400:.1f} days")

            # Load FAISS index
            if self.debug:
                debug_logger.start_timer("load_faiss_index")

            self.index = faiss.read_index(str(directory / "faiss.index"))

            if self.debug:
                debug_logger.end_timer("load_faiss_index")
                debug_logger.logger.info(f"Loaded FAISS index with dimension {self.dimension}")
                debug_logger.end_timer("load_vector_store")

                # Validate index size matches document count
                if self.index.ntotal != len(self.documents):
                    debug_logger.logger.warning(f"Index size mismatch: FAISS index has {self.index.ntotal} vectors but documents list has {len(self.documents)} items")

        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error loading vector store: {str(e)}")
                debug_logger.logger.debug(traceback.format_exc())
            raise

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the vector store.

        Returns:
            Dictionary with debug information
        """
        info = {
            "document_count": len(self.documents),
            "dimension": self.dimension,
            "index_size": self.index.ntotal if hasattr(self.index, "ntotal") else "unknown",
            "metadata_fields": set(),
            "source_files": set(),
            "source_types": set()
        }

        # Analyze documents
        for doc in self.documents[:min(100, len(self.documents))]:
            if "metadata" in doc:
                for field in doc["metadata"].keys():
                    info["metadata_fields"].add(field)

                if "source_file" in doc["metadata"]:
                    info["source_files"].add(doc["metadata"]["source_file"])

                if "source_type" in doc["metadata"]:
                    info["source_types"].add(doc["metadata"]["source_type"])

        # Convert sets to lists for JSON serialization
        info["metadata_fields"] = list(info["metadata_fields"])
        info["source_files"] = list(info["source_files"])
        info["source_types"] = list(info["source_types"])

        return info
