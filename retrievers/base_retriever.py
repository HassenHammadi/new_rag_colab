"""
Base retriever implementations for retrieving documents.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union
import time

from ..utils.debug_utils import debug_logger, debug_function

class BaseRetriever:
    """Base class for document retrievers."""
    
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to a query.
        
        Args:
            query: Query string
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents
        """
        raise NotImplementedError("Subclasses must implement retrieve")


class SimpleRetriever(BaseRetriever):
    """Simple retriever that uses vector similarity search."""
    
    def __init__(self, vector_store: Any, top_k: int = 4, debug: bool = True):
        """
        Initialize the simple retriever.
        
        Args:
            vector_store: Vector store to search
            top_k: Number of results to return
            debug: Whether to enable debugging
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized SimpleRetriever with top_k: {top_k}")
    
    @debug_function()
    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to a query using vector similarity.
        
        Args:
            query: Query string
            top_k: Number of results to return (overrides the default)
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents
        """
        if not query:
            if self.debug:
                debug_logger.logger.warning("Empty query provided")
            return []
        
        # Use provided top_k or default
        k = top_k if top_k is not None else self.top_k
        
        if self.debug:
            debug_logger.logger.info(f"Retrieving documents for query: '{query[:50]}...' with top_k={k}")
            debug_logger.start_timer("retrieve")
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=k)
            
            if self.debug:
                debug_logger.logger.info(f"Retrieved {len(results)} documents")
                debug_logger.end_timer("retrieve")
            
            return results
        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error retrieving documents: {str(e)}")
            raise
