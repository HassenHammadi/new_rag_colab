"""
Advanced retriever implementations for retrieving documents.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union, Set
import re
import time

from .base_retriever import BaseRetriever
from ..utils.debug_utils import debug_logger, debug_function

class QueryTransformationRetriever(BaseRetriever):
    """Retriever that transforms queries before retrieval."""
    
    def __init__(self, base_retriever: BaseRetriever, debug: bool = True):
        """
        Initialize the query transformation retriever.
        
        Args:
            base_retriever: Base retriever to use for retrieval
            debug: Whether to enable debugging
        """
        self.base_retriever = base_retriever
        self.debug = debug
        
        # Define query expansions
        self.expansions = {
            "ML": "machine learning",
            "AI": "artificial intelligence",
            "NLP": "natural language processing",
            "CV": "computer vision",
            "DL": "deep learning",
            "RL": "reinforcement learning",
            "NN": "neural network",
            "CNN": "convolutional neural network",
            "RNN": "recurrent neural network",
            "LSTM": "long short-term memory",
            "GAN": "generative adversarial network",
            "API": "application programming interface",
            "DB": "database",
            "UI": "user interface",
            "UX": "user experience"
        }
        
        if self.debug:
            debug_logger.logger.info(f"Initialized QueryTransformationRetriever with {len(self.expansions)} expansions")
    
    @debug_function()
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Transform the query and retrieve documents.
        
        Args:
            query: Query string
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents
        """
        if not query:
            if self.debug:
                debug_logger.logger.warning("Empty query provided")
            return []
        
        if self.debug:
            debug_logger.logger.info(f"Transforming query: '{query[:50]}...'")
            debug_logger.start_timer("transform_query")
        
        # Transform the query
        transformed_query = self._transform_query(query)
        
        if self.debug:
            if transformed_query != query:
                debug_logger.logger.info(f"Transformed query: '{transformed_query[:50]}...'")
            else:
                debug_logger.logger.info("Query not transformed")
            debug_logger.end_timer("transform_query")
        
        # Retrieve documents using the transformed query
        results = self.base_retriever.retrieve(transformed_query, **kwargs)
        
        # Add the original query to the results metadata
        for result in results:
            if "metadata" in result:
                result["metadata"]["original_query"] = query
                result["metadata"]["transformed_query"] = transformed_query
        
        return results
    
    def _transform_query(self, query: str) -> str:
        """
        Transform a query by expanding acronyms and adding synonyms.
        
        Args:
            query: Query string
            
        Returns:
            Transformed query
        """
        # Expand acronyms
        expanded_query = query
        for term, expansion in self.expansions.items():
            # Check if the term is a standalone word
            if re.search(r'\b' + re.escape(term) + r'\b', query, re.IGNORECASE):
                expanded_query += f" {expansion}"
        
        return expanded_query


class FusionRetriever(BaseRetriever):
    """Retriever that combines results from multiple retrievers."""
    
    def __init__(self, retrievers: List[BaseRetriever], weights: Optional[List[float]] = None, debug: bool = True):
        """
        Initialize the fusion retriever.
        
        Args:
            retrievers: List of retrievers to use
            weights: List of weights for each retriever (must match the length of retrievers)
            debug: Whether to enable debugging
        """
        self.retrievers = retrievers
        
        # If weights are not provided, use equal weights
        if weights is None:
            self.weights = [1.0] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                error_msg = f"Number of weights ({len(weights)}) must match number of retrievers ({len(retrievers)})"
                if debug:
                    debug_logger.logger.error(error_msg)
                raise ValueError(error_msg)
            self.weights = weights
        
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized FusionRetriever with {len(retrievers)} retrievers")
    
    @debug_function()
    def retrieve(self, query: str, top_k: int = 4, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents by combining results from multiple retrievers.
        
        Args:
            query: Query string
            top_k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents
        """
        if not query:
            if self.debug:
                debug_logger.logger.warning("Empty query provided")
            return []
        
        if self.debug:
            debug_logger.logger.info(f"Retrieving documents with fusion for query: '{query[:50]}...'")
            debug_logger.start_timer("retrieve_fusion")
        
        # Get results from each retriever
        all_results = []
        
        for i, retriever in enumerate(self.retrievers):
            try:
                if self.debug:
                    debug_logger.start_timer(f"retriever_{i}")
                
                # Get results from this retriever
                results = retriever.retrieve(query, top_k=top_k, **kwargs)
                
                # Add retriever ID and weight to each result
                for result in results:
                    result["retriever_id"] = i
                    result["weight"] = self.weights[i]
                    
                    # Adjust score by weight
                    if "score" in result:
                        result["score"] *= self.weights[i]
                
                all_results.extend(results)
                
                if self.debug:
                    debug_logger.end_timer(f"retriever_{i}")
                    debug_logger.logger.debug(f"Retriever {i} returned {len(results)} results")
            except Exception as e:
                if self.debug:
                    debug_logger.logger.error(f"Error in retriever {i}: {str(e)}")
        
        # Combine results
        combined_results = {}
        
        for result in all_results:
            doc_id = result.get("id")
            if doc_id in combined_results:
                # Update existing result
                combined_results[doc_id]["score"] += result["score"]
                combined_results[doc_id]["retrievers"].append(result["retriever_id"])
            else:
                # Add new result
                combined_result = result.copy()
                combined_result["retrievers"] = [result["retriever_id"]]
                combined_results[doc_id] = combined_result
        
        # Convert to list and sort by score
        results = list(combined_results.values())
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Limit to top_k results
        results = results[:top_k]
        
        if self.debug:
            debug_logger.logger.info(f"Fusion returned {len(results)} results")
            debug_logger.end_timer("retrieve_fusion")
        
        return results


class AdaptiveRetriever(BaseRetriever):
    """Retriever that adapts the retrieval strategy based on the query."""
    
    def __init__(self, vector_store: Any, debug: bool = True):
        """
        Initialize the adaptive retriever.
        
        Args:
            vector_store: Vector store to search
            debug: Whether to enable debugging
        """
        self.vector_store = vector_store
        self.debug = debug
        
        # Define query type patterns
        self.query_patterns = {
            "definition": [
                r"what is", r"define", r"meaning of", r"definition of",
                r"explain", r"describe", r"tell me about"
            ],
            "comparison": [
                r"compare", r"difference between", r"similarities between",
                r"versus", r"vs", r"better", r"worse", r"advantages"
            ],
            "list": [
                r"list", r"examples of", r"types of", r"kinds of",
                r"ways to", r"methods for", r"steps"
            ]
        }
        
        if self.debug:
            debug_logger.logger.info("Initialized AdaptiveRetriever")
    
    @debug_function()
    def retrieve(self, query: str, top_k: int = 4, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents using an adaptive strategy based on the query type.
        
        Args:
            query: Query string
            top_k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents
        """
        if not query:
            if self.debug:
                debug_logger.logger.warning("Empty query provided")
            return []
        
        # Determine the query type
        query_type = self._classify_query(query)
        
        if self.debug:
            debug_logger.logger.info(f"Query classified as: {query_type}")
            debug_logger.start_timer(f"retrieve_{query_type}")
        
        # Adapt the retrieval strategy based on the query type
        if query_type == "definition":
            # For definition queries, use precise retrieval with higher threshold
            results = self._retrieve_precise(query, top_k)
        elif query_type == "comparison":
            # For comparison queries, use diverse retrieval
            results = self._retrieve_diverse(query, top_k)
        elif query_type == "list":
            # For list queries, use comprehensive retrieval
            results = self._retrieve_comprehensive(query, top_k)
        else:
            # For other queries, use default retrieval
            results = self._retrieve_default(query, top_k)
        
        # Add query type to metadata
        for result in results:
            if "metadata" in result:
                result["metadata"]["query_type"] = query_type
        
        if self.debug:
            debug_logger.logger.info(f"Retrieved {len(results)} documents for {query_type} query")
            debug_logger.end_timer(f"retrieve_{query_type}")
        
        return results
    
    def _classify_query(self, query: str) -> str:
        """
        Classify a query based on patterns.
        
        Args:
            query: Query string
            
        Returns:
            Query type
        """
        query = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + re.escape(pattern) + r'\b', query):
                    return query_type
        
        return "general"
    
    def _retrieve_precise(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents with high precision.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Get more results than needed and filter by score
        results = self.vector_store.similarity_search(query, k=top_k * 2)
        
        # Filter results with high scores
        filtered_results = [r for r in results if r.get("score", 0) > 0.7]
        
        # If we don't have enough results, use the top ones
        if len(filtered_results) < top_k:
            filtered_results = results[:top_k]
        else:
            filtered_results = filtered_results[:top_k]
        
        return filtered_results
    
    def _retrieve_diverse(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve diverse documents.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Get more results than needed
        results = self.vector_store.similarity_search(query, k=top_k * 2)
        
        # Deduplicate results based on content similarity
        deduplicated_results = []
        seen_content = set()
        
        for result in results:
            # Create a content fingerprint (first 100 chars)
            content = result.get("content", "")
            fingerprint = content[:100].lower()
            
            # Check if we've seen similar content
            if not any(self._is_similar(fingerprint, seen) for seen in seen_content):
                deduplicated_results.append(result)
                seen_content.add(fingerprint)
                
                # Stop when we have enough results
                if len(deduplicated_results) >= top_k:
                    break
        
        return deduplicated_results[:top_k]
    
    def _retrieve_comprehensive(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve comprehensive results.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # For list queries, we want to get more results
        results = self.vector_store.similarity_search(query, k=top_k * 2)
        
        # Group results by source
        source_groups = {}
        
        for result in results:
            source = result.get("metadata", {}).get("source_file", "unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(result)
        
        # Take the top result from each source
        diverse_results = []
        
        for source, group in source_groups.items():
            # Sort by score
            group.sort(key=lambda x: x.get("score", 0), reverse=True)
            diverse_results.append(group[0])
        
        # If we don't have enough results, add more from the top sources
        if len(diverse_results) < top_k:
            remaining = top_k - len(diverse_results)
            additional_results = []
            
            for source, group in source_groups.items():
                if len(group) > 1:
                    additional_results.extend(group[1:])
            
            # Sort by score and add the top remaining
            additional_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            diverse_results.extend(additional_results[:remaining])
        
        # Sort by score
        diverse_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return diverse_results[:top_k]
    
    def _retrieve_default(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Default retrieval strategy.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        return self.vector_store.similarity_search(query, k=top_k)
    
    def _is_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two texts are similar.
        
        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold
            
        Returns:
            True if the texts are similar, False otherwise
        """
        # Simple similarity check based on character overlap
        if not text1 or not text2:
            return False
        
        # Count common characters
        common_chars = sum(1 for c in text1 if c in text2)
        
        # Calculate similarity
        similarity = common_chars / max(len(text1), len(text2))
        
        return similarity >= threshold
