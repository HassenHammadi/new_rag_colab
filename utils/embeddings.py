"""
Embedding utilities for the RAG system.
Adapted for Google Colab environment.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import os
import numpy as np
import requests
import json
import time
import traceback
from pathlib import Path

from .debug_utils import debug_logger, debug_function

class EmbeddingProvider:
    """Base class for embedding providers with caching and error handling."""
    
    def __init__(self, use_cache: bool = True, cache_dir: Optional[str] = "cache/embeddings", debug: bool = True):
        """
        Initialize the embedding provider.
        
        Args:
            use_cache: Whether to use caching
            cache_dir: Directory to store embedding cache
            debug: Whether to enable debugging
        """
        self.use_cache = use_cache
        self.debug = debug
        self.cache = {}
        
        if self.debug:
            debug_logger.logger.info(f"Initialized EmbeddingProvider with caching: {use_cache}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        raise NotImplementedError("Subclasses must implement get_embeddings")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first if enabled
        if self.use_cache and text in self.cache:
            if self.debug:
                debug_logger.logger.debug(f"Cache hit for text: {text[:50]}...")
            return self.cache[text]
        
        try:
            # Get embedding from provider
            embedding = self.get_embeddings([text])[0]
            
            # Cache the result if caching is enabled
            if self.use_cache:
                self.cache[text] = embedding
                
            return embedding
        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback (not ideal but prevents system crash)
            return [0.0] * self._get_default_dimension()
    
    def _get_default_dimension(self) -> int:
        """Get the default embedding dimension for fallback."""
        return 384  # Default dimension for many embedding models


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Hugging Face models with enhanced error handling."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 use_api: bool = False, use_cache: bool = True, max_retries: int = 3, debug: bool = True):
        """
        Initialize the Hugging Face embedding provider.
        
        Args:
            model_name: Name of the Hugging Face model
            use_api: Whether to use the Hugging Face API (requires API key) or local model
            use_cache: Whether to use caching
            max_retries: Maximum number of retries for API calls
            debug: Whether to enable debugging
        """
        super().__init__(use_cache=use_cache, debug=debug)
        self.model_name = model_name
        self.use_api = use_api
        self.max_retries = max_retries
        self.dimension = 384  # Default dimension for most sentence-transformers models
        
        if self.debug:
            debug_logger.logger.info(f"Initializing HuggingFaceEmbeddingProvider with model: {model_name}, API: {use_api}")
        
        if use_api:
            # Check for API key if using the API
            self.api_key = os.environ.get("HF_API_KEY")
            if not self.api_key:
                error_msg = "Hugging Face API key not found. Set the HF_API_KEY environment variable."
                if self.debug:
                    debug_logger.logger.error(error_msg)
                raise ValueError(error_msg)
            self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
            if self.debug:
                debug_logger.logger.info(f"Using Hugging Face API with URL: {self.api_url}")
        else:
            # Load the model locally
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                if self.debug:
                    debug_logger.logger.info("Loading Hugging Face model locally...")
                    debug_logger.start_timer("load_hf_model")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                
                if self.debug:
                    debug_logger.logger.info(f"Loaded Hugging Face model locally on device: {self.device}")
                    debug_logger.end_timer("load_hf_model")
                
                # Try to get the embedding dimension from the model config
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                    self.dimension = self.model.config.hidden_size
                    
            except ImportError as e:
                error_msg = f"Required packages not installed: {str(e)}"
                if self.debug:
                    debug_logger.logger.error(error_msg)
                raise ImportError("transformers and torch are not installed. "
                                "Install with: pip install transformers torch")
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                if self.debug:
                    debug_logger.logger.error(error_msg)
                raise RuntimeError(f"Failed to load Hugging Face model: {str(e)}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        import torch
        
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _get_default_dimension(self) -> int:
        """Get the default embedding dimension for fallback."""
        return self.dimension
    
    @debug_function()
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using Hugging Face models.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            if self.debug:
                debug_logger.logger.warning("Empty text list provided for embedding")
            return []
        
        # Truncate very long texts to prevent issues
        max_length = 8000  # Reasonable limit to prevent issues
        truncated_texts = [text[:max_length] if len(text) > max_length else text for text in texts]
        if any(len(text) > max_length for text in texts):
            if self.debug:
                debug_logger.logger.warning(f"Truncated {sum(1 for text in texts if len(text) > max_length)} texts to {max_length} characters")
        
        # Check cache for all texts if enabled
        if self.use_cache:
            cached_results = []
            texts_to_embed = []
            indices_to_embed = []
            
            for i, text in enumerate(truncated_texts):
                if text in self.cache:
                    cached_results.append((i, self.cache[text]))
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
            
            # If all embeddings are cached, return them
            if len(cached_results) == len(truncated_texts):
                if self.debug:
                    debug_logger.logger.debug(f"All {len(truncated_texts)} embeddings found in cache")
                # Sort by original index and return just the embeddings
                return [emb for _, emb in sorted(cached_results, key=lambda x: x[0])]
            
            # If some embeddings are cached, we'll need to merge results later
            partial_cache = len(cached_results) > 0
        else:
            texts_to_embed = truncated_texts
            partial_cache = False
        
        try:
            # Log the embedding request
            if self.debug:
                debug_logger.logger.debug(f"Generating embeddings for {len(texts_to_embed)} texts")
                debug_logger.start_timer("generate_embeddings")
            
            # Generate embeddings based on the method
            if self.use_api:
                new_embeddings = self._get_embeddings_api(texts_to_embed)
            else:
                new_embeddings = self._get_embeddings_local(texts_to_embed)
            
            if self.debug:
                debug_logger.logger.debug(f"Generated {len(new_embeddings)} embeddings with dimension {len(new_embeddings[0]) if new_embeddings else 0}")
                debug_logger.end_timer("generate_embeddings")
            
            # Cache the new embeddings if caching is enabled
            if self.use_cache:
                for i, embedding in enumerate(new_embeddings):
                    self.cache[texts_to_embed[i]] = embedding
            
            # If we had partial cache hits, merge the results
            if partial_cache:
                # Create a result array of the right size
                all_embeddings = [None] * len(truncated_texts)
                
                # Fill in cached results
                for idx, emb in cached_results:
                    all_embeddings[idx] = emb
                
                # Fill in new results
                for i, idx in enumerate(indices_to_embed):
                    all_embeddings[idx] = new_embeddings[i]
                
                return all_embeddings
            else:
                return new_embeddings
                
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            if self.debug:
                debug_logger.logger.error(error_msg)
                debug_logger.logger.debug(traceback.format_exc())
            
            # Return zero vectors as fallback
            return [[0.0] * self.dimension for _ in range(len(texts))]
    
    def _get_embeddings_api(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using the Hugging Face API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Process in batches to avoid timeouts
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Retry mechanism for API calls
            for attempt in range(self.max_retries):
                try:
                    if self.debug:
                        debug_logger.logger.debug(f"API call to HuggingFace, batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    
                    response = requests.post(
                        self.api_url, 
                        headers=headers, 
                        json={"inputs": batch_texts},
                        timeout=30  # Add timeout
                    )
                    
                    # Check if the model is still loading
                    if response.status_code == 503:
                        # Model is loading, wait and retry
                        response_json = json.loads(response.content.decode("utf-8"))
                        wait_time = response_json.get("estimated_time", 20)
                        if self.debug:
                            debug_logger.logger.warning(f"Model is loading, waiting {wait_time} seconds")
                        time.sleep(wait_time)
                        continue
                    
                    # Check for other error codes
                    if response.status_code == 401:
                        error_msg = "Invalid API key or authentication issue"
                        if self.debug:
                            debug_logger.logger.error(error_msg)
                        raise ValueError(error_msg)
                    elif response.status_code == 429:
                        error_msg = "Rate limit exceeded"
                        if self.debug:
                            debug_logger.logger.warning(error_msg)
                        # Wait longer for rate limits
                        time.sleep(5 * (attempt + 1))
                        continue
                        
                    response.raise_for_status()
                    
                    if self.debug:
                        debug_logger.logger.debug(f"API response received, status: {response.status_code}")
                    
                    batch_embeddings = response.json()
                    all_embeddings.extend(batch_embeddings)
                    break
                except requests.exceptions.Timeout:
                    error_msg = "Request timed out"
                    if self.debug:
                        debug_logger.logger.warning(error_msg)
                    if attempt == self.max_retries - 1:
                        raise TimeoutError(f"Failed to get embeddings after {self.max_retries} attempts: Request timed out")
                    time.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    error_msg = f"API call failed: {str(e)}"
                    if self.debug:
                        debug_logger.logger.warning(error_msg)
                    if attempt == self.max_retries - 1:
                        raise Exception(f"Failed to get embeddings after {self.max_retries} attempts: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return all_embeddings
    
    def _get_embeddings_local(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using the local Hugging Face model."""
        import torch
        
        # Process in batches to avoid OOM errors
        batch_size = 32
        all_embeddings = []
        
        try:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    
                    try:
                        if self.debug:
                            debug_logger.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                        
                        # Tokenize and prepare input
                        encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                                      max_length=512, return_tensors='pt')
                        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                        
                        # Get model output
                        model_output = self.model(**encoded_input)
                        
                        # Mean pooling
                        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                        
                        # Normalize embeddings
                        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                        
                        # Convert to list and add to results
                        batch_embeddings = sentence_embeddings.cpu().numpy().tolist()
                        all_embeddings.extend(batch_embeddings)
                        
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            # If we hit OOM, reduce batch size and try again
                            if self.debug:
                                debug_logger.logger.warning("CUDA out of memory, reducing batch size")
                            # Process one by one as fallback
                            for text in batch_texts:
                                try:
                                    encoded_input = self.tokenizer([text], padding=True, truncation=True, 
                                                                  max_length=512, return_tensors='pt')
                                    encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                                    model_output = self.model(**encoded_input)
                                    sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
                                    sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
                                    all_embeddings.append(sentence_embedding.cpu().numpy().tolist()[0])
                                except Exception as inner_e:
                                    # If individual processing fails, add a zero vector
                                    if self.debug:
                                        debug_logger.logger.error(f"Error processing individual text: {str(inner_e)}")
                                    all_embeddings.append([0.0] * self.dimension)
                        else:
                            # For other runtime errors, log and re-raise
                            if self.debug:
                                debug_logger.logger.error(f"Runtime error: {str(e)}")
                            raise
        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error in local embedding generation: {str(e)}")
                debug_logger.logger.debug(traceback.format_exc())
            raise
        
        return all_embeddings


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenRouter API."""
    
    def __init__(self, model: str = "openai/text-embedding-3-small", use_cache: bool = True, 
                max_retries: int = 3, debug: bool = True):
        """
        Initialize the OpenRouter embedding provider.
        
        Args:
            model: Model to use from OpenRouter
            use_cache: Whether to use caching
            max_retries: Maximum number of retries for API calls
            debug: Whether to enable debugging
        """
        super().__init__(use_cache=use_cache, debug=debug)
        self.model = model
        self.max_retries = max_retries
        
        # Set dimension based on model
        if "text-embedding-3" in model:
            self.dimension = 1536
        elif "text-embedding-ada-002" in model:
            self.dimension = 1536
        else:
            self.dimension = 1536  # Default for most OpenAI-compatible models
        
        # Check for API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            error_msg = "OpenRouter API key not found. Set the OPENROUTER_API_KEY environment variable."
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        
        if self.debug:
            debug_logger.logger.info(f"Initialized OpenRouterEmbeddingProvider with model: {model}")
    
    def _get_default_dimension(self) -> int:
        """Get the default embedding dimension for fallback."""
        return self.dimension
    
    @debug_function()
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using OpenRouter API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            if self.debug:
                debug_logger.logger.warning("Empty text list provided for embedding")
            return []
        
        # Truncate very long texts to prevent issues
        max_length = 8000  # Reasonable limit to prevent issues
        truncated_texts = [text[:max_length] if len(text) > max_length else text for text in texts]
        if any(len(text) > max_length for text in texts):
            if self.debug:
                debug_logger.logger.warning(f"Truncated {sum(1 for text in texts if len(text) > max_length)} texts to {max_length} characters")
        
        # Check cache for all texts if enabled
        if self.use_cache:
            cached_results = []
            texts_to_embed = []
            indices_to_embed = []
            
            for i, text in enumerate(truncated_texts):
                if text in self.cache:
                    cached_results.append((i, self.cache[text]))
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
            
            # If all embeddings are cached, return them
            if len(cached_results) == len(truncated_texts):
                if self.debug:
                    debug_logger.logger.debug(f"All {len(truncated_texts)} embeddings found in cache")
                # Sort by original index and return just the embeddings
                return [emb for _, emb in sorted(cached_results, key=lambda x: x[0])]
            
            # If some embeddings are cached, we'll need to merge results later
            partial_cache = len(cached_results) > 0
        else:
            texts_to_embed = truncated_texts
            partial_cache = False
        
        try:
            if self.debug:
                debug_logger.logger.debug(f"Generating embeddings for {len(texts_to_embed)} texts using OpenRouter API")
                debug_logger.start_timer("openrouter_embeddings")
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost"  # Required by OpenRouter
            }
            
            # Process in batches to avoid timeouts
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i+batch_size]
                
                # Retry mechanism for API calls
                for attempt in range(self.max_retries):
                    try:
                        if self.debug:
                            debug_logger.logger.debug(f"API call to OpenRouter, batch {i//batch_size + 1}/{(len(texts_to_embed)-1)//batch_size + 1}")
                        
                        data = {
                            "model": self.model,
                            "input": batch_texts
                        }
                        
                        response = requests.post(
                            self.api_url, 
                            headers=headers, 
                            json=data,
                            timeout=30  # Add timeout
                        )
                        
                        # Check for error codes
                        if response.status_code == 401:
                            error_msg = "Invalid API key or authentication issue"
                            if self.debug:
                                debug_logger.logger.error(error_msg)
                            raise ValueError(error_msg)
                        elif response.status_code == 429:
                            error_msg = "Rate limit exceeded"
                            if self.debug:
                                debug_logger.logger.warning(error_msg)
                            # Wait longer for rate limits
                            time.sleep(5 * (attempt + 1))
                            continue
                            
                        response.raise_for_status()
                        
                        if self.debug:
                            debug_logger.logger.debug(f"API response received, status: {response.status_code}")
                        
                        result = response.json()
                        batch_embeddings = [item["embedding"] for item in result["data"]]
                        all_embeddings.extend(batch_embeddings)
                        break
                    except requests.exceptions.Timeout:
                        error_msg = "Request timed out"
                        if self.debug:
                            debug_logger.logger.warning(error_msg)
                        if attempt == self.max_retries - 1:
                            raise TimeoutError(f"Failed to get embeddings after {self.max_retries} attempts: Request timed out")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    except Exception as e:
                        error_msg = f"API call failed: {str(e)}"
                        if self.debug:
                            debug_logger.logger.warning(error_msg)
                        if attempt == self.max_retries - 1:
                            raise Exception(f"Failed to get embeddings after {self.max_retries} attempts: {str(e)}")
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            if self.debug:
                debug_logger.logger.debug(f"Generated {len(all_embeddings)} embeddings with dimension {len(all_embeddings[0]) if all_embeddings else 0}")
                debug_logger.end_timer("openrouter_embeddings")
            
            # Cache the new embeddings if caching is enabled
            if self.use_cache:
                for i, embedding in enumerate(all_embeddings):
                    self.cache[texts_to_embed[i]] = embedding
            
            # If we had partial cache hits, merge the results
            if partial_cache:
                # Create a result array of the right size
                merged_embeddings = [None] * len(truncated_texts)
                
                # Fill in cached results
                for idx, emb in cached_results:
                    merged_embeddings[idx] = emb
                
                # Fill in new results
                for i, idx in enumerate(indices_to_embed):
                    merged_embeddings[idx] = all_embeddings[i]
                
                return merged_embeddings
            else:
                return all_embeddings
            
        except Exception as e:
            error_msg = f"Error generating embeddings with OpenRouter: {str(e)}"
            if self.debug:
                debug_logger.logger.error(error_msg)
                debug_logger.logger.debug(traceback.format_exc())
            
            # Return zero vectors as fallback
            return [[0.0] * self.dimension for _ in range(len(texts))]
