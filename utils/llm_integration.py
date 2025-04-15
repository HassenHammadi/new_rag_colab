"""
LLM integration module for enhancing RAG responses with structured formatting.
Adapted for Google Colab environment.
"""

import os
import json
import requests
import time
import traceback
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

from .debug_utils import debug_logger, debug_function

# Load environment variables
load_dotenv()

class LLMFormatter:
    """Base class for LLM-based response formatting."""
    
    def format_response(self, query: str, context: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format RAG results into a structured response.
        
        Args:
            query: The original query
            context: List of retrieved documents with content and metadata
            **kwargs: Additional arguments
            
        Returns:
            Formatted response as a string
        """
        raise NotImplementedError("Subclasses must implement format_response")


class OpenRouterLLM(LLMFormatter):
    """LLM formatter using OpenRouter API with enhanced error handling."""
    
    def __init__(self, model: str = "anthropic/claude-3-haiku", max_retries: int = 3, debug: bool = True):
        """
        Initialize the OpenRouter LLM formatter.
        
        Args:
            model: Model to use from OpenRouter
            max_retries: Maximum number of retries for API calls
            debug: Whether to enable debugging
        """
        self.model = model
        self.max_retries = max_retries
        self.debug = debug
        
        # Check for API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            error_msg = "OpenRouter API key not found. Set the OPENROUTER_API_KEY environment variable."
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if self.debug:
            debug_logger.logger.info(f"Initialized OpenRouterLLM with model: {model}")
    
    @debug_function()
    def format_response(self, query: str, context: List[Dict[str, Any]], 
                       response_format: str = "markdown", **kwargs) -> str:
        """
        Format RAG results into a structured response using OpenRouter.
        
        Args:
            query: The original query
            context: List of retrieved documents with content and metadata
            response_format: Format for the response (markdown, json, etc.)
            **kwargs: Additional arguments
            
        Returns:
            Formatted response as a string
        """
        if not context:
            if self.debug:
                debug_logger.logger.warning("Empty context provided for formatting")
            return f"```markdown\n# No Information Available\n\nI don't have any relevant information to answer your question about: {query}\n```"
        
        try:
            # Extract content and metadata from context
            context_texts = []
            for i, doc in enumerate(context):
                source_info = f"Source: {doc.get('metadata', {}).get('source_file', 'unknown')}"
                if 'source_type' in doc.get('metadata', {}):
                    source_info += f" (Type: {doc['metadata']['source_type']})"
                if 'page_number' in doc.get('metadata', {}):
                    source_info += f", Page: {doc['metadata']['page_number']}"
                elif 'row_number' in doc.get('metadata', {}):
                    source_info += f", Row: {doc['metadata']['row_number']}"
                
                # Add score if available
                score_info = ""
                if 'score' in doc:
                    score_info = f" [Relevance: {doc['score']:.2f}]"
                
                # Format the context entry and truncate long content
                content = doc.get('content', '')[:800]
                if len(doc.get('content', '')) > 800:
                    content += "..."
                context_text = f"[{i+1}] {source_info}{score_info}\n{content}"
                context_texts.append(context_text)
            
            # Join all context texts
            all_context = "\n\n---\n\n".join(context_texts)
            
            # Create the system prompt
            system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Follow these guidelines:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
3. Format your response in {response_format} format
4. Use headings, bullet points, and other formatting to make the answer clear and readable
5. Cite the sources used in your answer (they are numbered in the context)
6. Be concise but comprehensive"""
            
            # Create the user prompt
            user_prompt = f"""Question: {query}

Context:
{all_context}

Please provide a well-structured answer in {response_format} format, citing the relevant sources."""
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost"  # Required by OpenRouter
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            # Add optional parameters
            if "temperature" in kwargs:
                data["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                data["max_tokens"] = kwargs["max_tokens"]
            
            # Log the API call (without sensitive data)
            if self.debug:
                debug_logger.logger.debug(f"Calling OpenRouter API with model: {self.model}")
            
            # Make the API call with retries
            response = None
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        self.api_url, 
                        headers=headers, 
                        json=data,
                        timeout=60  # Longer timeout for LLM responses
                    )
                    
                    # Check for specific error codes
                    if response.status_code == 401:
                        error_msg = "Invalid API key or authentication issue"
                        if self.debug:
                            debug_logger.logger.error(error_msg)
                        return f"```markdown\n# Authentication Error\n\n{error_msg}\n```"
                    elif response.status_code == 429:
                        error_msg = "Rate limit exceeded"
                        if self.debug:
                            debug_logger.logger.warning(error_msg)
                        # Wait longer for rate limits
                        if attempt < self.max_retries - 1:
                            wait_time = 5 * (attempt + 1)
                            if self.debug:
                                debug_logger.logger.warning(f"Rate limited, waiting {wait_time} seconds before retry")
                            time.sleep(wait_time)
                            continue
                    
                    response.raise_for_status()
                    
                    # Parse the response
                    result = response.json()
                    formatted_answer = result["choices"][0]["message"]["content"]
                    
                    if self.debug:
                        debug_logger.logger.info("Successfully received response from OpenRouter API")
                    
                    return formatted_answer
                    
                except requests.exceptions.Timeout:
                    error_msg = "Request timed out"
                    if self.debug:
                        debug_logger.logger.warning(error_msg)
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return f"```markdown\n# Response Timeout\n\nThe request to generate a response timed out. Please try again later.\n```"
                except Exception as e:
                    error_msg = f"API call failed: {str(e)}"
                    if self.debug:
                        debug_logger.logger.warning(error_msg)
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        # If all retries failed, return a formatted error message
                        response_text = response.text if response else "No response"
                        return f"```markdown\n# Error Generating Response\n\n{error_msg}\nResponse: {response_text}\n```"
            
            # If we get here, all retries failed
            return f"```markdown\n# Error Generating Response\n\nFailed to generate a response after {self.max_retries} attempts.\n```"
            
        except Exception as e:
            # Catch any other exceptions in the formatting process
            if self.debug:
                debug_logger.logger.error(f"Error in format_response: {str(e)}")
                debug_logger.logger.debug(traceback.format_exc())
            return f"```markdown\n# Error Generating Response\n\nAn unexpected error occurred: {str(e)}\n```"


class MarkdownFormatter:
    """Utility class for formatting RAG results as Markdown without using an LLM."""
    
    @staticmethod
    def format_response(query: str, context: List[Dict[str, Any]]) -> str:
        """
        Format RAG results into a Markdown response without using an LLM.
        
        Args:
            query: The original query
            context: List of retrieved documents with content and metadata
            
        Returns:
            Markdown formatted response
        """
        if not context:
            return "```markdown\n# No results found\n\nNo relevant information was found for your query.\n```"
        
        # Start building the markdown response
        markdown = f"```markdown\n# Results for: {query}\n\n"
        
        # Add each context item
        for i, doc in enumerate(context):
            # Extract metadata
            metadata = doc.get('metadata', {})
            source_file = metadata.get('source_file', 'Unknown source')
            source_type = metadata.get('source_type', 'unknown')
            
            # Format based on source type
            if source_type == 'pdf':
                page_number = metadata.get('page_number', 'N/A')
                source_info = f"PDF: {source_file}, Page: {page_number}"
            elif source_type == 'csv':
                row_number = metadata.get('row_number', 'N/A')
                source_info = f"CSV: {source_file}, Row: {row_number}"
            elif source_type == 'json':
                json_path = metadata.get('json_path', '')
                source_info = f"JSON: {source_file}, Path: {json_path}"
            else:
                source_info = f"{source_type.upper()}: {source_file}"
            
            # Add score if available
            score = doc.get('score', None)
            score_text = f" (Relevance: {score:.2f})" if score is not None else ""
            
            # Add the section
            markdown += f"## Result {i+1}{score_text}\n\n"
            markdown += f"**Source:** {source_info}\n\n"
            
            # Add content with some formatting
            content = doc.get('content', 'No content available')
            # Limit content length and add ellipsis if needed
            if len(content) > 800:
                content = content[:800] + "..."
            
            markdown += f"{content}\n\n"
            
            # Add separator except for the last item
            if i < len(context) - 1:
                markdown += "---\n\n"
        
        markdown += "```"
        return markdown
