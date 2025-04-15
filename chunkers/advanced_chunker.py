"""
Advanced chunker implementations for splitting documents into chunks.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union
import re

from .base_chunker import BaseChunker
from ..utils.debug_utils import debug_logger, debug_function

class ContextualHeaderChunker(BaseChunker):
    """Chunker that adds contextual headers to chunks."""
    
    def __init__(self, base_chunker: BaseChunker, debug: bool = True):
        """
        Initialize the contextual header chunker.
        
        Args:
            base_chunker: Base chunker to use for initial chunking
            debug: Whether to enable debugging
        """
        self.base_chunker = base_chunker
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized ContextualHeaderChunker with base chunker: {type(base_chunker).__name__}")
    
    @debug_function()
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks with contextual headers.
        
        Args:
            document: Document to split
            
        Returns:
            List of chunks with contextual headers
        """
        if self.debug:
            debug_logger.logger.info("Chunking document with contextual headers")
            debug_logger.start_timer("chunk_document_contextual")
        
        # First, get the base chunks
        base_chunks = self.base_chunker.chunk_document(document)
        
        # Check if the document has pages (PDF-like)
        if "pages" in document:
            # Extract headers from each page
            enhanced_chunks = []
            
            for page in document["pages"]:
                # Extract headers from the page
                headers = self._extract_headers(page["content"])
                
                # Find chunks from this page
                page_number = page["metadata"].get("page_number")
                page_chunks = [chunk for chunk in base_chunks if chunk["metadata"].get("page_number") == page_number]
                
                # Add contextual headers to each chunk
                for chunk in page_chunks:
                    # Find relevant headers for this chunk
                    chunk_start = page["content"].find(chunk["content"][:100])
                    relevant_headers = self._find_relevant_headers(headers, chunk_start)
                    
                    # Add headers to the chunk content
                    if relevant_headers:
                        header_text = " > ".join(relevant_headers)
                        enhanced_content = f"CONTEXT: {header_text}\n\nCONTENT:\n{chunk['content']}"
                        chunk["content"] = enhanced_content
                        chunk["metadata"]["headers"] = relevant_headers
                    
                    enhanced_chunks.append(chunk)
            
            if self.debug:
                debug_logger.logger.info(f"Enhanced {len(enhanced_chunks)} chunks with contextual headers")
                debug_logger.end_timer("chunk_document_contextual")
            
            return enhanced_chunks
        
        # For non-PDF documents, extract headers from the entire content
        elif "content" in document:
            # Extract headers from the content
            content = document["content"]
            headers = self._extract_headers(content)
            
            # Add contextual headers to each chunk
            enhanced_chunks = []
            
            for chunk in base_chunks:
                # Find the position of this chunk in the original content
                chunk_start = content.find(chunk["content"][:100])
                
                # Find relevant headers for this chunk
                relevant_headers = self._find_relevant_headers(headers, chunk_start)
                
                # Add headers to the chunk content
                if relevant_headers:
                    header_text = " > ".join(relevant_headers)
                    enhanced_content = f"CONTEXT: {header_text}\n\nCONTENT:\n{chunk['content']}"
                    chunk["content"] = enhanced_content
                    chunk["metadata"]["headers"] = relevant_headers
                
                enhanced_chunks.append(chunk)
            
            if self.debug:
                debug_logger.logger.info(f"Enhanced {len(enhanced_chunks)} chunks with contextual headers")
                debug_logger.end_timer("chunk_document_contextual")
            
            return enhanced_chunks
        
        # If the document format is not supported, return the base chunks
        else:
            if self.debug:
                debug_logger.logger.warning("Document format not supported for contextual headers, returning base chunks")
                debug_logger.end_timer("chunk_document_contextual")
            
            return base_chunks
    
    def _extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract headers from text.
        
        Args:
            text: Text to extract headers from
            
        Returns:
            List of headers with text and position
        """
        headers = []
        
        # Match Markdown-style headers
        markdown_pattern = r'(^|\n)(#{1,6})\s+(.+?)(?=\n|$)'
        for match in re.finditer(markdown_pattern, text):
            level = len(match.group(2))
            header_text = match.group(3).strip()
            position = match.start()
            
            headers.append({
                "text": header_text,
                "level": level,
                "position": position
            })
        
        # Match underlined headers (=== or ---)
        underline_pattern = r'(^|\n)(.+?)\n([=\-]{3,})(?=\n|$)'
        for match in re.finditer(underline_pattern, text):
            header_text = match.group(2).strip()
            level = 1 if match.group(3)[0] == '=' else 2
            position = match.start()
            
            headers.append({
                "text": header_text,
                "level": level,
                "position": position
            })
        
        # Sort headers by position
        headers.sort(key=lambda h: h["position"])
        
        return headers
    
    def _find_relevant_headers(self, headers: List[Dict[str, Any]], position: int) -> List[str]:
        """
        Find headers relevant to a position in the text.
        
        Args:
            headers: List of headers
            position: Position in the text
            
        Returns:
            List of relevant header texts
        """
        relevant_headers = []
        current_levels = {}
        
        for header in headers:
            # If the header is after the position, stop
            if header["position"] > position:
                break
            
            # Update the current header for this level
            current_levels[header["level"]] = header["text"]
            
            # Remove headers with higher levels
            for level in list(current_levels.keys()):
                if level > header["level"]:
                    del current_levels[level]
        
        # Build the list of relevant headers from lowest to highest level
        for level in sorted(current_levels.keys()):
            relevant_headers.append(current_levels[level])
        
        return relevant_headers


class WindowEnrichmentChunker(BaseChunker):
    """Chunker that adds a window of surrounding context to chunks."""
    
    def __init__(self, base_chunker: BaseChunker, window_size: int = 200, debug: bool = True):
        """
        Initialize the window enrichment chunker.
        
        Args:
            base_chunker: Base chunker to use for initial chunking
            window_size: Size of the context window in characters
            debug: Whether to enable debugging
        """
        self.base_chunker = base_chunker
        self.window_size = window_size
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized WindowEnrichmentChunker with window size: {window_size}")
    
    @debug_function()
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks with window enrichment.
        
        Args:
            document: Document to split
            
        Returns:
            List of chunks with window enrichment
        """
        if self.debug:
            debug_logger.logger.info("Chunking document with window enrichment")
            debug_logger.start_timer("chunk_document_window")
        
        # First, get the base chunks
        base_chunks = self.base_chunker.chunk_document(document)
        
        # Check if the document has pages (PDF-like)
        if "pages" in document:
            # Add window enrichment to each chunk
            enriched_chunks = []
            
            for page in document["pages"]:
                # Get the page content
                text = page["content"]
                
                # Find chunks from this page
                page_number = page["metadata"].get("page_number")
                page_chunks = [chunk for chunk in base_chunks if chunk["metadata"].get("page_number") == page_number]
                
                # Add window enrichment to each chunk
                for chunk in page_chunks:
                    # Find the position of this chunk in the page content
                    chunk_content = chunk["content"]
                    if "CONTENT:\n" in chunk_content:
                        # Extract the actual content if it has been enhanced by another chunker
                        chunk_content = chunk_content.split("CONTENT:\n", 1)[1]
                    
                    chunk_start = text.find(chunk_content[:100])
                    
                    if chunk_start != -1:
                        # Calculate window boundaries
                        window_start = max(0, chunk_start - self.window_size)
                        chunk_end = chunk_start + len(chunk_content)
                        window_end = min(len(text), chunk_end + self.window_size)
                        
                        # Extract context windows
                        context_before = text[window_start:chunk_start]
                        context_after = text[chunk_end:window_end]
                        
                        # Add window enrichment to the chunk content
                        if "CONTEXT:" in chunk["content"]:
                            # Preserve existing context
                            parts = chunk["content"].split("\n\nCONTENT:\n", 1)
                            header = parts[0]
                            content = parts[1]
                            
                            enriched_content = f"{header}\n\nCONTEXT BEFORE:\n{context_before}\n\nCONTENT:\n{content}\n\nCONTEXT AFTER:\n{context_after}"
                        else:
                            enriched_content = f"CONTEXT BEFORE:\n{context_before}\n\nMAIN CONTENT:\n{chunk['content']}\n\nCONTEXT AFTER:\n{context_after}"
                        
                        chunk["content"] = enriched_content
                        chunk["metadata"]["window_enriched"] = True
                    
                    enriched_chunks.append(chunk)
            
            if self.debug:
                debug_logger.logger.info(f"Enriched {len(enriched_chunks)} chunks with window context")
                debug_logger.end_timer("chunk_document_window")
            
            return enriched_chunks
        
        # For non-PDF documents, add window enrichment to the entire content
        elif "content" in document:
            # Get the document content
            text = document["content"]
            
            # Add window enrichment to each chunk
            enriched_chunks = []
            
            for chunk in base_chunks:
                # Find the position of this chunk in the content
                chunk_content = chunk["content"]
                if "CONTENT:\n" in chunk_content:
                    # Extract the actual content if it has been enhanced by another chunker
                    chunk_content = chunk_content.split("CONTENT:\n", 1)[1]
                
                chunk_start = text.find(chunk_content[:100])
                
                if chunk_start != -1:
                    # Calculate window boundaries
                    window_start = max(0, chunk_start - self.window_size)
                    chunk_end = chunk_start + len(chunk_content)
                    window_end = min(len(text), chunk_end + self.window_size)
                    
                    # Extract context windows
                    context_before = text[window_start:chunk_start]
                    context_after = text[chunk_end:window_end]
                    
                    # Add window enrichment to the chunk content
                    if "CONTEXT:" in chunk["content"]:
                        # Preserve existing context
                        parts = chunk["content"].split("\n\nCONTENT:\n", 1)
                        header = parts[0]
                        content = parts[1]
                        
                        enriched_content = f"{header}\n\nCONTEXT BEFORE:\n{context_before}\n\nCONTENT:\n{content}\n\nCONTEXT AFTER:\n{context_after}"
                    else:
                        enriched_content = f"CONTEXT BEFORE:\n{context_before}\n\nMAIN CONTENT:\n{chunk['content']}\n\nCONTEXT AFTER:\n{context_after}"
                    
                    chunk["content"] = enriched_content
                    chunk["metadata"]["window_enriched"] = True
                
                enriched_chunks.append(chunk)
            
            if self.debug:
                debug_logger.logger.info(f"Enriched {len(enriched_chunks)} chunks with window context")
                debug_logger.end_timer("chunk_document_window")
            
            return enriched_chunks
        
        # If the document format is not supported, return the base chunks
        else:
            if self.debug:
                debug_logger.logger.warning("Document format not supported for window enrichment, returning base chunks")
                debug_logger.end_timer("chunk_document_window")
            
            return base_chunks
