"""
Base chunker implementations for splitting documents into chunks.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union
import re

from ..utils.debug_utils import debug_logger, debug_function

class BaseChunker:
    """Base class for document chunkers."""
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of chunks
        """
        raise NotImplementedError("Subclasses must implement chunk_document")


class FixedSizeChunker(BaseChunker):
    """Chunker that splits text into fixed-size chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, debug: bool = True):
        """
        Initialize the fixed-size chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            debug: Whether to enable debugging
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug = debug
        
        if chunk_overlap >= chunk_size:
            error_msg = f"Chunk overlap ({chunk_overlap}) must be less than chunk size ({chunk_size})"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.debug:
            debug_logger.logger.info(f"Initialized FixedSizeChunker with size: {chunk_size}, overlap: {chunk_overlap}")
    
    @debug_function()
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into fixed-size chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of chunks
        """
        if self.debug:
            debug_logger.logger.info("Chunking document")
            debug_logger.start_timer("chunk_document")
        
        chunks = []
        
        # Check if the document has pages (PDF-like)
        if "pages" in document:
            for page in document["pages"]:
                page_chunks = self._chunk_text(
                    text=page["content"],
                    metadata={
                        **document["metadata"],
                        **page["metadata"]
                    }
                )
                chunks.extend(page_chunks)
        # Check if the document has direct content (text-like)
        elif "content" in document:
            chunks = self._chunk_text(
                text=document["content"],
                metadata=document["metadata"]
            )
        else:
            error_msg = "Document format not supported: missing 'pages' or 'content'"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.debug:
            debug_logger.logger.info(f"Created {len(chunks)} chunks")
            debug_logger.end_timer("chunk_document")
        
        return chunks
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into fixed-size chunks.
        
        Args:
            text: Text to split
            metadata: Metadata to include with each chunk
            
        Returns:
            List of chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Adjust end to avoid splitting in the middle of a word or sentence
            if end < len(text):
                # Try to find a sentence boundary
                sentence_end = text.rfind(". ", start, end)
                if sentence_end != -1 and sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 2  # Include the period and space
                else:
                    # Try to find a word boundary
                    word_end = text.rfind(" ", start, end)
                    if word_end != -1 and word_end > start + self.chunk_size // 2:
                        end = word_end + 1  # Include the space
            
            # Create the chunk
            chunk_text = text[start:min(end, len(text))]
            chunks.append({
                "content": chunk_text,
                "metadata": metadata.copy()
            })
            
            # Move to the next chunk
            start = end - self.chunk_overlap
            
            # Avoid getting stuck
            if start >= len(text) or start <= 0:
                break
        
        return chunks


class SemanticChunker(BaseChunker):
    """Chunker that splits text based on semantic boundaries."""
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 2000, debug: bool = True):
        """
        Initialize the semantic chunker.
        
        Args:
            min_chunk_size: Minimum size of each chunk in characters
            max_chunk_size: Maximum size of each chunk in characters
            debug: Whether to enable debugging
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized SemanticChunker with min_size: {min_chunk_size}, max_size: {max_chunk_size}")
    
    @debug_function()
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into semantic chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of chunks
        """
        if self.debug:
            debug_logger.logger.info("Chunking document semantically")
            debug_logger.start_timer("chunk_document_semantic")
        
        chunks = []
        
        # Check if the document has pages (PDF-like)
        if "pages" in document:
            for page in document["pages"]:
                page_chunks = self._chunk_text_semantically(
                    text=page["content"],
                    metadata={
                        **document["metadata"],
                        **page["metadata"]
                    }
                )
                chunks.extend(page_chunks)
        # Check if the document has direct content (text-like)
        elif "content" in document:
            chunks = self._chunk_text_semantically(
                text=document["content"],
                metadata=document["metadata"]
            )
        else:
            error_msg = "Document format not supported: missing 'pages' or 'content'"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.debug:
            debug_logger.logger.info(f"Created {len(chunks)} semantic chunks")
            debug_logger.end_timer("chunk_document_semantic")
        
        return chunks
    
    def _chunk_text_semantically(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Text to split
            metadata: Metadata to include with each chunk
            
        Returns:
            List of chunks
        """
        if not text:
            return []
        
        # Split by sections (headers)
        section_pattern = r'(?:\n|^)#+\s+(.+?)(?=\n|$)'
        sections = re.split(section_pattern, text)
        
        # If no sections found, fall back to paragraphs
        if len(sections) <= 1:
            paragraphs = re.split(r'\n\s*\n', text)
            
            # Group paragraphs into chunks
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If adding this paragraph would exceed max_chunk_size, create a new chunk
                if len(current_chunk) + len(paragraph) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                    chunks.append({
                        "content": current_chunk,
                        "metadata": metadata.copy()
                    })
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += paragraph
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append({
                    "content": current_chunk,
                    "metadata": metadata.copy()
                })
        else:
            # Process sections
            chunks = []
            current_section = ""
            current_title = ""
            
            # Process each section
            for i in range(0, len(sections), 2):
                if i + 1 < len(sections):
                    title = sections[i]
                    content = sections[i + 1]
                    
                    # Create a new chunk for this section
                    section_text = f"# {title}\n\n{content}"
                    
                    # If this section is too large, split it further
                    if len(section_text) > self.max_chunk_size:
                        # Split by paragraphs
                        paragraphs = re.split(r'\n\s*\n', content)
                        
                        # Create the first chunk with the title
                        current_chunk = f"# {title}"
                        
                        for paragraph in paragraphs:
                            paragraph = paragraph.strip()
                            if not paragraph:
                                continue
                            
                            # If adding this paragraph would exceed max_chunk_size, create a new chunk
                            if len(current_chunk) + len(paragraph) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                                chunks.append({
                                    "content": current_chunk,
                                    "metadata": {**metadata.copy(), "section_title": title}
                                })
                                current_chunk = f"# {title} (continued)\n\n{paragraph}"
                            else:
                                if current_chunk and not current_chunk.endswith("\n"):
                                    current_chunk += "\n\n"
                                current_chunk += paragraph
                        
                        # Add the last chunk if it's not empty
                        if current_chunk:
                            chunks.append({
                                "content": current_chunk,
                                "metadata": {**metadata.copy(), "section_title": title}
                            })
                    else:
                        # Add the section as a single chunk
                        chunks.append({
                            "content": section_text,
                            "metadata": {**metadata.copy(), "section_title": title}
                        })
        
        return chunks
