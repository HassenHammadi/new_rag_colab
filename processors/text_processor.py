"""
Text processor for extracting text from plain text files.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union
import os
from pathlib import Path

from ..utils.debug_utils import debug_logger, debug_function

class TextProcessor:
    """Processor for extracting text from plain text files."""
    
    def __init__(self, encoding: str = "utf-8", debug: bool = True):
        """
        Initialize the text processor.
        
        Args:
            encoding: Text encoding to use
            debug: Whether to enable debugging
        """
        self.encoding = encoding
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized TextProcessor with encoding: {encoding}")
    
    @debug_function()
    def load_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a text file and extract text.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with metadata and content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if self.debug:
            debug_logger.logger.info(f"Loading text file: {file_path}")
            debug_logger.start_timer(f"load_text_{file_path.name}")
        
        try:
            # Read the file
            with open(file_path, "r", encoding=self.encoding) as f:
                content = f.read()
            
            # Create metadata
            metadata = {
                "source_file": file_path.name,
                "source_type": "text",
                "extension": file_path.suffix.lower(),
                "size_bytes": os.path.getsize(file_path),
                "encoding": self.encoding
            }
            
            if self.debug:
                debug_logger.logger.info(f"Loaded text file: {file_path.name} ({len(content)} characters)")
                debug_logger.end_timer(f"load_text_{file_path.name}")
            
            return {
                "metadata": metadata,
                "content": content
            }
        except UnicodeDecodeError as e:
            error_msg = f"Error decoding file with encoding {self.encoding}: {str(e)}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise UnicodeDecodeError(f"Error decoding {file_path.name}: {str(e)}. Try a different encoding.")
        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error loading text file: {str(e)}")
            raise
