"""
JSON processor for extracting text from JSON files.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union, Set
import os
import json
from pathlib import Path

from ..utils.debug_utils import debug_logger, debug_function

class JSONProcessor:
    """Processor for extracting text from JSON files."""
    
    def __init__(self, encoding: str = "utf-8", text_fields: Optional[List[str]] = None, debug: bool = True):
        """
        Initialize the JSON processor.
        
        Args:
            encoding: Text encoding to use
            text_fields: List of field names to extract (if None, extract all string values)
            debug: Whether to enable debugging
        """
        self.encoding = encoding
        self.text_fields = text_fields
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized JSONProcessor with encoding: {encoding}")
            if text_fields:
                debug_logger.logger.debug(f"Will extract specific fields: {', '.join(text_fields)}")
            else:
                debug_logger.logger.debug("Will extract all string values")
    
    @debug_function()
    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a JSON file and extract text.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with metadata and items
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if self.debug:
            debug_logger.logger.info(f"Loading JSON file: {file_path}")
            debug_logger.start_timer(f"load_json_{file_path.name}")
        
        try:
            # Read the file
            with open(file_path, "r", encoding=self.encoding) as f:
                data = json.load(f)
            
            # Create metadata
            metadata = {
                "source_file": file_path.name,
                "source_type": "json",
                "size_bytes": os.path.getsize(file_path),
                "encoding": self.encoding
            }
            
            # Extract text items
            items = []
            self._extract_text_from_json(data, items, path="$")
            
            if self.debug:
                debug_logger.logger.info(f"Loaded JSON file: {file_path.name} ({len(items)} text items extracted)")
                debug_logger.end_timer(f"load_json_{file_path.name}")
            
            return {
                "metadata": metadata,
                "items": items
            }
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON file: {str(e)}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise json.JSONDecodeError(f"Error decoding {file_path.name}: {str(e)}", e.doc, e.pos)
        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error loading JSON file: {str(e)}")
            raise
    
    def _extract_text_from_json(self, data: Any, items: List[Dict[str, Any]], path: str = "$", 
                               parent_keys: Optional[Set[str]] = None) -> None:
        """
        Recursively extract text from JSON data.
        
        Args:
            data: JSON data to extract from
            items: List to append extracted items to
            path: Current JSON path
            parent_keys: Set of parent keys for field matching
        """
        if parent_keys is None:
            parent_keys = set()
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}"
                current_keys = parent_keys.union({key})
                
                # Check if this is a text field we want to extract
                if isinstance(value, str) and (self.text_fields is None or key in self.text_fields or any(pk in self.text_fields for pk in parent_keys)):
                    items.append({
                        "content": value,
                        "metadata": {
                            "path": current_path,
                            "field": key
                        }
                    })
                
                # Recurse into nested structures
                self._extract_text_from_json(value, items, current_path, current_keys)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                
                # Check if this is a string in a list
                if isinstance(item, str) and (self.text_fields is None or any(pk in self.text_fields for pk in parent_keys)):
                    items.append({
                        "content": item,
                        "metadata": {
                            "path": current_path,
                            "field": f"item_{i}"
                        }
                    })
                
                # Recurse into nested structures
                self._extract_text_from_json(item, items, current_path, parent_keys)
