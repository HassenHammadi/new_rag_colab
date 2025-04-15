"""
CSV processor for extracting text from CSV files.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union
import os
import csv
from pathlib import Path

from ..utils.debug_utils import debug_logger, debug_function

class CSVProcessor:
    """Processor for extracting text from CSV files."""
    
    def __init__(self, delimiter: str = ",", encoding: str = "utf-8", 
                text_columns: Optional[List[str]] = None, debug: bool = True):
        """
        Initialize the CSV processor.
        
        Args:
            delimiter: CSV delimiter character
            encoding: Text encoding to use
            text_columns: List of column names to extract (if None, extract all columns)
            debug: Whether to enable debugging
        """
        self.delimiter = delimiter
        self.encoding = encoding
        self.text_columns = text_columns
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized CSVProcessor with delimiter: '{delimiter}', encoding: {encoding}")
            if text_columns:
                debug_logger.logger.debug(f"Will extract specific columns: {', '.join(text_columns)}")
            else:
                debug_logger.logger.debug("Will extract all columns")
    
    @debug_function()
    def load_csv(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a CSV file and extract text.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with metadata and rows
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if self.debug:
            debug_logger.logger.info(f"Loading CSV file: {file_path}")
            debug_logger.start_timer(f"load_csv_{file_path.name}")
        
        try:
            # Read the file
            with open(file_path, "r", encoding=self.encoding, newline='') as f:
                csv_reader = csv.reader(f, delimiter=self.delimiter)
                
                # Read headers
                headers = next(csv_reader, [])
                
                # Read rows
                rows = []
                for row_idx, row in enumerate(csv_reader):
                    # Create a dictionary for the row
                    row_dict = {}
                    for i, value in enumerate(row):
                        if i < len(headers):
                            row_dict[headers[i]] = value
                        else:
                            row_dict[f"column_{i}"] = value
                    
                    # Extract text from columns
                    text_parts = []
                    for header, value in row_dict.items():
                        if self.text_columns is None or header in self.text_columns:
                            if value:  # Only add non-empty values
                                text_parts.append(f"{header}: {value}")
                    
                    # Create row item
                    rows.append({
                        "content": "\n".join(text_parts),
                        "metadata": {
                            "row_number": row_idx + 2,  # +2 because 1-indexed and header is row 1
                            "row_data": row_dict
                        }
                    })
            
            # Create metadata
            metadata = {
                "source_file": file_path.name,
                "source_type": "csv",
                "size_bytes": os.path.getsize(file_path),
                "encoding": self.encoding,
                "delimiter": self.delimiter,
                "headers": headers,
                "row_count": len(rows) + 1  # +1 for header row
            }
            
            if self.debug:
                debug_logger.logger.info(f"Loaded CSV file: {file_path.name} ({len(rows)} rows, {len(headers)} columns)")
                debug_logger.end_timer(f"load_csv_{file_path.name}")
            
            return {
                "metadata": metadata,
                "rows": rows
            }
        except UnicodeDecodeError as e:
            error_msg = f"Error decoding CSV file with encoding {self.encoding}: {str(e)}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise UnicodeDecodeError(f"Error decoding {file_path.name}: {str(e)}. Try a different encoding.")
        except csv.Error as e:
            error_msg = f"CSV parsing error: {str(e)}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise csv.Error(f"Error parsing {file_path.name}: {str(e)}. Check the delimiter or file format.")
        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error loading CSV file: {str(e)}")
            raise
