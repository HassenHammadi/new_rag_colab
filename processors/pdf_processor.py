"""
PDF processor for extracting text from PDF files.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union
import os
from pathlib import Path
import tempfile

try:
    import pypdf
except ImportError:
    raise ImportError("pypdf is not installed. Install with: pip install pypdf")

from ..utils.debug_utils import debug_logger, debug_function

class PDFProcessor:
    """Processor for extracting text from PDF files."""
    
    def __init__(self, extraction_method: str = "pypdf", debug: bool = True):
        """
        Initialize the PDF processor.
        
        Args:
            extraction_method: Method to use for extraction (pypdf, unstructured, ocr)
            debug: Whether to enable debugging
        """
        self.extraction_method = extraction_method
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized PDFProcessor with extraction method: {extraction_method}")
        
        # Check if we have the required dependencies
        if extraction_method == "pypdf":
            try:
                import pypdf
            except ImportError:
                error_msg = "pypdf is not installed. Install with: pip install pypdf"
                if self.debug:
                    debug_logger.logger.error(error_msg)
                raise ImportError(error_msg)
        elif extraction_method == "unstructured":
            try:
                import unstructured
            except ImportError:
                error_msg = "unstructured is not installed. Install with: pip install unstructured"
                if self.debug:
                    debug_logger.logger.error(error_msg)
                raise ImportError(error_msg)
        elif extraction_method == "ocr":
            try:
                import pytesseract
                import pdf2image
            except ImportError:
                error_msg = "OCR dependencies not installed. Install with: pip install pytesseract pdf2image"
                if self.debug:
                    debug_logger.logger.error(error_msg)
                raise ImportError(error_msg)
        else:
            error_msg = f"Unsupported extraction method: {extraction_method}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise ValueError(error_msg)
    
    @debug_function()
    def load_pdf(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with metadata and pages
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if self.debug:
            debug_logger.logger.info(f"Loading PDF: {file_path}")
            debug_logger.start_timer(f"load_pdf_{file_path.name}")
        
        # Extract text based on the selected method
        if self.extraction_method == "pypdf":
            return self._extract_with_pypdf(file_path)
        elif self.extraction_method == "unstructured":
            return self._extract_with_unstructured(file_path)
        elif self.extraction_method == "ocr":
            return self._extract_with_ocr(file_path)
        else:
            error_msg = f"Unsupported extraction method: {self.extraction_method}"
            if self.debug:
                debug_logger.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _extract_with_pypdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using pypdf."""
        try:
            if self.debug:
                debug_logger.start_timer("pypdf_extraction")
            
            # Open the PDF file
            with open(file_path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                
                # Extract metadata
                metadata = {
                    "title": pdf.metadata.title if pdf.metadata and hasattr(pdf.metadata, "title") else None,
                    "author": pdf.metadata.author if pdf.metadata and hasattr(pdf.metadata, "author") else None,
                    "creator": pdf.metadata.creator if pdf.metadata and hasattr(pdf.metadata, "creator") else None,
                    "producer": pdf.metadata.producer if pdf.metadata and hasattr(pdf.metadata, "producer") else None,
                    "subject": pdf.metadata.subject if pdf.metadata and hasattr(pdf.metadata, "subject") else None,
                    "page_count": len(pdf.pages),
                    "source_file": file_path.name,
                    "source_type": "pdf"
                }
                
                # Extract text from each page
                pages = []
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        pages.append({
                            "page_number": i + 1,
                            "content": text,
                            "metadata": {"page_number": i + 1}
                        })
                    except Exception as e:
                        if self.debug:
                            debug_logger.logger.warning(f"Error extracting text from page {i+1}: {str(e)}")
                        pages.append({
                            "page_number": i + 1,
                            "content": f"[Error extracting text: {str(e)}]",
                            "metadata": {"page_number": i + 1, "extraction_error": str(e)}
                        })
            
            if self.debug:
                debug_logger.end_timer("pypdf_extraction")
                debug_logger.logger.info(f"Extracted {len(pages)} pages from {file_path.name}")
                debug_logger.end_timer(f"load_pdf_{file_path.name}")
            
            return {
                "metadata": metadata,
                "pages": pages
            }
        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error in pypdf extraction: {str(e)}")
            raise
    
    def _extract_with_unstructured(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using unstructured."""
        try:
            from unstructured.partition.pdf import partition_pdf
            
            if self.debug:
                debug_logger.start_timer("unstructured_extraction")
            
            # Extract elements using unstructured
            elements = partition_pdf(str(file_path))
            
            # Group elements by page
            pages_dict = {}
            for element in elements:
                page_number = getattr(element, "metadata", {}).get("page_number", 1)
                if page_number not in pages_dict:
                    pages_dict[page_number] = []
                pages_dict[page_number].append(str(element))
            
            # Create pages list
            pages = []
            for page_number, texts in sorted(pages_dict.items()):
                pages.append({
                    "page_number": page_number,
                    "content": "\n".join(texts),
                    "metadata": {"page_number": page_number}
                })
            
            # Create metadata
            metadata = {
                "page_count": len(pages),
                "source_file": file_path.name,
                "source_type": "pdf"
            }
            
            if self.debug:
                debug_logger.end_timer("unstructured_extraction")
                debug_logger.logger.info(f"Extracted {len(pages)} pages from {file_path.name} using unstructured")
                debug_logger.end_timer(f"load_pdf_{file_path.name}")
            
            return {
                "metadata": metadata,
                "pages": pages
            }
        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error in unstructured extraction: {str(e)}")
            raise
    
    def _extract_with_ocr(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using OCR."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            if self.debug:
                debug_logger.start_timer("ocr_extraction")
            
            # Convert PDF to images
            images = convert_from_path(str(file_path))
            
            # Extract text from each image
            pages = []
            for i, image in enumerate(images):
                try:
                    text = pytesseract.image_to_string(image)
                    pages.append({
                        "page_number": i + 1,
                        "content": text,
                        "metadata": {"page_number": i + 1}
                    })
                except Exception as e:
                    if self.debug:
                        debug_logger.logger.warning(f"Error extracting text from page {i+1} with OCR: {str(e)}")
                    pages.append({
                        "page_number": i + 1,
                        "content": f"[Error extracting text with OCR: {str(e)}]",
                        "metadata": {"page_number": i + 1, "extraction_error": str(e)}
                    })
            
            # Create metadata
            metadata = {
                "page_count": len(pages),
                "source_file": file_path.name,
                "source_type": "pdf"
            }
            
            if self.debug:
                debug_logger.end_timer("ocr_extraction")
                debug_logger.logger.info(f"Extracted {len(pages)} pages from {file_path.name} using OCR")
                debug_logger.end_timer(f"load_pdf_{file_path.name}")
            
            return {
                "metadata": metadata,
                "pages": pages
            }
        except Exception as e:
            if self.debug:
                debug_logger.logger.error(f"Error in OCR extraction: {str(e)}")
            raise
