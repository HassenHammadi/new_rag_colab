"""
Script to copy necessary modules from the original RAG system.

This script should be run from the parent directory containing both
the original PDF_RAG and the new new_rag_colab directories.
"""

import os
import shutil
from pathlib import Path

def copy_module(src_path, dest_path):
    """Copy a module file if it exists."""
    if os.path.exists(src_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")
    else:
        print(f"Source file not found: {src_path}")

def main():
    # Define source and destination directories
    src_dir = Path("PDF_RAG")
    dest_dir = Path("new_rag_colab")
    
    # Ensure directories exist
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return
    
    if not dest_dir.exists():
        print(f"Destination directory not found: {dest_dir}")
        return
    
    # Copy processor modules
    copy_module(src_dir / "processors" / "pdf_processor.py", dest_dir / "processors" / "pdf_processor.py")
    copy_module(src_dir / "processors" / "text_processor.py", dest_dir / "processors" / "text_processor.py")
    copy_module(src_dir / "processors" / "json_processor.py", dest_dir / "processors" / "json_processor.py")
    copy_module(src_dir / "processors" / "csv_processor.py", dest_dir / "processors" / "csv_processor.py")
    
    # Copy chunker modules
    copy_module(src_dir / "chunkers" / "base_chunker.py", dest_dir / "chunkers" / "base_chunker.py")
    copy_module(src_dir / "chunkers" / "advanced_chunker.py", dest_dir / "chunkers" / "advanced_chunker.py")
    
    # Copy retriever modules
    copy_module(src_dir / "retrievers" / "base_retriever.py", dest_dir / "retrievers" / "base_retriever.py")
    copy_module(src_dir / "retrievers" / "advanced_retriever.py", dest_dir / "retrievers" / "advanced_retriever.py")
    
    # Copy utility modules
    copy_module(src_dir / "utils" / "embeddings.py", dest_dir / "utils" / "embeddings.py")
    copy_module(src_dir / "utils" / "llm_integration.py", dest_dir / "utils" / "llm_integration.py")
    
    print("Module copying complete!")

if __name__ == "__main__":
    main()
