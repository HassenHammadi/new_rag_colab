"""
Debug utilities for the RAG system in Google Colab.
"""

import logging
import time
import traceback
import sys
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import inspect

class DebugLogger:
    """Debug logger for the RAG system."""
    
    def __init__(self, name: str = "RAG_Colab", level: int = logging.INFO, 
                 log_to_file: bool = False, log_file: Optional[str] = None):
        """
        Initialize the debug logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_to_file: Whether to log to a file
            log_file: Path to log file (if log_to_file is True)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add console handler to logger
        self.logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file:
            if log_file is None:
                log_file = f"{name.lower()}_debug.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Performance tracking
        self.timers = {}
    
    def start_timer(self, operation: str) -> None:
        """Start a timer for performance tracking."""
        self.timers[operation] = time.time()
        self.logger.debug(f"Started operation: {operation}")
    
    def end_timer(self, operation: str) -> Optional[float]:
        """
        End a timer and log the duration.
        
        Returns:
            Duration in seconds or None if timer not found
        """
        if operation in self.timers:
            duration = time.time() - self.timers[operation]
            self.logger.info(f"Operation '{operation}' completed in {duration:.4f} seconds")
            del self.timers[operation]
            return duration
        else:
            self.logger.warning(f"No timer found for operation: {operation}")
            return None
    
    def log_function_call(self, func_name: str, args: tuple, kwargs: dict) -> None:
        """Log a function call with its arguments."""
        args_str = ", ".join([repr(arg) for arg in args])
        kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        self.logger.debug(f"Function call: {func_name}({all_args})")
    
    def log_object_state(self, obj: Any, name: str = "object") -> None:
        """Log the state of an object."""
        try:
            # Get object attributes
            attributes = {attr: getattr(obj, attr) for attr in dir(obj) 
                         if not attr.startswith('_') and not callable(getattr(obj, attr))}
            
            # Filter out large attributes and convert to string representation
            filtered_attributes = {}
            for attr, value in attributes.items():
                try:
                    # Skip large objects
                    if isinstance(value, (list, dict)) and len(value) > 100:
                        filtered_attributes[attr] = f"{type(value).__name__} with {len(value)} items"
                    else:
                        filtered_attributes[attr] = repr(value)
                except:
                    filtered_attributes[attr] = "Error getting representation"
            
            self.logger.debug(f"{name} state: {json.dumps(filtered_attributes, indent=2)}")
        except Exception as e:
            self.logger.warning(f"Error logging object state: {str(e)}")
    
    def log_exception(self, e: Exception, context: str = "") -> None:
        """Log an exception with stack trace."""
        if context:
            self.logger.error(f"Exception in {context}: {str(e)}")
        else:
            self.logger.error(f"Exception: {str(e)}")
        self.logger.error(traceback.format_exc())


# Create a function decorator for debugging
def debug_function(logger: Optional[DebugLogger] = None, log_args: bool = True, 
                  log_result: bool = True, time_execution: bool = True):
    """
    Decorator for debugging functions.
    
    Args:
        logger: Debug logger to use
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        time_execution: Whether to time function execution
    
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create logger if not provided
            nonlocal logger
            if logger is None:
                logger = DebugLogger()
            
            # Get function name
            func_name = func.__qualname__
            
            # Log function call
            if log_args:
                logger.log_function_call(func_name, args, kwargs)
            
            # Start timer
            if time_execution:
                logger.start_timer(func_name)
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # End timer
                if time_execution:
                    logger.end_timer(func_name)
                
                # Log result
                if log_result:
                    if isinstance(result, (list, dict)) and len(result) > 100:
                        logger.logger.debug(f"{func_name} returned {type(result).__name__} with {len(result)} items")
                    else:
                        logger.logger.debug(f"{func_name} returned: {repr(result)}")
                
                return result
            except Exception as e:
                # Log exception
                logger.log_exception(e, func_name)
                raise
        
        return wrapper
    
    return decorator


class DebugInspector:
    """Utility for inspecting and debugging objects."""
    
    @staticmethod
    def inspect_object(obj: Any) -> Dict[str, Any]:
        """
        Inspect an object and return its attributes and methods.
        
        Args:
            obj: Object to inspect
            
        Returns:
            Dictionary with object information
        """
        # Get object type
        obj_type = type(obj).__name__
        
        # Get attributes
        attributes = {}
        for attr in dir(obj):
            if not attr.startswith('_'):
                try:
                    value = getattr(obj, attr)
                    if not callable(value):
                        if isinstance(value, (list, dict)) and len(value) > 100:
                            attributes[attr] = f"{type(value).__name__} with {len(value)} items"
                        else:
                            attributes[attr] = repr(value)
                except:
                    attributes[attr] = "Error getting value"
        
        # Get methods
        methods = []
        for attr in dir(obj):
            if not attr.startswith('_'):
                try:
                    value = getattr(obj, attr)
                    if callable(value):
                        try:
                            signature = str(inspect.signature(value))
                            methods.append(f"{attr}{signature}")
                        except:
                            methods.append(attr)
                except:
                    pass
        
        return {
            "type": obj_type,
            "attributes": attributes,
            "methods": methods
        }
    
    @staticmethod
    def print_object_info(obj: Any, name: str = "object") -> None:
        """
        Print information about an object.
        
        Args:
            obj: Object to inspect
            name: Name to use for the object
        """
        info = DebugInspector.inspect_object(obj)
        
        print(f"=== {name} ({info['type']}) ===")
        
        print("\nAttributes:")
        for attr, value in info['attributes'].items():
            print(f"  {attr}: {value}")
        
        print("\nMethods:")
        for method in info['methods']:
            print(f"  {method}")
    
    @staticmethod
    def inspect_vector_store(vector_store: Any) -> Dict[str, Any]:
        """
        Inspect a vector store and return information about it.
        
        Args:
            vector_store: Vector store to inspect
            
        Returns:
            Dictionary with vector store information
        """
        info = {
            "type": type(vector_store).__name__,
            "document_count": 0,
            "dimension": 0,
            "metadata_fields": set(),
            "sample_documents": []
        }
        
        try:
            # Get document count
            if hasattr(vector_store, 'documents'):
                info["document_count"] = len(vector_store.documents)
                
                # Get dimension
                if hasattr(vector_store, 'dimension'):
                    info["dimension"] = vector_store.dimension
                
                # Get metadata fields
                if info["document_count"] > 0:
                    for doc in vector_store.documents[:min(10, info["document_count"])]:
                        if "metadata" in doc:
                            for field in doc["metadata"].keys():
                                info["metadata_fields"].add(field)
                    
                    # Convert set to list for JSON serialization
                    info["metadata_fields"] = list(info["metadata_fields"])
                    
                    # Get sample documents
                    info["sample_documents"] = [
                        {
                            "id": doc.get("id", "unknown"),
                            "metadata": doc.get("metadata", {}),
                            "content_preview": doc.get("content", "")[:100] + "..." if len(doc.get("content", "")) > 100 else doc.get("content", "")
                        }
                        for doc in vector_store.documents[:min(5, info["document_count"])]
                    ]
        except Exception as e:
            info["error"] = str(e)
        
        return info
    
    @staticmethod
    def print_vector_store_info(vector_store: Any) -> None:
        """
        Print information about a vector store.
        
        Args:
            vector_store: Vector store to inspect
        """
        info = DebugInspector.inspect_vector_store(vector_store)
        
        print(f"=== Vector Store ({info['type']}) ===")
        print(f"Document count: {info['document_count']}")
        print(f"Embedding dimension: {info['dimension']}")
        
        if "metadata_fields" in info:
            print(f"\nMetadata fields: {', '.join(info['metadata_fields'])}")
        
        if "sample_documents" in info and info["sample_documents"]:
            print("\nSample documents:")
            for i, doc in enumerate(info["sample_documents"]):
                print(f"\n  Document {i+1} (ID: {doc['id']}):")
                print(f"    Metadata: {doc['metadata']}")
                print(f"    Content: {doc['content_preview']}")
        
        if "error" in info:
            print(f"\nError inspecting vector store: {info['error']}")


# Create a global debug logger
debug_logger = DebugLogger()
