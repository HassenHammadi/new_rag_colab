"""
Error handling utilities for the RAG system.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union, Callable, Type
import traceback
import sys
import time
from enum import Enum

from .debug_utils import debug_logger

class ErrorSeverity(Enum):
    """Enum for error severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RAGError(Exception):
    """Base class for RAG system errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                component: str = "RAGSystem", details: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG error.
        
        Args:
            message: Error message
            severity: Error severity
            component: Component that raised the error
            details: Additional error details
        """
        self.message = message
        self.severity = severity
        self.component = component
        self.details = details or {}
        self.timestamp = time.time()
        
        # Log the error
        self._log_error()
        
        super().__init__(message)
    
    def _log_error(self) -> None:
        """Log the error using the debug logger."""
        if self.severity == ErrorSeverity.INFO:
            debug_logger.logger.info(f"[{self.component}] {self.message}")
        elif self.severity == ErrorSeverity.WARNING:
            debug_logger.logger.warning(f"[{self.component}] {self.message}")
        elif self.severity == ErrorSeverity.ERROR:
            debug_logger.logger.error(f"[{self.component}] {self.message}")
        elif self.severity == ErrorSeverity.CRITICAL:
            debug_logger.logger.critical(f"[{self.component}] {self.message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary."""
        return {
            "message": self.message,
            "severity": self.severity.value,
            "component": self.component,
            "details": self.details,
            "timestamp": self.timestamp,
            "type": self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """Get a string representation of the error."""
        return f"{self.severity.value} [{self.component}]: {self.message}"


class FileError(RAGError):
    """Error related to file operations."""
    
    def __init__(self, message: str, file_path: str, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                component: str = "FileSystem", details: Optional[Dict[str, Any]] = None):
        """
        Initialize the file error.
        
        Args:
            message: Error message
            file_path: Path to the file that caused the error
            severity: Error severity
            component: Component that raised the error
            details: Additional error details
        """
        details = details or {}
        details["file_path"] = file_path
        
        super().__init__(message, severity, component, details)


class ProcessingError(RAGError):
    """Error related to document processing."""
    
    def __init__(self, message: str, document_type: str, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                component: str = "Processor", details: Optional[Dict[str, Any]] = None):
        """
        Initialize the processing error.
        
        Args:
            message: Error message
            document_type: Type of document being processed
            severity: Error severity
            component: Component that raised the error
            details: Additional error details
        """
        details = details or {}
        details["document_type"] = document_type
        
        super().__init__(message, severity, component, details)


class EmbeddingError(RAGError):
    """Error related to embedding generation."""
    
    def __init__(self, message: str, model: str, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                component: str = "EmbeddingProvider", details: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding error.
        
        Args:
            message: Error message
            model: Embedding model being used
            severity: Error severity
            component: Component that raised the error
            details: Additional error details
        """
        details = details or {}
        details["model"] = model
        
        super().__init__(message, severity, component, details)


class VectorStoreError(RAGError):
    """Error related to vector store operations."""
    
    def __init__(self, message: str, operation: str, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                component: str = "VectorStore", details: Optional[Dict[str, Any]] = None):
        """
        Initialize the vector store error.
        
        Args:
            message: Error message
            operation: Operation being performed
            severity: Error severity
            component: Component that raised the error
            details: Additional error details
        """
        details = details or {}
        details["operation"] = operation
        
        super().__init__(message, severity, component, details)


class DriveError(RAGError):
    """Error related to Google Drive operations."""
    
    def __init__(self, message: str, operation: str, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                component: str = "DriveHandler", details: Optional[Dict[str, Any]] = None):
        """
        Initialize the Drive error.
        
        Args:
            message: Error message
            operation: Operation being performed
            severity: Error severity
            component: Component that raised the error
            details: Additional error details
        """
        details = details or {}
        details["operation"] = operation
        
        super().__init__(message, severity, component, details)


class APIError(RAGError):
    """Error related to API calls."""
    
    def __init__(self, message: str, api: str, status_code: Optional[int] = None, 
                severity: ErrorSeverity = ErrorSeverity.ERROR, component: str = "APIClient", 
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize the API error.
        
        Args:
            message: Error message
            api: API being called
            status_code: HTTP status code
            severity: Error severity
            component: Component that raised the error
            details: Additional error details
        """
        details = details or {}
        details["api"] = api
        if status_code is not None:
            details["status_code"] = status_code
        
        super().__init__(message, severity, component, details)


def handle_error(error: Exception, fallback_value: Any = None, 
                log_error: bool = True, raise_error: bool = False) -> Any:
    """
    Handle an error with a fallback value.
    
    Args:
        error: Exception to handle
        fallback_value: Value to return if the error is handled
        log_error: Whether to log the error
        raise_error: Whether to re-raise the error
        
    Returns:
        Fallback value if the error is handled, otherwise raises the error
    """
    if log_error:
        debug_logger.logger.error(f"Error: {str(error)}")
        debug_logger.logger.debug(traceback.format_exc())
    
    if raise_error:
        raise error
    
    return fallback_value


def retry_on_error(max_retries: int = 3, retry_delay: float = 1.0, 
                  backoff_factor: float = 2.0, exceptions: Optional[List[Type[Exception]]] = None) -> Callable:
    """
    Decorator to retry a function on error.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by after each retry
        exceptions: List of exceptions to retry on (None for all exceptions)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = retry_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry this exception
                    if exceptions is not None and not any(isinstance(e, exc) for exc in exceptions):
                        raise e
                    
                    # Log the error
                    if attempt < max_retries:
                        debug_logger.logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}")
                        debug_logger.logger.debug(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        debug_logger.logger.error(f"All {max_retries + 1} attempts failed")
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    
    return decorator


def safe_execute(func: Callable, *args, fallback_value: Any = None, 
                log_error: bool = True, **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        fallback_value: Value to return if an error occurs
        log_error: Whether to log the error
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or fallback value if an error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            debug_logger.logger.error(f"Error executing {func.__name__}: {str(e)}")
            debug_logger.logger.debug(traceback.format_exc())
        
        return fallback_value
