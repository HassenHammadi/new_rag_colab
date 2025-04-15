"""
Unit tests for the RAG system components.
"""

import unittest
import os
import tempfile
from pathlib import Path
import json
import numpy as np

# Import components to test
from ..utils.debug_utils import debug_logger, DebugInspector
from ..utils.drive_utils import DriveHandler
from ..vector_stores.drive_vector_store import DriveVectorStore
from ..chunkers.base_chunker import FixedSizeChunker
from ..chunkers.advanced_chunker import ContextualHeaderChunker
from ..retrievers.base_retriever import SimpleRetriever
from ..retrievers.advanced_retriever import QueryTransformationRetriever
from ..utils.embeddings import HuggingFaceEmbeddingProvider
from ..utils.colab_rag_pipeline import ColabRAGPipeline

# Set debug logger to ERROR level to reduce test output
debug_logger.logger.setLevel("ERROR")

class TestEmbeddingProvider(unittest.TestCase):
    """Test the embedding provider."""
    
    def test_embedding_dimension(self):
        """Test that embeddings have the correct dimension."""
        # Create a mock embedding function
        def mock_embedding_fn(text):
            # Return a fixed-size vector based on the hash of the text
            text_hash = hash(text) % 1000
            return [float(text_hash) / 1000] * 384
        
        # Create test texts
        texts = ["This is a test", "Another test", "Third test text"]
        
        # Create embeddings
        embeddings = [mock_embedding_fn(text) for text in texts]
        
        # Check dimensions
        for embedding in embeddings:
            self.assertEqual(len(embedding), 384)


class TestVectorStore(unittest.TestCase):
    """Test the vector store."""
    
    def setUp(self):
        """Set up the test."""
        # Create a mock embedding function
        def mock_embedding_fn(text):
            # Return a fixed-size vector based on the hash of the text
            text_hash = hash(text) % 1000
            return [float(text_hash) / 1000] * 384
        
        # Create a vector store
        self.vector_store = DriveVectorStore(
            embedding_function=mock_embedding_fn,
            dimension=384,
            debug=False
        )
        
        # Add some documents
        self.documents = [
            {
                "content": "This is a test document about machine learning.",
                "metadata": {"source_file": "test1.txt", "source_type": "text"}
            },
            {
                "content": "Another document about artificial intelligence.",
                "metadata": {"source_file": "test2.txt", "source_type": "text"}
            },
            {
                "content": "Document about natural language processing.",
                "metadata": {"source_file": "test3.txt", "source_type": "text"}
            }
        ]
        
        self.doc_ids = self.vector_store.add_documents(self.documents)
    
    def test_add_documents(self):
        """Test adding documents to the vector store."""
        # Check that documents were added
        self.assertEqual(len(self.vector_store.documents), 3)
        
        # Check that document IDs were generated
        self.assertEqual(len(self.doc_ids), 3)
        
        # Check that documents have IDs
        for doc in self.vector_store.documents:
            self.assertIn("id", doc)
    
    def test_similarity_search(self):
        """Test similarity search."""
        # Search for documents
        results = self.vector_store.similarity_search("machine learning", k=2)
        
        # Check that results were returned
        self.assertEqual(len(results), 2)
        
        # Check that results have scores
        for result in results:
            self.assertIn("score", result)
    
    def test_save_load(self):
        """Test saving and loading the vector store."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the vector store
            self.vector_store.save(temp_dir)
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "documents.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "faiss.index")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "metadata.json")))
            
            # Create a new vector store
            def mock_embedding_fn(text):
                text_hash = hash(text) % 1000
                return [float(text_hash) / 1000] * 384
            
            new_vector_store = DriveVectorStore(
                embedding_function=mock_embedding_fn,
                dimension=384,
                debug=False
            )
            
            # Load the vector store
            new_vector_store.load(temp_dir)
            
            # Check that documents were loaded
            self.assertEqual(len(new_vector_store.documents), 3)
            
            # Check that search works
            results = new_vector_store.similarity_search("machine learning", k=2)
            self.assertEqual(len(results), 2)


class TestChunker(unittest.TestCase):
    """Test the chunker."""
    
    def test_fixed_size_chunker(self):
        """Test the fixed-size chunker."""
        # Create a chunker
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20, debug=False)
        
        # Create a document
        document = {
            "content": "This is a test document. " * 20,  # 480 characters
            "metadata": {"source_file": "test.txt", "source_type": "text"}
        }
        
        # Chunk the document
        chunks = chunker.chunk_document(document)
        
        # Check that chunks were created
        self.assertGreater(len(chunks), 1)
        
        # Check that chunks have content and metadata
        for chunk in chunks:
            self.assertIn("content", chunk)
            self.assertIn("metadata", chunk)
            self.assertLessEqual(len(chunk["content"]), 100)
    
    def test_contextual_header_chunker(self):
        """Test the contextual header chunker."""
        # Create a base chunker
        base_chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20, debug=False)
        
        # Create a contextual header chunker
        chunker = ContextualHeaderChunker(base_chunker, debug=False)
        
        # Create a document with headers
        document = {
            "content": "# Header 1\n\nThis is content under header 1.\n\n## Subheader\n\nThis is content under subheader.\n\n# Header 2\n\nThis is content under header 2.",
            "metadata": {"source_file": "test.txt", "source_type": "text"}
        }
        
        # Chunk the document
        chunks = chunker.chunk_document(document)
        
        # Check that chunks were created
        self.assertGreater(len(chunks), 0)


class TestRetriever(unittest.TestCase):
    """Test the retriever."""
    
    def setUp(self):
        """Set up the test."""
        # Create a mock embedding function
        def mock_embedding_fn(text):
            # Return a fixed-size vector based on the hash of the text
            text_hash = hash(text) % 1000
            return [float(text_hash) / 1000] * 384
        
        # Create a vector store
        self.vector_store = DriveVectorStore(
            embedding_function=mock_embedding_fn,
            dimension=384,
            debug=False
        )
        
        # Add some documents
        self.documents = [
            {
                "content": "This is a test document about machine learning.",
                "metadata": {"source_file": "test1.txt", "source_type": "text"}
            },
            {
                "content": "Another document about artificial intelligence.",
                "metadata": {"source_file": "test2.txt", "source_type": "text"}
            },
            {
                "content": "Document about natural language processing.",
                "metadata": {"source_file": "test3.txt", "source_type": "text"}
            }
        ]
        
        self.doc_ids = self.vector_store.add_documents(self.documents)
        
        # Create a retriever
        self.retriever = SimpleRetriever(self.vector_store, top_k=2, debug=False)
    
    def test_simple_retriever(self):
        """Test the simple retriever."""
        # Retrieve documents
        results = self.retriever.retrieve("machine learning")
        
        # Check that results were returned
        self.assertEqual(len(results), 2)
        
        # Check that results have scores
        for result in results:
            self.assertIn("score", result)
    
    def test_query_transformation_retriever(self):
        """Test the query transformation retriever."""
        # Create a query transformation retriever
        retriever = QueryTransformationRetriever(self.retriever, debug=False)
        
        # Retrieve documents
        results = retriever.retrieve("ML")
        
        # Check that results were returned
        self.assertEqual(len(results), 2)
        
        # Check that results have metadata with original and transformed queries
        for result in results:
            self.assertIn("metadata", result)
            self.assertIn("original_query", result["metadata"])
            self.assertIn("transformed_query", result["metadata"])


class TestRAGPipeline(unittest.TestCase):
    """Test the RAG pipeline."""
    
    def setUp(self):
        """Set up the test."""
        # Create a mock embedding function
        def mock_embedding_fn(text):
            # Return a fixed-size vector based on the hash of the text
            text_hash = hash(text) % 1000
            return [float(text_hash) / 1000] * 384
        
        # Create components
        self.vector_store = DriveVectorStore(
            embedding_function=mock_embedding_fn,
            dimension=384,
            debug=False
        )
        
        self.chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20, debug=False)
        self.retriever = SimpleRetriever(self.vector_store, top_k=2, debug=False)
        
        # Create a RAG pipeline
        self.rag_pipeline = ColabRAGPipeline(
            chunker=self.chunker,
            vector_store=self.vector_store,
            retriever=self.retriever,
            processors={},
            debug=False
        )
        
        # Add some documents directly to the vector store
        self.documents = [
            {
                "content": "This is a test document about machine learning.",
                "metadata": {"source_file": "test1.txt", "source_type": "text"}
            },
            {
                "content": "Another document about artificial intelligence.",
                "metadata": {"source_file": "test2.txt", "source_type": "text"}
            },
            {
                "content": "Document about natural language processing.",
                "metadata": {"source_file": "test3.txt", "source_type": "text"}
            }
        ]
        
        self.doc_ids = self.vector_store.add_documents(self.documents)
    
    def test_query(self):
        """Test querying the RAG pipeline."""
        # Query the pipeline
        results = self.rag_pipeline.query("machine learning")
        
        # Check that results were returned
        self.assertEqual(len(results), 2)
        
        # Check that results have scores
        for result in results:
            self.assertIn("score", result)
    
    def test_query_with_markdown(self):
        """Test querying the RAG pipeline with Markdown formatting."""
        # Query the pipeline with Markdown formatting
        markdown = self.rag_pipeline.query_with_markdown("machine learning")
        
        # Check that Markdown was returned
        self.assertIsInstance(markdown, str)
        self.assertIn("```markdown", markdown)


if __name__ == "__main__":
    unittest.main()
