import unittest
import sys
import os

# Add the project root to the path so we can import the components modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.components.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def test_vector_store_initialization(self):
        """Test that VectorStore can be initialized with Qdrant"""
        try:
            # Use a smaller model for testing
            vector_store = VectorStore("all-MiniLM-L6-v2")
            self.assertIsNotNone(vector_store.client)
            self.assertIsNotNone(vector_store.embedder)
        except Exception as e:
            self.fail(f"VectorStore initialization failed with error: {e}")
    
    def test_add_and_search_chunks(self):
        """Test adding chunks and searching"""
        # Use a smaller model for testing
        vector_store = VectorStore("all-MiniLM-L6-v2")
        
        # Sample chunks
        chunks = [
            {"text": "The quick brown fox jumps over the lazy dog", "source": "test1.txt"},
            {"text": "Machine learning is a subset of artificial intelligence", "source": "test2.txt"},
            {"text": "Python is a high-level programming language", "source": "test3.txt"}
        ]
        
        # Add chunks
        vector_store.add_chunks(chunks)
        
        # Search for similar chunks
        results = vector_store.search("Tell me about programming", k=2)
        
        # Verify results
        self.assertEqual(len(results), 2)
        # Check that the results contain expected fields
        for result in results:
            self.assertIn("text", result)
            self.assertIn("source", result)
            self.assertIn("distance", result)

if __name__ == "__main__":
    unittest.main()