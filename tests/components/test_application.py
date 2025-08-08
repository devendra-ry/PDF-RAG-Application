"""Tests for the main application module."""

import unittest
from src.components.application import RAGApp

class TestApplication(unittest.TestCase):
    def test_application_initialization(self):
        """Test that RAGApp can be initialized"""
        try:
            # Use a smaller model for testing
            app = RAGApp(embedding_model_name="all-MiniLM-L6-v2")
            self.assertIsNotNone(app.pdf_processor)
            self.assertIsNotNone(app.vector_store)
            self.assertIsNotNone(app.llm_interface)
        except Exception as e:
            self.fail(f"RAGApp initialization failed with error: {e}")

if __name__ == "__main__":
    unittest.main()