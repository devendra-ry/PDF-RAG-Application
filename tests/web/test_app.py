"""Tests for the web interface."""

import unittest

class TestWebInterface(unittest.TestCase):
    def test_web_import(self):
        """Test that the web module can be imported"""
        try:
            from src.web.app import main
            # If we get here, the import worked
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Web module import failed with error: {e}")

if __name__ == "__main__":
    unittest.main()