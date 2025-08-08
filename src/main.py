"""Main entry point for the PDF RAG Application."""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))

from src.web.app import main

if __name__ == "__main__":
    main()