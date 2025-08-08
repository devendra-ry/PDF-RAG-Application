# PDF RAG Application

This application allows you to upload PDF documents, index their content, and ask questions about them using any OpenAI API-compatible language model.

## Features

- PDF text extraction
- Text chunking with overlap
- Embedding generation using Qwen/Qwen3-Embedding-0.6B (Sentence Transformers)
- Vector search with Qdrant (in-memory or external cluster)
- Integration with any OpenAI API-compatible models (OpenAI, OpenRouter, Together AI, Mistral, etc.)
- Streamlit user interface

## Project Structure

```
pdf-rag-application/
├── pdf_rag_application/           # Main package
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # Main entry point
│   ├── core/                     # Core modules
│   │   ├── __init__.py          # Core package initialization
│   │   ├── pdf_processor.py     # PDF text extraction and chunking
│   │   ├── vector_store.py      # Vector store implementation
│   │   ├── llm_interface.py     # LLM interface
│   │   └── rag_app.py           # Main RAG application
│   └── ui/                       # User interface modules
│       ├── __init__.py          # UI package initialization
│       └── ui.py                # Streamlit UI
├── tests/                        # Test modules
│   ├── __init__.py              # Tests package initialization
│   ├── test_vector_store.py     # Vector store tests
│   └── run_tests.py             # Test runner
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── LICENSE                      # License information
```

## Setup

1. Clone this repository
2. Install [uv](https://github.com/astral-sh/uv) if you haven't already
3. Use the provided setup script:

   On Windows:
   ```bash
   setup.bat
   ```

   On Linux/Mac:
   ```bash
   ./setup.sh
   ```

   Or manually create a virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install -e .[dev]
   ```

4. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

5. Run the application:

```bash
streamlit run pdf_rag_application/ui/ui.py
```

## Usage

1. Start the application with `streamlit run pdf_rag_application/ui/ui.py`
2. In the sidebar, configure your settings:
   - **LLM API Configuration**:
     - Set the API Base URL (e.g., "https://api.openai.com/v1" for OpenAI, "https://openrouter.ai/api/v1" for OpenRouter)
     - Set the Model Name (e.g., "gpt-4o" for OpenAI, "mistralai/mistral-7b-instruct" for OpenRouter)
   - **Qdrant Configuration**:
     - Check "Use External Qdrant Cluster" to connect to your Qdrant cluster
     - The URL is pre-filled with your cluster endpoint
     - The API key is automatically loaded from your `.env` file
   - Select your preferred Embedding Model
3. Upload PDF documents in the "Upload & Index" tab
4. Ask questions about the documents in the "Query" tab

## Testing

To run tests:

```bash
python tests/run_tests.py
```

Or directly with pytest:

```bash
uv pip install -e .[dev]
pytest tests/test_vector_store.py
```

## Example Document

An example PDF document (`example_ml_document.pdf`) is included to help you test the application. This document contains information about machine learning concepts and applications.