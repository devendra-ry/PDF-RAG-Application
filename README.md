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
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # Main entry point
│   ├── components/               # Core components
│   │   ├── __init__.py          # Components package initialization
│   │   ├── pdf_processor.py     # PDF text extraction and chunking
│   │   ├── vector_store.py      # Vector store implementation
│   │   ├── llm_interface.py     # LLM interface
│   │   └── application.py       # Main application
│   └── web/                      # Web interface modules
│       ├── __init__.py          # Web package initialization
│       └── app.py               # Streamlit web application
├── tests/                        # Test modules
│   ├── __init__.py              # Tests package initialization
│   ├── components/              # Component tests
│   │   ├── __init__.py         # Component tests initialization
│   │   ├── test_vector_store.py # Vector store tests
│   │   └── test_application.py # Application tests
│   ├── web/                     # Web tests
│   │   ├── __init__.py         # Web tests initialization
│   │   └── test_app.py         # Web application tests
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
streamlit run src/web/app.py
```

## Usage

1. Start the application with `streamlit run src/web/app.py`
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
pytest tests/
```

## Example Document

An example PDF document (`example_ml_document.pdf`) is included to help you test the application. This document contains information about machine learning concepts and applications.