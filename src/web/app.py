"""Streamlit UI for the PDF RAG Application.

This module provides a web interface for:
1. Uploading and indexing PDF documents
2. Configuring LLM and embedding models
3. Querying the indexed documents
"""

import os
import sys
import streamlit as st
import tempfile

# Add the project root to the path so we can import the components modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.components.application import RAGApp

st.set_page_config(page_title="PDF RAG App", page_icon="📚", layout="wide")

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary directory and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def main():
    st.title("📚 PDF RAG Application")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Configuration
    st.sidebar.subheader("LLM API Configuration")
    api_base_url = st.sidebar.text_input(
        "API Base URL",
        value="https://openrouter.ai/api/v1",
        help="Base URL for the OpenAI-compatible API (e.g., OpenAI, OpenRouter)"
    )
    
    # Model selection
    model_name = st.sidebar.text_input(
        "Model Name",
        value="z-ai/glm-4.5-air:free",
        help="Name of the model to use"
    )
    
    # Qdrant Configuration
    st.sidebar.subheader("Qdrant Configuration")
    use_external_qdrant = st.sidebar.checkbox("Use External Qdrant Cluster", value=False)
    
    if use_external_qdrant:
        qdrant_url = st.sidebar.text_input(
            "Qdrant URL",
            value="https://dfc9f545-1fba-45eb-ac87-a2c620553ca6.eu-central-1-0.aws.cloud.qdrant.io:6333",
            help="URL of your Qdrant cluster"
        )
        # Try to get API key from environment variable first, with option to override
        default_qdrant_key = os.getenv("QDRANT_API_KEY", "")
        if default_qdrant_key:
            qdrant_api_key = st.sidebar.text_input(
                "Qdrant API Key",
                value=default_qdrant_key,
                type="password",
                help="API key for your Qdrant cluster (loaded from environment variable)"
            )
        else:
            qdrant_api_key = st.sidebar.text_input(
                "Qdrant API Key",
                value="",
                type="password",
                help="API key for your Qdrant cluster"
            )
    else:
        qdrant_url = None
        qdrant_api_key = None
    
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["Qwen/Qwen3-Embedding-0.6B", "all-MiniLM-L12-v2", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2", "BAAI/bge-m3"],
        index=0
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size (tokens)", 100, 2000, 800)
        chunk_overlap = st.slider("Chunk Overlap (tokens)", 0, 500, 200)
        retrieval_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
    
    # Check if vector store exists and is complete (has model_name.txt)
    vector_store_exists = os.path.exists("vector_store") and os.path.exists(os.path.join("vector_store", "model_name.txt"))
    
    # Initialize or load RAG app
    if "rag_app" not in st.session_state or st.sidebar.button("Reinitialize App"):
        if vector_store_exists:
            st.session_state.rag_app = RAGApp.load(
                llm_model_name=model_name,
                api_base_url=api_base_url,
                qdrant_url=qdrant_url if use_external_qdrant else None,
                qdrant_api_key=qdrant_api_key if use_external_qdrant else None,
                retrieval_k=retrieval_k
            )
            st.sidebar.success("Loaded existing vector store")
        else:
            st.session_state.rag_app = RAGApp(
                embedding_model_name=embedding_model,
                llm_model_name=model_name,
                api_base_url=api_base_url,
                qdrant_url=qdrant_url if use_external_qdrant else None,
                qdrant_api_key=qdrant_api_key if use_external_qdrant else None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                retrieval_k=retrieval_k
            )
            st.sidebar.success("Initialized new RAG app")
    
    # Main area
    tab1, tab2 = st.tabs(["Upload & Index", "Query"])
    
    # Upload & Index tab
    with tab1:
        st.header("Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                # Save uploaded files to temporary location
                pdf_paths = []
                for uploaded_file in uploaded_files:
                    path = save_uploaded_file(uploaded_file)
                    if path:
                        pdf_paths.append(path)
                
                # Index PDFs
                if pdf_paths:
                    st.session_state.rag_app.index_pdfs(pdf_paths)
                    st.success(f"Successfully processed {len(pdf_paths)} PDF files")
                    
                    # Clean up temporary files
                    for path in pdf_paths:
                        try:
                            os.unlink(path)
                        except:
                            pass
    
    # Query tab
    with tab2:
        st.header("Ask Questions")
        
        if not vector_store_exists and "rag_app" in st.session_state and not st.session_state.rag_app.vector_store.chunks:
            st.warning("Please upload and process PDF files first")
        else:
            query = st.text_input("Enter your question:")
            
            if query and st.button("Submit"):
                with st.spinner("Generating response..."):
                    result = st.session_state.rag_app.query(query)
                    
                    st.subheader("Response")
                    st.write(result["response"])
                    
                    with st.expander("View Retrieved Chunks"):
                        for i, chunk in enumerate(result["retrieved_chunks"]):
                            st.markdown(f"**Chunk {i+1}** (Source: {os.path.basename(chunk['source'])})")
                            st.text(chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"])
                            st.markdown("---")

if __name__ == "__main__":
    main()