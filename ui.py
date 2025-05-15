import os
import streamlit as st
import tempfile
from rag_app import RAGApp

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
    
    # Model selection
    api_option = st.sidebar.radio(
        "Select API Provider:",
        ["Gemini"]
        #["OpenAI", "OpenRouter", "Gemini"]
    )
    
    use_openrouter = api_option == "OpenRouter"
    use_gemini = api_option == "Gemini"
    
    # In the main function, update the model selection part:
    if use_gemini:
        model_name = st.sidebar.selectbox(
            "Gemini Model",
            ["gemini-2.0-flash", "gemini-2.5-flash-preview-04-17"],
            index=0
        )
    # elif use_openrouter:
    #     model_name = st.sidebar.text_input(
    #         "OpenRouter Model Name",
    #         value="openrouter/anthropic/claude-3-opus",
    #         help="Format: openrouter/provider/model_name"
    #     )
    # else:
    #     model_name = st.sidebar.text_input(
    #         "OpenAI Model Name",
    #         value="gpt-4o",
    #         help="OpenAI model name"
    #     )
    
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["all-MiniLM-L12-v2", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2", "BAAI/bge-m3"],
        index=0
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size (tokens)", 100, 2000, 800)
        chunk_overlap = st.slider("Chunk Overlap (tokens)", 0, 500, 200)
        retrieval_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
    
    # Check if vector store exists
    vector_store_exists = os.path.exists("vector_store/index.faiss")
    
    # Initialize or load RAG app
    if "rag_app" not in st.session_state or st.sidebar.button("Reinitialize App"):
        if vector_store_exists:
            st.session_state.rag_app = RAGApp.load(
                llm_model_name=model_name,
                use_openrouter=use_openrouter,
                use_gemini=use_gemini,
                retrieval_k=retrieval_k
            )
            st.sidebar.success("Loaded existing vector store")
        else:
            st.session_state.rag_app = RAGApp(
                embedding_model_name=embedding_model,
                llm_model_name=model_name,
                use_openrouter=use_openrouter,
                use_gemini=use_gemini,
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