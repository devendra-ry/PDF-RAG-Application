"""Main RAG application module for the PDF RAG Application.

This module provides the main application class that orchestrates:
1. PDF processing and chunking
2. Vector store management
3. LLM interaction for query processing
"""

import os
from typing import List, Dict, Any
from .pdf_processor import PDFProcessor
from .vector_store import VectorStore
from .llm_interface import LLMInterface

class RAGApp:
    def __init__(
        self,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        llm_model_name: str = "gpt-4o",
        api_base_url: str = "https://api.openai.com/v1",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        retrieval_k: int = 5
    ):
        """
        Initialize the RAG application.
        
        Args:
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the LLM model
            api_base_url: Base URL for the OpenAI-compatible API
            qdrant_url: URL of the Qdrant cluster (if None, uses in-memory)
            qdrant_api_key: API key for the Qdrant cluster (if needed)
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            retrieval_k: Number of chunks to retrieve for each query
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.api_base_url = api_base_url
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        
        # Initialize components
        self.pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(
            embedding_model_name=embedding_model_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        self.llm_interface = LLMInterface(model_name=llm_model_name, api_base_url=api_base_url)
        
        # Create vector store directory
        os.makedirs("vector_store", exist_ok=True)
    
    def index_pdfs(self, pdf_paths: List[str]) -> None:
        """
        Process and index PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
        """
        print(f"Processing {len(pdf_paths)} PDF files...")
        chunks = self.pdf_processor.process_pdfs(pdf_paths)
        print(f"Created {len(chunks)} chunks.")
        
        print("Computing embeddings and building index...")
        self.vector_store.add_chunks(chunks)
        print("Index built successfully.")
        
        # Save vector store
        self.vector_store.save()
        print("Vector store saved to disk.")
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a query and generate a response.
        
        Args:
            query_text: Query string
            
        Returns:
            Dictionary containing the query, retrieved chunks, and response
        """
        print(f"Processing query: {query_text}")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(query_text, k=self.retrieval_k)
        print(f"Retrieved {len(retrieved_chunks)} chunks.")
        
        # Generate response
        response = self.llm_interface.generate_response(query_text, retrieved_chunks)
        
        return {
            "query": query_text,
            "retrieved_chunks": retrieved_chunks,
            "response": response
        }
    
    @classmethod
    def load(
        cls,
        llm_model_name: str = "gpt-4o",
        api_base_url: str = "https://api.openai.com/v1",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        retrieval_k: int = 5
    ) -> 'RAGApp':
        """
        Load a RAG application with an existing vector store.
        
        Args:
            llm_model_name: Name of the LLM model
            api_base_url: Base URL for the OpenAI-compatible API
            qdrant_url: URL of the Qdrant cluster (if None, uses in-memory)
            qdrant_api_key: API key for the Qdrant cluster (if needed)
            retrieval_k: Number of chunks to retrieve for each query
            
        Returns:
            Loaded RAGApp instance
        """
        # Load vector store
        vector_store = VectorStore.load(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        # Create instance
        app = cls(
            embedding_model_name=vector_store.embedding_model_name,
            llm_model_name=llm_model_name,
            api_base_url=api_base_url,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            retrieval_k=retrieval_k
        )
        
        # Replace vector store with loaded one
        app.vector_store = vector_store
        
        return app