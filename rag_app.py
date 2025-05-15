import os
from typing import List, Dict, Any
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_interface import LLMInterface

class RAGApp:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = None,
        use_openrouter: bool = False,
        use_gemini: bool = False,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        retrieval_k: int = 5
    ):
        """
        Initialize the RAG application.
        
        Args:
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the LLM model
            use_openrouter: Whether to use OpenRouter API
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            retrieval_k: Number of chunks to retrieve for each query
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.use_openrouter = use_openrouter
        self.use_gemini = use_gemini
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        
        # Initialize components
        self.pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(embedding_model_name=embedding_model_name)
        self.llm_interface = LLMInterface(model_name=llm_model_name, use_openrouter=use_openrouter, use_gemini=use_gemini)
        
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
        llm_model_name: str = None,
        use_openrouter: bool = False,
        use_gemini: bool = False,
        retrieval_k: int = 5
    ) -> 'RAGApp':
        """
        Load a RAG application with an existing vector store.
        
        Args:
            llm_model_name: Name of the LLM model
            use_openrouter: Whether to use OpenRouter API
            retrieval_k: Number of chunks to retrieve for each query
            
        Returns:
            Loaded RAGApp instance
        """
        # Load vector store
        vector_store = VectorStore.load()
        
        # Create instance
        app = cls(
            embedding_model_name=vector_store.embedding_model_name,
            llm_model_name=llm_model_name,
            use_openrouter=use_openrouter,
            use_gemini=use_gemini,
            retrieval_k=retrieval_k
        )
        
        # Replace vector store with loaded one
        app.vector_store = vector_store
        
        return app