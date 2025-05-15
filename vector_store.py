import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store with an embedding model.
        
        Args:
            embedding_model_name: Name of the sentence-transformers model to use
        """
        self.embedding_model_name = embedding_model_name
        self.embedder = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        
    def add_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of dictionaries containing text chunks and metadata
        """
        if not chunks:
            return
            
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Compute embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store chunks
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            chunk["id"] = start_idx + i
            self.chunks.append(chunk)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        # Embed query
        query_embedding = self.embedder.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_embedding, k=min(k, len(self.chunks)))
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Ensure index is valid
                result = self.chunks[idx].copy()
                result["distance"] = float(distances[0][i])
                results.append(result)
                
        return results
    
    def save(self, directory: str = "vector_store") -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the vector store
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save chunks and metadata
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
            
        # Save model name
        with open(os.path.join(directory, "model_name.txt"), "w") as f:
            f.write(self.embedding_model_name)
    
    @classmethod
    def load(cls, directory: str = "vector_store") -> 'VectorStore':
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory containing the vector store
            
        Returns:
            Loaded VectorStore instance
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Vector store directory not found: {directory}")
            
        # Load model name
        with open(os.path.join(directory, "model_name.txt"), "r") as f:
            model_name = f.read().strip()
            
        # Create instance
        vector_store = cls(embedding_model_name=model_name)
        
        # Load index
        vector_store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load chunks
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            vector_store.chunks = pickle.load(f)
            
        return vector_store