"""Vector store implementation for the PDF RAG Application.

This module provides functionality to:
1. Generate embeddings using sentence-transformers
2. Store and retrieve document chunks using Qdrant
3. Perform similarity search on document chunks
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SearchParams
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch  # Import torch to check for CUDA

class VectorStore:
    def __init__(self, embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B", collection_name: str = "pdf_rag_collection", qdrant_url: str = None, qdrant_api_key: str = None):
        """
        Initialize the vector store with an embedding model.
        
        Args:
            embedding_model_name: Name of the sentence-transformers model to use
            collection_name: Name of the Qdrant collection
            qdrant_url: URL of the Qdrant cluster (if None, uses in-memory)
            qdrant_api_key: API key for the Qdrant cluster (if needed)
        """
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # Determine the device for the embedding model
        # Use CUDA if available, otherwise CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device for embedding model: {device}")
        
        # Load the embedding model
        # Special handling for Qwen model to avoid meta tensor issues
        if embedding_model_name == "Qwen/Qwen3-Embedding-0.6B":
            try:
                # Attempt to load Qwen model with trust_remote_code and explicit device
                # This can sometimes resolve issues with model initialization
                print("Loading Qwen/Qwen3-Embedding-0.6B with trust_remote_code=True...")
                # Load model without specifying device first to avoid potential conflicts
                self.embedder = SentenceTransformer(
                    embedding_model_name,
                    model_kwargs={"trust_remote_code": True},
                    tokenizer_kwargs={"padding_side": "left"}
                    # device=device  # Do not specify device in constructor for now
                )
                # If loaded successfully, try to move to the desired device
                if device != 'cpu':
                    print(f"Moving Qwen embedding model to device: {device}...")
                    self.embedder.to(device)
                    
            except NotImplementedError as e_ni:
                if "meta tensor" in str(e_ni) and "to_empty()" in str(e_ni):
                    print(f"Warning: Failed to move Qwen model to {device} due to meta tensor issue.")
                    print("Falling back to loading Qwen model directly on CPU.")
                    # Reload the model explicitly on CPU
                    self.embedder = SentenceTransformer(
                        embedding_model_name,
                        model_kwargs={"trust_remote_code": True},
                        tokenizer_kwargs={"padding_side": "left"},
                        device='cpu'
                    )
                    device = 'cpu' # Update device variable for consistency
                else:
                    # Re-raise if it's a different NotImplementedError
                    raise e_ni
            except Exception as e_qwen:
                print(f"Failed to load Qwen model with trust_remote_code: {e_qwen}")
                raise e_qwen # Re-raise to stop execution if Qwen is specifically requested
        else:
            # Load other models normally
            model_kwargs = {} # No special model kwargs
            self.embedder = SentenceTransformer(
                embedding_model_name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs={"padding_side": "left"}
                # device=device # Do not specify device in constructor for now
            )
            # Move other models to the desired device
            self.embedder.to(device)
            
        
        # Debug: Print the actual device of the model after loading and moving
        # The model object itself might be on CPU, but its components (like the transformer) should be on 'device'
        model_device = next(self.embedder.parameters()).device
        print(f"Embedding model '{embedding_model_name}' loaded. Expected device: {device}, Actual device of first parameter: {model_device}")
        
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        
        # Check if the model supports prompts
        self.supports_prompts = hasattr(self.embedder, 'prompts') and 'query' in self.embedder.prompts
        
        # Initialize Qdrant client
        if qdrant_url:
            # Connect to external Qdrant cluster
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
        else:
            # Use in-memory Qdrant
            self.client = QdrantClient(":memory:")
        
        # Check if collection exists and delete it if it does (for in-memory only)
        if not qdrant_url and self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
        
        # Create collection if it doesn't exist
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
            )
        
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
        
        # Debug: Print device info before encoding chunks
        model_device = next(self.embedder.parameters()).device
        print(f"Adding chunks: Embedding model is on device: {model_device}")
        
        # Compute embeddings
        if self.supports_prompts:
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
        else:
            # For models that don't support prompts, we encode directly
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Prepare points for Qdrant
        points = []
        start_idx = len(self.chunks)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = start_idx + i
            chunk["id"] = chunk_id
            
            # Store chunk
            self.chunks.append(chunk)
            
            # Create point for Qdrant
            points.append(PointStruct(
                id=chunk_id,
                vector=embedding.tolist(),
                payload=chunk  # Store the entire chunk as payload
            ))
        
        # Upload points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        # Debug: Print device info before encoding query
        model_device = next(self.embedder.parameters()).device
        print(f"Searching: Embedding model is on device: {model_device}")
        
        # Embed query with prompt if supported
        if self.supports_prompts:
            # Debug: Check if prompt_name is used correctly
            print(f"Searching: Using prompt_name='query' for model {self.embedding_model_name}")
            query_embedding = self.embedder.encode([query], prompt_name="query")[0].tolist()
        else:
            # For models that don't support prompts, we encode directly
            print(f"Searching: Encoding query directly for model {self.embedding_model_name}")
            query_embedding = self.embedder.encode([query])[0].tolist()
        
        # Search in Qdrant
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            search_params=SearchParams(exact=True)  # For precise results
        )
        
        # Format results
        results = []
        for result in search_result.points:
            chunk = result.payload.copy()
            chunk["distance"] = result.score
            results.append(chunk)
            
        return results
    
    def save(self, directory: str = "vector_store") -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the vector store
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save Qdrant data (only for in-memory)
        # For external Qdrant, data is already persisted
        if not self.client._client:
            self.client.dump_collection(
                collection_name=self.collection_name,
                path=os.path.join(directory, "qdrant_data")
            )
        
        # Save chunks and metadata (for compatibility with existing code)
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
            
        # Save model name
        with open(os.path.join(directory, "model_name.txt"), "w") as f:
            f.write(self.embedding_model_name)
            
        # Save collection name
        with open(os.path.join(directory, "collection_name.txt"), "w") as f:
            f.write(self.collection_name)
    
    @classmethod
    def load(cls, directory: str = "vector_store", qdrant_url: str = None, qdrant_api_key: str = None) -> 'VectorStore':
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory containing the vector store
            qdrant_url: URL of the Qdrant cluster (if None, uses in-memory)
            qdrant_api_key: API key for the Qdrant cluster (if needed)
            
        Returns:
            Loaded VectorStore instance
        """
        # Load model name
        with open(os.path.join(directory, "model_name.txt"), "r") as f:
            model_name = f.read().strip()
            
        # Load collection name
        with open(os.path.join(directory, "collection_name.txt"), "r") as f:
            collection_name = f.read().strip()
            
        # Create instance
        vector_store = cls(
            embedding_model_name=model_name, 
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        # For in-memory, load Qdrant data
        if not qdrant_url and os.path.exists(os.path.join(directory, "qdrant_data")):
            vector_store.client.restore_collection(
                collection_name=collection_name,
                path=os.path.join(directory, "qdrant_data")
            )
        
        # Load chunks
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            vector_store.chunks = pickle.load(f)
            
        return vector_store