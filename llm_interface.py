import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMInterface:
    def __init__(self, model_name: str = "gpt-4o", api_base_url: str = "https://api.openai.com/v1"):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: Name of the model to use
            api_base_url: Base URL for the OpenAI-compatible API
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        # Debug: Print key length (be careful not to print the key itself in logs)
        # print(f"DEBUG: Loaded API key length: {len(self.api_key)}")
        
        # Initialize the OpenAI client with custom base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=api_base_url
        )
        
        self.model_name = model_name
    
    def create_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a prompt combining the query and retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved text chunks
            
        Returns:
            Formatted prompt string
        """
        context = "\n\n---\n\n".join([
            f"Source: {chunk['source']}\n\n{chunk['text']}" 
            for chunk in retrieved_chunks
        ])
        
        prompt = f"""Answer the following question based on the provided context. If the answer cannot be determined from the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response to the query using the LLM.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved text chunks
            
        Returns:
            LLM response
        """
        prompt = self.create_prompt(query, retrieved_chunks)
        try:
            # Debug: Print request details
            print(f"DEBUG LLM Request:")
            print(f"  Base URL: {self.client.base_url}")
            print(f"  Model: {self.model_name}")
            # Note: Do not print the full API key for security
            print(f"  API Key Length: {len(self.api_key) if self.api_key else 'None'}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            # Debug: Print the full error
            print(f"DEBUG LLM Error: {e}")
            return f"Error generating response: {str(e)}"