import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types

# Load environment variables
load_dotenv()

class LLMInterface:
    def __init__(self, model_name: str = None, use_openrouter: bool = False, use_gemini: bool = False):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: Name of the model to use
            use_openrouter: Whether to use OpenRouter API
            use_gemini: Whether to use Google's Gemini API
        """
        self.use_openrouter = use_openrouter
        self.use_gemini = use_gemini
        
        if use_gemini:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=self.api_key)
            self.model_name = model_name or "gemini-2.5-flash-preview-04-17"
        elif use_openrouter:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
            # Default to a model if none specified
            self.model_name = model_name or "openrouter/anthropic/claude-3-opus"
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=self.api_key)
            # Default to a model if none specified
            self.model_name = model_name or "gpt-4o"
    
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
            if self.use_gemini:
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                return response.text
            else:
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
            return f"Error generating response: {str(e)}"