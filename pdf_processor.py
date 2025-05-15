import os
import fitz  # PyMuPDF
import tiktoken
from typing import List, Dict, Tuple

class PDFProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Initialize the PDF processor with chunking parameters.
        
        Args:
            chunk_size: Target size of each text chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        text = ""
        try:
            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text() + "\n"
                
            # Close the document
            doc.close()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
            
        return text
    
    def extract_text_from_pdfs(self, pdf_paths: List[str]) -> Dict[str, str]:
        """
        Extract text from multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping file paths to extracted text
        """
        result = {}
        for path in pdf_paths:
            result[path] = self.extract_text_from_pdf(path)
        return result
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def create_chunks(self, text: str, source: str = "") -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split into chunks
            source: Source identifier (e.g., file path)
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        i = 0
        while i < len(tokens):
            # Extract chunk_size tokens
            chunk_end = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:chunk_end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Add chunk with metadata
            chunks.append({
                "text": chunk_text,
                "source": source,
                "start_idx": i,
                "end_idx": chunk_end
            })
            
            # Move to next chunk, considering overlap
            i += self.chunk_size - self.chunk_overlap
            
        return chunks
    
    def process_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, str]]:
        """
        Process multiple PDFs into chunks.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        all_chunks = []
        pdf_texts = self.extract_text_from_pdfs(pdf_paths)
        
        for path, text in pdf_texts.items():
            chunks = self.create_chunks(text, source=path)
            all_chunks.extend(chunks)
            
        return all_chunks