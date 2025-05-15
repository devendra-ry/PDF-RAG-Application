import os
import fitz  # PyMuPDF
import tiktoken
import re
from typing import List, Dict, Tuple

class PDFProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Initialize the PDF processor with chunking parameters.
        
        Args:
            chunk_size: Maximum size of each text chunk in tokens (used as a safeguard)
            chunk_overlap: Number of tokens to overlap between chunks when fallback to token-based chunking
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
        Split text into semantic chunks based on document structure.
        
        Args:
            text: Text to split into chunks
            source: Source identifier (e.g., file path)
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        chunks = []
        
        # Split by sections using common section indicators
        # Look for headers, titles, or numbered sections
        section_patterns = [
            r'\n\s*#{1,6}\s+.+\n',  # Markdown headers
            r'\n\s*[A-Z][A-Za-z\s]+\n[-=]+\n',  # Underlined headers
            r'\n\s*[\d\.]+\s+[A-Z][^\n]+\n',  # Numbered sections like "1.1 Introduction"
            r'\n\s*[A-Z][A-Z\s]+[A-Z]:\s',  # ALL CAPS headers with colon
            r'\n\s*[A-Z][a-zA-Z\s]+\n',  # Potential title or header (capitalized line)
        ]
        
        # Combine patterns
        section_pattern = '|'.join(section_patterns)
        
        # First try to split by sections
        sections = re.split(f'({section_pattern})', text)
        
        # Recombine section headers with their content
        semantic_chunks = []
        for i in range(0, len(sections)-1, 2):
            if i+1 < len(sections):
                semantic_chunks.append(sections[i] + sections[i+1])
            else:
                semantic_chunks.append(sections[i])
        
        # If no sections were found, fall back to paragraph splitting
        if len(semantic_chunks) <= 1:
            semantic_chunks = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        # Process each semantic chunk
        current_position = 0
        for chunk_text in semantic_chunks:
            # Skip empty chunks
            if not chunk_text.strip():
                continue
                
            # Check if chunk is too large and needs further splitting
            chunk_tokens = self.encoding.encode(chunk_text)
            
            if len(chunk_tokens) <= self.chunk_size:
                # Chunk is an appropriate size, add it directly
                start_idx = current_position
                end_idx = current_position + len(chunk_tokens)
                
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "start_idx": start_idx,
                    "end_idx": end_idx
                })
                
                current_position = end_idx
            else:
                # Chunk is too large, apply token-based chunking as fallback
                i = 0
                while i < len(chunk_tokens):
                    # Extract chunk_size tokens
                    chunk_end = min(i + self.chunk_size, len(chunk_tokens))
                    sub_chunk_tokens = chunk_tokens[i:chunk_end]
                    sub_chunk_text = self.encoding.decode(sub_chunk_tokens)
                    
                    # Add chunk with metadata
                    chunks.append({
                        "text": sub_chunk_text,
                        "source": source,
                        "start_idx": current_position + i,
                        "end_idx": current_position + chunk_end
                    })
                    
                    # Move to next chunk, considering overlap
                    i += self.chunk_size - self.chunk_overlap
                
                current_position += len(chunk_tokens)
        
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