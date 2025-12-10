"""Lightweight preprocessing for Graphiti - LLM-First approach."""

import fitz  # PyMuPDF
import json
import os
from langdetect import detect
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    def __init__(self):
        """Initialize the lightweight preprocessor."""
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract clean text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
            return full_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def detect_language(self, text: str) -> str:
        """Detect language of text using langdetect."""
        try:
            return detect(text)
        except:
            return "unknown"
    
    def chunk_by_pages(self, text: str, filename: str) -> List[Dict[str, str]]:
        """
        Simple chunking by pages. Since we already have page breaks from PyMuPDF,
        we'll treat each page as a chunk.
        """
        # Split by the page breaks that PyMuPDF adds
        pages = text.split('\n\n')  # PyMuPDF adds double newlines between pages
        chunks = []
        
        for i, page_text in enumerate(pages):
            page_text = page_text.strip()
            if len(page_text) > 50:  # Only keep substantial pages
                chunk_id = f"{os.path.splitext(filename)[0]}_pagina_{i+1}"
                chunks.append({
                    "id": chunk_id,
                    "testo": page_text
                })
        
        # If no chunks were created (maybe different formatting), create one chunk per 1000 chars
        if not chunks:
            # Fallback: chunk by character count
            chunk_size = 2000  # Larger chunks for better context
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size].strip()
                if len(chunk_text) > 50:
                    chunk_id = f"{os.path.splitext(filename)[0]}_chunk_{i//chunk_size + 1}"
                    chunks.append({
                        "id": chunk_id,
                        "testo": chunk_text
                    })
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Lightweight processing: extract text, detect language, and chunk by pages.
        
        Returns a simple JSON structure with metadata and chunks.
        """
        try:
            logger.info(f"Processing {pdf_path}")
            
            # Get filename
            filename = os.path.basename(pdf_path)
            
            # Extract text
            full_text = self.extract_text_from_pdf(pdf_path)
            if not full_text:
                logger.error(f"No text extracted from {pdf_path}")
                return None
            
            # Detect language
            language = self.detect_language(full_text)
            
            # Create chunks
            chunks = self.chunk_by_pages(full_text, filename)
            
            # Create simple output structure
            output_data = {
                "file_sorgente": filename,
                "metadati": {
                    "lingua": language,
                    "numero_pagine": len(chunks)
                },
                "chunks": chunks
            }
            
            return output_data
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return None

def main():
    """Main function to process all PDFs in the input directory."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import INPUT_DOCUMENTS_PATH, OUTPUT_JSON_PATH
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_JSON_PATH, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor()
    
    # Process all PDF files
    pdf_files = [f for f in os.listdir(INPUT_DOCUMENTS_PATH) if f.endswith('.pdf')]
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DOCUMENTS_PATH, pdf_file)
        output_data = preprocessor.process_pdf(pdf_path)
        
        if output_data:
            # Save to JSON file
            output_filename = os.path.splitext(pdf_file)[0] + '.json'
            output_path = os.path.join(OUTPUT_JSON_PATH, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {output_path}")
        else:
            logger.error(f"Failed to process {pdf_file}")

if __name__ == "__main__":
    main()
