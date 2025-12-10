"""Unit tests for the GraphRAG preprocessing module."""

import pytest
import os
from unittest.mock import Mock, patch, mock_open
from src.preprocess import DocumentPreprocessor

class TestDocumentPreprocessor:
    """Test the DocumentPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.preprocessor = DocumentPreprocessor()
    
    def test_preprocessor_initialization(self):
        """Test that the preprocessor initializes correctly."""
        assert isinstance(self.preprocessor, DocumentPreprocessor)
    
    @patch('src.preprocess.fitz.open')
    def test_extract_text_from_pdf_success(self, mock_fitz_open):
        """Test successful PDF text extraction."""
        # Mock the PyMuPDF document
        mock_doc = Mock()
        mock_page1 = Mock()
        mock_page2 = Mock()
        mock_page1.get_text.return_value = "Page 1 content\n"
        mock_page2.get_text.return_value = "Page 2 content\n"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page1, mock_page2]))
        mock_doc.__len__ = Mock(return_value=2)
        mock_fitz_open.return_value = mock_doc
        
        result = self.preprocessor.extract_text_from_pdf("test.pdf")
        assert "Page 1 content" in result
        assert "Page 2 content" in result
        # The actual implementation strips the final result
        assert result == "Page 1 content\n\nPage 2 content"
    
    @patch('src.preprocess.fitz.open')
    def test_extract_text_from_pdf_error(self, mock_fitz_open):
        """Test PDF text extraction with error."""
        mock_fitz_open.side_effect = Exception("PDF error")
        result = self.preprocessor.extract_text_from_pdf("test.pdf")
        assert result == ""
    
    @patch('src.preprocess.detect')
    def test_detect_language_success(self, mock_detect):
        """Test successful language detection."""
        mock_detect.return_value = "it"
        result = self.preprocessor.detect_language("Testo in italiano")
        assert result == "it"
    
    @patch('src.preprocess.detect')
    def test_detect_language_error(self, mock_detect):
        """Test language detection with error."""
        mock_detect.side_effect = Exception("Detection error")
        result = self.preprocessor.detect_language("Test text")
        assert result == "unknown"
    
    def test_chunk_by_pages_with_valid_pages(self):
        """Test chunking by pages with valid content."""
        # Create text that will be split into multiple chunks
        text = "A" * 60 + "\n\n" + "B" * 60 + "\n\n" + "C" * 60  # Each part > 50 chars
        chunks = self.preprocessor.chunk_by_pages(text, "test_doc.pdf")
        assert len(chunks) == 3
        assert chunks[0]["id"] == "test_doc_pagina_1"
        assert "A" * 60 in chunks[0]["testo"]
    
    def test_chunk_by_pages_with_fallback_chunking(self):
        """Test fallback chunking when page splitting doesn't create substantial chunks."""
        # Create text that when split by \n\n results in chunks < 50 chars each
        # This should trigger fallback chunking
        text = "A" * 25 + "\n\n" + "B" * 25  # Two small chunks that will be filtered out
        chunks = self.preprocessor.chunk_by_pages(text, "test_doc.pdf")
        assert len(chunks) > 0
        # Should use fallback chunking since page chunks were too short
        assert "test_doc_chunk_" in chunks[0]["id"]
    
    def test_process_pdf_success(self):
        """Test successful PDF processing."""
        with patch.object(self.preprocessor, 'extract_text_from_pdf') as mock_extract, \
             patch.object(self.preprocessor, 'detect_language') as mock_detect, \
             patch.object(self.preprocessor, 'chunk_by_pages') as mock_chunk:
            
            mock_extract.return_value = "Test document content"
            mock_detect.return_value = "it"
            mock_chunk.return_value = [
                {"id": "test_pagina_1", "testo": "Page 1 content"},
                {"id": "test_pagina_2", "testo": "Page 2 content"}
            ]
            
            result = self.preprocessor.process_pdf("test.pdf")
            assert result is not None
            assert result["file_sorgente"] == "test.pdf"
            assert result["metadati"]["lingua"] == "it"
            assert result["metadati"]["numero_pagine"] == 2
            assert len(result["chunks"]) == 2
    
    def test_process_pdf_empty_text(self):
        """Test PDF processing with empty text."""
        with patch.object(self.preprocessor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = ""
            result = self.preprocessor.process_pdf("test.pdf")
            assert result is None

if __name__ == "__main__":
    pytest.main([__file__])
