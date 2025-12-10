"""Unit tests for the new RAG tools (rewrite_query_tool, grade_documents_tool, grade_answer_tool)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_nodes import rewrite_query_tool, grade_documents_tool, grade_answer_tool
from src.graders import GradeDocuments, GradeHallucinations, GradeAnswerUsefulness

class TestNewRAGTools:
    """Test the new RAG tools."""
    
    def test_rewrite_query_tool_invoke(self):
        """Test rewrite_query_tool invocation."""
        mock_rewriter_chain = Mock()
        mock_rewriter_chain.invoke.return_value = "Rewritten question about privacy regulations"
        
        result = rewrite_query_tool.invoke({
            "question": "What are the rules?", 
            "rewriter_chain": mock_rewriter_chain
        })
        assert result == "Rewritten question about privacy regulations"
    
    def test_grade_documents_tool_invoke(self):
        """Test grade_documents_tool invocation."""
        mock_relevance_grader = Mock()
        mock_grade = Mock()
        mock_grade.binary_score = "yes"
        mock_relevance_grader.invoke.return_value = mock_grade
        
        documents = [{"content": "Relevant document about privacy"}]
        
        result = grade_documents_tool.invoke({
            "documents": documents,
            "question": "Privacy question", 
            "relevance_grader": mock_relevance_grader
        })
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["content"] == "Relevant document about privacy"
    
    def test_grade_answer_tool_invoke(self):
        """Test grade_answer_tool invocation."""
        mock_hallucination_grader = Mock()
        mock_usefulness_grader = Mock()
        
        mock_hallucination_grade = Mock()
        mock_hallucination_grade.binary_score = "yes"
        mock_hallucination_grader.invoke.return_value = mock_hallucination_grade
        
        mock_usefulness_grade = Mock()
        mock_usefulness_grade.binary_score = "yes"
        mock_usefulness_grader.invoke.return_value = mock_usefulness_grade
        
        documents = [{"content": "Test document"}]
        
        result = grade_answer_tool.invoke({
            "generation": "Test answer",
            "documents": documents,
            "question": "Test question",
            "hallucination_grader": mock_hallucination_grader,
            "usefulness_grader": mock_usefulness_grader
        })
        assert result == "utile"

if __name__ == "__main__":
    pytest.main([__file__])
