"""Unit tests for the GraphRAG graph nodes module - Tool-based approach."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the raw functions for testing
import src.graph_nodes as graph_nodes_module
from src.graph_nodes import (
    create_hybrid_search_tool, create_structured_query_tool, create_web_search_tool,
    create_query_rewriter_chain, create_metadata_filter_tool
)
from src.graders import (
    create_relevance_grader, create_hallucination_grader, create_answer_usefulness_grader,
    GradeDocuments, GradeHallucinations, GradeAnswerUsefulness
)
from langchain_core.messages import HumanMessage, AIMessage

class TestGraphToolFunctions:
    """Test the graph tool functions."""
    
    def test_hybrid_search_tool_success(self):
        """Test successful hybrid search tool execution."""
        mock_vector_store = Mock()
        mock_docs = [Mock()]
        mock_docs[0].page_content = "Test content"
        mock_docs[0].metadata = {"source": "test"}
        mock_vector_store.similarity_search.return_value = mock_docs
        
        # Create the tool with dependencies
        hybrid_search_tool = create_hybrid_search_tool(vector_store=mock_vector_store)
        result = hybrid_search_tool.invoke({"query": "Test question"})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["content"] == "Test content"
        assert result[0]["metadata"]["source"] == "test"
    
    def test_hybrid_search_tool_no_vector_store(self):
        """Test hybrid search tool with no vector store."""
        hybrid_search_tool = create_hybrid_search_tool(vector_store=None)
        result = hybrid_search_tool.invoke({"query": "Test question"})
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_hybrid_search_tool_with_graph_context(self):
        """Test hybrid search tool with graph context enhancement."""
        mock_vector_store = Mock()
        mock_docs = [Mock()]
        mock_docs[0].page_content = "Test content"
        mock_docs[0].metadata = {"id": "test_doc"}
        mock_vector_store.similarity_search.return_value = mock_docs
        
        mock_graph = Mock()
        mock_graph.query.return_value = [{
            "document_name": "Test Document",
            "related_entities": ["Entity1"],
            "entity_types": ["Type1"]
        }]
        
        hybrid_search_tool = create_hybrid_search_tool(vector_store=mock_vector_store, graph=mock_graph)
        result = hybrid_search_tool.invoke({"query": "Test question"})
        assert isinstance(result, list)
        assert len(result) == 1
        assert "related_context" in result[0]
    
    def test_hybrid_search_tool_error_handling(self):
        """Test hybrid search tool error handling."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.side_effect = Exception("Search error")
        
        hybrid_search_tool = create_hybrid_search_tool(vector_store=mock_vector_store)
        result = hybrid_search_tool("Test question")
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_structured_query_tool_success(self):
        """Test successful structured query tool execution."""
        mock_graph_qa_chain = Mock()
        mock_graph_qa_chain.invoke.return_value = {"result": "Test result"}
        
        structured_query_tool = create_structured_query_tool(graph_qa_chain=mock_graph_qa_chain)
        result = structured_query_tool("Test question")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["question"] == "Test question"
        assert "result" in result[0]
    
    def test_structured_query_tool_fallback(self):
        """Test structured query tool with fallback to simple search."""
        mock_graph_qa_chain = Mock()
        mock_graph_qa_chain.invoke.side_effect = Exception("Chain error")
        mock_graph = Mock()
        mock_graph.query.return_value = [{"name": "Test", "type": "Entity", "description": "Test entity"}]
        
        structured_query_tool = create_structured_query_tool(graph_qa_chain=mock_graph_qa_chain, graph=mock_graph)
        result = structured_query_tool("Test question")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "name" in result[0]
        assert "type" in result[0]
        assert "description" in result[0]
    
    def test_structured_query_tool_no_dependencies(self):
        """Test structured query tool with no dependencies."""
        structured_query_tool = create_structured_query_tool(graph_qa_chain=None, graph=None)
        result = structured_query_tool("Test question")
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_web_search_tool_success(self):
        """Test successful web search tool execution."""
        mock_tavily_client = Mock()
        mock_tavily_client.search.return_value = {
            "results": [
                {"content": "Test result", "url": "http://test.com", "title": "Test"}
            ]
        }
        
        web_search_tool = create_web_search_tool(tavily_client=mock_tavily_client)
        result = web_search_tool("Test question")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["content"] == "Test result"
        assert result[0]["url"] == "http://test.com"
        assert result[0]["title"] == "Test"
    
    def test_web_search_tool_no_client(self):
        """Test web search tool with no client."""
        web_search_tool = create_web_search_tool(tavily_client=None)
        result = web_search_tool("Test question")
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_web_search_tool_error_handling(self):
        """Test web search tool error handling."""
        mock_tavily_client = Mock()
        mock_tavily_client.search.side_effect = Exception("Search error")
        
        web_search_tool = create_web_search_tool(tavily_client=mock_tavily_client)
        result = web_search_tool("Test question")
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_metadata_filter_tool_success(self):
        """Test successful metadata filter tool execution."""
        mock_vector_store = Mock()
        mock_docs = [Mock()]
        mock_docs[0].page_content = "Filtered content"
        mock_docs[0].metadata = {"document_type": "Provvedimento", "year": "2024"}
        mock_vector_store.similarity_search.return_value = mock_docs
        
        metadata_filter_tool = create_metadata_filter_tool(vector_store=mock_vector_store)
        result = metadata_filter_tool.invoke({"query": "Test question", "filter_dict": {"document_type": "Provvedimento"}})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["content"] == "Filtered content"
        assert result[0]["metadata"]["document_type"] == "Provvedimento"
    
    def test_metadata_filter_tool_no_vector_store(self):
        """Test metadata filter tool with no vector store."""
        metadata_filter_tool = create_metadata_filter_tool(vector_store=None)
        result = metadata_filter_tool.invoke({"query": "Test question", "filter_dict": {"document_type": "Provvedimento"}})
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_metadata_filter_tool_error_handling(self):
        """Test metadata filter tool error handling."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.side_effect = Exception("Filter error")
        
        metadata_filter_tool = create_metadata_filter_tool(vector_store=mock_vector_store)
        result = metadata_filter_tool.invoke({"query": "Test question", "filter_dict": {"document_type": "Provvedimento"}})
        assert isinstance(result, list)
        assert len(result) == 0
    

if __name__ == "__main__":
    pytest.main([__file__])
