"""Unit tests for the GraphRAG chatbot module - Agent-based approach."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import GraphRAGChatbot

class TestGraphRAGChatbot:
    """Test the GraphRAGChatbot class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('src.chatbot.ChatOpenAI'), \
             patch('src.chatbot.Neo4jGraph'), \
             patch('src.chatbot.GraphCypherQAChain'), \
             patch('src.chatbot.Neo4jVector.from_existing_index'):
            self.chatbot = GraphRAGChatbot(openai_api_key="test-key")
    
    def test_chatbot_initialization(self):
        """Test that the chatbot initializes correctly."""
        assert isinstance(self.chatbot, GraphRAGChatbot)
        assert hasattr(self.chatbot, 'llm')
        assert hasattr(self.chatbot, 'graph')
        # Router and decomposer no longer exist in agent-based approach
        assert not hasattr(self.chatbot, 'query_router')
        assert not hasattr(self.chatbot, 'query_decomposer')
    
    def test_build_workflow(self):
        """Test workflow building."""
        with patch('src.chatbot.StateGraph') as mock_state_graph:
            mock_workflow = Mock()
            mock_state_graph.return_value = mock_workflow
            
            compiled_app = Mock()
            mock_workflow.compile.return_value = compiled_app
            
            app = self.chatbot.build_workflow()
            assert app is not None
            # Should have agent and tools nodes
            assert mock_workflow.add_node.call_count >= 2
            # Check that we have the expected edges (at least entry point and one edge)
            assert mock_workflow.set_entry_point.call_count >= 1
            assert mock_workflow.add_conditional_edges.call_count >= 1
    
    def test_ask_success(self):
        """Test successful question answering."""
        with patch.object(self.chatbot, 'build_workflow') as mock_build:
            mock_app = Mock()
            mock_app.invoke.return_value = {
                "messages": [AIMessage(content="Test answer")]
            }
            mock_build.return_value = mock_app
            
            result = self.chatbot.ask("Test question")
            assert result == "Test answer"
    
    def test_ask_success_with_dict_message(self):
        """Test successful question answering with dict message."""
        with patch.object(self.chatbot, 'build_workflow') as mock_build:
            mock_app = Mock()
            mock_app.invoke.return_value = {
                "messages": [{"content": "Test answer"}]
            }
            mock_build.return_value = mock_app
            
            result = self.chatbot.ask("Test question")
            assert result == "Test answer"
    
    def test_ask_empty_messages(self):
        """Test question answering with empty messages."""
        with patch.object(self.chatbot, 'build_workflow') as mock_build:
            mock_app = Mock()
            mock_app.invoke.return_value = {
                "messages": []
            }
            mock_build.return_value = mock_app
            
            result = self.chatbot.ask("Test question")
            assert "dispiace" in result.lower()
    
    def test_ask_error(self):
        """Test question answering with error."""
        with patch.object(self.chatbot, 'build_workflow') as mock_build:
            mock_build.side_effect = Exception("Workflow error")
            
            result = self.chatbot.ask("Test question")
            assert "errore" in result.lower()
    
    def test_ask_with_tool_calls(self):
        """Test question answering with tool calls in the workflow."""
        with patch.object(self.chatbot, 'build_workflow') as mock_build:
            mock_app = Mock()
            # Simulate a conversation with tool calls and final answer
            mock_app.invoke.return_value = {
                "messages": [
                    HumanMessage(content="Test question"),
                    AIMessage(content="Final answer based on tool results")
                ]
            }
            mock_build.return_value = mock_app
            
            result = self.chatbot.ask("Test question")
            assert result == "Final answer based on tool results"

if __name__ == "__main__":
    pytest.main([__file__])
