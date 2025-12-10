"""Unit tests for the GraphRAG agent state module."""

import pytest
from src.graph_state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

class TestAgentState:
    """Test the AgentState class."""
    
    def test_agent_state_initialization(self):
        """Test that AgentState initializes correctly."""
        state = AgentState(messages=[HumanMessage(content="Test question")])
        assert isinstance(state, dict)  # TypedDict is a dict at runtime
        assert "messages" in state
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "Test question"
    
    def test_agent_state_with_multiple_messages(self):
        """Test AgentState with multiple messages."""
        messages = [
            HumanMessage(content="Test question"),
            AIMessage(content="Test response"),
            HumanMessage(content="Follow up")
        ]
        state = AgentState(messages=messages)
        
        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
        assert isinstance(state["messages"][2], HumanMessage)
        assert state["messages"][0].content == "Test question"
        assert state["messages"][1].content == "Test response"
        assert state["messages"][2].content == "Follow up"
    
    def test_agent_state_field_types(self):
        """Test that AgentState fields have correct types."""
        state = AgentState(messages=[HumanMessage(content="Test question")])
        assert isinstance(state["messages"], list)
        assert isinstance(state["messages"][0], HumanMessage)
    
    def test_agent_state_empty_messages(self):
        """Test AgentState with empty messages."""
        state = AgentState(messages=[])
        assert isinstance(state, dict)
        assert "messages" in state
        assert state["messages"] == []
    
    def test_agent_state_message_content(self):
        """Test that AgentState messages contain correct content."""
        messages = [
            HumanMessage(content="What is GDPR?"),
            AIMessage(content="GDPR stands for General Data Protection Regulation.")
        ]
        state = AgentState(messages=messages)
        
        assert state["messages"][0].content == "What is GDPR?"
        assert state["messages"][1].content == "GDPR stands for General Data Protection Regulation."

if __name__ == "__main__":
    pytest.main([__file__])
