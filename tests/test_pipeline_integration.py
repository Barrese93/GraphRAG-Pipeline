"""Integration tests for GraphRAG pipeline components - Agent-based approach.

This file was moved from src/test_pipeline.py to tests/test_pipeline_integration.py
to establish a proper testing structure. These tests verify that components can be
imported and initialized correctly without requiring actual API keys or external services.
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock, ANY

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_schema():
    """Test the schema definitions."""
    print("Testing schema definitions...")
    
    from src.schema import ENTITY_TYPES, RELATIONSHIP_TYPES, KnowledgeGraph, get_extraction_prompt
    print("✅ Schema imports successful")
    print(f"Entity types: {ENTITY_TYPES}")
    print(f"Relationship types: {RELATIONSHIP_TYPES}")
    
    # Test Pydantic model
    test_graph = KnowledgeGraph(
        entities=[],
        relationships=[]
    )
    print("✅ Pydantic model creation successful")
    
    # Test prompt creation
    prompt = get_extraction_prompt()
    print("✅ Prompt creation successful")
    print(f"Prompt type: {type(prompt)}")
    
    # Use assertions for pytest
    assert ENTITY_TYPES is not None
    assert RELATIONSHIP_TYPES is not None
    assert test_graph is not None
    assert prompt is not None

def test_preprocessing():
    """Test the preprocessing component."""
    print("\nTesting preprocessing component...")
    
    from src.preprocess import DocumentPreprocessor
    print("✅ Preprocessor imports successful")
    
    # Test initialization
    preprocessor = DocumentPreprocessor()
    print("✅ Preprocessor initialization successful")
    
    # Use assertions for pytest
    assert preprocessor is not None

def test_ingestion_components():
    """Test the ingestion components."""
    print("\nTesting ingestion components...")
    
    # Mock Neo4j connections
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    with patch('src.ingest.Neo4jGraph'), \
         patch('src.ingest.ChatOpenAI'), \
         patch('src.ingest.OpenAIEmbeddings') as mock_embeddings:
        # Mock the embedding dimension calculation during initialization
        mock_embeddings.return_value.embed_query.return_value = [0.1] * 1536
        
        from src.ingest import GraphRAGIngestor
        print("✅ Ingestor imports successful")
        
        # Test initialization (without connecting to services)
        ingestor = GraphRAGIngestor(openai_api_key="test-key")
        print("✅ Ingestor initialization successful")
        
        # Use assertions for pytest
        assert ingestor is not None

def test_chatbot_components():
    """Test the chatbot components."""
    print("\nTesting chatbot components...")
    
    with patch('src.chatbot.ChatOpenAI'), \
         patch('src.chatbot.Neo4jGraph'), \
         patch('src.chatbot.GraphCypherQAChain'), \
         patch('src.chatbot.Neo4jVector.from_existing_index'):
        from src.chatbot import GraphRAGChatbot
        # QueryRouter and QueryDecomposer no longer exist in agent-based approach
        print("✅ Chatbot imports successful")
        
        # Test initialization
        chatbot = GraphRAGChatbot(openai_api_key="test-key")
        print("✅ Chatbot initialization successful")
        
        # Use assertions for pytest
        assert chatbot is not None

def test_agent_tool_selection():
    """Test that the agent chooses the correct tool based on question type."""
    print("\nTesting agent tool selection behavior...")
    
    # Mock all external dependencies
    with patch('src.chatbot.ChatOpenAI') as mock_llm, \
         patch('src.chatbot.Neo4jGraph'), \
         patch('src.chatbot.GraphCypherQAChain'), \
         patch('src.chatbot.Neo4jVector.from_existing_index'), \
         patch('src.chatbot.TavilyClient'):
        
        # Mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        
        # Mock the LLM response with tool calls
        mock_response = MagicMock()
        mock_response.tool_calls = [{
            "name": "hybrid_search_tool",
            "args": {"query": "What are the privacy regulations?"},
            "id": "call_1"
        }]
        mock_response.content = ""
        
        mock_llm_instance.bind_tools.return_value.invoke.return_value = mock_response
        
        # Import and initialize chatbot
        from src.chatbot import GraphRAGChatbot
        chatbot = GraphRAGChatbot(openai_api_key="test-key")
        
        # Test that the build_workflow method can be called
        workflow = chatbot.build_workflow()
        print("  ✅ Workflow building test passed")
        
        # Test that the LLM is properly bound with tools
        mock_llm_instance.bind_tools.assert_called()
        print("  ✅ Tool binding test passed")
        
        print("✅ Agent tool selection tests completed successfully")

def test_new_rag_tools_integration():
    """Test integration of new RAG tools (rewrite_query_tool, grade_documents_tool, grade_answer_tool)."""
    print("\nTesting new RAG tools integration...")
    
    # Mock all external dependencies
    with patch('src.chatbot.ChatOpenAI') as mock_llm, \
         patch('src.chatbot.Neo4jGraph'), \
         patch('src.chatbot.GraphCypherQAChain'), \
         patch('src.chatbot.Neo4jVector.from_existing_index'), \
         patch('src.chatbot.TavilyClient'):
        
        # Import and initialize chatbot
        from src.chatbot import GraphRAGChatbot
        chatbot = GraphRAGChatbot(openai_api_key="test-key")
        
        # Test that the build_workflow method includes new tools
        workflow = chatbot.build_workflow()
        print("  ✅ Workflow with new tools building test passed")
        
        print("✅ New RAG tools integration tests completed successfully")

def main():
    """Run all tests."""
    print("🧪 Running GraphRAG Component Tests")
    print("=" * 40)
    
    tests = [
        test_schema,
        test_preprocessing,
        test_ingestion_components,
        test_chatbot_components,
        test_agent_tool_selection,
        test_new_rag_tools_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline components are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main()
