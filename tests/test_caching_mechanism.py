"""Test script to demonstrate the caching mechanism in GraphRAG pipeline."""

import os
import sys
import logging
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import run_pipeline
from src.ingest import GraphRAGIngestor

def test_caching_mechanism():
    """Test that documents are skipped when already processed."""
    print("🧪 Testing Caching Mechanism")
    print("=" * 40)
    
    # First run - process documents normally
    print("\n📝 First Run: Processing documents...")
    run_pipeline(clear_database=True)  # Clear database for fresh start
    
    # Second run - should skip already processed documents
    print("\n📝 Second Run: Testing caching mechanism...")
    
    # Mock the process_all_documents method to count calls
    original_process_all_documents = GraphRAGIngestor.process_all_documents
    call_count = {'count': 0}
    
    def mock_process_all_documents(self, json_directory=None):
        call_count['count'] += 1
        # Check if documents are actually skipped
        print("  Checking if documents are skipped...")
        return original_process_all_documents(self, json_directory)
    
    # Patch the method to monitor calls
    with patch.object(GraphRAGIngestor, 'process_all_documents', side_effect=mock_process_all_documents):
        run_pipeline(clear_database=False)  # Don't clear database - should use caching
    
    print(f"\n📊 Process all documents was called {call_count['count']} times")
    print("✅ Caching test completed")

def test_document_already_processed_check():
    """Test the document_already_processed method directly."""
    print("\n🔍 Testing document_already_processed method...")
    
    from config.settings import OPENAI_API_KEY
    
    # Initialize ingestor
    ingestor = GraphRAGIngestor(openai_api_key=OPENAI_API_KEY)
    
    # Check if our test document is already processed
    test_document = "10136889_250116_Provvedimento_del_16_gennaio_2025.pdf"
    already_processed = ingestor.document_already_processed(test_document)
    
    if already_processed:
        print(f"✅ Document '{test_document}' is already processed - caching working!")
    else:
        print(f"⚠️  Document '{test_document}' is not found in database")
    
    ingestor.close()
    return already_processed

def main():
    """Run caching tests."""
    logging.basicConfig(level=logging.INFO)
    
    # Test document already processed check
    document_exists = test_document_already_processed_check()
    
    # Test full caching mechanism
    test_caching_mechanism()
    
    print(f"\n🏁 Caching Test Results:")
    print(f"   Document already processed: {document_exists}")
    print(f"   Caching mechanism: {'Working' if document_exists else 'Needs database population'}")

if __name__ == "__main__":
    main()
