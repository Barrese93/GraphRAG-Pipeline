"""Unit tests for the GraphRAG ingestion module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from src.ingest import GraphRAGIngestor
from src.schema import KnowledgeGraph, Entity, Relationship

class TestGraphRAGIngestor:
    """Test the GraphRAGIngestor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('src.ingest.ChatOpenAI'), \
             patch('src.ingest.Neo4jGraph'), \
             patch('src.ingest.get_extraction_prompt'), \
             patch('src.ingest.OpenAIEmbeddings'), \
             patch('src.ingest.Neo4jVector'):
            self.ingestor = GraphRAGIngestor(openai_api_key="test-key")
    
    def test_ingestor_initialization(self):
        """Test that the ingestor initializes correctly."""
        assert isinstance(self.ingestor, GraphRAGIngestor)
        assert hasattr(self.ingestor, 'llm')
        assert hasattr(self.ingestor, 'graph')
        assert hasattr(self.ingestor, 'extraction_chain')
    
    def test_extract_knowledge_graph_success(self):
        """Test successful knowledge graph extraction."""
        with patch.object(self.ingestor.extraction_chain, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "entities": [{"id": "test_1", "type": "Provvedimento", "name": "Test Doc", "description": "A test document", "publication_date": "2024-01-01", "document_type": "Provvedimento", "reference_number": "123"}],
                "relationships": []
            }
            
            result = self.ingestor.extract_knowledge_graph("Test text")
            assert isinstance(result, KnowledgeGraph)
            assert len(result.entities) == 1
            assert result.entities[0].name == "Test Doc"
            assert result.entities[0].publication_date == "2024-01-01"
    
    def test_extract_knowledge_graph_error(self):
        """Test knowledge graph extraction with error."""
        with patch.object(self.ingestor.extraction_chain, 'invoke') as mock_invoke:
            mock_invoke.side_effect = Exception("Extraction error")
            
            result = self.ingestor.extract_knowledge_graph("Test text")
            assert isinstance(result, KnowledgeGraph)
            assert len(result.entities) == 0
            assert len(result.relationships) == 0
    
    def test_convert_to_graph_document_success(self):
        """Test successful conversion to GraphDocument."""
        kg = KnowledgeGraph(
            entities=[
                Entity(id="entity_1", type="Provvedimento", name="Test Doc", description="A test document", publication_date="2024-01-01", document_type="Provvedimento", reference_number="123"),
                Entity(id="entity_2", type="FonteNormativa", name="Test Source", description="A test source")
            ],
            relationships=[
                Relationship(id="rel_1", source_entity_id="entity_1", target_entity_id="entity_2", type="CITA", description="Doc cites source")
            ]
        )
        
        result = self.ingestor.convert_to_graph_document(kg, "test_chunk_1", "test_document.pdf")
        assert result is not None
        assert len(result.nodes) == 2
        assert result.nodes[0].properties["publication_date"] == "2024-01-01"
        assert len(result.relationships) == 1
        assert result.relationships[0].source.id == "entity_1"
    
    def test_convert_to_graph_document_error(self):
        """Test conversion to GraphDocument with error."""
        with patch('langchain_community.graphs.graph_document.Node', side_effect=Exception("Node creation error")):
            kg = KnowledgeGraph(
                entities=[Entity(id="entity_1", type="Provvedimento", name="Test Doc")],
                relationships=[]
            )
            result = self.ingestor.convert_to_graph_document(kg, "test_chunk_1", "test_document.pdf")
            assert result is None
    
    def test_store_graph_and_embeddings_success(self):
        """Test successful storage of graph and embeddings."""
        with patch.object(self.ingestor.graph, 'add_graph_documents') as mock_add_graph, \
             patch.object(self.ingestor.vector_store, 'add_documents') as mock_add_docs:
            
            # Create a more realistic mock for GraphDocument
            mock_node = Mock()
            mock_node.properties = {"description": "test description"}
            
            # The GraphDocument needs to be an object with a 'nodes' attribute that is a list
            mock_graph_document = MagicMock()
            mock_graph_document.nodes = [mock_node]
            mock_graph_document.relationships = [] # Also mock relationships
            
            self.ingestor.store_graph_and_embeddings(mock_graph_document)
            
            mock_add_graph.assert_called_once_with([mock_graph_document])
            mock_add_docs.assert_called_once()

    def test_store_graph_and_embeddings_no_doc(self):
        """Test storage when no graph document is provided."""
        with patch.object(self.ingestor.graph, 'add_graph_documents') as mock_add_graph, \
             patch.object(self.ingestor.vector_store, 'add_documents') as mock_add_docs:
            
            self.ingestor.store_graph_and_embeddings(None)
            
            mock_add_graph.assert_not_called()
            mock_add_docs.assert_not_called()
    
    def test_process_chunk_success(self):
        """Test successful chunk processing."""
        chunk_data = {
            "id": "test_chunk_1",
            "testo": "Test chunk content"
        }
        
        with patch.object(self.ingestor, 'extract_knowledge_graph') as mock_extract, \
             patch.object(self.ingestor, 'convert_to_graph_document') as mock_convert, \
             patch.object(self.ingestor, 'store_graph_and_embeddings') as mock_store:
            
            mock_extract.return_value = KnowledgeGraph(entities=[], relationships=[])
            mock_convert.return_value = Mock()
            
            self.ingestor.process_chunk(chunk_data, "test_document.pdf")
            mock_extract.assert_called_once_with("Test chunk content")
            mock_convert.assert_called_once()
            mock_store.assert_called_once()
    
    def test_process_chunk_empty_text(self):
        """Test chunk processing with empty text."""
        chunk_data = {
            "id": "test_chunk_1",
            "testo": ""
        }
        
        with patch.object(self.ingestor, 'extract_knowledge_graph') as mock_extract:
            self.ingestor.process_chunk(chunk_data, "test_document.pdf")
            mock_extract.assert_not_called()
    
    def test_process_document_json_success(self):
        """Test successful document JSON processing."""
        test_data = {
            "file_sorgente": "test_document.pdf",
            "chunks": [
                {"id": "chunk_1", "testo": "Content 1"},
                {"id": "chunk_2", "testo": "Content 2"}
            ]
        }
        
        with patch('builtins.open', mock_open(read_data=str(test_data).replace("'", '"'))), \
             patch.object(self.ingestor, 'process_chunk') as mock_process:
            
            self.ingestor.process_document_json("test.json")
            assert mock_process.call_count == 2

if __name__ == "__main__":
    pytest.main([__file__])
