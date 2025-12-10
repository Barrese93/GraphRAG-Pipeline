"""Unit tests for the GraphRAG schema definitions."""

import pytest
from src.schema import (
    ENTITY_TYPES, RELATIONSHIP_TYPES, DOCUMENT_TYPES,
    Entity, Relationship, KnowledgeGraph,
    get_extraction_prompt, get_system_prompt
)

class TestSchemaDefinitions:
    """Test the schema constants and data structures."""
    
    def test_entity_types_defined(self):
        """Test that entity types are properly defined."""
        assert isinstance(ENTITY_TYPES, list)
        assert len(ENTITY_TYPES) > 0
        assert 'Provvedimento' in ENTITY_TYPES
        assert 'FonteNormativa' in ENTITY_TYPES
        assert 'Persona' in ENTITY_TYPES
    
    def test_relationship_types_defined(self):
        """Test that relationship types are properly defined."""
        assert isinstance(RELATIONSHIP_TYPES, list)
        assert len(RELATIONSHIP_TYPES) > 0
        assert 'CITA' in RELATIONSHIP_TYPES
        assert 'FIRMA' in RELATIONSHIP_TYPES
        assert 'EMESSO_DA' in RELATIONSHIP_TYPES
    
    def test_document_types_defined(self):
        """Test that document types are properly defined."""
        assert isinstance(DOCUMENT_TYPES, list)
        assert len(DOCUMENT_TYPES) > 0
        assert 'Deliberazione' in DOCUMENT_TYPES
        assert 'Parere' in DOCUMENT_TYPES
    
    def test_entity_model_creation(self):
        """Test Entity model creation."""
        entity = Entity(
            id="test_entity_1",
            type="Provvedimento",
            name="Test Document",
            description="A test legal document"
        )
        assert entity.id == "test_entity_1"
        assert entity.type == "Provvedimento"
        assert entity.name == "Test Document"
        assert entity.description == "A test legal document"
    
    def test_relationship_model_creation(self):
        """Test Relationship model creation."""
        relationship = Relationship(
            id="test_rel_1",
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            type="CITA",
            description="Entity 1 cites Entity 2"
        )
        assert relationship.id == "test_rel_1"
        assert relationship.source_entity_id == "entity_1"
        assert relationship.target_entity_id == "entity_2"
        assert relationship.type == "CITA"
        assert relationship.description == "Entity 1 cites Entity 2"
    
    def test_knowledge_graph_model_creation(self):
        """Test KnowledgeGraph model creation."""
        entity = Entity(
            id="test_entity_1",
            type="Provvedimento",
            name="Test Document",
            description="A test legal document"
        )
        relationship = Relationship(
            id="test_rel_1",
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            type="CITA",
            description="Entity 1 cites Entity 2"
        )
        
        kg = KnowledgeGraph(
            entities=[entity],
            relationships=[relationship]
        )
        assert len(kg.entities) == 1
        assert len(kg.relationships) == 1
        assert kg.entities[0].id == "test_entity_1"
        assert kg.relationships[0].id == "test_rel_1"
    
    def test_get_extraction_prompt_returns_prompt(self):
        """Test that get_extraction_prompt returns a ChatPromptTemplate."""
        from langchain_core.prompts import ChatPromptTemplate
        prompt = get_extraction_prompt()
        assert isinstance(prompt, ChatPromptTemplate)
        assert len(prompt.messages) > 0
    
    def test_get_system_prompt_returns_string(self):
        """Test that get_system_prompt returns a string."""
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Provvedimento" in prompt

if __name__ == "__main__":
    pytest.main([__file__])
