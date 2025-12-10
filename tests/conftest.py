"""Pytest configuration and fixtures for GraphRAG tests."""

import pytest
import os
import sys
import warnings
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress PyMuPDF/SWIG deprecation warnings from frozen importlib
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPyPacked.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPyObject.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="<frozen importlib._bootstrap>")

# Suppress SWIG builtin type warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type SwigPyPacked.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type SwigPyObject.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type swigvarlink.*")

# Mock environment variables for testing
os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
os.environ['NEO4J_USERNAME'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'password'
os.environ['OPENAI_API_KEY'] = 'test-key'
os.environ['INPUT_DOCUMENTS_PATH'] = 'input document'
os.environ['OUTPUT_JSON_PATH'] = 'output/json'

@pytest.fixture
def sample_graph_state():
    """Fixture providing a sample GraphState for testing."""
    from src.graph_state import GraphState
    return GraphState(question="Test question")

@pytest.fixture
def sample_knowledge_graph():
    """Fixture providing a sample KnowledgeGraph for testing."""
    from src.schema import KnowledgeGraph, Entity, Relationship
    return KnowledgeGraph(
        entities=[
            Entity(
                id="test_entity_1",
                type="Provvedimento",
                name="Test Document",
                description="A test legal document"
            )
        ],
        relationships=[
            Relationship(
                id="test_rel_1",
                source_entity_id="test_entity_1",
                target_entity_id="test_entity_2",
                type="CITA",
                description="Test entity cites another"
            )
        ]
    )

@pytest.fixture
def sample_document_data():
    """Fixture providing sample document data for testing."""
    return {
        "file_sorgente": "test_document.pdf",
        "metadati": {
            "lingua": "it",
            "numero_pagine": 2
        },
        "chunks": [
            {
                "id": "test_document_pagina_1",
                "testo": "Test content for page 1"
            },
            {
                "id": "test_document_pagina_2", 
                "testo": "Test content for page 2"
            }
        ]
    }
