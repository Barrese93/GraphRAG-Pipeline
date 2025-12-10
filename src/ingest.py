"""Ingestion pipeline for GraphRAG using LangChain extraction chains."""

import json
import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document

from config.settings import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OUTPUT_JSON_PATH, OPENAI_API_KEY
from src.schema import KnowledgeGraph, get_extraction_prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGIngestor:
    def __init__(self, openai_api_key: str = None):
        """Initialize the GraphRAG ingestor with LangChain components."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # or your preferred model
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        # Initialize Neo4j graph
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        
        # Create extraction chain using with_structured_output for better reliability
        self.extraction_prompt = get_extraction_prompt()
        self.extraction_chain = (
            {"text": RunnablePassthrough()}
            | self.extraction_prompt
            | self.llm.with_structured_output(KnowledgeGraph)
        )
        
        # Initialize Embeddings and Vector Store for ingestion
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = Neo4jVector(
            embedding=self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name="vector_index",
            keyword_index_name="keyword_index",
            search_type="hybrid",
            node_label="Entity",
            text_node_property="description",
            embedding_node_property="embedding"
        )

    def clear_database(self):
        """Clear all nodes and relationships from the Neo4j database."""
        try:
            logger.info("Clearing the Neo4j database...")
            self.graph.query("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully.")
        except Exception as e:
            logger.error(f"Error clearing the database: {str(e)}")
    
    def extract_knowledge_graph(self, text: str) -> KnowledgeGraph:
        """Extract knowledge graph from text using LangChain chain."""
        try:
            result = self.extraction_chain.invoke(text)
            # The result from the chain is a dictionary, so we instantiate the KnowledgeGraph object
            if isinstance(result, dict):
                return KnowledgeGraph(**result)
            # If it's already a KnowledgeGraph object, return it directly
            return result
        except Exception as e:
            logger.error(f"Error extracting knowledge graph: {str(e)}")
            # Return empty graph on error
            return KnowledgeGraph(entities=[], relationships=[])
    
    def convert_to_graph_document(self, kg: KnowledgeGraph, chunk_id: str, source_document: str) -> GraphDocument:
        """Convert KnowledgeGraph to LangChain GraphDocument format."""
        try:
            from langchain_core.documents import Document
            from langchain_community.graphs.graph_document import Node, Relationship
            
            # Convert entities to nodes
            nodes = []
            for entity in kg.entities:
                node = Node(
                    id=entity.id,
                    type=entity.type,
                    properties={
                        "name": entity.name,
                        "description": entity.description or "",
                        "publication_date": entity.publication_date,
                        "document_type": entity.document_type,
                        "reference_number": entity.reference_number,
                        "chunk_id": chunk_id,
                        "source_document": source_document
                    }
                )
                nodes.append(node)
            
            # Convert relationships
            relationships = []
            # Create a mapping of node IDs to Node objects for easy lookup
            node_map = {node.id: node for node in nodes}
            
            for rel in kg.relationships:
                # Get the source and target nodes
                source_node = node_map.get(rel.source_entity_id)
                target_node = node_map.get(rel.target_entity_id)
                
                if source_node and target_node:
                    relationship = Relationship(
                        source=source_node,
                        target=target_node,
                        type=rel.type,
                        properties={
                            "description": rel.description or "",
                            "chunk_id": chunk_id,
                            "source_document": source_document
                        }
                    )
                    relationships.append(relationship)
            
            # Create document for the chunk
            document = Document(
                page_content=f"Chunk {chunk_id} from {source_document}",
                metadata={
                    "chunk_id": chunk_id,
                    "source_document": source_document
                }
            )
            
            # Create GraphDocument
            graph_document = GraphDocument(
                nodes=nodes,
                relationships=relationships,
                source=document
            )
            
            return graph_document
            
        except Exception as e:
            logger.error(f"Error converting to GraphDocument: {str(e)}")
            return None
    
    def store_graph_and_embeddings(self, graph_document: GraphDocument):
        """Store the graph document and generate embeddings for the nodes."""
        try:
            if not graph_document:
                logger.warning("No graph document to store.")
                return

            # Create documents for vector store from the nodes' descriptions
            # We will embed the description of each entity
            documents_to_embed = [
                Document(page_content=node.properties.get("description", ""), metadata=node.properties)
                for node in graph_document.nodes if node.properties.get("description")
            ]

            # Store graph structure
            self.graph.add_graph_documents([graph_document])
            logger.info(f"Stored graph document with {len(graph_document.nodes)} nodes and {len(graph_document.relationships)} relationships.")

            # Add documents to the vector store, which also creates embeddings
            if documents_to_embed:
                self.vector_store.add_documents(documents_to_embed)
                logger.info(f"Generated and stored embeddings for {len(documents_to_embed)} nodes.")

        except Exception as e:
            logger.error(f"Error storing graph and embeddings: {str(e)}")
    
    def create_vector_indices(self):
        """Create vector indices in Neo4j for semantic search."""
        try:
            # Create vector index using the new CREATE VECTOR INDEX syntax (Neo4j 5.x+)
            self.graph.query("""
                CREATE VECTOR INDEX vector_index IF NOT EXISTS
                FOR (n:Entity)
                ON n.description
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            logger.info("Vector index created successfully")
        except Exception as e:
            logger.warning(f"Could not create vector index: {str(e)}")
    
    def create_keyword_indices(self):
        """Create keyword indices in Neo4j for full-text search."""
        try:
            self.graph.query("""
                CREATE FULLTEXT INDEX keyword_index IF NOT EXISTS FOR (n:Entity) ON EACH [n.description, n.name]
            """)
            logger.info("Keyword index created successfully")
        except Exception as e:
            logger.warning(f"Could not create keyword index: {str(e)}")
    
    def process_chunk(self, chunk_data: Dict[str, Any], source_document: str):
        """Process a single chunk through the extraction and storage pipeline."""
        chunk_id = chunk_data.get("id", "")
        text = chunk_data.get("testo", "")
        
        if not text:
            logger.warning(f"Empty text for chunk {chunk_id}")
            return
        
        logger.info(f"Processing chunk {chunk_id}")
        
        # Extract knowledge graph
        kg = self.extract_knowledge_graph(text)
        
        # Convert to GraphDocument
        graph_document = self.convert_to_graph_document(kg, chunk_id, source_document)
        
        # Store graph and generate embeddings
        self.store_graph_and_embeddings(graph_document)
    
    def document_already_processed(self, source_document: str) -> bool:
        """Check if a document has already been processed by querying Neo4j."""
        try:
            # Check if any nodes exist with this source document
            query = """
                MATCH (n:Entity {source_document: $source_document})
                RETURN count(n) as count
            """
            result = self.graph.query(query, {"source_document": source_document})
            return result[0]["count"] > 0
        except Exception as e:
            logger.warning(f"Error checking if document already processed: {str(e)}")
            return False
    
    def process_document_json(self, json_path: str):
        """Process a document JSON file through the pipeline."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            source_document = data.get("file_sorgente", "")
            chunks = data.get("chunks", [])
            
            # Check if document already processed
            if self.document_already_processed(source_document):
                logger.info(f"Document {source_document} already processed, skipping...")
                return
            
            logger.info(f"Processing document {source_document} with {len(chunks)} chunks")
            
            for chunk in chunks:
                self.process_chunk(chunk, source_document)
                
        except Exception as e:
            logger.error(f"Error processing document {json_path}: {str(e)}")
    
    def process_all_documents(self, json_directory: str = None):
        """Process all document JSON files in the directory."""
        if json_directory is None:
            json_directory = OUTPUT_JSON_PATH
        
        json_path = Path(json_directory)
        json_files = list(json_path.glob("*.json"))
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            self.process_document_json(str(json_file))
        
        # Create indices after processing all documents
        self.create_vector_indices()
        self.create_keyword_indices()
    
    def close(self):
        """Close the Neo4j connection."""
        if hasattr(self, 'graph') and self.graph:
            # Neo4jGraph handles connection cleanup automatically
            pass

def main():
    """Main function to run the ingestion pipeline."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import OPENAI_API_KEY
    
    # Initialize ingestor
    ingestor = GraphRAGIngestor(openai_api_key=OPENAI_API_KEY)
    
    try:
        # Process all documents
        ingestor.process_all_documents()
        logger.info("Ingestion pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {str(e)}")
    finally:
        ingestor.close()

if __name__ == "__main__":
    main()
