# GraphRAG: Advanced Knowledge Graph System with LangGraph Agents

## 📊 Overview

This project implements a state-of-the-art knowledge graph construction and querying pipeline using a hybrid architecture that combines the best of **LangChain** for structured extraction and **LangGraph** for advanced, adaptive querying. The system is designed to ingest a large corpus of PDF documents and create a sophisticated GraphRAG system capable of intelligent question routing and hybrid reasoning.

## 🏗️ System Architecture

### Complete Architecture Diagram

![Architecture Diagram](docs/architecture_diagram.md)

### Core Architecture Components

The system is built around three core phases that work together to create a powerful knowledge graph system:

### Phase 1: Lightweight Preprocessing (`src/preprocess.py`)
- Extracts clean text from PDFs using PyMuPDF
- Performs language detection
- Creates manageable text chunks for downstream processing
- Outputs structured JSON for ingestion

### Phase 2: Controlled Ingestion with LangChain (`src/ingest.py`)
- Uses LangChain's structured output capabilities for accurate entity and relationship extraction
- Leverages Pydantic models defined in `src/schema.py` to enforce schema compliance
- Stores extracted knowledge graphs in Neo4j using LangChain's Neo4j integration
- Creates vector indices for semantic search capabilities

### Phase 3: Advanced Querying with LangGraph Agent (`src/chatbot.py`)
- Implements an intelligent agent using LangGraph's ReAct (Reason-Act) pattern
- Features tool-calling capabilities where the agent autonomously selects appropriate tools
- Supports hybrid reasoning combining semantic search, structured graph queries, and web search
- Includes CRAG (Corrective RAG) and Adaptive RAG capabilities for improved accuracy

## 🚀 Key Features

### 🤖 Intelligent Agent Architecture
- **LangGraph Self-Reflective RAG**: Advanced agent with reasoning and self-evaluation capabilities
- **Tool-Calling Ecosystem**: 7 specialized tools for different query patterns
- **Dynamic Workflow**: Adaptive decision-making based on query type

### 🛠️ Advanced RAG Capabilities
- **Adaptive RAG**: Rewrites queries for better retrieval using query rewriting tool
- **CRAG (Corrective RAG)**: Grades retrieved documents for relevance using document grading tool
- **Self-RAG**: Evaluates answer quality and detects hallucinations using answer grading tool
- **Query Decomposition**: Breaks complex questions into simpler sub-queries
- **Hybrid Search**: Combines vector similarity search with keyword-based search for comprehensive retrieval
- **Metadata Filtering**: Filters search results based on document metadata properties

### 🔧 Scalable & Modular Design
- **Modular Components**: Independent development and testing of each component
- **Stateful Workflow**: Context-aware conversation management
- **Vector Search Integration**: Semantic understanding capabilities
- **Extensible Tool System**: Easy addition of new capabilities

## 📋 Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- Docker (for running Neo4j locally)
- OpenAI API key
- Tavily API key (for web search functionality)

## 📈 Performance Notes

For detailed information about API call behavior, optimization opportunities, and caching mechanisms, see [PERFORMANCE_NOTES.md](docs/PERFORMANCE_NOTES.md).

## ⚙️ Setup & Configuration

### 1. Install Dependencies

```bash
poetry install
```

### 2. Environment Configuration

Create a `.env` file in the project root with the following variables:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Tavily Configuration (for web search)
TAVILY_API_KEY=your_tavily_api_key

# Processing Configuration
INPUT_DOCUMENTS_PATH=input document
OUTPUT_JSON_PATH=output/json
```

### 3. Start Neo4j with Docker

You can run a local Neo4j instance with the APOC plugin using Docker:

```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS='["apoc"]'  \
    neo4j:latest
```

Access the Neo4j Browser at `http://localhost:7474` (default credentials: neo4j/password).

## 🎯 Usage

### 1. Run the Full Ingestion Pipeline

To run the complete pipeline, which includes cleaning the database, preprocessing local documents, and ingesting the data into Neo4j (including embedding generation), execute:

```bash
poetry run python src/main.py
```

This single command handles all the necessary steps to get your knowledge graph ready.

### 2. Query with the Chatbot

```bash
poetry run python src/chatbot.py
```

Starts an interactive chatbot session that supports:
- Intelligent question routing (semantic, structured, or hybrid)
- Contextual document grading (CRAG)
- Adaptive web search (Adaptive RAG)
- Complex query decomposition

## 🧪 Testing

Run the integration tests to verify component functionality:

```bash
poetry run python tests/test_pipeline_integration.py
```

For more comprehensive testing:

```bash
poetry run pytest tests/
```

## 🛠️ Development

### Adding New Entity Types

1. Update `ENTITY_TYPES` and `RELATIONSHIP_TYPES` in `src/schema.py`
2. Modify the Pydantic models to reflect new schema requirements
3. Update prompts in `get_extraction_prompt()` if needed

### Customizing the Chatbot Workflow

The LangGraph workflow in `src/chatbot.py` can be customized by:
1. Adding new nodes in `src/graph_nodes.py`
2. Modifying the workflow edges in the `build_workflow()` method
3. Updating the `GraphState` definition in `src/graph_state.py`

## 🆘 Troubleshooting

### Neo4j Connection Issues
- Ensure the Docker container is running: `docker ps`
- Verify Neo4j credentials in `.env` file
- Check if ports 7474 and 7687 are available

### No Entities Extracted
- Check OpenAI API key configuration
- Verify document quality and language
- Review LLM response logs in the ingestion process

### Vector Search Not Working
- Ensure the ingestion process completed successfully
- Verify that vector indices were created in Neo4j

## 🚀 Next Steps & Roadmap

See [ROADMAP.md](docs/ROADMAP.md) for detailed future improvements and development plans.
