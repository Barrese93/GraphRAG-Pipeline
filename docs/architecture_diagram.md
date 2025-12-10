# GraphRAG Architecture Diagram

```mermaid
graph TD
    A[User Input - Questions] --> B[GraphRAG Chatbot]
    B --> C[LangGraph Agent]
    
    subgraph "Agent Workflow"
        C --> D[Tool Selection]
        D --> E{Tool Decision}
        E -->|Hybrid Search| F[Hybrid Search Tool]
        E -->|Structured Query| G[Structured Query Tool]
        E -->|Web Search| H[Web Search Tool]
        E -->|Query Rewrite| I[Query Rewrite Tool]
        E -->|Document Grading| J[Document Grading Tool]
        E -->|Answer Grading| K[Answer Grading Tool]
        E -->|Metadata Filter| L[Metadata Filter Tool]
    end
    
    subgraph "Data Processing Pipeline"
        M[PDF Documents] --> N[Preprocessing]
        N --> O[Chunking]
        O --> P[Knowledge Graph Extraction]
        P --> Q[Graph Storage - Neo4j]
        P --> R[Vector Embeddings]
        Q --> S[Graph Indices]
        R --> T[Vector Indices]
    end
    
    subgraph "External Services"
        U[OpenAI API]
        V[Tavily Search API]
        W[Neo4j Database]
    end
    
    F --> W
    G --> W
    H --> V
    I --> U
    J --> U
    K --> U
    L --> W
    P --> U
    C --> U
    
    W --> B
    S --> F
    T --> F
    
    subgraph "Output"
        X[Formatted Answers] --> B
    end
    
    B --> X
    
    style B fill:#4CAF50,stroke:#388E3C
    style C fill:#2196F3,stroke:#0D47A1
    style W fill:#FF9800,stroke:#E65100
    style U fill:#9C27B0,stroke:#4A148C
    style V fill:#9C27B0,stroke:#4A148C
    
    classDef process fill:#2196F3,stroke:#0D47A1,color:white;
    classDef data fill:#FF9800,stroke:#E65100,color:white;
    classDef external fill:#9C27B0,stroke:#4A148C,color:white;
    classDef output fill:#4CAF50,stroke:#388E3C,color:white;
```

## Architecture Components

### 1. Data Ingestion Pipeline
- **Preprocessing**: Extracts clean text from PDF documents
- **Chunking**: Creates manageable text chunks for processing
- **Knowledge Graph Extraction**: Uses LangChain structured extraction to convert text to graph format
- **Storage**: Stores both graph structure and vector embeddings in Neo4j

### 2. Query Processing Layer
- **LangGraph Agent**: Intelligent agent that reasons about query type and selects appropriate tools
- **Tool Ecosystem**: Specialized tools for different query patterns and validation

### 3. Retrieval System
- **Hybrid Search**: Combines vector and keyword search for comprehensive results
- **Structured Queries**: Direct Cypher queries for precise graph traversal
- **Web Search**: External search for current information

### 4. Quality Control
- **Adaptive RAG**: Query rewriting for optimization
- **CRAG**: Document relevance grading
- **Self-RAG**: Answer quality and hallucination detection

### 5. External Services
- **OpenAI**: LLM processing for extraction and reasoning
- **Tavily**: Web search capabilities
- **Neo4j**: Graph database with vector search capabilities
