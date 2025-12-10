# GraphRAG Performance Notes

## API Call Behavior Explanation

During the ingestion process, you may notice a high number of API calls to OpenAI. This is expected behavior due to the following reasons:

### 1. Chunk-level Processing
- Each PDF document is split into multiple text chunks for better processing
- Each chunk requires separate API calls for optimal extraction quality
- For each chunk, two API calls are made:
  - **Chat Completions API**: Extracts entities and relationships from the text
  - **Embeddings API**: Generates vector embeddings for semantic search capabilities

### 2. Document Processing Flow
```
PDF Document → Multiple Chunks → [Extraction + Embedding] per Chunk → Knowledge Graph
```

### 3. Why Multiple Calls Are Necessary
- **Accuracy**: Processing smaller chunks yields better entity/relationship extraction
- **Context Window**: Large documents exceed LLM context limits
- **Scalability**: Chunking allows parallel processing and better error handling
- **Vector Search**: Each entity needs embeddings for semantic search functionality

### 4. Normal API Call Count
For a single document with 3 chunks, you should expect:
- **3 Chat Completions API calls** (one per chunk for extraction)
- **3 Embeddings API calls** (one per chunk for vector generation)
- **Total: 6 API calls** for complete document processing

## Optimization Opportunities
For future improvements, consider:
- **Batch Processing**: Group multiple chunks into single API calls where possible
- **Caching**: Store results to avoid reprocessing identical chunks
- **Rate Limiting**: Implement queuing to manage API call frequency
- **Selective Processing**: Skip already processed documents

## Cost Management
- Monitor API usage through OpenAI dashboard
- Consider using smaller, more cost-effective models for embedding generation
- Implement preprocessing to filter out low-value content
- Use local embedding models for development/testing

## Performance Tips

### For Development
- Use smaller test documents during development
- Enable caching to avoid redundant API calls
- Monitor logs for duplicate processing patterns

### For Production
- Implement proper rate limiting
- Use environment variables for API keys
- Consider async processing for better throughput
- Monitor and log API usage for cost optimization

## Troubleshooting Duplicate Processing

If you notice the same documents being processed multiple times:
1. Check the `output/json` directory for duplicate JSON files (files with _v1, _v2, _v3 suffixes)
2. Clear the output directory before running the pipeline: `del output\json\*.json`
3. Verify that preprocessing isn't creating multiple versions of the same document
4. Ensure the ingestion process runs on a clean output directory

The system now includes **smart caching** - it checks Neo4j to see if a document has already been processed and skips it if found. This prevents redundant API calls while maintaining data consistency.

## Ingestion Behavior Explained

**Does re-running main.py cause re-ingestion?**
- **Default Behavior** - Smart caching enabled (`clear_database=False` by default)
- **Smart Skipping** - Documents already processed are automatically skipped to prevent redundant API calls
- **API Cost Savings** - No redundant OpenAI API calls for existing documents when database isn't cleared
- **Full Re-processing** - Use `run_pipeline(clear_database=True)` when you want to completely refresh the database

**Controlling Database Clearing:**
- **Incremental Processing**: `python src/main.py` (default - uses caching, no database clearing)
- **Full Re-processing**: `python src/main.py` with `clear_database=True` parameter or modify main.py

The normal behavior is **one JSON file per PDF document**, which results in appropriate API call counts based on the number of chunks in each document.
