# GraphRAG Testing Framework

This directory contains the comprehensive testing framework for the GraphRAG project. All tests are designed to run without requiring actual API keys or external services by using mocking and dummy data.

> **Note on Neo4j Requirements**: While most tests use mocking to avoid external dependencies, some integration tests may require a running Neo4j instance for full functionality. See the main README.md for Docker setup instructions. If you encounter Neo4j connection errors during testing, ensure the Neo4j Docker container is running with the correct credentials configured in your `.env` file.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_advanced_rag_integration.py  # Advanced RAG integration tests
├── test_chatbot.py          # Tests for chatbot functionality
├── test_graph_nodes.py      # Tests for graph nodes
├── test_graph_state.py      # Tests for graph state
├── test_ingest.py           # Tests for ingestion module
├── test_new_tools.py        # Tests for new tools
├── test_pipeline_integration.py  # Pipeline integration tests
├── test_preprocess.py       # Tests for preprocessing
├── test_schema.py           # Tests for schema definitions
├── run_tests.py            # Test runner script
├── test_caching_mechanism.py # Caching mechanism tests
├── verify_integration.py    # Integration verification
└── README.md               # This file
```

## Running Tests

### Prerequisites

Make sure you have the test dependencies installed. The project uses Poetry for dependency management:

```bash
# Install test dependencies
poetry install --with test

# Or install all dependencies including tests
poetry install
```

### Running All Tests

```bash
# Run all tests
pytest tests/

# Run all tests with verbose output
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Running Specific Test Files

```bash
# Run schema tests
pytest tests/test_schema.py

# Run preprocessing tests
pytest tests/test_preprocess.py

# Run ingestion tests
pytest tests/test_ingest.py
```

## Test Design Philosophy

### Unit Testing Approach

1. **Mock External Dependencies**: All external services (OpenAI, Neo4j, Tavily) are mocked to ensure tests run without API keys
2. **Focus on Logic**: Tests verify the core logic and data transformations rather than external service responses
3. **Isolation**: Each test is independent and doesn't rely on external state
4. **Fast Execution**: Tests run quickly without network calls or file I/O

### Integration Testing

The `test_pipeline_integration.py` file contains integration tests that verify the components can be imported and initialized correctly. These tests use dummy API keys and don't make actual API calls.

## Test Coverage

### Core Components

- **Schema**: Entity and relationship definitions, prompt templates
- **Preprocessing**: PDF text extraction, language detection, chunking
- **Ingestion**: Knowledge graph extraction, GraphDocument conversion, storage
- **Chatbot**: Query routing, decomposition, search, answer generation
- **Graph Nodes**: Individual workflow components
- **Graph State**: State management for the LangGraph workflow

### Mocking Strategy

- **LLM Calls**: Mocked using `unittest.mock.patch`
- **External APIs**: Neo4j, OpenAI, and Tavily clients are mocked
- **File Operations**: File I/O is mocked using `mock_open`
- **Network Calls**: All network operations are mocked to prevent external dependencies

## Adding New Tests

### Test File Naming

Follow the pattern `test_<module_name>.py` for consistency.

### Using Fixtures

Common test data and configurations are provided via pytest fixtures in `conftest.py`:

```python
def test_my_function(sample_graph_state):
    # Use the sample_graph_state fixture
    result = my_function(sample_graph_state)
    assert result is not None
```

### Mocking Best Practices

1. **Patch at the Right Level**: Patch the module where the dependency is imported
2. **Use Context Managers**: Use `patch` as a context manager or decorator
3. **Verify Calls**: Use `assert_called_with()` to verify mock interactions
4. **Clean Up**: Ensure mocks don't leak between tests

## Continuous Integration

Tests are designed to be run in CI/CD pipelines without any external dependencies. All tests should pass with a clean installation of the project dependencies.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in the Python path via `conftest.py`
2. **Missing Mocks**: Add appropriate mocks for new external dependencies
3. **Test Isolation**: Ensure tests don't share state between runs

### Debugging Tests

```bash
# Run a single test with output
pytest tests/test_schema.py::TestSchemaDefinitions::test_entity_types_defined -v

# Run with debug output
pytest tests/ -s -v

# Run with Python debugger
pytest tests/ --pdb
