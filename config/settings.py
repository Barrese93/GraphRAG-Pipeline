import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Tavily API Key for web search
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')

# Processing Configuration
INPUT_DOCUMENTS_PATH = os.getenv('INPUT_DOCUMENTS_PATH', 'input document')
OUTPUT_JSON_PATH = os.getenv('OUTPUT_JSON_PATH', 'output/json')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
