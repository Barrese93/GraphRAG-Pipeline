"""Main orchestration script for GraphRAG pipeline."""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import DocumentPreprocessor
from src.ingest import GraphRAGIngestor
from config.settings import INPUT_DOCUMENTS_PATH, OUTPUT_JSON_PATH, OPENAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(clear_database: bool = False):
    """Run the complete GraphRAG pipeline."""
    
    print("🚀 Starting GraphRAG Pipeline")
    print("=" * 50)
    
    # Step 1: Preprocessing
    print("\n📋 Step 1: Preprocessing Documents")
    print("-" * 30)
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_JSON_PATH, exist_ok=True)
        
        # Initialize preprocessor
        preprocessor = DocumentPreprocessor()
        
        # Process all PDF files
        pdf_files = [f for f in os.listdir(INPUT_DOCUMENTS_PATH) if f.endswith('.pdf')]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        print(f"Found {len(pdf_files)} PDF files to process")
        
        processed_files = 0
        for pdf_file in pdf_files:
            pdf_path = os.path.join(INPUT_DOCUMENTS_PATH, pdf_file)
            output_data = preprocessor.process_pdf(pdf_path)
            
            if output_data:
                # Save to JSON file
                output_filename = os.path.splitext(pdf_file)[0] + '.json'
                output_path = os.path.join(OUTPUT_JSON_PATH, output_filename)
                
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {output_path}")
                print(f"✅ Processed: {pdf_file}")
                processed_files += 1
            else:
                logger.error(f"Failed to process {pdf_file}")
                print(f"❌ Failed: {pdf_file}")
        
        print(f"\nPreprocessing completed: {processed_files}/{len(pdf_files)} files processed")
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        print(f"❌ Preprocessing failed: {str(e)}")
        return
    
    # Step 2: Ingestion
    print("\n📥 Step 2: Ingesting Documents into Knowledge Graph")
    print("-" * 40)
    
    try:
        # Initialize ingestor
        ingestor = GraphRAGIngestor(openai_api_key=OPENAI_API_KEY)
        
        # Clear the database before ingestion if requested
        if clear_database:
            ingestor.clear_database()
        
        # Process all documents
        ingestor.process_all_documents()
        
        print("✅ Ingestion completed successfully")
        ingestor.close()
        
    except Exception as e:
        logger.error(f"Error in ingestion: {str(e)}")
        print(f"❌ Ingestion failed: {str(e)}")
        return
    
    print("\n🎉 GraphRAG Pipeline completed successfully!")
    print("\nYou can now run the chatbot with:")
    print("python src/chatbot.py")

if __name__ == "__main__":
    run_pipeline()
