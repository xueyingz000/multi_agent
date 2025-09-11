#!/usr/bin/env python3
"""
Example script demonstrating the use of .ifc and .json file formats
with the IFC Semantic Agent (No Knowledge Graph version).

This example shows how to:
1. Load IFC data from a standard .ifc file
2. Load regulatory text from a structured .json file
3. Process queries using the loaded data
"""

import sys
import os
from pathlib import Path
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG
from data_processing import IFCProcessor, TextProcessor
from utils import get_logger

def main():
    """Main example function."""
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("IFC Semantic Agent - New File Formats Example")
    logger.info("=" * 60)
    
    try:
        # Initialize the agent
        logger.info("Initializing IFC Semantic Agent (No KG version)...")
        agent = IFCSemanticAgentNoKG()
        
        # Initialize processors
        ifc_processor = IFCProcessor()
        text_processor = TextProcessor()
        
        # File paths
        ifc_file_path = "sample_data.ifc"
        regulations_file_path = "sample_regulations.json"
        
        # Check if files exist
        if not os.path.exists(ifc_file_path):
            logger.error(f"IFC file not found: {ifc_file_path}")
            logger.info("Please ensure sample_data.ifc exists in the project directory")
            return False
            
        if not os.path.exists(regulations_file_path):
            logger.error(f"Regulations file not found: {regulations_file_path}")
            logger.info("Please ensure sample_regulations.json exists in the project directory")
            return False
        
        # Load and process IFC data
        logger.info(f"Loading IFC data from: {ifc_file_path}")
        ifc_data = ifc_processor.process_file(ifc_file_path)
        logger.info(f"✓ Loaded IFC data with {len(ifc_data.get('entities', {}))} entities")
        
        # Display some IFC entities
        entities = ifc_data.get('entities', [])
        if entities:
            logger.info("Sample IFC entities:")
            for i, entity_data in enumerate(entities[:3]):
                entity_id = entity_data.get('global_id', f'Entity_{i}')
                entity_type = entity_data.get('type', 'Unknown')
                entity_name = entity_data.get('name', 'Unnamed')
                logger.info(f"  - {entity_id}: {entity_type} ({entity_name})")
        
        # Load and process regulatory data
        logger.info(f"Loading regulatory data from: {regulations_file_path}")
        with open(regulations_file_path, 'r', encoding='utf-8') as f:
            regulations_data = json.load(f)
        
        # Convert JSON to text format
        regulatory_text = text_processor.process_json_regulations(regulations_data)
        logger.info(f"✓ Processed regulatory text with {len(regulations_data.get('sections', []))} sections")
        
        # Display some regulatory sections
        sections = regulations_data.get('sections', [])
        if sections:
            logger.info("Sample regulatory sections:")
            for section in sections[:2]:
                section_id = section.get('section_id', '')
                title = section.get('title', '')
                req_count = len(section.get('requirements', []))
                logger.info(f"  - Section {section_id}: {title} ({req_count} requirements)")
        
        # Example queries
        queries = [
            "How do the IFC wall entities align with building code wall requirements?",
            "What are the thickness requirements for structural elements?",
            "Which IFC entities correspond to load-bearing elements in the regulations?"
        ]
        
        logger.info("\n" + "=" * 40)
        logger.info("Processing Example Queries")
        logger.info("=" * 40)
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\nQuery {i}: {query}")
            logger.info("-" * 50)
            
            try:
                # Process the query with loaded data
                response = agent.process_query(
                    query=query,
                    ifc_data=ifc_data,
                    regulatory_text=regulatory_text
                )
                
                logger.info(f"✓ Query processed successfully")
                logger.info(f"Final Answer: {response.final_answer}")
                logger.info(f"Confidence Score: {response.confidence_score:.2f}")
                logger.info(f"Processing Steps: {response.total_steps}")
                logger.info(f"Execution Time: {response.execution_time:.2f}s")
                
                if response.semantic_mappings:
                    logger.info(f"Semantic Mappings Found: {len(response.semantic_mappings)}")
                
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                continue
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Example completed successfully!")
        logger.info("\nKey Features Demonstrated:")
        logger.info("• Loading IFC data from standard .ifc files")
        logger.info("• Loading regulatory text from structured .json files")
        logger.info("• Semantic alignment between IFC entities and regulations")
        logger.info("• Multi-step reasoning with ReAct framework")
        logger.info("• Confidence scoring and validation")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)