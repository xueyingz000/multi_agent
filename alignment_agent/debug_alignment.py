#!/usr/bin/env python3
"""
Debug script for semantic alignment issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG
from utils.logger import get_logger

logger = get_logger(__name__)

def debug_alignment():
    """Debug semantic alignment process"""
    print("\n" + "="*60)
    print("DEBUG: Semantic Alignment Analysis")
    print("="*60)
    
    try:
        # Initialize agent
        print("\n1. Initializing agent...")
        agent = IFCSemanticAgentNoKG()
        print("   - Agent initialized successfully")
        
        # Load sample data
        ifc_file = "sample_data.ifc"
        json_file = "output-1.json"
        
        if not os.path.exists(ifc_file):
            print(f"   - Error: {ifc_file} not found")
            return
        if not os.path.exists(json_file):
            print(f"   - Error: {json_file} not found")
            return
            
        print(f"\n2. Loading IFC file: {ifc_file}")
        ifc_data = agent.ifc_processor.process_ifc_file(ifc_file)
        print(f"   - IFC data type: {type(ifc_data)}")
        print(f"   - IFC data keys: {ifc_data.keys() if isinstance(ifc_data, dict) else 'Not a dict'}")
        
        entities = ifc_data.get('entities', []) if isinstance(ifc_data, dict) else ifc_data
        print(f"   - IFC entities found: {len(entities) if hasattr(entities, '__len__') else 'Unknown'}")
        if entities and hasattr(entities, '__iter__'):
            print("   - Sample IFC entities:")
            entity_list = list(entities)[:3] if hasattr(entities, '__iter__') else []
            for i, entity in enumerate(entity_list):
                if hasattr(entity, 'entity_type'):
                    print(f"     [{i+1}] Type: {entity.entity_type}")
                    print(f"         Name: {getattr(entity, 'name', 'N/A')}")
                    print(f"         Context: {getattr(entity, 'context', 'N/A')}...")
                else:
                    print(f"     [{i+1}] Entity: {entity}")
        
        print(f"\n3. Loading JSON file: {json_file}")
        reg_entities = agent.text_processor._process_json_regulatory_text(json_file)
        print(f"   - Regulatory sections: {len(reg_entities.get('regulations', []))}")
        print(f"   - Regulatory entities: {len(reg_entities.get('entities', []))}")
        
        reg_entities_list = reg_entities.get('entities', [])
        if reg_entities_list:
            print("   - Sample regulatory entities:")
            for i, entity in enumerate(reg_entities_list[:3]):
                print(f"     [{i+1}] Type: {type(entity)}")
                print(f"         Content: {entity}...")
        
        print("\n4. Testing semantic alignment")
         if entities and reg_entities_list:
             # Test semantic alignment directly with updated config
             # Force lower thresholds for debugging
             agent.semantic_alignment.similarity_threshold = 0.2
             agent.semantic_alignment.confidence_threshold = 0.3
             entity_list = list(entities) if hasattr(entities, '__iter__') else []
             alignments = agent.semantic_alignment.align_entities(
                 ifc_entities=entity_list[:5],  # Test with first 5 entities
                 regulatory_entities=reg_entities_list[:50]  # Test with first 50 entities
             )
            print(f"   - Alignments found: {len(alignments)}")
            
            # Debug similarity calculation if no alignments found
            if len(alignments) == 0:
                print("   - No alignments found. Debugging similarity calculations...")
                print("\n5. Debug similarity calculation:")
                
                if entities and reg_entities_list:
                    ifc_entity = entities[0]
                    reg_entity = reg_entities_list[0]
                    
                    print(f"   IFC Entity: {ifc_entity.entity_type} - {ifc_entity.name}")
                    reg_text = reg_entity.get('text', '') if isinstance(reg_entity, dict) else reg_entity.text
                    print(f"   REG Entity: {reg_text}")
                    
                    # Calculate similarity manually
                    similarity = agent.semantic_alignment._calculate_semantic_similarity(
                        ifc_entity.name, reg_text
                    )
                    print(f"   Semantic similarity: {similarity:.3f}")
                    print(f"   Similarity threshold: {agent.semantic_alignment.similarity_threshold}")
                    print(f"   Confidence threshold: {agent.semantic_alignment.confidence_threshold}")
                    
                    # Check predefined mappings
                    predefined_mappings = agent.semantic_alignment.predefined_mappings
                    if ifc_entity.entity_type in predefined_mappings:
                        mapping_info = predefined_mappings[ifc_entity.entity_type]
                        print(f"   Predefined mapping found for {ifc_entity.entity_type}:")
                        print(f"   Regulatory terms: {mapping_info['regulatory_terms']}")
                    else:
                        print(f"   No predefined mapping for {ifc_entity.entity_type}")
                
                # Check if any regulatory entities contain building-related terms
                print("\n6. Searching for building-related terms in regulatory entities:")
                building_terms = ['wall', 'floor', 'slab', 'building', 'storey', 'structure', '墙', '楼板', '建筑', '层']
                found_matches = []
                for i, reg_ent in enumerate(reg_entities_list[:50]):  # Check first 50
                    reg_text = reg_ent.get('text', '') if isinstance(reg_ent, dict) else str(reg_ent)
                    for term in building_terms:
                        if term.lower() in reg_text.lower():
                            found_matches.append((i, reg_text, term))
                            if len(found_matches) >= 5:  # Limit output
                                break
                    if len(found_matches) >= 5:
                        break
                
                if found_matches:
                    print("   Found potential matches:")
                    for idx, text, term in found_matches:
                        print(f"     [{idx}] '{text}' (contains '{term}')")
                else:
                    print("   No building-related terms found in regulatory entities")
        else:
            print("   - No entities to align!")
            if not entities:
                print("     - No IFC entities found")
            if not reg_entities_list:
                print("     - No regulatory entities found")
        
    except Exception as e:
        print(f"   - Error during debugging: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("DEBUG: Analysis Complete")
    print("="*60)

if __name__ == "__main__":
    debug_alignment()