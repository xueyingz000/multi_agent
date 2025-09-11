#!/usr/bin/env python3
"""
Simple debug script for semantic alignment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG

def debug_simple():
    print("\n=== Simple Alignment Debug ===")
    
    try:
        # Initialize agent
        agent = IFCSemanticAgentNoKG()
        
        # Process files
        ifc_data = agent.ifc_processor.process_ifc_file("sample_data.ifc")
        reg_data = agent.text_processor._process_json_regulatory_text("output-1.json")
        
        print(f"IFC data type: {type(ifc_data)}")
        if isinstance(ifc_data, dict):
            entities = ifc_data.get('entities', [])
        else:
            entities = list(ifc_data) if hasattr(ifc_data, '__iter__') else []
        
        reg_entities = reg_data.get('entities', [])
        
        print(f"IFC entities: {len(entities)}")
        print(f"Regulatory entities: {len(reg_entities)}")
        
        if entities and reg_entities:
            # Set low thresholds
            agent.semantic_alignment.similarity_threshold = 0.1
            agent.semantic_alignment.confidence_threshold = 0.2
            
            # Test alignment
            alignments = agent.semantic_alignment.align_entities(
                ifc_entities=entities[:3],
                regulatory_entities=reg_entities[:20]
            )
            
            print(f"Alignments found: {len(alignments)}")
            
            if len(alignments) == 0:
                print("\nTesting manual similarity:")
                if entities and reg_entities:
                    entity = entities[0]
                    reg_entity = reg_entities[0]
                    
                    entity_name = entity.get('name', 'Unknown') if isinstance(entity, dict) else str(entity)
                    reg_text = reg_entity.get('text', '') if isinstance(reg_entity, dict) else str(reg_entity)
                    
                    print(f"IFC: {entity_name}")
                    print(f"REG: {reg_text}")
                    
                    similarity = agent.semantic_alignment._calculate_semantic_similarity(entity_name, reg_text)
                    print(f"Similarity: {similarity:.3f}")
                    print(f"Threshold: {agent.semantic_alignment.similarity_threshold}")
                    
                    # Check for building terms in regulatory text
                    building_terms = ['wall', 'floor', 'slab', 'building', '墙', '楼板', '建筑']
                    matches = []
                    for i, reg_ent in enumerate(reg_entities[:30]):
                        text = reg_ent.get('text', '') if isinstance(reg_ent, dict) else str(reg_ent)
                        for term in building_terms:
                            if term.lower() in text.lower():
                                matches.append((i, text, term))
                                break
                    
                    print(f"\nBuilding-related matches found: {len(matches)}")
                    for i, (idx, text, term) in enumerate(matches[:3]):
                        print(f"  {i+1}. '{text}' (contains '{term}')")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple()