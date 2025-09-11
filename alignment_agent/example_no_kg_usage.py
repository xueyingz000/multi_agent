#!/usr/bin/env python3
"""Example usage of IFC Semantic Agent without Knowledge Graph."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG
from utils import get_logger

def example_basic_usage():
    """Basic usage example."""
    logger = get_logger(__name__)
    
    # Initialize the agent (without knowledge graph)
    agent = IFCSemanticAgentNoKG()
    
    # Simple query without additional data
    query = "What are the key semantic relationships between IFC building elements and regulatory requirements?"
    
    logger.info(f"Processing query: {query}")
    
    response = agent.process_query(query)
    
    print("\n" + "="*60)
    print("BASIC USAGE EXAMPLE")
    print("="*60)
    print(f"Query: {query}")
    print(f"\nAnswer: {response.final_answer}")
    print(f"Confidence: {response.confidence_score:.2f}")
    print(f"Execution Time: {response.execution_time:.2f}s")
    print(f"Steps Taken: {response.total_steps}")
    
    return response

def example_with_ifc_data():
    """Example with IFC data."""
    logger = get_logger(__name__)
    
    # Initialize the agent
    agent = IFCSemanticAgentNoKG()
    
    # Sample IFC data representing a simple building
    ifc_data = {
        'entities': {
            'building_001': {
                'type': 'IfcBuilding',
                'attributes': {'Name': 'Office Building A', 'Description': 'Commercial office building'},
                'properties': {'TotalFloorArea': 5000, 'NumberOfFloors': 10}
            },
            'wall_ext_001': {
                'type': 'IfcWall',
                'attributes': {'Name': 'ExteriorWall_North', 'Description': 'Load bearing exterior wall'},
                'properties': {'Width': 250, 'Height': 3200, 'Material': 'Reinforced Concrete', 'ThermalTransmittance': 0.25}
            },
            'wall_int_001': {
                'type': 'IfcWall',
                'attributes': {'Name': 'InteriorWall_Partition', 'Description': 'Non-load bearing partition'},
                'properties': {'Width': 100, 'Height': 3000, 'Material': 'Gypsum Board'}
            },
            'slab_001': {
                'type': 'IfcSlab',
                'attributes': {'Name': 'FloorSlab_Level1', 'Description': 'Structural floor slab'},
                'properties': {'Thickness': 200, 'Material': 'Reinforced Concrete', 'LoadCapacity': 5000}
            },
            'door_001': {
                'type': 'IfcDoor',
                'attributes': {'Name': 'FireDoor_Exit', 'Description': 'Emergency exit door'},
                'properties': {'Width': 900, 'Height': 2100, 'FireRating': 60, 'Material': 'Steel'}
            },
            'window_001': {
                'type': 'IfcWindow',
                'attributes': {'Name': 'Window_Office', 'Description': 'Office window'},
                'properties': {'Width': 1200, 'Height': 1500, 'GlazingType': 'Double', 'UValue': 1.8}
            }
        },
        'relationships': [
            {'source': 'wall_ext_001', 'target': 'slab_001', 'type': 'supports'},
            {'source': 'slab_001', 'target': 'building_001', 'type': 'contained_in'},
            {'source': 'door_001', 'target': 'wall_ext_001', 'type': 'opens_through'},
            {'source': 'window_001', 'target': 'wall_ext_001', 'type': 'opens_through'}
        ]
    }
    
    query = "Analyze the IFC building elements and identify potential compliance issues with building codes."
    
    logger.info(f"Processing query with IFC data: {query}")
    
    response = agent.process_query(query=query, ifc_data=ifc_data)
    
    print("\n" + "="*60)
    print("IFC DATA ANALYSIS EXAMPLE")
    print("="*60)
    print(f"Query: {query}")
    print(f"\nIFC Entities Analyzed: {len(ifc_data['entities'])}")
    print(f"Relationships: {len(ifc_data['relationships'])}")
    print(f"\nAnswer: {response.final_answer}")
    print(f"Confidence: {response.confidence_score:.2f}")
    print(f"Knowledge Sources: {', '.join(response.knowledge_sources)}")
    
    return response

def example_with_regulatory_text():
    """Example with regulatory text."""
    logger = get_logger(__name__)
    
    # Initialize the agent
    agent = IFCSemanticAgentNoKG()
    
    # Sample regulatory text
    regulatory_text = """
    BUILDING CODE REQUIREMENTS - SECTION 4: STRUCTURAL ELEMENTS
    
    4.1 WALLS
    - Load-bearing walls shall have minimum thickness of 200mm
    - Exterior walls must achieve thermal transmittance (U-value) ≤ 0.30 W/m²K
    - Fire-rated walls shall maintain integrity for specified duration
    - Partition walls may be non-load bearing with minimum 75mm thickness
    
    4.2 FLOOR SLABS
    - Structural slabs shall support minimum live load of 2.5 kN/m²
    - Slab thickness shall be minimum 150mm for spans up to 6m
    - Reinforced concrete slabs require proper reinforcement coverage
    
    4.3 OPENINGS
    - Door openings in fire-rated walls require fire-rated doors
    - Emergency exit doors shall be minimum 800mm wide
    - Window openings shall not exceed 40% of wall area
    - Window U-values shall not exceed 2.0 W/m²K
    
    4.4 ACCESSIBILITY
    - Door thresholds shall not exceed 13mm height
    - Corridor widths shall be minimum 1200mm
    - Ramp slopes shall not exceed 1:12 gradient
    """
    
    query = "Extract key building code requirements and identify the main regulatory entities and their specifications."
    
    logger.info(f"Processing regulatory text: {query}")
    
    response = agent.process_query(query=query, regulatory_text=regulatory_text)
    
    print("\n" + "="*60)
    print("REGULATORY TEXT ANALYSIS EXAMPLE")
    print("="*60)
    print(f"Query: {query}")
    print(f"\nRegulatory Text Length: {len(regulatory_text)} characters")
    print(f"\nAnswer: {response.final_answer}")
    print(f"Confidence: {response.confidence_score:.2f}")
    print(f"Knowledge Sources: {', '.join(response.knowledge_sources)}")
    
    return response

def example_comprehensive_alignment():
    """Comprehensive example with both IFC data and regulatory text."""
    logger = get_logger(__name__)
    
    # Initialize the agent
    agent = IFCSemanticAgentNoKG()
    
    # IFC data for a wall
    ifc_data = {
        'entities': {
            'wall_001': {
                'type': 'IfcWall',
                'attributes': {'Name': 'ExteriorWall_South', 'Description': 'Load bearing exterior wall'},
                'properties': {
                    'Width': 300,  # 300mm thick
                    'Height': 3000,
                    'Material': 'Reinforced Concrete',
                    'ThermalTransmittance': 0.28,  # U-value
                    'FireRating': 120,  # minutes
                    'LoadBearing': True
                }
            }
        },
        'relationships': []
    }
    
    # Corresponding regulatory requirements
    regulatory_text = """
    WALL REQUIREMENTS:
    - Load-bearing walls: minimum 200mm thickness
    - Exterior walls: U-value ≤ 0.30 W/m²K
    - Fire resistance: minimum 60 minutes for structural walls
    - Concrete walls: minimum grade C25/30
    """
    
    query = "Check if the IFC wall entity complies with the regulatory requirements and identify any alignment issues."
    
    logger.info(f"Processing comprehensive alignment: {query}")
    
    response = agent.process_query(
        query=query,
        ifc_data=ifc_data,
        regulatory_text=regulatory_text
    )
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ALIGNMENT EXAMPLE")
    print("="*60)
    print(f"Query: {query}")
    print(f"\nIFC Wall Properties:")
    wall_props = ifc_data['entities']['wall_001']['properties']
    for prop, value in wall_props.items():
        print(f"  - {prop}: {value}")
    
    print(f"\nAnswer: {response.final_answer}")
    print(f"Confidence: {response.confidence_score:.2f}")
    print(f"Semantic Mappings Found: {len(response.semantic_mappings)}")
    
    # Show reasoning steps
    print(f"\nReasoning Steps ({response.total_steps} total):")
    for i, step in enumerate(response.react_steps[:3]):  # Show first 3 steps
        print(f"  {i+1}. {step.action.action_type.value}: {step.action.reasoning}")
    
    return response

def example_with_file_formats():
    """Example showing usage with .ifc and .json file formats."""
    logger = get_logger(__name__)
    
    print("\n" + "="*60)
    print("FILE FORMAT EXAMPLE (.ifc + .json)")
    print("="*60)
    
    # Note: This example shows the concept - actual files would need to be created
    print("Command line usage examples:")
    print("\n1. Process .ifc file with .json regulations:")
    print("   python run_agent.py --query \"Analyze building compliance\" \\")
    print("                       --ifc-file building_model.ifc \\")
    print("                       --text-file building_codes.json")
    
    print("\n2. Supported file formats:")
    print("   - IFC files: .ifc (Industry Foundation Classes)")
    print("   - Regulatory text: .json, .txt, .md, .pdf, .docx")
    
    print("\n3. JSON regulatory format structure:")
    json_example = {
        "sections": [
            {
                "section": "4.1",
                "regulations": [
                    {
                        "regulation_id": "4.1.1",
                        "title": "Wall Requirements",
                        "content": "Load-bearing walls shall have minimum thickness of 200mm",
                        "mandatory": True,
                        "category": "structural"
                    }
                ]
            }
        ]
    }
    
    import json
    print(json.dumps(json_example, indent=2))
    
    return None

def main():
    """Run all examples."""
    logger = get_logger(__name__)
    
    print("🏗️  IFC Semantic Agent (No Knowledge Graph) - Usage Examples")
    print("=" * 70)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("IFC Data Analysis", example_with_ifc_data),
        ("Regulatory Text Analysis", example_with_regulatory_text),
        ("Comprehensive Alignment", example_comprehensive_alignment),
        ("File Format Examples", example_with_file_formats)
    ]
    
    for example_name, example_func in examples:
        try:
            print(f"\n🔄 Running: {example_name}")
            response = example_func()
            print(f"✅ {example_name} completed successfully")
        except Exception as e:
            print(f"❌ {example_name} failed: {e}")
            logger.error(f"Example {example_name} failed: {e}")
    
    print("\n" + "="*70)
    print("🎉 All examples completed!")
    print("\nThe IFC Semantic Agent (No-KG version) is ready for use.")
    print("Key benefits of this version:")
    print("  ✓ No dependency on knowledge graph infrastructure")
    print("  ✓ Faster initialization and processing")
    print("  ✓ Direct semantic alignment using LLM and rule-based methods")
    print("  ✓ Suitable for standalone applications and testing")
    print("  ✓ Supports .ifc and .json file formats")
    print("\nTo use in your code:")
    print("  from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG")
    print("  agent = IFCSemanticAgentNoKG()")
    print("  response = agent.process_query(query, ifc_data, regulatory_text)")
    print("\nFor .ifc and .json file examples, see: example_new_formats.py")

if __name__ == "__main__":
    main()