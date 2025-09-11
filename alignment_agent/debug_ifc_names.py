#!/usr/bin/env python3
"""
Debug IFC entity name extraction
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.ifc_processor import IFCProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


def debug_ifc_names():
    """Debug IFC entity name extraction."""
    print("=== IFC Entity Name Extraction Debug ===")

    # Initialize IFC processor
    ifc_processor = IFCProcessor()

    # Sample IFC file path
    ifc_file_path = "outlinewall.ifc"

    if not os.path.exists(ifc_file_path):
        print(f"IFC file not found: {ifc_file_path}")
        return

    print(f"Processing IFC file: {ifc_file_path}")

    try:
        # Process IFC file
        result = ifc_processor.process_ifc_file(ifc_file_path)

        print(f"\nProcessing result keys: {list(result.keys())}")

        entities = result.get("entities", [])
        print(f"Number of entities: {len(entities)}")

        # Check first few entities
        for i, entity in enumerate(entities[:5]):
            print(f"\nEntity {i+1}:")
            print(f"  Type: {entity.get('type', 'N/A')}")
            print(f"  Name: {entity.get('name', 'N/A')}")
            print(f"  Global ID: {entity.get('global_id', 'N/A')}")
            print(f"  Description: {entity.get('description', 'N/A')}")

            # Check attributes for additional name info
            attributes = entity.get("attributes", {})
            if attributes:
                print(f"  Attributes keys: {list(attributes.keys())}")

                # Look for name-related attributes
                for attr_key, attr_value in attributes.items():
                    if "name" in attr_key.lower() or "tag" in attr_key.lower():
                        print(f"    {attr_key}: {attr_value}")

                # Check property sets
                for pset_name, pset_props in attributes.items():
                    if isinstance(pset_props, dict):
                        for prop_name, prop_value in pset_props.items():
                            if "name" in prop_name.lower() or "tag" in prop_name.lower():
                                print(f"    {pset_name}.{prop_name}: {prop_value}")

        # Try to load IFC file directly to check raw data
        print("\n=== Direct IFC File Analysis ===")
        try:
            import ifcopenshell

            ifc_file = ifcopenshell.open(ifc_file_path)

            # Get first few entities of supported types
            supported_types = ["IfcWall", "IfcSlab", "IfcBeam", "IfcColumn", "IfcDoor", "IfcWindow"]

            for entity_type in supported_types:
                entities_of_type = ifc_file.by_type(entity_type)
                if entities_of_type:
                    print(f"\n{entity_type} entities found: {len(entities_of_type)}")

                    for i, entity in enumerate(entities_of_type[:2]):
                        print(f"  Entity {i+1}:")
                        print(f"    GlobalId: {getattr(entity, 'GlobalId', 'N/A')}")
                        print(f"    Name: {getattr(entity, 'Name', 'N/A')}")
                        print(f"    Description: {getattr(entity, 'Description', 'N/A')}")
                        print(f"    ObjectType: {getattr(entity, 'ObjectType', 'N/A')}")
                        print(f"    Tag: {getattr(entity, 'Tag', 'N/A')}")

        except ImportError:
            print("ifcopenshell not available for direct analysis")
        except Exception as e:
            print(f"Error in direct IFC analysis: {e}")

    except Exception as e:
        print(f"Error processing IFC file: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_ifc_names()
