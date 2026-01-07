import ifcopenshell
import ifcopenshell.geom
import numpy as np

def get_element_geometry_height(element, settings):
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = shape.geometry.verts
        z_coords = [verts[i+2] for i in range(0, len(verts), 3)]
        min_z = min(z_coords)
        max_z = max(z_coords)
        return min_z, max_z, max_z - min_z
    except Exception as e:
        return None, None, None

def inspect_ids(ifc_file_path, target_ids):
    ifc_file = ifcopenshell.open(ifc_file_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    print(f"Inspecting IDs: {target_ids}")
    
    for eid in target_ids:
        try:
            element = ifc_file.by_id(eid)
            print(f"\n--- Element ID: {eid} ---")
            print(f"Type: {element.is_a()}")
            print(f"Name: {element.Name}")
            
            # Get geometry height
            min_z, max_z, height = get_element_geometry_height(element, settings)
            if min_z is not None:
                print(f"Geometry Z range: {min_z:.2f} to {max_z:.2f}")
                print(f"Calculated Height: {height:.2f}")
            else:
                print("Could not extract geometry.")
                
            # Print Property Sets
            for definition in element.IsDefinedBy:
                if definition.is_a('IfcRelDefinesByProperties'):
                    pset = definition.RelatingPropertyDefinition
                    if pset.is_a('IfcPropertySet'):
                        print(f"Pset: {pset.Name}")
                        for prop in pset.HasProperties:
                            if prop.is_a('IfcPropertySingleValue'):
                                print(f"  {prop.Name}: {prop.NominalValue.wrappedValue if prop.NominalValue else 'None'}")
                                
        except Exception as e:
            print(f"Error inspecting {eid}: {e}")

if __name__ == "__main__":
    inspect_ids("/Users/zhuxueying/ifc/ifc_files/academic b.ifc", [17063, 7861, 7886])
