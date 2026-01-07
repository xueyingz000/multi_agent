import ifcopenshell
import ifcopenshell.util.placement
import collections


def get_accumulated_z(element):
    """递归获取累积Z坐标"""
    z = 0.0
    try:
        placement = element.ObjectPlacement
        while placement:
            matrix = ifcopenshell.util.placement.get_local_placement(placement)
            if matrix is not None:
                z += matrix[2][3]
            if hasattr(placement, "PlacementRelTo") and placement.PlacementRelTo:
                placement = placement.PlacementRelTo
            else:
                break
    except Exception:
        pass
    return z


def run_inspection():
    file_path = "/Users/zhuxueying/ifc/ifc_files/academic b.ifc"
    print(f"Loading {file_path}...")
    ifc_file = ifcopenshell.open(file_path)

    # Target elevations to inspect
    targets = [0.0, 4.5]
    tolerance = 1.0

    print(f"Inspecting elements near elevations: {targets} with tolerance {tolerance}m")

    # Broad category of elements to check
    types_to_check = [
        "IfcWall",
        "IfcWallStandardCase",
        "IfcCurtainWall",
        "IfcColumn",
        "IfcBeam",
        "IfcMember",
        "IfcPlate",
        "IfcBuildingElementProxy",
        "IfcCovering",
        "IfcWindow",
        "IfcDoor",
        "IfcRailing",
        "IfcStair",
    ]

    all_elements = []
    for t in types_to_check:
        all_elements.extend(ifc_file.by_type(t))

    print(f"Found {len(all_elements)} total elements of interest.")

    hits = collections.defaultdict(list)

    for el in all_elements:
        z = get_accumulated_z(el)
        # Check against targets
        for t_elev in targets:
            if abs(z - t_elev) < tolerance:
                hits[t_elev].append(el)

    # 1. Histogram of Z values
    z_counts = collections.defaultdict(int)
    z_type_counts = collections.defaultdict(lambda: collections.defaultdict(int))

    for el in all_elements:
        z = get_accumulated_z(el)
        z_rounded = round(z, 1)  # Round to nearest 0.1m
        z_counts[z_rounded] += 1
        z_type_counts[z_rounded][el.is_a()] += 1

    print("\n=== Z Coordinate Distribution (All Elements) ===")
    sorted_z = sorted(z_counts.keys())
    for z in sorted_z:
        if z_counts[z] > 5:  # Only show significant clusters
            print(f"Z={z}m: {z_counts[z]} elements")
            print(f"  Breakdown: {dict(z_type_counts[z])}")

    # 3. Inspect Walls at 0.0m for Height
    print("\n=== Inspecting Walls at 0.0m for Height ===")
    walls_0 = [el for el in hits[0.0] if el.is_a() == "IfcWallStandardCase"]
    for w in walls_0[:5]:  # Check first 5
        print(f"Wall: {w.Name} (ID: {w.id()})")
        # Try to find height
        height = None
        # Check properties
        for rel in w.IsDefinedBy:
            if rel.is_a() == "IfcRelDefinesByProperties":
                props = rel.RelatingPropertyDefinition
                if props.is_a() == "IfcPropertySet":
                    for prop in props.HasProperties:
                        if prop.Name == "Height":  # Common property name?
                            height = prop.NominalValue.wrappedValue
        print(f"  Property Height: {height}")

        # Check Quantities
        for rel in w.IsDefinedBy:
            if (
                rel.is_a() == "IfcRelDefinesByProperties"
                and rel.RelatingPropertyDefinition.is_a() == "IfcElementQuantity"
            ):
                for q in rel.RelatingPropertyDefinition.Quantities:
                    if q.Name == "Height":
                        print(f"  Quantity Height: {q.LengthValue}")


if __name__ == "__main__":
    run_inspection()
