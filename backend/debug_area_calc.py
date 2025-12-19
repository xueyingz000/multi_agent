import ifcopenshell
import ifcopenshell.util.placement
import ifcopenshell.util.unit
import sys
import os


# Mock the helper functions
def get_storey_elevation(storey):
    z = 0.0
    try:
        placement = storey.ObjectPlacement
        while placement:
            matrix = ifcopenshell.util.placement.get_local_placement(placement)
            if matrix is not None:
                z += matrix[2][3]
            if hasattr(placement, "PlacementRelTo") and placement.PlacementRelTo:
                placement = placement.PlacementRelTo
            else:
                break
    except:
        pass
    return z


def get_accumulated_z(element):
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


def analyze_stories(ifc_file):
    unit_scale = ifcopenshell.util.unit.calculate_unit_scale(ifc_file)
    print(f"DEBUG: Unit Scale = {unit_scale}")

    stories = ifc_file.by_type("IfcBuildingStorey")
    stories_data = []
    for s in stories:
        elev = get_storey_elevation(s)
        # Check if attribute Elevation exists and is different
        attr_elev = s.Elevation if hasattr(s, "Elevation") else "N/A"

        elev_m = (elev * unit_scale) if elev is not None else 0.0

        stories_data.append(
            {
                "name": s.Name,
                "raw_placement_z": elev,
                "attr_elevation": attr_elev,
                "elevation": elev_m,
                "global_id": s.GlobalId,
            }
        )

    # Sort
    stories_data.sort(key=lambda x: x["elevation"])

    # Filter
    unique_stories = []
    if stories_data:
        current_story = stories_data[0]
        unique_stories.append(current_story)
        for i in range(1, len(stories_data)):
            next_story = stories_data[i]
            if next_story["elevation"] - current_story["elevation"] > 0.2:
                unique_stories.append(next_story)
                current_story = next_story
            else:
                print(
                    f"DEBUG: Skipping duplicate {next_story['name']} at {next_story['elevation']} (near {current_story['elevation']})"
                )

    stories_data = unique_stories

    # Calculate Heights
    for i, s in enumerate(stories_data):
        if i < len(stories_data) - 1:
            height = stories_data[i + 1]["elevation"] - s["elevation"]
        else:
            height = 3.0
        s["height"] = height
        print(
            f"DEBUG: Story {s['name']} | Elev: {s['elevation']:.4f}m | Height: {s['height']:.4f}m"
        )

    return stories_data, unit_scale


def check_walls(ifc_file, stories_data, unit_scale):
    walls = ifc_file.by_type("IfcWall")
    print(f"DEBUG: Total Walls: {len(walls)}")

    # Check first 5 walls
    for w in walls[:5]:
        raw_z = get_accumulated_z(w)
        z_m = raw_z * unit_scale
        print(f"DEBUG: Wall {w.GlobalId} | Raw Z: {raw_z} | Z (m): {z_m}")

    # Try grouping
    target_elevations = [s["elevation"] for s in stories_data]
    tolerance = 1.0  # Simple tolerance

    result = {elev: 0 for elev in target_elevations}

    for w in walls:
        raw_z = get_accumulated_z(w)
        z_m = raw_z * unit_scale

        closest_elev = None
        min_dist = float("inf")
        for elev in target_elevations:
            dist = abs(z_m - elev)
            if dist < min_dist:
                min_dist = dist
                closest_elev = elev

        if closest_elev is not None and min_dist < tolerance:
            result[closest_elev] += 1

    print("DEBUG: Wall Grouping Results:")
    for elev, count in result.items():
        print(f"  Elev {elev:.4f}m: {count} walls")


# Main execution
if __name__ == "__main__":
    # Find the IFC file - assuming there is one in the current dir or subdirs
    # For now, let's look for .ifc files
    import glob

    files = glob.glob("**/*.ifc", recursive=True)
    if not files:
        print("No IFC file found.")
        sys.exit(1)

    file_path = files[0]
    print(f"DEBUG: Analyzing {file_path}")

    ifc_file = ifcopenshell.open(file_path)
    stories, unit_scale = analyze_stories(ifc_file)
    check_walls(ifc_file, stories, unit_scale)
