import sys
import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from shapely.geometry import Polygon, MultiPolygon
import ifcopenshell
import ifcopenshell.util.placement

# Add parent directory to path to import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from external_wall import get_external_wall_outline
from ifc_slab_classifier import get_storey_elevation, get_slab_geometry
from shapely.ops import unary_union

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class AreaCalculationAgent:
    def __init__(self):
        self.name = "Area Calculation Agent"

    def calculate(self, ifc_file, rules_data: Dict, alignment_data: List[Dict]) -> Dict:
        """
        Main entry point for area calculation.

        Args:
            ifc_file: Opened ifcopenshell file object.
            rules_data: Output from RegulationAnalysisAgent (Agent 1).
            alignment_data: Output from IfcSemanticAlignmentAgent (Agent 2).

        Returns:
            Dict containing detailed area calculation results.
        """
        logger.info("🏗️ [Agent 3] Starting Area Calculation...")

        # 1. Analyze Stories & Heights
        stories_info = self._analyze_stories(ifc_file)

        # 2. Get External Wall Outlines (Base Area - Inside)
        # This returns a dictionary keyed by elevation
        # Pass target elevations to ensure all stories are considered
        target_elevations = [s["elevation"] for s in stories_info]
        wall_outlines = get_external_wall_outline(
            ifc_file, target_elevations=target_elevations
        )
        # Temporary Fix: Disable external wall detection as per user request
        # wall_outlines = {}

        # 2.5 Get All Slabs and group by story for Geometric Analysis
        # We need this to determine what is "Outside" the wall outline (Balconies)
        slabs_by_story = self._group_slabs_by_story(ifc_file, stories_info)

        # 3. Parse Rules
        height_rules = rules_data.get("height_requirements", [])
        enclosure_rules = rules_data.get("enclosure_requirements", [])
        special_space_rules = rules_data.get("special_space_requirements", [])

        # 4. Process Semantic Alignment Data (Group by Story)
        semantic_by_story = self._group_semantics_by_story(
            ifc_file, alignment_data, stories_info
        )

        # 5. Calculate Area per Story
        detailed_results = []
        total_area = 0.0

        for story in stories_info:
            elevation = story["elevation"]
            story_name = story["name"]
            height = story["height"]

            # A. Geometric Analysis (Inside vs Outside)
            outline_data = self._find_outline_for_elevation(wall_outlines, elevation)
            # outline_data = None  # Disabled
            inside_area = 0.0
            outside_area = 0.0  # Balconies (Geometric)
            outline_poly = None

            # Get Slab Geometry for this story
            story_slabs = slabs_by_story.get(story["obj"].GlobalId, [])
            merged_slab_poly = None
            if story_slabs:
                polys = []
                for slab in story_slabs:
                    geom = get_slab_geometry(ifc_file, slab)
                    if geom and geom.get("top_polygon"):
                        polys.append(geom["top_polygon"])
                if polys:
                    merged_slab_poly = unary_union(polys)
                    if not merged_slab_poly.is_valid:
                        merged_slab_poly = merged_slab_poly.buffer(0)

            # Temporary Logic: Treat all slab area as "Inside"
            # if merged_slab_poly:
            #     inside_area = merged_slab_poly.area
            #     outline_poly = merged_slab_poly  # Use slab as the outline

            # Original Logic Disabled
            if outline_data and outline_data.get("merged_polygon"):
                outline_poly = outline_data["merged_polygon"]
                # Repair if invalid
                if not outline_poly.is_valid:
                    outline_poly = outline_poly.buffer(0)

                # Base Area (Inside) is the Wall Outline
                # Note: We assume the Wall Outline represents the "Enclosed" area.
                # However, to be safe, we should perhaps intersect with the slab?
                # Usually GFA is measured to the outside face of the wall, even if there is no slab (e.g. atriums might be handled by Void deduction).
                # But if we rely on Outline for GFA:
                inside_area = outline_poly.area

                # Calculate Outside Area (Balconies)
                if merged_slab_poly:
                    if not merged_slab_poly.is_valid:
                        merged_slab_poly = merged_slab_poly.buffer(0)

                    # Difference: Slab - WallOutline
                    try:
                        diff_poly = merged_slab_poly.difference(outline_poly)
                        outside_area = diff_poly.area
                    except Exception as e:
                        logger.warning(
                            f"Error calculating outside area for {story_name}: {e}"
                        )

            # If no outline found, but we have slabs, maybe it's all "Outside" (e.g. roof deck?)?
            # Or maybe we just take the slab area?
            # For now, if no outline, we assume it's 0 inside.

            base_area = inside_area  # This is the 1.0 area candidate

            # B. Determine Base Coefficient (Height)
            height_coeff = self._determine_height_coefficient(height, height_rules)

            # --- BREAKDOWN TRACKING ---
            # Track area components by coefficient
            breakdown_1_0 = 0.0
            breakdown_0_5 = 0.0
            breakdown_excluded = 0.0

            # Initialize from Base Area
            if height_coeff == 1.0:
                breakdown_1_0 += base_area
            elif height_coeff == 0.5:
                breakdown_0_5 += base_area
            else:
                # If there are other coefficients, for now assume they fall into excluded or custom
                # But typically height coeff is 1.0 or 0.5 (for low headroom)
                # If it's 0 (excluded), it goes to excluded.
                if height_coeff == 0:
                    breakdown_excluded += base_area
                else:
                    # Treat as custom, but for this breakdown we might need to be careful.
                    # Let's put it in 0.5 if it's < 1.0? Or just leave it as is.
                    # For strict 1.0, 0.5, excluded categories:
                    pass

            # C. Adjustments based on Semantic Elements AND Geometric Findings
            story_semantics = semantic_by_story.get(story["obj"].GlobalId, [])
            adjustments_area = 0.0
            adjustment_details = []

            # 1. Add Geometric Balconies (Outside Area)
            # We treat this as an adjustment or part of the calc.
            # Let's add it to adjustments.
            # Default coeff for outside is 0.5
            outside_coeff = 0.5
            # Check enclosure rules if "Outside" area should be 1.0?
            # Usually outside = 0.5.
            if outside_area > 0.01:  # Tolerance
                added_outside = outside_area * outside_coeff
                adjustments_area += added_outside
                adjustment_details.append(
                    f"Geometric Balcony/Outside ({round(outside_area, 2)}m2 * {outside_coeff})"
                )
                # Add to breakdown
                if outside_coeff == 0.5:
                    breakdown_0_5 += outside_area
                elif outside_coeff == 1.0:
                    breakdown_1_0 += outside_area

            for item in story_semantics:
                category = item.get("category", "").upper()
                dimensions = item.get("dimensions", {})
                elem_area = float(dimensions.get("area", 0.0))

                # 1. Balconies (Semantic)
                # Since we calculated Geometric Balconies, we should NOT double count.
                # If we rely on geometry, we skip semantic balconies UNLESS they are enclosed (which might be inside?).
                # If a semantic balcony is "Inside" the outline, it's already in base_area (1.0).
                # If it's "Outside", it's in outside_area.
                # So we should probably IGNORE "BALCONY" semantics for Area Addition if we trust geometry.
                # BUT, maybe Agent 2 knows about specific "Enclosed Balconies" that are outside?
                # Let's assume Geometric approach supercedes Semantic approach for "Adding Area".
                if "BALCONY" in category:
                    pass  # Handled by Geometric Analysis (Outside Area)

                # 2. Voids / Shafts (Subtract from area, usually inside external wall)
                elif category in [
                    "ATRIUM",
                    "SHAFT",
                    "VOID",
                    "ELEVATOR_SHAFT",
                    "STAIRWELL",
                ]:
                    # Simplified logic: If it's a void, subtract it
                    if category in ["SHAFT", "VOID", "ATRIUM"]:
                        adjustments_area -= elem_area
                        adjustment_details.append(
                            f"Subtracted {category} ({elem_area}m2)"
                        )
                        # Remove from breakdown (it was likely in base_area)
                        if height_coeff == 1.0:
                            breakdown_1_0 -= elem_area
                        elif height_coeff == 0.5:
                            breakdown_0_5 -= elem_area

                        breakdown_excluded += elem_area

                # 3. Special Spaces (Parking, Refuge, etc.)
                else:
                    # Check against special_space_requirements
                    matched_rule = None
                    for rule in special_space_rules:
                        # Simple keyword matching
                        rule_desc = rule.get("description", "").upper()
                        # If rule says "Parking" and category is "PARKING", match.
                        if rule_desc in category or category in rule_desc:
                            matched_rule = rule
                            break

                    if matched_rule:
                        coeff = float(matched_rule.get("coefficient", 1.0))
                        # Assumption: These spaces are INCLUDED in the Base Area (external wall).
                        # So we need to adjust from 1.0 to coeff.
                        # Adjustment = Area * (Coeff - 1.0)
                        diff = elem_area * (coeff - 1.0)
                        adjustments_area += diff
                        adjustment_details.append(
                            f"Adjusted {category} ({elem_area}m2 * {coeff})"
                        )

                        # Adjust breakdown
                        # Move area from current height_coeff bucket to new coeff bucket
                        # Typically moves from 1.0 to 0.5 or 0.0

                        # Remove from original bucket
                        if height_coeff == 1.0:
                            breakdown_1_0 -= elem_area
                        elif height_coeff == 0.5:
                            breakdown_0_5 -= elem_area

                        # Add to new bucket
                        if coeff == 1.0:
                            breakdown_1_0 += elem_area
                        elif coeff == 0.5:
                            breakdown_0_5 += elem_area
                        elif coeff == 0.0:
                            breakdown_excluded += elem_area

            # Calculate Final Area for this story
            # Formula: (Base Area * Height Coeff) + Adjustments
            calculated_area = (base_area * height_coeff) + adjustments_area

            # Ensure non-negative
            calculated_area = max(0.0, calculated_area)
            breakdown_1_0 = max(0.0, breakdown_1_0)
            breakdown_0_5 = max(0.0, breakdown_0_5)
            breakdown_excluded = max(0.0, breakdown_excluded)

            story_result = {
                "story_name": story_name,
                "elevation": elevation,
                "height": height,
                "base_area": round(base_area, 2),
                "height_coefficient": height_coeff,
                "adjustments_area": round(adjustments_area, 2),
                "adjustment_details": adjustment_details,
                "calculated_area": round(calculated_area, 2),
                "area_full": round(breakdown_1_0, 2),
                "area_half": round(breakdown_0_5, 2),
                "area_excluded": round(breakdown_excluded, 2),
                "outline_polygon": str(outline_poly) if outline_poly else None,
            }

            detailed_results.append(story_result)
            total_area += calculated_area

        # 6. Generate Report
        report = {
            "total_area": round(total_area, 2),
            "stories": detailed_results,
            "regulation_applied": rules_data.get("region", "Unknown"),
            "summary_text": f"Total Calculated Area: {round(total_area, 2)} sq.m across {len(detailed_results)} stories.",
        }

        logger.info(f"✅ [Agent 3] Calculation Complete. Total Area: {total_area}")
        return report

    def export_to_excel(
        self, report: Dict, output_path: str = "area_calculation_report.xlsx"
    ):
        """
        Export the calculation report to an Excel file.
        """
        try:
            # Calculate aggregates
            stories = report.get("stories", [])
            total_full = sum(s.get("area_full", 0.0) for s in stories)
            total_half = sum(s.get("area_half", 0.0) for s in stories)
            total_excluded = sum(s.get("area_excluded", 0.0) for s in stories)

            # 1. Summary Data
            summary_data = {
                "Item": [
                    "Region",
                    "Total Calculated Area (sq.m)",
                    "Total Full Area (100%)",
                    "Total Half Area (50%)",
                    "Total Excluded Area",
                    "Story Count",
                    "Date",
                ],
                "Value": [
                    report.get("regulation_applied", "Unknown"),
                    report.get("total_area", 0.0),
                    round(total_full, 2),
                    round(total_half, 2),
                    round(total_excluded, 2),
                    len(stories),
                    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                ],
            }
            df_summary = pd.DataFrame(summary_data)

            # 2. Detailed Data
            detail_data = []
            for s in stories:
                detail_data.append(
                    {
                        "Story": s.get("story_name"),
                        "Elevation (m)": s.get("elevation"),
                        "Height (m)": s.get("height"),
                        "Base Area (sq.m)": s.get("base_area"),
                        "Height Coeff": s.get("height_coefficient"),
                        "Adjustments (sq.m)": s.get("adjustments_area"),
                        "Adjustment Details": "; ".join(
                            s.get("adjustment_details", [])
                        ),
                        "Full Area (100%)": s.get("area_full", 0.0),
                        "Half Area (50%)": s.get("area_half", 0.0),
                        "Excluded Area": s.get("area_excluded", 0.0),
                        "Calculated Area (sq.m)": s.get("calculated_area"),
                    }
                )
            df_details = pd.DataFrame(detail_data)

            # Write to Excel
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df_summary.to_excel(writer, sheet_name="Summary", index=False)
                df_details.to_excel(
                    writer, sheet_name="Detailed Breakdown", index=False
                )

            logger.info(f"✅ Report exported to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ Failed to export Excel: {e}")
            return None

    def _group_slabs_by_story(self, ifc_file, stories_info):
        """
        Group all IfcSlab elements by story using spatial containment or placement Z.
        """
        grouped = {s["obj"].GlobalId: [] for s in stories_info}

        # Get all slabs
        slabs = ifc_file.by_type("IfcSlab")

        for slab in slabs:
            # Try to get story from spatial containment first (more reliable)
            # But simpler here: use elevation
            try:
                # 1. Check spatial structure (RelContainedInSpatialStructure)
                # This is standard IFC but might be missing in some files.
                # If found, use it.
                rel_contained = slab.ContainedInStructure
                story_found = False
                if rel_contained:
                    for rel in rel_contained:
                        if rel.RelatingStructure.is_a("IfcBuildingStorey"):
                            if rel.RelatingStructure.GlobalId in grouped:
                                grouped[rel.RelatingStructure.GlobalId].append(slab)
                                story_found = True
                                break

                if story_found:
                    continue

                # 2. Fallback to Z placement
                local_placement = ifcopenshell.util.placement.get_local_placement(
                    slab.ObjectPlacement
                )
                z = local_placement[2][3]

                # Find nearest story below
                matched_story_guid = None
                for i, story in enumerate(stories_info):
                    story_elev = story["elevation"]
                    next_story_elev = (
                        stories_info[i + 1]["elevation"]
                        if i + 1 < len(stories_info)
                        else float("inf")
                    )

                    if story_elev - 0.5 <= z < next_story_elev:
                        matched_story_guid = story["obj"].GlobalId
                        break

                if matched_story_guid:
                    grouped[matched_story_guid].append(slab)

            except Exception as e:
                pass

        return grouped

    def _group_semantics_by_story(
        self, ifc_file, alignment_data, stories_info
    ) -> Dict[str, List[Dict]]:
        """
        Map semantic elements to stories.
        Returns: Dict {story_guid: [element_data, ...]}
        """
        grouped = {}
        for story in stories_info:
            grouped[story["obj"].GlobalId] = []

        # Helper to find story by elevation
        # Assuming stories_info is sorted by elevation
        elevations = [s["elevation"] for s in stories_info]

        for item in alignment_data:
            guid = item.get("guid")
            if not guid:
                continue

            try:
                element = (
                    ifc_file.by_id(guid) if len(guid) < 22 else ifc_file.by_guid(guid)
                )
                if not element:
                    continue

                # Get element elevation
                local_placement = ifcopenshell.util.placement.get_local_placement(
                    element.ObjectPlacement
                )
                z = local_placement[2][3]

                # Find nearest story below z
                matched_story_guid = None
                for i, story in enumerate(stories_info):
                    story_elev = story["elevation"]
                    next_story_elev = (
                        stories_info[i + 1]["elevation"]
                        if i + 1 < len(stories_info)
                        else float("inf")
                    )

                    # Tolerance 0.1m
                    if story_elev - 0.1 <= z < next_story_elev:
                        matched_story_guid = story["obj"].GlobalId
                        break

                if matched_story_guid:
                    grouped[matched_story_guid].append(item)

            except Exception as e:
                logger.warning(f"Failed to map element {guid} to story: {e}")

        return grouped

    def _analyze_stories(self, ifc_file):
        """
        Extract story information using get_storey_elevation tool logic.
        Filters out storeys that are too close to each other to avoid zero-height issues.
        Ensures all units are converted to Meters.
        """
        import ifcopenshell.util.unit

        # Calculate unit scale (to meters)
        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(ifc_file)
        logger.info(f"Unit Scale to Meters: {unit_scale}")

        stories = ifc_file.by_type("IfcBuildingStorey")
        stories_data = []
        for s in stories:
            elev = get_storey_elevation(s)
            # Normalize to meters
            elev_m = (elev * unit_scale) if elev is not None else 0.0

            stories_data.append(
                {
                    "obj": s,
                    "name": s.Name,
                    "elevation": elev_m,
                    "raw_elevation": elev,  # Keep raw for debugging
                }
            )

        # Sort by elevation
        stories_data.sort(key=lambda x: x["elevation"])

        # Filter out duplicates/near-duplicates (e.g. < 0.2m difference)
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
                    logger.warning(
                        f"Skipping duplicate/close story: {next_story['name']} (Elev: {next_story['elevation']}) near {current_story['name']} (Elev: {current_story['elevation']})"
                    )

        stories_data = unique_stories

        # Calculate Height (diff with next story)
        for i, s in enumerate(stories_data):
            if i < len(stories_data) - 1:
                height = stories_data[i + 1]["elevation"] - s["elevation"]
            else:
                height = 3.0  # Default for top floor
            s["height"] = height
            logger.info(
                f"Story {s['name']}: Elev={s['elevation']:.3f}m, Height={s['height']:.3f}m"
            )

        return stories_data

    def _find_outline_for_elevation(self, wall_outlines, elevation):
        """
        Match exact elevation to wall outlines (which are keyed by elevation).
        Allow small tolerance.
        """
        for elev, data in wall_outlines.items():
            if abs(elev - elevation) < 0.1:
                return data
        return None

    def _determine_height_coefficient(self, height, height_rules):
        """
        Apply Agent 1 rules to determine height coefficient.
        """
        coeff = 1.0  # Default
        for rule in height_rules:
            logic = rule.get("condition_logic", "")
            rule_coeff = float(rule.get("coefficient", 1.0))

            # Simple eval (security risk in prod, acceptable for prototype)
            # logic example: "h < 2.2"
            try:
                if eval(logic, {"h": height}):
                    coeff = rule_coeff
                    # Assumption: If multiple rules match, take the min? or last?
                    # Usually "h < 2.2" is specific. "h >= 2.2" is 1.0.
            except:
                pass
        return coeff


if __name__ == "__main__":
    # Test execution is handled by external script
    pass
