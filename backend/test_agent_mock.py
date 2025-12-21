import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from shapely.geometry import Polygon

# Add agents path
sys.path.append(os.path.abspath("agents"))

from agents.area_calculation_agent import AreaCalculationAgent


class TestAreaCalculationAgent(unittest.TestCase):

    def setUp(self):
        self.agent = AreaCalculationAgent()

        # Mock IFC File
        self.mock_ifc = MagicMock()

        # Mock Rules (Agent 1 Output)
        self.rules_data = {
            "region": "Test Region",
            "height_requirements": [
                {"category": "height", "condition_logic": "h < 2.2", "coefficient": 0.5}
            ],
            "enclosure_requirements": [
                {
                    "category": "enclosure",
                    "description": "Unenclosed Balcony",
                    "coefficient": 0.5,
                }
            ],
            "special_space_requirements": [],
        }

        # Mock Alignment Data (Agent 2 Output)
        self.alignment_data = [
            {
                "guid": "guid_balcony",
                "category": "BALCONY",
                "dimensions": {"area": 10.0},
            },
            {"guid": "guid_void", "category": "VOID", "dimensions": {"area": 2.0}},
        ]

    @patch("agents.area_calculation_agent.get_external_wall_outline")
    @patch("agents.area_calculation_agent.AreaCalculationAgent._analyze_stories")
    @patch(
        "agents.area_calculation_agent.ifcopenshell.util.placement.get_local_placement"
    )
    @patch("agents.area_calculation_agent.get_slab_geometry")
    @patch("agents.area_calculation_agent.AreaCalculationAgent._group_slabs_by_story")
    def test_calculate(
        self,
        mock_group_slabs,
        mock_get_slab_geom,
        mock_get_placement,
        mock_analyze_stories,
        mock_get_outline,
    ):

        # 1. Setup Mock Stories
        story1 = MagicMock()
        story1.GlobalId = "story_1_guid"
        story1.Name = "Level 1"

        story2 = MagicMock()
        story2.GlobalId = "story_2_guid"
        story2.Name = "Level 2"

        mock_analyze_stories.return_value = [
            {"obj": story1, "name": "Level 1", "elevation": 0.0, "height": 3.0},
            {
                "obj": story2,
                "name": "Level 2",
                "elevation": 3.0,
                "height": 2.0,
            },  # Low height -> 0.5 coeff
        ]

        # 2. Setup Mock External Outlines
        # Story 1: 100m2 square (0-10, 0-10)
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        # Story 2: 100m2 square
        poly2 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        mock_get_outline.return_value = {
            0.0: {"merged_polygon": poly1},
            3.0: {"merged_polygon": poly2},
        }

        # Mock Slabs
        slab1 = MagicMock()
        slab2 = MagicMock()

        mock_group_slabs.return_value = {
            "story_1_guid": [slab1],
            "story_2_guid": [slab2],
        }

        # Mock Geometry
        # Side effect to return different geom for different slabs
        def side_effect_geom(ifc_file, slab):
            if slab == slab1:
                return {
                    "top_polygon": Polygon([(0, 0), (12, 0), (12, 10), (0, 10)])
                }  # 120m2 total, 20m2 outside
            else:
                return {
                    "top_polygon": Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
                }  # 100m2 total

        mock_get_slab_geom.side_effect = side_effect_geom

        # 3. Setup Mock Element Placement (for mapping to stories)
        # We need to map guid_balcony and guid_void to stories.
        # guid_balcony -> Level 1 (z=0)
        # guid_void -> Level 1 (z=0)

        def side_effect_placement(placement):
            # This is a bit hacky, we assume the mock passed in corresponds to our elements
            # But in the code, it calls ifc_file.by_id(guid).ObjectPlacement
            return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]  # Z=0

        mock_get_placement.return_value = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]  # Default Z=0

        # We also need to mock ifc_file.by_id or by_guid to return objects with ObjectPlacement
        mock_elem = MagicMock()
        self.mock_ifc.by_id.return_value = mock_elem
        self.mock_ifc.by_guid.return_value = mock_elem

        # Refine side effect to put elements on specific levels if needed
        # For this test, let's say all elements are at Z=0 (Level 1)

        # Run Calculation
        result = self.agent.calculate(
            self.mock_ifc, self.rules_data, self.alignment_data
        )

        print("\nTest Result Report:")
        print(result["summary_text"])
        for s in result["stories"]:
            print(
                f"Story: {s['story_name']}, Base: {s['base_area']}, Coeff: {s['height_coefficient']}, Adj: {s['adjustments_area']}, Final: {s['calculated_area']}"
            )

        # Assertions

        # Level 1 (Height 3.0 > 2.2 -> Coeff 1.0)
        # Base Area: 120.0 (Slab Area)
        # Geometric Balcony: 0.0 (Inside Slab)
        # Semantic Balcony: 10.0 (Inside Slab) -> Subtracted from Base, Added at 0.5 Coeff
        # Void (Area 2.0) -> Subtracted from Base
        # Expected: 120.0 - 10.0 + (10.0 * 0.5) - 2.0 = 113.0
        story1_res = result["stories"][0]
        self.assertAlmostEqual(story1_res["calculated_area"], 113.0)

        # Level 2 (Height 2.0 < 2.2 -> Coeff 0.5)
        # Base Area: 100.0 (Slab Area)
        # Geometric Balcony: 0.0
        # Expected: 100.0 * 0.5 = 50.0
        story2_res = result["stories"][1]
        self.assertAlmostEqual(story2_res["calculated_area"], 50.0)

        # Total
        self.assertAlmostEqual(result["total_area"], 163.0)

        # 4. Test Excel Export
        output_file = "test_report.xlsx"
        exported_path = self.agent.export_to_excel(result, output_file)
        self.assertIsNotNone(exported_path)
        self.assertTrue(os.path.exists(exported_path))

        # Cleanup
        if os.path.exists(output_file):
            os.remove(output_file)


if __name__ == "__main__":
    unittest.main()
