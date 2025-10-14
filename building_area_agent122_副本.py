from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.types import AgentType
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from typing import List, Dict, Any, Tuple
import os
import numpy as np
import math
from io import StringIO
import ifcopenshell

from langchain.agents import create_react_agent, AgentExecutor
from boolean.external_wall import get_external_wall_outline


class BuildingAreaCalculator:
    def __init__(self, ifc_file_path: str):
        self.ifc_file = ifcopenshell.open(ifc_file_path)
        # Add unit conversion factor
        self.length_unit_factor = self.get_length_unit_factor()

    def get_length_unit_factor(self) -> float:
        """Get length unit conversion factor"""
        project = self.ifc_file.by_type("IFCPROJECT")[0]
        units = project.UnitsInContext.Units
        for unit in units:
            if unit.is_a("IFCSIUNIT"):
                if unit.UnitType == "LENGTHUNIT":
                    if unit.Prefix == "MILLI":
                        return 0.001  # millimeter to meter
        return 1.0  # default to meter

    def calculate_building_area(self) -> Tuple[float, List[dict]]:
        """Calculate building area, return total area and detailed information"""
        # 1. Get and classify stories
        stories = self.ifc_file.by_type("IFCBUILDINGSTOREY")
        stories = sorted(stories, key=lambda x: x.Elevation)  # Sort by elevation

        high_stories = []  # Stories higher than 2.2m
        low_stories = []  # Stories not higher than 2.2m
        detailed_info = []  # Store detailed information

        for i, story in enumerate(stories):
            height = self.get_story_height(story, stories, i)
            if height > 2.2:
                high_stories.append(story)
            else:
                low_stories.append(story)

        total_area = 0

        # Process stories higher than 2.2m
        for story in high_stories:
            story_info = {
                "story_name": story.Name,
                "story_elevation": story.Elevation * self.length_unit_factor,
                "calculation_coefficient": 1.0,
                "slab_list": [],
            }

            slabs = self.get_slabs_for_story(story)
            story_total_area = 0

            for slab in slabs:
                area = self.calculate_slab_area(slab)
                story_total_area += area
                slab_info = {
                    "slab_name": slab.Name if hasattr(slab, "Name") else "Unnamed slab",
                    "area": area,
                }
                story_info["slab_list"].append(slab_info)

            story_info["story_total_area"] = story_total_area
            total_area += story_total_area
            detailed_info.append(story_info)

        # Process stories lower than 2.2m
        for story in low_stories:
            story_info = {
                "story_name": story.Name,
                "story_elevation": story.Elevation * self.length_unit_factor,
                "calculation_coefficient": 0.5,
                "slab_list": [],
            }

            slabs = self.get_slabs_for_story(story)
            story_total_area = 0

            for slab in slabs:
                area = self.calculate_slab_area(slab) * 0.5  # Half area
                story_total_area += area
                slab_info = {
                    "slab_name": slab.Name if hasattr(slab, "Name") else "Unnamed slab",
                    "area": area,
                }
                story_info["slab_list"].append(slab_info)

            story_info["story_total_area"] = story_total_area
            total_area += story_total_area
            detailed_info.append(story_info)

        return total_area, detailed_info

    def get_story_height(self, story, stories: List, current_index: int) -> float:
        """Get story height"""
        current_elevation = story.Elevation * self.length_unit_factor
        if current_index < len(stories) - 1:
            next_elevation = (
                stories[current_index + 1].Elevation * self.length_unit_factor
            )
            return abs(next_elevation - current_elevation)
        else:
            # For the top floor, try to get height from other attributes
            if hasattr(story, "TotalHeight"):
                return story.TotalHeight
            # If there's no clear height information, use default value or infer from component height
            return self.get_height_from_components(story)

    def get_height_from_components(self, story) -> float:
        """Infer height from components in the story"""
        max_height = 0
        related_elements = self.get_related_elements(story)
        for element in related_elements:
            if hasattr(element, "Height"):
                max_height = max(max_height, element.Height)
        return max_height if max_height > 0 else 3.0  # Default height 3.0m

    def get_related_elements(self, story) -> List:
        """Get all elements related to the story"""
        related_elements = []
        containment = self.ifc_file.by_type("IFCRELCONTAINEDINSPATIALSTRUCTURE")
        for rel in containment:
            if rel.RelatingStructure == story:
                related_elements.extend(rel.RelatedElements)
        return related_elements

    def get_slabs_for_story(self, story) -> List:
        """Get slabs corresponding to the story"""
        slabs = []
        story_elevation = story.Elevation * self.length_unit_factor
        all_slabs = self.ifc_file.by_type("IFCSLAB")

        for slab in all_slabs:
            slab_elevation = self.get_slab_elevation(slab)
            # Allow 0.1 meter error range
            if abs(slab_elevation - story_elevation) < 0.1:
                slabs.append(slab)
        return slabs

    def get_slab_elevation(self, slab) -> float:
        """Get absolute elevation of the slab"""
        if not slab.ObjectPlacement:
            return 0.0

        elevation = 0.0
        placement = slab.ObjectPlacement

        # If it's IfcLocalPlacement type
        if placement.is_a("IfcLocalPlacement"):
            # Get relative position
            rel_placement = placement.RelativePlacement
            if rel_placement and hasattr(rel_placement, "Location"):
                location = rel_placement.Location
                if location:
                    elevation = location.Coordinates[2]  # Z coordinate is elevation

        # If there's PlacementRelTo, need to accumulate all relative elevations
        while hasattr(placement, "PlacementRelTo") and placement.PlacementRelTo:
            placement = placement.PlacementRelTo
            if hasattr(placement, "RelativePlacement"):
                rel_placement = placement.RelativePlacement
                if rel_placement and hasattr(rel_placement, "Location"):
                    location = rel_placement.Location
                    if location:
                        elevation += location.Coordinates[2]

        return elevation * self.length_unit_factor

    def calculate_slab_area(self, slab) -> float:
        """Calculate slab area"""
        representation = slab.Representation
        if not representation:
            return 0

        total_area = 0
        for item in representation.Representations:
            if hasattr(item, "Items"):
                for shape in item.Items:
                    area = self.calculate_shape_area(shape)
                    total_area += area

        # Handle openings
        openings = self.get_slab_openings(slab)
        for opening in openings:
            if not self.is_elevator_opening(opening):
                opening_area = self.calculate_opening_area(opening)
                total_area -= opening_area

        return total_area

    def calculate_shape_area(self, shape) -> float:
        """Calculate shape area"""
        if shape.is_a("IFCEXTRUDEDAREASOLID"):
            profile = shape.SweptArea

            # Add debug information
            print(f"Profile type: {profile.is_a()}")

            if profile.is_a("IFCCIRCLEPROFILEDEF"):
                radius = profile.Radius * self.length_unit_factor
                return math.pi * radius * radius
            else:
                points = self.get_profile_points(profile)
                if points:
                    # Add debug information
                    print(f"Profile point coordinates: {points}")

                    points = [
                        (x * self.length_unit_factor, y * self.length_unit_factor)
                        for x, y in points
                    ]
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    area = 0.5 * np.abs(
                        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                    )
                    print(f"Calculated area: {area:.2f} square meters")
                    return area
        return 0

    def get_profile_points(self, profile) -> List[Tuple[float, float]]:
        """Get profile point coordinates"""
        points = []
        if hasattr(profile, "OuterCurve"):
            # Handle IFCARBITRARYCLOSEDPROFILEDEF
            curve = profile.OuterCurve
            if curve.is_a("IFCPOLYLINE"):
                points = [
                    (point.Coordinates[0], point.Coordinates[1])
                    for point in curve.Points
                ]
        elif hasattr(profile, "XDim") and hasattr(profile, "YDim"):
            # Handle IFCRECTANGLEPROFILEDEF
            x, y = profile.XDim / 2, profile.YDim / 2
            points = [(-x, -y), (x, -y), (x, y), (-x, y)]
        return points

    def get_slab_openings(self, slab) -> List:
        """Get slab openings"""
        openings = []
        if hasattr(slab, "HasOpenings"):
            for rel in slab.HasOpenings:
                if hasattr(rel, "RelatedOpeningElement"):
                    openings.append(rel.RelatedOpeningElement)
        return openings

    def is_elevator_opening(self, opening) -> bool:
        """Determine if it's an elevator shaft opening"""
        # Can be determined by opening attributes, name or associated components
        # Here uses simple name judgment, actual projects need more complex judgment logic
        if hasattr(opening, "Name"):
            name_lower = opening.Name.lower()
            return "elevator" in name_lower or "电梯" in name_lower
        return False

    def calculate_opening_area(self, opening) -> float:
        """Calculate opening area"""
        representation = opening.Representation
        if not representation:
            return 0

        total_area = 0
        for item in representation.Representations:
            if hasattr(item, "Items"):
                for shape in item.Items:
                    area = self.calculate_shape_area(shape)
                    total_area += area
        return total_area


class ExternalWallTool:
    def __init__(self, ifc_file_path):
        import ifcopenshell

        self.ifc_file = ifcopenshell.open(ifc_file_path)

    def get_external_walls(self):
        return get_external_wall_outline(self.ifc_file)


def create_ifc_tools(
    calculator: BuildingAreaCalculator, external_wall_tool: ExternalWallTool
):
    return [
        Tool(
            name="CalculateBuildingArea",
            func=lambda _: calculator.calculate_building_area(),
            description="Calculate total building area, return total area and detailed information for each floor. Return value is tuple (total_area, detailed_info_list)",
        ),
        Tool(
            name="GetStoryHeight",
            func=lambda story, stories, idx: calculator.get_story_height(
                story, stories, idx
            ),
            description="Get the height of specified story",
        ),
        Tool(
            name="GetSlabsForStory",
            func=lambda story: calculator.get_slabs_for_story(story),
            description="Get all slabs for specified story",
        ),
        Tool(
            name="CalculateSlabArea",
            func=lambda slab: calculator.calculate_slab_area(slab),
            description="Calculate area of single slab, including opening deduction",
        ),
        Tool(
            name="GetExternalWalls",
            func=lambda _: external_wall_tool.get_external_walls(),
            description="Identify and return external wall outline and external wall ID list for each story",
        ),
    ]


# Define agent class
class BuildingAreaAgent:
    def __init__(self, llm, ifc_file_path: str):
        self.llm = llm

        # Read IFC file content
        # with open(ifc_file_path, 'r', encoding='utf-8') as file:
        #     ifc_content = file.read()

        # Create tools
        # tools = create_ifc_tools(BuildingAreaCalculator(ifc_file_path))
        calculator = BuildingAreaCalculator(ifc_file_path)
        external_wall_tool = ExternalWallTool(ifc_file_path)
        tools = create_ifc_tools(calculator, external_wall_tool)

        # First define base template
        BASE_TEMPLATE = """You are a professional building area calculation assistant. Use the available tools to help analyze IFC files

        Please use the provided tools to analyze the IFC file and calculate areas following these steps:

        1. Get and classify stories:
        - Retrieve all IFCBUILDINGSTOREY elements
        - Sort by elevation
        - Calculate story heights
        - Classify stories into two categories:
            * Stories higher than 2.2m (coefficient 1.0)
            * Stories not higher than 2.2m (coefficient 0.5)

        2. Calculate area for each story:
        - Get all slabs (IFCSLAB) for the story
        - Calculate area for each slab
        - Subtract openings (excluding elevator shafts)
        - Apply appropriate coefficient based on story height

        3. Use the GetExternalWalls tool to identify the external wall outline and external wall IDs for each story.

        4. Summarize results:
        - Calculate total building area
        - Generate detailed calculation report

        Available tools: {tools}
        Tool Names: {tool_names}

        To use a tool, please use the following format:
        Thought: Consider what to do next
        Action: Tool name
        Action Input: Tool input
        Observation: Tool output
        ... (repeat Thought/Action/Action Input/Observation if needed)
        Thought: I know what to do
        Final Answer: Final response to the user with detailed area calculation report

        Begin!"""

        # Create new prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", BASE_TEMPLATE),  # 直接使用包含工具描述的模板
                ("human", "{input}"),
                ("human", "This is the current conversation:\n{agent_scratchpad}"),
            ]
        )

        # Create the agent
        agent = create_react_agent(llm, tools, prompt)

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )

    def calculate_area(self, ifc_file_path: str) -> float:
        """Calculate building area from IFC file content"""
        try:
            return self.agent_executor.invoke(
                {
                    "input": "Please calculate the building area for the given IFC content"
                }
            )
        except Exception as e:
            raise Exception(f"Error reading IFC file: {str(e)}")


# Usage example
def main():
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = "sk-Dppz8ZLK62lWyZc7G3w3LtkJx7sNgmjdF65kzR7hKtQlnQDL"

    base_url = os.getenv("YUNWU_URL", "https://yunwu.zeabur.app/v1")
    #
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key="sk-Dppz8ZLK62lWyZc7G3w3LtkJx7sNgmjdF65kzR7hKtQlnQDL",
        base_url=base_url,
        temperature=0,
    )

    ifc_file_path = "endtoend.ifc"

    agent = BuildingAreaAgent(llm, ifc_file_path)

    result = agent.calculate_area(ifc_file_path)

    # Print complete return result
    print("Agent response:")
    print(result)

    # If you need to extract specific area values, handle according to actual returned data structure
    # For example:
    # if isinstance(result, dict) and 'total_area' in result:
    #     print(f"Total building area: {result['total_area']:.2f} square meters")


if __name__ == "__main__":
    main()
