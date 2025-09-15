"""Prompt Templates for LLM-based Semantic Alignment

This module contains all prompt templates used by the semantic alignment agent
for various LLM-powered analysis tasks.
"""

from typing import Dict, Any, Optional
from datetime import datetime


class PromptTemplates:
    """Collection of prompt templates for semantic alignment tasks."""

    @staticmethod
    def geometry_analyzer_prompt() -> str:
        """Prompt template for geometry analysis."""
        return """
Analyze and extract pure geometric and spatial characteristics from IFC building data. Focus on describing geometric features WITHOUT making any functional inferences or semantic interpretations.

## Input Data:
- IFC Element Type: {ifc_type}
- Element GUID: {guid}
- Geometric Properties: {geometry_data}
- Spatial Coordinates: {coordinates}
- Dimensional Data: {dimensions}
- Related Elements: {related_elements}
- Building Context: {building_info}

## Analysis Tasks:

### 1. DIMENSIONAL ANALYSIS
Extract and categorize all dimensional characteristics:
- Primary dimensions: Length={length}m, Width={width}m, Height/Thickness={height}m
- Dimensional ratios: L/W ratio, H/Area ratio

### 2. SHAPE AND FORM ANALYSIS
Describe geometric form characteristics:
- Basic shape: [rectangular, square, circular, L-shaped, T-shaped, irregular, complex]
- Shape complexity: [simple_polygon, complex_polygon, curved_boundaries, multi_part_geometry]
- Aspect ratio classification: [narrow: >3:1, elongated: 2-3:1, balanced: 1-2:1, square: ~1:1]
- Geometric regularity: [highly_regular, moderately_regular, irregular, very_irregular]

### 3. SPATIAL POSITION ANALYSIS
Analyze positional and elevation characteristics:
- Absolute position: Floor level={floor_level}, Elevation={elevation}m
- Relative position: [corner_location, central_location, perimeter_location, isolated]
- Vertical position: [ground_level, intermediate_floor, top_floor, basement, sub_basement]
- Building zone: [core_zone, perimeter_zone, transition_zone]

### 4. ADJACENCY AND CONNECTIVITY ANALYSIS
Map spatial relationships with surrounding elements:
- Direct adjacencies: Connected to {adjacent_elements} via {connection_types}
- Boundary sharing: Shared boundaries with [{shared_boundary_elements}]
- Proximity relationships: Within {proximity_distance}m of [{nearby_elements}]
- Connectivity degree: [isolated: 0-1 connections, low: 2-3, medium: 4-6, high: >6]

### 5. OPENING AND ACCESS ANALYSIS
Analyze openings and access points:
- Opening count: Total openings={opening_count}
- Opening types: [{door_openings}, {window_openings}, {generic_openings}]
- Opening dimensions: Average opening size={avg_opening_size}m²
- Opening positions: [{wall_positions}] of openings
- Access characteristics: [single_access, multiple_access, through_circulation, dead_end]

### 6. VERTICAL RELATIONSHIPS
Analyze vertical spatial connections:
- Floor-to-ceiling height: Clear height={clear_height}m
- Vertical continuity: [single_floor, multi_floor_connection, vertical_void]
- Above/below relationships: Above=[{elements_above}], Below=[{elements_below}]
- Vertical access: [stairs_present, elevator_access, ramp_access, no_vertical_access]

### 7. BOUNDARY CONDITIONS
Analyze interface with building envelope and structure:
- Exterior exposure: [{exterior_wall_count}] exterior walls, total exposure length={exposure_length}m
- Interior boundaries: [{interior_wall_count}] interior partitions
- Structural relationships: Adjacent to [{structural_elements}]
- Environmental exposure: [north_facing, south_facing, east_facing, west_facing, interior_only]

### 8. GEOMETRIC COMPLEXITY METRICS
Quantify geometric complexity:
- Boundary complexity: Perimeter-to-area ratio={perimeter_area_ratio}
- Shape convexity: [convex, mostly_convex, concave, highly_irregular]
- Corner count: {corner_count} corners/vertices
- Geometric efficiency: Area utilization ratio={efficiency_ratio}

## Output Format:
Provide a structured analysis in JSON format with the following structure:
{{
  "element_identification": {{
    "ifc_type": "string",
    "guid": "string",
    "analysis_timestamp": "ISO_datetime"
  }},
  "dimensional_characteristics": {{
    "primary_dimensions": {{
      "length": float,
      "width": float,
      "height": float
    }},
  "geometric_form": {{
    "basic_shape": "string",
    "shape_complexity": "string",
    "regularity_level": "string"
  }},
  "spatial_position": {{
    "floor_level": "string",
    "elevation": float,
    "building_zone": "string",
    "relative_position": "string"
  }},
  "adjacency_analysis": {{
    "adjacent_elements": ["array_of_adjacent_elements"],
    "shared_boundaries": ["array_of_shared_boundaries"],
    "connectivity_degree": "string"
  }},
  "opening_characteristics": {{
    "total_openings": integer,
    "opening_types": {{"doors": int, "windows": int, "others": int}},
    "access_pattern": "string"
  }},
  "vertical_relationships": {{
    "clear_height": float,
    "vertical_continuity": "string",
    "vertical_connections": ["array_of_connections"]
  }},
  "boundary_conditions": {{
    "exterior_exposure": float,
    "structural_adjacency": ["array_of_structural_elements"],
    "environmental_orientation": ["array_of_orientations"]
  }},
  "complexity_metrics": {{
    "perimeter_area_ratio": float,
    "corner_count": integer,
    "geometric_efficiency": float
  }}
}}
"""

    @staticmethod
    def element_classifier_prompt() -> str:
        """Prompt template for element classification."""
        return """
Infer functional semantics from comprehensive building element data by combining geometric analysis, IFC properties, and contextual information to determine the functional classification and usage characteristics of building elements.

## Input Data:
- IFC Element Type: {ifc_type}
- Element GUID: {guid}
- Geometric Analysis Results: {geometric_features}  # From Geometry Analyzer Agent
- IFC Property Sets: {property_sets}
- Spatial Context: {spatial_context}
- Building Information: {building_context}
- Regional Context: {region_info}
- Project Phase: {project_phase}

## Classification Framework:

### 1. PRIMARY FUNCTION CLASSIFICATION
Determine the primary functional role based on comprehensive evidence:

#### For IfcSpace Elements:
- **Habitable Functions**: [residential, office, commercial, educational, healthcare]
- **Support Functions**: [mechanical, electrical, storage, circulation, service]
- **Special Functions**: [parking, recreational, assembly, industrial, other]

#### For IfcSlab Elements:
- **Structural Functions**: [floor_slab, roof_slab, foundation_slab]
- **Architectural Functions**: [landing, platform, deck]
- **Special Functions**: [equipment_base, ramp, other]

### 2. SUB-CATEGORY CLASSIFICATION
Provide detailed sub-categorization:

#### Space Sub-categories:
- Office: [private_office, open_office, meeting_room, conference_room]
- Residential: [bedroom, living_room, kitchen, bathroom]
- Commercial: [retail, restaurant, showroom, warehouse]
- Support: [mechanical_room, electrical_room, janitor_closet, storage]
- Circulation: [corridor, lobby, stairwell, elevator_shaft]

#### Slab Sub-categories:
- Floor: [typical_floor, mezzanine, raised_floor, access_floor]
- Roof: [flat_roof, pitched_roof, green_roof, mechanical_roof]
- Foundation: [ground_slab, basement_slab, pile_cap]

### 3. USAGE INTENSITY CLASSIFICATION
Assess expected usage patterns:
- **High Intensity**: Frequent daily use, high occupancy
- **Medium Intensity**: Regular use, moderate occupancy
- **Low Intensity**: Occasional use, low occupancy
- **Minimal Intensity**: Rare use, maintenance access only

### 4. REGULATORY CLASSIFICATION HINTS
Provide hints for building regulation compliance:
- **Occupancy Classification**: [A-assembly, B-business, E-educational, I-institutional, M-mercantile, R-residential, S-storage, U-utility]
- **Fire Safety Requirements**: [egress_requirements, fire_rating, sprinkler_requirements]
- **Accessibility Requirements**: [ada_compliance, barrier_free_access]
- **Ventilation Requirements**: [natural_ventilation, mechanical_ventilation, special_exhaust]

## Analysis Process:

### Step 1: Evidence Gathering
Analyze all available evidence:
- Geometric characteristics from spatial analysis
- IFC properties and naming conventions
- Spatial relationships and adjacencies
- Building context and project type
- Regional building practices

### Step 2: Pattern Recognition
Identify classification patterns:
- Size and proportion patterns typical for function
- Adjacency patterns indicating use relationships
- Access patterns suggesting circulation requirements
- Equipment and fixture patterns

### Step 3: Ambiguity Resolution
Address classification uncertainties:
- Identify conflicting evidence
- Assess reliability of different data sources
- Consider multiple possible classifications
- Evaluate context-dependent factors

### Step 4: Confidence Assessment
Evaluate classification confidence:
- Strong evidence alignment = High confidence
- Mixed evidence = Medium confidence
- Insufficient/conflicting evidence = Low confidence

## Output Format:
Provide classification results in JSON format:
{{
  "element_identification": {{
    "ifc_type": "string",
    "guid": "string",
    "classification_timestamp": "ISO_datetime"
  }},
  "primary_classification": {{
    "function_category": "string",
    "confidence_score": float,
    "evidence_strength": "string"
  }},
  "sub_classification": {{
    "detailed_type": "string",
    "usage_intensity": "string",
    "special_characteristics": ["array_of_characteristics"]
  }},
  "regulatory_hints": {{
    "occupancy_classification": "string",
    "fire_safety_requirements": ["array_of_requirements"],
    "accessibility_requirements": ["array_of_requirements"],
    "ventilation_requirements": ["array_of_requirements"]
  }},
  "classification_reasoning": {{
    "supporting_evidence": ["array_of_evidence"],
    "conflicting_evidence": ["array_of_conflicts"],
    "assumptions_made": ["array_of_assumptions"],
    "uncertainty_areas": ["array_of_uncertainties"]
  }},
  "alternative_classifications": [
    {{
      "function_category": "string",
      "confidence_score": float,
      "reasoning": "string"
    }}
  ]
}}
"""

    @staticmethod
    def semantic_alignment_prompt() -> str:
        """Prompt template for semantic alignment."""
        return """
I need to align IFC element semantics with building regulation terminology.

IFC Data:
- Element: {ifc_type}
- Properties: {properties}
- Spatial context: {context}

Regulation Requirements:
- Categories: {reg_categories}
- Rules: {calculation_rules}

Think step by step:
1. What is the most likely functional purpose of this IFC element?
2. Which regulation categories could this element belong to?
3. What are the potential semantic ambiguities?
4. What additional context would resolve uncertainties?
5. What is the confidence level of this alignment?

Provide reasoning and final mapping decision.
"""

    @staticmethod
    def confidence_assessment_prompt() -> str:
        """Prompt template for confidence assessment."""
        return """
Assess the confidence level of the semantic alignment between IFC element and regulation category.

Alignment Data:
- IFC Element: {ifc_element}
- Proposed Category: {proposed_category}
- Supporting Evidence: {evidence}
- Conflicting Evidence: {conflicts}
- Context Information: {context}

Evaluation Criteria:
1. **Evidence Strength** (0.0-1.0):
   - How strong is the supporting evidence?
   - Are there clear geometric/spatial indicators?
   - Do IFC properties align with the category?

2. **Context Consistency** (0.0-1.0):
   - Does the classification fit the building context?
   - Are adjacent elements consistent with this classification?
   - Does the spatial arrangement support this function?

3. **Ambiguity Level** (0.0-1.0, inverted):
   - How many alternative interpretations exist?
   - Are there conflicting indicators?
   - Is additional context needed for certainty?

4. **Regulation Alignment** (0.0-1.0):
   - How well does this match regulation definitions?
   - Are the calculation rules applicable?
   - Does this support compliance checking?

Provide detailed confidence assessment in JSON format:
{{
  "overall_confidence": float,
  "confidence_breakdown": {{
    "evidence_strength": float,
    "context_consistency": float,
    "ambiguity_level": float,
    "regulation_alignment": float
  }},
  "confidence_factors": {{
    "supporting_factors": ["list of factors that increase confidence"],
    "limiting_factors": ["list of factors that decrease confidence"]
  }},
  "improvement_suggestions": ["what additional data would improve confidence"],
  "alternative_interpretations": [
    {{
      "category": "string",
      "likelihood": float,
      "reasoning": "string"
    }}
  ]
}}
"""

    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """Format a prompt template with provided arguments.

        Args:
            template: Prompt template string
            **kwargs: Template variables

        Returns:
            Formatted prompt string
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            # Handle missing template variables gracefully
            missing_var = str(e).strip("'")
            kwargs[missing_var] = f"[{missing_var}_not_provided]"
            return template.format(**kwargs)

    @classmethod
    def get_geometry_analysis_prompt(cls, **context) -> str:
        """Get formatted geometry analysis prompt."""
        return cls.format_prompt(cls.geometry_analyzer_prompt(), **context)

    @classmethod
    def get_element_classification_prompt(cls, **context) -> str:
        """Get formatted element classification prompt."""
        return cls.format_prompt(cls.element_classifier_prompt(), **context)

    @classmethod
    def get_semantic_alignment_prompt(cls, **context) -> str:
        """Get formatted semantic alignment prompt."""
        return cls.format_prompt(cls.semantic_alignment_prompt(), **context)

    @classmethod
    def get_confidence_assessment_prompt(cls, **context) -> str:
        """Get formatted confidence assessment prompt."""
        return cls.format_prompt(cls.confidence_assessment_prompt(), **context)
