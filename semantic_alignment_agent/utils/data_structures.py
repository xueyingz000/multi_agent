from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from enum import Enum


class RegulationCategory(Enum):
    """法规分类枚举"""

    EXCLUDE = "exclude"
    INCLUDE_PARTIAL = "include_partial"
    INCLUDE_FULL = "include_full"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


class IfcElementType(Enum):
    """IFC元素类型枚举"""

    SLAB = "IfcSlab"
    SPACE = "IfcSpace"
    OPENING_ELEMENT = "IfcOpeningElement"
    VOID = "IfcVoid"
    SHAFT = "IfcShaft"
    BUILDING_ELEMENT_PROXY = "IfcBuildingElementProxy"


class FunctionType(Enum):
    """功能类型枚举"""

    STRUCTURAL_SLAB = "structural_slab"
    EQUIPMENT_PLATFORM = "equipment_platform"
    DECORATION_PLATFORM = "decoration_platform"
    HABITABLE_OFFICE = "habitable_office"
    MECHANICAL_ROOM = "mechanical_room"
    AUXILIARY_CONSTRUCTION = "auxiliary_construction"
    ATRIUM = "atrium"
    SHAFT = "shaft"
    STAIRWELL = "stairwell"
    UNKNOWN = "unknown"


class VerticalSpaceType(Enum):
    """垂直空间类型枚举"""

    ATRIUM = "atrium"
    SHAFT = "shaft"
    STAIRCASE = "staircase"
    ELEVATOR_SHAFT = "elevator_shaft"
    OPENING = "opening"
    VOID = "void"


@dataclass
class Point3D:
    """三维点"""

    x: float
    y: float
    z: float


@dataclass
class BoundingBox:
    """边界框"""

    min_point: Point3D
    max_point: Point3D

    @property
    def width(self) -> float:
        return self.max_point.x - self.min_point.x

    @property
    def depth(self) -> float:
        return self.max_point.y - self.min_point.y

    @property
    def height(self) -> float:
        return self.max_point.z - self.min_point.z

    @property
    def center(self) -> Point3D:
        return Point3D(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2,
        )


@dataclass
class GeometricFeatures:
    """几何特征"""

    bounding_box: BoundingBox
    area: float
    volume: float
    thickness: Optional[float] = None
    centroid: Optional[Point3D] = None
    floor_level: Optional[float] = None
    floors_spanned: int = 1
    is_vertical_element: bool = False
    cross_section_area: Optional[float] = None


@dataclass
class Evidence:
    """证据信息"""

    text: str
    source: str
    confidence: float = 1.0
    start: int = -1
    end: int = -1


@dataclass
class FunctionalInference:
    """功能推断结果"""

    primary_function: FunctionType
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    alternatives: List[Tuple[FunctionType, float]] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class IfcElementInfo:
    """IFC元素信息"""

    guid: str
    ifc_type: IfcElementType
    name: Optional[str] = None
    description: Optional[str] = None
    predefined_type: Optional[str] = None
    object_type: Optional[str] = None
    geometric_features: Optional[GeometricFeatures] = None
    functional_inference: Optional[FunctionalInference] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    material_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerticalSpaceInfo:
    """垂直空间信息"""

    space_id: str
    space_type: VerticalSpaceType
    floors_penetrated: List[str]
    geometric_features: GeometricFeatures
    is_continuous: bool
    deduction_rule: str  # "deduct_per_floor", "no_deduction"
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class AlignmentDecision:
    """对齐决策"""

    element_id: str
    regulation_category: RegulationCategory
    coefficient: float
    confidence: float
    reasoning_path: str
    evidence: List[Evidence] = field(default_factory=list)
    category_type: str  # "A1", "A2", "B1", "B2"
    requires_review: bool = False


@dataclass
class SemanticAlignmentResult:
    """语义对齐结果"""

    alignment_decisions: List[AlignmentDecision]
    vertical_spaces: List[VerticalSpaceInfo]
    confidence_assessment: Dict[str, float]
    processing_summary: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class RegulationRule:
    """法规规则（从regulation analysis agent输出解析）"""

    region: str
    rule_type: str  # "height", "cover_enclosure", "special_use"
    feature_key: str
    label: str  # "full", "half", "excluded", "conditional"
    coefficient: float
    conditions: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class ProcessingContext:
    """处理上下文"""

    ifc_file_path: str
    regulation_rules: List[RegulationRule]
    target_region: str
    building_type: Optional[str] = None
    processing_options: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    """Result of semantic alignment analysis."""

    element_guid: str
    element_type: str
    function_classification: str
    confidence_score: float
    reasoning: str
    alternative_classifications: List[Dict[str, Any]]
    timestamp: str


@dataclass
class ClassificationResult:
    """Result of LLM-based element classification."""

    element_guid: str
    element_type: str
    primary_function: str
    sub_category: str
    usage_intensity: str
    confidence_score: float
    reasoning: str
    alternative_classifications: List[Dict[str, Any]]
    regulatory_hints: List[str]
    timestamp: str


@dataclass
class LLMAnalysisResult:
    """Result of LLM analysis with confidence metrics."""

    analysis: str
    conclusion: str
    confidence_score: float
    confidence_factors: List[str]
    uncertainty_areas: List[str]
    additional_context_needed: List[str]
    timestamp: str


@dataclass
class GeometricAnalysisResult:
    """Enhanced geometric analysis result from LLM."""

    element_identification: Dict[str, Any]
    dimensional_characteristics: Dict[str, Any]
    geometric_form: Dict[str, Any]
    spatial_position: Dict[str, Any]
    adjacency_analysis: Dict[str, Any]
    opening_characteristics: Dict[str, Any]
    vertical_relationships: Dict[str, Any]
    boundary_conditions: Dict[str, Any]
    complexity_metrics: Dict[str, Any]
    confidence_score: float
    analysis_method: str  # 'llm' or 'rule_based'
    timestamp: str
