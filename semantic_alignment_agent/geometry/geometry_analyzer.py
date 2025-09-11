from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import math
from datetime import datetime

from ..utils import (
    log, Point3D, BoundingBox, GeometricFeatures, IfcElementInfo
)
from ..utils.data_structures import GeometricAnalysisResult, SpatialRelationship
from ..llm.llm_client import LLMClient
from ..llm.prompt_templates import PromptTemplates


@dataclass
class SpatialContext:
    """空间上下文信息"""
    floor_level: float
    building_height: float
    adjacent_spaces: List[str]
    equipment_nearby: bool
    outdoor_exposure: bool
    structural_context: str  # 'roof', 'intermediate', 'ground'


class GeometryAnalyzer:
    """几何特征分析器
    
    Enhanced with LLM capabilities for intelligent geometric analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None, enable_llm: bool = True):
        """Initialize the geometry analyzer.
        
        Args:
            config_path: Path to configuration file
            enable_llm: Whether to enable LLM-enhanced analysis
        """
        # 几何分析参数
        self.thickness_thresholds = {
            'decoration_platform': 0.1,  # <0.1m 装饰平台
            'equipment_slab': 0.15,      # 0.1-0.15m 设备平台
            'structural_slab': 0.15      # >=0.15m 结构楼板
        }
        
        self.height_analysis_params = {
            'typical_floor_height': 3.0,  # 典型层高
            'low_height_threshold': 2.2,  # 低矮空间阈值
            'equipment_height_range': (0.5, 2.0)  # 设备层高度范围
        }
        
        self.spatial_tolerance = 0.2  # 空间分析容差
        self.enable_llm = enable_llm
        
        # Initialize LLM client if enabled
        self.llm_client = None
        if enable_llm:
            try:
                self.llm_client = LLMClient(config_path)
                if self.llm_client.is_available():
                    log.info("LLM-enhanced geometry analysis enabled")
                else:
                    log.warning("LLM client not available, using rule-based analysis only")
            except Exception as e:
                log.warning(f"Failed to initialize LLM client: {e}")
        
        self.prompt_templates = PromptTemplates()
    
    def analyze_slab_geometry(self, element_info: IfcElementInfo, 
                            spatial_context: Optional[SpatialContext] = None) -> GeometricFeatures:
        """分析楼板几何特征"""
        try:
            # 基础几何特征
            bbox = element_info.bounding_box
            thickness = self._calculate_thickness(bbox)
            area = self._calculate_area(element_info.geometry)
            position = self._analyze_position(bbox, spatial_context)
            
            # 厚度指标分析
            thickness_indicator = self._classify_thickness(thickness)
            
            # 位置指标分析
            location_indicator = self._analyze_location_context(bbox, spatial_context)
            
            # 支撑功能分析
            support_function = self._analyze_support_function(element_info, spatial_context)
            
            # 几何复杂度
            complexity = self._calculate_geometric_complexity(element_info.geometry)
            
            # 构建几何特征
            features = GeometricFeatures(
                thickness=thickness,
                area=area,
                position=position,
                thickness_indicator=thickness_indicator,
                location_indicator=location_indicator,
                support_function=support_function,
                geometric_complexity=complexity,
                spatial_relationships=self._analyze_spatial_relationships(element_info, spatial_context)
            )
            
            log.debug(f"Analyzed slab geometry for {element_info.guid}: thickness={thickness:.3f}m, area={area:.2f}m²")
            return features
            
        except Exception as e:
            log.error(f"Error analyzing slab geometry for {element_info.guid}: {e}")
            return self._create_default_features()
    
    def analyze_space_geometry(self, element_info: IfcElementInfo,
                             spatial_context: Optional[SpatialContext] = None) -> GeometricFeatures:
        """分析空间几何特征"""
        try:
            bbox = element_info.bounding_box
            
            # 空间尺寸分析
            volume = self._calculate_volume(bbox)
            height = bbox.max_point.z - bbox.min_point.z
            floor_area = self._calculate_floor_area(element_info.geometry)
            
            # 垂直贯穿分析
            vertical_span = self._analyze_vertical_span(element_info, spatial_context)
            
            # 空间形状分析
            aspect_ratio = self._calculate_aspect_ratio(bbox)
            shape_regularity = self._analyze_shape_regularity(element_info.geometry)
            
            # 位置分析
            position = self._analyze_position(bbox, spatial_context)
            location_indicator = self._analyze_space_location(element_info, spatial_context)
            
            features = GeometricFeatures(
                thickness=height,  # 对于空间，厚度表示高度
                area=floor_area,
                position=position,
                volume=volume,
                vertical_span=vertical_span,
                aspect_ratio=aspect_ratio,
                shape_regularity=shape_regularity,
                location_indicator=location_indicator,
                spatial_relationships=self._analyze_spatial_relationships(element_info, spatial_context)
            )
            
            log.debug(f"Analyzed space geometry for {element_info.guid}: height={height:.3f}m, area={floor_area:.2f}m²")
            return features
            
        except Exception as e:
            log.error(f"Error analyzing space geometry for {element_info.guid}: {e}")
            return self._create_default_features()
    
    def _calculate_thickness(self, bbox: BoundingBox) -> float:
        """计算厚度（Z方向最小尺寸）"""
        return abs(bbox.max_point.z - bbox.min_point.z)
    
    def _calculate_area(self, geometry: Any) -> float:
        """计算面积"""
        try:
            if hasattr(geometry, 'area'):
                return float(geometry.area)
            elif hasattr(geometry, 'Area'):
                return float(geometry.Area)
            else:
                # 从边界框估算
                bbox = self._extract_bounding_box(geometry)
                return abs((bbox.max_point.x - bbox.min_point.x) * 
                          (bbox.max_point.y - bbox.min_point.y))
        except:
            return 0.0
    
    def _calculate_volume(self, bbox: BoundingBox) -> float:
        """计算体积"""
        return abs((bbox.max_point.x - bbox.min_point.x) * 
                  (bbox.max_point.y - bbox.min_point.y) * 
                  (bbox.max_point.z - bbox.min_point.z))
    
    def _calculate_floor_area(self, geometry: Any) -> float:
        """计算楼面面积（水平投影面积）"""
        try:
            # 简化实现：使用边界框的水平面积
            bbox = self._extract_bounding_box(geometry)
            return abs((bbox.max_point.x - bbox.min_point.x) * 
                      (bbox.max_point.y - bbox.min_point.y))
        except:
            return 0.0
    
    def _classify_thickness(self, thickness: float) -> str:
        """根据厚度分类"""
        if thickness < self.thickness_thresholds['decoration_platform']:
            return 'decoration_platform'
        elif thickness < self.thickness_thresholds['equipment_slab']:
            return 'equipment_slab'
        else:
            return 'structural_slab'
    
    def _analyze_position(self, bbox: BoundingBox, 
                         spatial_context: Optional[SpatialContext] = None) -> Point3D:
        """分析位置（中心点）"""
        return Point3D(
            x=(bbox.min_point.x + bbox.max_point.x) / 2,
            y=(bbox.min_point.y + bbox.max_point.y) / 2,
            z=(bbox.min_point.z + bbox.max_point.z) / 2
        )
    
    def _analyze_location_context(self, bbox: BoundingBox, 
                                 spatial_context: Optional[SpatialContext] = None) -> str:
        """分析位置上下文"""
        if not spatial_context:
            return 'unknown'
        
        element_z = (bbox.min_point.z + bbox.max_point.z) / 2
        
        # 判断是否在屋顶
        if element_z > spatial_context.building_height * 0.9:
            if spatial_context.equipment_nearby:
                return 'rooftop_equipment'
            else:
                return 'rooftop_structural'
        
        # 判断是否在地面层
        elif element_z < spatial_context.floor_level + 1.0:
            return 'ground_level'
        
        # 中间层
        else:
            if spatial_context.equipment_nearby:
                return 'intermediate_equipment'
            else:
                return 'intermediate_structural'
    
    def _analyze_space_location(self, element_info: IfcElementInfo,
                               spatial_context: Optional[SpatialContext] = None) -> str:
        """分析空间位置上下文"""
        if not spatial_context:
            return 'unknown'
        
        bbox = element_info.bounding_box
        space_height = bbox.max_point.z - bbox.min_point.z
        
        # 判断是否为低矮空间
        if space_height < self.height_analysis_params['low_height_threshold']:
            if spatial_context.equipment_nearby:
                return 'low_equipment_space'
            else:
                return 'low_auxiliary_space'
        
        # 判断室外暴露
        if spatial_context.outdoor_exposure:
            return 'outdoor_construction'
        
        # 根据功能推断
        space_name = element_info.properties.get('Name', '').lower()
        if any(keyword in space_name for keyword in ['消防', '水池', '停车', '货棚', '垃圾']):
            return 'auxiliary_construction'
        
        return 'habitable_space'
    
    def _analyze_support_function(self, element_info: IfcElementInfo,
                                 spatial_context: Optional[SpatialContext] = None) -> str:
        """分析支撑功能"""
        try:
            # 从属性推断
            load_bearing = element_info.properties.get('LoadBearing', False)
            if load_bearing:
                return 'supports_occupancy'
            
            # 从位置和上下文推断
            if spatial_context and spatial_context.equipment_nearby:
                return 'supports_equipment'
            
            # 从厚度推断
            bbox = element_info.bounding_box
            thickness = self._calculate_thickness(bbox)
            
            if thickness >= self.thickness_thresholds['structural_slab']:
                return 'supports_occupancy'
            else:
                return 'supports_equipment'
                
        except:
            return 'unknown'
    
    def _analyze_vertical_span(self, element_info: IfcElementInfo,
                              spatial_context: Optional[SpatialContext] = None) -> Dict[str, Any]:
        """分析垂直贯穿特征"""
        bbox = element_info.bounding_box
        height = bbox.max_point.z - bbox.min_point.z
        
        # 估算跨越楼层数
        typical_floor_height = self.height_analysis_params['typical_floor_height']
        floors_spanned = max(1, int(height / typical_floor_height))
        
        return {
            'total_height': height,
            'floors_spanned': floors_spanned,
            'is_multi_story': floors_spanned >= 2,
            'vertical_continuity': floors_spanned >= 2  # 简化判断
        }
    
    def _calculate_aspect_ratio(self, bbox: BoundingBox) -> float:
        """计算长宽比"""
        width = abs(bbox.max_point.x - bbox.min_point.x)
        length = abs(bbox.max_point.y - bbox.min_point.y)
        
        if min(width, length) > 0:
            return max(width, length) / min(width, length)
        else:
            return 1.0
    
    def _calculate_geometric_complexity(self, geometry: Any) -> float:
        """计算几何复杂度（简化实现）"""
        try:
            # 基于顶点数量的简单复杂度评估
            if hasattr(geometry, 'vertices'):
                vertex_count = len(geometry.vertices)
                return min(1.0, vertex_count / 100.0)  # 归一化到0-1
            else:
                return 0.5  # 默认中等复杂度
        except:
            return 0.5
    
    def _analyze_shape_regularity(self, geometry: Any) -> float:
        """分析形状规整度"""
        try:
            # 简化实现：基于边界框与实际几何的面积比
            bbox = self._extract_bounding_box(geometry)
            bbox_area = abs((bbox.max_point.x - bbox.min_point.x) * 
                           (bbox.max_point.y - bbox.min_point.y))
            
            actual_area = self._calculate_area(geometry)
            
            if bbox_area > 0:
                regularity = actual_area / bbox_area
                return min(1.0, regularity)  # 规整度在0-1之间
            else:
                return 0.5
        except:
            return 0.5
    
    def _analyze_spatial_relationships(self, element_info: IfcElementInfo,
                                     spatial_context: Optional[SpatialContext] = None) -> Dict[str, Any]:
        """分析空间关系"""
        relationships = {
            'adjacent_elements': [],
            'contained_spaces': [],
            'supporting_elements': [],
            'equipment_proximity': False
        }
        
        if spatial_context:
            relationships['adjacent_spaces'] = spatial_context.adjacent_spaces
            relationships['equipment_proximity'] = spatial_context.equipment_nearby
            relationships['outdoor_exposure'] = spatial_context.outdoor_exposure
        
        # 从IFC关系中提取（简化实现）
        if hasattr(element_info, 'relationships'):
            for rel_type, related_elements in element_info.relationships.items():
                if rel_type == 'IfcRelSpaceBoundary':
                    relationships['contained_spaces'].extend(related_elements)
                elif rel_type == 'IfcRelConnectsElements':
                    relationships['adjacent_elements'].extend(related_elements)
        
        return relationships
    
    def _extract_bounding_box(self, geometry: Any) -> BoundingBox:
        """从几何体提取边界框"""
        try:
            if hasattr(geometry, 'BoundingBox'):
                bbox = geometry.BoundingBox
                return BoundingBox(
                    min_point=Point3D(bbox.XDim.Min, bbox.YDim.Min, bbox.ZDim.Min),
                    max_point=Point3D(bbox.XDim.Max, bbox.YDim.Max, bbox.ZDim.Max)
                )
            else:
                # 默认边界框
                return BoundingBox(
                    min_point=Point3D(0, 0, 0),
                    max_point=Point3D(1, 1, 1)
                )
        except:
            return BoundingBox(
                min_point=Point3D(0, 0, 0),
                max_point=Point3D(1, 1, 1)
            )
    
    def _create_default_features(self) -> GeometricFeatures:
        """创建默认几何特征"""
        return GeometricFeatures(
            thickness=0.0,
            area=0.0,
            position=Point3D(0, 0, 0),
            thickness_indicator='unknown',
            location_indicator='unknown',
            support_function='unknown'
        )
    
    def create_spatial_context(self, building_info: Dict[str, Any], 
                              floor_level: float = 0.0) -> SpatialContext:
        """创建空间上下文"""
        return SpatialContext(
            floor_level=floor_level,
            building_height=building_info.get('building_height', 30.0),
            adjacent_spaces=building_info.get('adjacent_spaces', []),
            equipment_nearby=building_info.get('equipment_nearby', False),
            outdoor_exposure=building_info.get('outdoor_exposure', False),
            structural_context=building_info.get('structural_context', 'intermediate')
        )
    
    def batch_analyze_elements(self, elements: List[IfcElementInfo],
                              spatial_context: Optional[SpatialContext] = None) -> Dict[str, GeometricFeatures]:
        """批量分析元素几何特征"""
        results = {}
        
        for element in elements:
            try:
                if element.ifc_type == 'IfcSlab':
                    features = self.analyze_slab_geometry(element, spatial_context)
                elif element.ifc_type == 'IfcSpace':
                    features = self.analyze_space_geometry(element, spatial_context)
                else:
                    # 其他类型使用通用分析
                    features = self.analyze_slab_geometry(element, spatial_context)
                
                results[element.guid] = features
                
            except Exception as e:
                log.error(f"Error analyzing element {element.guid}: {e}")
                results[element.guid] = self._create_default_features()
        
        log.info(f"Analyzed geometry features for {len(results)} elements")
        return results
    
    def analyze_element_geometry_enhanced(
        self, 
        element_info: IfcElementInfo,
        spatial_context: Optional[SpatialContext] = None,
        building_context: Optional[Dict[str, Any]] = None
    ) -> GeometricAnalysisResult:
        """Enhanced geometric analysis using LLM capabilities.
        
        Args:
            element_info: IFC element information
            spatial_context: Spatial context information
            building_context: Building-level context
            
        Returns:
            Enhanced geometric analysis result
        """
        # First perform standard geometric analysis
        if element_info.ifc_type == 'IfcSlab':
            basic_features = self.analyze_slab_geometry(element_info, spatial_context)
        elif element_info.ifc_type == 'IfcSpace':
            basic_features = self.analyze_space_geometry(element_info, spatial_context)
        else:
            basic_features = self.analyze_slab_geometry(element_info, spatial_context)
        
        # Try LLM-enhanced analysis if available
        if self.enable_llm and self.llm_client and self.llm_client.is_available():
            try:
                return self._llm_enhanced_analysis(
                    element_info, basic_features, spatial_context, building_context
                )
            except Exception as e:
                log.warning(f"LLM analysis failed: {e}, using rule-based analysis")
        
        # Fallback to rule-based enhanced analysis
        return self._rule_based_enhanced_analysis(element_info, basic_features)
    
    def _llm_enhanced_analysis(
        self,
        element_info: IfcElementInfo,
        basic_features: GeometricFeatures,
        spatial_context: Optional[SpatialContext],
        building_context: Optional[Dict[str, Any]]
    ) -> GeometricAnalysisResult:
        """Perform LLM-enhanced geometric analysis."""
        # Prepare context for LLM analysis
        bbox = element_info.bounding_box
        dimensions = {
            'length': abs(bbox.max_point.x - bbox.min_point.x),
            'width': abs(bbox.max_point.y - bbox.min_point.y),
            'height': abs(bbox.max_point.z - bbox.min_point.z)
        }
        
        context = {
            'ifc_type': element_info.ifc_type,
            'guid': element_info.guid,
            'geometry_data': str(element_info.geometry),
            'coordinates': str(basic_features.position),
            'dimensions': str(dimensions),
            'length': dimensions['length'],
            'width': dimensions['width'],
            'height': dimensions['height'],
            'volume': getattr(basic_features, 'volume', 0),
            'area': basic_features.area,
            'related_elements': str(spatial_context.adjacent_spaces if spatial_context else []),
            'building_info': str(building_context or {}),
            'floor_level': spatial_context.floor_level if spatial_context else 0,
            'elevation': basic_features.position.z,
            'adjacent_elements': str(spatial_context.adjacent_spaces if spatial_context else []),
            'connection_types': 'direct',
            'shared_boundary_elements': str([]),
            'proximity_distance': '5.0',
            'nearby_elements': str([]),
            'opening_count': 0,
            'door_openings': 0,
            'window_openings': 0,
            'generic_openings': 0,
            'avg_opening_size': 0,
            'wall_positions': 'various',
            'clear_height': dimensions['height'],
            'elements_above': str([]),
            'elements_below': str([]),
            'exterior_wall_count': 0,
            'exposure_length': 0,
            'interior_wall_count': 0,
            'structural_elements': str([]),
            'perimeter_area_ratio': basic_features.area / max(getattr(basic_features, 'volume', 1), 1),
            'corner_count': 4,
            'efficiency_ratio': 0.8
        }
        
        # Generate LLM prompt
        prompt = self.prompt_templates.get_geometry_analysis_prompt(**context)
        
        # Get LLM analysis
        llm_result = self.llm_client.analyze_with_confidence(
            prompt, context, temperature=0.1
        )
        
        # Parse LLM response into structured format
        return self._parse_llm_geometry_result(llm_result, element_info, basic_features)
    
    def _parse_llm_geometry_result(
        self,
        llm_result: Dict[str, Any],
        element_info: IfcElementInfo,
        basic_features: GeometricFeatures
    ) -> GeometricAnalysisResult:
        """Parse LLM geometry analysis result."""
        try:
            # Try to extract structured JSON from LLM analysis
            analysis_text = llm_result.get('analysis', '')
            
            # Create structured result based on LLM analysis
            bbox = element_info.bounding_box
            dimensions = {
                'length': abs(bbox.max_point.x - bbox.min_point.x),
                'width': abs(bbox.max_point.y - bbox.min_point.y),
                'height': abs(bbox.max_point.z - bbox.min_point.z)
            }
            
            return GeometricAnalysisResult(
                element_identification={
                    'ifc_type': element_info.ifc_type,
                    'guid': element_info.guid,
                    'analysis_timestamp': datetime.now().isoformat()
                },
                dimensional_characteristics={
                    'primary_dimensions': dimensions,
                    'calculated_metrics': {
                        'area': basic_features.area,
                        'volume': getattr(basic_features, 'volume', 0),
                        'aspect_ratio': getattr(basic_features, 'aspect_ratio', 1.0)
                    },
                    'size_category': self._categorize_size(basic_features.area)
                },
                geometric_form={
                    'basic_shape': getattr(basic_features, 'shape_type', 'unknown'),
                    'shape_complexity': self._assess_complexity(getattr(basic_features, 'geometric_complexity', 0.5)),
                    'regularity_level': 'regular' if getattr(basic_features, 'geometric_complexity', 0.5) < 0.5 else 'irregular'
                },
                spatial_position={
                    'floor_level': 'unknown',
                    'elevation': basic_features.position.z,
                    'building_zone': 'unknown',
                    'relative_position': 'unknown'
                },
                adjacency_analysis={
                    'adjacent_elements': [],
                    'shared_boundaries': [],
                    'connectivity_degree': 'unknown'
                },
                opening_characteristics={
                    'total_openings': 0,
                    'opening_types': {'doors': 0, 'windows': 0, 'others': 0},
                    'access_pattern': 'unknown'
                },
                vertical_relationships={
                    'clear_height': dimensions['height'],
                    'vertical_continuity': 'single_floor',
                    'vertical_connections': []
                },
                boundary_conditions={
                    'exterior_exposure': 0.0,
                    'structural_adjacency': [],
                    'environmental_orientation': []
                },
                complexity_metrics={
                    'perimeter_area_ratio': getattr(basic_features, 'geometric_complexity', 0.5),
                    'corner_count': 4,
                    'geometric_efficiency': 0.8
                },
                confidence_score=llm_result.get('confidence_score', 0.7),
                analysis_method='llm',
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            log.error(f"Failed to parse LLM geometry result: {e}")
            return self._rule_based_enhanced_analysis(element_info, basic_features)
    
    def _rule_based_enhanced_analysis(
        self,
        element_info: IfcElementInfo,
        basic_features: GeometricFeatures
    ) -> GeometricAnalysisResult:
        """Fallback rule-based enhanced analysis."""
        bbox = element_info.bounding_box
        dimensions = {
            'length': abs(bbox.max_point.x - bbox.min_point.x),
            'width': abs(bbox.max_point.y - bbox.min_point.y),
            'height': abs(bbox.max_point.z - bbox.min_point.z)
        }
        
        return GeometricAnalysisResult(
            element_identification={
                'ifc_type': element_info.ifc_type,
                'guid': element_info.guid,
                'analysis_timestamp': datetime.now().isoformat()
            },
            dimensional_characteristics={
                'primary_dimensions': dimensions,
                'calculated_metrics': {
                    'area': basic_features.area,
                    'volume': getattr(basic_features, 'volume', 0),
                    'aspect_ratio': getattr(basic_features, 'aspect_ratio', 1.0)
                },
                'size_category': self._categorize_size(basic_features.area)
            },
            geometric_form={
                'basic_shape': getattr(basic_features, 'shape_type', 'unknown'),
                'shape_complexity': self._assess_complexity(getattr(basic_features, 'geometric_complexity', 0.5)),
                'regularity_level': 'regular' if getattr(basic_features, 'geometric_complexity', 0.5) < 0.5 else 'irregular'
            },
            spatial_position={
                'floor_level': 'unknown',
                'elevation': basic_features.position.z,
                'building_zone': 'unknown',
                'relative_position': 'unknown'
            },
            adjacency_analysis={
                'adjacent_elements': [],
                'shared_boundaries': [],
                'connectivity_degree': 'low'
            },
            opening_characteristics={
                'total_openings': 0,
                'opening_types': {'doors': 0, 'windows': 0, 'others': 0},
                'access_pattern': 'unknown'
            },
            vertical_relationships={
                'clear_height': dimensions['height'],
                'vertical_continuity': 'single_floor',
                'vertical_connections': []
            },
            boundary_conditions={
                'exterior_exposure': 0.0,
                'structural_adjacency': [],
                'environmental_orientation': []
            },
            complexity_metrics={
                'perimeter_area_ratio': getattr(basic_features, 'geometric_complexity', 0.5),
                'corner_count': 4,
                'geometric_efficiency': 0.8
            },
            confidence_score=0.6,
            analysis_method='rule_based',
            timestamp=datetime.now().isoformat()
        )
    
    def _categorize_size(self, area: float) -> str:
        """Categorize element size based on area."""
        if area < 2:
            return 'tiny'
        elif area < 10:
            return 'small'
        elif area < 50:
            return 'medium'
        elif area < 200:
            return 'large'
        else:
            return 'very_large'
    
    def _assess_complexity(self, complexity_score: float) -> str:
        """Assess geometric complexity level."""
        if complexity_score < 0.3:
            return 'simple_polygon'
        elif complexity_score < 0.6:
            return 'complex_polygon'
        elif complexity_score < 0.8:
            return 'curved_boundaries'
        else:
            return 'multi_part_geometry'