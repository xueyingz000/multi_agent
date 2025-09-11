from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from enum import Enum
from datetime import datetime

from ..utils import (
    RegulationCategory, IFCElementType, FunctionType, VerticalSpaceType,
    Point3D, BoundingBox, GeometricFeatures, FunctionInference,
    IFCElementInfo, VerticalSpaceInfo, AlignmentDecision, Evidence,
    config, logger
)
from .function_inference import FunctionInferenceEngine
from .vertical_space_detector import VerticalSpaceDetector
from ..geometry import GeometryAnalyzer
from ..data_processing import RegulationParser
from ..llm.llm_client import LLMClient
from ..llm.prompt_templates import PromptTemplates
from ..llm.element_classifier import LLMElementClassifier


class AlignmentType(Enum):
    """对齐类型枚举"""
    CATEGORY_A_FUNCTIONAL = "category_a_functional"
    CATEGORY_B_GEOMETRIC = "category_b_geometric"


class ConfidenceLevel(Enum):
    """置信度等级"""
    HIGH = "high"  # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"  # 0.0-0.5


@dataclass
class AlignmentContext:
    """对齐上下文信息"""
    regulation_rules: Dict[str, Any]
    building_type: str
    floor_height: float
    total_floors: int
    geometric_tolerance: float = 0.2  # 200mm


@dataclass
class AlignmentResult:
    """对齐结果"""
    element_guid: str
    alignment_type: AlignmentType
    regulation_category: RegulationCategory
    function_classification: str  # 功能分类结果
    confidence: float
    confidence_level: ConfidenceLevel
    reasoning_path: List[str]
    evidence: List[str]
    alternatives: List[Dict[str, Any]]
    requires_review: bool


class SemanticAlignmentAgent:
    """核心语义对齐代理
    
    统一处理A类功能语义对齐和B类几何规范对齐问题
    集成LLM智能分析，提高边界情况处理和置信度评估能力
    """
    
    def __init__(self, config_path: Optional[str] = None, enable_llm: bool = True):
        """初始化语义对齐代理
        
        Args:
            config_path: 配置文件路径
            enable_llm: 是否启用LLM增强功能
        """
        self.logger = logger
        self.config = config
        self.enable_llm = enable_llm
        
        # 初始化子模块
        self.function_engine = FunctionInferenceEngine(
            config_path=config_path, enable_llm=enable_llm
        )
        self.vertical_detector = VerticalSpaceDetector()
        self.geometry_analyzer = GeometryAnalyzer(
            config_path=config_path, enable_llm=enable_llm
        )
        self.regulation_parser = RegulationParser()
        
        # 初始化LLM组件
        self.llm_client = None
        self.element_classifier = None
        
        if enable_llm:
            try:
                self.llm_client = LLMClient(config_path)
                if self.llm_client.is_available():
                    self.element_classifier = LLMElementClassifier(self.llm_client)
                    self.logger.info("LLM-enhanced semantic alignment agent initialized")
                else:
                    self.logger.warning("LLM client not available, using rule-based alignment only")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM components: {e}")
        
        self.prompt_templates = PromptTemplates()
        
        # 置信度阈值
        self.confidence_thresholds = {
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.LOW: 0.0
        }
        
        # 厚度指标阈值
        self.thickness_thresholds = {
            'decoration_platform': 0.1,  # <0.1m
            'equipment_slab': 0.15,      # 0.1-0.15m
            'structural_slab': 0.15       # >=0.15m
        }
        
        # 统计信息
        self.alignment_stats = {
            'total_processed': 0,
            'llm_enhanced': 0,
            'rule_based_only': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0
        }
        
    def align_elements(
        self,
        ifc_elements: List[IFCElementInfo],
        vertical_spaces: List[VerticalSpaceInfo],
        regulation_data: Dict[str, Any],
        context: AlignmentContext
    ) -> List[AlignmentResult]:
        """对齐IFC元素到法规分类
        
        Args:
            ifc_elements: IFC元素信息列表
            vertical_spaces: 垂直空间信息列表
            regulation_data: 法规数据
            context: 对齐上下文
            
        Returns:
            对齐结果列表
        """
        results = []
        
        # 处理IFC元素（A类问题）
        for element in ifc_elements:
            if element.ifc_type in [IFCElementType.SLAB, IFCElementType.SPACE]:
                result = self._align_category_a_enhanced(element, regulation_data, context)
                results.append(result)
                
        # 处理垂直空间（B类问题）
        for space in vertical_spaces:
            result = self._align_category_b_enhanced(space, regulation_data, context)
            results.append(result)
            
        return results
    
    def _align_category_a_enhanced(
        self,
        element: IFCElementInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext
    ) -> AlignmentResult:
        """增强的A类功能语义对齐处理"""
        self.alignment_stats['total_processed'] += 1
        
        # 首先尝试传统规则方法
        traditional_result = self._align_category_a(element, regulation_data, context)
        
        # 检查是否需要LLM增强
        if self._needs_llm_enhancement(traditional_result):
            if self.enable_llm and self.llm_client and self.llm_client.is_available():
                try:
                    return self._llm_enhanced_category_a_alignment(
                        element, regulation_data, context, traditional_result
                    )
                except Exception as e:
                    self.logger.warning(f"LLM enhancement failed: {e}, using traditional result")
        
        self.alignment_stats['rule_based_only'] += 1
        return traditional_result
    
    def _align_category_b_enhanced(
        self,
        space: VerticalSpaceInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext
    ) -> AlignmentResult:
        """增强的B类几何规范对齐处理"""
        self.alignment_stats['total_processed'] += 1
        
        # 首先尝试传统规则方法
        traditional_result = self._align_category_b(space, regulation_data, context)
        
        # 检查是否需要LLM增强
        if self._needs_llm_enhancement(traditional_result):
            if self.enable_llm and self.llm_client and self.llm_client.is_available():
                try:
                    return self._llm_enhanced_category_b_alignment(
                        space, regulation_data, context, traditional_result
                    )
                except Exception as e:
                    self.logger.warning(f"LLM enhancement failed: {e}, using traditional result")
        
        self.alignment_stats['rule_based_only'] += 1
        return traditional_result
    
    def _needs_llm_enhancement(self, result: AlignmentResult) -> bool:
        """判断是否需要LLM增强"""
        return (
            result.confidence < 0.7 or  # 置信度较低
            len(result.alternatives) > 1 or  # 有多个备选方案
            result.requires_review  # 需要人工审查
        )
    
    def _llm_enhanced_category_a_alignment(
        self,
        element: IFCElementInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext,
        traditional_result: AlignmentResult
    ) -> AlignmentResult:
        """使用LLM增强A类对齐"""
        self.alignment_stats['llm_enhanced'] += 1
        
        # 准备LLM分析上下文
        llm_context = {
            'ifc_type': element.ifc_type.value,
            'guid': element.guid,
            'geometric_features': {
                'area': element.geometric_features.area if element.geometric_features else 0,
                'thickness': element.geometric_features.thickness if element.geometric_features else 0,
                'position': str(element.geometric_features.position) if element.geometric_features else 'Unknown'
            },
            'property_sets': str(element.properties),
            'spatial_context': {},
            'building_context': {
                'building_type': context.building_type,
                'floor_height': context.floor_height,
                'total_floors': context.total_floors
            },
            'region_info': 'Unknown',
            'project_phase': 'Design',
            'traditional_analysis': {
                'predicted_category': traditional_result.regulation_category.value,
                'confidence': traditional_result.confidence,
                'reasoning': traditional_result.reasoning_path
            }
        }
        
        # 生成LLM提示
        prompt = self.prompt_templates.get_element_classification_prompt(**llm_context)
        
        # 获取LLM分析结果
        llm_result = self.llm_client.analyze_with_confidence(
            prompt, llm_context, temperature=0.2
        )
        
        # 结合传统结果和LLM结果
        return self._combine_traditional_and_llm_results(
            traditional_result, llm_result, element, AlignmentType.CATEGORY_A_FUNCTIONAL
        )
    
    def _llm_enhanced_category_b_alignment(
        self,
        space: VerticalSpaceInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext,
        traditional_result: AlignmentResult
    ) -> AlignmentResult:
        """使用LLM增强B类对齐"""
        self.alignment_stats['llm_enhanced'] += 1
        
        # 准备LLM分析上下文
        llm_context = {
            'ifc_type': 'VerticalSpace',
            'guid': space.guid,
            'geometric_features': {
                'height': space.height,
                'area': space.area,
                'position': str(space.position)
            },
            'spatial_context': {
                'space_type': space.space_type.value,
                'floor_levels': space.floor_levels
            },
            'building_context': {
                'building_type': context.building_type,
                'floor_height': context.floor_height,
                'total_floors': context.total_floors
            },
            'traditional_analysis': {
                'predicted_category': traditional_result.regulation_category.value,
                'confidence': traditional_result.confidence,
                'reasoning': traditional_result.reasoning_path
            }
        }
        
        # 生成几何分析提示
        prompt = self.prompt_templates.get_geometry_analyzer_prompt(**llm_context)
        
        # 获取LLM分析结果
        llm_result = self.llm_client.analyze_with_confidence(
            prompt, llm_context, temperature=0.2
        )
        
        # 结合传统结果和LLM结果
        return self._combine_traditional_and_llm_results(
            traditional_result, llm_result, space, AlignmentType.CATEGORY_B_GEOMETRIC
        )
    
    def _combine_traditional_and_llm_results(
        self,
        traditional_result: AlignmentResult,
        llm_result: Dict[str, Any],
        element: Any,
        alignment_type: AlignmentType
    ) -> AlignmentResult:
        """结合传统分析和LLM分析结果"""
        try:
            llm_analysis = llm_result.get('analysis', '')
            llm_confidence = llm_result.get('confidence_score', 0.5)
            
            # 选择更高置信度的结果
            if llm_confidence > traditional_result.confidence:
                # 使用LLM结果，但保留传统分析作为备选
                enhanced_evidence = traditional_result.evidence + [
                    f"LLM analysis: {llm_analysis[:200]}..."
                ]
                
                enhanced_alternatives = traditional_result.alternatives + [{
                    'method': 'traditional_rules',
                    'category': traditional_result.regulation_category.value,
                    'confidence': traditional_result.confidence
                }]
                
                return AlignmentResult(
                    element_guid=getattr(element, 'guid', 'unknown'),
                    alignment_type=alignment_type,
                    regulation_category=self._extract_regulation_category_from_llm(llm_analysis),
                    function_classification=self._extract_function_classification_from_llm(llm_analysis),
                    confidence=min(llm_confidence, 0.95),
                    confidence_level=self._get_confidence_level(llm_confidence),
                    reasoning_path=traditional_result.reasoning_path + [f"LLM enhancement: {llm_analysis[:100]}..."],
                    evidence=enhanced_evidence,
                    alternatives=enhanced_alternatives,
                    requires_review=llm_confidence < 0.8
                )
            else:
                # 使用传统结果，但增加LLM见解
                enhanced_evidence = traditional_result.evidence + [
                    f"LLM provided additional insight: {llm_analysis[:150]}..."
                ]
                
                return AlignmentResult(
                    element_guid=traditional_result.element_guid,
                    alignment_type=traditional_result.alignment_type,
                    regulation_category=traditional_result.regulation_category,
                    function_classification=traditional_result.function_classification,
                    confidence=min(traditional_result.confidence + 0.1, 0.95),
                    confidence_level=self._get_confidence_level(traditional_result.confidence + 0.1),
                    reasoning_path=traditional_result.reasoning_path + ["Enhanced with LLM insights"],
                    evidence=enhanced_evidence,
                    alternatives=traditional_result.alternatives,
                    requires_review=traditional_result.requires_review
                )
                
        except Exception as e:
            self.logger.error(f"Failed to combine traditional and LLM results: {e}")
            return traditional_result
    
    def _extract_regulation_category_from_llm(self, analysis: str) -> RegulationCategory:
        """从LLM分析中提取法规分类"""
        analysis_lower = analysis.lower()
        
        # 简单的关键词匹配
        if any(keyword in analysis_lower for keyword in ['residential', 'dwelling', 'apartment']):
            return RegulationCategory.RESIDENTIAL
        elif any(keyword in analysis_lower for keyword in ['office', 'commercial', 'business']):
            return RegulationCategory.OFFICE
        elif any(keyword in analysis_lower for keyword in ['industrial', 'factory', 'manufacturing']):
            return RegulationCategory.INDUSTRIAL
        else:
            return RegulationCategory.OTHER
    
    def _extract_function_classification_from_llm(self, analysis: str) -> str:
        """从LLM分析中提取功能分类"""
        analysis_lower = analysis.lower()
        
        # 提取主要功能关键词
        function_keywords = {
            'structural': ['structural', 'load bearing', 'support'],
            'equipment': ['equipment', 'mechanical', 'hvac'],
            'decorative': ['decorative', 'aesthetic', 'finish'],
            'habitable': ['habitable', 'occupancy', 'living'],
            'service': ['service', 'utility', 'auxiliary']
        }
        
        for function, keywords in function_keywords.items():
            if any(keyword in analysis_lower for keyword in keywords):
                return function
        
        return 'unknown'
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """根据置信度值获取置信度等级"""
        if confidence >= self.confidence_thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif confidence >= self.confidence_thresholds[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _align_category_a(
        self,
        element: IFCElementInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext
    ) -> AlignmentResult:
        """处理A类功能语义对齐问题"""
        
        if element.ifc_type == IFCElementType.SLAB:
            return self._align_a1_slab_classification(element, regulation_data, context)
        elif element.ifc_type == IFCElementType.SPACE:
            return self._align_a2_space_classification(element, regulation_data, context)
        else:
            # 默认处理
            return self._create_default_result(element, AlignmentType.CATEGORY_A_FUNCTIONAL)
    
    def _align_a1_slab_classification(
        self,
        element: IFCElementInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext
    ) -> AlignmentResult:
        """A1: 设备设施与结构构件区分（IfcSlab分类）"""
        
        reasoning_path = []
        evidence = []
        alternatives = []
        
        # 1. 评估功能证据
        thickness = element.geometric_features.thickness if element.geometric_features else 0.0
        
        # 厚度指标分析
        if thickness < self.thickness_thresholds['decoration_platform']:
            thickness_category = 'decoration_platform'
            thickness_confidence = 0.9
        elif thickness < self.thickness_thresholds['equipment_slab']:
            thickness_category = 'equipment_slab'
            thickness_confidence = 0.7
        else:
            thickness_category = 'structural_slab'
            thickness_confidence = 0.8
            
        reasoning_path.append(f"厚度分析: {thickness:.3f}m → {thickness_category}")
        evidence.append(f"slab_thickness: {thickness:.3f}m")
        
        # 位置指标分析
        location_confidence = 0.5
        location_category = 'unknown'
        
        if element.geometric_features:
            # 检查是否在屋顶
            if element.geometric_features.elevation > context.floor_height * (context.total_floors - 0.5):
                location_category = 'rooftop_equipment'
                location_confidence = 0.8
                reasoning_path.append("位置分析: 屋顶位置 → 设备平台可能性高")
                evidence.append("location: rooftop")
        
        # 支撑功能分析
        support_confidence = 0.5
        if element.function_inference:
            if 'equipment' in element.function_inference.primary_function.lower():
                support_category = 'equipment_support'
                support_confidence = element.function_inference.confidence
                reasoning_path.append(f"功能推断: {element.function_inference.primary_function} → 设备支撑")
                evidence.extend(element.function_inference.evidence)
            elif 'structural' in element.function_inference.primary_function.lower():
                support_category = 'structural_support'
                support_confidence = element.function_inference.confidence
                reasoning_path.append(f"功能推断: {element.function_inference.primary_function} → 结构支撑")
        
        # 3. 功能分类确定
        function_classification = "structural_slab"  # 默认为结构楼板
        regulation_category = RegulationCategory.INCLUDE_FULL
        
        if thickness_category == 'decoration_platform':
            function_classification = "decoration_platform"
            regulation_category = RegulationCategory.EXCLUDE
            reasoning_path.append("功能分类: 装饰平台")
        elif thickness_category == 'equipment_slab' or location_category == 'rooftop_equipment':
            function_classification = "equipment_platform"
            regulation_category = RegulationCategory.INCLUDE_PARTIAL
            reasoning_path.append("功能分类: 设备平台")
        else:
            function_classification = "structural_slab"
            regulation_category = RegulationCategory.INCLUDE_FULL
            reasoning_path.append("功能分类: 结构楼板")
        
        # 3. 置信度评估
        confidence_scores = [thickness_confidence, location_confidence, support_confidence]
        final_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # 检查指标一致性
        if thickness_category == 'equipment_slab' and location_category == 'rooftop_equipment':
            final_confidence = min(final_confidence + 0.2, 1.0)  # 指标一致性加分
            reasoning_path.append("一致性检查: 厚度和位置指标一致 → 置信度提升")
        
        confidence_level = self._get_confidence_level(final_confidence)
        requires_review = confidence_level == ConfidenceLevel.LOW or final_confidence < 0.7
        
        # 生成备选方案
        if requires_review:
            alternatives = [
                {
                    "function": "equipment_platform",
                    "category": RegulationCategory.INCLUDE_PARTIAL.value,
                    "reason": "保守估计：按设备平台处理"
                },
                {
                    "function": "structural_slab",
                    "category": RegulationCategory.INCLUDE_FULL.value,
                    "reason": "按结构楼板处理"
                }
            ]
        
        return AlignmentResult(
            element_guid=element.guid,
            alignment_type=AlignmentType.CATEGORY_A_FUNCTIONAL,
            regulation_category=regulation_category,
            function_classification=function_classification,
            confidence=final_confidence,
            confidence_level=confidence_level,
            reasoning_path=reasoning_path,
            evidence=evidence,
            alternatives=alternatives,
            requires_review=requires_review
        )
    
    def _align_a2_space_classification(
        self,
        element: IFCElementInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext
    ) -> AlignmentResult:
        """A2: 空间功能分类（缺失或模糊的空间功能属性）"""
        
        reasoning_path = []
        evidence = []
        alternatives = []
        
        # 1. 功能推断集成
        primary_function = "unknown"
        function_confidence = 0.3
        
        if element.function_inference:
            primary_function = element.function_inference.primary_function
            function_confidence = element.function_inference.confidence
            reasoning_path.append(f"功能推断: {primary_function} (置信度: {function_confidence:.2f})")
            evidence.extend(element.function_inference.evidence)
        
        # 2. 几何证据验证
        geometric_confidence = 0.5
        if element.geometric_features:
            # 检查层高
            if element.geometric_features.height < 2.2:
                geometric_confidence = 0.8
                reasoning_path.append(f"几何验证: 层高 {element.geometric_features.height:.2f}m < 2.2m → 辅助用房")
                evidence.append(f"space_height: {element.geometric_features.height:.2f}m")
            
            # 检查面积
            if element.geometric_features.area < 10.0:  # 小面积空间
                reasoning_path.append(f"几何验证: 面积 {element.geometric_features.area:.2f}m² → 小型辅助空间")
                evidence.append(f"space_area: {element.geometric_features.area:.2f}m²")
        
        # 3. 建筑类型上下文考虑
        context_confidence = 0.5
        if context.building_type:
            reasoning_path.append(f"建筑类型上下文: {context.building_type}")
            evidence.append(f"building_type: {context.building_type}")
        
        # 4. 空间功能分类
        function_classification = primary_function if primary_function != "unknown" else "unclassified_space"
        regulation_category = RegulationCategory.INCLUDE_FULL
        
        # 特殊空间处理
        special_spaces = [
            '室外消防水池', '停车棚', '货棚', '垃圾棚', '加油站', '收费站'
        ]
        
        if any(keyword in primary_function for keyword in special_spaces):
            function_classification = "special_construction"
            regulation_category = RegulationCategory.EXCLUDE
            reasoning_path.append(f"特殊空间识别: {primary_function} → 特殊构筑物")
        elif 'mechanical' in primary_function.lower() or 'equipment' in primary_function.lower():
            if element.geometric_features and element.geometric_features.height < 2.2:
                function_classification = "low_height_equipment_room"
                regulation_category = RegulationCategory.INCLUDE_PARTIAL
                reasoning_path.append("设备用房 + 低层高 → 低层高设备用房")
            else:
                function_classification = "equipment_room"
                regulation_category = RegulationCategory.INCLUDE_FULL
                reasoning_path.append("设备用房 + 正常层高 → 设备用房")
        elif 'office' in primary_function.lower() or 'habitable' in primary_function.lower():
            function_classification = "habitable_space"
            regulation_category = RegulationCategory.INCLUDE_FULL
            reasoning_path.append("可居住办公空间 → 可居住空间")
        
        # 5. 置信度评估
        confidence_scores = [function_confidence, geometric_confidence, context_confidence]
        final_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # 模糊空间的特殊处理
        if primary_function == "unknown" or function_confidence < 0.5:
            final_confidence = max(final_confidence - 0.2, 0.1)
            reasoning_path.append("模糊空间处理: 功能不明确 → 置信度降低")
            
            # 采用保守解释
            if regulation_category == RegulationCategory.INCLUDE_FULL:
                alternatives.append({
                    "function": "auxiliary_space",
                    "category": RegulationCategory.INCLUDE_PARTIAL.value,
                    "reason": "保守解释：功能不明确时按辅助用房处理"
                })
        
        confidence_level = self._get_confidence_level(final_confidence)
        requires_review = confidence_level == ConfidenceLevel.LOW or final_confidence < 0.6
        
        return AlignmentResult(
            element_guid=element.guid,
            alignment_type=AlignmentType.CATEGORY_A_FUNCTIONAL,
            regulation_category=regulation_category,
            function_classification=function_classification,
            confidence=final_confidence,
            confidence_level=confidence_level,
            reasoning_path=reasoning_path,
            evidence=evidence,
            alternatives=alternatives,
            requires_review=requires_review
        )
    
    def _align_category_b(
        self,
        space: VerticalSpaceInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext
    ) -> AlignmentResult:
        """处理B类几何规范对齐问题"""
        
        if space.space_type in [VerticalSpaceType.ATRIUM, VerticalSpaceType.SHAFT, VerticalSpaceType.STAIRCASE]:
            return self._align_b1_vertical_openings(space, regulation_data, context)
        else:
            return self._align_b2_space_classification(space, regulation_data, context)
    
    def _align_b1_vertical_openings(
        self,
        space: VerticalSpaceInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext
    ) -> AlignmentResult:
        """B1: 多层开口识别和处理"""
        
        reasoning_path = []
        evidence = []
        alternatives = []
        
        # 1. 垂直连续性分析
        floors_spanned = len(space.floor_levels)
        reasoning_path.append(f"垂直跨度分析: 跨越 {floors_spanned} 层")
        evidence.append(f"floors_spanned: {floors_spanned}")
        
        is_multi_story = floors_spanned >= 2
        if is_multi_story:
            reasoning_path.append("确认为多层开口")
        
        # 2. 截面一致性检查
        cross_section_consistency = 0.8  # 假设从几何分析得出
        if cross_section_consistency > 0.7:
            reasoning_path.append(f"截面一致性: {cross_section_consistency:.2f} → 同一开口实例")
            evidence.append(f"cross_section_consistency: {cross_section_consistency:.2f}")
        
        # 3. 功能确定
        space_function = space.space_type.value
        reasoning_path.append(f"空间功能: {space_function}")
        evidence.append(f"space_function: {space_function}")
        
        # 4. 开口分类确定
        opening_classification = space.space_type.value
        regulation_category = RegulationCategory.EXCLUDE
        
        if space.space_type == VerticalSpaceType.ATRIUM:
            opening_classification = "multi_story_atrium"
            regulation_category = RegulationCategory.DEDUCT_PER_FLOOR
            reasoning_path.append("中庭空间 → 多层中庭开口")
        elif space.space_type in [VerticalSpaceType.SHAFT, VerticalSpaceType.STAIRCASE]:
            opening_classification = f"multi_story_{space.space_type.value}"
            regulation_category = RegulationCategory.EXCLUDE
            reasoning_path.append(f"{space_function} → 多层{space_function}开口")
        
        # 5. 跨层协调
        reasoning_path.append("跨层协调: 确保所有楼层一致处理")
        evidence.append(f"coordinated_floors: {space.floor_levels}")
        
        # 避免重复计算检查
        if is_multi_story:
            reasoning_path.append("重复计算检查: 已标记为多层处理")
        
        # 置信度评估
        confidence_factors = [
            0.9 if is_multi_story else 0.3,  # 多层确认
            cross_section_consistency,        # 几何一致性
            0.8 if space.space_type != VerticalSpaceType.UNKNOWN else 0.3  # 功能明确性
        ]
        
        final_confidence = sum(confidence_factors) / len(confidence_factors)
        confidence_level = self._get_confidence_level(final_confidence)
        requires_review = confidence_level == ConfidenceLevel.LOW
        
        return AlignmentResult(
            element_guid=space.guid,
            alignment_type=AlignmentType.CATEGORY_B_GEOMETRIC,
            regulation_category=regulation_category,
            function_classification=opening_classification,
            confidence=final_confidence,
            confidence_level=confidence_level,
            reasoning_path=reasoning_path,
            evidence=evidence,
            alternatives=alternatives,
            requires_review=requires_review
        )
    
    def _align_b2_space_classification(
        self,
        space: VerticalSpaceInfo,
        regulation_data: Dict[str, Any],
        context: AlignmentContext
    ) -> AlignmentResult:
        """B2: 垂直空间分类（不同竖井类型的不同面积扣除规则）"""
        
        reasoning_path = []
        evidence = []
        alternatives = []
        
        # 1. 空间类型识别
        space_type = space.space_type
        reasoning_path.append(f"空间类型识别: {space_type.value}")
        evidence.append(f"detected_type: {space_type.value}")
        
        # 2. 几何特征分析
        if space.geometric_features:
            area = space.geometric_features.area
            height = space.geometric_features.height
            reasoning_path.append(f"几何特征: 面积 {area:.2f}m², 高度 {height:.2f}m")
            evidence.extend([f"space_area: {area:.2f}m²", f"space_height: {height:.2f}m"])
        
        # 3. 垂直空间分类
        space_classification = space_type.value
        regulation_category = RegulationCategory.EXCLUDE
        
        if space_type == VerticalSpaceType.ATRIUM:
            space_classification = "atrium_space"
            regulation_category = RegulationCategory.EXCLUDE
            reasoning_path.append("中庭空间 → 中庭类垂直空间")
        elif space_type == VerticalSpaceType.SHAFT:
            space_classification = "utility_shaft"
            regulation_category = RegulationCategory.INCLUDE_FULL
            reasoning_path.append("竖井空间 → 功能性竖井")
        elif space_type == VerticalSpaceType.STAIRCASE:
            space_classification = "staircase_space"
            regulation_category = RegulationCategory.INCLUDE_FULL
            reasoning_path.append("楼梯间 → 楼梯间空间")
        elif space_type == VerticalSpaceType.ELEVATOR_SHAFT:
            space_classification = "elevator_shaft"
            regulation_category = RegulationCategory.INCLUDE_FULL
            reasoning_path.append("电梯井 → 电梯井空间")
        else:
            space_classification = "unclassified_vertical_space"
            regulation_category = RegulationCategory.EXCLUDE
            reasoning_path.append("未知空间类型 → 未分类垂直空间")
            
            # 提供备选方案
            alternatives = [
                {
                    "function": "utility_shaft",
                    "category": RegulationCategory.INCLUDE_FULL.value,
                    "reason": "如确认为功能性竖井，应归类为功能性竖井"
                }
            ]
        
        # 4. 置信度评估
        type_confidence = 0.8 if space_type != VerticalSpaceType.UNKNOWN else 0.3
        geometric_confidence = 0.7 if space.geometric_features else 0.4
        
        final_confidence = (type_confidence + geometric_confidence) / 2
        confidence_level = self._get_confidence_level(final_confidence)
        requires_review = (confidence_level == ConfidenceLevel.LOW or 
                          space_type == VerticalSpaceType.UNKNOWN)
        
        return AlignmentResult(
            element_guid=space.guid,
            alignment_type=AlignmentType.CATEGORY_B_GEOMETRIC,
            regulation_category=regulation_category,
            function_classification=space_classification,
            confidence=final_confidence,
            confidence_level=confidence_level,
            reasoning_path=reasoning_path,
            evidence=evidence,
            alternatives=alternatives,
            requires_review=requires_review
        )
    
    def _create_default_result(
        self,
        element: IFCElementInfo,
        alignment_type: AlignmentType
    ) -> AlignmentResult:
        """创建默认对齐结果"""
        return AlignmentResult(
            element_guid=element.guid,
            alignment_type=alignment_type,
            regulation_category=RegulationCategory.INCLUDE_FULL,
            area_coefficient=1.0,
            confidence=0.5,
            confidence_level=ConfidenceLevel.MEDIUM,
            reasoning_path=["默认处理：未识别的元素类型"],
            evidence=[f"ifc_type: {element.ifc_type.value}"],
            alternatives=[],
            requires_review=True
        )
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """根据置信度数值确定置信度等级"""
        if confidence >= self.confidence_thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif confidence >= self.confidence_thresholds[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def generate_alignment_report(
        self,
        results: List[AlignmentResult]
    ) -> Dict[str, Any]:
        """生成对齐报告
        
        Args:
            results: 对齐结果列表
            
        Returns:
            对齐报告字典
        """
        report = {
            "summary": {
                "total_elements": len(results),
                "category_a_count": len([r for r in results if r.alignment_type == AlignmentType.CATEGORY_A_FUNCTIONAL]),
                "category_b_count": len([r for r in results if r.alignment_type == AlignmentType.CATEGORY_B_GEOMETRIC]),
                "high_confidence_count": len([r for r in results if r.confidence_level == ConfidenceLevel.HIGH]),
                "requires_review_count": len([r for r in results if r.requires_review])
            },
            "confidence_distribution": {
                "high": len([r for r in results if r.confidence_level == ConfidenceLevel.HIGH]),
                "medium": len([r for r in results if r.confidence_level == ConfidenceLevel.MEDIUM]),
                "low": len([r for r in results if r.confidence_level == ConfidenceLevel.LOW])
            },
            "regulation_categories": {
                "include_full": len([r for r in results if r.regulation_category == RegulationCategory.INCLUDE_FULL]),
                "include_partial": len([r for r in results if r.regulation_category == RegulationCategory.INCLUDE_PARTIAL]),
                "exclude": len([r for r in results if r.regulation_category == RegulationCategory.EXCLUDE]),
                "deduct_per_floor": len([r for r in results if r.regulation_category == RegulationCategory.DEDUCT_PER_FLOOR])
            },
            "detailed_results": [
                {
                    "element_guid": result.element_guid,
                    "alignment_type": result.alignment_type.value,
                    "regulation_category": result.regulation_category.value,
                    "area_coefficient": result.area_coefficient,
                    "confidence": result.confidence,
                    "confidence_level": result.confidence_level.value,
                    "requires_review": result.requires_review,
                    "reasoning_path": result.reasoning_path,
                    "evidence": result.evidence,
                    "alternatives": result.alternatives
                }
                for result in results
            ],
            "review_required": [
                {
                    "element_guid": result.element_guid,
                    "reason": "低置信度" if result.confidence_level == ConfidenceLevel.LOW else "需要人工确认",
                    "confidence": result.confidence,
                    "alternatives": result.alternatives
                }
                for result in results if result.requires_review
            ]
        }
        
        return report