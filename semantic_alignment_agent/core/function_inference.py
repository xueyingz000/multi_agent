from __future__ import annotations

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..utils import (
    log,
    GeometricFeatures,
    IfcElementInfo,
    FunctionalInference as FunctionInference,
    FunctionType,
    Evidence,
)
from ..geometry import SpatialContext
from ..llm.llm_client import LLMClient
from ..llm.prompt_templates import PromptTemplates
from ..utils.config_loader import ConfigLoader


@dataclass
class InferenceRule:
    """推断规则"""

    name: str
    conditions: Dict[str, Any]
    function_type: FunctionType
    confidence_base: float
    evidence_keywords: List[str]
    priority: int = 1  # 优先级，数字越小优先级越高


class FunctionInferenceEngine:
    """功能推断引擎

    集成LLM智能分析，用于处理模糊情况和边界情况的功能推断
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_llm: bool = True,
        enable_linear_adjustments: bool = False,
    ):
        """初始化功能推断引擎

        Args:
            config_path: 配置文件路径
            enable_llm: 是否启用LLM增强分析
            enable_linear_adjustments: 是否启用线性证据加权调整
        """
        self.inference_rules = self._initialize_rules()
        self.keyword_patterns = self._initialize_keyword_patterns()

        # 置信度调整参数
        self.confidence_adjustments = {
            "geometric_consistency": 0.2,
            "keyword_match": 0.15,
            "spatial_context": 0.1,
            "property_evidence": 0.15,
        }

        self.enable_llm = enable_llm
        self.enable_linear_evidence_adjustments = enable_linear_adjustments

        # 初始化LLM客户端
        self.llm_client = None
        # 从配置加载语义相关设置（如 LLM 触发阈值和强制开关）
        try:
            _loader = ConfigLoader()
            semantic_cfg = _loader.get_semantic_config()
        except Exception:
            semantic_cfg = {}
        self.llm_confidence_threshold = float(semantic_cfg.get("llm_confidence_threshold", 0.6))
        self.force_llm = bool(semantic_cfg.get("force_llm", False))

        if enable_llm:
            try:
                self.llm_client = LLMClient(config_path)
                if self.llm_client.is_available():
                    log.info("LLM-enhanced function inference enabled")
                else:
                    log.warning(
                        "LLM client not available, using rule-based inference only"
                    )
            except Exception as e:
                log.warning(f"Failed to initialize LLM client: {e}")

        self.prompt_templates = PromptTemplates()

    def _initialize_rules(self) -> List[InferenceRule]:
        """初始化推断规则"""
        return [
            # IfcSlab 推断规则
            InferenceRule(
                name="structural_slab_thick",
                conditions={
                    "ifc_type": "IfcSlab",
                    "thickness_min": 0.15,
                    "support_function": "supports_occupancy",
                },
                function_type=FunctionType.STRUCTURAL_FLOOR,
                confidence_base=0.8,
                evidence_keywords=["structural", "floor", "slab", "load bearing"],
                priority=1,
            ),
            InferenceRule(
                name="decoration_platform_thin",
                conditions={
                    "ifc_type": "IfcSlab",
                    "thickness_max": 0.1,
                    "support_function": ["supports_equipment", "unknown"],
                },
                function_type=FunctionType.DECORATION_PLATFORM,
                confidence_base=0.7,
                evidence_keywords=["decoration", "decorative", "thin", "canopy"],
                priority=3,
            ),
            # IfcSpace 推断规则
            InferenceRule(
                name="general_use_space",
                conditions={
                    "ifc_type": "IfcSpace",
                    "height_min": 2.2,
                    "location_indicator": ["habitable_space"],
                },
                function_type=FunctionType.GENERAL_USE_SPACE,
                confidence_base=0.8,
                evidence_keywords=[
                    "general",
                    "space",
                    "room",
                    "indoor",
                    "通用",
                    "普通",
                    "室内",
                    "空间",
                ],
                priority=1,
            ),
            InferenceRule(
                name="outdoor_construction",
                conditions={
                    "ifc_type": "IfcSpace",
                    "outdoor_exposure": True,
                    "location_indicator": [
                        "outdoor_construction",
                        "auxiliary_construction",
                    ],
                },
                function_type=FunctionType.AUXILIARY_CONSTRUCTION,
                confidence_base=0.8,
                evidence_keywords=[
                    "outdoor",
                    "消防",
                    "水池",
                    "停车",
                    "货棚",
                    "垃圾",
                    "加油",
                    "收费",
                ],
                priority=1,
            ),
            # 垂直空间推断规则
            InferenceRule(
                name="atrium_multi_story",
                conditions={
                    "ifc_type": "IfcSpace",
                    "floors_spanned_min": 2,
                    "aspect_ratio_max": 3.0,  # 相对规整的形状
                },
                function_type=FunctionType.ATRIUM,
                confidence_base=0.75,
                evidence_keywords=["atrium", "lobby", "hall", "void"],
                priority=2,
            ),
            InferenceRule(
                name="shaft_vertical",
                conditions={
                    "ifc_type": "IfcSpace",
                    "floors_spanned_min": 2,
                    "aspect_ratio_min": 3.0,  # 细长形状
                },
                function_type=FunctionType.SHAFT,
                confidence_base=0.8,
                evidence_keywords=["shaft", "duct", "pipe", "elevator", "stair"],
                priority=1,
            ),
        ]

    def _initialize_keyword_patterns(self) -> Dict[str, List[str]]:
        """初始化关键词模式"""
        return {
            "structural": ["结构", "structural", "load", "bearing", "承重", "楼板"],
            "equipment": ["设备", "equipment", "mechanical", "plant", "机房", "设备间"],
            "decoration": ["装饰", "decorative", "canopy", "雨篷", "装饰性"],
            "office": ["办公", "office", "workspace", "工作", "房间"],
            "auxiliary": ["辅助", "auxiliary", "utility", "service", "附属"],
            "outdoor": ["室外", "outdoor", "external", "户外"],
            "construction": ["构筑物", "construction", "建筑物", "设施"],
            "atrium": ["中庭", "atrium", "lobby", "hall", "大厅"],
            "shaft": ["竖井", "shaft", "duct", "pipe", "管井", "电梯井"],
        }

    def infer_function(
        self,
        element_info: IfcElementInfo,
        geometric_features: GeometricFeatures,
        spatial_context: Optional[SpatialContext] = None,
    ) -> FunctionInference:
        """推断元素功能"""
        try:
            # 收集所有匹配的规则
            matching_rules = self._find_matching_rules(
                element_info, geometric_features, spatial_context
            )

            if not matching_rules:
                return self._create_default_inference(element_info)

            # 按优先级排序
            matching_rules.sort(
                key=lambda x: (x[0].priority, -x[1])
            )  # 优先级升序，置信度降序

            # 选择最佳规则
            best_rule, base_confidence = matching_rules[0]

            # 调整置信度
            adjusted_confidence = self._adjust_confidence(
                base_confidence,
                element_info,
                geometric_features,
                spatial_context,
                best_rule,
            )

            # 收集证据
            evidence = self._collect_evidence(
                element_info, geometric_features, spatial_context, best_rule
            )

            # 生成备选方案
            alternatives = self._generate_alternatives(
                matching_rules[1:5]
            )  # 最多5个备选

            inference = FunctionInference(
                primary_function=best_rule.function_type,
                confidence=min(
                    0.95, max(0.1, adjusted_confidence)
                ),  # 限制在0.1-0.95之间
                evidence=evidence,
                alternatives=alternatives,
                reasoning=self._generate_reasoning_path(
                    best_rule, element_info, geometric_features
                ),
            )

            log.debug(
                f"Inferred function for {element_info.guid}: {best_rule.function_type.value} (confidence: {inference.confidence:.3f})"
            )
            return inference

        except Exception as e:
            log.error(f"Error inferring function for {element_info.guid}: {e}")
            return self._create_default_inference(element_info)

    def infer_element_function(
        self,
        element: IfcElementInfo,
        spatial_context: Optional[SpatialContext] = None,
    ) -> FunctionInference:
        """兼容包装方法：接受元素对象，内部调用 infer_function

        与主流程保持一致：优先使用已计算的几何特征。
        """
        geometric = element.geometric_features
        return self.infer_function(element, geometric, spatial_context)

    def infer_function_enhanced(
        self,
        element_info: IfcElementInfo,
        geometric_features: GeometricFeatures,
        spatial_context: Optional[SpatialContext] = None,
        building_context: Optional[Dict[str, Any]] = None,
    ) -> FunctionInference:
        """增强的功能推断，集成LLM智能分析

        Args:
            element_info: IFC元素信息
            geometric_features: 几何特征
            spatial_context: 空间上下文
            building_context: 建筑上下文

        Returns:
            功能推断结果
        """
        # 首先进行传统规则推断
        rule_based_result = self.infer_function(
            element_info, geometric_features, spatial_context
        )

        # 检查是否需要LLM增强分析
        if self._needs_llm_analysis(rule_based_result):
            if self.enable_llm and self.llm_client and self.llm_client.is_available():
                try:
                    return self._llm_enhanced_inference(
                        element_info,
                        geometric_features,
                        spatial_context,
                        building_context,
                        rule_based_result,
                    )
                except Exception as e:
                    log.warning(f"LLM inference failed: {e}, using rule-based result")

        return rule_based_result

    def _needs_llm_analysis(self, rule_based_result: FunctionInference) -> bool:
        """判断是否需要LLM增强分析

        在以下情况下使用LLM：
        1. 置信度较低（模糊情况）
        2. 有多个可能的功能选项
        3. 存在不确定性因素
        """
        if self.force_llm:
            return True
        return (
            rule_based_result.confidence < self.llm_confidence_threshold  # 配置化置信度阈值
            or len(rule_based_result.alternatives) > 1
            or len(rule_based_result.evidence) < 2
        )

    def _llm_enhanced_inference(
        self,
        element_info: IfcElementInfo,
        geometric_features: GeometricFeatures,
        spatial_context: Optional[SpatialContext],
        building_context: Optional[Dict[str, Any]],
        rule_based_result: FunctionInference,
    ) -> FunctionInference:
        """使用LLM进行增强的功能推断"""
        # 准备LLM分析的上下文
        context = {
            "ifc_type": element_info.ifc_type,
            "guid": element_info.guid,
            "geometric_features": {
                "area": geometric_features.area,
                "thickness": geometric_features.thickness,
                "position": str(geometric_features.position),
                "location_indicator": geometric_features.location_indicator,
            },
            "property_sets": str(element_info.properties),
            "spatial_context": {
                "adjacent_spaces": str(
                    spatial_context.adjacent_spaces if spatial_context else []
                ),
                "floor_level": spatial_context.floor_level if spatial_context else 0,
                "support_relationships": str(
                    spatial_context.support_relationships if spatial_context else []
                ),
            },
            "building_context": str(building_context or {}),
            "region_info": "Unknown",
            "project_phase": "Design",
            "rule_based_analysis": {
                "predicted_function": rule_based_result.primary_function.value,
                "confidence": rule_based_result.confidence,
                "alternatives": [
                    (alt[0].value, alt[1]) for alt in rule_based_result.alternatives
                ],
                "evidence": [ev.text for ev in rule_based_result.evidence],
            },
        }

        # 生成LLM提示
        prompt = self.prompt_templates.get_element_classification_prompt(**context)

        # 获取LLM分析结果
        llm_result = self.llm_client.analyze_with_confidence(
            prompt, context, temperature=0.2
        )

        # 解析LLM结果并与规则推断结合
        return self._combine_llm_and_rule_results(
            llm_result, rule_based_result, context
        )

    def _combine_llm_and_rule_results(
        self,
        llm_result: Dict[str, Any],
        rule_based_result: FunctionInference,
        context: Dict[str, Any],
    ) -> FunctionInference:
        """结合LLM分析和规则推断的结果"""
        try:
            # 提取LLM分析的功能预测
            llm_analysis = llm_result.get("analysis", "")
            llm_confidence = llm_result.get("confidence_score", 0.5)

            # 简单的结合策略：如果LLM置信度更高，使用LLM结果
            if llm_confidence > rule_based_result.confidence:
                # 尝试从LLM分析中提取功能类型
                predicted_function = self._extract_function_from_llm_analysis(
                    llm_analysis
                )

                # 创建增强的证据链
                enhanced_evidence = rule_based_result.evidence + [
                    Evidence(
                        text=f"LLM analysis: {llm_analysis[:200]}...",
                        source="llm_analysis",
                        confidence=llm_confidence,
                    )
                ]

                return FunctionInference(
                    primary_function=predicted_function,
                    confidence=min(llm_confidence, 0.9),  # 限制最大置信度
                    evidence=enhanced_evidence,
                    alternatives=rule_based_result.alternatives
                    + [
                        (
                            rule_based_result.primary_function,
                            rule_based_result.confidence,
                        )
                    ],
                    reasoning=f"Enhanced with LLM: {rule_based_result.reasoning} → LLM analysis suggests {predicted_function.value}",
                )
            else:
                # 使用规则推断结果，但增加LLM的见解
                enhanced_evidence = rule_based_result.evidence + [
                    Evidence(
                        text=f"LLM provided additional insight: {llm_analysis[:150]}...",
                        source="llm_insight",
                        confidence=llm_confidence,
                    )
                ]

                return FunctionInference(
                    primary_function=rule_based_result.primary_function,
                    confidence=min(
                        rule_based_result.confidence + 0.1, 0.95
                    ),  # 轻微提升置信度
                    evidence=enhanced_evidence,
                    alternatives=rule_based_result.alternatives,
                    reasoning=f"{rule_based_result.reasoning} (Enhanced with LLM insights)",
                )

        except Exception as e:
            log.error(f"Failed to combine LLM and rule results: {e}")
            return rule_based_result

    def _extract_function_from_llm_analysis(self, llm_analysis: str) -> FunctionType:
        """从LLM分析中提取功能类型"""
        # 简单的关键词匹配策略
        analysis_lower = llm_analysis.lower()

        function_keywords = {
            FunctionType.STRUCTURAL_FLOOR: [
                "structural",
                "load bearing",
                "support",
                "beam",
                "foundation",
            ],
            FunctionType.EQUIPMENT_PLATFORM: [
                "equipment",
                "platform",
                "mechanical",
                "rooftop",
                "hvac",
            ],
            FunctionType.DECORATION_PLATFORM: [
                "decorative",
                "decoration",
                "thin",
                "canopy",
                "aesthetic",
            ],
            FunctionType.GENERAL_USE_SPACE: [
                "office",
                "habitable",
                "workspace",
                "room",
                "occupancy",
            ],
            FunctionType.MECHANICAL_ROOM: [
                "mechanical",
                "equipment room",
                "plant",
                "utility",
                "service",
            ],
            FunctionType.AUXILIARY_ROOM: [
                "auxiliary",
                "storage",
                "utility",
                "service",
                "support",
            ],
            FunctionType.AUXILIARY_CONSTRUCTION: [
                "outdoor",
                "construction",
                "parking",
                "garage",
            ],
            FunctionType.ATRIUM: ["atrium", "lobby", "hall", "void", "multi-story"],
            FunctionType.SHAFT: [
                "shaft",
                "duct",
                "pipe",
                "elevator",
                "stair",
                "vertical",
            ],
        }

        # 计算每个功能类型的匹配分数
        scores = {}
        for function_type, keywords in function_keywords.items():
            score = sum(1 for keyword in keywords if keyword in analysis_lower)
            if score > 0:
                scores[function_type] = score

        # 返回得分最高的功能类型
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return FunctionType.STRUCTURAL_FLOOR  # 默认值

    def _find_matching_rules(
        self,
        element_info: IfcElementInfo,
        geometric_features: GeometricFeatures,
        spatial_context: Optional[SpatialContext] = None,
    ) -> List[Tuple[InferenceRule, float]]:
        """查找匹配的规则"""
        matching_rules = []

        for rule in self.inference_rules:
            match_score = self._evaluate_rule_match(
                rule, element_info, geometric_features, spatial_context
            )
            if match_score > 0:
                matching_rules.append((rule, rule.confidence_base * match_score))

        return matching_rules

    def _evaluate_rule_match(
        self,
        rule: InferenceRule,
        element_info: IfcElementInfo,
        geometric_features: GeometricFeatures,
        spatial_context: Optional[SpatialContext] = None,
    ) -> float:
        """评估规则匹配度"""
        conditions = rule.conditions
        match_score = 1.0

        # 检查IFC类型
        if "ifc_type" in conditions:
            if element_info.ifc_type != conditions["ifc_type"]:
                return 0.0

        # 检查厚度条件
        if "thickness_min" in conditions:
            if geometric_features.thickness < conditions["thickness_min"]:
                match_score *= 0.5

        if "thickness_max" in conditions:
            if geometric_features.thickness > conditions["thickness_max"]:
                match_score *= 0.5

        # 检查高度条件（对于空间）
        if "height_min" in conditions:
            if (
                geometric_features.thickness < conditions["height_min"]
            ):  # 对空间，thickness表示高度
                match_score *= 0.5

        if "height_max" in conditions:
            if geometric_features.thickness > conditions["height_max"]:
                match_score *= 0.5

        # 检查支撑功能
        if "support_function" in conditions:
            expected_functions = conditions["support_function"]
            if isinstance(expected_functions, str):
                expected_functions = [expected_functions]

            if geometric_features.support_function not in expected_functions:
                match_score *= 0.7

        # 检查位置指标
        if "location_indicator" in conditions:
            expected_locations = conditions["location_indicator"]
            if isinstance(expected_locations, str):
                expected_locations = [expected_locations]

            if geometric_features.location_indicator not in expected_locations:
                match_score *= 0.8

        # 检查空间上下文
        if spatial_context:
            if "equipment_proximity" in conditions:
                if (
                    spatial_context.equipment_nearby
                    != conditions["equipment_proximity"]
                ):
                    match_score *= 0.8

            if "outdoor_exposure" in conditions:
                if spatial_context.outdoor_exposure != conditions["outdoor_exposure"]:
                    match_score *= 0.8

        # 检查垂直跨度
        if "floors_spanned_min" in conditions:
            floors_spanned = getattr(geometric_features, "vertical_span", {}).get(
                "floors_spanned", 1
            )
            if floors_spanned < conditions["floors_spanned_min"]:
                match_score *= 0.3

        # 检查长宽比
        if "aspect_ratio_min" in conditions:
            aspect_ratio = getattr(geometric_features, "aspect_ratio", 1.0)
            if aspect_ratio < conditions["aspect_ratio_min"]:
                match_score *= 0.7

        if "aspect_ratio_max" in conditions:
            aspect_ratio = getattr(geometric_features, "aspect_ratio", 1.0)
            if aspect_ratio > conditions["aspect_ratio_max"]:
                match_score *= 0.7

        return match_score

    def _adjust_confidence(
        self,
        base_confidence: float,
        element_info: IfcElementInfo,
        geometric_features: GeometricFeatures,
        spatial_context: Optional[SpatialContext],
        rule: InferenceRule,
    ) -> float:
        """调整置信度"""
        adjusted = base_confidence

        if self.enable_linear_evidence_adjustments:
            # 线性证据调整（可开关）
            geometric_consistency = self._evaluate_geometric_consistency(
                geometric_features, rule
            )
            adjusted += (
                geometric_consistency
                * self.confidence_adjustments["geometric_consistency"]
            )

            keyword_match = self._evaluate_keyword_match(element_info, rule)
            adjusted += keyword_match * self.confidence_adjustments["keyword_match"]

            if spatial_context:
                context_match = self._evaluate_spatial_context_match(
                    spatial_context, rule
                )
                adjusted += (
                    context_match * self.confidence_adjustments["spatial_context"]
                )

            property_evidence = self._evaluate_property_evidence(element_info, rule)
            adjusted += (
                property_evidence * self.confidence_adjustments["property_evidence"]
            )

        # 当关闭线性调整时，直接返回基础置信度
        return adjusted

    def _evaluate_geometric_consistency(
        self, geometric_features: GeometricFeatures, rule: InferenceRule
    ) -> float:
        """评估几何一致性"""
        consistency_score = 0.0

        # 厚度一致性
        if rule.function_type == FunctionType.STRUCTURAL_FLOOR:
            if geometric_features.thickness >= 0.15:
                consistency_score += 0.5
        elif rule.function_type == FunctionType.EQUIPMENT_PLATFORM:
            if 0.1 <= geometric_features.thickness < 0.15:
                consistency_score += 0.5
        elif rule.function_type == FunctionType.DECORATION_PLATFORM:
            if geometric_features.thickness < 0.1:
                consistency_score += 0.5

        # 支撑功能一致性
        if rule.function_type in [
            FunctionType.STRUCTURAL_FLOOR,
            FunctionType.GENERAL_USE_SPACE,
        ]:
            if geometric_features.support_function == "supports_occupancy":
                consistency_score += 0.3
        elif rule.function_type in [
            FunctionType.EQUIPMENT_PLATFORM,
            FunctionType.MECHANICAL_ROOM,
        ]:
            if geometric_features.support_function == "supports_equipment":
                consistency_score += 0.3

        return min(1.0, consistency_score)

    def _evaluate_keyword_match(
        self, element_info: IfcElementInfo, rule: InferenceRule
    ) -> float:
        """评估关键词匹配"""
        text_sources = [
            element_info.properties.get("Name", ""),
            element_info.properties.get("Description", ""),
            element_info.properties.get("LongName", ""),
            element_info.properties.get("ObjectType", ""),
        ]

        combined_text = " ".join(text_sources).lower()

        match_count = 0
        for keyword in rule.evidence_keywords:
            if keyword.lower() in combined_text:
                match_count += 1

        if rule.evidence_keywords:
            return match_count / len(rule.evidence_keywords)
        else:
            return 0.0

    def _evaluate_spatial_context_match(
        self, spatial_context: SpatialContext, rule: InferenceRule
    ) -> float:
        """评估空间上下文匹配"""
        match_score = 0.0

        # 设备邻近性
        if rule.function_type in [
            FunctionType.EQUIPMENT_PLATFORM,
            FunctionType.MECHANICAL_ROOM,
        ]:
            if spatial_context.equipment_nearby:
                match_score += 0.4

        # 室外暴露
        if rule.function_type == FunctionType.AUXILIARY_CONSTRUCTION:
            if spatial_context.outdoor_exposure:
                match_score += 0.4

        # 结构上下文
        if rule.function_type == FunctionType.STRUCTURAL_FLOOR:
            if spatial_context.structural_context in ["intermediate", "ground"]:
                match_score += 0.2

        return min(1.0, match_score)

    def _evaluate_property_evidence(
        self, element_info: IfcElementInfo, rule: InferenceRule
    ) -> float:
        """评估属性证据"""
        evidence_score = 0.0

        # 承重属性
        load_bearing = element_info.properties.get("LoadBearing", False)
        if rule.function_type == FunctionType.STRUCTURAL_FLOOR and load_bearing:
            evidence_score += 0.5

        # 预定义类型
        predefined_type = element_info.properties.get("PredefinedType", "").upper()
        if rule.function_type == FunctionType.SHAFT and predefined_type == "SHAFT":
            evidence_score += 0.5
        elif rule.function_type == FunctionType.ATRIUM and predefined_type == "ATRIUM":
            evidence_score += 0.5

        return min(1.0, evidence_score)

    def _collect_evidence(
        self,
        element_info: IfcElementInfo,
        geometric_features: GeometricFeatures,
        spatial_context: Optional[SpatialContext],
        rule: InferenceRule,
    ) -> List[Evidence]:
        """收集证据"""
        evidence = []

        # 几何证据
        evidence.append(
            Evidence(
                text=f"厚度: {geometric_features.thickness:.3f}m, 分类: {geometric_features.thickness_indicator}",
                source="geometric_analysis",
                confidence=0.8,
            )
        )

        evidence.append(
            Evidence(
                text=f"位置指标: {geometric_features.location_indicator}",
                source="geometric_analysis",
                confidence=0.7,
            )
        )

        evidence.append(
            Evidence(
                text=f"支撑功能: {geometric_features.support_function}",
                source="geometric_analysis",
                confidence=0.7,
            )
        )

        # 属性证据
        for prop_name, prop_value in element_info.properties.items():
            if prop_name in [
                "Name",
                "Description",
                "LongName",
                "ObjectType",
                "PredefinedType",
            ]:
                if prop_value and any(
                    keyword.lower() in str(prop_value).lower()
                    for keyword in rule.evidence_keywords
                ):
                    evidence.append(
                        Evidence(
                            text=f"{prop_name}: {prop_value}",
                            source="ifc_properties",
                            confidence=0.6,
                        )
                    )

        # 空间上下文证据
        if spatial_context:
            if spatial_context.equipment_nearby:
                evidence.append(
                    Evidence(
                        text="设备邻近性: 是", source="spatial_context", confidence=0.6
                    )
                )

            if spatial_context.outdoor_exposure:
                evidence.append(
                    Evidence(
                        text="室外暴露: 是", source="spatial_context", confidence=0.6
                    )
                )

        return evidence

    def _generate_alternatives(
        self, other_rules: List[Tuple[InferenceRule, float]]
    ) -> List[Tuple[FunctionType, float]]:
        """生成备选方案"""
        alternatives = []

        for rule, confidence in other_rules:
            alternatives.append((rule.function_type, confidence))

        return alternatives

    def _generate_reasoning_path(
        self,
        rule: InferenceRule,
        element_info: IfcElementInfo,
        geometric_features: GeometricFeatures,
    ) -> str:
        """生成推理路径"""
        path_parts = [
            f"应用规则: {rule.name}",
            f"IFC类型: {element_info.ifc_type}",
            f"几何特征: 厚度={geometric_features.thickness:.3f}m, 位置={geometric_features.location_indicator}",
            f"功能推断: {rule.function_type.value}",
        ]

        return " → ".join(path_parts)

    def _create_default_inference(
        self, element_info: IfcElementInfo
    ) -> FunctionInference:
        """创建默认推断结果"""
        # 基于IFC类型的默认推断
        if element_info.ifc_type == "IfcSlab":
            default_function = FunctionType.STRUCTURAL_FLOOR
        elif element_info.ifc_type == "IfcSpace":
            default_function = FunctionType.GENERAL_USE_SPACE
        else:
            default_function = FunctionType.STRUCTURAL_FLOOR

        return FunctionInference(
            primary_function=default_function,
            confidence=0.3,  # 低置信度
            evidence=[
                Evidence(
                    text=f"默认推断基于IFC类型: {element_info.ifc_type}",
                    source="default_inference",
                    confidence=0.3,
                )
            ],
            alternatives=[],
            reasoning=f"默认推断: {element_info.ifc_type} → {default_function.value}",
        )

    def batch_infer_functions(
        self,
        elements_data: Dict[str, Tuple[IfcElementInfo, GeometricFeatures]],
        spatial_context: Optional[SpatialContext] = None,
    ) -> Dict[str, FunctionInference]:
        """批量推断功能"""
        results = {}

        for guid, (element_info, geometric_features) in elements_data.items():
            try:
                inference = self.infer_function(
                    element_info, geometric_features, spatial_context
                )
                results[guid] = inference
            except Exception as e:
                log.error(f"Error inferring function for {guid}: {e}")
                results[guid] = self._create_default_inference(element_info)

        log.info(f"Completed function inference for {len(results)} elements")
        return results
