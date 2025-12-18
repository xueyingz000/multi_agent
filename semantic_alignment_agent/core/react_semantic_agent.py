from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from ..utils import (
    log,
    RegulationCategory,
    IfcElementType,
    IfcElementInfo,
    VerticalSpaceInfo,
    Evidence,
    GeometricFeatures,
    BoundingBox,
    Point3D,
)
from ..llm.llm_client import LLMClient
from ..llm.prompt_templates import PromptTemplates
from ..geometry.geometry_analyzer import GeometryAnalyzer, SpatialContext
from .function_inference import FunctionInferenceEngine
from .vertical_space_detector import VerticalSpaceDetector
from ..data_processing.regulation_parser import RegulationParser
from ..llm.element_classifier import LLMElementClassifier
from ..utils.memory import AgentMemory


@dataclass
class AgentConfig:
    max_steps: int = 6
    stop_confidence: float = 0.8
    enable_llm: bool = True
    config_path: Optional[str] = None


@dataclass
class AgentOutcome:
    element_guid: str
    regulation_category: RegulationCategory
    function_classification: str
    confidence: float
    reasoning_path: List[str]
    evidence: List[str]
    alternatives: List[Dict[str, Any]]
    requires_review: bool


class SemanticAlignmentReActAgent:
    """基于 ReAct/CoT 的语义对齐代理。

    通过计划-行动-观测-记忆循环，调用现有的工具模块（几何分析、功能推断、LLM 分类、法规解析），
    将 IFC 元素对齐到法规分类，并在记忆中记录推理轨迹。
    """

    def __init__(self, agent_config: Optional[AgentConfig] = None):
        self.cfg = agent_config or AgentConfig()

        # 工具模块
        self.geometry_analyzer = GeometryAnalyzer(
            config_path=self.cfg.config_path, enable_llm=self.cfg.enable_llm
        )
        self.function_engine = FunctionInferenceEngine(
            config_path=self.cfg.config_path, enable_llm=self.cfg.enable_llm
        )
        self.vertical_detector = VerticalSpaceDetector()
        self.regulation_parser = RegulationParser()

        # LLM
        self.llm_client = LLMClient(self.cfg.config_path) if self.cfg.enable_llm else None
        self.element_classifier = LLMElementClassifier(self.cfg.config_path) if self.cfg.enable_llm else None
        self.prompt_templates = PromptTemplates()

        # 记忆
        self.memory = AgentMemory()

    # -------- 工具（Actions） --------
    def _tool_get_geometry(self, element: IfcElementInfo, context: Optional[SpatialContext]) -> Dict[str, Any]:
        # 提供健壮的几何特征兜底，避免字段不匹配导致异常
        try:
            bbox = element.bounding_box or self._default_bbox()
            area = abs((bbox.max_point.x - bbox.min_point.x) * (bbox.max_point.y - bbox.min_point.y))
            volume = abs((bbox.max_point.x - bbox.min_point.x) * (bbox.max_point.y - bbox.min_point.y) * (bbox.max_point.z - bbox.min_point.z))
            thickness = abs(bbox.max_point.z - bbox.min_point.z) if element.ifc_type == IfcElementType.SLAB else None
            features = GeometricFeatures(
                bounding_box=bbox,
                area=area,
                volume=volume,
                thickness=thickness,
                centroid=bbox.center,
                floor_level=getattr(context, "floor_level", None) if context else None,
                floors_spanned=1,
                is_vertical_element=False,
                cross_section_area=None,
            )
            element.geometric_features = features
            log.info(f"Geometry features computed for {element.guid}: area={area:.3f}, volume={volume:.3f}, thickness={thickness}")
            return {"type": "geometry_features", "features": self._safe_serialize(features)}
        except Exception as e:
            log.error(f"Geometry feature computation failed for {element.guid}: {e}")
            return {"type": "geometry_features_error", "error": str(e)}

    def _tool_infer_function(self, element: IfcElementInfo, context: Optional[SpatialContext]) -> Dict[str, Any]:
        inference = self.function_engine.infer_element_function(element, context)
        element.functional_inference = inference
        log.info(f"Function inferred for {element.guid}: primary={inference.primary_function.value}, confidence={inference.confidence}")
        return {
            "type": "function_inference",
            "primary_function": str(inference.primary_function.value),
            "confidence": inference.confidence,
            "evidence": [ev.text for ev in inference.evidence],
            "alternatives": [(alt[0].value, alt[1]) for alt in inference.alternatives],
        }

    def _tool_llm_classify(self, element: IfcElementInfo, building_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.element_classifier or not self.llm_client or not self.llm_client.is_available():
            log.warning(f"LLM classification unavailable for {element.guid}")
            return {"type": "llm_classification", "error": "LLM unavailable"}
        geom = self._safe_serialize(element.geometric_features)
        spatial_context = {"adjacent_spaces": [], "floor_level": geom.get("floor_level")}
        result = self.element_classifier.classify_element(
            {
                "ifc_type": element.ifc_type.value,
                "guid": element.guid,
                "properties": element.properties,
            },
            geom,
            spatial_context,
            building_context or {},
        )
        log.info(f"LLM classified {element.guid}: primary_function={result.primary_function}, confidence={result.confidence_score}")
        return {
            "type": "llm_classification",
            "primary_function": result.primary_function,
            "sub_category": result.sub_category,
            "usage_intensity": result.usage_intensity,
            "confidence": result.confidence_score,
            "regulatory_hints": result.regulatory_hints,
        }

    def _tool_parse_regulations(self, regulation_data: Dict[str, Any], target_region: Optional[str]) -> Dict[str, Any]:
        rules = self.regulation_parser.parse_regulation_output(regulation_data, target_region)
        feature_map = self.regulation_parser.create_feature_mapping(rules)
        return {"type": "regulation_features", "feature_map": self._safe_serialize(feature_map)}

    def _tool_rule_align_a(self, element: IfcElementInfo) -> Dict[str, Any]:
        # 简化版规则映射：基于功能推断 + 厚度指标
        func = (element.functional_inference.primary_function.value if element.functional_inference else "unknown")
        thickness = getattr(element.geometric_features, "thickness", None)
        category = RegulationCategory.INCLUDE_FULL
        confidence = 0.6

        if func in ["decoration_platform"]:
            category = RegulationCategory.INCLUDE_PARTIAL
            confidence = 0.7
        elif func in ["auxiliary_room", "auxiliary_construction"]:
            category = RegulationCategory.CONDITIONAL
            confidence = 0.65
        elif func in ["mechanical_room"]:
            category = RegulationCategory.INCLUDE_PARTIAL
            confidence = 0.75
        elif func in ["atrium"]:
            category = RegulationCategory.INCLUDE_FULL
            confidence = 0.8
        elif func in ["shaft"]:
            category = RegulationCategory.INCLUDE_PARTIAL
            confidence = 0.7
        else:
            # 基于厚度的兜底判断
            if thickness is not None and thickness < 0.1:
                category = RegulationCategory.INCLUDE_PARTIAL
                confidence = 0.7

        return {
            "type": "rule_align_a",
            "regulation_category": category.value,
            "confidence": confidence,
        }

    # -------- Agent 主循环 --------
    def align_element(self,
                      element: IfcElementInfo,
                      regulation_data: Dict[str, Any],
                      building_context: Optional[Dict[str, Any]] = None,
                      spatial_context: Optional[SpatialContext] = None,
                      target_region: Optional[str] = None) -> AgentOutcome:
        self.memory.start_episode()
        guid = element.guid
        reasoning: List[str] = []
        evidence: List[str] = []
        alternatives: List[Dict[str, Any]] = []

        final_category = RegulationCategory.UNKNOWN
        final_function = "unknown"
        final_conf = 0.5
        requires_review = True

        for step in range(1, self.cfg.max_steps + 1):
            plan_context = {
                "element": {"guid": guid, "type": element.ifc_type.value},
                "episode": self.memory.get_episode(),
            }
            plan_prompt = self.prompt_templates.get_react_plan_prompt(context=self._to_json(plan_context))
            plan = self._gen_json(plan_prompt)
            if not plan:
                break

            thought = plan.get("thought", "")
            action = plan.get("action", {})
            self.memory.add_plan(guid, step, thought, action)
            reasoning.append(f"Thought[{step}]: {thought}")
            log.info(f"Step[{step}] planned: action={action.get('tool')} | thought={thought}")

            obs: Dict[str, Any] = {"type": "noop"}
            tool = (action or {}).get("tool")
            args = (action or {}).get("args", {})
            if tool == "GetGeometryFeatures":
                obs = self._tool_get_geometry(element, spatial_context)
            elif tool == "RuleAlignA":
                obs = self._tool_rule_align_a(element)
            elif tool == "RuleAlignB":
                # 目前不处理 B 类，保留接口
                obs = {"type": "rule_align_b", "note": "not_implemented"}
            elif tool == "LLMClassifyElement":
                obs = self._tool_llm_classify(element, building_context)
            elif tool == "ParseRegulations":
                obs = self._tool_parse_regulations(regulation_data, args.get("region"))
            elif tool == "InferFunction":
                obs = self._tool_infer_function(element, spatial_context)
            else:
                obs = {"type": "invalid_tool", "tool": tool}

            # 记录 observation
            self.memory.add_observation(guid, step, obs)
            evidence.append(f"Obs[{step}]: {str(obs)[:200]}")
            log.info(f"Step[{step}] observation: type={obs.get('type')}")

            # 反思与停机判定
            reflect_ctx = {
                "element": {"guid": guid, "type": element.ifc_type.value},
                "last_observation": obs,
                "episode": self.memory.get_episode(),
            }
            reflect_prompt = self.prompt_templates.get_react_reflect_prompt(
                context=self._to_json(reflect_ctx), observation=self._to_json(obs)
            )
            reflect = self._gen_json(reflect_prompt)
            if reflect and reflect.get("should_stop"):
                # 汇总当前最好的结论
                final_category, final_function, final_conf = self._summarize_alignment(element, obs)
                log.info(f"Stopping at step {step}: category={final_category}, function={final_function}, confidence={final_conf}")
                break

        # 若循环结束仍无明确结论，进行兜底汇总
        if final_category == RegulationCategory.UNKNOWN:
            final_category, final_function, final_conf = self._summarize_alignment(element, None)
            log.info(f"Final summary (fallback): category={final_category}, function={final_function}, confidence={final_conf}")

        requires_review = final_conf < self.cfg.stop_confidence
        self.memory.commit_long_term(guid)

        return AgentOutcome(
            element_guid=guid,
            regulation_category=final_category,
            function_classification=final_function,
            confidence=final_conf,
            reasoning_path=reasoning,
            evidence=evidence,
            alternatives=alternatives,
            requires_review=requires_review,
        )

    # -------- 辅助方法 --------
    def _summarize_alignment(self, element: IfcElementInfo, last_obs: Optional[Dict[str, Any]]):
        # 优先：使用最近一次 RuleAlignA 的输出
        if last_obs and last_obs.get("type") == "rule_align_a":
            cat = last_obs.get("regulation_category", RegulationCategory.UNKNOWN.value)
            conf = float(last_obs.get("confidence", 0.6))
            func = (element.functional_inference.primary_function.value if element.functional_inference else "unknown")
            return RegulationCategory(cat), func, conf

        # 次优：使用 LLM 分类的 regulatory_hints
        func = (element.functional_inference.primary_function.value if element.functional_inference else "unknown")
        conf = (element.functional_inference.confidence if element.functional_inference else 0.5)
        cat = RegulationCategory.INCLUDE_FULL
        if func in ["decoration_platform"]:
            cat = RegulationCategory.INCLUDE_PARTIAL
        elif func in ["auxiliary_room", "auxiliary_construction", "shaft"]:
            cat = RegulationCategory.CONDITIONAL
        return cat, func, conf

    def _gen_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not self.llm_client or not self.llm_client.is_available():
            return None
        try:
            return self.llm_client.generate_json_completion(prompt, temperature=0.2)
        except Exception:
            return None

    @staticmethod
    def _safe_serialize(obj: Any) -> Dict[str, Any]:
        try:
            if hasattr(obj, "__dict__"):
                return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                        for k, v in obj.__dict__.items()}
            if isinstance(obj, dict):
                return obj
            return {"value": str(obj)}
        except Exception:
            return {"value": "serialization_error"}

    @staticmethod
    def _to_json(obj: Any) -> str:
        import json
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    @staticmethod
    def _default_bbox():
        return BoundingBox(
            min_point=Point3D(0.0, 0.0, 0.0),
            max_point=Point3D(1.0, 1.0, 0.2),
        )