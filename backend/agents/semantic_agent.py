import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import json
import os
import math
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# ===========================
# 模块 1: 全能几何与属性分析器 (融合版)
# ===========================
class GeometryAnalyzer:
    def __init__(self, ifc_file):
        # 接收已打开的 ifc_file 对象，而不是路径，避免重复 IO
        self.file = ifc_file
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)

    def get_element_features(self, element):
        psets = ifcopenshell.util.element.get_psets(element)
        geo_data = self._get_geometry(element)

        # 1. 提取 IsExternal
        is_external = "UNKNOWN"
        for pset_name, props in psets.items():
            if "Common" in pset_name and "IsExternal" in props:
                val = props["IsExternal"]
                is_external = str(val).upper()
                break

        # 2. 提取空间上下文
        spatial_data = self._analyze_spatial_context(element, geo_data)

        features = {
            "guid": element.GlobalId,
            "ifc_type": element.is_a(),
            "predefined_type": ifcopenshell.util.element.get_predefined_type(element),
            "name": element.Name if element.Name else "",
            "is_external": is_external,
            "dimensions": geo_data,
            "spatial_context": spatial_data,
            "psets_raw": psets,
        }
        return features

    def _get_geometry(self, element):
        data = {
            "height": 0.0,
            "width": 0.0,
            "depth": 0.0,
            "area": 0.0,
            "volume": 0.0,
            "thickness": 0.0,
            "aspect_ratio": 1.0,
            "bbox_dims": [0, 0, 0],
        }
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            verts = shape.geometry.verts
            xs, ys, zs = verts[0::3], verts[1::3], verts[2::3]
            dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)

            data["bbox_dims"] = [round(dx, 2), round(dy, 2), round(dz, 2)]
            data["height"] = round(dz, 2)
            dims_xy = sorted([dx, dy])
            width, depth = dims_xy[0], dims_xy[1]
            data["width"] = round(width, 2)
            data["depth"] = round(depth, 2)

            if width > 0.1:
                data["aspect_ratio"] = round(depth / width, 1)
            else:
                data["aspect_ratio"] = 99.9

            if element.is_a("IfcSpace") or element.is_a("IfcOpeningElement"):
                data["area"] = round(dx * dy, 2)
            elif element.is_a("IfcSlab"):
                data["thickness"] = round(dz, 2)

            data["volume"] = round(dx * dy * dz, 2)

        except:
            data["note"] = "No Geometry"
        return data

    def _analyze_spatial_context(self, element, geo_data):
        context = {
            "contains_transport": False,
            "contains_stair": False,
            "contains_mep": False,
            "boundary_material": "UNKNOWN",
            "floor_span_est": 1.0,
        }
        if geo_data["height"] > 0:
            context["floor_span_est"] = round(geo_data["height"] / 3.5, 1)

        if element.is_a("IfcSpace"):
            if hasattr(element, "ContainsElements"):
                for rel in element.ContainsElements:
                    for related_elem in rel.RelatedElements:
                        etype = related_elem.is_a()
                        if etype in ["IfcStair", "IfcRamp"]:
                            context["contains_stair"] = True
                        elif etype in ["IfcTransportElement", "IfcElevator"]:
                            context["contains_transport"] = True
                        elif "Flow" in etype:
                            context["contains_mep"] = True

            walls, railings = 0, 0
            if hasattr(element, "BoundedBy"):
                for rel in element.BoundedBy:
                    if not rel.RelatedBuildingElement:
                        continue
                    b_elem = rel.RelatedBuildingElement
                    if b_elem.is_a("IfcWall") or b_elem.is_a("IfcWallStandardCase"):
                        walls += 1
                    elif b_elem.is_a("IfcRailing"):
                        railings += 1
            if walls > railings:
                context["boundary_material"] = "WALL"
            elif railings > 0:
                context["boundary_material"] = "RAILING"

        return context


# ===========================
# 模块 2: 双模式智能分类代理
# ===========================
class IntelligentClassifier:
    def __init__(self, model_name="gpt-4o"):  # 使用 gpt-4o 或 gpt-4-turbo
        api_key = os.getenv("OPENAI_API_KEY")
        # 如果你有自定义 base_url，可以从 env 读取，否则保持默认
        base_url = os.getenv("OPENAI_BASE_URL", None)

        self.llm = ChatOpenAI(
            model=model_name, temperature=0, api_key=api_key, base_url=base_url
        )

        # Prompt A (General)
        self.prompt_general = PromptTemplate.from_template(
            """
            You are a BIM Data Governance Expert working under the regulation: {rule_context}.
            Map the IFC element to a Functional Category.
            
            [Element Data]
            - IFC Class: {ifc_type}
            - PredefinedType: {predefined_type}
            - IsExternal: {is_external}
            - Name: {name}
            - Dimensions: {dimensions}
            - Psets Snippet: {psets_snippet}
            
            [Allowed Categories]
            {valid_candidates}
            
            [Output Requirement]
            Return JSON ONLY:
            {{
                "category": "CATEGORY_NAME",
                "confidence": 0.0-1.0,
                "reasoning": "Brief explanation"
            }}
            """
        )

        # Prompt B (Vertical)
        self.prompt_vertical = PromptTemplate.from_template(
            """
            You are a BIM Code Expert working under the regulation: {rule_context}.
            Distinguish between 'FUNCTIONAL_SHAFT' and 'ATRIUM'.

            [Input Data]
            - Type: {ifc_type}
            - Name: {name}
            - Dimensions: H={height}m, Area={area}m2, AspectRatio={aspect_ratio}
            - Floor Span: ~{floor_span} floors
            - Contents: Stair={has_stair}, Elevator={has_transport}, MEP={has_mep}
            - Enclosure: {boundary_type} (Bounded by Wall or Railing?)

            [Decision Logic for Vertical Spaces]
            1. **Multi-floor Check**: If Floor Span < 1.5, likely 'GENERAL_ROOM' or 'CORRIDOR'.
            2. **Content Check**: 
               - Stair/Ramp -> 'STAIRWELL'
               - Elevator -> 'ELEVATOR_SHAFT'
               - MEP -> 'MEP_SHAFT'
            3. **Enclosure Check**:
               - RAILING -> Likely 'ATRIUM'
               - WALL -> Likely 'FUNCTIONAL_SHAFT'
            4. **Size**: Large+Square -> ATRIUM; Small+Narrow -> SHAFT.

            [Allowed Categories]
            {valid_candidates}

            [Task]
            Classify the element. Return JSON ONLY:
            {{
                "category": "CATEGORY_NAME",
                "confidence": 0.0-1.0,
                "reasoning": "Reasoning..."
            }}
            """
        )

    def classify(self, features, all_categories, rule_context_desc="Standard"):
        # 1. Router
        ifc_type = features["ifc_type"]
        span = features["spatial_context"]["floor_span_est"]
        name = features["name"].lower()
        mode = "GENERAL"

        if ifc_type in ["IfcOpeningElement", "IfcBuildingElementProxy"]:
            mode = "VERTICAL"
        elif ifc_type == "IfcSpace" and span >= 1.5:
            mode = "VERTICAL"
        elif ifc_type == "IfcSpace" and any(
            k in name for k in ["shaft", "atrium", "void", "riser", "lift"]
        ):
            mode = "VERTICAL"

        # 2. Filter Candidates
        relevant_candidates = []
        vertical_keywords = ["SHAFT", "ATRIUM", "STAIRWELL", "ELEVATOR", "VOID"]

        for c in all_categories:
            if ifc_type in c["valid_ifc_types"]:
                if mode == "VERTICAL":
                    cat_upper = c["name"].upper()
                    # Also include dynamic categories if they seem vertical or if we want to be permissive
                    # For now, stick to the keywords, but maybe add the new ones if they are relevant?
                    is_vertical_cat = any(k in cat_upper for k in vertical_keywords)
                    if is_vertical_cat or c["name"] == "GENERAL_ROOM":
                        relevant_candidates.append(c["name"])
                else:
                    relevant_candidates.append(c["name"])

        if not relevant_candidates:
            relevant_candidates = ["GENERIC_ELEMENT"]

        # 3. Input Construction
        if mode == "GENERAL":
            psets_str = str(features["psets_raw"])[:1000]
            input_vars = {
                "rule_context": rule_context_desc,
                "ifc_type": features["ifc_type"],
                "predefined_type": features["predefined_type"],
                "is_external": features["is_external"],
                "name": features["name"],
                "dimensions": json.dumps(features["dimensions"]),
                "psets_snippet": psets_str,
                "valid_candidates": json.dumps(relevant_candidates),
            }
            prompt = self.prompt_general
        else:
            geo = features["dimensions"]
            spatial = features["spatial_context"]
            input_vars = {
                "rule_context": rule_context_desc,
                "ifc_type": features["ifc_type"],
                "name": features["name"],
                "height": geo["height"],
                "area": geo["area"],
                "aspect_ratio": geo["aspect_ratio"],
                "floor_span": spatial["floor_span_est"],
                "has_stair": str(spatial["contains_stair"]),
                "has_transport": str(spatial["contains_transport"]),
                "has_mep": str(spatial["contains_mep"]),
                "boundary_type": spatial["boundary_material"],
                "valid_candidates": json.dumps(relevant_candidates),
            }
            prompt = self.prompt_vertical

        # 4. Invoke LLM
        try:
            chain = prompt | self.llm
            response = chain.invoke(input_vars)
            content = response.content.strip().replace("```json", "").replace("```", "")
            result = json.loads(content)
        except Exception as e:
            return {"category": "ERROR", "confidence": 0.0, "reasoning": str(e)}

        # 5. Calibration
        if mode == "GENERAL":
            final_conf = self._calibrate_general(features, result)
        else:
            final_conf = self._calibrate_vertical(features, result)

        result["confidence"] = final_conf
        return result

    def _calibrate_general(self, features, llm_result):
        # 保持你原始的校准逻辑
        raw_llm_score = float(llm_result.get("confidence", 0.5))
        score = 0.5 + (raw_llm_score - 0.5) * 0.5
        p_type = features.get("predefined_type", "")
        is_ext = features.get("is_external", "UNKNOWN")
        cat_name = llm_result.get("category", "")
        elem_name = features.get("name", "").upper()
        dims = features.get("dimensions", {})

        simulated_area = 0.0
        if features["ifc_type"] == "IfcSpace":
            simulated_area = dims.get("area", 0.0)
        simulated_volume = 0.0

        strong_types = ["ROOF", "BEAM", "COLUMN", "FLOOR", "BASESLAB", "STAIR", "RAMP"]
        if p_type and p_type not in ["NOTDEFINED", "USERDEFINED", "None"]:
            if p_type in strong_types:
                score += 0.20
            else:
                score += 0.10
        if is_ext in ["TRUE", "FALSE"]:
            score += 0.08
        clean_cat = cat_name.replace("_", "").replace(" ", "")
        if clean_cat in elem_name.replace(" ", "") or elem_name in cat_name:
            score += 0.12
        if simulated_area > 0.1 or simulated_volume > 0.1:
            score += 0.05
        if cat_name == "GENERIC_ELEMENT":
            score -= 0.15
            if score > 0.75:
                score = 0.75
        if simulated_area == 0 and simulated_volume == 0:
            score -= 0.30
        if len(elem_name) < 3:
            score -= 0.05
        return round(max(0.1, min(score, 0.99)), 2)

    def _calibrate_vertical(self, features, llm_result):
        # 保持你原始的垂直校准逻辑
        score = float(llm_result.get("confidence", 0.5))
        cat_name = llm_result.get("category", "").upper()
        elem_name = features.get("name", "").lower()
        dims = features.get("dimensions", {})
        area = dims.get("area", 0.0)
        spatial = features.get("spatial_context", {})

        is_fallback_cat = cat_name in ["GENERAL_ROOM", "GENERIC_ELEMENT", "UNKNOWN"]
        if is_fallback_cat:
            if any(kw in elem_name for kw in ["opening", "void", "hole"]):
                score -= 0.45
            elif area < 2.5:
                score -= 0.3
            if "room" not in elem_name and "office" not in elem_name:
                if score > 0.7:
                    score = 0.7
        if cat_name == "ATRIUM" and area < 15.0:
            score -= 0.5
        if "SHAFT" in cat_name:
            strong_ev = (
                spatial.get("contains_stair")
                or spatial.get("contains_transport")
                or spatial.get("boundary_material") == "WALL"
                or "shaft" in elem_name
            )
            if not strong_ev:
                score -= 0.25
                if score > 0.65:
                    score = 0.65
        if cat_name.replace("_", "") in elem_name.replace("-", "").upper():
            score += 0.15
        if cat_name == "STAIRWELL" and spatial.get("contains_stair"):
            score += 0.25
        elif cat_name == "ATRIUM" and spatial.get("boundary_material") == "RAILING":
            score += 0.2
        return round(max(0.2, min(score, 0.99)), 2)


# ===========================
# 模块 3: Agent 2 入口类 (供 main.py 调用)
# ===========================
class IfcSemanticAlignmentAgent:
    def __init__(self):
        # 定义分类标准 (这里可以从配置文件读取，也可以保留硬编码)
        self.full_categories = [
            {"name": "INTERIOR_FLOOR_SLAB", "valid_ifc_types": ["IfcSlab"]},
            {"name": "ROOF_SLAB", "valid_ifc_types": ["IfcSlab", "IfcRoof"]},
            {"name": "BALCONY", "valid_ifc_types": ["IfcSlab"]},
            {
                "name": "GENERAL_ROOM",
                "valid_ifc_types": [
                    "IfcSpace",
                    "IfcOpeningElement",
                    "IfcBuildingElementProxy",
                ],
            },
            {"name": "CORRIDOR", "valid_ifc_types": ["IfcSpace"]},
            {
                "name": "ATRIUM",
                "valid_ifc_types": [
                    "IfcSpace",
                    "IfcOpeningElement",
                    "IfcBuildingElementProxy",
                ],
            },
            {
                "name": "ELEVATOR_SHAFT",
                "valid_ifc_types": [
                    "IfcSpace",
                    "IfcOpeningElement",
                    "IfcBuildingElementProxy",
                ],
            },
            {
                "name": "STAIRWELL",
                "valid_ifc_types": [
                    "IfcSpace",
                    "IfcOpeningElement",
                    "IfcBuildingElementProxy",
                ],
            },
            {
                "name": "FUNCTIONAL_SHAFT",
                "valid_ifc_types": [
                    "IfcSpace",
                    "IfcOpeningElement",
                    "IfcBuildingElementProxy",
                ],
            },
        ]

    def _parse_rules_to_categories(self, rules_data):
        """
        从 Agent 1 的规则中提取动态分类
        """
        new_cats = []
        if not rules_data:
            return new_cats

        # Extract from special_space_requirements
        for req in rules_data.get("special_space_requirements", []):
            desc = req.get("description", "").lower()
            logic = req.get("condition_logic", "").lower()

            # 简单的关键词匹配，实际可以更复杂
            potential_names = [
                "parking",
                "auxiliary_room",
                "deformation_joint",
                "basement",
                "shared_area",
                "fire_refuge",
                "equipment_room",
            ]

            found_name = None
            for name in potential_names:
                if name.replace("_", " ") in desc or name in logic:
                    found_name = name.upper()
                    break

            if found_name:
                new_cats.append(
                    {
                        "name": found_name,
                        "valid_ifc_types": [
                            "IfcSpace",
                            "IfcBuildingElementProxy",
                            "IfcSlab",
                        ],
                    }
                )
        return new_cats

    def align(self, ifc_model, rules_data=None, stop_callback=None):
        """
        执行语义对齐
        :param ifc_model: 已由 ifcopenshell 打开的模型对象 (from main.py)
        :param rules_data: Agent 1 的输出结果 (dict)
        :param stop_callback: 可选的回调函数，返回 True 则中止分析
        """
        print("🧠 [Agent 2] Starting Semantic Alignment...")

        # 1. 提取法规名称与详细描述作为上下文
        rule_name = (
            rules_data.get("region", "Standard Regulations")
            if rules_data
            else "Standard Regulations"
        )

        rule_desc = f"Regulation Context: {rule_name}\n"
        if rules_data:
            for key in [
                "height_requirements",
                "enclosure_requirements",
                "special_space_requirements",
            ]:
                if key in rules_data and rules_data[key]:
                    rule_desc += f"\n[{key.replace('_', ' ').title()}]\n"
                    for r in rules_data[key]:
                        rule_desc += f"- {r.get('description')} (Logic: {r.get('condition_logic')})\n"

        # 2. 动态扩展分类体系
        dynamic_cats = self._parse_rules_to_categories(rules_data)
        active_categories = self.full_categories.copy()
        existing_names = {c["name"] for c in active_categories}

        for dc in dynamic_cats:
            if dc["name"] not in existing_names:
                active_categories.append(dc)
                existing_names.add(dc["name"])
                print(f"   + Added dynamic category: {dc['name']}")

        # 3. 初始化工具
        analyzer = GeometryAnalyzer(ifc_model)
        classifier = IntelligentClassifier()  # API Key 自动从 env 读取

        # 4. 筛选要分析的构件 (为了演示速度，这里只分析 Space 和 Slab)
        # 实际生产中应分析所有相关构件
        elements = ifc_model.by_type("IfcSpace") + ifc_model.by_type("IfcSlab")

        results = {}
        hitl_queue = []
        CONFIDENCE_THRESHOLD = 0.65

        # 5. 批处理循环
        total = len(elements)
        print(f"   Analyzing {total} elements under context: {rule_name}")

        for idx, elem in enumerate(elements):
            if stop_callback and stop_callback():
                print("🛑 [Agent 2] Analysis stopped by user request.")
                break

            try:
                # 特征提取
                features = analyzer.get_element_features(elem)
                # LLM 分类 (传入 active_categories 和 rule_desc)
                cls_result = classifier.classify(features, active_categories, rule_desc)

                guid = features["guid"]
                conf = cls_result["confidence"]
                cat = cls_result["category"]

                status = (
                    "AUTO_MAPPED" if conf >= CONFIDENCE_THRESHOLD else "NEEDS_REVIEW"
                )

                # 结果结构化
                result_entry = {
                    "guid": guid,
                    "name": features["name"],
                    "ifc_type": features["ifc_type"],
                    "semantic_category": cat,
                    "confidence": conf,
                    "status": status,
                    "reasoning": cls_result["reasoning"],
                    # "geometry": features["dimensions"] # 可选：传回几何信息
                }

                results[guid] = result_entry

                if status == "NEEDS_REVIEW":
                    hitl_queue.append(result_entry)

                if idx % 10 == 0:
                    print(f"   Processed {idx}/{total}...")

            except Exception as e:
                print(f"   Error processing {elem.GlobalId}: {e}")

        print(
            f"✅ [Agent 2] Alignment Complete. Review needed: {len(hitl_queue)}/{total}"
        )

        return {
            "alignment_results": results,
            "hitl_queue": hitl_queue,
            "meta": {"total": total, "needs_review": len(hitl_queue)},
        }
