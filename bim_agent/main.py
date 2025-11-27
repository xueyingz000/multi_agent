from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import json
import os
import math


# ===========================
# 模块 1: 全能几何与属性分析器 (融合版)
# ===========================
class GeometryAnalyzer:
    def __init__(self, ifc_file_path):
        print(f"[系统] 正在加载模型: {ifc_file_path} ...")
        self.file = ifcopenshell.open(ifc_file_path)
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)

    def get_element_features(self, element):
        psets = ifcopenshell.util.element.get_psets(element)
        geo_data = self._get_geometry(element)

        # 1. 提取 IsExternal (兼容 main_a 逻辑)
        is_external = "UNKNOWN"
        for pset_name, props in psets.items():
            if "Common" in pset_name and "IsExternal" in props:
                val = props["IsExternal"]
                is_external = str(val).upper()
                break

        # 2. 提取空间上下文 (兼容 main_c 逻辑)
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
        # 融合逻辑：确保包含 main_a 需要的 thickness/bbox_dims
        data = {
            "height": 0.0,
            "width": 0.0,
            "depth": 0.0,
            "area": 0.0,
            "volume": 0.0,
            "thickness": 0.0,
            "aspect_ratio": 1.0,
        }
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            verts = shape.geometry.verts
            xs, ys, zs = verts[0::3], verts[1::3], verts[2::3]
            dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)

            # main_a 需要 bbox_dims
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
            elif element.is_a("IfcWall"):
                data["thickness"] = round(min(dx, dy), 2)

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
    def __init__(self, api_key, model_name="gpt-4o", base_url=None):
        self.llm = ChatOpenAI(
            model=model_name, temperature=0, api_key=api_key, base_url=base_url
        )

        # Prompt A (main_a.py 原版)
        self.prompt_general = PromptTemplate.from_template(
            """
            You are a BIM Data Governance Expert. Map the IFC element to a Functional Category.
            
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

        # Prompt B (main_c.py 原版)
        self.prompt_vertical = PromptTemplate.from_template(
            """
            You are a BIM Code Expert. Distinguish between 'FUNCTIONAL_SHAFT' and 'ATRIUM' based on the rules below.

            [Input Data]
            - Type: {ifc_type}
            - Name: {name}
            - Dimensions: H={height}m, Area={area}m2, AspectRatio={aspect_ratio}
            - Floor Span: ~{floor_span} floors
            - Contents: Stair={has_stair}, Elevator={has_transport}, MEP={has_mep}
            - Enclosure: {boundary_type} (Bounded by Wall or Railing?)

            [Decision Logic for Vertical Spaces]
            1. **Multi-floor Check**: If Floor Span < 1.5, likely 'GENERAL_ROOM' or 'CORRIDOR', not a shaft/atrium.
            2. **Content Check**: 
               - If contains Stair/Ramp -> 'STAIRWELL' (a type of Shaft).
               - If contains Elevator -> 'ELEVATOR_SHAFT'.
               - If contains MEP -> 'MEP_SHAFT'.
            3. **Enclosure Check (Crucial)**:
               - If enclosed by **RAILING** -> Likely 'ATRIUM' (Open void).
               - If enclosed by **WALL** -> Likely 'FUNCTIONAL_SHAFT'.
            4. **Size & Shape Check**:
               - If Area > 50m2 AND AspectRatio < 2.0 (Square/Large) -> 'ATRIUM'.
               - If Area < 15m2 OR AspectRatio > 3.0 (Narrow/Small) -> 'FUNCTIONAL_SHAFT'.

            [Allowed Categories]
            {valid_candidates}

            [Task]
            Classify the element based on the logic above.
            Return JSON ONLY:
            {{
                "category": "CATEGORY_NAME",
                "confidence": 0.0-1.0,
                "reasoning": "Step-by-step check: 1. Span=... 2. Content=... 3. Shape=... therefore..."
            }}
            """
        )

    def classify(self, features, all_categories):
        # 1. 路由 (Router)
        ifc_type = features["ifc_type"]
        span = features["spatial_context"]["floor_span_est"]
        name = features["name"].lower()

        mode = "GENERAL"

        # 触发垂直模式的条件：洞口，代理，或者高层空间，或者特定命名空间
        if ifc_type in ["IfcOpeningElement", "IfcBuildingElementProxy"]:
            mode = "VERTICAL"
        elif ifc_type == "IfcSpace" and span >= 1.5:
            mode = "VERTICAL"
        elif ifc_type == "IfcSpace" and any(
            k in name for k in ["shaft", "atrium", "void", "riser", "lift"]
        ):
            mode = "VERTICAL"

        # 2. 候选集过滤 (Critical Fix)
        relevant_candidates = []
        vertical_keywords = ["SHAFT", "ATRIUM", "STAIRWELL", "ELEVATOR", "VOID"]

        for c in all_categories:
            if ifc_type in c["valid_ifc_types"]:
                if mode == "VERTICAL":
                    # Vertical 模式：只保留 Shaft/Atrium 等
                    cat_upper = c["name"].upper()
                    is_vertical_cat = any(k in cat_upper for k in vertical_keywords)
                    if is_vertical_cat or c["name"] == "GENERAL_ROOM":
                        relevant_candidates.append(c["name"])
                else:
                    # General 模式：完全放开，和 main_a 一样，不做任何过滤
                    relevant_candidates.append(c["name"])

        if not relevant_candidates:
            relevant_candidates = ["GENERIC_ELEMENT"]

        # 3. 构造 Input
        if mode == "GENERAL":
            # [严格复刻 main_a.py]
            psets_str = str(features["psets_raw"])[:1000]
            input_vars = {
                "ifc_type": features["ifc_type"],
                "predefined_type": features["predefined_type"],
                "is_external": features["is_external"],
                "name": features["name"],
                "dimensions": json.dumps(features["dimensions"]),
                "psets_snippet": psets_str,
                "valid_candidates": json.dumps(relevant_candidates),
            }
            prompt = self.prompt_general

        else:  # VERTICAL
            # [严格复刻 main_c.py]
            geo = features["dimensions"]
            spatial = features["spatial_context"]
            input_vars = {
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

        # 4. 执行 LLM
        try:
            chain = prompt | self.llm
            response = chain.invoke(input_vars)
            content = response.content.strip().replace("```json", "").replace("```", "")
            result = json.loads(content)
        except Exception as e:
            return {"category": "ERROR", "confidence": 0.0, "reasoning": str(e)}

        # 5. 置信度校准 (分流)
        if mode == "GENERAL":
            final_conf = self._calibrate_general(features, result)
        else:
            final_conf = self._calibrate_vertical(features, result)

        result["confidence"] = final_conf
        return result

    # def _calibrate_general(self, features, llm_result):
    #     """
    #     [回滚逻辑] 严格复刻 main_a.py 的校准逻辑
    #     """
    #     raw_llm_score = float(llm_result.get("confidence", 0.5))
    #     score = 0.5 + (raw_llm_score - 0.5) * 0.5

    #     p_type = features.get("predefined_type", "")
    #     is_ext = features.get("is_external", "UNKNOWN")
    #     cat_name = llm_result.get("category", "")
    #     elem_name = features.get("name", "").upper()
    #     dims = features.get("dimensions", {})

    #     # [完全一致的 Bonus]
    #     strong_types = ["ROOF", "BEAM", "COLUMN", "FLOOR", "BASESLAB", "STAIR", "RAMP"]
    #     if p_type and p_type not in ["NOTDEFINED", "USERDEFINED", "None"]:
    #         if p_type in strong_types:
    #             score += 0.20
    #         else:
    #             score += 0.10

    #     if is_ext in ["TRUE", "FALSE"]:
    #         score += 0.08

    #     clean_cat = cat_name.replace("_", "").replace(" ", "")
    #     if clean_cat in elem_name.replace(" ", "") or elem_name in cat_name:
    #         score += 0.12

    #     if dims.get("area", 0) > 0.1 or dims.get("volume", 0) > 0.1:
    #         score += 0.05

    #     # [完全一致的 Penalties]
    #     if cat_name == "GENERIC_ELEMENT":
    #         score -= 0.15
    #         if score > 0.75:
    #             score = 0.75

    #     if dims.get("area", 0) == 0 and dims.get("volume", 0) == 0:
    #         score -= 0.30

    #     if len(elem_name) < 3:
    #         score -= 0.05

    #     return round(max(0.1, min(score, 0.99)), 2)
    def _calibrate_general(self, features, llm_result):
        """
        [回滚逻辑] 严格复刻 main_a.py 的校准逻辑
        注意：为了确保分数完全一致，必须模拟 main_a.py 中“较弱”的几何数据输入。
        main_a.py 中只有 IfcSpace 有 Area，且几乎所有构件 Volume 均为 0。
        """
        raw_llm_score = float(llm_result.get("confidence", 0.5))
        score = 0.5 + (raw_llm_score - 0.5) * 0.5

        p_type = features.get("predefined_type", "")
        is_ext = features.get("is_external", "UNKNOWN")
        cat_name = llm_result.get("category", "")
        elem_name = features.get("name", "").upper()
        dims = features.get("dimensions", {})

        # --- 关键修正开始: 模拟 main_a.py 的数据环境 ---
        # main.py 的 GeometryAnalyzer 计算了所有构件的 Volume 和 Opening 的 Area。
        # 但 main_a.py 中，只有 IfcSpace 有 Area，且 Volume 始终为 0 (初始化后未赋值)。
        # 为了分数一致，我们必须在这里“降级”数据，忽略多算出来的几何信息。

        simulated_area = 0.0
        if features["ifc_type"] == "IfcSpace":
            simulated_area = dims.get("area", 0.0)

        simulated_volume = 0.0  # main_a.py 中 Volume 始终为 0
        # --- 关键修正结束 ---

        # [完全一致的 Bonus]
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

        # 使用模拟数据进行判断
        if simulated_area > 0.1 or simulated_volume > 0.1:
            score += 0.05

        # [完全一致的 Penalties]
        if cat_name == "GENERIC_ELEMENT":
            score -= 0.15
            if score > 0.75:
                score = 0.75

        # 使用模拟数据进行判断 (这会导致 Wall/Slab 再次被扣 0.3 分，与 main_a 行为一致)
        if simulated_area == 0 and simulated_volume == 0:
            score -= 0.30

        if len(elem_name) < 3:
            score -= 0.05

        return round(max(0.1, min(score, 0.99)), 2)

    def _calibrate_vertical(self, features, llm_result):
        """
        [严格复刻 main_c.py V4]
        """
        score = float(llm_result.get("confidence", 0.5))
        cat_name = llm_result.get("category", "").upper()
        elem_name = features.get("name", "").lower()
        dims = features.get("dimensions", {})
        area = dims.get("area", 0.0)
        spatial = features.get("spatial_context", {})

        is_fallback_cat = cat_name in ["GENERAL_ROOM", "GENERIC_ELEMENT", "UNKNOWN"]

        if is_fallback_cat:
            void_keywords = ["opening", "void", "hole", "penetration", "recess"]
            if any(kw in elem_name for kw in void_keywords):
                score -= 0.45
            elif area < 2.5:
                score -= 0.3
            if "room" not in elem_name and "office" not in elem_name:
                if score > 0.7:
                    score = 0.7

        if cat_name == "ATRIUM" and area < 15.0:
            score -= 0.5

        if "SHAFT" in cat_name:
            has_strong_evidence = (
                spatial.get("contains_stair")
                or spatial.get("contains_transport")
                or spatial.get("boundary_material") == "WALL"
                or "shaft" in elem_name
            )
            if not has_strong_evidence:
                score -= 0.25
                if score > 0.65:
                    score = 0.65

        clean_cat = cat_name.replace("_", "").replace(" ", "")
        clean_name = elem_name.replace(" ", "").replace("-", "").upper()
        if clean_cat in clean_name:
            score += 0.15

        if cat_name == "STAIRWELL" and spatial.get("contains_stair"):
            score += 0.25
        elif cat_name == "ATRIUM" and spatial.get("boundary_material") == "RAILING":
            score += 0.2

        return round(max(0.2, min(score, 0.99)), 2)


# ===========================
# 模块 3: 主流程
# ===========================
def main():
    ifc_path = "group_a_dataset.ifc"

    if not os.path.exists(ifc_path):
        print(f"错误: 未找到 {ifc_path}")
        return

    CONFIDENCE_THRESHOLD = 0.65
    api_key = "sk-Dppz8ZLK62lWyZc7G3w3LtkJx7sNgmjdF65kzR7hKtQlnQDL"
    base_url = "https://yunwu.ai/v1"

    # [关键调整] 这里的列表范围决定了 LLM 的“视野”
    # 如果想和 main_a.py 完全一致，这里就不能包含 Beam/Column
    # 如果想用更好的逻辑，就把 Beam/Column 加回来
    full_categories = [
        # --- A/B 类 (严格匹配 main_a.py 的范围) ---
        {"name": "INTERIOR_FLOOR_SLAB", "valid_ifc_types": ["IfcSlab"]},
        {"name": "ROOF_SLAB", "valid_ifc_types": ["IfcSlab", "IfcRoof"]},
        {"name": "BALCONY", "valid_ifc_types": ["IfcSlab"]},
        {"name": "FOUNDATION_SLAB", "valid_ifc_types": ["IfcSlab"]},
        {"name": "STAIR_LANDING", "valid_ifc_types": ["IfcSlab"]},
        {
            "name": "EXTERIOR_WALL",
            "valid_ifc_types": ["IfcWall", "IfcWallStandardCase"],
        },
        {
            "name": "INTERIOR_WALL",
            "valid_ifc_types": ["IfcWall", "IfcWallStandardCase"],
        },
        {"name": "SHEAR_WALL", "valid_ifc_types": ["IfcWall"]},
        {"name": "CURTAIN_WALL", "valid_ifc_types": ["IfcCurtainWall", "IfcWall"]},
        {"name": "MECHANICAL_ROOM", "valid_ifc_types": ["IfcSpace"]},
        {
            "name": "GENERAL_ROOM",
            "valid_ifc_types": [
                "IfcSpace",
                "IfcOpeningElement",
                "IfcBuildingElementProxy",
            ],
        },
        {"name": "CORRIDOR", "valid_ifc_types": ["IfcSpace"]},
        {"name": "INDOOR_PARKING", "valid_ifc_types": ["IfcSpace"]},
        # --- C/D 类 ---
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
            "name": "MEP_SHAFT",
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
        # 注意：我注释掉了 Beam/Column，为了让 Group A 跑出和你 main_a.py 一样的“低分”结果，证明逻辑的一致性。
        # 如果你希望 Beam 跑出高分，请取消注释。
        # {"name": "STRUCTURAL_COLUMN",   "valid_ifc_types": ["IfcColumn"]},
        # {"name": "STRUCTURAL_BEAM",     "valid_ifc_types": ["IfcBeam"]},
        # {"name": "GENERIC_ELEMENT",     "valid_ifc_types": ["IfcBuildingElementProxy", "IfcRailing", "IfcRamp", "IfcStair"]},
    ]

    analyzer = GeometryAnalyzer(ifc_path)
    classifier = IntelligentClassifier(api_key, base_url=base_url)

    elements = analyzer.file.by_type("IfcElement") + analyzer.file.by_type("IfcSpace")
    print(f"开始分析 {len(elements)} 个构件 (Mode: General vs Vertical)...\n")

    results = []
    hitl_queue = []

    for idx, elem in enumerate(elements):
        features = analyzer.get_element_features(elem)
        cls_result = classifier.classify(features, full_categories)

        conf = cls_result["confidence"]
        cat = cls_result["category"]
        status = "AUTO_MAPPED" if conf >= CONFIDENCE_THRESHOLD else "NEEDS_REVIEW"
        log_prefix = "[✓]" if status == "AUTO_MAPPED" else "[!]"

        if status == "NEEDS_REVIEW":
            hitl_queue.append(
                {
                    "guid": features["guid"],
                    "name": features["name"],
                    "guess": cat,
                    "conf": conf,
                    "reason": cls_result["reasoning"],
                }
            )

        print(
            f"{idx+1}. {log_prefix} {features['name'][:30]:<30} -> {cat:<20} (Conf: {conf})"
        )

        results.append(
            {
                "guid": features["guid"],
                "name": features["name"],
                "category": cat,
                "confidence": conf,
                "status": status,
                "reasoning": cls_result["reasoning"],
            }
        )

    # 收集待复核队列和全部分类列表
    output_data = {
        "meta": {
            "total_count": len(results),
            "review_count": len(hitl_queue),
            # 提取所有类别的名称供前端下拉框使用
            "all_categories": [c["name"] for c in full_categories],
        },
        "results": results,  # 包含所有构件，前端根据 status == "NEEDS_REVIEW" 过滤
    }

    print(f"\n分析完成。需复核: {len(hitl_queue)}/{len(results)}")

    # 保存为 hitl_data.json
    with open("hitl_data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
