from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import json
import os
import math


# ===========================
# 模块 1: 几何与空间关系分析器 (针对 Shaft/Atrium 增强)
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

        # --- 新增：高级空间分析 ---
        spatial_data = self._analyze_spatial_context(element, geo_data)

        features = {
            "guid": element.GlobalId,
            "ifc_type": element.is_a(),
            "predefined_type": ifcopenshell.util.element.get_predefined_type(element),
            "name": element.Name if element.Name else "",
            "is_external": str(
                psets.get("Pset_SpaceCommon", {}).get("IsExternal", "UNKNOWN")
            ),
            "dimensions": geo_data,
            "spatial_context": spatial_data,  # <--- 关键的新增特征
            "psets_raw": psets,
        }
        return features

    def _get_geometry(self, element):
        data = {
            "height": 0.0,
            "width": 0.0,
            "depth": 0.0,
            "area": 0.0,
            "aspect_ratio": 1.0,
        }
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            verts = shape.geometry.verts
            xs, ys, zs = verts[0::3], verts[1::3], verts[2::3]
            dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)

            data["height"] = round(dz, 2)
            # 计算长宽（平面投影）
            dims_xy = sorted([dx, dy])
            width, depth = dims_xy[0], dims_xy[1]  # width 是短边，depth 是长边

            data["width"] = round(width, 2)
            data["depth"] = round(depth, 2)

            # 长宽比 (Aspect Ratio)
            if width > 0.1:
                data["aspect_ratio"] = round(depth / width, 1)
            else:
                data["aspect_ratio"] = 99.9

            if element.is_a("IfcSpace"):
                data["area"] = round(dx * dy, 2)  # 简化投影面积
            elif element.is_a("IfcOpeningElement"):
                data["area"] = round(dx * dy, 2)

        except:
            data["note"] = "No Geometry"
        return data

    def _analyze_spatial_context(self, element, geo_data):
        """
        专门用于区分 Atrium 和 Functional Shaft 的逻辑
        """
        context = {
            "contains_transport": False,  # 是否包含电梯/扶梯
            "contains_stair": False,  # 是否包含楼梯
            "contains_mep": False,  # 是否包含机电管线
            "boundary_material": "UNKNOWN",  # 围护结构类型 (WALL / RAILING)
            "floor_span_est": 1.0,  # 预估贯通层数
        }

        # 1. 估算层数 (Step 1: Spans >= 2 floors?)
        # 假设层高 3.5m (可以根据项目调整)
        if geo_data["height"] > 0:
            context["floor_span_est"] = round(geo_data["height"] / 3.5, 1)

        # 2. 内容物检查 (Step 3: Internal Content Check)
        # 检查空间内部包含了什么 (IfcRelContainedInSpatialStructure)
        if element.is_a("IfcSpace"):
            for rel in element.ContainsElements:
                for related_elem in rel.RelatedElements:
                    etype = related_elem.is_a()
                    if etype in ["IfcStair", "IfcRamp"]:
                        context["contains_stair"] = True
                    elif etype in ["IfcTransportElement", "IfcElevator"]:
                        context["contains_transport"] = True
                    elif etype in [
                        "IfcFlowSegment",
                        "IfcFlowFitting",
                        "IfcDistributionControlElement",
                    ]:
                        context["contains_mep"] = True

        # 3. 边界检查 (Step 4: Edge Enclosure Check)
        # 检查由什么构件定义了空间的边界 (IfcRelSpaceBoundary)
        # 注意：这步计算量较大，简化逻辑只看前几个边界
        if element.is_a("IfcSpace"):
            walls_count = 0
            railings_count = 0
            # hasattr check because Openings don't have BoundedBy usually
            if hasattr(element, "BoundedBy"):
                for rel in element.BoundedBy:
                    if not rel.RelatedBuildingElement:
                        continue
                    b_elem = rel.RelatedBuildingElement
                    if b_elem.is_a("IfcWall") or b_elem.is_a("IfcWallStandardCase"):
                        walls_count += 1
                    elif b_elem.is_a("IfcRailing"):
                        railings_count += 1

            if walls_count > railings_count:
                context["boundary_material"] = "WALL"
            elif railings_count > 0 and railings_count >= walls_count:
                context["boundary_material"] = "RAILING"

        return context


# ===========================
# 模块 2: 智能分类代理 (含混合置信度校准)
# ===========================
class IntelligentClassifier:
    def __init__(self, api_key, model_name="gpt-4o", base_url=None):
        self.llm = ChatOpenAI(
            model=model_name, temperature=0, api_key=api_key, base_url=base_url
        )

        # 升级 Prompt：植入你的 5 步逻辑
        self.prompt = PromptTemplate.from_template(
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
        relevant_candidates = [
            c["name"]
            for c in all_categories
            if features["ifc_type"] in c["valid_ifc_types"]
        ]
        if not relevant_candidates:
            relevant_candidates = ["GENERIC_ELEMENT"]

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

        try:
            chain = self.prompt | self.llm
            response = chain.invoke(input_vars)
            content = response.content.strip().replace("```json", "").replace("```", "")
            result = json.loads(content)
        except Exception as e:
            return {"category": "ERROR", "confidence": 0.0, "reasoning": str(e)}

        result["confidence"] = self._calibrate_confidence(features, result)
        return result

    # def _calibrate_confidence(self, features, llm_result):
    #     # 保持你之前那个优秀的“积分制”逻辑
    #     # 这里特别增加针对 Shaft/Atrium 的加分项

    #     base_score = float(llm_result.get("confidence", 0.5))
    #     score = 0.5 + (base_score - 0.5) * 0.5

    #     cat = llm_result.get("category", "")
    #     spatial = features.get("spatial_context", {})

    #     # 强逻辑加分：如果有实物证据 (楼梯/电梯)，直接置信度拉满
    #     if cat in ["STAIRWELL", "ELEVATOR_SHAFT"]:
    #         if spatial.get("contains_stair") or spatial.get("contains_transport"):
    #             score += 0.3

    #     # 边界加分：如果判定为 Atrium 且探测到了 Railing
    #     if cat == "ATRIUM" and spatial.get("boundary_material") == "RAILING":
    #         score += 0.25

    #     # ... (保留原有的 PredefinedType 加分逻辑) ...
    #     if features.get("predefined_type") not in ["NOTDEFINED", "None", "USERDEFINED"]:
    #         score += 0.15

    #     return round(max(0.1, min(score, 0.99)), 2)
    def _calibrate_confidence(self, features, llm_result):
        """
        [优化版 V4] 混合置信度计算：修复“排除法”导致的虚高置信度
        核心目标：解决 "Opening" 被高分归类为 "GENERAL_ROOM" 的问题
        """
        # 1. 初始分：获取 LLM 原始分
        score = float(llm_result.get("confidence", 0.5))

        # 提取特征
        cat_name = llm_result.get("category", "").upper()
        elem_name = features.get("name", "").lower()
        dims = features.get("dimensions", {})
        area = dims.get("area", 0.0)
        spatial = features.get("spatial_context", {})

        # 定义“兜底分类” (Fallback Categories)
        # 这些分类通常是 LLM 在没找到更好匹配时选的
        is_fallback_cat = cat_name in ["GENERAL_ROOM", "GENERIC_ELEMENT", "UNKNOWN"]

        # =========================================
        # [Step 1] 兜底分类的严格审查 (The "Not A Room" Check)
        # =========================================
        if is_fallback_cat:
            # 1.1 语义冲突：名字叫“洞”，却分类为“房”
            void_keywords = ["opening", "void", "hole", "penetration", "recess"]
            if any(kw in elem_name for kw in void_keywords):
                print(
                    f"    [冲突] {features['guid']} 名字含Opening/Void，却被归类为 {cat_name}，重罚。"
                )
                score -= 0.45  # 直接扣到不及格

            # 1.2 尺寸冲突：面积太小，不可能是普通房间
            # 1.0m2 的东西叫 GENERAL_ROOM 是不合理的
            elif area < 2.5:
                print(
                    f"    [疑点] {features['guid']} 面积({area}m2)过小，不像 {cat_name}，扣分。"
                )
                score -= 0.3

            # 1.3 强制封顶：排除法得出的结论，置信度不能太高
            # 除非名字里真的有 "Room" (比如 "Storage Room")
            if "room" not in elem_name and "office" not in elem_name:
                if score > 0.7:
                    score = 0.7  # 强制压低上限

        # =========================================
        # [Step 2] 常规冲突检测 (Shaft/Atrium)
        # =========================================
        # Atrium 必须大
        if cat_name == "ATRIUM" and area < 15.0:
            score -= 0.5

        # Shaft 必须有围护或内容，否则如果是纯几何推断，降分
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

        # =========================================
        # [Step 3] 加分项 (Bonuses)
        # =========================================
        # 只有非冲突、非兜底的情况，或者兜底但名字匹配的情况，才加分

        # 3.1 名字与分类完美匹配 (e.g. Name="Mech Room", Cat="MECHANICAL_ROOM")
        # 移除下划线和空格进行模糊比对
        clean_cat = cat_name.replace("_", "").replace(" ", "")
        clean_name = elem_name.replace(" ", "").replace("-", "").upper()
        if clean_cat in clean_name:
            score += 0.15

        # 3.2 强证据匹配
        if cat_name == "STAIRWELL" and spatial.get("contains_stair"):
            score += 0.25
        elif cat_name == "ATRIUM" and spatial.get("boundary_material") == "RAILING":
            score += 0.2

        # =========================================
        # [Step 4] 最终边界
        # =========================================
        return round(max(0.2, min(score, 0.99)), 2)

    # def _calibrate_confidence(self, features, llm_result):
    #     """
    #     [优化版 V2] 混合置信度计算逻辑：引入冲突惩罚机制 (Conflict Penalties)
    #     """
    #     # 1. 初始分：直接使用 LLM 的原始置信度，不再人为抬高基准线
    #     # 如果 LLM 觉得只有 0.6，我们就从 0.6 开始算，而不是强行拉到 0.8
    #     score = float(llm_result.get("confidence", 0.5))

    #     # 提取关键信息
    #     cat_name = llm_result.get("category", "").upper()
    #     elem_name = features.get("name", "").lower()
    #     dims = features.get("dimensions", {})
    #     area = dims.get("area", 0.0)
    #     aspect = dims.get("aspect_ratio", 1.0)
    #     p_type = features.get("predefined_type", "")
    #     spatial = features.get("spatial_context", {})

    #     # ===========================
    #     # 2. 冲突惩罚 (Conflict Penalties) - 这是一个扣分系统
    #     # ===========================

    #     # [逻辑矛盾 A] 尺寸 vs 功能
    #     # Atrium 必须是大空间。如果是小洞口被归类为 Atrium，大概率错了
    #     if cat_name == "ATRIUM" and area < 20.0:
    #         score -= 0.45  # 重罚！变成 < 0.5

    #     # Functional Shaft 通常比较窄。如果很大的方正空间被归类为 Shaft，可能有误
    #     if cat_name == "FUNCTIONAL_SHAFT" and area > 50.0 and aspect < 1.5:
    #         score -= 0.2

    #     # [逻辑矛盾 B] 名字 vs 分类 (Semantic Mismatch)
    #     # 如果名字里明确写了 Stair/Elevator，却被分到了 General Room，说明 LLM 没对上号
    #     critical_keywords = {
    #         "stair": ["STAIR", "SHAFT", "CIRCULATION"],
    #         "elevator": ["ELEVATOR", "SHAFT", "TRANSPORT"],
    #         "lift": ["ELEVATOR", "SHAFT", "TRANSPORT"],
    #         "plumbing": ["MEP", "SHAFT", "PIPING"],
    #         "hvac": ["MEP", "SHAFT", "DUCT"],
    #     }
    #     for kw, valid_cats in critical_keywords.items():
    #         if kw in elem_name:
    #             # 检查当前分类是否在允许的列表里
    #             is_match = any(vc in cat_name for vc in valid_cats)
    #             if not is_match:
    #                 score -= 0.25  # 名字叫电梯却没分到电梯类，扣分

    #     # [逻辑矛盾 C] 模拟辅助构件降权
    #     # 你的数据集中有很多 Boundary_for_xxx，这些是辅助体，不是真实构件，置信度不应太高
    #     if "boundary_" in elem_name or "content_" in elem_name:
    #         score -= 0.15

    #     # ===========================
    #     # 3. 兜底分类封顶 (Cap for Fallbacks)
    #     # ===========================
    #     # 如果分类是“普通房间”或“通用构件”，说明没有识别出特殊性
    #     # 这时置信度不应该很高，除非面积真的很大且规整
    #     if cat_name in ["GENERAL_ROOM", "GENERIC_ELEMENT"]:
    #         # 基础封顶 0.8
    #         if score > 0.8:
    #             score = 0.8

    #         # 如果面积还很小 (比如 < 2m2 的洞口被归类为普通房间)，说明甚至可能是漏判的管井
    #         if area < 3.0:
    #             score -= 0.25  # 极小面积的 General Room 非常可疑

    #     # ===========================
    #     # 4. 加分项 (Bonuses) - 只有完全匹配才加分
    #     # ===========================

    #     # [加分 A] 强类型匹配
    #     # IFC 类型明确且匹配
    #     if p_type and p_type not in [
    #         "NOTDEFINED",
    #         "USERDEFINED",
    #         "None",
    #         "PROVISIONFORVOID",
    #     ]:
    #         score += 0.15

    #     # [加分 B] 空间线索完美匹配
    #     # 比如：分类是 STAIRWELL，且确实探测到了楼梯
    #     if cat_name == "STAIRWELL" and spatial.get("contains_stair"):
    #         score += 0.25
    #     elif cat_name == "ELEVATOR_SHAFT" and spatial.get("contains_transport"):
    #         score += 0.25
    #     elif cat_name == "ATRIUM" and spatial.get("boundary_material") == "RAILING":
    #         score += 0.25
    #     elif "SHAFT" in cat_name and spatial.get("boundary_material") == "WALL":
    #         score += 0.1

    #     # ===========================
    #     # 5. 边界归一化
    #     # ===========================
    #     final_score = max(0.2, min(score, 0.99))
    #     return round(final_score, 2)


# ===========================
# 模块 3: 主流程 (含 HITL 分流)
# ===========================
def main():
    ifc_path = "group_d_dataset.ifc"  # <--- 你的 IFC 文件路径
    if not os.path.exists(ifc_path):
        print(f"错误: 未找到 {ifc_path}")
        return

    # --- 配置 ---
    CONFIDENCE_THRESHOLD = 0.65  # 低于此值进入人工复核

    # 你的 API Key
    api_key = "sk-Dppz8ZLK62lWyZc7G3w3LtkJx7sNgmjdF65kzR7hKtQlnQDL"
    base_url = "https://yunwu.ai/v1"

    # 定义全量分类 (根据你的 Excel 图片补充)
    full_categories = [
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
        {"name": "GENERAL_ROOM", "valid_ifc_types": ["IfcSpace"]},
        {"name": "CORRIDOR", "valid_ifc_types": ["IfcSpace"]},
        {"name": "INDOOR_PARKING", "valid_ifc_types": ["IfcSpace"]},
        {"name": "ATRIUM", "valid_ifc_types": ["IfcSpace", "IfcOpeningElement"]},
        {
            "name": "ELEVATOR_SHAFT",
            "valid_ifc_types": ["IfcSpace", "IfcOpeningElement"],
        },
        {"name": "STAIRWELL", "valid_ifc_types": ["IfcSpace", "IfcOpeningElement"]},
        {"name": "MEP_SHAFT", "valid_ifc_types": ["IfcSpace", "IfcOpeningElement"]},
        {
            "name": "FUNCTIONAL_SHAFT",
            "valid_ifc_types": ["IfcSpace", "IfcOpeningElement"],
        },  # 通用管井
    ]

    # --- 初始化 ---
    analyzer = GeometryAnalyzer(ifc_path)
    classifier = IntelligentClassifier(api_key, base_url=base_url)

    # --- 开始处理 ---
    elements = analyzer.file.by_type("IfcElement") + analyzer.file.by_type("IfcSpace")
    # 为了演示，只取前 20 个。生产环境请去掉 [:20]
    elements_to_process = elements

    print(
        f"开始分析 {len(elements_to_process)} 个构件 (HITL 阈值: {CONFIDENCE_THRESHOLD})...\n"
    )

    results = []
    hitl_queue = []

    for idx, elem in enumerate(elements_to_process):
        # 1. 提取
        features = analyzer.get_element_features(elem)

        # 2. 分类 + 校准
        cls_result = classifier.classify(features, full_categories)

        conf = cls_result["confidence"]
        cat = cls_result["category"]

        # 3. 判定状态
        status = "AUTO_MAPPED"
        log_prefix = "[✓]"

        if conf < CONFIDENCE_THRESHOLD:
            status = "NEEDS_REVIEW"
            log_prefix = "[!]"
            hitl_queue.append(
                {
                    "guid": features["guid"],
                    "name": features["name"],
                    "ai_guess": cat,
                    "confidence": conf,
                    "reason": cls_result["reasoning"],
                }
            )

        print(
            f"{idx+1}. {log_prefix} {features['guid']} -> {cat:<20} (Conf: {conf}) | {status}"
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

    # --- 输出总结 ---
    print("\n" + "=" * 40)
    print(
        f"分析结束。自动通过: {len(results)-len(hitl_queue)} | 需人工复核: {len(hitl_queue)}"
    )
    print("=" * 40)

    if hitl_queue:
        print("\n--- 待复核清单 (HITL Queue) ---")
        for item in hitl_queue:
            print(f"GUID: {item['guid']}")
            print(f"  Guess: {item['ai_guess']} (Conf: {item['confidence']})")
            print(f"  Why: {item['reason']}")
            print("-" * 20)

    # 保存
    with open("final_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
