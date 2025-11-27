from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import json
import os


# ===========================
# 模块 1: 几何与属性分析器
# ===========================
class GeometryAnalyzer:
    def __init__(self, ifc_file_path):
        print(f"[系统] 正在加载模型: {ifc_file_path} ...")
        self.file = ifcopenshell.open(ifc_file_path)
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)

    def get_element_features(self, element):
        """提取用于分类的关键特征"""
        psets = ifcopenshell.util.element.get_psets(element)

        # 1. 提取 IsExternal
        is_external = "UNKNOWN"
        for pset_name, props in psets.items():
            if "Common" in pset_name and "IsExternal" in props:
                val = props["IsExternal"]
                is_external = str(val).upper()
                break

        # 2. 提取几何尺寸
        geo_data = self._get_geometry(element)

        # 3. 组装特征
        features = {
            "guid": element.GlobalId,
            "ifc_type": element.is_a(),
            "predefined_type": ifcopenshell.util.element.get_predefined_type(element),
            "name": element.Name if element.Name else "",
            "is_external": is_external,
            "dimensions": geo_data,
            "psets_raw": psets,
        }
        return features

    def _get_geometry(self, element):
        data = {"height": 0.0, "thickness": 0.0, "area": 0.0, "volume": 0.0}
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            verts = shape.geometry.verts
            xs, ys, zs = verts[0::3], verts[1::3], verts[2::3]
            dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)

            data["bbox_dims"] = [round(dx, 2), round(dy, 2), round(dz, 2)]
            if element.is_a("IfcSpace"):
                data["height"] = round(dz, 2)
                data["area"] = round(dx * dy, 2)
            elif element.is_a("IfcSlab"):
                data["thickness"] = round(dz, 2)
            elif element.is_a("IfcWall"):
                data["thickness"] = round(min(dx, dy), 2)
        except:
            data["note"] = "No Geometry Representation"
        return data


# ===========================
# 模块 2: 智能分类代理 (含混合置信度校准)
# ===========================
class IntelligentClassifier:
    def __init__(self, api_key, model_name="gpt-4o", base_url=None):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
            base_url=base_url,
        )

        self.prompt = PromptTemplate.from_template(
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

    def classify(self, features, all_categories):
        # 1. 筛选候选集
        relevant_candidates = [
            c["name"]
            for c in all_categories
            if features["ifc_type"] in c["valid_ifc_types"]
        ]
        if not relevant_candidates:
            relevant_candidates = ["GENERIC_ELEMENT"]

        # 2. 准备 Prompt
        psets_str = str(features["psets_raw"])
        if len(psets_str) > 1000:
            psets_str = psets_str[:1000] + "..."

        input_vars = {
            "ifc_type": features["ifc_type"],
            "predefined_type": features["predefined_type"],
            "is_external": features["is_external"],
            "name": features["name"],
            "dimensions": json.dumps(features["dimensions"]),
            "psets_snippet": psets_str,
            "valid_candidates": json.dumps(relevant_candidates),
        }

        # 3. LLM 推理
        try:
            chain = self.prompt | self.llm
            response = chain.invoke(input_vars)
            content = response.content.strip().replace("```json", "").replace("```", "")
            result = json.loads(content)
        except Exception as e:
            return {"category": "ERROR", "confidence": 0.0, "reasoning": str(e)}

        # 4. [关键步骤] 混合置信度校准
        final_confidence = self._calibrate_confidence(features, result)
        result["confidence"] = final_confidence

        return result

    def _calibrate_confidence(self, features, llm_result):
        """
        [优化版] 混合置信度计算逻辑：积分制
        使分数呈现 0.4 ~ 0.99 的渐进分布，体现数据质量的差异
        """
        # 1. 设定基准分 (Base Score)
        # 我们稍微抑制一下 LLM 的原始自信，把它压缩到 0.5 ~ 0.7 之间作为起跑线
        # 这样留出空间给后续的规则去加分
        raw_llm_score = float(llm_result.get("confidence", 0.5))
        score = 0.5 + (raw_llm_score - 0.5) * 0.5  # 映射到 0.5-0.75 区间

        # 获取关键数据
        p_type = features.get("predefined_type", "")
        is_ext = features.get("is_external", "UNKNOWN")
        cat_name = llm_result.get("category", "")
        elem_name = features.get("name", "").upper()
        dims = features.get("dimensions", {})

        # ===========================
        # 2. 加分项 (Bonuses)
        # ===========================

        # [A] 类型明确度加分 (权重最大)
        # 如果 IFC 类型非常明确 (如 ROOF, BEAM)，而不是 USERDEFINED，加分
        strong_types = ["ROOF", "BEAM", "COLUMN", "FLOOR", "BASESLAB", "STAIR", "RAMP"]
        if p_type and p_type not in ["NOTDEFINED", "USERDEFINED", "None"]:
            if p_type in strong_types:
                score += 0.20  # 强类型，大幅加分
            else:
                score += 0.10  # 普通类型 (如 LANDING)，小幅加分

        # [B] 属性完整度加分
        # 如果 IsExternal 被明确定义了 (TRUE/FALSE)，说明数据质量好
        if is_ext in ["TRUE", "FALSE"]:
            score += 0.08

        # [C] 语义一致性加分
        # 如果 LLM 选出的分类词，直接出现在了构件名字里
        # 例如 名字叫 "Mech Room"，分类是 "MECHANICAL_ROOM"
        # 简单的字符串包含检查
        clean_cat = cat_name.replace("_", "").replace(" ", "")
        if clean_cat in elem_name.replace(" ", "") or elem_name in cat_name:
            score += 0.12

        # [D] 几何数据存在加分
        if dims.get("area", 0) > 0.1 or dims.get("volume", 0) > 0.1:
            score += 0.05

        # ===========================
        # 3. 扣分项 (Penalties)
        # ===========================

        # [X] 泛型惩罚 (最重要)
        # 如果分类结果是 GENERIC_ELEMENT，说明没识别出来，上限不能太高
        if cat_name == "GENERIC_ELEMENT":
            score -= 0.15
            # 设个软上限，通用构件再怎么好，通常也不建议超过 0.75
            if score > 0.75:
                score = 0.75

        # [Y] 几何缺失惩罚
        if dims.get("area", 0) == 0 and dims.get("volume", 0) == 0:
            score -= 0.30  # 没几何信息，严重扣分

        # [Z] 命名模糊惩罚
        # 如果名字太短或没有名字
        if len(elem_name) < 3:
            score -= 0.05

        # ===========================
        # 4. 边界处理 (Clamping)
        # ===========================
        # 确保分数在 0.1 ~ 0.99 之间
        final_score = max(0.1, min(score, 0.99))

        # 保留两位小数
        return round(final_score, 2)

    # def _calibrate_confidence(self, features, llm_result):
    #     """
    #     混合置信度计算逻辑：
    #     Python 规则 (Hard Rules) + LLM 语义判断 (Soft Logic)
    #     """
    #     # 获取 LLM 原始打分 (如果没有则默认 0.5)，并限制在 [0, 1]
    #     score = float(llm_result.get("confidence", 0.5))
    #     score = max(0.0, min(1.0, score))

    #     # --- 规则 1: PredefinedType 权威性加成（渐进式靠拢高锚点）---
    #     # 强信号存在时，不直接跳到 0.95，而是以一定比例靠近 0.95
    #     p_type = features.get("predefined_type", "")
    #     if p_type and p_type not in ["NOTDEFINED", "USERDEFINED", "None"]:
    #         # 向 0.95 靠拢（60% 的幅度），避免一刀切 0.95
    #         score = score + 0.6 * (0.95 - score)

    #     # --- 规则 2: IsExternal 明确性加成（小幅加成）---
    #     if features.get("is_external") in ["TRUE", "FALSE"]:
    #         # 轻微向 1.0 靠拢，力度较温和，避免上限过快
    #         score = score + 0.08 * (1.0 - score)

    #     # --- 规则 3: 几何缺失惩罚（渐进式靠拢低锚点）---
    #     dims = features.get("dimensions", {})
    #     if dims.get("area", 0) == 0 and dims.get("volume", 0) == 0:
    #         # 向低锚点 0.35 靠拢（50% 的幅度），避免强制卡在 0.4
    #         score = score + 0.5 * (0.35 - score)

    #     # --- 规则 4: 关键词匹配加成（温和加成）---
    #     if llm_result.get("category", "").lower() in features.get("name", "").lower():
    #         # 温和地向 0.95 靠拢（5% 的幅度），让文本匹配产生细腻影响
    #         score = score + 0.05 * (0.95 - score)

    #     # 最终边界控制，避免过度偏向两端，并保留两位小数
    #     score = max(0.05, min(0.98, score))
    #     return round(score, 2)


# ===========================
# 模块 3: 主流程 (含 HITL 分流)
# ===========================
def main():
    ifc_path = "group_a_dataset.ifc"  # <--- 你的 IFC 文件路径
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
