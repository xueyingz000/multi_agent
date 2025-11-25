from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import json
import os


# ===========================
# 模块 1: 几何与属性分析器 (增强版)
# ===========================
class GeometryAnalyzer:
    def __init__(self, ifc_file_path):
        print(f"[系统] 正在加载模型: {ifc_file_path} ...")
        self.file = ifcopenshell.open(ifc_file_path)
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)

    def get_element_features(self, element):
        """
        提取用于分类的关键特征，特别是 PredefinedType 和 IsExternal
        """
        psets = ifcopenshell.util.element.get_psets(element)

        # 1. 提取 IsExternal (通常在 Pset_WallCommon, Pset_SlabCommon 等)
        is_external = "UNKNOWN"
        for pset_name, props in psets.items():
            if "Common" in pset_name and "IsExternal" in props:
                val = props["IsExternal"]
                # 处理 IFC bool (True/False/1/0)
                is_external = str(val).upper()
                break

        # 2. 提取几何尺寸
        geo_data = self._get_geometry(element)

        # 3. 组装特征包
        features = {
            "guid": element.GlobalId,
            "ifc_type": element.is_a(),
            "predefined_type": ifcopenshell.util.element.get_predefined_type(
                element
            ),  # 关键：获取 FLOOR, ROOF, BASESLAB 等
            "name": element.Name if element.Name else "",
            "is_external": is_external,
            "dimensions": geo_data,
            "psets_raw": psets,  # 仅供 LLM 参考其他线索
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

            # 简单的体积估算
            if element.is_a("IfcSpace"):
                data["height"] = round(dz, 2)
                data["area"] = round(dx * dy, 2)  # 简化计算
            elif element.is_a("IfcSlab"):
                data["thickness"] = round(dz, 2)
            elif element.is_a("IfcWall"):
                data["thickness"] = round(min(dx, dy), 2)

        except:
            data["note"] = "No Geometry Representation"
        return data


# ===========================
# 模块 2: 真实的 LLM 代理 (核心逻辑)
# ===========================
class IntelligentClassifier:
    def __init__(self, api_key, model_name="gpt-4o", base_url=None):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,  # 设为0以保证分类稳定性
            api_key=api_key,
            base_url=base_url,
        )

        # 升级后的 Prompt：教会 LLM 阅读你的 Excel 逻辑
        self.prompt = PromptTemplate.from_template(
            """
            You are a BIM Data Governance Expert. Your task is to map a raw IFC element to a specific "Functional Category" based on its properties.
            
            [Element Data]
            - IFC Class: {ifc_type}
            - PredefinedType: {predefined_type} (Crucial!)
            - IsExternal: {is_external} (Crucial: TRUE=External, FALSE=Internal)
            - Name/Description: {name}
            - Dimensions: {dimensions}
            - Other Properties: {psets_snippet}
            
            [Allowed Categories (Candidates) for {ifc_type}]
            {valid_candidates}
            
            [Reasoning Logic Guidelines]
            1. **Slabs**: 
               - If PredefinedType=ROOF -> "ROOF_SLAB"
               - If PredefinedType=FLOOR and IsExternal=TRUE -> "BALCONY" or "EXTERIOR_FLOOR"
               - If PredefinedType=FLOOR and IsExternal=FALSE -> "INTERIOR_FLOOR"
               - If PredefinedType=BASESLAB -> "FOUNDATION_SLAB"
               - If Name contains "Landing" -> "STAIR_LANDING"
            2. **Walls**:
               - If Type=IfcCurtainWall -> "CURTAIN_WALL"
               - If IsExternal=TRUE -> "EXTERIOR_WALL"
               - If Name contains "Shear" -> "SHEAR_WALL"
               - If Name contains "Plumbing" -> "PLUMBING_WALL"
            3. **Spaces**:
               - Use the 'Name' and dimensions to infer function (e.g., "Mech" -> MECHANICAL_ROOM).
               - Small area + "Shaft" -> "SHAFT".
               - "GFA" -> "GROSS_FLOOR_AREA".
            
            [Output Requirement]
            Return a JSON object ONLY.
            {{
                "category": "ONE_OF_THE_CANDIDATES",
                "confidence": 0.0-1.0,
                "reasoning": "Explain why based on PredefinedType, External status, or Name."
            }}
            """
        )

    def classify(self, features, all_categories):
        # 1. 过滤候选集：只提供跟当前 IFC 类型相关的分类，减少 LLM 幻觉
        relevant_candidates = [
            c["name"]
            for c in all_categories
            if features["ifc_type"] in c["valid_ifc_types"]
        ]

        # 如果没有对应的规则，给一个通用候选项
        if not relevant_candidates:
            relevant_candidates = ["GENERIC_ELEMENT"]

        # 2. 准备 Prompt 输入
        # 截取 psets 字符串，只保留有意义的部分（去噪）
        psets_str = str(features["psets_raw"])
        if len(psets_str) > 1500:
            psets_str = psets_str[:1500] + "..."

        input_vars = {
            "ifc_type": features["ifc_type"],
            "predefined_type": features["predefined_type"],
            "is_external": features["is_external"],
            "name": features["name"],
            "dimensions": json.dumps(features["dimensions"]),
            "psets_snippet": psets_str,
            "valid_candidates": json.dumps(relevant_candidates, indent=2),
        }

        # 3. 调用 LLM
        try:
            print(
                f"    >>> [LLM] 分析 {features['ifc_type']} (Predefined: {features['predefined_type']}, Ext: {features['is_external']})..."
            )
            chain = self.prompt | self.llm
            response = chain.invoke(input_vars)

            content = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(content)
        except Exception as e:
            print(f"    !!! LLM Error: {e}")
            return {"category": "ERROR", "confidence": 0.0, "reasoning": str(e)}


# ===========================
# 模块 3: 主流程 (带人工审核分流)
# ===========================
def main():
    ifc_path = "group_a_dataset.ifc"
    if not os.path.exists(ifc_path):
        print(f"未找到 {ifc_path}")
        return

    # --- 配置阈值 ---
    # 置信度低于此值的，将被标记为“需人工复核”
    CONFIDENCE_THRESHOLD = 0.65

    # --- 定义全量分类标准 (同上，此处省略以节省篇幅，保持不变) ---
    full_categories = [
        # ... (把上一段代码里的 categories 复制到这里) ...
        {"name": "INTERIOR_FLOOR_SLAB", "valid_ifc_types": ["IfcSlab"]},
        {"name": "ROOF_SLAB", "valid_ifc_types": ["IfcSlab", "IfcRoof"]},
        {"name": "BALCONY", "valid_ifc_types": ["IfcSlab"]},
        {
            "name": "EXTERIOR_WALL",
            "valid_ifc_types": ["IfcWall", "IfcWallStandardCase"],
        },
        {
            "name": "INTERIOR_WALL",
            "valid_ifc_types": ["IfcWall", "IfcWallStandardCase"],
        },
        {"name": "MECHANICAL_ROOM", "valid_ifc_types": ["IfcSpace"]},
        {"name": "GENERAL_ROOM", "valid_ifc_types": ["IfcSpace"]},
        # ... 确保你的列表是完整的
    ]
    # 如果为了测试方便，防止报错，加一个默认兜底
    if not full_categories:
        print("请确保 full_categories 列表已定义！")
        return

    # --- 初始化 ---
    analyzer = GeometryAnalyzer(ifc_path)
    # 记得替换 API Key
    api_key = "sk-Dppz8ZLK62lWyZc7G3w3LtkJx7sNgmjdF65kzR7hKtQlnQDL"
    base_url = "https://yunwu.ai/v1"
    classifier = IntelligentClassifier(api_key, base_url=base_url)

    # --- 运行分析 ---
    elements = analyzer.file.by_type("IfcElement") + analyzer.file.by_type("IfcSpace")
    print(
        f"共发现 {len(elements)} 个构件，开始智能分类 (阈值: {CONFIDENCE_THRESHOLD})...\n"
    )

    results = []
    hitl_queue = []  # 人工复核队列

    # 遍历构件 (这里建议处理全部，或者取前20个测试)
    for index, elem in enumerate(elements):
        # 1. 提取特征
        features = analyzer.get_element_features(elem)

        # 2. LLM 推理 + 混合置信度校准
        cls_result = classifier.classify(features, full_categories)

        confidence = cls_result["confidence"]
        category = cls_result["category"]

        # 3. 判定状态 (核心逻辑)
        status = "AUTO_MAPPED"  # 默认：自动映射成功
        status_icon = "[✓]"

        if confidence < CONFIDENCE_THRESHOLD:
            status = "NEEDS_REVIEW"  # 标记为需人工介入
            status_icon = "[!]"
            hitl_queue.append(
                {
                    "guid": features["guid"],
                    "name": features["name"],
                    "ai_guess": category,
                    "confidence": confidence,
                    "reason": cls_result["reasoning"],
                }
            )

        # 4. 打印实时日志 (区分颜色或标记)
        print(
            f"{index+1}. {status_icon} {features['guid']} | {category:<20} | Conf: {confidence:.2f} | {status}"
        )

        # 5. 收集结果
        results.append(
            {
                "guid": features["guid"],
                "ifc_type": features["ifc_type"],
                "original_name": features["name"],
                "mapped_category": category,
                "confidence": confidence,
                "status": status,  # <--- 输出中增加了 Status 字段
                "reasoning": cls_result["reasoning"],
            }
        )

    # --- 输出报告 ---
    print("\n" + "=" * 50)
    print(f" 分析完成: 总计 {len(results)} 个")
    print(f" 自动通过: {len(results) - len(hitl_queue)} 个")
    print(f" 需人工介入: {len(hitl_queue)} 个")
    print("=" * 50)

    if hitl_queue:
        print("\n--- 等待人工复核列表 (HITL Queue) ---")
        for item in hitl_queue:
            print(f"GUID: {item['guid']}")
            print(f"  Name: {item['name']}")
            print(f"  AI Suggestion: {item['ai_guess']} (Conf: {item['confidence']})")
            print(f"  Reason: {item['reason']}")
            print("-" * 30)

    # 保存文件
    with open("classification_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存至 classification_results.json")


if __name__ == "__main__":
    main()
