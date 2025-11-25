import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import numpy as np
import json
import os


# ===========================
# 模块 1: 几何分析器 (Tools)
# ===========================
class GeometryAnalyzer:
    def __init__(self, ifc_file_path):
        print(f"[系统] 正在加载模型: {ifc_file_path} ...")
        self.file = ifcopenshell.open(ifc_file_path)
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)

    def get_element_geometry_data(self, element):
        data = {
            "guid": element.GlobalId,
            "type": element.is_a(),
            "height": 0.0,
            "thickness": 0.0,
            "area": 0.0,
            "aspect_ratio": 0.0,
            "is_vertical_span": False,  # 简化逻辑，默认 False
        }
        try:
            # 尝试提取几何
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            verts = shape.geometry.verts
            xs, ys, zs = verts[0::3], verts[1::3], verts[2::3]

            dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)

            if element.is_a("IfcSlab"):
                data["thickness"] = dz
                data["area"] = dx * dy
            else:
                data["height"] = dz
                data["thickness"] = min(dx, dy) if dx > 0 and dy > 0 else 0.1

            # 简单的长宽比
            dims = sorted([dx, dy])
            if dims[0] > 0.01:  # 避免除以零
                data["aspect_ratio"] = dims[1] / dims[0]

        except Exception as e:
            # 某些构件可能没有几何表达
            pass
        return data

    def get_psets(self, element):
        return ifcopenshell.util.element.get_psets(element)


# ===========================
# 模块 2: 规则推理引擎 (Logic)
# ===========================
class RuleInferenceEngine:
    def __init__(self, regulations):
        self.regulations = regulations

    def evaluate(self, element_geo, element_psets, element_type):
        best_match = {"term": "UNKNOWN", "confidence": 0.0, "reasoning": "No match"}

        for rule in self.regulations:
            if element_type not in rule.get("valid_ifc_types", []):
                continue

            match_score = 1.0
            reasons = []

            # 模拟一些简单的规则检查
            if (
                "min_thickness" in rule
                and element_geo["thickness"] < rule["min_thickness"]
            ):
                match_score *= 0.5
                reasons.append(
                    f"Thickness {element_geo['thickness']:.2f} < {rule['min_thickness']}"
                )

            # 模拟属性检查
            elem_name = element_psets.get("Identity", {}).get("Name", "Unknown").lower()
            if rule.get("required_keywords"):
                if not any(k in elem_name for k in rule["required_keywords"]):
                    match_score *= 0.8  # 稍微降低

            # 简化版置信度计算
            base_confidence = 0.8  # 假设基准
            final_conf = base_confidence * match_score

            if final_conf > best_match["confidence"]:
                best_match = {
                    "term": rule["term_name"],
                    "confidence": round(final_conf, 2),
                    "reasoning": (
                        f"Rule matched. Penalties: {reasons}"
                        if reasons
                        else "Perfect rule match"
                    ),
                }

        return best_match


# ===========================
# 模块 3: LLM 代理 (Brain - 模拟版)
# ===========================
class MockLLMAgent:
    """
    这是一个模拟的 LLM。在实际生产中，你会用 LangChain + OpenAI 替换它。
    """

    def reasoning(self, element_data, rule_result, candidates):
        print(f"    >>> [LLM 介入] 正在思考 {element_data['guid']}...")

        # 模拟逻辑：检查属性里有没有特殊关键词
        psets_str = str(element_data["psets"]).lower()

        if "mech" in psets_str or "riser" in psets_str:
            return {
                "selected_term": "FUNCTIONAL_SHAFT",
                "confidence": 0.92,
                "reasoning": "LLM 分析属性集发现 'Mech'/'Riser' 关键词，判定为管井。",
            }
        elif "balcony" in psets_str:
            return {
                "selected_term": "BALCONY",
                "confidence": 0.88,
                "reasoning": "LLM 发现 'Balcony' 命名，判定为阳台。",
            }
        else:
            # 如果 LLM 也没发现什么，就稍微增加一点规则引擎的置信度
            return {
                "selected_term": rule_result["term"],
                "confidence": min(rule_result["confidence"] + 0.1, 0.9),
                "reasoning": "LLM 复核了几何数据，未发现异常，支持规则判定。",
            }


# ===========================
# 模块 4: 主控制器 (Orchestrator)
# ===========================
class SemanticAlignmentAgent:
    def __init__(self, ifc_path, regulation_json):
        self.geo_analyzer = GeometryAnalyzer(ifc_path)
        self.rule_engine = RuleInferenceEngine(regulation_json)
        self.llm_agent = MockLLMAgent()  # 使用模拟 LLM
        self.hitl_queue = []
        self.results = []

    def run(self):
        # 为了演示，我们只处理少量构件，避免刷屏
        ifc_file = self.geo_analyzer.file
        # 选取所有板和空间
        elements = ifc_file.by_type("IfcSlab")[:5] + ifc_file.by_type("IfcSpace")[:5]

        print(f"[流程] 开始分析 {len(elements)} 个构件...\n")

        for elem in elements:
            guid = elem.GlobalId

            # 1. 几何 & 属性分析
            geo_data = self.geo_analyzer.get_element_geometry_data(elem)
            psets = self.geo_analyzer.get_psets(elem)
            geo_data["psets"] = psets

            if geo_data["thickness"] == 0 and geo_data["height"] == 0:
                continue  # 跳过无效构件

            # 2. 规则引擎推断
            rule_res = self.rule_engine.evaluate(geo_data, psets, elem.is_a())

            final_term = rule_res["term"]
            final_conf = rule_res["confidence"]
            final_reason = rule_res["reasoning"]
            source = "RuleEngine"

            print(
                f"[{elem.is_a()}] {guid} -> 初步判定: {final_term} (置信度: {final_conf})"
            )

            # 3. 置信度检查 (阈值 0.6)
            if final_conf < 0.6:
                # 调用 LLM
                candidates = [r["term_name"] for r in self.rule_engine.regulations]
                llm_res = self.llm_agent.reasoning(geo_data, rule_res, candidates)

                # 合并结果
                if llm_res["confidence"] > final_conf:
                    final_term = llm_res["selected_term"]
                    final_conf = llm_res["confidence"]
                    final_reason = llm_res["reasoning"]
                    source = "LLM_Enhanced"
                    print(f"    >>> LLM 修正结果: {final_term} (置信度: {final_conf})")

            # 4. HITL 人工审核队列 (仍然低于 0.6)
            status = "ALIGNED"
            if final_conf < 0.6:
                status = "PENDING_HUMAN_REVIEW"
                self.hitl_queue.append(
                    {"guid": guid, "system_guess": final_term, "confidence": final_conf}
                )

            self.results.append(
                {
                    "guid": guid,
                    "term": final_term,
                    "confidence": final_conf,
                    "source": source,
                    "status": status,
                }
            )

        return self.results, self.hitl_queue


# ===========================
# 运行入口
# ===========================
if __name__ == "__main__":
    # 1. 检查文件是否存在
    ifc_path = "test1.ifc"
    if not os.path.exists(ifc_path):
        print(
            f"错误: 找不到文件 {ifc_path}。请将一个 IFC 文件放入该目录并重命名为 model.ifc"
        )
        exit()

    # 2. 定义来自 Agent 1 的规则 (模拟输入)
    mock_regulations = [
        {
            "term_name": "STRUCTURAL_FLOOR",  # 结构楼板
            "valid_ifc_types": ["IfcSlab"],
            "min_thickness": 0.15,  # 假设大于 150mm 才是结构板
        },
        {
            "term_name": "DECORATIVE_FINISH",  # 装饰面层
            "valid_ifc_types": ["IfcSlab"],
            "min_thickness": 0.0,  # 任何厚度
            # 规则引擎会优先匹配结构板，如果厚度不够，结构板分低，这个可能分高
        },
        {
            "term_name": "FUNCTIONAL_SHAFT",  # 管道井
            "valid_ifc_types": ["IfcSpace"],
            "required_keywords": ["shaft", "riser"],
        },
        {"term_name": "GENERAL_ROOM", "valid_ifc_types": ["IfcSpace"]},  # 普通房间
    ]

    # 3. 启动 Agent
    agent = SemanticAlignmentAgent(ifc_path, mock_regulations)
    results, hitl = agent.run()

    # 4. 输出结果报告
    print("\n" + "=" * 40)
    print("最终对齐报告 (Alignment Report)")
    print("=" * 40)
    for res in results:
        print(
            f"GUID: {res['guid']} | Term: {res['term']:<20} | Conf: {res['confidence']} | Source: {res['source']} | Status: {res['status']}"
        )

    print("\n" + "=" * 40)
    print(f"需要人工介入 (HITL Queue): {len(hitl)} 个")
    print("=" * 40)
    print(json.dumps(hitl, indent=2))
