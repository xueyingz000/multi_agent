import os
import json
import pdfplumber
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
import dotenv

# 加载环境变量 (确保你有 OPENAI_API_KEY)
dotenv.load_dotenv()

# ==========================================
# 1. 定义数据结构 (Schema Definition)
# 这是 Agent 1 和 Agent 2 之间的 "协议"
# ==========================================


class RuleCondition(BaseModel):
    """定义单条规则的逻辑"""

    category: Literal["height", "enclosure", "special_use"] = Field(
        ..., description="规则类别"
    )
    description: str = Field(..., description="规则的简短文字描述，如'层高小于2.2米'")
    condition_logic: str = Field(
        ..., description="用于后续代码匹配的逻辑伪代码，如 'h < 2.2'"
    )
    coefficient: float = Field(..., description="计算系数: 1.0, 0.5, or 0.0")
    citation: str = Field(..., description="引用法规原文条款，用于 UI 展示")


class RegulationOutput(BaseModel):
    """Agent 1 的最终输出结构"""

    region: str = Field(..., description="法规适用的地区/版本")
    height_requirements: List[RuleCondition] = Field(description="关于层高的规则集合")
    enclosure_requirements: List[RuleCondition] = Field(
        description="关于围护结构/阳台的规则集合"
    )
    special_space_requirements: List[RuleCondition] = Field(
        description="关于特殊用途空间的规则集合"
    )

    # CoT: 让模型输出它的思考过程
    reasoning_trace: str = Field(
        description="模型的思维链(CoT)摘要，解释它是如何提取这些规则的"
    )


# ==========================================
# 2. Regulation Analysis Agent 类
# ==========================================


class RegulationAnalysisAgent:
    def __init__(self, model_name="gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """工具函数：从 PDF 提取文本"""
        print(f"📄 [Agent 1] Reading PDF: {pdf_path}...")
        text_content = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() + "\n"
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return ""

        # 简单截断防止超出 token 限制 (实际生产中可以使用 RAG 技术分块检索)
        # 增加限制到 200,000 字符，以确保覆盖大多数完整法规
        return text_content[:200000]

    def analyze(self, pdf_path: str, region_name: str) -> dict:
        """
        执行 ReAct 流程：
        1. 获取输入 (PDF)
        2. 构建 Prompt (包含 CoT 指令)
        3. 调用 LLM
        4. 结构化输出
        """
        raw_text = self._extract_text_from_pdf(pdf_path)
        if not raw_text:
            return {"error": "No text extracted"}

        print(f"🧠 [Agent 1] Analyzing regulations for {region_name} ...")

        # --- System Prompt: 定义角色与思维方式 ---
        system_prompt = """
        You are an expert Architect and Compliance Analyst Agent. 
        Your goal is to extract 'Area Calculation Rules' from building regulation texts.
        
        You must verify the rules against three specific categories:
        1. **Story Height Requirements**: Look for threshold values (e.g., 2.2m, 3.6m) that change the calculation coefficient (1.0 vs 0.5).
        2. **Covering/Enclosure**: Look for keywords like 'Balcony' (阳台), 'Enclosed' (封闭), 'Unenclosed' (未封闭), 'Canopy' (雨棚). Determine if they calculate full (1.0) or half (0.5) area.
        3. **Special Use**: Look for 'Basement', 'Shared Area', 'Fire Refuge', 'Equipment Room', 'Parking', 'Auxiliary Room', 'Deformation Joint'.
        
        **CRITICAL INSTRUCTIONS:**
        - **BE EXHAUSTIVE**: Do not summarize or omit any rules. Extract EVERY single clause that mentions area calculation logic.
        - **DETAILS MATTER**: If a rule has multiple sub-conditions (e.g. "Bay window > 2.1m" AND "Bay window < 2.1m"), create SEPARATE entries for each.
        - **NO HALLUCINATION**: Only extract rules explicitly present in the text.
        - **KEYWORDS**: Pay special attention to: 阳台 (Balcony), 飘窗/凸窗 (Bay Window), 地下室 (Basement), 雨篷 (Canopy), 变形缝 (Deformation Joint), 结构层高 (Story Height).
        
        **Chain of Thought Process:**
        1. First, scan the text for keywords related to 'Area Calculation' (建筑面积计算).
        2. Quote the specific clause.
        3. Determine the logic: IF condition THEN coefficient.
        4. Finally, output the structured JSON.
        """

        # --- User Prompt: 传入数据 ---
        user_prompt = f"""
        Region/Standard Name: {region_name}
        
        Raw Regulation Text:
        {raw_text}
        
        Please output the result strictly matching the JSON Schema provided.
        """

        try:
            # 使用 OpenAI 的 Structured Outputs (JSON Mode) 确保格式稳定
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=RegulationOutput,
                temperature=0,  # Set to 0 to ensure deterministic output
            )

            result = response.choices[0].message.parsed

            print("✅ [Agent 1] Analysis Complete.")
            return result.model_dump()

        except Exception as e:
            print(f"❌ [Agent 1] LLM Error: {e}")
            return {}


# ==========================================
# 3. 模拟运行 (Mock Execution)
# ==========================================

if __name__ == "__main__":
    # 假设你有一个测试用的 PDF (你需要真的放一个 PDF 在这里才能运行，比如《建筑工程建筑面积计算规范 GB/T 50353-2013》)
    # 为了演示，我将模拟一个 PDF 的文本内容，绕过文件读取，直接测试 LLM 逻辑

    agent = RegulationAnalysisAgent()

    # 模拟 PDF 文本 (实际使用时调用 agent.analyze(pdf_path="..."))
    mock_pdf_text = """
    ...
    3.0.1 在主体结构内的阳台，应按其结构外围水平面积计算全面积；在主体结构外的阳台，应按其结构底板水平投影面积计算1/2面积。
    3.0.2 建筑物的建筑面积应按自然层外墙结构外围水平面积之和计算。结构层高在2.20m及以上的，应计算全面积；结构层高在2.20m以下的，应计算1/2面积。
    3.0.24 建筑物内的变形缝，应按其自然层合并在建筑物面积内计算。
    ...
    """

    # 由于没有真实 PDF，这里我手动 patch 一下 _extract_text_from_pdf 用于演示
    agent._extract_text_from_pdf = lambda x: mock_pdf_text

    # 运行分析
    json_result = agent.analyze("fake_path.pdf", "National Standard 2013")

    # 打印结果 (Pretty Print)
    print(json.dumps(json_result, indent=2, ensure_ascii=False))
