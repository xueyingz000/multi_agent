import os
import shutil
import ifcopenshell
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict

# --- 导入 Agent ---
from agents.regulation_agent import RegulationAnalysisAgent
from agents.semantic_agent import IfcSemanticAlignmentAgent
from agents.area_calculation_agent import AreaCalculationAgent

app = FastAPI()

# 配置 CORS (允许前端 React 访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 全局状态存储 (模拟数据库) ---
# 在生产环境中，这些应该存入 Redis 或 SQL 数据库
session_state = {
    "current_ifc_path": None,  # 当前上传的 IFC 文件路径
    "current_model": None,  # 缓存已打开的 ifcopenshell 模型对象
    "current_rules": None,  # Agent 1 的输出结果 (JSON)
    "current_rule_name": None,  # 当前选用的法规名称
    "semantic_results": None,  # 缓存 Agent 2 的分析结果
    "stop_analysis": False,  # 控制分析中止的信号
}

# 初始化 Agents
# 注意：确保环境变量中有 OPENAI_API_KEY
reg_agent = RegulationAnalysisAgent()
semantic_agent = IfcSemanticAlignmentAgent()
calc_agent = AreaCalculationAgent()


# ============================================================
# 1. Import: 上传 IFC 文件
# ============================================================
@app.post("/upload/ifc")
async def upload_ifc(file: UploadFile = File(...)):
    try:
        # 1. 确保临时目录存在
        os.makedirs("temp", exist_ok=True)
        file_path = f"temp/{file.filename}"

        # 2. 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. 更新状态
        session_state["current_ifc_path"] = file_path
        # 预加载模型，供后续 Agent 使用
        session_state["current_model"] = ifcopenshell.open(file_path)
        # 清空之前的分析结果
        session_state["semantic_results"] = None

        return {
            "status": "success",
            "message": f"IFC File '{file.filename}' uploaded successfully.",
            "step": "Import Complete",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 2. Select Rule: 触发 Agent 1 (法规分析)
# ============================================================
@app.post("/upload/regulation")
async def analyze_regulation(
    file: UploadFile = File(...), region_name: str = Form(...)
):
    """
    用户上传 PDF 法规 -> 触发 Agent 1 -> 返回结构化 JSON
    """
    try:
        print(f"🚀 Triggering Agent 1 for region: {region_name}")

        # 1. 保存 PDF
        os.makedirs("temp", exist_ok=True)
        pdf_path = f"temp/{file.filename}"
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 【核心】调用 Agent 1 进行分析
        # 这里会消耗 Token 调用 LLM
        rules_json = reg_agent.analyze(pdf_path, region_name)

        if not rules_json:
            raise HTTPException(status_code=500, detail="Agent 1 failed to analyze PDF")

        # 3. 存储规则供后续 Agent 使用
        session_state["current_rules"] = rules_json
        session_state["current_rule_name"] = region_name

        return {
            "status": "success",
            "message": "Regulation analyzed by Agent 1",
            "data": rules_json,  # 将结果返回给前端展示，让用户看到提取了什么规则
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 触发 Agent 2 全量分析 (Batch Analysis)
# ============================================================
@app.post("/analyze/semantic")
def run_semantic_alignment():
    """
    Trigger Agent 2: Perform semantic alignment on the entire model
    """
    if not session_state["current_model"]:
        raise HTTPException(status_code=400, detail="No IFC model loaded")

    # 重置停止信号
    session_state["stop_analysis"] = False

    # 获取 Agent 1 的结果 (如果有的话，没有就用默认)
    rules = session_state.get("current_rules")

    def check_stop():
        return session_state.get("stop_analysis", False)

    try:
        # 调用 Agent 2
        results = semantic_agent.align(
            session_state["current_model"], rules, stop_callback=check_stop
        )
        session_state["semantic_results"] = results

        # Enrich queue with Express IDs for frontend highlighting
        if "hitl_queue" in results:
            for item in results["hitl_queue"]:
                item["express_id"] = get_express_id(item["guid"])

        return {
            "status": "success",
            "message": (
                "Semantic alignment complete"
                if not session_state["stop_analysis"]
                else "Semantic alignment stopped"
            ),
            "meta": results["meta"],
            "data": {"hitl_queue": results["hitl_queue"]},
        }
    except Exception as e:
        print(f"Agent 2 Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/stop")
async def stop_semantic_alignment():
    """
    Stop the running Agent 2 analysis
    """
    print("🛑 Received stop signal")
    session_state["stop_analysis"] = True
    return {"status": "success", "message": "Stop signal sent"}


def get_express_id(guid):
    """Helper to get Express ID from GUID using the loaded model."""
    model = session_state.get("current_model")
    if not model:
        return None
    try:
        entity = model.by_guid(guid)
        return entity.id()
    except:
        return None


class CalculationRequest(BaseModel):
    element_guid: str


# ============================================================
# 3. Process & Review: 触发 Agent 3 (语义对齐与计算)
# ============================================================


@app.post("/analyze/element")
async def analyze_element_logic(req: CalculationRequest):
    """
    前端点击某个构件 -> 返回 Agent 2 的缓存结果 (未来加上 Agent 3 计算)
    User Journey: Debug / Review 阶段
    """
    if not session_state["current_ifc_path"]:
        raise HTTPException(status_code=400, detail="Please upload IFC first")
    if not session_state["current_rules"]:
        raise HTTPException(
            status_code=400, detail="Please select/upload regulation first"
        )

    # 获取上下文
    guid = req.element_guid
    rules = session_state["current_rules"]
    ifc_path = session_state["current_ifc_path"]

    # 1. 检查 Agent 2 是否已运行
    semantic_data = session_state.get("semantic_results")

    if semantic_data and guid in semantic_data["alignment_results"]:
        # 命中缓存：直接返回 Agent 2 的分析结果
        agent2_res = semantic_data["alignment_results"][guid]

        # [Future Agent 3 Placeholder] 计算逻辑
        # calc_res = calc_agent.calculate(agent2_res, session_state["current_rules"])

        # 临时模拟 Agent 3 计算
        is_balcony = agent2_res["semantic_category"] == "BALCONY"
        factor = 0.5 if is_balcony else 1.0

        return {
            "element_id": guid,
            "express_id": get_express_id(guid),
            "type": agent2_res["ifc_type"],
            "calc_factor": factor,
            "reason": f"Agent 2 Identified as {agent2_res['semantic_category']} ({agent2_res['confidence']}).\nReasoning: {agent2_res['reasoning']}",
            "is_dirty": agent2_res["status"] == "NEEDS_REVIEW",
        }

    else:
        # 如果 Agent 2 还没跑，或者没找到该构件
        return {
            "element_id": guid,
            "express_id": get_express_id(guid),
            "reason": "Analysis not run yet. Please click 'Run Analysis' first.",
            "calc_factor": 0.0,
        }


# ============================================================
# 4. Calculate: 触发 Agent 3 (面积计算)
# ============================================================
@app.post("/calculate/area")
def calculate_area():
    """
    Trigger Agent 3: Calculate area based on semantic alignment and rules.
    """
    if not session_state["current_model"]:
        raise HTTPException(status_code=400, detail="No IFC model loaded")

    rules = session_state.get("current_rules", {})

    # 准备 Agent 2 的对齐数据
    alignment_list = []
    agent2_res = session_state.get("semantic_results", {})
    if agent2_res and "alignment_results" in agent2_res:
        for guid, info in agent2_res["alignment_results"].items():
            alignment_list.append(
                {
                    "guid": guid,
                    "category": info.get("semantic_category"),
                    "dimensions": info.get("dimensions", {}),
                }
            )

    try:
        print("🚀 Triggering Agent 3 (Area Calculation)...")
        results = calc_agent.calculate(
            session_state["current_model"], rules, alignment_list
        )
        session_state["calculation_results"] = results

        return {
            "status": "success",
            "message": "Area calculation complete",
            "data": results,
        }
    except Exception as e:
        print(f"Agent 3 Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export/report")
def export_report():
    """
    Export calculation report to Excel.
    """
    if not session_state.get("calculation_results"):
        raise HTTPException(status_code=400, detail="No calculation results available")

    try:
        report_path = calc_agent.export_to_excel(session_state["calculation_results"])
        return FileResponse(
            path=report_path,
            filename="Area_Calculation_Report.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # 启动服务器 (开启 reload 模式，方便开发)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
