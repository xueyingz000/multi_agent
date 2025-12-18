# 此文件用于导出 Agent 类，使外部调用更简洁

# 1. 导入 Agent 1 (已完成)
from .regulation_agent import RegulationAnalysisAgent

# 2. 导入 Agent 2 (已完成)
from .semantic_agent import IfcSemanticAlignmentAgent

# 3. 导入 Agent 3 (待开发)
# 当你写好 calculation_agent.py 后，取消下面这行的注释:
# from .calculation_agent import AreaCalculationAgent

# 定义对外暴露的列表
__all__ = [
    "RegulationAnalysisAgent",
    "IfcSemanticAlignmentAgent",
    # "AreaCalculationAgent",
]
