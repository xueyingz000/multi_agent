"""语义对齐代理包

这是一个用于处理IFC文件与建筑法规术语语义对齐问题的Python包。
主要解决IFC文件存储形式与法规描述术语不匹配的问题。

主要功能:
- A类功能语义对齐: 设备设施vs结构构件区分，空间功能分类
- B类几何规范对齐: 多层开口识别，垂直空间分类
- 面积计算结果生成和置信度评估

使用示例:
    from semantic_alignment_agent import SemanticAlignmentPipeline
    
    pipeline = SemanticAlignmentPipeline()
    result = pipeline.process(
        ifc_file_path="building.ifc",
        regulation_json_path="regulation_rules.json"
    )
"""

from .main import SemanticAlignmentPipeline
from .core import (
    SemanticAlignmentAgent,
    AlignmentType,
    ConfidenceLevel,
    AlignmentContext,
    AlignmentResult,
    FunctionInferenceEngine,
    VerticalSpaceDetector
)
from .utils import (
    RegulationCategory,
    IFCElementType,
    FunctionType,
    VerticalSpaceType,
    config,
    logger
)
from .data_processing import (
    IfcExtractor,
    RegulationParser
)
from .geometry import (
    GeometryAnalyzer,
    SpatialContext
)

__version__ = "1.0.0"
__author__ = "Semantic Alignment Agent Team"
__description__ = "IFC-Regulation Semantic Alignment Agent for Building Area Calculation"

__all__ = [
    # 主要类
    'SemanticAlignmentPipeline',
    'SemanticAlignmentAgent',
    
    # 核心组件
    'FunctionInferenceEngine',
    'VerticalSpaceDetector',
    'GeometryAnalyzer',
    'IfcExtractor',
    'RegulationParser',
    
    # 数据类型
    'AlignmentType',
    'ConfidenceLevel',
    'AlignmentContext',
    'AlignmentResult',
    'SpatialContext',
    
    # 枚举类型
    'RegulationCategory',
    'IFCElementType',
    'FunctionType',
    'VerticalSpaceType',
    
    # 工具
    'config',
    'logger'
]