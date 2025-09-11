# Semantic Alignment Agent

基于LLM的智能语义对齐代理，用于处理IFC建筑模型与建筑法规之间的语义匹配问题。

## 项目概述

本项目是三个大语言模型代理协作框架的第二个组件，负责解决IFC文件存储形式与法规描述术语不完全匹配的问题。集成了先进的LLM智能分析能力，能够处理复杂的边界情况和模糊语义。

### 🚀 新特性 (LLM智能升级)

- **智能几何分析**: 使用LLM增强几何特征识别和空间关系理解
- **智能功能推断**: LLM驱动的功能分类，处理模糊和边界情况
- **智能置信度评估**: 基于多维度证据的动态置信度计算
- **混合推理模式**: 结合传统规则和LLM智能分析的双重保障
- **上下文感知**: 充分利用建筑信息、空间上下文进行智能推理

### 主要功能

- **Category A: 功能语义冲突处理**
  - A1: 区分设备设施与结构构件（如IfcSlab的不同功能分类）
  - A2: 处理缺失或模糊的空间功能属性

- **Category B: 几何-法规对齐**
  - B1: 多层开口识别
  - B2: 垂直空间分类

## 项目结构

```
semantic_alignment_agent/
├── core/                    # 核心模块
│   ├── semantic_alignment_agent.py # 核心语义对齐代理 (LLM增强)
│   ├── function_inference.py       # 功能推断模块 (LLM增强)
│   └── vertical_space_detector.py  # 垂直空间检测
├── data_processing/         # 数据处理模块
│   ├── ifc_extractor.py        # IFC数据提取
│   └── regulation_parser.py    # 法规数据解析
├── geometry/               # 几何分析模块
│   └── geometry_analyzer.py    # 几何特征分析器 (LLM增强)
├── llm/                    # 🆕 LLM智能模块
│   ├── llm_client.py           # 统一LLM客户端接口
│   ├── prompt_templates.py     # 专业prompt模板库
│   └── element_classifier.py   # 基于LLM的元素分类器
├── utils/                  # 工具模块
│   ├── config_loader.py        # 配置加载
│   ├── logger.py               # 日志工具
│   └── data_structures.py      # 数据结构定义
├── examples/               # 示例代码
│   └── example_usage.py        # 基本使用示例
├── config.yaml             # 配置文件 (包含LLM设置)
├── requirements.txt        # 依赖包 (包含OpenAI)
├── main.py                 # 主程序入口
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```python
from semantic_alignment_agent import SemanticAlignmentAgent

# 初始化代理 (启用LLM增强功能)
agent = SemanticAlignmentAgent(
    config_path="config.yaml",
    enable_llm=True  # 启用LLM智能分析
)

# 输入数据
regulation_rules = {...}  # 来自regulation analysis agent的输出
ifc_file_path = "path/to/building.ifc"

# 执行语义对齐
result = agent.align(
    regulation_rules=regulation_rules,
    ifc_file_path=ifc_file_path
)

# 输出结果
print(result.area_calculation_results)
print(result.alignment_decisions)
print(result.confidence_scores)
```

### LLM增强功能使用

```python
# 使用LLM增强的几何分析
from geometry import GeometryAnalyzer

analyzer = GeometryAnalyzer(enable_llm=True)
enhanced_result = analyzer.analyze_element_geometry_enhanced(
    element_info, spatial_context, building_context
)

# 使用LLM增强的功能推断
from core import FunctionInferenceEngine

engine = FunctionInferenceEngine(enable_llm=True)
function_result = engine.infer_function_enhanced(
    element_info, geometric_features, spatial_context
)
```

### 输入格式

1. **法规规则** (来自Regulation Analysis Agent):
```json
{
  "per_region": {
    "CN": {
      "height_rules": [...],
      "cover_enclosure_rules": [...],
      "special_use_rules": [...]
    }
  }
}
```

2. **IFC文件**: 标准IFC格式的建筑模型文件

### 输出格式

```json
{
  "area_calculation_results": {
    "elements": [
      {
        "element_id": "guid",
        "ifc_type": "IfcSlab",
        "regulation_category": "include_partial",
        "coefficient": 0.5,
        "area": 100.0,
        "calculated_area": 50.0,
        "confidence": 0.85,
        "reasoning_path": "..."
      }
    ]
  },
  "alignment_decisions": {...},
  "confidence_assessment": {...}
}
```

## 核心算法

### Category A: 功能语义对齐

#### A1: 设备vs结构构件判断 (LLM增强)
- **传统规则**: 厚度指标 (<0.1m → 装饰平台; 0.1-0.15m → 设备平台; ≥0.15m → 结构楼板)
- **位置分析**: 屋顶+设备邻接 → 设备平台
- **LLM智能分析**: 综合几何特征、空间上下文、建筑信息进行智能判断
- **混合推理**: 结合传统规则和LLM分析，提供最可靠的分类结果

#### A2: 空间功能分类 (LLM增强)
- **智能功能推断**: LLM驱动的功能分类，处理模糊和边界情况
- **上下文感知**: 考虑空间关系、建筑类型、使用模式
- **动态置信度**: 基于多维度证据的智能置信度评估
- **法规映射**: 自动匹配最适合的法规分类

### Category B: 几何-法规对齐

#### B1: 垂直贯穿空间检测 (LLM增强)
- **智能扫描**: LLM辅助的多实体识别和关联分析
- **几何连续性验证**: 结合传统算法和LLM空间理解
- **智能整合**: LLM驱动的去重和空间关系优化

#### B2: 垂直空间分类 (LLM增强)
- **智能分类**: LLM基于空间特征和使用功能进行分类
- **规则应用**: 中庭 → 每层扣除开口面积; 竖井 → 不扣除; 楼梯间 → 不扣除
- **边界情况处理**: LLM处理复杂和模糊的垂直空间类型

## 环境变量

```bash
# OpenAI API配置 (LLM功能必需)
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
export OPENAI_MODEL="gpt-4o-mini"  # 可选
```

## LLM配置说明

### config.yaml中的LLM设置

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.2
  max_tokens: 4000
  timeout: 60
  
  # LLM增强功能开关
  enhancement:
    enable_geometry_analysis: true
    enable_function_inference: true
    enable_confidence_assessment: true
    enable_boundary_case_handling: true
    
    # 置信度阈值设置
    llm_enhancement_threshold: 0.6  # 低于此值时启用LLM
    min_traditional_confidence: 0.3
```

### 智能功能特性

- **自适应分析**: 当传统方法置信度低于0.6时自动启用LLM增强
- **混合推理**: 结合规则推理和LLM智能分析的最佳结果
- **回退机制**: LLM不可用时自动切换到传统分析方法
- **上下文感知**: 充分利用建筑信息和空间关系进行智能推理

## 许可证

MIT License