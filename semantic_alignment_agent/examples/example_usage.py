#!/usr/bin/env python3
"""
语义对齐代理使用示例

展示如何使用语义对齐代理处理IFC文件和法规数据
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import SemanticAlignmentPipeline
from utils import info, warning, error


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("基本使用示例")
    print("=" * 60)
    
    # 示例文件路径（需要根据实际情况调整）
    ifc_file = "path/to/your/building.ifc"
    regulation_json = "path/to/regulation_analysis_output.json"
    output_file = "semantic_alignment_result.json"
    
    try:
        # 创建流水线实例
        pipeline = SemanticAlignmentPipeline()
        
        # 处理文件
        result = pipeline.process(
            ifc_file_path=ifc_file,
            regulation_json_path=regulation_json,
            output_path=output_file
        )
        
        # 输出结果摘要
        print("\n处理结果摘要:")
        summary = result["alignment_report"]["summary"]
        print(f"  总元素数量: {summary['total_elements']}")
        print(f"  A类问题处理: {summary['category_a_count']}")
        print(f"  B类问题处理: {summary['category_b_count']}")
        print(f"  高置信度结果: {summary['high_confidence_count']}")
        print(f"  需要审查: {summary['requires_review_count']}")
        
        classification_summary = result["classification_summary"]
        print(f"\n分类结果:")
        print(f"  功能分类数量: {len(classification_summary['function_classifications'])}")
        print(f"  开口分类数量: {len(classification_summary['opening_classifications'])}")
        print(f"  A类问题处理: {classification_summary['category_breakdown']['A']}")
        print(f"  B类问题处理: {classification_summary['category_breakdown']['B']}")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保IFC文件和法规JSON文件路径正确")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


def example_with_mock_data():
    """使用模拟数据的示例"""
    print("\n" + "=" * 60)
    print("模拟数据示例")
    print("=" * 60)
    
    # 创建模拟的法规数据
    mock_regulation_data = {
        "metadata": {
            "source": "模拟法规数据",
            "version": "1.0",
            "country": "中国"
        },
        "rules": [
            {
                "id": "rule_001",
                "category": "slab_classification",
                "description": "楼板分类规则",
                "conditions": {
                    "thickness_threshold": 0.15,
                    "equipment_platform_coefficient": 0.5,
                    "structural_slab_coefficient": 1.0
                }
            },
            {
                "id": "rule_002",
                "category": "vertical_space",
                "description": "垂直空间处理规则",
                "conditions": {
                    "atrium_calculation": "exclude",
                    "shaft_calculation": "include",
                    "staircase_calculation": "include"
                }
            }
        ],
        "thresholds": {
            "story_height_limit": 2.2,
            "equipment_room_coefficient": 0.5,
            "auxiliary_construction_coefficient": 0.0
        }
    }
    
    # 保存模拟数据到临时文件
    import json
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(mock_regulation_data, f, ensure_ascii=False, indent=2)
        temp_regulation_file = f.name
    
    print(f"创建了模拟法规文件: {temp_regulation_file}")
    print("\n模拟法规数据内容:")
    print(json.dumps(mock_regulation_data, ensure_ascii=False, indent=2))
    
    # 清理临时文件
    try:
        os.unlink(temp_regulation_file)
        print(f"\n已清理临时文件: {temp_regulation_file}")
    except:
        pass


def example_analyze_results():
    """结果分析示例"""
    print("\n" + "=" * 60)
    print("结果分析示例")
    print("=" * 60)
    
    # 模拟一个处理结果
    mock_result = {
        "alignment_report": {
            "summary": {
                "total_elements": 25,
                "category_a_count": 15,
                "category_b_count": 10,
                "high_confidence_count": 20,
                "requires_review_count": 5
            },
            "confidence_distribution": {
                "high": 20,
                "medium": 3,
                "low": 2
            },
            "regulation_categories": {
                "include_full": 18,
                "include_partial": 4,
                "exclude": 2,
                "deduct_per_floor": 1
            }
        },
        "classification_summary": {
            "function_classifications": [
                {"element_guid": "elem_001", "classification": "structural_floor", "confidence": 0.95},
                {"element_guid": "elem_002", "classification": "equipment_platform", "confidence": 0.88}
            ],
            "opening_classifications": [
                {"element_guid": "space_001", "classification": "multi_story_atrium", "confidence": 0.92},
                {"element_guid": "space_002", "classification": "staircase_opening", "confidence": 0.85}
            ],
            "category_breakdown": {"A": 15, "B": 10}
        }
    }
    
    def analyze_confidence_distribution(result):
        """分析置信度分布"""
        dist = result["alignment_report"]["confidence_distribution"]
        total = sum(dist.values())
        
        print("置信度分布分析:")
        for level, count in dist.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {level.upper()}: {count} ({percentage:.1f}%)")
        
        if dist["low"] > 0:
            print(f"  ⚠️  有 {dist['low']} 个低置信度结果需要关注")
    
    def analyze_classification_accuracy(result):
        """分析分类准确性"""
        classification = result["classification_summary"]
        
        print("\n分类准确性分析:")
        func_classifications = classification["function_classifications"]
        opening_classifications = classification["opening_classifications"]
        
        # 功能分类置信度分析
        if func_classifications:
            avg_func_confidence = sum(item["confidence"] for item in func_classifications) / len(func_classifications)
            print(f"  功能分类平均置信度: {avg_func_confidence:.2f}")
        
        # 开口分类置信度分析
        if opening_classifications:
            avg_opening_confidence = sum(item["confidence"] for item in opening_classifications) / len(opening_classifications)
            print(f"  开口分类平均置信度: {avg_opening_confidence:.2f}")
    
    def analyze_processing_quality(result):
        """分析处理质量"""
        summary = result["alignment_report"]["summary"]
        
        print("\n处理质量分析:")
        total = summary["total_elements"]
        high_conf = summary["high_confidence_count"]
        review_needed = summary["requires_review_count"]
        
        if total > 0:
            quality_score = ((high_conf / total) * 0.7 + 
                           ((total - review_needed) / total) * 0.3) * 100
            
            print(f"  质量评分: {quality_score:.1f}/100")
            print(f"  高置信度比例: {(high_conf/total)*100:.1f}%")
            print(f"  需要审查比例: {(review_needed/total)*100:.1f}%")
            
            if quality_score >= 80:
                print("  ✅ 处理质量优秀")
            elif quality_score >= 60:
                print("  ⚠️  处理质量良好，建议审查低置信度结果")
            else:
                print("  ❌ 处理质量需要改进，建议人工干预")
    
    # 执行分析
    analyze_confidence_distribution(mock_result)
    analyze_classification_accuracy(mock_result)
    analyze_processing_quality(mock_result)


def example_integration_workflow():
    """集成工作流示例"""
    print("\n" + "=" * 60)
    print("集成工作流示例")
    print("=" * 60)
    
    print("""
完整的三个Agent协作流程:

1. Regulation Analysis Agent
   输入: 法规PDF文件
   输出: 结构化法规规则JSON
   
2. Semantic Alignment Agent (当前Agent)
   输入: IFC文件 + 法规规则JSON
   输出: 语义对齐结果 + 面积计算
   
3. 下一个Agent (待开发)
   输入: 语义对齐结果
   输出: 最终面积计算报告

当前Agent的处理流程:
├── 数据加载
│   ├── 加载IFC文件
│   └── 加载法规JSON
├── 特征提取
│   ├── IFC元素提取
│   ├── 几何特征分析
│   └── 功能推断
├── 问题处理
│   ├── A类: 功能语义对齐
│   │   ├── A1: 设备vs结构区分
│   │   └── A2: 空间功能分类
│   └── B类: 几何规范对齐
│       ├── B1: 多层开口识别
│       └── B2: 垂直空间分类
└── 结果生成
    ├── 对齐决策
    ├── 置信度评估
    └── 面积计算
""")


def main():
    """主函数"""
    print("语义对齐代理使用示例")
    
    # 运行各个示例
    example_basic_usage()
    example_with_mock_data()
    example_analyze_results()
    example_integration_workflow()
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)
    print("""
使用说明:
1. 确保已安装所有依赖: pip install -r requirements.txt
2. 准备IFC文件和法规JSON文件
3. 运行主程序: python main.py <ifc_file> <regulation_json>
4. 查看输出结果文件

注意事项:
- IFC文件需要包含楼板、空间等面积计算相关元素
- 法规JSON需要符合regulation analysis agent的输出格式
- 处理大型IFC文件时可能需要较长时间
- 低置信度结果建议人工审查
""")


if __name__ == "__main__":
    main()