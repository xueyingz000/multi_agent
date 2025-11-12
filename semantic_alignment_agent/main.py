#!/usr/bin/env python3
"""
语义对齐代理主程序

这是第二个agent，接收regulation analysis agent的输出和IFC文件作为输入，
解决IFC文件存储形式与法规术语不匹配的问题。

主要功能：
1. 处理A类功能语义对齐问题（设备设施vs结构构件，空间功能分类）
2. 处理B类几何开洞规范对齐问题（多层开口识别，垂直空间分类）
3. 生成语义对齐决策和分类结果
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import config, logger, info, warning, error
from data_processing import IfcExtractor, RegulationParser
from geometry import GeometryAnalyzer
from core import (
    SemanticAlignmentAgent,
    AlignmentContext,
    AlignmentResult,
    FunctionInferenceEngine,
    VerticalSpaceDetector
)


class SemanticAlignmentPipeline:
    """语义对齐处理流水线"""
    
    def __init__(self):
        self.logger = logger
        self.config = config
        
        # 初始化各个组件
        self.ifc_extractor = IfcExtractor()
        self.regulation_parser = RegulationParser()
        self.geometry_analyzer = GeometryAnalyzer()
        self.function_engine = FunctionInferenceEngine()
        self.vertical_detector = VerticalSpaceDetector()
        self.alignment_agent = SemanticAlignmentAgent()
        
        info("语义对齐流水线初始化完成")
    
    def process(
        self,
        ifc_file_path: str,
        regulation_json_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理IFC文件和法规数据，生成语义对齐结果
        
        Args:
            ifc_file_path: IFC文件路径
            regulation_json_path: 法规分析结果JSON文件路径
            output_path: 输出文件路径（可选）
            
        Returns:
            处理结果字典
        """
        try:
            info(f"开始处理IFC文件: {ifc_file_path}")
            info(f"法规数据文件: {regulation_json_path}")
            
            # 1. 加载和解析输入数据
            regulation_data = self._load_regulation_data(regulation_json_path)
            ifc_model = self._load_ifc_file(ifc_file_path)
            
            # 2. 提取IFC元素和几何特征
            ifc_elements = self.ifc_extractor.extract_elements(ifc_model)
            info(f"提取到 {len(ifc_elements)} 个IFC元素")
            
            # 3. 几何特征分析
            for element in ifc_elements:
                if element.geometric_features is None:
                    element.geometric_features = self.geometry_analyzer.analyze_element_geometry(
                        element, ifc_model
                    )
            
            # 4. 功能推断
            for element in ifc_elements:
                if element.function_inference is None:
                    element.function_inference = self.function_engine.infer_element_function(
                        element
                    )
            
            # 5. 垂直空间检测
            vertical_spaces = self.vertical_detector.detect_vertical_spaces(
                ifc_elements, ifc_model
            )
            info(f"检测到 {len(vertical_spaces)} 个垂直贯穿空间")
            
            # 6. 创建对齐上下文
            context = self._create_alignment_context(regulation_data, ifc_model)
            
            # 7. 执行语义对齐
            alignment_results = self.alignment_agent.align_elements(
                ifc_elements=ifc_elements,
                vertical_spaces=vertical_spaces,
                regulation_data=regulation_data,
                context=context
            )
            
            info(f"完成 {len(alignment_results)} 个元素的语义对齐")
            
            # 8. 生成对齐报告
            alignment_report = self.alignment_agent.generate_alignment_report(alignment_results)
            
            # 9. 整理分类结果
            classification_results = self._organize_classification_results(alignment_results, ifc_elements, vertical_spaces)
            
            # 10. 组装最终结果
            final_result = {
                "metadata": {
                    "ifc_file": ifc_file_path,
                    "regulation_file": regulation_json_path,
                    "processing_timestamp": self._get_timestamp(),
                    "agent_version": "1.0.0"
                },
                "input_summary": {
                    "total_ifc_elements": len(ifc_elements),
                    "total_vertical_spaces": len(vertical_spaces),
                    "regulation_rules_count": len(regulation_data.get('rules', []))
                },
                "alignment_report": alignment_report,
                "classification_summary": classification_results,
                "detailed_elements": [
                    self._serialize_element_result(element, result)
                    for element, result in zip(ifc_elements + vertical_spaces, alignment_results)
                ]
            }
            
            # 11. 保存结果
            if output_path:
                self._save_results(final_result, output_path)
                info(f"结果已保存到: {output_path}")
            
            info("语义对齐处理完成")
            return final_result
            
        except Exception as e:
            error(f"处理过程中发生错误: {str(e)}")
            raise
    
    def _load_regulation_data(self, regulation_json_path: str) -> Dict[str, Any]:
        """加载法规数据"""
        try:
            with open(regulation_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            info(f"成功加载法规数据: {len(data.get('rules', []))} 条规则")
            return data
        except Exception as e:
            error(f"加载法规数据失败: {str(e)}")
            raise
    
    def _load_ifc_file(self, ifc_file_path: str):
        """加载IFC文件"""
        try:
            model = self.ifc_extractor.load_ifc_file(ifc_file_path)
            info(f"成功加载IFC文件: {ifc_file_path}")
            return model
        except Exception as e:
            error(f"加载IFC文件失败: {str(e)}")
            raise
    
    def _create_alignment_context(self, regulation_data: Dict[str, Any], ifc_model) -> AlignmentContext:
        """创建对齐上下文"""
        # 从IFC模型中提取建筑信息
        building_info = self.ifc_extractor.get_building_info(ifc_model)
        
        # 估算楼层信息
        floor_height = 3.0  # 默认层高
        total_floors = 1
        building_type = "unknown"
        
        if building_info:
            floor_height = building_info.get('typical_floor_height', 3.0)
            total_floors = building_info.get('total_floors', 1)
            building_type = building_info.get('building_type', 'unknown')
        
        return AlignmentContext(
            regulation_rules=regulation_data,
            building_type=building_type,
            floor_height=floor_height,
            total_floors=total_floors,
            geometric_tolerance=0.2
        )
    
    def _organize_classification_results(self, alignment_results: List[AlignmentResult], ifc_elements, vertical_spaces) -> Dict[str, Any]:
        """整理分类结果"""
        function_classifications = []
        opening_classifications = []
        category_breakdown = {"A": [], "B": []}
        
        # 合并所有元素
        all_elements = ifc_elements + vertical_spaces
        
        for element, result in zip(all_elements, alignment_results):
            classification_info = {
                "element_guid": result.element_guid,
                "ifc_type": element.ifc_type.value if hasattr(element, 'ifc_type') else "unknown",
                "regulation_category": result.regulation_category.value,
                "confidence": result.confidence,
                "confidence_level": result.confidence_level.value,
                "reasoning_path": result.reasoning_path,
                "evidence": result.evidence
            }
            
            if result.alignment_type.name.startswith('CATEGORY_A'):
                function_classifications.append(classification_info)
                category_breakdown["A"].append(classification_info)
            elif result.alignment_type.name.startswith('CATEGORY_B'):
                opening_classifications.append(classification_info)
                category_breakdown["B"].append(classification_info)
        
        return {
            "function_classifications": function_classifications,
            "opening_classifications": opening_classifications,
            "category_breakdown": category_breakdown,
            "total_elements": len(alignment_results)
        }
    
    def _serialize_element_result(self, element, result: AlignmentResult) -> Dict[str, Any]:
        """序列化元素结果"""
        element_data = {
            "guid": result.element_guid,
            "ifc_type": element.ifc_type.value if hasattr(element, 'ifc_type') else "unknown",
            "alignment_result": {
                "alignment_type": result.alignment_type.value,
                "regulation_category": result.regulation_category.value,

                "confidence": result.confidence,
                "confidence_level": result.confidence_level.value,
                "requires_review": result.requires_review,
                "reasoning_path": result.reasoning_path,
                "evidence": result.evidence,
                "alternatives": result.alternatives
            }
        }
        
        # 添加几何特征
        if hasattr(element, 'geometric_features') and element.geometric_features:
            element_data["geometric_features"] = {
                "area": element.geometric_features.area,
                "thickness": element.geometric_features.thickness,
                "height": element.geometric_features.height,
                "elevation": element.geometric_features.elevation
            }
        
        # 添加功能推断
        if hasattr(element, 'function_inference') and element.function_inference:
            element_data["function_inference"] = {
                "primary_function": element.function_inference.primary_function,
                "confidence": element.function_inference.confidence,
                "evidence": element.function_inference.evidence
            }
        
        return element_data
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """保存结果到文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            error(f"保存结果失败: {str(e)}")
            raise
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="语义对齐代理 - 处理IFC文件与法规术语的语义对齐问题"
    )
    
    parser.add_argument(
        "ifc_file",
        help="输入的IFC文件路径"
    )
    
    parser.add_argument(
        "regulation_json",
        help="法规分析结果JSON文件路径"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="输出文件路径（默认：semantic_alignment_result.json）",
        default="semantic_alignment_result.json"
    )
    
    parser.add_argument(
        "--config",
        help="配置文件路径",
        default=None
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 检查输入文件
        if not os.path.exists(args.ifc_file):
            error(f"IFC文件不存在: {args.ifc_file}")
            sys.exit(1)
        
        if not os.path.exists(args.regulation_json):
            error(f"法规JSON文件不存在: {args.regulation_json}")
            sys.exit(1)
        
        # 创建输出目录
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 初始化并运行流水线
        pipeline = SemanticAlignmentPipeline()
        
        info("=" * 60)
        info("语义对齐代理开始处理")
        info(f"IFC文件: {args.ifc_file}")
        info(f"法规文件: {args.regulation_json}")
        info(f"输出文件: {args.output}")
        info("=" * 60)
        
        result = pipeline.process(
            ifc_file_path=args.ifc_file,
            regulation_json_path=args.regulation_json,
            output_path=args.output
        )
        
        # 输出摘要信息
        summary = result["alignment_report"]["summary"]
        classification_summary = result["classification_summary"]
        
        info("=" * 60)
        info("处理完成摘要:")
        info(f"  总元素数量: {summary['total_elements']}")
        info(f"  A类问题处理: {summary['category_a_count']}")
        info(f"  B类问题处理: {summary['category_b_count']}")
        info(f"  高置信度结果: {summary['high_confidence_count']}")
        info(f"  需要审查: {summary['requires_review_count']}")
        info(f"  功能分类数量: {len(classification_summary['function_classifications'])}")
        info(f"  开口分类数量: {len(classification_summary['opening_classifications'])}")
        info("=" * 60)
        
        if summary['requires_review_count'] > 0:
            warning(f"有 {summary['requires_review_count']} 个元素需要人工审查")
        
        info("语义对齐代理处理完成")
        
    except KeyboardInterrupt:
        warning("用户中断处理")
        sys.exit(1)
    except Exception as e:
        error(f"处理失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()