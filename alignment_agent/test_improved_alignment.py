#!/usr/bin/env python3
"""
测试改进后的语义对齐功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG
from utils.logger import get_logger

logger = get_logger(__name__)

def test_improved_alignment():
    """测试改进后的语义对齐功能"""
    
    print("=" * 60)
    print("测试改进后的语义对齐功能")
    print("=" * 60)
    
    # 初始化代理
    agent = IFCSemanticAgentNoKG()
    
    # 处理IFC和JSON数据
    ifc_file = "outlinewall.ifc"
    json_file = "output-1.json"
    
    print(f"\n1. 处理IFC文件: {ifc_file}")
    ifc_result = agent.ifc_processor.process_ifc_file(ifc_file)
    
    print(f"\n2. 处理JSON法规文件: {json_file}")
    json_result = agent.text_processor._process_json_regulatory_text(json_file)
    
    # 提取实体数据
    ifc_entities = ifc_result.get('entities', [])
    reg_entities = json_result.get('entities', [])
    
    print(f"\nIFC实体数量: {len(ifc_entities)}")
    print(f"法规实体数量: {len(reg_entities)}")
    
    # 设置较低的阈值以便测试
    agent.semantic_alignment.similarity_threshold = 0.1
    
    print(f"\n3. 执行语义对齐 (阈值: {agent.semantic_alignment.similarity_threshold})")
    alignment_results = agent.semantic_alignment.align_entities(ifc_entities, reg_entities)
    
    print(f"\n找到的对齐数量: {len(alignment_results)}")
    
    # 按IFC实体分组显示对齐结果
    print("\n4. 详细对齐结果:")
    ifc_entity_groups = {}
    for alignment in alignment_results:
        ifc_entity = alignment.ifc_entity
        if ifc_entity not in ifc_entity_groups:
            ifc_entity_groups[ifc_entity] = []
        ifc_entity_groups[ifc_entity].append(alignment)
    
    for ifc_entity, alignment_list in ifc_entity_groups.items():
        print(f"\n  IFC实体: {ifc_entity}")
        for i, alignment in enumerate(alignment_list[:3]):  # 显示前3个对齐
            print(f"    对齐 {i+1}:")
            print(f"      法规实体: {alignment.regulatory_entity}")
            print(f"      置信度: {alignment.confidence_score:.3f}")
            print(f"      语义相似度: {alignment.semantic_similarity:.3f}")
            print(f"      上下文相似度: {alignment.context_similarity:.3f}")
            print(f"      对齐证据: {alignment.alignment_evidence}")
            print(f"      映射方法: {alignment.attributes.get('mapping_method', 'unknown')}")
    
    # 测试特定的中文术语匹配
    print("\n5. 测试中文建筑术语匹配:")
    test_cases = [
        ("ExteriorWall", "建筑面积"),
        ("ExteriorWall", "外墙"),
        ("Wall", "墙体"),
        ("Slab", "楼板")
    ]
    
    for ifc_term, reg_term in test_cases:
        similarity = agent.semantic_alignment._calculate_semantic_similarity(ifc_term, reg_term)
        print(f"  {ifc_term} <-> {reg_term}: 相似度 = {similarity:.3f}")
    
    # 检查预定义映射
    print("\n6. 检查预定义映射:")
    mappings = agent.semantic_alignment.predefined_mappings
    for ifc_type, mapping_info in mappings.items():
        if 'IfcWall' in ifc_type or 'IfcBuildingElement' in ifc_type:
            print(f"  {ifc_type}:")
            chinese_terms = [term for term in mapping_info['regulatory_terms'] if any('\u4e00' <= char <= '\u9fff' for char in term)]
            if chinese_terms:
                print(f"    中文术语: {chinese_terms[:5]}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_improved_alignment()