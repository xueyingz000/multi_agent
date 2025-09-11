#!/usr/bin/env python3
"""Basic usage example for IFC Semantic Agent."""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from core import IFCSemanticAgent
from utils import get_logger


def main():
    """Demonstrate basic usage of IFC Semantic Agent."""
    logger = get_logger(__name__)
    
    # Initialize the agent
    config_path = Path(__file__).parent.parent / "config.yaml"
    agent = IFCSemanticAgent(str(config_path))
    
    logger.info("IFC Semantic Agent initialized successfully")
    
    # Example 1: Simple semantic alignment query
    print("\n" + "="*60)
    print("Example 1: Simple Semantic Alignment Query")
    print("="*60)
    
    query1 = "如何将IfcSlab映射到监管文档中的平台概念？"
    
    response1 = agent.process_query(
        query=query1,
        ifc_data=None,  # No specific IFC data provided
        regulatory_text=None  # No specific regulatory text provided
    )
    
    print(f"查询: {query1}")
    print(f"最终答案: {response1.final_answer}")
    print(f"置信度: {response1.confidence_score:.2f}")
    print(f"推理步骤数: {response1.total_steps}")
    print(f"执行时间: {response1.execution_time:.2f}秒")
    
    # Example 2: Query with mock IFC data
    print("\n" + "="*60)
    print("Example 2: Query with Mock IFC Data")
    print("="*60)
    
    # Mock IFC data structure
    mock_ifc_data = {
        "entities": {
            "wall_001": {
                "type": "IfcWall",
                "name": "ExteriorWall",
                "properties": {
                    "Width": 200,  # mm
                    "Height": 3000,  # mm
                    "Material": "Concrete"
                },
                "location": "Building_A_Floor_1"
            },
            "slab_001": {
                "type": "IfcSlab",
                "name": "FloorSlab",
                "properties": {
                    "Thickness": 150,  # mm
                    "Material": "ReinforcedConcrete",
                    "LoadBearing": True
                },
                "location": "Building_A_Floor_1"
            }
        },
        "relationships": [
            {
                "type": "IfcRelContainedInSpatialStructure",
                "relating_structure": "Building_A_Floor_1",
                "related_elements": ["wall_001", "slab_001"]
            }
        ]
    }
    
    query2 = "分析这些IFC实体与建筑监管要求的对应关系"
    
    response2 = agent.process_query(
        query=query2,
        ifc_data=mock_ifc_data,
        regulatory_text=None
    )
    
    print(f"查询: {query2}")
    print(f"最终答案: {response2.final_answer}")
    print(f"置信度: {response2.confidence_score:.2f}")
    print(f"推理步骤数: {response2.total_steps}")
    
    # Example 3: Query with regulatory text
    print("\n" + "="*60)
    print("Example 3: Query with Regulatory Text")
    print("="*60)
    
    regulatory_text = """
    建筑设计规范要求：
    1. 承重墙厚度不应小于180mm
    2. 楼板厚度应根据跨度确定，一般不小于120mm
    3. 外墙应具备保温、防水功能
    4. 结构构件应满足抗震设防要求
    5. 建筑材料应符合防火等级要求
    """
    
    query3 = "检查IFC模型中的墙体和楼板是否符合监管要求"
    
    response3 = agent.process_query(
        query=query3,
        ifc_data=mock_ifc_data,
        regulatory_text=regulatory_text
    )
    
    print(f"查询: {query3}")
    print(f"最终答案: {response3.final_answer}")
    print(f"置信度: {response3.confidence_score:.2f}")
    print(f"推理步骤数: {response3.total_steps}")
    
    # Display detailed reasoning trace for the last example
    print("\n" + "="*60)
    print("Detailed Reasoning Trace (Last Example)")
    print("="*60)
    
    for i, step in enumerate(response3.react_steps, 1):
        print(f"\n步骤 {i}: {step.thought.reasoning_type.upper()}")
        print(f"思考: {step.thought.content}")
        print(f"行动: {step.action.action_type.value} - {step.action.reasoning}")
        print(f"观察: {step.observation.observation_type} (置信度: {step.observation.confidence:.2f})")
        print(f"成功: {'是' if step.success else '否'}")
        if step.error_message:
            print(f"错误: {step.error_message}")
    
    # Display agent state
    print("\n" + "="*60)
    print("Agent State Information")
    print("="*60)
    
    agent_state = agent.get_agent_state()
    print(f"当前步骤: {agent_state['current_step']}")
    print(f"推理历史长度: {agent_state['react_history_length']}")
    print(f"整体置信度: {agent_state['overall_confidence']:.2f}")
    print(f"可用行动: {[action.value for action in agent_state['available_actions']]}")
    
    # Reset agent for next use
    agent.reset_agent()
    logger.info("Agent reset completed")
    
    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()