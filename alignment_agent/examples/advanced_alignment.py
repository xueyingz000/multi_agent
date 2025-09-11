#!/usr/bin/env python3
"""Advanced semantic alignment example for IFC Semantic Agent."""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from core import IFCSemanticAgent, SemanticAlignment
from knowledge_graph import GraphBuilder, RAGSystem
from utils import get_logger


class AdvancedAlignmentDemo:
    """Advanced demonstration of IFC-regulatory semantic alignment."""
    
    def __init__(self, config_path: str):
        """Initialize the demo with configuration."""
        self.logger = get_logger(__name__)
        self.agent = IFCSemanticAgent(config_path)
        self.alignment = SemanticAlignment()
        
    def load_complex_ifc_data(self) -> Dict[str, Any]:
        """Load complex IFC data for demonstration."""
        return {
            "project_info": {
                "name": "Commercial Building Project",
                "description": "Multi-story commercial building with mixed use",
                "location": "Beijing, China"
            },
            "spatial_structure": {
                "building": {
                    "id": "building_001",
                    "type": "IfcBuilding",
                    "name": "MainBuilding",
                    "properties": {
                        "BuildingType": "Commercial",
                        "NumberOfStoreys": 15,
                        "TotalHeight": 60.0,  # meters
                        "GrossFloorArea": 12000.0  # square meters
                    }
                },
                "storeys": [
                    {
                        "id": f"storey_{i:03d}",
                        "type": "IfcBuildingStorey",
                        "name": f"Floor_{i}",
                        "elevation": i * 4.0,
                        "properties": {
                            "FloorArea": 800.0,
                            "CeilingHeight": 3.6
                        }
                    } for i in range(1, 16)
                ]
            },
            "structural_elements": {
                "walls": [
                    {
                        "id": "wall_ext_001",
                        "type": "IfcWall",
                        "subtype": "STANDARD",
                        "name": "ExteriorWall_North",
                        "properties": {
                            "Width": 300,  # mm
                            "Height": 3600,  # mm
                            "Length": 20000,  # mm
                            "Material": "ReinforcedConcrete",
                            "ThermalTransmittance": 0.35,  # W/m²K
                            "FireResistance": 120,  # minutes
                            "LoadBearing": True
                        },
                        "location": "storey_001"
                    },
                    {
                        "id": "wall_int_001",
                        "type": "IfcWall",
                        "subtype": "PARTITIONING",
                        "name": "InteriorWall_Office",
                        "properties": {
                            "Width": 120,  # mm
                            "Height": 3600,  # mm
                            "Length": 8000,  # mm
                            "Material": "LightweightConcrete",
                            "SoundInsulation": 45,  # dB
                            "FireResistance": 60,  # minutes
                            "LoadBearing": False
                        },
                        "location": "storey_001"
                    }
                ],
                "slabs": [
                    {
                        "id": "slab_floor_001",
                        "type": "IfcSlab",
                        "subtype": "FLOOR",
                        "name": "FloorSlab_Level1",
                        "properties": {
                            "Thickness": 200,  # mm
                            "Area": 800.0,  # m²
                            "Material": "ReinforcedConcrete",
                            "LoadCapacity": 5.0,  # kN/m²
                            "FireResistance": 120,  # minutes
                            "VibrationControl": True
                        },
                        "location": "storey_001"
                    }
                ],
                "columns": [
                    {
                        "id": "column_001",
                        "type": "IfcColumn",
                        "name": "StructuralColumn_A1",
                        "properties": {
                            "CrossSection": "400x400",  # mm
                            "Height": 4000,  # mm
                            "Material": "ReinforcedConcrete",
                            "ConcreteGrade": "C30",
                            "SteelGrade": "HRB400",
                            "LoadCapacity": 2000,  # kN
                            "FireResistance": 180  # minutes
                        },
                        "location": "storey_001"
                    }
                ]
            },
            "building_services": {
                "hvac": [
                    {
                        "id": "hvac_unit_001",
                        "type": "IfcUnitaryEquipment",
                        "name": "AirHandlingUnit_Floor1",
                        "properties": {
                            "AirFlowRate": 5000,  # m³/h
                            "CoolingCapacity": 50,  # kW
                            "HeatingCapacity": 40,  # kW
                            "EnergyEfficiency": "A+",
                            "NoiseLevel": 45  # dB
                        }
                    }
                ],
                "electrical": [
                    {
                        "id": "panel_001",
                        "type": "IfcElectricDistributionBoard",
                        "name": "MainElectricalPanel_Floor1",
                        "properties": {
                            "RatedVoltage": 380,  # V
                            "RatedCurrent": 630,  # A
                            "PowerCapacity": 400,  # kW
                            "ProtectionClass": "IP54",
                            "FireResistance": 60  # minutes
                        }
                    }
                ]
            },
            "relationships": [
                {
                    "type": "IfcRelAggregates",
                    "relating_object": "building_001",
                    "related_objects": [f"storey_{i:03d}" for i in range(1, 16)]
                },
                {
                    "type": "IfcRelContainedInSpatialStructure",
                    "relating_structure": "storey_001",
                    "related_elements": ["wall_ext_001", "wall_int_001", "slab_floor_001", "column_001"]
                }
            ]
        }
    
    def load_comprehensive_regulations(self) -> str:
        """Load comprehensive regulatory text."""
        return """
        建筑设计防火规范 GB 50016-2014 相关条款：
        
        第5.1.1条：建筑高度大于54m的住宅建筑，建筑高度大于50m的公共建筑为一类高层建筑。
        
        第5.3.1条：建筑的耐火等级应根据其建筑高度、使用功能、重要性和火灾扑救难度等确定。
        
        第5.3.2条：一类高层建筑的耐火等级不应低于一级，二类高层建筑的耐火等级不应低于二级。
        
        第5.4.1条：一级耐火等级建筑的主要构件的燃烧性能和耐火极限不应低于下表的规定：
        - 承重墙：不燃性，3.00h
        - 楼板：不燃性，1.50h
        - 柱：不燃性，3.00h
        - 梁：不燃性，2.00h
        
        第6.1.1条：建筑内疏散楼梯间及其前室的门的净宽度不应小于0.90m，疏散走道和其他疏散门的净宽度不应小于1.40m。
        
        第6.1.9条：公共建筑内房间疏散门、安全出口、疏散走道和疏散楼梯的各自总净宽度，应根据疏散人数按每100人的最小疏散净宽度计算确定。
        
        建筑节能设计标准 GB 50189-2015 相关条款：
        
        第4.2.1条：建筑围护结构热工性能应满足下列要求：
        - 外墙平均传热系数：严寒地区≤0.35 W/(m²·K)
        - 屋面传热系数：严寒地区≤0.30 W/(m²·K)
        
        第4.3.1条：建筑外窗的气密性不应低于现行国家标准《建筑外门窗气密、水密、抗风压性能分级及检测方法》GB/T 7106规定的6级。
        
        建筑结构荷载规范 GB 50009-2012 相关条款：
        
        第4.1.1条：民用建筑楼面均布活荷载标准值及其组合值系数、频遇值系数和准永久值系数，应按下表采用：
        - 办公楼：2.0 kN/m²
        - 商店：3.5 kN/m²
        - 展览厅：3.5 kN/m²
        
        第5.1.1条：雪荷载标准值应按下式计算：sk = μr·s0
        其中：μr为屋面积雪分布系数；s0为基本雪压。
        
        建筑抗震设计规范 GB 50011-2010 相关条款：
        
        第3.1.2条：抗震设防烈度为6度及以上地区的建筑，必须进行抗震设计。
        
        第6.1.1条：钢筋混凝土结构的抗震等级应根据结构类型、房屋高度、设防烈度确定。
        
        第6.3.1条：框架结构的抗震等级为一级时，其梁柱节点核芯区应符合下列构造要求：
        - 箍筋应采用封闭式
        - 箍筋直径不应小于8mm
        - 箍筋间距不应大于100mm
        """
    
    def demonstrate_entity_alignment(self, ifc_data: Dict, regulatory_text: str):
        """Demonstrate entity-level semantic alignment."""
        print("\n" + "="*80)
        print("实体级语义对齐演示")
        print("="*80)
        
        # Extract IFC entities
        ifc_entities = []
        for category, elements in ifc_data["structural_elements"].items():
            for element in elements:
                ifc_entities.append({
                    "id": element["id"],
                    "type": element["type"],
                    "name": element["name"],
                    "properties": element["properties"]
                })
        
        # Process alignment for each entity
        for entity in ifc_entities[:3]:  # Limit to first 3 for demo
            query = f"将IFC实体 {entity['type']} ({entity['name']}) 与监管要求进行语义对齐"
            
            response = self.agent.process_query(
                query=query,
                ifc_data={"entity": entity},
                regulatory_text=regulatory_text
            )
            
            print(f"\n实体: {entity['type']} - {entity['name']}")
            print(f"对齐结果: {response.final_answer}")
            print(f"置信度: {response.confidence_score:.2f}")
    
    def demonstrate_property_alignment(self, ifc_data: Dict, regulatory_text: str):
        """Demonstrate property-level semantic alignment."""
        print("\n" + "="*80)
        print("属性级语义对齐演示")
        print("="*80)
        
        # Focus on specific properties
        wall_properties = ifc_data["structural_elements"]["walls"][0]["properties"]
        
        property_queries = [
            ("FireResistance", "耐火极限", wall_properties["FireResistance"]),
            ("ThermalTransmittance", "传热系数", wall_properties["ThermalTransmittance"]),
            ("Width", "墙体厚度", wall_properties["Width"])
        ]
        
        for ifc_prop, cn_prop, value in property_queries:
            query = f"检查IFC属性 {ifc_prop} (值: {value}) 是否符合{cn_prop}的监管要求"
            
            response = self.agent.process_query(
                query=query,
                ifc_data={"property": {ifc_prop: value}},
                regulatory_text=regulatory_text
            )
            
            print(f"\n属性: {ifc_prop} ({cn_prop})")
            print(f"当前值: {value}")
            print(f"合规性分析: {response.final_answer}")
            print(f"置信度: {response.confidence_score:.2f}")
    
    def demonstrate_relationship_alignment(self, ifc_data: Dict, regulatory_text: str):
        """Demonstrate relationship-level semantic alignment."""
        print("\n" + "="*80)
        print("关系级语义对齐演示")
        print("="*80)
        
        # Analyze spatial relationships
        spatial_query = "分析建筑空间结构与疏散要求的关系对齐"
        
        response = self.agent.process_query(
            query=spatial_query,
            ifc_data={
                "spatial_structure": ifc_data["spatial_structure"],
                "relationships": ifc_data["relationships"]
            },
            regulatory_text=regulatory_text
        )
        
        print(f"空间关系分析: {response.final_answer}")
        print(f"置信度: {response.confidence_score:.2f}")
        
        # Analyze structural relationships
        structural_query = "分析结构构件之间的承载关系与荷载规范的对齐"
        
        response2 = self.agent.process_query(
            query=structural_query,
            ifc_data={
                "structural_elements": ifc_data["structural_elements"],
                "relationships": ifc_data["relationships"]
            },
            regulatory_text=regulatory_text
        )
        
        print(f"\n结构关系分析: {response2.final_answer}")
        print(f"置信度: {response2.confidence_score:.2f}")
    
    def demonstrate_compliance_checking(self, ifc_data: Dict, regulatory_text: str):
        """Demonstrate comprehensive compliance checking."""
        print("\n" + "="*80)
        print("综合合规性检查演示")
        print("="*80)
        
        compliance_queries = [
            "检查建筑高度是否符合高层建筑分类要求",
            "验证结构构件耐火极限是否满足一级耐火等级要求",
            "评估外墙热工性能是否符合节能标准",
            "分析楼板荷载能力是否满足使用要求"
        ]
        
        compliance_results = []
        
        for query in compliance_queries:
            response = self.agent.process_query(
                query=query,
                ifc_data=ifc_data,
                regulatory_text=regulatory_text
            )
            
            compliance_results.append({
                "query": query,
                "result": response.final_answer,
                "confidence": response.confidence_score,
                "steps": response.total_steps
            })
            
            print(f"\n检查项目: {query}")
            print(f"检查结果: {response.final_answer}")
            print(f"置信度: {response.confidence_score:.2f}")
            print(f"推理步骤: {response.total_steps}")
        
        # Generate compliance summary
        print("\n" + "-"*60)
        print("合规性检查汇总")
        print("-"*60)
        
        total_confidence = sum(r["confidence"] for r in compliance_results) / len(compliance_results)
        print(f"整体合规置信度: {total_confidence:.2f}")
        
        high_confidence = [r for r in compliance_results if r["confidence"] > 0.8]
        medium_confidence = [r for r in compliance_results if 0.6 <= r["confidence"] <= 0.8]
        low_confidence = [r for r in compliance_results if r["confidence"] < 0.6]
        
        print(f"高置信度检查项 (>0.8): {len(high_confidence)}")
        print(f"中等置信度检查项 (0.6-0.8): {len(medium_confidence)}")
        print(f"低置信度检查项 (<0.6): {len(low_confidence)}")
        
        if low_confidence:
            print("\n需要进一步验证的项目:")
            for item in low_confidence:
                print(f"- {item['query']} (置信度: {item['confidence']:.2f})")
    
    def run_demo(self):
        """Run the complete advanced alignment demonstration."""
        self.logger.info("Starting advanced semantic alignment demonstration")
        
        # Load data
        ifc_data = self.load_complex_ifc_data()
        regulatory_text = self.load_comprehensive_regulations()
        
        print("高级IFC-监管语义对齐演示")
        print("="*80)
        print(f"项目: {ifc_data['project_info']['name']}")
        print(f"描述: {ifc_data['project_info']['description']}")
        print(f"位置: {ifc_data['project_info']['location']}")
        
        # Run demonstrations
        self.demonstrate_entity_alignment(ifc_data, regulatory_text)
        self.demonstrate_property_alignment(ifc_data, regulatory_text)
        self.demonstrate_relationship_alignment(ifc_data, regulatory_text)
        self.demonstrate_compliance_checking(ifc_data, regulatory_text)
        
        print("\n" + "="*80)
        print("高级语义对齐演示完成！")
        print("="*80)
        
        self.logger.info("Advanced semantic alignment demonstration completed")


def main():
    """Main function to run the advanced alignment demo."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    demo = AdvancedAlignmentDemo(str(config_path))
    demo.run_demo()


if __name__ == "__main__":
    main()