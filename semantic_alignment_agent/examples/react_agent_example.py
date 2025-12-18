from typing import Dict, Any

from semantic_alignment_agent.core.react_semantic_agent import (
    SemanticAlignmentReActAgent,
    AgentConfig,
)
from semantic_alignment_agent.utils.data_structures import IfcElementInfo, IfcElementType


def build_sample_element() -> IfcElementInfo:
    return IfcElementInfo(
        guid="TEST-IFC-0001",
        ifc_type=IfcElementType.SLAB,
        properties={
            "Name": "Roof Slab",
            "Material": "Concrete",
            "Storey": "L3",
        },
        geometric_features=None,
        functional_inference=None,
    )


def build_regulation_data() -> Dict[str, Any]:
    # 最简示例：可替换为解析后的法规 JSON
    return {
        "region": "CN",
        "rules": [],
    }


def main():
    cfg = AgentConfig(max_steps=5, stop_confidence=0.8, enable_llm=True)
    agent = SemanticAlignmentReActAgent(cfg)

    element = build_sample_element()
    regulation_data = build_regulation_data()

    outcome = agent.align_element(
        element=element,
        regulation_data=regulation_data,
        building_context={"project_name": "Demo Building"},
        spatial_context=None,
        target_region="CN",
    )

    print("Element GUID:", outcome.element_guid)
    print("Regulation Category:", outcome.regulation_category)
    print("Function Classification:", outcome.function_classification)
    print("Confidence:", outcome.confidence)
    print("Requires Review:", outcome.requires_review)
    print("Reasoning Path:", " | ".join(outcome.reasoning_path))
    print("Evidence (truncated):", [e[:120] for e in outcome.evidence])


if __name__ == "__main__":
    main()