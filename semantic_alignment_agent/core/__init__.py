try:
    from .function_inference import (
        FunctionInferenceEngine,
        InferenceRule,
    )
except Exception:
    FunctionInferenceEngine = None
    InferenceRule = None

try:
    from .vertical_space_detector import (
        VerticalSpaceDetector,
        DetectionCandidate,
    )
except Exception:
    VerticalSpaceDetector = None
    DetectionCandidate = None

# Avoid importing heavy semantic_alignment_agent by default to keep core lightweight
try:
    from .semantic_alignment_agent import (
        SemanticAlignmentAgent,
        AlignmentType,
        ConfidenceLevel,
        AlignmentContext,
        AlignmentResult,
    )
except Exception:
    SemanticAlignmentAgent = None
    AlignmentType = None
    ConfidenceLevel = None
    AlignmentContext = None
    AlignmentResult = None

__all__ = [
    'SemanticAlignmentAgent',
    'AlignmentType',
    'ConfidenceLevel',
    'AlignmentContext',
    'AlignmentResult',
    'FunctionInferenceEngine',
    'InferenceRule',
    'VerticalSpaceDetector',
    'DetectionCandidate',
]