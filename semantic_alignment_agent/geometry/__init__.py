try:
    from .geometry_analyzer import GeometryAnalyzer, SpatialContext
except Exception:
    GeometryAnalyzer = None
    SpatialContext = None

__all__ = [
    'GeometryAnalyzer',
    'SpatialContext',
]