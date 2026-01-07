import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement
import numpy as np
import logging
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_accumulated_z(element):
    """递归获取累积Z坐标"""
    z = 0.0
    try:
        placement = element.ObjectPlacement
        while placement:
            matrix = ifcopenshell.util.placement.get_local_placement(placement)
            if matrix is not None:
                z += matrix[2][3]
            if hasattr(placement, "PlacementRelTo") and placement.PlacementRelTo:
                placement = placement.PlacementRelTo
            else:
                break
    except Exception:
        pass
    return z

def check_vertical_spans(ifc_path):
    logger.info(f"Loading IFC file: {ifc_path}")
    ifc_file = ifcopenshell.open(ifc_path)
    
    # 定义要检查的构件类型
    types_to_check = [
        "IfcWall", "IfcWallStandardCase", "IfcCurtainWall", 
        "IfcMember", "IfcPlate", "IfcColumn", "IfcWindow"
    ]
    
    all_elements = []
    for t in types_to_check:
        all_elements.extend(ifc_file.by_type(t))
    
    logger.info(f"Total elements to check: {len(all_elements)}")
    
    # 设置几何提取
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    # 目标标高
    target_elevations = [0.0, 4.5, 8.4, 12.3]
    coverage_stats = {elev: [] for elev in target_elevations}
    
    # 简单的点Z坐标分组统计（模拟当前逻辑）
    point_z_stats = {elev: [] for elev in target_elevations}
    
    count = 0
    for element in all_elements:
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} elements...", end="\r")
            
        try:
            # 1. 获取几何包围盒 Z 范围
            if element.Representation:
                try:
                    shape = ifcopenshell.geom.create_shape(settings, element)
                    verts = shape.geometry.verts
                    z_coords = [verts[i+2] for i in range(0, len(verts), 3)]
                    min_z = min(z_coords)
                    max_z = max(z_coords)
                    
                    # 检查覆盖哪些标高 (允许 0.1m 的容差)
                    for elev in target_elevations:
                        # 逻辑：构件的垂直范围包含了标高切面
                        # min_z <= elev <= max_z
                        # 稍微放宽一点，允许切面在构件内部或边缘
                        if min_z - 0.1 <= elev <= max_z + 0.1:
                            coverage_stats[elev].append(element.id())
                            
                except Exception as e:
                    # logger.warning(f"Failed to create shape for #{element.id()}: {e}")
                    pass

            # 2. 获取点 Z 坐标（模拟当前逻辑）
            elem_z = get_accumulated_z(element)
            for elev in target_elevations:
                if abs(elem_z - elev) < 2.0: # 假设容差 2.0
                    point_z_stats[elev].append(element.id())
                    
        except Exception as e:
            pass
            
    print("\n")
    logger.info("=== Analysis Results ===")
    
    for elev in target_elevations:
        span_count = len(coverage_stats[elev])
        point_count = len(point_z_stats[elev])
        logger.info(f"Elevation {elev}m:")
        logger.info(f"  - Elements covering this elevation (Span Logic): {span_count}")
        logger.info(f"  - Elements near this elevation (Point Logic): {point_count}")
        logger.info(f"  - Diff (Span - Point): {span_count - point_count}")
        
        # 如果差异大，打印一些差异构件的类型
        if span_count > point_count:
            span_ids = set(coverage_stats[elev])
            point_ids = set(point_z_stats[elev])
            diff_ids = span_ids - point_ids
            
            diff_types = {}
            for eid in list(diff_ids)[:20]: # 只检查前20个
                el = ifc_file.by_id(eid)
                t = el.is_a()
                diff_types[t] = diff_types.get(t, 0) + 1
            
            logger.info(f"  - Sample missing types in Point Logic: {diff_types}")

if __name__ == "__main__":
    ifc_path = "/Users/zhuxueying/ifc/ifc_files/academic b.ifc"
    check_vertical_spans(ifc_path)
