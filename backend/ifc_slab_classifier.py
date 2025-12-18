import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.util.placement
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, box
from shapely import ops
from shapely.ops import unary_union  # Add this import line
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import logging
from scipy.spatial import ConvexHull
# from visualization import (
#     visualize_classification_by_storey,
#     visualize_section_lines,
#     create_height_heatmap,
# )
from shapely.geometry import MultiLineString
from shapely.ops import polygonize
from shapely.geometry import MultiPoint

# Import area calculation module
# from area_calculator import (
#     calculate_areas_for_slab,
#     process_classification_results,
#     print_area_statistics,
#     generate_area_report,
# )

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_ifc_file(file_path):
    """Load IFC file"""
    return ifcopenshell.open(file_path)


def get_storey_elevation(storey):
    """Get the elevation of a storey"""
    placement = storey.ObjectPlacement
    if placement:
        matrix = ifcopenshell.util.placement.get_local_placement(placement)
        return matrix[2][3]  # Z-axis position is the elevation
    return None


def get_slab_geometry(ifc_file, slab):
    """Get slab geometry information, including detailed 3D geometry data"""
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    shape = ifcopenshell.geom.create_shape(settings, slab)
    verts = shape.geometry.verts
    faces = shape.geometry.faces

    # Extract top and bottom surface Z coordinates
    z_coords = [verts[i + 2] for i in range(0, len(verts), 3)]
    min_z = min(z_coords)
    max_z = max(z_coords)

    # Extract all vertex 3D coordinates
    vertices_3d = []
    for i in range(0, len(verts), 3):
        vertices_3d.append([verts[i], verts[i + 1], verts[i + 2]])

    # Extract all faces
    face_vertices = []
    for i in range(0, len(faces), 3):
        v1_idx = faces[i]
        v2_idx = faces[i + 1]
        v3_idx = faces[i + 2]

        v1 = [verts[v1_idx * 3], verts[v1_idx * 3 + 1], verts[v1_idx * 3 + 2]]
        v2 = [verts[v2_idx * 3], verts[v2_idx * 3 + 1], verts[v2_idx * 3 + 2]]
        v3 = [verts[v3_idx * 3], verts[v3_idx * 3 + 1], verts[v3_idx * 3 + 2]]

        face_vertices.append([v1, v2, v3])

    # Extract top and bottom faces
    top_faces = []
    bottom_faces = []

    for face in face_vertices:
        if all(abs(v[2] - max_z) < 0.01 for v in face):
            top_faces.append(face)
        if all(abs(v[2] - min_z) < 0.01 for v in face):
            bottom_faces.append(face)

    # Construct top and bottom surface polygons
    top_polygon = create_polygon_from_faces(top_faces)
    bottom_polygon = create_polygon_from_faces(bottom_faces)

    # If polygons were successfully created, return geometry information
    if top_polygon and bottom_polygon:
        return {
            "min_z": min_z,
            "max_z": max_z,
            "top_polygon": top_polygon,
            "bottom_polygon": bottom_polygon,
            "vertices_3d": vertices_3d,
            "faces": face_vertices,
            "name": (
                slab.Name
                if hasattr(slab, "Name") and slab.Name
                else f"Slab-{slab.id()}"
            ),
        }
    return None


def get_object_geometry(ifc_file, obj):
    """Get geometry information for any IFC object"""
    try:
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        shape = ifcopenshell.geom.create_shape(settings, obj)
        verts = shape.geometry.verts
        faces = shape.geometry.faces

        # 提取Z坐标范围
        z_coords = [verts[i + 2] for i in range(0, len(verts), 3)]
        min_z = min(z_coords) if z_coords else None
        max_z = max(z_coords) if z_coords else None

        # 增强面数据结构检查和处理
        face_vertices = []
        for i in range(0, len(faces), 3):
            if i + 2 < len(faces):
                v1_idx = faces[i]
                v2_idx = faces[i + 1]
                v3_idx = faces[i + 2]

                if (
                    v1_idx * 3 + 2 < len(verts)
                    and v2_idx * 3 + 2 < len(verts)
                    and v3_idx * 3 + 2 < len(verts)
                ):

                    v1 = [
                        verts[v1_idx * 3],
                        verts[v1_idx * 3 + 1],
                        verts[v1_idx * 3 + 2],
                    ]
                    v2 = [
                        verts[v2_idx * 3],
                        verts[v2_idx * 3 + 1],
                        verts[v2_idx * 3 + 2],
                    ]
                    v3 = [
                        verts[v3_idx * 3],
                        verts[v3_idx * 3 + 1],
                        verts[v3_idx * 3 + 2],
                    ]

                    face_vertices.append([v1, v2, v3])

        # 使用处理后的面数据
        element_info = {
            "min_z": min_z,
            "max_z": max_z,
            "verts": verts,
            "faces": face_vertices,  # 使用重新组织的面数据
            "type": obj.is_a(),
            "name": (
                obj.Name
                if hasattr(obj, "Name") and obj.Name
                else f"{obj.is_a()}-{obj.id()}"
            ),
        }

        return element_info
    except Exception as e:
        logger.warning(f"Failed to get object geometry information: {e}")
        return None


def create_polygon_from_faces(faces):
    """Create polygon from triangle face collection"""
    if not faces:
        return None

    # Extract all edges
    edges = []
    for face in faces:
        edges.append((tuple(face[0][0:2]), tuple(face[1][0:2])))
        edges.append((tuple(face[1][0:2]), tuple(face[2][0:2])))
        edges.append((tuple(face[2][0:2]), tuple(face[0][0:2])))

    # Remove duplicate edges (considering direction)
    unique_edges = []
    for edge in edges:
        reverse_edge = (edge[1], edge[0])
        if reverse_edge in unique_edges:
            unique_edges.remove(reverse_edge)
        else:
            unique_edges.append(edge)

    # 从边构建轮廓
    if not unique_edges:
        return None

    try:
        # 构建边的连接关系
        from collections import defaultdict

        edge_map = defaultdict(list)
        for start, end in unique_edges:
            edge_map[start].append(end)

        # 按顺序连接轮廓点
        boundary_points = []
        current = next(iter(edge_map.keys()))
        start_point = current

        while edge_map:
            boundary_points.append(current)
            if current in edge_map and edge_map[current]:
                next_point = edge_map[current].pop(0)
                if not edge_map[current]:
                    del edge_map[current]
                current = next_point
                if current == start_point:  # 如果回到起点，完成一个环
                    break
            else:
                break

        # 创建多边形
        if len(boundary_points) >= 3:
            polygon = Polygon(boundary_points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            return polygon
    except Exception as e:
        logger.warning(f"多边形创建错误: {e}")
        # 如果上述方法失败，尝试使用凸包
        try:
            # 收集所有不重复的点
            all_points = set()
            for face in faces:
                for v in face:
                    all_points.add((v[0], v[1]))

            all_points = list(all_points)
            if len(all_points) < 3:
                return None

            # 使用凸包算法
            hull = ConvexHull(all_points)
            hull_points = [all_points[i] for i in hull.vertices]

            polygon = Polygon(hull_points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            return polygon
        except Exception as e:
            logger.warning(f"凸包创建错误: {e}")
            return None

    return None


def find_elements_above_slab(
    ifc_file, slab_geometry, all_elements_geometry, next_level_elevation=None
):
    """Find all elements above the slab and calculate slab clearance height

    Modified logic:
    1. If there are slabs above, use the distance between upper and lower slabs as clearance height
    2. If there are no slabs above, use the distance from roof to this slab as clearance height
    """
    slab_max_z = slab_geometry["max_z"]
    slab_polygon = slab_geometry["top_polygon"]
    elements_above = []

    # 计算搜索的最大高度（下一层或默认高度）
    max_search_height = (
        next_level_elevation if next_level_elevation else slab_max_z + 5.0
    )

    # 第一步：检查上方是否有楼板
    slabs_above = []

    for element_id, element_geom in all_elements_geometry.items():
        # 只检查类型为"IFCSLAB"的元素
        if element_geom["type"] == "IfcSlab":
            # 检查楼板是否有部分在当前楼板上方
            if element_geom["min_z"] > slab_max_z:
                # 检查水平投影是否相交
                element_points = []
                element_verts = element_geom["verts"]

                for i in range(0, len(element_verts), 3):
                    element_points.append((element_verts[i], element_verts[i + 1]))

                try:
                    # 创建元素的2D凸包
                    if len(element_points) >= 3:
                        hull = ConvexHull(element_points)
                        hull_points = [element_points[i] for i in hull.vertices]
                        element_polygon = Polygon(hull_points)

                        # 检查是否与当前楼板相交
                        if element_polygon.intersects(slab_polygon):
                            # 计算这个楼板对当前楼板的实际影响高度
                            min_effective_z = element_geom["min_z"]

                            # 添加到上方楼板列表
                            slab_info = element_geom.copy()
                            slab_info["effective_min_z"] = min_effective_z
                            slabs_above.append(slab_info)
                except Exception as e:
                    logger.warning(f"处理上方楼板时出错: {e}")
                    continue

    # 按照高度排序上方楼板
    slabs_above.sort(key=lambda x: x["effective_min_z"])

    # 第二步：根据是否有上层楼板采用不同的处理逻辑
    if slabs_above:
        # 情况1：有上层楼板，直接使用最近的上层楼板计算净空高度
        logger.info(
            f"Found {len(slabs_above)} slabs above slab {slab_geometry['name']}"
        )
        nearest_slab = slabs_above[0]
        min_clearance = nearest_slab["effective_min_z"] - slab_max_z
        logger.info(
            f"Nearest slab above: {nearest_slab['name']}, distance: {min_clearance:.2f}m"
        )

        # 只返回上方的楼板作为影响元素
        return [nearest_slab], min_clearance
    else:
        # 情况2：没有上层楼板，查找上方的所有元素（屋顶等）
        logger.info(
            f"No other slabs above slab {slab_geometry['name']}, checking roofs and other elements"
        )

        # 查找上方的所有元素
        for element_id, element_geom in all_elements_geometry.items():
            # 跳过楼板类型，因为已经在前面处理过
            if element_geom["type"] == "IfcSlab":
                continue

            # 检查元素是否有任何部分在楼板上方
            has_part_above = False
            element_verts = element_geom["verts"]

            # 检查是否有任何顶点在楼板上方
            for i in range(0, len(element_verts), 3):
                if element_verts[i + 2] > slab_max_z:  # 检查z坐标
                    has_part_above = True
                    break

            # 如果元素没有部分在楼板上方，跳过此元素
            if not has_part_above:
                continue

            # 检查水平投影是否相交
            element_points = []
            for i in range(0, len(element_verts), 3):
                element_points.append((element_verts[i], element_verts[i + 1]))

            try:
                # 创建元素的2D凸包
                if len(element_points) >= 3:
                    hull = ConvexHull(element_points)
                    hull_points = [element_points[i] for i in hull.vertices]
                    element_polygon = Polygon(hull_points)

                    if element_polygon.intersects(slab_polygon):
                        # 计算这个元素对楼板的实际影响高度
                        min_effective_z = calculate_min_effective_height(
                            element_geom, slab_polygon, slab_max_z
                        )

                        # 添加有效最低高度信息
                        element_info = element_geom.copy()
                        element_info["effective_min_z"] = min_effective_z
                        elements_above.append(element_info)
            except Exception as e:
                logger.warning(f"处理上方元素时出错: {e}")
                # 如果凸包创建失败，检查是否有点在楼板上方
                for point in element_points:
                    if slab_polygon.contains(Point(point)) or slab_polygon.touches(
                        Point(point)
                    ):
                        # 计算这个元素对楼板的实际影响高度
                        min_effective_z = calculate_min_effective_height(
                            element_geom, slab_polygon, slab_max_z
                        )

                        element_info = element_geom.copy()
                        element_info["effective_min_z"] = min_effective_z
                        elements_above.append(element_info)
                        break

        # 按有效Z坐标排序
        elements_above.sort(key=lambda x: x["effective_min_z"])

        # 计算净空高度
        if elements_above:
            # 使用有效最低高度计算净空
            min_clearance = elements_above[0]["effective_min_z"] - slab_max_z
        else:
            # 如果没有上方元素但有下一层楼层，使用下一层标高
            if next_level_elevation:
                min_clearance = next_level_elevation - slab_max_z
            else:
                # 对于顶层楼板，使用合理的默认值
                min_clearance = 3.0  # 假设一个足够高的默认值

        logger.info(
            f"Found {len(elements_above)} elements above slab {slab_geometry['name']}, minimum clearance height: {min_clearance:.2f}m"
        )
        return elements_above, min_clearance


def calculate_min_effective_height(element_geom, slab_polygon, slab_max_z):
    """计算元素对楼板的有效影响高度

    分析元素与楼板重叠区域的最低点高度。对于倾斜元素（如坡屋顶），
    会正确计算其在楼板投影区域内的最低高度。
    """
    element_verts = element_geom["verts"]
    element_faces = element_geom["faces"]
    min_effective_z = float("inf")

    # 提取楼板多边形的边界框
    minx, miny, maxx, maxy = slab_polygon.bounds

    # 采样网格大小
    grid_size = 0.5  # 可根据精度需求调整

    # 创建采样网格点
    x_points = np.arange(minx, maxx + grid_size, grid_size)
    y_points = np.arange(miny, maxy + grid_size, grid_size)

    # 对楼板上每个采样点，计算元素在该点上方的最低高度
    for x in x_points:
        for y in y_points:
            point = Point(x, y)
            if not slab_polygon.contains(point):
                continue

            # 从该点向上发射射线，找出与元素各面的交点
            for face in element_faces:
                # 在新结构中，face直接是三个点的坐标
                v1, v2, v3 = face

                # 检查三角形是否在点的正上方（水平投影）
                face_polygon = Polygon([(v1[0], v1[1]), (v2[0], v2[1]), (v3[0], v3[1])])
                if not face_polygon.contains(point) and not face_polygon.intersects(
                    point
                ):
                    continue

                # 使用重心坐标法计算交点Z值
                try:
                    # 假设三角形不是完全垂直的
                    # 使用平面方程计算近似Z值
                    a = (v2[1] - v1[1]) * (v3[2] - v1[2]) - (v3[1] - v1[1]) * (
                        v2[2] - v1[2]
                    )
                    b = (v2[2] - v1[2]) * (v3[0] - v1[0]) - (v3[2] - v1[2]) * (
                        v2[0] - v1[0]
                    )
                    c = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (
                        v2[1] - v1[1]
                    )

                    # 如果三角形接近垂直，跳过
                    if abs(c) < 1e-6:
                        continue

                    d = -a * v1[0] - b * v1[1] - c * v1[2]

                    # 计算交点Z值
                    intersection_z = (-d - a * x - b * y) / c

                    # 仅考虑楼板上方的交点
                    if intersection_z > slab_max_z:
                        min_effective_z = min(min_effective_z, intersection_z)
                except Exception as e:
                    continue

    # 如果没有找到有效交点，使用元素的min_z
    if min_effective_z == float("inf"):
        # 对于完全在楼板上方的元素，直接使用其最低点
        if element_geom["min_z"] > slab_max_z:
            min_effective_z = element_geom["min_z"]
        else:
            # 对于部分在楼板下方的元素，计算在楼板上方部分的最低Z值
            for i in range(0, len(element_verts), 3):
                z = element_verts[i + 2]
                if z > slab_max_z:
                    min_effective_z = min(min_effective_z, z)

            # 如果仍未找到，使用略高于楼板的值
            if min_effective_z == float("inf"):
                min_effective_z = slab_max_z + 0.01

    return min_effective_z


def analyze_horizontal_section_with_lines(
    slab_geometry, elements_above, threshold=2.2, roof_only=True
):
    """使用水平剖面法分析楼板并返回交线信息

    Args:
        slab_geometry: 楼板几何信息
        elements_above: 楼板上方元素列表
        threshold: 高度阈值，默认2.2米
        roof_only: 是否只考虑屋顶/屋檐元素，默认True

    返回：水平剖面法结果和找到的交线列表
    """
    from shapely.ops import unary_union
    from visualization import visualize_section_lines

    slab_polygon = slab_geometry["top_polygon"]
    slab_max_z = slab_geometry["max_z"]
    section_z = slab_max_z + threshold  # 剖面高度

    # 查找与剖面相交的线段
    all_intersect_lines = []

    # 检查每个元素与剖面的交线
    for element in elements_above:
        # 跳过高度完全低于剖面的元素
        if element.get("max_z", 0) <= section_z:
            continue

        # 跳过高度完全高于剖面的元素
        if element.get("min_z", float("inf")) >= section_z:
            continue

        element_type = element.get("type", "")
        element_name = element.get("name", "Unknown")

        # 如果只考虑屋顶/屋檐元素，则过滤掉墙体等其他元素
        if roof_only and not (
            element_type in ["IfcRoof", "IfcCovering"]
            or "屋檐" in element_name
            or "屋顶" in element_name
        ):
            logger.info(
                f"  Skip non-roof/eave element: {element_name} ({element_type})"
            )
            continue

        logger.info(f"  Check element: {element_name} ({element_type})")

        # 查找与剖面相交的面
        intersect_lines = find_section_lines(element, section_z)

        if intersect_lines:
            logger.info(
                f"  Element {element_name} has {len(intersect_lines)} faces crossing the section"
            )
            if "屋檐" in element_name or element_type == "IfcCovering":
                logger.info(f"【Eave Bottom Plate Details】")
                for i, face in enumerate(element.get("faces", [])):
                    if i < 10:  # 限制输出数量
                        pts = [
                            f"Point{j+1}=({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})"
                            for j, pt in enumerate(face)
                        ]
                        logger.info(f"  Face {i+1}: {', '.join(pts)}")

                logger.info(
                    f"【Line Segments Generated by Eave Bottom Plate】Count: {len(intersect_lines)}"
                )
                for i, line in enumerate(intersect_lines):
                    coords = list(line.coords)
                    logger.info(
                        f"  Segment {i+1}: Start=({coords[0][0]:.2f}, {coords[0][1]:.2f}), End=({coords[1][0]:.2f}, {coords[1][1]:.2f})"
                    )

            all_intersect_lines.extend(intersect_lines)

    logger.info(f"Found {len(all_intersect_lines)} line segments crossing the section")

    # 可视化交线
    if all_intersect_lines:
        visualize_section_lines(
            slab_geometry,
            all_intersect_lines,
            elements_above,  # 正确传递elements_above参数
            f'horizontal_section_lines_{slab_geometry.get("name", "unknown")}.png',
        )
        logger.info("【All Intersection Lines Details】")
        for i, line in enumerate(all_intersect_lines):
            coords = list(line.coords)
            logger.info(
                f"Segment {i+1}: Start=({coords[0][0]:.2f}, {coords[0][1]:.2f}), End=({coords[1][0]:.2f}, {coords[1][1]:.2f})"
            )

    # 处理交线与楼板边界，形成封闭区域
    high_areas, low_areas = process_section_with_boundary(
        all_intersect_lines, slab_polygon, slab_max_z, elements_above, threshold
    )

    # 合并结果
    result = {}
    if high_areas:
        result["high"] = unary_union(high_areas)
    else:
        result["high"] = None

    if low_areas:
        result["low"] = unary_union(low_areas)
    else:
        result["low"] = None

    return result, all_intersect_lines


def find_section_lines(element, section_z):
    """查找元素与指定高度水平剖面的交线"""
    from shapely.geometry import LineString

    # 获取元素的所有面
    faces = element.get("faces", [])

    # 调试输出
    if faces and (not isinstance(faces, list) or (faces and isinstance(faces[0], int))):
        logger.warning(
            f"元素 {element.get('name', 'unknown')} 的面数据结构异常: {type(faces)}"
        )
        logger.warning(f"第一个面的类型: {type(faces[0]) if faces else 'N/A'}")
        logger.warning(f"面数据: {faces[:5] if len(faces) > 5 else faces}")
        return []

    # 找出与剖面相交的线段
    intersect_lines = []

    # 确保faces是正确的数据结构
    if not faces or not isinstance(faces, list):
        return []

    for face_idx, face in enumerate(faces):
        # 确保face是点列表而不是整数或其他类型
        if not isinstance(face, list):
            logger.warning(f"跳过非点列表的面: 索引={face_idx}, 类型={type(face)}")
            continue

        # 确保face中的元素是包含坐标的点
        if not face or not isinstance(face[0], (list, tuple)) or len(face[0]) < 3:
            logger.warning(f"跳过格式错误的面: 索引={face_idx}, 内容={face[:2]}")
            continue

        try:
            # 检查面的顶点是否跨过剖面
            z_values = [p[2] for p in face]
            min_z = min(z_values)
            max_z = max(z_values)

            # 如果面穿过剖面高度
            if min_z <= section_z <= max_z:
                # 计算面与剖面的所有交点
                intersections = []

                for i in range(len(face)):
                    p1 = face[i]
                    p2 = face[(i + 1) % len(face)]

                    # 检查这条边是否穿过剖面
                    if (p1[2] <= section_z <= p2[2]) or (p2[2] <= section_z <= p1[2]):
                        # 两点不在同一高度，可以计算交点
                        if p1[2] != p2[2]:
                            # 计算交点
                            t = (section_z - p1[2]) / (p2[2] - p1[2])
                            intersection_x = p1[0] + t * (p2[0] - p1[0])
                            intersection_y = p1[1] + t * (p2[1] - p1[1])

                            # 存储交点
                            intersections.append((intersection_x, intersection_y))

                # 如果找到两个或更多交点，创建线段
                if len(intersections) >= 2:
                    # 如果是闭合面，交点数量应为偶数
                    if len(intersections) % 2 != 0:
                        logger.warning(f"面与剖面交点数量为奇数: {len(intersections)}")

                    # 按顺序连接交点形成线段
                    for i in range(0, len(intersections) - 1, 2):
                        if i + 1 < len(intersections):
                            line = LineString([intersections[i], intersections[i + 1]])
                            if line.length > 0.001:  # 忽略太短的线段
                                intersect_lines.append(line)
        except Exception as e:
            logger.warning(f"处理面 {face_idx} 时出错: {e}")
            continue

    return intersect_lines


def get_element_geometry(ifc_file, element):
    """获取元素的几何信息，包括面的详细数据"""
    try:
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        shape = ifcopenshell.geom.create_shape(settings, element)

        # 获取几何信息的原始数据
        verts = shape.geometry.verts
        faces = shape.geometry.faces
        materials = shape.geometry.materials

        # 提取三维几何数据
        vertices = []
        for i in range(0, len(verts), 3):
            if i + 2 < len(verts):
                vertices.append((verts[i], verts[i + 1], verts[i + 2]))

        # 提取面数据 - 使用更底层的接口获取真正的面数据
        face_data = []
        if hasattr(shape, "geometry") and hasattr(shape.geometry, "faces"):
            # 索引数据转换为实际坐标点
            i = 0
            while i < len(faces):
                # 第一个数字表示面中的点数
                if i < len(faces):
                    num_points = faces[i]
                    i += 1

                    # 收集这个面的所有点
                    face_points = []
                    for j in range(num_points):
                        if i < len(faces):
                            idx = faces[i]
                            if idx < len(vertices):
                                face_points.append(vertices[idx])
                            i += 1

                    if len(face_points) >= 3:  # 至少需要3个点才能形成面
                        face_data.append(face_points)

        # 计算边界框
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        z_coords = [v[2] for v in vertices]

        min_x = min(x_coords) if x_coords else 0
        max_x = max(x_coords) if x_coords else 0
        min_y = min(y_coords) if y_coords else 0
        max_y = max(y_coords) if y_coords else 0
        min_z = min(z_coords) if z_coords else 0
        max_z = max(z_coords) if z_coords else 0

        # 返回元素几何信息
        name = (
            element.Name
            if hasattr(element, "Name") and element.Name
            else f"Element {element.id()}"
        )
        element_type = element.is_a()

        return {
            "id": element.id(),
            "name": name,
            "type": element_type,
            "vertices": vertices,
            "faces": face_data,  # 使用正确解析的面数据
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z,
            "bbox": (min_x, min_y, min_z, max_x, max_y, max_z),
        }
    except Exception as e:
        logger.error(f"获取元素 {element.id()} 几何信息时出错: {e}")
        return None


def analyze_slab_height(slab_geometry, elements_above, threshold=2.2):
    """分析楼板上方净空高度并识别变高度区域

    Args:
        slab_geometry: 楼板的几何信息
        elements_above: 楼板上方的元素列表
        threshold: 净空高度阈值，默认2.2米

    Returns:
        分析结果的字典：包含高区域和低区域的多边形
        最小净空高度
    """
    slab_polygon = slab_geometry["top_polygon"]
    slab_max_z = slab_geometry["max_z"]

    # 计算楼板整体净空高度
    min_clearance = float("inf")
    if elements_above:
        min_effective_z = min(
            [element.get("effective_min_z", float("inf")) for element in elements_above]
        )
        min_clearance = min_effective_z - slab_max_z

    # 检查是否有屋顶/屋檐元素
    has_roof_elements = False
    for element in elements_above:
        element_type = element.get("type", "")
        element_name = element.get("name", "")
        if element_type in ["IfcRoof", "IfcCovering"] or "屋檐" in element_name:
            has_roof_elements = True
            break

    # 检查屋檐/屋顶是否跨过剖面高度
    section_z = slab_max_z + threshold
    has_roof_crossing_section = False

    if has_roof_elements:
        for element in elements_above:
            element_type = element.get("type", "")
            element_name = element.get("name", "")
            if element_type in ["IfcRoof", "IfcCovering"] or "屋檐" in element_name:
                element_min_z = element.get("min_z", float("inf"))
                element_max_z = element.get("max_z", float("-inf"))
                if element_min_z <= section_z <= element_max_z:
                    has_roof_crossing_section = True
                    logger.info(
                        f"Detected eave/roof element {element_name} crossing section height"
                    )
                    logger.info(
                        f"Height range: {element_min_z}m - {element_max_z}m, section height: {section_z}m"
                    )
                    break

    # 首先使用水平剖面法
    logger.info(
        f"Using horizontal section method to analyze slab {slab_geometry.get('name', 'unknown')}"
    )
    section_result, section_lines = analyze_horizontal_section_with_lines(
        slab_geometry, elements_above, threshold, roof_only=True
    )

    # 检查水平剖面法结果
    has_high = section_result["high"] is not None and not (
        hasattr(section_result["high"], "is_empty") and section_result["high"].is_empty
    )
    has_low = section_result["low"] is not None and not (
        hasattr(section_result["low"], "is_empty") and section_result["low"].is_empty
    )

    if has_high and has_low:
        logger.info(
            "Horizontal section method successfully identified variable height areas"
        )
        return section_result, min_clearance

    # 如果水平剖面法没有找到变高度区域，且检测到屋顶跨过剖面，使用网格采样法
    if has_roof_crossing_section:
        logger.info(
            "Horizontal section method did not find variable height areas, but detected roof crossing section, using grid sampling method"
        )
        grid_result = analyze_using_grid_sampling(
            slab_geometry, elements_above, threshold, fine_for_roof=True
        )

        # 检查网格采样结果
        grid_has_high = grid_result["high"] is not None and not (
            hasattr(grid_result["high"], "is_empty") and grid_result["high"].is_empty
        )
        grid_has_low = grid_result["low"] is not None and not (
            hasattr(grid_result["low"], "is_empty") and grid_result["low"].is_empty
        )

        if grid_has_high and grid_has_low:
            logger.info(
                "Grid sampling method successfully identified variable height areas"
            )
            return grid_result, min_clearance

    # 如果两种方法都没有找到变高度区域，根据整体净空高度判断
    if min_clearance >= threshold:
        logger.info(
            f"No variable height areas detected, overall clearance height {min_clearance:.2f}m >= {threshold}m, classified as high area"
        )
        return {"high": slab_polygon, "low": None}, min_clearance
    else:
        logger.info(
            f"No variable height areas detected, overall clearance height {min_clearance:.2f}m < {threshold}m, classified as low area"
        )
        return {"high": None, "low": slab_polygon}, min_clearance


def get_concave_hull(points, alpha=0.5):
    """创建多点集的凹包

    Args:
        points: 点列表或MultiPoint对象
        alpha: alpha值，控制凹包的紧凑度，值越小越紧凑

    Returns:
        凹包多边形
    """
    from shapely.ops import triangulate
    from shapely.geometry import MultiPoint

    # 确保points是MultiPoint对象
    if isinstance(points, list):
        points = MultiPoint([(p.x, p.y) for p in points])
    elif isinstance(points, np.ndarray):
        points = MultiPoint([(p[0], p[1]) for p in points])

    # 创建三角剖分
    triangles = triangulate(points)

    # 计算每个三角形的外接圆半径
    triangle_radii = []
    for triangle in triangles:
        coords = list(triangle.exterior.coords)
        # 三角形三个顶点
        a = np.array(coords[0])
        b = np.array(coords[1])
        c = np.array(coords[2])

        # 计算边长
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)

        # 计算半周长
        s = (ab + bc + ca) / 2

        # 计算面积
        area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))

        # 计算外接圆半径 R = (abc)/(4*Area)
        if area > 0:
            radius = (ab * bc * ca) / (4 * area)
            triangle_radii.append((triangle, radius))

    # 根据alpha值过滤三角形
    if triangle_radii:
        max_radius = max([r for _, r in triangle_radii])
        filtered_triangles = [t for t, r in triangle_radii if r < alpha * max_radius]

        # 合并所有三角形
        if filtered_triangles:
            concave_hull = unary_union(filtered_triangles)
            if concave_hull.geom_type == "MultiPolygon":
                # 取最大的多边形
                concave_hull = max(concave_hull.geoms, key=lambda x: x.area)
            return concave_hull

    # 如果没有有效三角形，返回凸包
    from scipy.spatial import ConvexHull

    coords = np.array([(p.x, p.y) for p in points.geoms])
    if len(coords) >= 3:
        hull = ConvexHull(coords)
        hull_points = [coords[i] for i in hull.vertices]
        return Polygon(hull_points)

    # 如果点不足以构成多边形，返回空多边形
    return Polygon()


def process_section_with_boundary(
    all_intersect_lines, slab_polygon, slab_max_z, elements_above, threshold
):
    # 添加容差处理
    tolerance = 0.001  # 1mm的容差

    # 对所有线段端点进行容差处理
    processed_lines = []
    for line in all_intersect_lines:
        coords = list(line.coords)
        # 对端点进行四舍五入到指定精度
        processed_coords = [
            (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
            for x, y in coords
        ]
        processed_lines.append(LineString(processed_coords))

    # 合并处理后的线段
    merged_lines = unary_union(processed_lines)

    # 如果没有交线，说明没有与2.2米水平剖面相交
    if merged_lines.is_empty:
        # 在楼板内部随机取两个点
        minx, miny, maxx, maxy = slab_polygon.bounds
        sample_points = []
        attempts = 0
        while len(sample_points) < 2 and attempts < 100:
            x = minx + (maxx - minx) * np.random.random()
            y = miny + (maxy - miny) * np.random.random()
            point = Point(x, y)
            if slab_polygon.contains(point):
                sample_points.append(point)
            attempts += 1

        # 计算采样点的高度并与阈值比较
        heights = []
        for point in sample_points:
            height = calculate_point_height(point, elements_above, slab_max_z)
            clearance = height - slab_max_z
            heights.append(clearance)

        # 判断是否高于阈值
        is_high = all(h >= threshold for h in heights)
        logger.info(
            f"No intersection lines, sampling point heights: {', '.join([f'{h:.2f}m' for h in heights])}"
        )
        return ([slab_polygon], []) if is_high else ([], [slab_polygon])

    # 获取线段数量
    if isinstance(merged_lines, LineString):
        num_lines = 1
    elif isinstance(merged_lines, MultiLineString):
        num_lines = len(list(merged_lines.geoms))
    else:
        num_lines = 0

    logger.info(f"Starting to process {num_lines} intersection lines")

    # 1. 分析每条交线的方向性和高度变化
    line_analysis = []
    for line in all_intersect_lines:
        coords = list(line.coords)
        if len(coords) < 2:
            continue

        # 计算线段的方向向量
        start_point = Point(coords[0])
        end_point = Point(coords[-1])
        dx = end_point.x - start_point.x
        dy = end_point.y - start_point.y
        length = (dx**2 + dy**2) ** 0.5

        if length > 0:
            # 单位向量
            dx /= length
            dy /= length

            # 法向量（垂直于线段）
            nx = -dy
            ny = dx

            # 在线段上均匀取点
            num_samples = 5
            sample_points = []
            for i in range(num_samples):
                t = i / (num_samples - 1)
                x = start_point.x + t * (end_point.x - start_point.x)
                y = start_point.y + t * (end_point.y - start_point.y)
                sample_points.append(Point(x, y))

            # 计算每个采样点的高度
            heights = []
            for point in sample_points:
                height = calculate_point_height(point, elements_above, slab_max_z)
                if height != float("inf"):
                    heights.append(height)

            if heights:
                # 计算高度变化
                height_diff = max(heights) - min(heights)
                avg_height = sum(heights) / len(heights)

                # 在法向量方向上取点
                mid_point = Point(coords[len(coords) // 2])
                offset = 0.1  # 偏移距离
                p1 = Point(mid_point.x + nx * offset, mid_point.y + ny * offset)
                p2 = Point(mid_point.x - nx * offset, mid_point.y - ny * offset)

                # 计算法向量方向上的高度
                h1 = calculate_point_height(p1, elements_above, slab_max_z)
                h2 = calculate_point_height(p2, elements_above, slab_max_z)

                # 记录分析结果
                line_analysis.append(
                    {
                        "line": line,
                        "direction": (dx, dy),
                        "normal": (nx, ny),
                        "mid_point": mid_point,
                        "heights": heights,
                        "height_diff": height_diff,
                        "avg_height": avg_height,
                        "normal_heights": (h1, h2),
                        "normal_diff": h1 - h2,
                    }
                )

    # 2. 根据交线分析结果确定区域类型
    if line_analysis:
        # 计算整体高度变化趋势
        total_height_diff = sum(analysis["height_diff"] for analysis in line_analysis)
        avg_height_diff = total_height_diff / len(line_analysis)
        logger.info(f"Average height change: {avg_height_diff:.2f}m")

        # 确定主要方向
        normal_diffs = [analysis["normal_diff"] for analysis in line_analysis]
        avg_normal_diff = sum(normal_diffs) / len(normal_diffs)
        logger.info(
            f"Normal vector direction average height difference: {avg_normal_diff:.2f}m"
        )

        # 确定高区域方向
        high_direction = 1 if avg_normal_diff > 0 else -1

        # 3. 处理交线形成的内部区域
    inner_polygons = list(polygonize(merged_lines))
    logger.info(
        f"Intersection lines formed {len(inner_polygons)} internal closed areas"
    )

    # 4. 处理与边界形成的区域
    slab_boundary = LineString(list(slab_polygon.exterior.coords))
    all_lines = unary_union([merged_lines, slab_boundary])
    all_polygons = list(polygonize(all_lines))
    logger.info(f"After adding boundaries, formed {len(all_polygons)} areas")

    # 5. 处理每个区域
    high_areas = []
    low_areas = []

    for i, polygon in enumerate(all_polygons):
        if not polygon.is_valid or polygon.is_empty:
            continue

        # 检查是否与楼板相交
        if not polygon.intersects(slab_polygon):
            continue

        # 裁剪与楼板的交集
        valid_area = polygon.intersection(slab_polygon)
        if valid_area.is_empty:
            continue

        # 计算面积占比
        area_ratio = valid_area.area / slab_polygon.area * 100
        if area_ratio < 0.1:  # 忽略太小的区域
            continue

            # 判断区域类型
            # 取区域中心点
            centroid = valid_area.centroid

            # 计算区域中心点与所有交线的关系
            high_count = 0
            low_count = 0
            total_height = 0
            height_count = 0

            for analysis in line_analysis:
                mid_point = analysis["mid_point"]
                normal = analysis["normal"]

                # 计算区域中心点与交线中点的向量
                dx = centroid.x - mid_point.x
                dy = centroid.y - mid_point.y

                # 计算点积，判断区域中心点在法向量的哪一侧
                dot_product = dx * normal[0] + dy * normal[1]

                if dot_product * high_direction > 0:
                    high_count += 1
                else:
                    low_count += 1

                # 计算区域内的平均高度
                if analysis["heights"]:
                    total_height += sum(analysis["heights"])
                    height_count += len(analysis["heights"])

            # 计算区域平均高度
            avg_area_height = total_height / height_count if height_count > 0 else 0
            clearance = avg_area_height - slab_max_z

            # 根据多数和高度判断区域类型
            area_type = "high" if clearance >= threshold else "low"

            # 记录区域信息
            logger.info(f"Area {i+1}:")
            logger.info(f"  Area ratio: {area_ratio:.2f}%")
            logger.info(f"  Average clearance height: {clearance:.2f}m")
            logger.info(f"  High side count: {high_count}, Low side count: {low_count}")
            logger.info(f"  Classified as {area_type} area")

            # 添加到相应列表
            if area_type == "high":
                high_areas.append(valid_area)
            else:
                low_areas.append(valid_area)

    # 统计最终结果
    total_high_area = sum(area.area for area in high_areas)
    total_low_area = sum(area.area for area in low_areas)
    total_area = slab_polygon.area

    logger.info("Final classification results:")
    logger.info(
        f"  High areas: {len(high_areas)} areas, total area ratio: {total_high_area/total_area*100:.2f}%"
    )
    logger.info(
        f"  Low areas: {len(low_areas)} areas, total area ratio: {total_low_area/total_area*100:.2f}%"
    )

    return high_areas, low_areas

    # 如果没有有效的交线分析结果，使用网格采样法
    logger.info(
        "No valid intersection line analysis results, using grid sampling method"
    )

    # 计算合适的网格大小
    bounds = slab_polygon.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    grid_size = min(width, height) / 20  # 将楼板分成20x20的网格

    # 创建采样点网格
    x_coords = np.linspace(bounds[0], bounds[2], 20)
    y_coords = np.linspace(bounds[1], bounds[3], 20)
    grid_points = []

    # 添加边界采样点
    boundary_points = []
    for i in range(len(slab_polygon.exterior.coords) - 1):
        p1 = slab_polygon.exterior.coords[i]
        p2 = slab_polygon.exterior.coords[i + 1]
        # 在边界上添加更密集的点
        num_points = int(
            np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) / (grid_size / 2)
        )
        if num_points > 0:
            t = np.linspace(0, 1, num_points)
            for ti in t:
                x = p1[0] + ti * (p2[0] - p1[0])
                y = p1[1] + ti * (p2[1] - p1[1])
                boundary_points.append((x, y))

    # 合并网格点和边界点
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if slab_polygon.contains(point):
                grid_points.append(point)

    for point in boundary_points:
        if slab_polygon.contains(Point(point)):
            grid_points.append(Point(point))

    # 对每个点计算高度
    high_points = []
    low_points = []

    for point in grid_points:
        height = calculate_point_height(point, elements_above, slab_max_z)
        if height == float("inf"):
            continue

        if height >= threshold:
            high_points.append(point)
        else:
            low_points.append(point)

    # 创建高低区域
    high_areas = []
    low_areas = []

    if high_points:
        # 使用凹包创建高区域
        high_hull = get_concave_hull(high_points)
        if high_hull is not None:
            high_areas.append(high_hull.intersection(slab_polygon))

    if low_points:
        # 使用凹包创建低区域
        low_hull = get_concave_hull(low_points)
        if low_hull is not None:
            low_areas.append(low_hull.intersection(slab_polygon))

    return high_areas, low_areas


def integrate_section_boundaries(
    grid_result, intersect_lines, slab_polygon, elements_above, slab_max_z, threshold
):
    """整合水平剖面法的边界线与网格采样法的结果

    Args:
        grid_result: 网格采样法的结果 {'high': 高区域, 'low': 低区域}
        intersect_lines: 水平剖面法找到的交线
        slab_polygon: 楼板多边形
        elements_above: 楼板上方的元素
        slab_max_z: 楼板顶面高度
        threshold: 净空高度阈值

    Returns:
        整合后的结果 {'high': 高区域, 'low': 低区域}
    """
    from shapely.ops import unary_union, linemerge
    from shapely.geometry import LineString, MultiLineString, Point

    # 如果没有交线或网格采样没有生成有效结果，直接返回网格采样结果
    if not intersect_lines or not grid_result["high"] or not grid_result["low"]:
        return grid_result

    logger.info(
        f"整合水平剖面法的边界线与网格采样法结果 (交线数量: {len(intersect_lines)})"
    )

    # 合并所有交线
    merged_lines = unary_union(intersect_lines)
    # 创建缓冲区表示交线影响区域
    boundary_buffer = merged_lines.buffer(0.1)

    # 提取网格采样结果
    high_area = grid_result["high"]
    low_area = grid_result["low"]

    # 移除交线附近的区域
    high_inner = high_area.difference(boundary_buffer) if high_area else None
    low_inner = low_area.difference(boundary_buffer) if low_area else None

    # 使用交线重新划分边界区域
    if merged_lines and high_inner and low_inner:
        try:
            # 创建直线化的边界区域
            # 为每条边界线创建小缓冲区
            straight_buffer = merged_lines.buffer(0.05)

            # 根据两侧区域的类型划分边界
            high_side = []
            low_side = []

            # 如果是MultiLineString，分别处理每条线
            if isinstance(merged_lines, MultiLineString):
                for line in merged_lines.geoms:
                    buffer = line.buffer(0.05)
                    # 取线两端的点判断所属区域
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        p1 = Point(coords[0][0], coords[0][1])
                        p2 = Point(coords[-1][0], coords[-1][1])

                        # 向线的法线方向偏移一点取样
                        dx = coords[-1][0] - coords[0][0]
                        dy = coords[-1][1] - coords[0][1]
                        length = (dx**2 + dy**2) ** 0.5
                        if length > 0:
                            nx, ny = -dy / length, dx / length  # 法线向量

                            p1_side1 = Point(p1.x + nx * 0.1, p1.y + ny * 0.1)
                            p1_side2 = Point(p1.x - nx * 0.1, p1.y - ny * 0.1)
                            p2_side1 = Point(p2.x + nx * 0.1, p2.y + ny * 0.1)
                            p2_side2 = Point(p2.x - nx * 0.1, p2.y - ny * 0.1)

                            # 判断点所在的区域类型
                            points = [p1_side1, p1_side2, p2_side1, p2_side2]
                            heights = [
                                calculate_point_height(p, elements_above, slab_max_z)
                                for p in points
                            ]

                            # 根据多数点的高度确定线段所属
                            high_count = sum(
                                1 for h in heights if h - slab_max_z >= threshold
                            )
                            if high_count >= 2:
                                high_side.append(buffer)
                            else:
                                low_side.append(buffer)
            else:
                # 单线段情况类似处理...
                high_side.append(straight_buffer)

            # 合并边界区域
            high_boundary = unary_union(high_side) if high_side else None
            low_boundary = unary_union(low_side) if low_side else None

            # 组合内部区域和边界区域
            final_high = (
                unary_union([high_inner, high_boundary])
                if high_boundary
                else high_inner
            )
            final_low = (
                unary_union([low_inner, low_boundary]) if low_boundary else low_inner
            )

            # 确保与原始楼板相交
            final_high = final_high.intersection(slab_polygon) if final_high else None
            final_low = final_low.intersection(slab_polygon) if final_low else None

            # 确保区域有效
            if (
                final_high
                and not final_high.is_empty
                and final_low
                and not final_low.is_empty
            ):
                logger.info("成功整合水平剖面法边界与网格采样结果")
                return {"high": final_high, "low": final_low}

        except Exception as e:
            logger.warning(f"整合边界失败: {e}")

    return grid_result


def analyze_using_grid_sampling(
    slab_geometry, elements_above, threshold=2.2, fine_for_roof=False
):
    """使用网格采样方法分析楼板高度变化

    Args:
        slab_geometry: 楼板几何信息
        elements_above: 楼板上方元素列表
        threshold: 高度阈值
        fine_for_roof: 是否对屋顶进行精细分析

    Returns:
        包含高低区域分类的字典
    """
    slab_polygon = slab_geometry["top_polygon"]
    slab_max_z = slab_geometry["max_z"]

    # 计算合适的网格大小
    bounds = slab_polygon.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    grid_size = min(width, height) / 20  # 将楼板分成20x20的网格

    # 创建采样点网格
    x_coords = np.linspace(bounds[0], bounds[2], 20)
    y_coords = np.linspace(bounds[1], bounds[3], 20)
    grid_points = []

    # 添加边界采样点
    boundary_points = []
    for i in range(len(slab_polygon.exterior.coords) - 1):
        p1 = slab_polygon.exterior.coords[i]
        p2 = slab_polygon.exterior.coords[i + 1]
        # 在边界上添加更密集的点
        num_points = int(
            np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) / (grid_size / 2)
        )
        if num_points > 0:
            t = np.linspace(0, 1, num_points)
            for ti in t:
                x = p1[0] + ti * (p2[0] - p1[0])
                y = p1[1] + ti * (p2[1] - p1[1])
                boundary_points.append((x, y))

    # 合并网格点和边界点
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if slab_polygon.contains(point):
                grid_points.append(point)

    for point in boundary_points:
        if slab_polygon.contains(Point(point)):
            grid_points.append(Point(point))

    # 对每个点计算高度
    high_points = []
    low_points = []

    for point in grid_points:
        height = calculate_point_height(point, elements_above, slab_max_z)
        if height == float("inf"):
            continue

        if height >= threshold:
            high_points.append(point)
        else:
            low_points.append(point)

    # 创建高低区域
    result = {"high": None, "low": None}

    if high_points:
        # 使用凹包创建高区域
        high_hull = get_concave_hull(high_points)
        if high_hull is not None:
            result["high"] = high_hull.intersection(slab_polygon)

    if low_points:
        # 使用凹包创建低区域
        low_hull = get_concave_hull(low_points)
        if low_hull is not None:
            result["low"] = low_hull.intersection(slab_polygon)

    # 处理边界问题
    if result["high"] is not None:
        result["high"] = align_boundary(result["high"], slab_polygon, grid_size)

    if result["low"] is not None:
        result["low"] = align_boundary(result["low"], slab_polygon, grid_size)

    # 直线化内部边界
    if result["high"] is not None and result["low"] is not None:
        result["high"], result["low"] = straighten_inner_boundary(
            result["high"], result["low"], grid_size
        )

    # 平滑分类结果
    result = smooth_classification_result(result, slab_polygon)

    return result


def calculate_point_height(point, elements_above, slab_max_z):
    """计算指定点上方的最低高度点

    Args:
        point: 要检查的点(Point对象)
        elements_above: 上方元素列表
        slab_max_z: 楼板顶面高度

    Returns:
        point上方的最低元素高度，如无元素则返回float('inf')
    """
    min_height = float("inf")

    for element in elements_above:
        element_faces = element.get("faces", [])

        for face in element_faces:
            # 在新结构中，face直接是三个点的坐标
            if len(face) < 3:
                continue

            v1, v2, v3 = face[0:3]  # 确保我们只取前三个点

            # 检查点是否在三角形的水平投影下方
            try:
                tri_polygon = Polygon([(v1[0], v1[1]), (v2[0], v2[1]), (v3[0], v3[1])])

                if tri_polygon.contains(point):
                    # 计算点正上方的高度 - 使用重心插值
                    try:
                        # 三角形法向量的计算
                        normal_x = (v2[1] - v1[1]) * (v3[2] - v1[2]) - (
                            v3[1] - v1[1]
                        ) * (v2[2] - v1[2])
                        normal_y = (v2[2] - v1[2]) * (v3[0] - v1[0]) - (
                            v3[2] - v1[2]
                        ) * (v2[0] - v1[0])
                        normal_z = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (
                            v3[0] - v1[0]
                        ) * (v2[1] - v1[1])

                        if abs(normal_z) < 1e-6:  # 避免除以接近零的值
                            continue

                        # 平面方程: ax + by + cz + d = 0
                        a, b, c = normal_x, normal_y, normal_z
                        d = -(a * v1[0] + b * v1[1] + c * v1[2])

                        # 计算交点高度 z = -(ax + by + d)/c
                        z_intersect = -(a * point.x + b * point.y + d) / c

                        # 确保交点在楼板上方
                        if z_intersect > slab_max_z:
                            min_height = min(min_height, z_intersect)
                    except Exception as e:
                        continue
            except Exception:
                # 如果创建多边形失败，跳过此面
                continue

    return min_height


def calculate_area_m2(geometry):
    """计算几何对象的面积（平方米）"""
    if geometry is None or geometry.is_empty:
        return 0.0
    return geometry.area


def classify_slabs_by_height(file_path, threshold=2.2):
    """根据净空高度分类楼板"""
    # 加载IFC文件
    ifc_file = load_ifc_file(file_path)

    # 获取所有楼层
    storeys = get_all_storeys(ifc_file)
    storeys_by_elevation = {storey_id: data for storey_id, data in storeys.items()}

    # 按标高排序楼层
    sorted_storeys = sorted(
        storeys_by_elevation.items(), key=lambda x: x[1]["elevation"]
    )

    # 获取所有楼板
    all_slabs = get_all_slabs(ifc_file)

    # 获取所有元素几何信息（一次性获取以提高性能）
    logger.info("开始获取所有元素的几何信息...")
    all_elements_geometry = get_all_elements_geometry(ifc_file)
    logger.info(f"获取了 {len(all_elements_geometry)} 个元素的几何信息")

    # 按楼层分类楼板
    results_by_storey = {}

    # 初始化每层的分类结果
    for storey_id, storey_data in storeys.items():
        results_by_storey[storey_id] = {
            "name": storey_data["name"],
            "elevation": storey_data["elevation"],
            "high_slabs": [],
            "low_slabs": [],
            "variable_height_slabs": [],
        }

    # 对每个楼板进行分析
    for slab_id, slab_data in all_slabs.items():
        # 找出楼板所在的楼层
        storey_id = slab_data["storey_id"]
        storey_data = storeys.get(storey_id)

        if not storey_data:
            logger.warning(f"找不到楼板 {slab_data['name']} 所在的楼层")
            continue

        # 找出下一层楼层的标高（如果有）
        next_level_elevation = None
        for idx, (curr_storey_id, curr_storey_data) in enumerate(sorted_storeys):
            if curr_storey_id == storey_id and idx < len(sorted_storeys) - 1:
                next_level_elevation = sorted_storeys[idx + 1][1]["elevation"]
                break

        # 找出楼板上方的元素 - 使用改进的逻辑
        logger.info(f"使用改进的高度计算逻辑分析楼板 {slab_data['name']}")
        elements_above, min_clearance = find_elements_above_slab(
            ifc_file, slab_data, all_elements_geometry, next_level_elevation
        )

        # 使用新的分析方法
        logger.info(f"分析楼板 {slab_data['name']} 的高度变化")
        height_result, min_clearance = analyze_slab_height(
            slab_data, elements_above, threshold
        )

        # 根据分析结果分类
        if height_result["high"] and height_result["low"]:
            # 变高度区域
            results_by_storey[storey_id]["variable_height_slabs"].append(
                {
                    "id": slab_id,
                    "name": slab_data["name"],
                    "geometry": slab_data,
                    "min_clearance": min_clearance,
                    "high_polygon": height_result["high"],
                    "low_polygon": height_result["low"],
                }
            )
        elif height_result["high"]:
            # 高区域
            results_by_storey[storey_id]["high_slabs"].append(
                {
                    "id": slab_id,
                    "name": slab_data["name"],
                    "geometry": slab_data,
                    "min_clearance": min_clearance,
                }
            )
        else:
            # 低区域
            results_by_storey[storey_id]["low_slabs"].append(
                {
                    "id": slab_id,
                    "name": slab_data["name"],
                    "geometry": slab_data,
                    "min_clearance": min_clearance,
                }
            )

    # 处理分类结果，添加面积信息
    results_by_storey = process_classification_results(results_by_storey)

    return results_by_storey


def get_all_storeys(ifc_file):
    """获取所有楼层信息，返回以楼层ID为键的字典"""
    storeys = {}
    for storey in ifc_file.by_type("IFCBUILDINGSTOREY"):
        storey_id = storey.id()
        elevation = get_storey_elevation(storey)
        name = (
            storey.Name
            if hasattr(storey, "Name") and storey.Name
            else f"Level {len(storeys) + 1}"
        )

        # 将中文"标高"替换为"Level"以避免字体显示问题
        if "标高" in name:
            name = name.replace("标高", "Level")

        storeys[storey_id] = {
            "id": storey_id,
            "name": name,
            "elevation": elevation,
            "object": storey,
        }

    return storeys


def get_all_slabs(ifc_file):
    """获取所有楼板及其几何信息，返回以楼板ID为键的字典"""
    slabs = {}
    ifc_slabs = ifc_file.by_type("IFCSLAB")

    # 获取所有楼层
    storeys = get_all_storeys(ifc_file)
    storeys_by_elevation = sorted(
        [(k, v["elevation"]) for k, v in storeys.items()], key=lambda x: x[1]
    )

    for slab in ifc_slabs:
        slab_id = slab.id()
        slab_geometry = get_slab_geometry(ifc_file, slab)

        if not slab_geometry:
            continue

        # 确定楼板所在楼层
        storey_id = find_slab_storey(slab, slab_geometry, storeys, storeys_by_elevation)

        if storey_id:
            slab_geometry["storey_id"] = storey_id
            slabs[slab_id] = slab_geometry

    return slabs


def find_slab_storey(slab, slab_geometry, storeys, storeys_by_elevation):
    """确定楼板所在的楼层"""
    # 方法1：通过ObjectPlacement关联
    slab_placement = slab.ObjectPlacement
    for storey_id, storey_data in storeys.items():
        storey = storey_data["object"]
        storey_placement = storey.ObjectPlacement
        if slab_placement == storey_placement or (
            slab_placement
            and storey_placement
            and hasattr(slab_placement, "PlacementRelTo")
            and slab_placement.PlacementRelTo == storey_placement
        ):
            return storey_id

    # 方法2：通过高度判断
    slab_z = (slab_geometry["min_z"] + slab_geometry["max_z"]) / 2

    for i, (storey_id, elevation) in enumerate(storeys_by_elevation):
        if i < len(storeys_by_elevation) - 1:
            next_elevation = storeys_by_elevation[i + 1][1]
            if elevation <= slab_z < next_elevation:
                return storey_id
        else:
            # 对于最高层
            if elevation <= slab_z:
                return storey_id

    return None


def get_all_elements_geometry(ifc_file):
    """获取所有可能影响净空高度的元素的几何信息"""
    element_types = [
        "IFCROOF",
        "IFCSLAB",
        "IFCCOVERING",
    ]
    all_elements_geometry = {}

    for element_type in element_types:
        elements = ifc_file.by_type(element_type)
        for element in elements:
            geometry = get_object_geometry(ifc_file, element)
            if geometry:
                all_elements_geometry[element.id()] = geometry

    return all_elements_geometry


def align_boundary(area, slab_polygon, grid_size):
    """将区域边界与楼板边界对齐"""
    if area is None or area.is_empty:
        return area

    # 确保区域在楼板内部
    area = area.intersection(slab_polygon)

    # 获取区域边界
    boundary = area.boundary

    # 对每个边界点进行对齐
    aligned_points = []
    for point in boundary.coords:
        # 找到最近的网格点
        x = round(point[0] / grid_size) * grid_size
        y = round(point[1] / grid_size) * grid_size
        aligned_points.append((x, y))

    # 创建新的多边形
    try:
        aligned_area = Polygon(aligned_points)
        # 确保结果仍然在楼板内部
        return aligned_area.intersection(slab_polygon)
    except:
        return area


def straighten_inner_boundary(high_area, low_area, grid_size):
    """将内部边界直线化"""
    if high_area is None or low_area is None:
        return high_area, low_area

    # 获取两个区域的边界
    high_boundary = high_area.boundary
    low_boundary = low_area.boundary

    # 找到共同的边界点
    common_points = []
    for point in high_boundary.coords:
        if low_boundary.distance(Point(point)) < grid_size / 2:
            common_points.append(point)

    if not common_points:
        return high_area, low_area

    # 使用最小二乘法拟合直线
    x = np.array([p[0] for p in common_points])
    y = np.array([p[1] for p in common_points])

    # 计算主方向
    cov_matrix = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    main_direction = eigenvectors[:, np.argmax(eigenvalues)]

    # 创建直线
    line = LineString(
        [
            (min(x) - 10 * main_direction[0], min(y) - 10 * main_direction[1]),
            (max(x) + 10 * main_direction[0], max(y) + 10 * main_direction[1]),
        ]
    )

    # 将直线投影到楼板边界
    slab_polygon = high_area.union(low_area).convex_hull
    line = line.intersection(slab_polygon)

    # 使用直线分割区域
    high_side = line.buffer(grid_size / 2)
    low_side = line.buffer(-grid_size / 2)

    # 更新区域
    new_high = high_area.intersection(high_side)
    new_low = low_area.intersection(low_side)

    return new_high, new_low


def smooth_classification_result(result, slab_polygon):
    """平滑分类结果"""
    if result["high"] is not None:
        result["high"] = result["high"].buffer(-0.1).buffer(0.2)
        result["high"] = result["high"].intersection(slab_polygon)

    if result["low"] is not None:
        result["low"] = result["low"].buffer(-0.1).buffer(0.2)
        result["low"] = result["low"].intersection(slab_polygon)

    return result


if __name__ == "__main__":
    # 使用示例
    ifc_file_path = "overall.ifc"
    results_by_storey = classify_slabs_by_height(ifc_file_path)

    # 打印面积统计信息
    print_area_statistics(results_by_storey)

    # 生成详细报告
    report = generate_area_report(results_by_storey)

    # 将报告保存到文件
    with open("area_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 可视化结果
    from visualization import (
        visualize_classification_by_storey,
        create_height_heatmap,
    )

    # 生成分类图
    visualize_classification_by_storey(results_by_storey)

    # 生成热力图
    create_height_heatmap(results_by_storey)

    # 避免交互式显示导致的卡住问题
    print("Images and reports have been saved:")
    print("- 'slab_classification_by_storey.png' - Shows high/low area classification")
    print("- 'height_heatmap.png' - Shows clearance height heatmap")
    print("- 'area_report.txt' - Area statistics report")
    plt.close("all")  # 关闭所有图形，避免交互式显示
