import os
import sys
import numpy as np
import ifcopenshell
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.font_manager as font_manager
import logging
import ifcopenshell.util.placement

# 配置logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 将父目录添加到路径中以便导入
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# 导入ifc_loader模块
# from boolean.ifc_loader import IFCLoader


def get_entity_id(wall):
    """从墙体对象获取实体ID"""
    # 尝试不同方法获取ID
    try:
        # 如果wall是IfcEntity对象
        return wall.id()
    except (AttributeError, TypeError):
        try:
            # 尝试从字符串表示中提取ID
            wall_str = str(wall)
            if "#" in wall_str:
                # 提取形如 "#123=..." 中的数字部分
                id_str = wall_str.split("#")[1].split("=")[0]
                return int(id_str)
        except (IndexError, ValueError):
            pass

    # 如果上述方法都失败，返回None
    return None


def plot_walls_and_outline(wall_geometries, external_outline=None, external_walls=None):
    """绘制墙体和外轮廓，使用统一的处理方式"""
    # 配置中文字体支持
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "STHeiti Light"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    plt.figure(figsize=(10, 10))

    # 绘制所有墙体
    for wall_id, wall_geom in wall_geometries.items():
        if isinstance(wall_geom, dict) and "start" in wall_geom and "end" in wall_geom:
            start = wall_geom["start"]
            end = wall_geom["end"]

            # 根据是否为外墙选择不同颜色
            if external_walls and wall_id in external_walls:
                color = "red"  # 外墙用红色表示
                linewidth = 2
            else:
                color = "blue"  # 内墙用蓝色表示
                linewidth = 1

            # 检查是否为弧形墙体
            if (
                "is_curved" in wall_geom
                and wall_geom["is_curved"]
                and "curve_points" in wall_geom
            ):
                # 绘制弧形墙体
                curve_points = wall_geom["curve_points"]
                if len(curve_points) > 2:
                    # 对曲线点进行过滤和平滑处理
                    filtered_points = filter_curve_points(curve_points)

                    # 确保有足够的点绘制曲线
                    if len(filtered_points) > 2:
                        # 使用过滤后的点绘制曲线
                        xs = [p[0] for p in filtered_points]
                        ys = [p[1] for p in filtered_points]
                        plt.plot(xs, ys, color=color, linewidth=linewidth)

                        # 显示墙体ID在曲线中点位置
                        mid_idx = len(filtered_points) // 2
                        mid_point = filtered_points[mid_idx]
                        plt.text(
                            mid_point[0],
                            mid_point[1],
                            str(wall_id),
                            fontsize=8,
                            ha="center",
                            va="center",
                            color="black",
                            bbox=dict(facecolor="white", alpha=0.7),
                        )
                    else:
                        # 如果过滤后点数不足，退回到直线绘制
                        plt.plot(
                            [start[0], end[0]],
                            [start[1], end[1]],
                            color=color,
                            linewidth=linewidth,
                        )
                        plt.text(
                            (start[0] + end[0]) / 2,
                            (start[1] + end[1]) / 2,
                            str(wall_id),
                            fontsize=8,
                            ha="center",
                            va="center",
                            color="black",
                            bbox=dict(facecolor="white", alpha=0.7),
                        )
                else:
                    # 如果曲线点不足，退回到直线绘制
                    plt.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        color=color,
                        linewidth=linewidth,
                    )
                    plt.text(
                        (start[0] + end[0]) / 2,
                        (start[1] + end[1]) / 2,
                        str(wall_id),
                        fontsize=8,
                        ha="center",
                        va="center",
                        color="black",
                        bbox=dict(facecolor="white", alpha=0.7),
                    )
            else:
                # 绘制直线墙体
                plt.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=color,
                    linewidth=linewidth,
                )
                plt.text(
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2,
                    str(wall_id),
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.7),
                )

    # 绘制外轮廓
    if external_outline:
        if hasattr(external_outline, "exterior"):
            # 绘制外轮廓的外边界曲线
            xs, ys = external_outline.exterior.xy
            plt.plot(xs, ys, "g-", linewidth=2)

            # 绘制外轮廓的内部边界（如果有）
            if hasattr(external_outline, "interiors"):
                for interior in external_outline.interiors:
                    int_xs, int_ys = interior.xy
                    plt.plot(int_xs, int_ys, "g--", linewidth=1)

    plt.title("Walls and External Outline")
    plt.axis("equal")
    plt.grid(True)

    # 保存图像
    plt.savefig("walls_and_outline.png")
    print(f"Saved wall and outline image to 'walls_and_outline.png'")

    # 显示图像
    plt.show()


def filter_curve_points(curve_points, tolerance=0.05):
    """过滤和简化曲线点集，去除重复点和异常点

    Args:
        curve_points: 原始曲线点集
        tolerance: 容差值，用于确定点是否重复

    Returns:
        list: 过滤后的点集
    """
    if len(curve_points) <= 2:
        return curve_points

    # 过滤重复或过近的点
    filtered = [curve_points[0]]  # 保留第一个点

    for i in range(1, len(curve_points)):
        last_point = filtered[-1]
        current = curve_points[i]

        # 计算与上一点的距离
        distance = np.sqrt(
            (current[0] - last_point[0]) ** 2 + (current[1] - last_point[1]) ** 2
        )

        # 距离大于容差才保留
        if distance > tolerance:
            filtered.append(current)

    # 确保至少保留起点和终点
    if len(filtered) == 1 and len(curve_points) > 1:
        filtered.append(curve_points[-1])

    return filtered


def smooth_curve_points(points, window=3):
    """对曲线点进行平滑处理并过滤异常点

    Args:
        points: 曲线点列表
        window: 平滑窗口大小

    Returns:
        list: 平滑后的曲线点
    """
    if len(points) <= window:
        return points

    # 首先去除异常跳跃点
    filtered_points = []
    filtered_points.append(points[0])  # 保留第一个点

    for i in range(1, len(points) - 1):
        # 计算到前后点的距离
        prev_dist = (
            (points[i][0] - points[i - 1][0]) ** 2
            + (points[i][1] - points[i - 1][1]) ** 2
        ) ** 0.5
        next_dist = (
            (points[i][0] - points[i + 1][0]) ** 2
            + (points[i][1] - points[i + 1][1]) ** 2
        ) ** 0.5

        # 如果与前后点的距离过大，可能是异常点
        max_allowed_dist = 2.0  # 设置合理的阈值
        if prev_dist < max_allowed_dist and next_dist < max_allowed_dist:
            filtered_points.append(points[i])

    filtered_points.append(points[-1])  # 保留最后一个点

    # 再进行平滑处理
    smoothed = []
    smoothed.append(filtered_points[0])  # 保留首点不变

    # 对中间点进行移动平均平滑
    for i in range(1, len(filtered_points) - 1):
        # 确定窗口范围
        start = max(0, i - window // 2)
        end = min(len(filtered_points), i + window // 2 + 1)
        window_points = filtered_points[start:end]

        # 计算窗口内点的平均值
        x_avg = sum(p[0] for p in window_points) / len(window_points)
        y_avg = sum(p[1] for p in window_points) / len(window_points)

        smoothed.append((x_avg, y_avg))

    smoothed.append(filtered_points[-1])  # 保留尾点不变

    # 对于长度超过一定值的曲线进行降采样，但确保保留形状特征
    if len(smoothed) > 25:
        # 使用Douglas-Peucker算法进行简化
        # 创建临时LineString
        from shapely.geometry import LineString

        line = LineString(smoothed)
        # 简化曲线，但保留关键特征点
        simplified_line = line.simplify(0.1, preserve_topology=True)
        # 转回点列表
        smoothed = list(simplified_line.coords)

    return smoothed


def create_curved_wall_polygon(curve_points, thickness):
    """根据曲线点和厚度创建弧形墙体多边形，改进版本

    Args:
        curve_points: 曲线点列表
        thickness: 墙体厚度

    Returns:
        Polygon: 弧形墙体多边形
    """
    if len(curve_points) < 3:
        return None

    # 确保曲线点是连续的，不含异常跳跃
    filtered_points = []
    filtered_points.append(curve_points[0])

    for i in range(1, len(curve_points)):
        # 计算与前一点的距离
        dist = (
            (curve_points[i][0] - filtered_points[-1][0]) ** 2
            + (curve_points[i][1] - filtered_points[-1][1]) ** 2
        ) ** 0.5

        # 如果距离合理，添加此点
        if dist < 1.0:  # 使用更小的阈值以过滤掉更多异常点
            filtered_points.append(curve_points[i])

    # 如果过滤后点数太少，返回None
    if len(filtered_points) < 3:
        return None

    # 创建曲线的偏移多边形
    half_thickness = thickness / 2.0

    # 创建内外轮廓点
    inner_points = []
    outer_points = []

    for i in range(len(filtered_points)):
        # 计算每点的法向量
        if i == 0:
            # 第一个点
            dx = filtered_points[1][0] - filtered_points[0][0]
            dy = filtered_points[1][1] - filtered_points[0][1]
        elif i == len(filtered_points) - 1:
            # 最后一个点
            dx = filtered_points[-1][0] - filtered_points[-2][0]
            dy = filtered_points[-1][1] - filtered_points[-2][1]
        else:
            # 中间点，使用前后点的平均法向量
            dx1 = filtered_points[i][0] - filtered_points[i - 1][0]
            dy1 = filtered_points[i][1] - filtered_points[i - 1][1]
            dx2 = filtered_points[i + 1][0] - filtered_points[i][0]
            dy2 = filtered_points[i + 1][1] - filtered_points[i][1]

            # 使用平均向量
            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2

        # 计算单位法向量
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx = -dy / length
            ny = dx / length
        else:
            # 如果长度为0，使用默认法向量
            nx, ny = 0, 1

        # 创建内外轮廓点
        inner_points.append(
            (
                filtered_points[i][0] + nx * half_thickness,
                filtered_points[i][1] + ny * half_thickness,
            )
        )
        outer_points.append(
            (
                filtered_points[i][0] - nx * half_thickness,
                filtered_points[i][1] - ny * half_thickness,
            )
        )

    # 合并点集形成完整多边形，外点需要逆序
    polygon_points = inner_points + outer_points[::-1]

    try:
        poly = Polygon(polygon_points)
        if poly.is_valid:
            return poly
        else:
            # 尝试修复无效多边形
            return poly.buffer(0)
    except Exception as e:
        print(f"创建弧形墙体多边形时出错: {str(e)}")
        return None


def create_simple_curved_wall_polygon(start_point, end_point, thickness):
    """创建简化的弧形墙体多边形（矩形近似）

    Args:
        start_point: 起点
        end_point: 终点
        thickness: 墙体厚度

    Returns:
        Polygon: 简化的墙体多边形
    """
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # 计算法向量
    if abs(dx) + abs(dy) > 0:
        length = np.sqrt(dx**2 + dy**2)
        nx, ny = -dy / length, dx / length
    else:
        nx, ny = 0, 1

    half_thickness = thickness / 2
    p1 = (start_point[0] + nx * half_thickness, start_point[1] + ny * half_thickness)
    p2 = (start_point[0] - nx * half_thickness, start_point[1] - ny * half_thickness)
    p3 = (end_point[0] - nx * half_thickness, end_point[1] - ny * half_thickness)
    p4 = (end_point[0] + nx * half_thickness, end_point[1] + ny * half_thickness)

    try:
        return Polygon([p1, p2, p3, p4])
    except Exception:
        return None


def plot_merged_polygon(merged_polygon):
    """单独绘制合并后的多边形

    Args:
        merged_polygon: 合并后的多边形
    """
    plt.figure(figsize=(10, 10))

    if isinstance(merged_polygon, Polygon):
        # 绘制多边形外边界
        xs, ys = merged_polygon.exterior.xy
        plt.plot(xs, ys, "r-", linewidth=2, label="Merged Polygon Boundary")

        # 绘制多边形内部边界（如果有）
        if hasattr(merged_polygon, "interiors"):
            for i, interior in enumerate(merged_polygon.interiors):
                int_xs, int_ys = interior.xy
                plt.plot(
                    int_xs,
                    int_ys,
                    "r--",
                    linewidth=1,
                    label=f"Interior Boundary {i+1}" if i == 0 else "",
                )

    elif isinstance(merged_polygon, MultiPolygon):
        # 绘制多个多边形
        for i, poly in enumerate(merged_polygon.geoms):
            xs, ys = poly.exterior.xy
            plt.plot(
                xs, ys, "r-", linewidth=2, label=f"Polygon {i+1}" if i == 0 else ""
            )

            # 绘制内部边界
            if hasattr(poly, "interiors"):
                for j, interior in enumerate(poly.interiors):
                    int_xs, int_ys = interior.xy
                    plt.plot(int_xs, int_ys, "r--", linewidth=1)

    plt.title("Merged Polygon")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # 保存图像
    plt.savefig("merged_polygon.png")
    print(f"Saved merged polygon image to 'merged_polygon.png'")

    # 显示图像
    plt.show()


def validate_outline(outline, min_area=0.1, min_length=1.0):
    """验证外轮廓的有效性

    Args:
        outline: 外轮廓多边形
        min_area: 最小面积阈值
        min_length: 最小周长阈值

    Returns:
        bool: 外轮廓是否有效
    """
    if not outline or not outline.is_valid:
        logger.warning("外轮廓无效")
        return False

    if outline.area < min_area:
        logger.warning(f"外轮廓面积过小: {outline.area}")
        return False

    if outline.length < min_length:
        logger.warning(f"外轮廓周长过小: {outline.length}")
        return False

    return True


def repair_geometry(polygon, buffer_distance=0.001, simplify_tolerance=0.01):
    """修复和优化几何形状

    Args:
        polygon: 输入多边形
        buffer_distance: 缓冲区距离
        simplify_tolerance: 简化容差

    Returns:
        Polygon: 修复后的多边形
    """
    try:
        if not polygon.is_valid:
            # 使用缓冲区修复自相交
            polygon = polygon.buffer(buffer_distance)

        # 简化复杂边界
        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

        # 确保多边形有效
        if not polygon.is_valid:
            logger.warning("几何修复后多边形仍然无效")
            return None

        return polygon
    except Exception as e:
        logger.error(f"几何修复失败: {str(e)}")
        return None


def calculate_intersection_ratio(wall_poly, outline_boundary):
    """计算墙体与轮廓边界的相交比例

    Args:
        wall_poly: 墙体多边形
        outline_boundary: 轮廓边界

    Returns:
        float: 相交比例
    """
    try:
        if not wall_poly.intersects(outline_boundary):
            return 0.0

        intersection = wall_poly.intersection(outline_boundary)
        if not hasattr(intersection, "length"):
            return 0.0

        wall_length = wall_poly.length
        if wall_length == 0:
            return 0.0

        return intersection.length / wall_length
    except Exception as e:
        logger.error(f"计算相交比例失败: {str(e)}")
        return 0.0


def get_root_object_placement(obj):
    """递归获取ObjectPlacement的最顶层对象"""
    placement = getattr(obj, "ObjectPlacement", None)
    while (
        placement and hasattr(placement, "PlacementRelTo") and placement.PlacementRelTo
    ):
        placement = placement.PlacementRelTo
    return placement


def group_walls_by_storey_objectplacement_recursive(ifc_file):
    storeys = ifc_file.by_type("IfcBuildingStorey")
    for s in storeys:
        print(f"楼层ID: {s.GlobalId}, 标高: {getattr(s, 'Elevation', None)}")
    storey_placements = {}
    for storey in storeys:
        root_placement = get_root_object_placement(storey)
        if root_placement:
            storey_placements[root_placement.id()] = {
                "storey": storey,
                "elevation": getattr(storey, "Elevation", 0.0),
                "walls": [],
            }

    # 获取所有墙体
    walls = []
    walls.extend(ifc_file.by_type("IfcWall"))
    walls.extend(ifc_file.by_type("IfcWallStandardCase"))
    walls.extend(ifc_file.by_type("IfcCurtainWall"))
    walls.extend(ifc_file.by_type("IfcPlate"))
    walls.extend(ifc_file.by_type("IfcCovering"))
    walls.extend(ifc_file.by_type("IfcMember"))

    for wall in walls:
        root_placement = get_root_object_placement(wall)
        if root_placement and root_placement.id() in storey_placements:
            storey_placements[root_placement.id()]["walls"].append(wall)

    # 按楼层标高分组返回
    result = {}
    for info in storey_placements.values():
        elevation = round(info["elevation"], 3)
        if elevation not in result:
            result[elevation] = []
        result[elevation].extend(info["walls"])
    return result


def group_walls_by_z(ifc_file, z_tolerance=1000):
    """按Z坐标分组墙体，z_tolerance单位为mm（如1000=1米）"""
    walls = []
    walls.extend(ifc_file.by_type("IfcWall"))
    walls.extend(ifc_file.by_type("IfcWallStandardCase"))
    walls.extend(ifc_file.by_type("IfcCurtainWall"))
    walls.extend(ifc_file.by_type("IfcPlate"))
    walls.extend(ifc_file.by_type("IfcCovering"))
    walls.extend(ifc_file.by_type("IfcMember"))
    result = {}
    for wall in walls:
        placement = getattr(wall, "ObjectPlacement", None)
        z = 0.0
        if placement and hasattr(placement, "RelativePlacement"):
            loc = placement.RelativePlacement.Location
            if loc and hasattr(loc, "Coordinates"):
                coords = loc.Coordinates
                if len(coords) > 2:
                    z = coords[2]
        # 按z_tolerance分组
        z_key = round(z / z_tolerance) * z_tolerance
        if z_key not in result:
            result[z_key] = []
        result[z_key].append(wall)
    return result


def group_walls_by_global_z(ifc_file, z_tolerance=1000):
    """按全局Z坐标分组墙体，z_tolerance单位为mm（如1000=1米）"""
    walls = []
    walls.extend(ifc_file.by_type("IfcWall"))
    walls.extend(ifc_file.by_type("IfcWallStandardCase"))
    walls.extend(ifc_file.by_type("IfcCurtainWall"))
    walls.extend(ifc_file.by_type("IfcPlate"))
    walls.extend(ifc_file.by_type("IfcCovering"))
    walls.extend(ifc_file.by_type("IfcMember"))
    # 按id去重，避免重复计数
    unique_walls = {}
    for wall in walls:
        unique_walls[wall.id()] = wall
    walls = list(unique_walls.values())
    result = {}
    for wall in walls:
        try:
            matrix = ifcopenshell.util.placement.get_local_placement(
                wall.ObjectPlacement
            )
            z = matrix[2][3] if matrix is not None else 0.0
        except Exception:
            z = 0.0
        z_key = round(z / z_tolerance) * z_tolerance
        # 插入调试打印
        print(f"墙体ID: {wall.id()} 全局Z: {z}，分组key: {z_key}")
        if z_key not in result:
            result[z_key] = []
        result[z_key].append(wall)
    return result


def get_external_wall_outline(ifc_file, wall_thickness=None):
    """使用布尔运算获取外墙轮廓，按楼层分别计算"""
    # 优先用结构分组
    walls_by_elevation = group_walls_by_storey_objectplacement_recursive(ifc_file)
    if len(walls_by_elevation) == 1:
        logger.info("结构分组只有一个楼层，自动切换为Z坐标分组")
        walls_by_elevation = group_walls_by_z(ifc_file)
    if len(walls_by_elevation) == 1:
        logger.info("局部Z分组只有一个楼层，自动切换为全局Z坐标分组")
        walls_by_elevation = group_walls_by_global_z(ifc_file)
    logger.info(f"墙体按标高分组: {len(walls_by_elevation)} 个楼层")

    # 计算标准墙厚度（如果未提供）
    all_walls = []
    for wall_list in walls_by_elevation.values():
        all_walls.extend(wall_list)
    if wall_thickness is None:
        thickness_values = []
        for wall in all_walls:
            try:
                if not wall.Representation:
                    continue

                settings = ifcopenshell.geom.settings()
                settings.set(settings.USE_WORLD_COORDS, True)
                shape = ifcopenshell.geom.create_shape(settings, wall)

                if not shape or not shape.geometry:
                    continue

                verts = shape.geometry.verts
                if not verts or len(verts) < 3:
                    continue

                # 提取坐标范围
                x_coords = [verts[i] for i in range(0, len(verts), 3)]
                y_coords = [verts[i + 1] for i in range(0, len(verts), 3)]

                # 计算包围盒
                min_x = min(x_coords)
                max_x = max(x_coords)
                min_y = min(y_coords)
                max_y = max(y_coords)

                # 估算厚度（取短边）
                width = max_x - min_x
                height = max_y - min_y

                if width > 0 and height > 0:
                    thickness = min(width, height)
                    thickness_values.append(thickness)
            except Exception as e:
                logger.warning(f"计算墙体 #{wall.id()} 厚度时出错: {e}")

        # 使用中位数作为标准厚度
        if thickness_values:
            thickness_values.sort()
            wall_thickness = thickness_values[len(thickness_values) // 2]
        else:
            wall_thickness = 0.2  # 默认厚度为20cm

    logger.info(f"使用墙厚: {wall_thickness:.3f}m")

    # 按楼层处理墙体
    results_by_elevation = {}

    for elevation, floor_walls in walls_by_elevation.items():
        logger.info(f"处理标高 {elevation}m 的墙体，共 {len(floor_walls)} 个")

        # 为当前楼层的每个墙体创建多边形
        polygons = []
        wall_ids = []
        id_to_polygon = {}

        for wall in floor_walls:
            try:
                if not wall.Representation:
                    continue

                settings = ifcopenshell.geom.settings()
                settings.set(settings.USE_WORLD_COORDS, True)
                shape = ifcopenshell.geom.create_shape(settings, wall)

                if not shape or not shape.geometry:
                    continue

                verts = shape.geometry.verts
                if not verts or len(verts) < 3:
                    continue

                # 提取顶点
                points = []
                for i in range(0, len(verts), 3):
                    if i + 2 < len(verts):
                        points.append((verts[i], verts[i + 1]))

                if len(points) < 3:
                    continue

                # 创建墙体多边形
                try:
                    from scipy.spatial import ConvexHull

                    hull = ConvexHull(points)
                    hull_points = [points[i] for i in hull.vertices]
                    wall_polygon = Polygon(hull_points)

                    # 修复和优化几何形状
                    wall_polygon = repair_geometry(wall_polygon)

                    if wall_polygon and wall_polygon.is_valid and wall_polygon.area > 0:
                        polygons.append(wall_polygon)
                        wall_ids.append(wall.id())
                        id_to_polygon[wall.id()] = wall_polygon
                    else:
                        logger.warning(f"墙体 #{wall.id()} 创建的多边形无效或面积为0")
                except Exception as e:
                    logger.warning(f"创建墙体 #{wall.id()} 多边形时出错: {e}")
            except Exception as e:
                logger.warning(f"处理墙体 #{wall.id()} 时出错: {e}")

        # 使用布尔运算合并当前楼层的多边形
        if not polygons:
            logger.warning(f"标高 {elevation}m 没有有效的墙体多边形")
            continue

        logger.info(
            f"使用布尔运算合并标高 {elevation}m 的 {len(polygons)} 个墙体多边形..."
        )

        try:
            # 合并所有多边形前，先做buffer填补缝隙
            buffer_distance = 0.05  # 5厘米，实际可根据模型精度调整
            buffered_polygons = [poly.buffer(buffer_distance) for poly in polygons]
            merged_polygon = unary_union(buffered_polygons)
            # 合并后再收缩
            if isinstance(merged_polygon, MultiPolygon):
                merged_polygon = max(merged_polygon.geoms, key=lambda p: p.area)
            merged_polygon = merged_polygon.buffer(-buffer_distance)

            # 验证外轮廓
            if not validate_outline(merged_polygon):
                logger.error(f"标高 {elevation}m 的外轮廓验证失败")
                continue

            # 识别构成外轮廓的墙体
            external_wall_ids = []
            outline_boundary = merged_polygon.boundary
            intersection_threshold = 0.1  # 相交比例阈值

            for wall_id, wall_poly in id_to_polygon.items():
                # 计算相交比例
                intersection_ratio = calculate_intersection_ratio(
                    wall_poly, outline_boundary
                )
                if intersection_ratio > intersection_threshold:
                    external_wall_ids.append(wall_id)

            logger.info(f"标高 {elevation}m 识别出 {len(external_wall_ids)} 个外墙")

            # 存储当前楼层的结果
            results_by_elevation[elevation] = {
                "outline": merged_polygon,
                "external_wall_ids": external_wall_ids,
                "merged_polygon": merged_polygon,
                "id_to_polygon": id_to_polygon,
            }

        except Exception as e:
            logger.error(f"标高 {elevation}m 合并多边形时出错: {e}")
            continue

    return results_by_elevation


def create_improved_wall_polygon(start_point, end_point, thickness, outward=False):
    """创建改进的墙体多边形，可以向外偏移"""
    # 计算墙体方向向量
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # 计算法向量 (垂直于墙线)
    if abs(dx) + abs(dy) > 0:
        length = np.sqrt(dx**2 + dy**2)
        nx, ny = -dy / length, dx / length
    else:
        nx, ny = 0, 1  # 默认向上的法向量

    # 使用实际墙体一半的厚度进行偏移
    half_thickness = thickness / 2.0  # 使用准确的一半厚度

    if outward:
        # 外侧多边形创建代码
        p1 = (start_point[0], start_point[1])
        p2 = (start_point[0] + nx * thickness, start_point[1] + ny * thickness)
        p3 = (end_point[0] + nx * thickness, end_point[1] + ny * thickness)
        p4 = (end_point[0], end_point[1])

        try:
            poly = Polygon([p1, p2, p3, p4])
            return poly if poly.is_valid else None
        except Exception:
            return None
    else:
        # 创建标准墙体多边形，包含内外两侧
        p1 = (
            start_point[0] + nx * half_thickness,
            start_point[1] + ny * half_thickness,
        )
        p2 = (
            start_point[0] - nx * half_thickness,
            start_point[1] - ny * half_thickness,
        )
        p3 = (end_point[0] - nx * half_thickness, end_point[1] - ny * half_thickness)
        p4 = (end_point[0] + nx * half_thickness, end_point[1] + ny * half_thickness)

        try:
            poly = Polygon([p1, p2, p3, p4])
            return poly if poly.is_valid else None
        except Exception:
            return None


def create_improved_curved_wall_polygon(curve_points, thickness, outward=False):
    """创建改进的弧形墙体多边形，从中心线向两侧偏移"""
    if len(curve_points) < 3:
        return None

    # 过滤点，确保连续性
    filtered_points = filter_curve_points(curve_points)
    if len(filtered_points) < 3:
        filtered_points = curve_points

    # 使用实际墙体一半的厚度
    half_thickness = thickness / 2.0  # 使用准确的一半厚度

    # 计算每点的法向量和内外侧点
    outer_points = []
    inner_points = []

    for i in range(len(filtered_points)):
        # 计算每个点的法向量
        if i == 0:
            # 第一个点
            dx = filtered_points[1][0] - filtered_points[0][0]
            dy = filtered_points[1][1] - filtered_points[0][1]
        elif i == len(filtered_points) - 1:
            # 最后一个点
            dx = filtered_points[i][0] - filtered_points[i - 1][0]
            dy = filtered_points[i][1] - filtered_points[i - 1][1]
        else:
            # 中间点，使用前后点的平均法向量
            dx1 = filtered_points[i][0] - filtered_points[i - 1][0]
            dy1 = filtered_points[i][1] - filtered_points[i - 1][1]
            dx2 = filtered_points[i + 1][0] - filtered_points[i][0]
            dy2 = filtered_points[i + 1][1] - filtered_points[i][1]

            # 使用平均向量
            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2

        # 计算单位法向量
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx = -dy / length
            ny = dx / length
        else:
            # 如果长度为0，使用默认法向量
            nx, ny = 0, 1

        # 创建内外侧点，精确偏移
        outer_points.append(
            (
                filtered_points[i][0] + nx * half_thickness,
                filtered_points[i][1] + ny * half_thickness,
            )
        )

        inner_points.append(
            (
                filtered_points[i][0] - nx * half_thickness,
                filtered_points[i][1] - ny * half_thickness,
            )
        )

    # 根据outward参数决定如何创建多边形
    try:
        if outward:
            # 改进的外轮廓绘制方法
            # 1. 使用所有外轮廓点
            # 2. 在起点和终点处添加额外的点以确保平滑过渡
            start = filtered_points[0]
            end = filtered_points[-1]

            # 计算起点和终点的法向量
            start_dx = filtered_points[1][0] - start[0]
            start_dy = filtered_points[1][1] - start[1]
            start_length = np.sqrt(start_dx**2 + start_dy**2)
            if start_length > 0:
                start_nx = -start_dy / start_length
                start_ny = start_dx / start_length
            else:
                start_nx, start_ny = 0, 1

            end_dx = end[0] - filtered_points[-2][0]
            end_dy = end[1] - filtered_points[-2][1]
            end_length = np.sqrt(end_dx**2 + end_dy**2)
            if end_length > 0:
                end_nx = -end_dy / end_length
                end_ny = end_dx / end_length
            else:
                end_nx, end_ny = 0, 1

            # 创建完整的外轮廓点集
            polygon_points = []

            # 添加起点
            polygon_points.append(start)

            # 添加所有外轮廓点
            polygon_points.extend(outer_points)

            # 添加终点
            polygon_points.append(end)

            # 添加所有内轮廓点（逆序）
            polygon_points.extend(reversed(inner_points))

            # 创建多边形
            poly = Polygon(polygon_points)

            # 如果多边形无效，尝试修复
            if not poly.is_valid:
                poly = poly.buffer(0)

            return poly
        else:
            # 正常墙体多边形
            polygon_points = outer_points + list(reversed(inner_points))
            poly = Polygon(polygon_points)

        if poly.is_valid:
            return poly
        else:
            # 尝试修复多边形
            return poly.buffer(0)
    except Exception as e:
        print(f"创建曲线多边形时出错: {str(e)}")
        return None


def determine_building_centroid(wall_geometries):
    """确定建筑物的中心点

    Args:
        wall_geometries: 墙体几何信息字典

    Returns:
        tuple: 中心点坐标 (x, y)
    """
    if not wall_geometries:
        return (0, 0)

    # 收集所有墙体的端点
    points = []
    for geom in wall_geometries.values():
        if isinstance(geom, dict) and "start" in geom and "end" in geom:
            points.append(geom["start"])
            points.append(geom["end"])

    # 计算平均中心点
    if points:
        x_sum = sum(p[0] for p in points)
        y_sum = sum(p[1] for p in points)
        return (x_sum / len(points), y_sum / len(points))

    return (0, 0)


def is_point_inside_polygon(point, polygon):
    """判断点是否在多边形内部

    Args:
        point: 点坐标 (x, y)
        polygon: 多边形对象

    Returns:
        bool: 点在多边形内部返回True
    """
    return Point(point).within(polygon)


def interactive_external_wall_adjustment(
    wall_geometries, external_outline, id_to_polygon
):
    """交互式调整外墙识别阈值

    Args:
        wall_geometries: 墙体几何信息
        external_outline: 外轮廓多边形
        id_to_polygon: 墙体ID到多边形的映射

    Returns:
        list: 调整后的外墙ID列表
    """
    # 默认阈值
    threshold = 0.5

    while True:
        # 使用当前阈值识别外墙
        external_wall_ids = []
        for wall_id, poly in id_to_polygon.items():
            wall_boundary = poly.boundary
            outline_boundary = external_outline.boundary

            if wall_boundary.intersects(outline_boundary):
                intersection = wall_boundary.intersection(outline_boundary)
                if hasattr(intersection, "length") and intersection.length > 0:
                    overlap_ratio = intersection.length / wall_boundary.length
                    if overlap_ratio > threshold:
                        external_wall_ids.append(wall_id)

        # 显示识别结果
        print(
            f"\n使用阈值 {threshold} 识别到 {len(external_wall_ids)} 个外墙: {external_wall_ids}"
        )

        # 绘制结果
        plot_walls_and_outline(wall_geometries, external_outline, external_wall_ids)

        # 询问是否调整阈值
        user_input = input("\n结果是否满意？(y/n): ")
        if user_input.lower() == "y":
            return external_wall_ids

        # 调整阈值
        new_threshold = input(f"请输入新的阈值 (当前: {threshold}): ")
        try:
            threshold = float(new_threshold)
        except ValueError:
            print("输入无效，使用原阈值")


def visualize_wall_centerlines(wall_geometries, external_walls=None):
    """可视化墙体中心线和偏移情况

    Args:
        wall_geometries: 墙体几何信息字典
        external_walls: 外墙ID列表
    """
    plt.figure(figsize=(12, 12))

    # 确定标准墙厚
    standard_thickness = 0.2  # 默认厚度
    thickness_values = []

    for wall_id, wall_geom in wall_geometries.items():
        if isinstance(wall_geom, dict) and "bbox" in wall_geom:
            bbox = wall_geom["bbox"]
            if len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox
                if abs(x_max - x_min) > abs(y_max - y_min):
                    thickness = abs(y_max - y_min)
                else:
                    thickness = abs(x_max - x_min)
                thickness_values.append(thickness)

    if thickness_values:
        thickness_values.sort()
        standard_thickness = thickness_values[len(thickness_values) // 2]

    print(f"使用标准墙厚: {standard_thickness:.3f}m")
    half_thickness = standard_thickness / 2.0

    # 绘制所有墙体
    for wall_id, wall_geom in wall_geometries.items():
        if isinstance(wall_geom, dict) and "start" in wall_geom and "end" in wall_geom:
            # 根据是否为外墙选择颜色
            is_external = external_walls and wall_id in external_walls
            centerline_color = "red" if is_external else "blue"
            offset_color = "darkred" if is_external else "darkblue"

            # 检查是否为弧形墙
            if (
                "is_curved" in wall_geom
                and wall_geom["is_curved"]
                and "curve_points" in wall_geom
            ):
                # 弧形墙处理
                curve_points = wall_geom["curve_points"]
                if len(curve_points) > 2:
                    # 过滤处理曲线点 - 直接使用原始点，但去除重复点
                    filtered_points = filter_curve_points(curve_points)

                    if len(filtered_points) > 2:
                        # 绘制中心线
                        xs = [p[0] for p in filtered_points]
                        ys = [p[1] for p in filtered_points]

                        # 使用正确的matplotlib绘图格式
                        plt.plot(
                            xs,
                            ys,
                            color=centerline_color,
                            linestyle="-",
                            linewidth=2,
                            label=(
                                f"Wall {wall_id} Centerline"
                                if wall_id == list(wall_geometries.keys())[0]
                                else ""
                            ),
                        )

                        # 计算每点的法向量
                        normals = []
                        outer_points = []
                        inner_points = []

                        for i in range(len(filtered_points)):
                            # 法向量计算
                            if i == 0:
                                # 第一个点，使用前两点计算方向
                                dx = filtered_points[1][0] - filtered_points[0][0]
                                dy = filtered_points[1][1] - filtered_points[0][1]
                            elif i == len(filtered_points) - 1:
                                # 最后一个点，使用最后两点计算方向
                                dx = filtered_points[i][0] - filtered_points[i - 1][0]
                                dy = filtered_points[i][1] - filtered_points[i - 1][1]
                            else:
                                # 中间点，使用前后点计算平均方向
                                dx1 = filtered_points[i][0] - filtered_points[i - 1][0]
                                dy1 = filtered_points[i][1] - filtered_points[i - 1][1]
                                dx2 = filtered_points[i + 1][0] - filtered_points[i][0]
                                dy2 = filtered_points[i + 1][1] - filtered_points[i][1]

                                # 使用平均向量
                                dx = (dx1 + dx2) / 2
                                dy = (dy1 + dy2) / 2

                            # 计算法向量
                            length = np.sqrt(dx**2 + dy**2)
                            if length > 0:
                                nx = -dy / length
                                ny = dx / length
                            else:
                                nx, ny = 0, 1

                            normals.append((nx, ny))

                            # 计算偏移点
                            outer_points.append(
                                (
                                    filtered_points[i][0] + nx * half_thickness,
                                    filtered_points[i][1] + ny * half_thickness,
                                )
                            )

                            inner_points.append(
                                (
                                    filtered_points[i][0] - nx * half_thickness,
                                    filtered_points[i][1] - ny * half_thickness,
                                )
                            )

                        # 绘制偏移线
                        outer_xs = [p[0] for p in outer_points]
                        outer_ys = [p[1] for p in outer_points]
                        inner_xs = [p[0] for p in inner_points]
                        inner_ys = [p[1] for p in inner_points]

                        plt.plot(
                            outer_xs,
                            outer_ys,
                            color=offset_color,
                            linestyle="--",
                            linewidth=1,
                            label=(
                                f"Wall {wall_id} Outer Offset"
                                if wall_id == list(wall_geometries.keys())[0]
                                else ""
                            ),
                        )
                        plt.plot(
                            inner_xs,
                            inner_ys,
                            color=offset_color,
                            linestyle="--",
                            linewidth=1,
                            label=(
                                f"Wall {wall_id} Inner Offset"
                                if wall_id == list(wall_geometries.keys())[0]
                                else ""
                            ),
                        )

                        # 显示代表性法向量（不显示全部，避免过于拥挤）
                        for i in range(
                            0, len(filtered_points), max(1, len(filtered_points) // 5)
                        ):
                            if i < len(filtered_points) and i < len(normals):
                                p = filtered_points[i]
                                nx, ny = normals[i]
                                plt.arrow(
                                    p[0],
                                    p[1],
                                    nx * half_thickness,
                                    ny * half_thickness,
                                    head_width=0.1,
                                    head_length=0.1,
                                    fc=offset_color,
                                    ec=offset_color,
                                )

                        # 添加墙体ID标签
                        mid_idx = len(filtered_points) // 2
                        mid_point = filtered_points[mid_idx]
                        plt.text(
                            mid_point[0],
                            mid_point[1],
                            str(wall_id),
                            fontsize=10,
                            ha="center",
                            va="center",
                            color="black",
                            bbox=dict(facecolor="white", alpha=0.7),
                        )
            else:
                # 直线墙体处理
                start = wall_geom["start"]
                end = wall_geom["end"]

                # 绘制中心线
                plt.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=centerline_color,
                    linestyle="-",
                    linewidth=2,
                    label=(
                        f"Wall {wall_id} Centerline"
                        if wall_id == list(wall_geometries.keys())[0]
                        else ""
                    ),
                )

                # 计算法向量
                dx = end[0] - start[0]
                dy = end[1] - start[1]

                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    nx = -dy / length
                    ny = dx / length
                else:
                    nx, ny = 0, 1

                # 计算偏移点
                start_outer = (
                    start[0] + nx * half_thickness,
                    start[1] + ny * half_thickness,
                )
                start_inner = (
                    start[0] - nx * half_thickness,
                    start[1] - ny * half_thickness,
                )
                end_outer = (end[0] + nx * half_thickness, end[1] + ny * half_thickness)
                end_inner = (end[0] - nx * half_thickness, end[1] - ny * half_thickness)

                # 绘制偏移线
                plt.plot(
                    [start_outer[0], end_outer[0]],
                    [start_outer[1], end_outer[1]],
                    color=offset_color,
                    linestyle="--",
                    linewidth=1,
                    label=(
                        f"Wall {wall_id} Outer Offset"
                        if wall_id == list(wall_geometries.keys())[0]
                        else ""
                    ),
                )
                plt.plot(
                    [start_inner[0], end_inner[0]],
                    [start_inner[1], end_inner[1]],
                    color=offset_color,
                    linestyle="--",
                    linewidth=1,
                    label=(
                        f"Wall {wall_id} Inner Offset"
                        if wall_id == list(wall_geometries.keys())[0]
                        else ""
                    ),
                )

                # 可视化法向量
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                plt.arrow(
                    mid_x,
                    mid_y,
                    nx * half_thickness,
                    ny * half_thickness,
                    head_width=0.1,
                    head_length=0.1,
                    fc=offset_color,
                    ec=offset_color,
                )

                # 添加墙体ID标签
                plt.text(
                    mid_x,
                    mid_y,
                    str(wall_id),
                    fontsize=10,
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.7),
                )

    plt.title("Wall Centerlines and Offsets")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # 保存图像
    plt.savefig("wall_centerlines.png")
    print(f"Saved wall centerlines image to 'wall_centerlines.png'")

    # 显示图像
    plt.show()


def main():
    loader = IFCLoader()

    if loader.load_file("outline wall_3.ifc"):
        print("\n文件信息:")
        print(loader.get_file_info())

        walls = loader.get_all_walls()
        print(f"\n找到 {len(walls)} 个墙体")

        # 打印所有墙体的几何信息
        print("\n打印所有墙体的几何信息:")
        for wall in walls:
            loader.print_wall_info(wall)

        wall_geometries = loader.get_all_wall_geometries()

        # 创建墙体ID到墙体对象的映射
        wall_dict = {}
        for wall in walls:
            wall_id = get_entity_id(wall)
            if wall_id is not None:
                wall_dict[wall_id] = wall

        # 检查墙体几何数据的格式
        print(f"\n墙体几何数据格式检查:")
        print(f"墙体总数: {len(wall_geometries)}")
        if wall_geometries:
            sample_id = next(iter(wall_geometries))
            print(f"示例墙体 ID: {sample_id}")
            print(f"几何数据类型: {type(wall_geometries[sample_id])}")
            print(f"几何数据: {wall_geometries[sample_id]}")

        # 找出外轮廓 - 使用改进的方法
        print("\n计算外墙轮廓...")
        results_by_elevation = get_external_wall_outline(ifc_file=loader.ifc_file)

        # 先获取外墙ID后再可视化
        visualize_wall_centerlines(
            wall_geometries, results_by_elevation[0]["external_wall_ids"]
        )

        if results_by_elevation[0]["outline"]:
            print(f"\n外轮廓信息:")
            print(f"类型: {type(results_by_elevation[0]['outline']).__name__}")
            print(f"面积: {results_by_elevation[0]['outline'].area}")
            print(f"周长: {results_by_elevation[0]['outline'].length}")

            if hasattr(results_by_elevation[0]["outline"], "exterior"):
                print(
                    f"外轮廓顶点数: {len(list(results_by_elevation[0]['outline'].exterior.coords)) - 1}"
                )

                # 识别构成外轮廓的墙体
                print("\n构成外轮廓的墙体:")
                for wall_id in results_by_elevation[0]["external_wall_ids"]:
                    # 使用字典查找墙体对象
                    if wall_id in wall_dict:
                        wall_obj = wall_dict[wall_id]
                        print(f"墙体ID: {wall_id}")
                        loader.print_wall_info(wall_obj)
                    else:
                        # 如果在墙体字典中找不到，尝试从几何数据获取信息
                        if wall_id in wall_geometries:
                            print(f"墙体ID: {wall_id}")
                            print(f"  起点: {wall_geometries[wall_id]['start']}")
                            print(f"  终点: {wall_geometries[wall_id]['end']}")
                            print(f"  长度: {wall_geometries[wall_id]['length']}m")

                print(
                    f"\n总共有 {len(results_by_elevation[0]['external_wall_ids'])} 个墙体构成外轮廓"
                )

                # 绘制墙体和外轮廓
                plot_walls_and_outline(
                    wall_geometries,
                    results_by_elevation[0]["outline"],
                    results_by_elevation[0]["external_wall_ids"],
                )
            else:
                print("外轮廓没有外边界属性，可能不是多边形")
        else:
            print("无法计算外墙轮廓")

        # 显示合并多边形
        plot_merged_polygon(results_by_elevation[0]["merged_polygon"])

        # 交互式调整外墙识别阈值
        new_external_wall_ids = interactive_external_wall_adjustment(
            wall_geometries,
            results_by_elevation[0]["outline"],
            results_by_elevation[0]["id_to_polygon"],
        )

        # 使用新的外墙ID再次可视化
        visualize_wall_centerlines(wall_geometries, new_external_wall_ids)


if __name__ == "__main__":
    main()
