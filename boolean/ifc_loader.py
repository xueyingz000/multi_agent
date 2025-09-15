import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.representation
import ifcopenshell.geom
import numpy as np
import math
import time
import os
from collections import defaultdict
from boolean.improved_curve_extractor import sort_points_by_path
from boolean.arc_extractor import extract_wall_centerline
from boolean.wall_centerline_extractor import extract_wall_true_centerline


class IFCLoader:
    """IFC文件加载和处理类，用于提取建筑物外墙和几何信息"""

    def __init__(self, ifc_path=None):
        """初始化IFC加载器

        Args:
            ifc_path: IFC文件路径
        """
        self.ifc_path = ifc_path
        self.ifc_file = None
        self.geometry_cache = {}  # 几何信息缓存
        self.debug_mode = False  # 调试模式
        self.tolerance = 200  # 几何连接容差（毫米）

        if ifc_path:
            self.load_file(ifc_path)

    def load_file(self, ifc_path):
        """加载IFC文件

        Args:
            ifc_path: IFC文件路径

        Returns:
            bool: 是否成功加载
        """
        try:
            t1 = time.time()
            print(f"正在加载IFC文件: {ifc_path}")
            self.ifc_file = ifcopenshell.open(ifc_path)
            self.ifc_path = ifc_path
            print(f"文件加载时间: {time.time() - t1:.2f}秒")
            return True
        except Exception as e:
            print(f"加载IFC文件失败: {str(e)}")
            return False

    def get_all_walls(self):
        """获取IFC文件中的所有墙体

        Returns:
            list: 墙体对象列表
        """
        if not self.ifc_file:
            print("错误: 未加载IFC文件")
            return []

        # 尝试获取标准墙和通用墙
        walls = self.ifc_file.by_type("IfcWall")
        if not walls:
            walls = self.ifc_file.by_type("IfcWallStandardCase")

        print(f"找到 {len(walls)} 个墙体构件")
        return walls

    def get_wall_geometry(self, wall):
        """获取墙的几何信息，修复版本

        Args:
            wall: IFC墙体对象

        Returns:
            dict: 墙的几何信息，包含起点、终点、长度等
        """
        try:
            # 检查缓存
            wall_id = wall.id()
            if wall_id in self.geometry_cache:
                return self.geometry_cache[wall_id]

            # 创建几何处理器设置
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)

            # 创建形状
            shape = ifcopenshell.geom.create_shape(settings, wall)
            if not shape:
                print(f"警告: 墙体 {wall_id} 无法创建形状")
                return None

            # 获取墙体几何信息的替代方法
            try:
                # 方法1: 从顶点直接计算边界框
                verts = shape.geometry.verts
                if verts and len(verts) > 0:
                    # 重新组织顶点数据为坐标点列表
                    points = []
                    for i in range(0, len(verts), 3):
                        if i + 2 < len(verts):
                            points.append((verts[i], verts[i + 1], verts[i + 2]))

                    if points:
                        # 计算边界框
                        min_x = min(p[0] for p in points)
                        max_x = max(p[0] for p in points)
                        min_y = min(p[1] for p in points)
                        max_y = max(p[1] for p in points)

                        # 判断墙体方向（根据边界框的长宽比）
                        width = max_x - min_x
                        height = max_y - min_y

                        # 检查这是否是弧形墙体
                        is_curved = False
                        actual_ifc_curve = None  # 用于存储找到的实际IfcCurve对象

                        # 遍历墙体的Representation以寻找弧形定义
                        if hasattr(wall, "Representation") and wall.Representation:
                            for rep in wall.Representation.Representations:
                                if hasattr(rep, "Items"):
                                    for item in rep.Items:
                                        if item.is_a("IfcExtrudedAreaSolid"):
                                            swept_area = item.SweptArea
                                            if hasattr(swept_area, "OuterCurve"):
                                                outer_curve = swept_area.OuterCurve
                                                if outer_curve.is_a(
                                                    (
                                                        "IfcCircle",
                                                        "IfcTrimmedCurve",
                                                        "IfcCompositeCurve",
                                                    )
                                                ):
                                                    is_curved = True
                                                    actual_ifc_curve = outer_curve
                                                    break  # 找到曲线，退出内层循环
                                                elif outer_curve.is_a(
                                                    "IfcArbitraryClosedProfileDef"
                                                ):  # 处理任意闭合轮廓
                                                    if hasattr(
                                                        outer_curve, "OuterCurve"
                                                    ) and outer_curve.OuterCurve.is_a(
                                                        (
                                                            "IfcCircle",
                                                            "IfcTrimmedCurve",
                                                            "IfcCompositeCurve",
                                                        )
                                                    ):
                                                        is_curved = True
                                                        actual_ifc_curve = (
                                                            outer_curve.OuterCurve
                                                        )
                                                        break
                                    if (
                                        is_curved
                                    ):  # 如果在当前Representation中找到，则退出外层循环
                                        break

                        if is_curved and actual_ifc_curve:
                            curve_points = self.extract_curve_points(
                                wall, actual_ifc_curve
                            )

                            # 如果成功提取曲线点
                            if curve_points and len(curve_points) > 2:
                                start_point = curve_points[0]
                                end_point = curve_points[-1]

                                # 计算弧长
                                length = 0
                                for i in range(len(curve_points) - 1):
                                    dx = curve_points[i + 1][0] - curve_points[i][0]
                                    dy = curve_points[i + 1][1] - curve_points[i][1]
                                    length += math.sqrt(dx * dx + dy * dy)

                                result = {
                                    "id": wall_id,
                                    "start": start_point,
                                    "end": end_point,
                                    "length": length,
                                    "wall": wall,
                                    "bbox": (min_x, min_y, max_x, max_y),
                                    "is_curved": True,
                                    "curve_points": curve_points,
                                }

                                self.geometry_cache[wall_id] = result

                                if self.debug_mode:
                                    print(
                                        f"弧形墙体 {wall_id} 曲线点数: {len(curve_points)}, 长度: {length:.2f}"
                                    )

                                return result

                        # --- 直线墙体处理 --- #
                        # 获取墙体的所有2D点
                        xy_points = [
                            (p[0], p[1]) for p in points
                        ]  # `points` 已经包含 (x,y,z)

                        if len(xy_points) < 2:
                            print(f"警告: 墙体 {wall_id} 没有足够的2D点来确定轴线。")
                            return None

                        # 将点转换为numpy数组
                        np_points = np.array(xy_points)

                        # 居中点
                        mean_coords = np.mean(np_points, axis=0)
                        centered_points = np_points - mean_coords

                        # 计算协方差矩阵
                        covariance_matrix = np.cov(centered_points, rowvar=False)

                        # 计算特征值和特征向量
                        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

                        # 找到主轴（最大特征值对应的特征向量）
                        main_axis_idx = np.argmax(eigenvalues)
                        main_axis = eigenvectors[:, main_axis_idx]

                        # 找到次轴（垂直于主轴的特征向量）
                        minor_axis_idx = 1 - main_axis_idx
                        minor_axis = eigenvectors[:, minor_axis_idx]

                        # 将所有点投影到主轴上，找到起点和终点
                        projections_on_main_axis = np.dot(centered_points, main_axis)
                        start_proj = np.min(projections_on_main_axis)
                        end_proj = np.max(projections_on_main_axis)

                        start_point = mean_coords + start_proj * main_axis
                        end_point = mean_coords + end_proj * main_axis

                        # 计算长度
                        length = np.linalg.norm(end_point - start_point)

                        # 计算墙体宽度（厚度）
                        projections_on_minor_axis = np.dot(centered_points, minor_axis)
                        wall_width = np.max(projections_on_minor_axis) - np.min(
                            projections_on_minor_axis
                        )

                        # 如果长度过小，可能是不正常的墙体
                        if length < 0.01:
                            print(
                                f"警告: 墙体 {wall_id} 沿主轴长度过小 ({length:.3f}m)，可能不是有效墙体。"
                            )
                            return None

                        result = {
                            "id": wall_id,
                            "start": tuple(start_point),  # Convert numpy array to tuple
                            "end": tuple(end_point),  # Convert numpy array to tuple
                            "length": length,
                            "wall": wall,
                            "bbox": (min_x, min_y, max_x, max_y),
                            "is_curved": False,
                            "width": wall_width,  # Store calculated width
                        }

                        # 存入缓存
                        self.geometry_cache[wall_id] = result

                        if self.debug_mode:
                            print(
                                f"直线墙体 {wall_id} 中心线: {start_point} -> {end_point}, 长度: {length:.2f}, 宽度: {wall_width:.2f}"
                            )

                        return result

                print(f"警告: 墙体 {wall_id} 无法从顶点计算边界")
                return None

            except Exception as geo_err:
                print(f"从形状获取墙体 {wall_id} 几何信息失败: {str(geo_err)}")

                # 备选方法: 尝试从墙体的位置信息获取
                try:
                    # 从ObjectPlacement获取墙体位置
                    if hasattr(wall, "ObjectPlacement") and wall.ObjectPlacement:
                        placement = wall.ObjectPlacement

                        # 获取位置坐标
                        if (
                            hasattr(placement, "RelativePlacement")
                            and placement.RelativePlacement
                        ):
                            rel_placement = placement.RelativePlacement

                            # 获取位置点
                            if (
                                hasattr(rel_placement, "Location")
                                and rel_placement.Location
                            ):
                                location = rel_placement.Location

                                if hasattr(location, "Coordinates"):
                                    coords = location.Coordinates
                                    base_point = (coords[0], coords[1])

                                    # 获取墙体方向
                                    direction = (1, 0)  # 默认X轴方向
                                    if (
                                        hasattr(rel_placement, "RefDirection")
                                        and rel_placement.RefDirection
                                    ):
                                        ref_dir = rel_placement.RefDirection
                                        if hasattr(ref_dir, "DirectionRatios"):
                                            dir_ratios = ref_dir.DirectionRatios
                                            if len(dir_ratios) >= 2:
                                                direction = (
                                                    dir_ratios[0],
                                                    dir_ratios[1],
                                                )

                                    # 估算墙体长度
                                    length = 3.0  # 默认长度为3米

                                    # 根据方向计算终点
                                    end_point = (
                                        base_point[0] + direction[0] * length,
                                        base_point[1] + direction[1] * length,
                                    )

                                    result = {
                                        "id": wall_id,
                                        "start": base_point,
                                        "end": end_point,
                                        "length": length,
                                        "wall": wall,
                                        "estimated": True,  # 标记为估计值
                                    }

                                    # 存入缓存
                                    self.geometry_cache[wall_id] = result

                                    print(
                                        f"墙体 {wall_id} 使用估计轴线: {base_point} -> {end_point}"
                                    )
                                    return result

                    print(f"警告: 墙体 {wall_id} 无法从位置信息获取几何数据")
                    return None

                except Exception as place_err:
                    print(
                        f"从位置信息获取墙体 {wall_id} 几何数据失败: {str(place_err)}"
                    )
                    return None

        except Exception as e:
            print(f"获取墙体 {wall.id()} 几何信息失败: {str(e)}")
            return None

    def get_all_wall_geometries(self):
        """获取所有墙体的几何信息

        Returns:
            dict: 墙体ID到几何信息的映射
        """
        walls = self.get_all_walls()
        wall_geometries = {}

        for wall in walls:
            geom = self.get_wall_geometry(wall)
            if geom:
                wall_geometries[wall.id()] = geom

        print(f"成功获取 {len(wall_geometries)} 个墙体的几何信息")
        return wall_geometries

    def get_wall_elevation(self, wall):
        """获取墙体的标高

        Args:
            wall: IFC墙体对象

        Returns:
            float: 墙体标高，如果无法获取则返回None
        """
        try:
            # 方法1: 从属性集中获取
            psets = ifcopenshell.util.element.get_psets(wall)
            for pset_name, props in psets.items():
                # 检查常见的标高属性名
                for height_prop in [
                    "BaseConstraintHeight",
                    "Height",
                    "Elevation",
                    "BaseHeight",
                ]:
                    if height_prop in props:
                        elevation = float(props[height_prop])
                        if self.debug_mode:
                            print(
                                f"墙体 {wall.id()} 从属性集 {pset_name} 获取标高: {elevation}m"
                            )
                        return elevation

            # 方法2: 从几何表示中获取
            shape = ifcopenshell.util.representation.get_representation(
                wall, "Model", "Body"
            )
            if shape:
                try:
                    bbox = shape.BoundingBox
                    if bbox:
                        elevation = float(bbox.Corner.Coordinates[2])
                        if self.debug_mode:
                            print(f"墙体 {wall.id()} 从边界框获取标高: {elevation}m")
                        return elevation
                except:
                    pass

            # 方法3: 从ObjectPlacement中获取
            if wall.ObjectPlacement:
                if hasattr(wall.ObjectPlacement, "PlacementRelTo"):
                    # 考虑相对位置
                    relative_placement = wall.ObjectPlacement.PlacementRelTo
                    if relative_placement and hasattr(
                        relative_placement, "RelativePlacement"
                    ):
                        base_elevation = (
                            relative_placement.RelativePlacement.Location.Coordinates[2]
                        )
                        local_elevation = (
                            wall.ObjectPlacement.RelativePlacement.Location.Coordinates[
                                2
                            ]
                        )
                        elevation = float(base_elevation + local_elevation)
                        if self.debug_mode:
                            print(f"墙体 {wall.id()} 从相对位置获取标高: {elevation}m")
                        return elevation
                else:
                    # 直接位置
                    elevation = float(
                        wall.ObjectPlacement.RelativePlacement.Location.Coordinates[2]
                    )
                    if self.debug_mode:
                        print(f"墙体 {wall.id()} 从直接位置获取标高: {elevation}m")
                    return elevation

            # 方法4: 从关联的楼层获取
            containment = ifcopenshell.util.element.get_container(wall)
            if containment and hasattr(containment, "Elevation"):
                elevation = float(containment.Elevation)
                if self.debug_mode:
                    print(f"墙体 {wall.id()} 从所属楼层获取标高: {elevation}m")
                return elevation

            if self.debug_mode:
                print(f"警告: 墙体 {wall.id()} 无法获取标高")
            return None

        except Exception as e:
            print(f"获取墙体 {wall.id()} 标高失败: {str(e)}")
            return None

    def group_walls_by_elevation(self):
        """将墙体按标高分组

        Returns:
            dict: 标高到墙体列表的映射
        """
        walls = self.get_all_walls()
        walls_by_elevation = defaultdict(list)

        for wall in walls:
            elevation = self.get_wall_elevation(wall)
            if elevation is None:
                elevation = 0.0  # 如果无法获取标高，默认为0

            walls_by_elevation[elevation].append(wall)

        # 打印分组结果
        print("\n墙体按标高分组结果:")
        for elevation, elevation_walls in walls_by_elevation.items():
            print(f"标高 {elevation}m: {len(elevation_walls)} 个墙体")

        return walls_by_elevation

    def are_points_connected(self, point1, point2, tolerance=None):
        """检查两个点是否连接

        Args:
            point1: 第一个点坐标 (x, y)
            point2: 第二个点坐标 (x, y)
            tolerance: 容差值，如果为None则使用默认值

        Returns:
            bool: 两点是否可视为连接
        """
        if tolerance is None:
            tolerance = self.tolerance / 1000  # 将毫米转换为米

        dx = abs(point1[0] - point2[0])
        dy = abs(point1[1] - point2[1])
        distance = math.sqrt(dx * dx + dy * dy)

        return distance <= tolerance

    def find_wall_connections(self, wall_geometries, tolerance=None):
        """查找墙体之间的连接关系

        Args:
            wall_geometries: 墙体几何信息字典
            tolerance: 连接判断容差

        Returns:
            dict: 墙体连接关系图
        """
        if tolerance is None:
            tolerance = self.tolerance / 1000  # 毫米转米

        connections = {}
        for wall_id in wall_geometries:
            connections[wall_id] = []

        # 检查所有墙体对之间的连接关系
        for wall1_id, wall1 in wall_geometries.items():
            for wall2_id, wall2 in wall_geometries.items():
                if wall1_id != wall2_id:
                    # 检查所有可能的连接
                    if self.are_points_connected(
                        wall1["end"], wall2["start"], tolerance
                    ):
                        connections[wall1_id].append(
                            {"wall_id": wall2_id, "type": "end-start"}
                        )
                    elif self.are_points_connected(
                        wall1["end"], wall2["end"], tolerance
                    ):
                        connections[wall1_id].append(
                            {"wall_id": wall2_id, "type": "end-end"}
                        )
                    elif self.are_points_connected(
                        wall1["start"], wall2["start"], tolerance
                    ):
                        connections[wall1_id].append(
                            {"wall_id": wall2_id, "type": "start-start"}
                        )
                    elif self.are_points_connected(
                        wall1["start"], wall2["end"], tolerance
                    ):
                        connections[wall1_id].append(
                            {"wall_id": wall2_id, "type": "start-end"}
                        )

        return connections

    def get_file_info(self):
        """获取IFC文件的基本信息

        Returns:
            dict: 文件信息
        """
        if not self.ifc_file:
            return {"error": "未加载IFC文件"}

        try:
            # 获取项目信息
            projects = self.ifc_file.by_type("IfcProject")
            project_info = {}
            if projects:
                project = projects[0]
                project_info = {
                    "项目名称": project.Name if hasattr(project, "Name") else "未知",
                    "项目描述": (
                        project.Description
                        if hasattr(project, "Description") and project.Description
                        else "无描述"
                    ),
                    "项目全局ID": (
                        project.GlobalId if hasattr(project, "GlobalId") else "未知"
                    ),
                }

            # 获取构件统计
            element_counts = {}
            for entity in [
                "IfcWall",
                "IfcSlab",
                "IfcColumn",
                "IfcBeam",
                "IfcDoor",
                "IfcWindow",
            ]:
                element_counts[entity] = len(self.ifc_file.by_type(entity))

            # 文件基本信息
            file_info = {
                "文件路径": self.ifc_path,
                "文件大小": (
                    f"{os.path.getsize(self.ifc_path) / (1024*1024):.2f} MB"
                    if os.path.exists(self.ifc_path)
                    else "未知"
                ),
                "项目信息": project_info,
                "构件统计": element_counts,
            }

            return file_info

        except Exception as e:
            print(f"获取文件信息失败: {str(e)}")
            return {"error": str(e)}

    def print_wall_info(self, wall):
        """打印墙体的基本几何信息

        Args:
            wall: IFC墙体对象
        """
        try:
            wall_id = wall.id()
            print(f"\n墙体 ID: {wall_id} {'='*30}")

            # 基本信息
            print(f"全局ID: {wall.GlobalId if hasattr(wall, 'GlobalId') else '未知'}")
            print(
                f"名称: {wall.Name if hasattr(wall, 'Name') and wall.Name else '未命名'}"
            )
            print(f"类型: {wall.is_a()}")

            # 获取并打印几何信息
            geom = self.get_wall_geometry(wall)
            if geom:
                print("\n几何信息:")
                print(f"  起点: ({geom['start'][0]:.3f}, {geom['start'][1]:.3f})")
                print(f"  终点: ({geom['end'][0]:.3f}, {geom['end'][1]:.3f})")
                print(f"  长度: {geom['length']:.3f}m")
                if "bbox" in geom:
                    bbox = geom["bbox"]
                    print(
                        f"  边界框: X({bbox[0]:.3f}, {bbox[2]:.3f}), Y({bbox[1]:.3f}, {bbox[3]:.3f})"
                    )
                if "estimated" in geom and geom["estimated"]:
                    print("  注意: 几何信息为估计值")

            # 获取标高
            elevation = self.get_wall_elevation(wall)
            print(
                f"\n标高: {elevation:.3f}m" if elevation is not None else "\n标高: 未知"
            )

            print("=" * 50)

        except Exception as e:
            print(
                f"打印墙体 {wall.id() if hasattr(wall, 'id') else '未知'} 信息失败: {str(e)}"
            )

    def check_if_curved_wall(self, wall):
        """检查墙体是否为弧形墙（仅通过IFC表示类型判断）

        Args:
            wall: IFC墙体对象

        Returns:
            bool: 是否为弧形墙
        """
        try:
            if hasattr(wall, "Representation") and wall.Representation:
                for rep in wall.Representation.Representations:
                    if hasattr(rep, "Items"):
                        for item in rep.Items:
                            if item.is_a("IfcExtrudedAreaSolid"):
                                swept_area = item.SweptArea
                                if hasattr(swept_area, "OuterCurve"):
                                    outer_curve = swept_area.OuterCurve
                                    # 更精确地检查是否为曲线类型
                                    if outer_curve.is_a(
                                        (
                                            "IfcCircle",
                                            "IfcTrimmedCurve",
                                            "IfcCompositeCurve",
                                        )
                                    ):
                                        return True
                                    elif outer_curve.is_a(
                                        "IfcArbitraryClosedProfileDef"
                                    ):  # 处理任意闭合轮廓
                                        if hasattr(
                                            outer_curve, "OuterCurve"
                                        ) and outer_curve.OuterCurve.is_a(
                                            (
                                                "IfcCircle",
                                                "IfcTrimmedCurve",
                                                "IfcCompositeCurve",
                                            )
                                        ):
                                            return True
            return False

        except Exception as e:
            print(f"检查墙体 {wall.id()} 是否为弧形时出错: {str(e)}")
            return False

    def extract_curve_points(self, wall, ifc_curve_entity):
        """从IFC曲线实体中提取曲线点

        Args:
            wall: IFC墙体对象 (用于日志)
            ifc_curve_entity: IfcCurve, IfcCircle, IfcTrimmedCurve, IfcCompositeCurve等实体

        Returns:
            list: 曲线点列表
        """
        try:
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)  # 确保使用世界坐标

            # 从曲线实体创建形状
            curve_shape = ifcopenshell.geom.create_shape(settings, ifc_curve_entity)
            if not curve_shape or not curve_shape.geometry:
                print(f"警告: 墙体 {wall.id()} 的曲线实体无法创建形状。")
                return []

            verts = curve_shape.geometry.verts
            points_3d = []
            for i in range(0, len(verts), 3):
                if i + 2 < len(verts):
                    points_3d.append((verts[i], verts[i + 1], verts[i + 2]))

            xy_points = [(p[0], p[1]) for p in points_3d]

            # 过滤重复点 (使用小容差)
            unique_points = []
            seen = set()
            for p in xy_points:
                rounded = (round(p[0], 3), round(p[1], 3))  # 稍高精度
                if rounded not in seen:
                    seen.add(rounded)
                    unique_points.append(p)

            # 对点进行简化 (如果点数过多)
            if len(unique_points) > 50:
                # 简单步长降采样
                step = len(unique_points) // 50
                simplified_points = [
                    unique_points[i] for i in range(0, len(unique_points), max(1, step))
                ]
                if unique_points[-1] not in simplified_points:  # 确保包含最后一个点
                    simplified_points.append(unique_points[-1])
                return simplified_points

            return unique_points

        except Exception as e:
            print(f"从IFC曲线实体提取墙体 {wall.id()} 曲线点时出错: {str(e)}")
            return []


# 测试代码示例
if __name__ == "__main__":
    loader = IFCLoader()
    if loader.load_file("../outline wall_1.ifc"):
        print("\n文件信息:")
        print(loader.get_file_info())

        walls = loader.get_all_walls()
        print(f"\n找到 {len(walls)} 个墙体")

        # 打印所有墙体的几何信息
        print("\n打印所有墙体的几何信息:")
        for wall in walls:
            loader.print_wall_info(wall)

        wall_geometries = loader.get_all_wall_geometries()

        # 按标高分组墙体
        walls_by_elevation = loader.group_walls_by_elevation()
