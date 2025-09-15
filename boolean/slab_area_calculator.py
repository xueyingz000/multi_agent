import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

# 将父目录添加到路径中以便导入
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# 导入相关模块
from boolean.ifc_loader import IFCLoader
from boolean.external_wall import get_external_wall_outline


def get_slab_geometry(loader):
    """从IFC文件中提取所有楼板的几何信息"""
    slabs = loader.get_all_elements_by_type("IfcSlab")
    print(f"找到 {len(slabs)} 个楼板")

    slab_geometries = {}
    for i, slab in enumerate(slabs):
        try:
            # 获取楼板ID
            slab_id = slab.id()

            # 获取楼板的外边界点
            boundary_points = loader.get_element_boundary_points(slab)

            if boundary_points:
                # 创建楼板多边形
                slab_polygon = Polygon(boundary_points)

                # 计算面积
                area = slab_polygon.area

                # 存储楼板几何信息
                slab_geometries[slab_id] = {
                    "polygon": slab_polygon,
                    "area": area,
                    "boundary_points": boundary_points,
                }

                print(f"楼板 #{slab_id} - 面积: {area:.2f} 平方米")
            else:
                print(f"无法获取楼板 #{slab_id} 的边界点")
        except Exception as e:
            print(f"处理楼板 #{i} 时发生错误: {e}")

    return slab_geometries


def classify_slabs(slab_geometries, external_outline):
    """将楼板分类为内部和外部"""
    internal_slabs = {}
    external_slabs = {}

    for slab_id, slab_data in slab_geometries.items():
        slab_polygon = slab_data["polygon"]

        # 判断楼板是否在外轮廓内
        # 使用重叠面积比例来决定归类
        intersection = slab_polygon.intersection(external_outline)
        intersection_area = intersection.area if not intersection.is_empty else 0

        # 如果楼板超过50%在外轮廓内，则视为内部楼板
        if intersection_area / slab_polygon.area > 0.5:
            internal_slabs[slab_id] = slab_data
        else:
            external_slabs[slab_id] = slab_data

    return internal_slabs, external_slabs


def calculate_areas(internal_slabs, external_slabs):
    """计算内部和外部楼板的总面积"""
    internal_area = sum(slab["area"] for slab in internal_slabs.values())
    external_area = sum(slab["area"] for slab in external_slabs.values())

    return internal_area, external_area


def plot_slabs_and_outline(
    slab_geometries, external_outline, internal_slabs, external_slabs
):
    """绘制楼板和外轮廓"""
    plt.figure(figsize=(12, 10))

    # 绘制外轮廓
    if external_outline:
        if hasattr(external_outline, "exterior"):
            xs, ys = external_outline.exterior.xy
            plt.plot(xs, ys, "g-", linewidth=3, label="外墙轮廓")

    # 绘制内部楼板
    for slab_id, slab_data in internal_slabs.items():
        polygon = slab_data["polygon"]
        x, y = polygon.exterior.xy
        plt.fill(x, y, "blue", alpha=0.3)
        plt.plot(x, y, "b-", linewidth=1)
        centroid = polygon.centroid
        plt.text(
            centroid.x,
            centroid.y,
            f"#{slab_id}\n内部",
            fontsize=8,
            ha="center",
            va="center",
        )

    # 绘制外部楼板
    for slab_id, slab_data in external_slabs.items():
        polygon = slab_data["polygon"]
        x, y = polygon.exterior.xy
        plt.fill(x, y, "red", alpha=0.3)
        plt.plot(x, y, "r-", linewidth=1)
        centroid = polygon.centroid
        plt.text(
            centroid.x,
            centroid.y,
            f"#{slab_id}\n外部",
            fontsize=8,
            ha="center",
            va="center",
        )

    plt.title("楼板分类与外墙轮廓")
    plt.axis("equal")
    plt.grid(True)

    # 添加图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", alpha=0.3, label="内部楼板"),
        Patch(facecolor="red", alpha=0.3, label="外部楼板"),
        plt.Line2D([0], [0], color="g", linewidth=3, label="外墙轮廓"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    # 保存图像
    plt.savefig("slabs_classification.png")
    print(f"已保存楼板分类图像到 'slabs_classification.png'")

    # 显示图像
    plt.show()


def main():
    # 实例化IFC加载器
    loader = IFCLoader()

    # 加载IFC文件
    if loader.load_file("outline wall_2.ifc"):
        print("IFC文件加载成功")

        # 获取墙体几何数据
        wall_geometries = loader.get_all_wall_geometries()

        # 获取外墙轮廓
        print("计算外墙轮廓...")
        external_outline, external_wall_ids = get_external_wall_outline(wall_geometries)

        if external_outline:
            print(f"外墙轮廓计算成功，面积: {external_outline.area:.2f} 平方米")

            # 获取楼板几何信息
            print("提取楼板几何信息...")
            slab_geometries = get_slab_geometry(loader)

            if slab_geometries:
                # 将楼板分类为内部和外部
                print("将楼板分类为内部和外部...")
                internal_slabs, external_slabs = classify_slabs(
                    slab_geometries, external_outline
                )

                # 计算内部和外部楼板的总面积
                internal_area, external_area = calculate_areas(
                    internal_slabs, external_slabs
                )

                # 打印结果
                print("\n楼板面积计算结果:")
                print(
                    f"内部楼板: {len(internal_slabs)} 个，总面积: {internal_area:.2f} 平方米"
                )
                print(
                    f"外部楼板: {len(external_slabs)} 个，总面积: {external_area:.2f} 平方米"
                )
                print(f"总楼板面积: {internal_area + external_area:.2f} 平方米")

                # 绘制结果
                plot_slabs_and_outline(
                    slab_geometries, external_outline, internal_slabs, external_slabs
                )
            else:
                print("未找到楼板几何信息")
        else:
            print("无法计算外墙轮廓")
    else:
        print("IFC文件加载失败")


if __name__ == "__main__":
    main()
