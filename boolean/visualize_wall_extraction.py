import sys
import matplotlib.pyplot as plt
from pathlib import Path

# 修复导入路径问题
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))  # 添加父目录到路径

# 导入IFC加载器 - 修复导入路径
from ifc_loader import IFCLoader  # 直接导入，不需要boolean前缀


def visualize_raw_wall_extraction(ifc_path):
    """直接可视化从IFC文件提取的墙体中心线"""
    # 加载IFC文件
    loader = IFCLoader()
    if not loader.load_file(ifc_path):
        print("Failed to load IFC file")
        return

    # 获取墙体几何信息
    wall_geometries = loader.get_all_wall_geometries()

    # 创建图表
    plt.figure(figsize=(12, 12))
    plt.title("Wall Centerline Extraction from IFC")
    plt.axis("equal")
    plt.grid(True)

    # 遍历所有墙体并绘制
    for wall_id, geom in wall_geometries.items():
        if geom:
            if "is_curved" in geom and geom["is_curved"] and "curve_points" in geom:
                # 绘制弧形墙 - 红色
                curve_points = geom["curve_points"]
                if len(curve_points) > 1:
                    xs = [p[0] for p in curve_points]
                    ys = [p[1] for p in curve_points]
                    plt.plot(
                        xs,
                        ys,
                        "r-",
                        linewidth=2,
                        label=(
                            f"Curved Wall #{wall_id}"
                            if wall_id == list(wall_geometries.keys())[0]
                            else ""
                        ),
                    )

                    # 在曲线中点显示ID
                    mid_idx = len(curve_points) // 2
                    plt.text(
                        curve_points[mid_idx][0],
                        curve_points[mid_idx][1],
                        f"{wall_id}",
                        fontsize=12,
                        ha="center",
                    )

                    # 可视化每个曲线点
                    plt.scatter(xs, ys, color="red", s=20, alpha=0.5)
            else:
                # 绘制直线墙 - 蓝色
                start = geom["start"]
                end = geom["end"]
                plt.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    "b-",
                    linewidth=2,
                    label=(
                        f"Straight Wall #{wall_id}"
                        if wall_id == list(wall_geometries.keys())[0]
                        else ""
                    ),
                )

                # 在线段中点显示ID
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                plt.text(mid_x, mid_y, f"{wall_id}", fontsize=12, ha="center")

    plt.legend()
    plt.savefig("wall_centerline_extraction.png")
    print(f"Image saved to: wall_centerline_extraction.png")
    plt.show()


def visualize_curve_extraction_details(ifc_path):
    """详细可视化弧形墙的点提取过程"""
    # 加载IFC文件
    loader = IFCLoader()
    if not loader.load_file(ifc_path):
        print("Failed to load IFC file")
        return

    # 获取所有墙体
    walls = loader.get_all_walls()

    # 找出弧形墙
    curved_walls = []
    for wall in walls:
        geom = loader.get_wall_geometry(wall)
        if geom and "is_curved" in geom and geom["is_curved"]:
            curved_walls.append((wall, geom))

    if not curved_walls:
        print("No curved walls found")
        return

    # 为每个弧形墙创建单独的可视化
    for i, (wall, geom) in enumerate(curved_walls):
        plt.figure(figsize=(15, 10))
        wall_id = wall.id()
        plt.suptitle(f"Curved Wall #{wall_id} Extraction Details", fontsize=16)

        # 获取曲线点
        if "curve_points" in geom:
            curve_points = geom["curve_points"]

            # 绘制所有提取的点
            xs = [p[0] for p in curve_points]
            ys = [p[1] for p in curve_points]

            plt.subplot(1, 2, 1)
            plt.title("Original Curve Points Extraction")
            plt.plot(xs, ys, "r-", linewidth=2)
            plt.scatter(xs, ys, color="red", s=30)

            # 标记起点和终点
            plt.scatter(
                [curve_points[0][0]],
                [curve_points[0][1]],
                color="green",
                s=100,
                label="Start Point",
            )
            plt.scatter(
                [curve_points[-1][0]],
                [curve_points[-1][1]],
                color="blue",
                s=100,
                label="End Point",
            )

            # 添加点索引标签
            for idx, (x, y) in enumerate(zip(xs, ys)):
                if idx % max(1, len(curve_points) // 10) == 0:  # 每10%标记一个点
                    plt.text(x, y, f"{idx}", fontsize=8)

            plt.grid(True)
            plt.axis("equal")
            plt.legend()

            # 绘制直线参考和偏差
            plt.subplot(1, 2, 2)
            plt.title("Curve vs Straight Line Reference")

            # 画出理想直线
            start = curve_points[0]
            end = curve_points[-1]
            plt.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                "b--",
                linewidth=1,
                label="Straight Reference",
            )

            # 画出实际曲线
            plt.plot(xs, ys, "r-", linewidth=2, label="Actual Curve")

            # 计算每个点到参考直线的距离
            # 直线方程 ax + by + c = 0
            a = end[1] - start[1]
            b = start[0] - end[0]
            c = end[0] * start[1] - start[0] * end[1]
            line_len = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5

            max_dist = 0
            max_dist_idx = 0

            for idx, point in enumerate(curve_points):
                # 计算点到直线的距离
                dist = abs(a * point[0] + b * point[1] + c) / ((a**2 + b**2) ** 0.5)

                if dist > max_dist:
                    max_dist = dist
                    max_dist_idx = idx

                # 画出到直线的垂线
                # 计算在直线上的投影点
                t = -(a * point[0] + b * point[1] + c) / (a**2 + b**2)
                proj_x = point[0] + t * a
                proj_y = point[1] + t * b

                # 每隔几个点画一条垂线，避免图形过于拥挤
                if idx % max(1, len(curve_points) // 20) == 0:
                    plt.plot(
                        [point[0], proj_x], [point[1], proj_y], "g-", linewidth=0.5
                    )

            # 标记最大偏离点
            max_point = curve_points[max_dist_idx]
            plt.scatter(
                [max_point[0]],
                [max_point[1]],
                color="purple",
                s=100,
                label=f"Max Deviation Point (Dist={max_dist:.3f}m)",
            )

            plt.grid(True)
            plt.axis("equal")
            plt.legend()

            plt.savefig(f"curved_wall_{wall_id}_analysis.png")
            print(f"Image saved to: curved_wall_{wall_id}_analysis.png")

    plt.show()


# 主函数
if __name__ == "__main__":
    # 使用您的IFC文件路径
    ifc_path = "overall.ifc"

    print(f"Analyzing IFC file: {ifc_path}")
    visualize_raw_wall_extraction(ifc_path)
    visualize_curve_extraction_details(ifc_path)
