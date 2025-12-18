import ifcopenshell
import ifcopenshell.geom
import open3d as o3d
import numpy as np
import multiprocessing
import tkinter as tk
from tkinter import filedialog
import os


def load_ifc_geometry(ifc_file_path):
    """
    解析 IFC 文件并转换为 Open3D 的 Mesh 对象列表
    """
    print(f"正在加载模型: {ifc_file_path} ...")

    # 打开 IFC 文件
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
    except Exception as e:
        print(f"打开文件失败: {e}")
        return []

    # 配置几何转换设置
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)  # 使用世界坐标

    # 创建几何迭代器 (利用多核加速)
    # exclude=['IfcSpace', 'IfcOpeningElement'] 意味着不渲染空间体量和开洞占位符
    iterator = ifcopenshell.geom.iterator(
        settings,
        ifc_file,
        multiprocessing.cpu_count(),
        exclude=["IfcSpace", "IfcOpeningElement"],
    )

    meshes = []

    # 遍历 IFC 中的构件
    if iterator.initialize():
        while True:
            shape = iterator.get()

            # 提取顶点和面
            faces = shape.geometry.faces
            verts = shape.geometry.verts

            # IfcOpenShell 返回的是扁平列表，需要 reshape
            # 顶点是 (x, y, z, x, y, z ...)
            grouped_verts = np.array(verts).reshape((-1, 3))
            # 面是 (v1, v2, v3, v1, v2, v3 ...)
            grouped_faces = np.array(faces).reshape((-1, 3))

            # 创建 Open3D 网格对象
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(grouped_verts)
            mesh.triangles = o3d.utility.Vector3iVector(grouped_faces)

            # 计算法线以获得更好的光照效果
            mesh.compute_vertex_normals()

            # 以此给构件上色（这里简单的根据ID生成随机颜色，或者使用统一灰色）
            # 实际应用中可以解析 IFC 的材质颜色
            color = np.random.rand(3)
            mesh.paint_uniform_color(color)

            meshes.append(mesh)

            if not iterator.next():
                break

    print(f"加载完成，共解析 {len(meshes)} 个构件。")
    return meshes


def main():
    # 1. 创建一个简单的文件选择窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    print("请在弹出的窗口中选择 .ifc 文件")
    file_path = filedialog.askopenfilename(
        title="选择 IFC 文件", filetypes=[("IFC Files", "*.ifc")]
    )

    if not file_path:
        print("未选择文件，程序退出。")
        return

    # 2. 解析几何数据
    geometry_meshes = load_ifc_geometry(file_path)

    if not geometry_meshes:
        print("未能提取到几何信息。")
        return

    # 3. 启动可视化窗口
    print("正在启动 3D 查看器...")
    print("操作说明:")
    print(" - 左键拖动: 旋转")
    print(" - 滚轮: 缩放")
    print(" - Shift + 左键拖动: 平移")
    print(" - Ctrl + 左键拖动: 调整光照")

    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"IFC Viewer - {os.path.basename(file_path)}",
        width=1280,
        height=720,
    )

    # 添加所有网格到场景
    for mesh in geometry_meshes:
        vis.add_geometry(mesh)

    # 添加坐标轴 (X:红, Y:绿, Z:蓝) - 原点大小为 2 米
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    vis.add_geometry(axis)

    # 运行渲染循环
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
