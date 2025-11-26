import ifcopenshell
import ifcopenshell.api.root
import ifcopenshell.api.unit
import ifcopenshell.api.context
import ifcopenshell.api.project
import ifcopenshell.api.spatial
import ifcopenshell.api.geometry
import ifcopenshell.api.aggregate
import ifcopenshell.api.type
import ifcopenshell.api.pset

# --- 1. Group C 数据集 (No. 66 - 90) ---
# 格式: (No, IfcType, Features_Dict, ActualFunction)
dataset = [
    (
        66,
        "IfcSpace",
        {"F1": "H>20m", "F2": "W<3m", "F3": "Contains elevator"},
        "Elevator shaft",
    ),
    (
        67,
        "IfcSpace",
        {"F1": "H>15m", "F2": "W<2m", "F3": "Contains stair"},
        "Stairwell",
    ),
    (
        68,
        "IfcSpace",
        {"F1": "H>15m", "F2": "aspect>10", "F3": "Vertical ductwork"},
        "HVAC shaft",
    ),
    (
        69,
        "IfcSpace",
        {"F1": "H>15m", "F2": "aspect>15", "F3": "Cable vertical shaft"},
        "Electrical shaft",
    ),
    (70, "IfcSpace", {"F1": "H>12m", "F2": "W>8m", "F3": "floor_count>=2"}, "Atrium"),
    (
        71,
        "IfcOpeningElement",
        {"F1": "H>12m", "F2": "W>8m", "F3": "Penetrates multiple floors"},
        "Atrium void",
    ),
    (
        72,
        "IfcOpeningElement",
        {"F1": "H>15m", "F2": "W<3m", "F3": "Contains elevator"},
        "Elevator shaft opening",
    ),
    (
        73,
        "IfcOpeningElement",
        {"F1": "H>15m", "F2": "W<2m", "F3": "Contains stair"},
        "Stairwell opening",
    ),
    (
        74,
        "IfcOpeningElement",
        {"F1": "H>12m", "F2": "aspect>10", "F3": "Vertical ductwork"},
        "HVAC shaft opening",
    ),
    (
        75,
        "IfcOpeningElement",
        {"F1": "H>12m", "F2": "aspect>12", "F3": "Vertical piping"},
        "Plumbing shaft opening",
    ),
    (
        76,
        "IfcOpeningElement",
        {"F1": "H>10m", "F2": "aspect>15", "F3": "Cable vertical shaft"},
        "Electrical shaft opening",
    ),
    (
        77,
        "IfcOpeningElement",
        {"F1": "In slab", "F2": "D<0.5m", "F3": "Circular"},
        "Small service penetration",
    ),
    (
        78,
        "IfcOpeningElement",
        {"F1": "In slab", "F2": "D>1m", "F3": "Rectangular"},
        "Large equipment opening",
    ),
    (
        79,
        "IfcOpeningElement",
        {"F1": "Building boundary", "F2": "Narrow gap", "F3": "Vertical"},
        "Expansion joint",
    ),
    (
        80,
        "IfcOpeningElement",
        {"F1": "In slab", "F2": "D<0.3m", "F3": "Round"},
        "Floor drain opening",
    ),
    (
        81,
        "IfcOpeningElement",
        {"F1": "Underground exterior wall", "F2": "Light well", "F3": "-"},
        "Window well",
    ),
    (
        82,
        "IfcOpeningElement",
        {"F1": "In slab", "F2": "Temporary", "F3": "Rigging"},
        "Equipment rigging opening",
    ),
    (
        83,
        "IfcOpeningElement",
        {"F1": "Multi-floor", "F2": "Large void", "F3": "Central"},
        "Central void/atrium",
    ),
    (
        84,
        "IfcOpeningElement",
        {"F1": "In slab", "F2": "Stair penetration", "F3": "-"},
        "Stair opening",
    ),
    (
        85,
        "IfcOpeningElement",
        {"F1": "In slab", "F2": "Escalator penetration", "F3": "-"},
        "Escalator opening",
    ),
    (
        86,
        "IfcOpeningElement",
        {"F1": "In slab", "F2": "Vertical circulation", "F3": "-"},
        "Vertical circulation void",
    ),
    (
        87,
        "IfcOpeningElement",
        {"F1": "In roof", "F2": "Large area", "F3": "Glass"},
        "Glazed roof opening",
    ),
    (
        88,
        "IfcOpeningElement",
        {"F1": "Between floors", "F2": "Vertical void", "F3": "Open"},
        "Open floor connection",
    ),
    (
        89,
        "IfcOpeningElement",
        {"F1": "In slab", "F2": "Loading access", "F3": "-"},
        "Loading dock opening",
    ),
    (
        90,
        "IfcOpeningElement",
        {"F1": "Partial floor", "F2": "Balcony edge", "F3": "-"},
        "Balcony void boundary",
    ),
]

# --- 2. 初始化 IFC 模型 ---
model = ifcopenshell.file()

# 基础层级
project = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcProject", name="Group C Dataset"
)
ifcopenshell.api.run("unit.assign_unit", model)
context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
body = ifcopenshell.api.run(
    "context.add_context",
    model,
    context_type="Model",
    context_identifier="Body",
    target_view="MODEL_VIEW",
    parent=context,
)

site = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcSite", name="Default Site"
)
building = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcBuilding", name="Test Building"
)
storey = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcBuildingStorey", name="Level 1"
)

# 构建树状结构
ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=project, products=[site]
)
ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=site, products=[building]
)
ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=building, products=[storey]
)


# --- 3. 几何解析器 (针对 Group C 优化) ---
def parse_geometry_group_c(features_dict):
    """
    解析特征字典，返回 (Length, Width, Height)
    """
    # 拼接所有特征字符串以便搜索
    f_str = " ".join(features_dict.values()).lower()

    # 默认值 (普通开洞)
    l, w, h = 1.0, 1.0, 0.3

    # --- 1. 高度逻辑 (Z轴) ---
    if "h>20" in f_str:
        h = 22.0
    elif "h>15" in f_str:
        h = 16.0
    elif "h>12" in f_str:
        h = 13.0
    elif "h>10" in f_str:
        h = 11.0
    elif "multi-floor" in f_str:
        h = 10.0
    elif "vertical void" in f_str:
        h = 6.0
    elif "in slab" in f_str or "partial floor" in f_str:
        h = 0.3  # 板厚
    elif "in roof" in f_str:
        h = 0.2

    # --- 2. 宽度/长度逻辑 (XY平面) ---
    # Atrium / Large void
    if (
        "w>8" in f_str
        or "large area" in f_str
        or "large void" in f_str
        or "atrium" in f_str
    ):
        l, w = 10.0, 10.0
    # Shafts (Narrow)
    elif "w<2" in f_str or "aspect>15" in f_str:
        l, w = 1.5, 1.5
    elif "w<3" in f_str or "aspect>10" in f_str:
        l, w = 2.5, 2.5
    # Small penetrations
    elif "d<0.5" in f_str or "d<0.3" in f_str or "small" in f_str:
        l, w = 0.4, 0.4
    elif "d>1" in f_str:
        l, w = 1.5, 1.5
    # Narrow Gap
    elif "narrow gap" in f_str:
        l, w = 10.0, 0.2

    # Stair/Escalator (长方形)
    if "stair" in f_str or "escalator" in f_str:
        l, w = 4.0, 2.0

    return l, w, h


# --- 4. 生成循环 ---
row_length = 5
spacing_x = 15.0
spacing_y = 15.0

print("正在生成 Group C IFC 文件...")

for item in dataset:
    no, ifc_type, feat_dict, actual_func = item

    # 1. 创建实体
    name_str = f"No.{no}_{actual_func.replace(' ', '_').replace('/', '-')}"
    element = ifcopenshell.api.run(
        "root.create_entity", model, ifc_class=ifc_type, name=name_str
    )

    # 2. 写入自定义属性集 (Pset_SimulatedData)
    # 将字典扁平化存入，供 Agent 分析
    pset = ifcopenshell.api.run(
        "pset.add_pset", model, product=element, name="Pset_SimulatedData"
    )
    props = {
        "Feature1": feat_dict["F1"],
        "Feature2": feat_dict["F2"],
        "Feature3": feat_dict["F3"],
        "ActualFunction": actual_func,
    }
    ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties=props)

    # 3. 生成几何
    l, w, h = parse_geometry_group_c(feat_dict)

    # 为了可视化 Opening，我们将其作为普通的体块生成
    # 这样在浏览器里是一个实体的块，代表“这里是洞的体积”
    thickness_val = w  # 借用 thickness 参数传递宽度

    representation = ifcopenshell.api.run(
        "geometry.add_wall_representation",
        model,
        context=body,
        length=l,
        height=h,
        thickness=thickness_val,
    )

    # 4. 定位
    idx = no - 66
    pos_x = (float(idx) % row_length) * spacing_x
    pos_y = (float(idx) // row_length) * spacing_y

    ifcopenshell.api.run(
        "geometry.edit_object_placement",
        model,
        product=element,
        matrix=[
            [1.0, 0.0, 0.0, pos_x],
            [0.0, 1.0, 0.0, pos_y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

    ifcopenshell.api.run(
        "geometry.assign_representation",
        model,
        product=element,
        representation=representation,
    )

    # 5. 空间层级分配
    # IfcSpace 属于 aggregate (空间聚合)
    if ifc_type == "IfcSpace":
        ifcopenshell.api.run(
            "aggregate.assign_object", model, relating_object=storey, products=[element]
        )
    # IfcOpeningElement 通常属于构件，但这里作为独立展示，放入楼层容器 (Spatial)
    else:
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            relating_structure=storey,
            products=[element],
        )

# --- 5. 保存 ---
filename = "group_c_dataset.ifc"
model.write(filename)
print(f"成功生成: {filename}")
print(f"包含 No.66 到 No.90 共 {len(dataset)} 个构件。")
