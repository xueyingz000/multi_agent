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
project = ifcopenshell.api.run(
    "root.create_entity",
    model,
    ifc_class="IfcProject",
    name="Group C Dataset (Visible)",
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

ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=project, products=[site]
)
ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=site, products=[building]
)
ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=building, products=[storey]
)


# --- 3. 几何解析器 ---
def parse_geometry_group_c(features_dict):
    f_str = " ".join(features_dict.values()).lower()
    l, w, h = 1.0, 1.0, 0.3

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
        h = 0.3
    elif "in roof" in f_str:
        h = 0.2

    if (
        "w>8" in f_str
        or "large area" in f_str
        or "large void" in f_str
        or "atrium" in f_str
    ):
        l, w = 10.0, 10.0
    elif "w<2" in f_str or "aspect>15" in f_str:
        l, w = 1.5, 1.5
    elif "w<3" in f_str or "aspect>10" in f_str:
        l, w = 2.5, 2.5
    elif "d<0.5" in f_str or "d<0.3" in f_str or "small" in f_str:
        l, w = 0.4, 0.4
    elif "d>1" in f_str:
        l, w = 1.5, 1.5
    elif "narrow gap" in f_str:
        l, w = 10.0, 0.2
    if "stair" in f_str or "escalator" in f_str:
        l, w = 4.0, 2.0

    return l, w, h


# --- 4. 生成循环 ---
row_length = 5
spacing_x = 15.0
spacing_y = 15.0

print("正在生成可见版 Group C IFC 文件...")

for item in dataset:
    no, original_ifc_type, feat_dict, actual_func = item

    # 【关键修改】如果它是 Opening，我们强制改为 Proxy 以便可视化
    # 但我们会在 Pset 属性里记录它原本应该是 Opening
    final_ifc_class = original_ifc_type
    if original_ifc_type == "IfcOpeningElement":
        final_ifc_class = "IfcBuildingElementProxy"

    name_str = f"No.{no}_{actual_func.replace(' ', '_').replace('/', '-')}"
    element = ifcopenshell.api.run(
        "root.create_entity", model, ifc_class=final_ifc_class, name=name_str
    )

    # 写入属性集
    pset = ifcopenshell.api.run(
        "pset.add_pset", model, product=element, name="Pset_SimulatedData"
    )
    props = {
        "Feature1": feat_dict["F1"],
        "Feature2": feat_dict["F2"],
        "Feature3": feat_dict["F3"],
        "ActualFunction": actual_func,
        "OriginalType": original_ifc_type,  # 记录原始类型
    }
    ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties=props)

    # 生成几何
    l, w, h = parse_geometry_group_c(feat_dict)

    representation = ifcopenshell.api.run(
        "geometry.add_wall_representation",
        model,
        context=body,
        length=l,
        height=h,
        thickness=w,
    )

    # 定位
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

    # 空间层级
    if final_ifc_class == "IfcSpace":
        ifcopenshell.api.run(
            "aggregate.assign_object", model, relating_object=storey, products=[element]
        )
    else:
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            relating_structure=storey,
            products=[element],
        )

filename = "group_c_dataset_visible.ifc"
model.write(filename)
print(f"成功生成: {filename}")
print("所有 IfcOpeningElement 已转换为 IfcBuildingElementProxy 以便查看。")
