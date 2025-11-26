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

# --- 1. Group C 数据集 ---
# [修改点 1] 将所有的 "IfcOpeningElement" 替换为 "IfcBuildingElementProxy"
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
    # 下面全是 Proxy (原 Opening)
    (
        71,
        "IfcBuildingElementProxy",
        {"F1": "H>12m", "F2": "W>8m", "F3": "Penetrates multiple floors"},
        "Atrium void",
    ),
    (
        72,
        "IfcBuildingElementProxy",
        {"F1": "H>15m", "F2": "W<3m", "F3": "Contains elevator"},
        "Elevator shaft opening",
    ),
    (
        73,
        "IfcBuildingElementProxy",
        {"F1": "H>15m", "F2": "W<2m", "F3": "Contains stair"},
        "Stairwell opening",
    ),
    (
        74,
        "IfcBuildingElementProxy",
        {"F1": "H>12m", "F2": "aspect>10", "F3": "Vertical ductwork"},
        "HVAC shaft opening",
    ),
    (
        75,
        "IfcBuildingElementProxy",
        {"F1": "H>12m", "F2": "aspect>12", "F3": "Vertical piping"},
        "Plumbing shaft opening",
    ),
    (
        76,
        "IfcBuildingElementProxy",
        {"F1": "H>10m", "F2": "aspect>15", "F3": "Cable vertical shaft"},
        "Electrical shaft opening",
    ),
    (
        77,
        "IfcBuildingElementProxy",
        {"F1": "In slab", "F2": "D<0.5m", "F3": "Circular"},
        "Small service penetration",
    ),
    (
        78,
        "IfcBuildingElementProxy",
        {"F1": "In slab", "F2": "D>1m", "F3": "Rectangular"},
        "Large equipment opening",
    ),
    (
        79,
        "IfcBuildingElementProxy",
        {"F1": "Building boundary", "F2": "Narrow gap", "F3": "Vertical"},
        "Expansion joint",
    ),
    (
        80,
        "IfcBuildingElementProxy",
        {"F1": "In slab", "F2": "D<0.3m", "F3": "Round"},
        "Floor drain opening",
    ),
    (
        81,
        "IfcBuildingElementProxy",
        {"F1": "Underground exterior wall", "F2": "Light well", "F3": "-"},
        "Window well",
    ),
    (
        82,
        "IfcBuildingElementProxy",
        {"F1": "In slab", "F2": "Temporary", "F3": "Rigging"},
        "Equipment rigging opening",
    ),
    (
        83,
        "IfcBuildingElementProxy",
        {"F1": "Multi-floor", "F2": "Large void", "F3": "Central"},
        "Central void/atrium",
    ),
    (
        84,
        "IfcBuildingElementProxy",
        {"F1": "In slab", "F2": "Stair penetration", "F3": "-"},
        "Stair opening",
    ),
    (
        85,
        "IfcBuildingElementProxy",
        {"F1": "In slab", "F2": "Escalator penetration", "F3": "-"},
        "Escalator opening",
    ),
    (
        86,
        "IfcBuildingElementProxy",
        {"F1": "In slab", "F2": "Vertical circulation", "F3": "-"},
        "Vertical circulation void",
    ),
    (
        87,
        "IfcBuildingElementProxy",
        {"F1": "In roof", "F2": "Large area", "F3": "Glass"},
        "Glazed roof opening",
    ),
    (
        88,
        "IfcBuildingElementProxy",
        {"F1": "Between floors", "F2": "Vertical void", "F3": "Open"},
        "Open floor connection",
    ),
    (
        89,
        "IfcBuildingElementProxy",
        {"F1": "In slab", "F2": "Loading access", "F3": "-"},
        "Loading dock opening",
    ),
    (
        90,
        "IfcBuildingElementProxy",
        {"F1": "Partial floor", "F2": "Balcony edge", "F3": "-"},
        "Balcony void boundary",
    ),
]

# --- 2. 初始化 IFC 模型 (同上) ---
model = ifcopenshell.file()
project = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcProject", name="Group C Dataset (Visual)"
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


# --- 3. 几何解析器 (同上) ---
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

    if any(
        x in f_str
        for x in ["w>8", "large area", "large void", "atrium", "central void"]
    ):
        l, w = 10.0, 10.0
    elif any(x in f_str for x in ["w<2", "aspect>15", "cable"]):
        l, w = 1.5, 1.5
    elif any(x in f_str for x in ["w<3", "aspect>10", "hvac"]):
        l, w = 2.5, 2.5
    elif any(x in f_str for x in ["d<0.5", "d<0.3", "small"]):
        l, w = 0.4, 0.4

    if "stair" in f_str or "escalator" in f_str:
        l, w = 4.0, 2.0
    return l, w, h


# --- 4. 上下文模拟器 (适配 Proxy) ---
def simulate_context(
    model, element, function_name, features_dict, storey, body_context
):
    func = function_name.lower()

    # ---------------------------
    # A. 边界判定
    # ---------------------------
    boundary_type = None
    if any(
        x in func for x in ["shaft", "stairwell", "stair", "elevator", "duct", "pipe"]
    ):
        boundary_type = "IfcWall"
    elif any(x in func for x in ["atrium", "void", "balcony", "open"]):
        boundary_type = "IfcRailing"

    if boundary_type:
        boundary_elem = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class=boundary_type,
            name=f"Boundary_for_{element.Name}",
        )
        rep = ifcopenshell.api.run(
            "geometry.add_wall_representation",
            model,
            context=body_context,
            length=1,
            height=1,
            thickness=0.1,
        )
        ifcopenshell.api.run(
            "geometry.assign_representation",
            model,
            product=boundary_elem,
            representation=rep,
        )
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            relating_structure=storey,
            products=[boundary_elem],
        )

        # 仅 IfcSpace 支持 RelSpaceBoundary
        if element.is_a("IfcSpace"):
            rel = ifcopenshell.api.run(
                "root.create_entity",
                model,
                ifc_class="IfcRelSpaceBoundary",
                name="BoundaryRel",
            )
            rel.RelatingSpace = element
            rel.RelatedBuildingElement = boundary_elem
            rel.PhysicalOrVirtualBoundary = "PHYSICAL"

    # ---------------------------
    # B. 内容物判定
    # ---------------------------
    content_type = None
    if "elevator" in func or "lift" in func:
        content_type = "IfcTransportElement"
    elif "stair" in func:
        content_type = "IfcStair"
    elif "escalator" in func:
        content_type = "IfcTransportElement"
    elif any(
        x in func for x in ["hvac", "duct", "pipe", "drain", "cable", "electrical"]
    ):
        content_type = "IfcFlowSegment"

    if content_type:
        content_elem = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class=content_type,
            name=f"Content_inside_{element.Name}",
        )
        rep_c = ifcopenshell.api.run(
            "geometry.add_wall_representation",
            model,
            context=body_context,
            length=0.5,
            height=0.5,
            thickness=0.5,
        )
        ifcopenshell.api.run(
            "geometry.assign_representation",
            model,
            product=content_elem,
            representation=rep_c,
        )

        if element.is_a("IfcSpace"):
            ifcopenshell.api.run(
                "aggregate.assign_object",
                model,
                relating_object=element,
                products=[content_elem],
            )
        else:
            # 对于 Proxy，为了可视化，我们把内容物放在同一个 Storey 即可，位置重叠
            ifcopenshell.api.run(
                "spatial.assign_container",
                model,
                relating_structure=storey,
                products=[content_elem],
            )


# --- 5. 生成循环 ---
row_length = 5
spacing_x = 15.0
spacing_y = 15.0

print("正在生成可视化版 Group C IFC (Use Proxy)...")

for item in dataset:
    no, ifc_type, feat_dict, actual_func = item

    # 1. 创建实体
    name_str = f"No.{no}_{actual_func.replace(' ', '_').replace('/', '-')}"

    # [修改点 2] 为 Proxy 增加 PredefinedType='PROVISIONFORVOID'
    # 这样在 Revit 中虽然是常规模型，但属性上知道它是留洞
    predefined_type = None
    if ifc_type == "IfcBuildingElementProxy":
        predefined_type = "PROVISIONFORVOID"

    element = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class=ifc_type,
        name=name_str,
        predefined_type=predefined_type,
    )

    # 2. 写入属性
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
    # 对于 Proxy，我们把它生成为实心体块，这样在 Revit 里才看得到
    representation = ifcopenshell.api.run(
        "geometry.add_wall_representation",
        model,
        context=body,
        length=l,
        height=h,
        thickness=w,
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

    # 5. 空间层级
    if ifc_type == "IfcSpace":
        ifcopenshell.api.run(
            "aggregate.assign_object", model, relating_object=storey, products=[element]
        )
    else:
        # Proxy 和 Opening 一样，放入 Spatial Container
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            relating_structure=storey,
            products=[element],
        )

    # 6. 上下文环境
    simulate_context(model, element, actual_func, feat_dict, storey, body)

# --- 6. 保存 ---
filename = "group_c_dataset_viz.ifc"
model.write(filename)
print(f"成功生成: {filename}")
print("  - 所有 Openings 已转换为 IfcBuildingElementProxy (PROVISIONFORVOID)")
print("  - 可直接在 Revit 中链接或打开查看几何体块")
