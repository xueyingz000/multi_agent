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

# --- 1. Group D 数据集 (No. 91 - 100) ---
dataset = [
    (
        91,
        "Semi-outdoor space",
        "IfcSpace",
        "EXTERNAL=ambiguous",
        "IfcSlab",
        "IsExternal=TRUE, HasRoof=Partial",
        "Has partial enclosure + roof overhang",
    ),
    (
        92,
        "Sunken plaza",
        "IfcSpace",
        "Elevation<0",
        "IfcSlab",
        "IsExternal=TRUE, BelowGrade",
        "Below grade + open to sky",
    ),
    (
        93,
        "Covered parking",
        "IfcSpace",
        "PARKING + HasRoof",
        "IfcSlab",
        "PARKING + Covered",
        "Has roof but open sides",
    ),
    (
        94,
        "Mechanical penthouse",
        "IfcSpace",
        "Rooftop equipment",
        "IfcSlab",
        "ROOF + Equipment area",
        "On roof + enclosed vs open",
    ),
    (
        95,
        "Double-height lobby",
        "IfcSpace",
        "H>6m, single space",
        "IfcSlab",
        "Two IfcSlabs (upper void)",
        "Tall ceiling vs mezzanine potential",
    ),
    (
        96,
        "Enclosed bridge",
        "IfcSpace",
        "connecting buildings",
        "IfcSlab",
        "Bridge + IsEnclosed",
        "Between buildings + heated",
    ),
    (
        97,
        "Light well",
        "IfcSpace",
        "basement",
        "IfcOpeningElement",
        "-",
        "Underground exterior wall + well",
    ),
    (
        98,
        "Equipment opening",
        "IfcOpeningElement",
        "-",
        "IfcSlab",
        "HasOpening",
        "In slab + small hole",
    ),
    (
        99,
        "Bridge space",
        "IfcSpace",
        "-",
        "IfcSlab",
        "Bridge",
        "Connecting buildings + elevated",
    ),
    (
        100,
        "Central void",
        "IfcSpace",
        "multi-floor",
        "IfcOpeningElement",
        "void",
        "Multi-floor + central + large",
    ),
]

# --- 2. 初始化 IFC 模型 ---
model = ifcopenshell.file()
project = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcProject", name="Group D Dataset"
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
storey_ground = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcBuildingStorey", name="Level 1 (Ground)"
)
storey_basement = ifcopenshell.api.run(
    "root.create_entity",
    model,
    ifc_class="IfcBuildingStorey",
    name="Level 0 (Basement)",
)
storey_roof = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcBuildingStorey", name="Level Roof"
)

ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=project, products=[site]
)
ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=site, products=[building]
)
ifcopenshell.api.run(
    "aggregate.assign_object",
    model,
    relating_object=building,
    products=[storey_ground, storey_basement, storey_roof],
)


# --- 3. 几何逻辑 ---
def get_geometry_and_elevation(desc_str, func_name, ifc_type):
    desc = desc_str.lower()
    func = func_name.lower()
    l, w, h = 5.0, 5.0, 3.0
    z_offset = 0.0

    if (
        "below" in desc
        or "sunken" in func
        or "basement" in desc
        or "underground" in desc
    ):
        z_offset = -3.0
    if "roof" in desc or "penthouse" in func:
        z_offset = 10.0
    if "elevated" in desc or "bridge" in func:
        z_offset = 5.0

    if "h>6m" in desc or "double-height" in func or "multi-floor" in desc:
        h = 8.0
    if ifc_type == "IfcSlab":
        h = 0.3
    if "bridge" in func:
        l, w = 15.0, 3.0
    if ifc_type == "IfcOpeningElement":
        l, w, h = 2.0, 2.0, 0.5
        if "central void" in func:
            l, w, h = 10.0, 10.0, 10.0

    return (l, w, h), z_offset


# --- 4. 生成循环 ---
spacing_x = 20.0
spacing_sub = 6.0

print("正在生成 Group D IFC 文件...")

for i, item in enumerate(dataset):
    no, func_name, type1, desc1, type2, desc2, discriminator = item

    expressions = [(1, type1, desc1), (2, type2, desc2)]

    for exp_id, original_type, desc in expressions:
        # 【关键修改】不再进行 Proxy 转换，直接使用原始类型 (如 IfcOpeningElement)
        # 这确保了 Agent 测试时的类型判断逻辑是真实的

        name_str = f"No.{no}_Exp{exp_id}_{func_name.replace(' ', '_')}"
        element = ifcopenshell.api.run(
            "root.create_entity", model, ifc_class=original_type, name=name_str
        )

        # 属性集
        pset = ifcopenshell.api.run(
            "pset.add_pset", model, product=element, name="Pset_SimulatedData"
        )
        props = {
            "Function": func_name,
            "ExpressionID": f"Expression {exp_id}",
            "Description": desc,
            "OriginalType": original_type,
            "Discriminator": discriminator,
        }
        ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties=props)

        # 几何
        (l, w, h), z = get_geometry_and_elevation(desc, func_name, original_type)
        thick_val = w
        representation = ifcopenshell.api.run(
            "geometry.add_wall_representation",
            model,
            context=body,
            length=l,
            height=h,
            thickness=thick_val,
        )

        # 定位
        pos_x = i * spacing_x
        pos_y = (exp_id - 1) * spacing_sub

        ifcopenshell.api.run(
            "geometry.edit_object_placement",
            model,
            product=element,
            matrix=[
                [1.0, 0.0, 0.0, pos_x],
                [0.0, 1.0, 0.0, pos_y],
                [0.0, 0.0, 1.0, z],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

        ifcopenshell.api.run(
            "geometry.assign_representation",
            model,
            product=element,
            representation=representation,
        )

        # 层级
        target_storey = storey_ground
        if z < -1.0:
            target_storey = storey_basement
        if z > 8.0:
            target_storey = storey_roof

        # 空间使用聚合，物理构件(含Opening)使用空间包含
        if original_type == "IfcSpace":
            ifcopenshell.api.run(
                "aggregate.assign_object",
                model,
                relating_object=target_storey,
                products=[element],
            )
        else:
            ifcopenshell.api.run(
                "spatial.assign_container",
                model,
                relating_structure=target_storey,
                products=[element],
            )

# --- 5. 保存 ---
filename = "group_d_dataset.ifc"
model.write(filename)
print(f"成功生成: {filename}")
print(
    "此文件包含真实的 IfcOpeningElement 类型。在 Viewer 中可能不可见，但适合 Agent 代码测试。"
)
