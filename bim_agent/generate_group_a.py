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

# --- 1. 数据集定义 ---
dataset = [
    (1, "IfcSlab", "FLOOR", False),
    (2, "IfcSlab", "ROOF", None),
    (3, "IfcSlab", "LANDING", False),
    (4, "IfcSlab", "BASESLAB", False),
    (5, "IfcSlab", "FLOOR", True),
    (6, "IfcSlab", "LANDING", True),
    (7, "IfcSpace", "PARKING", False),
    (8, "IfcSpace", "EXTERNAL", True),
    (9, "IfcSpace", "INTERNAL", False),
    (10, "IfcSpace", "GFA", None),
    (11, "IfcSpace", "PARKING", True),
    (12, "IfcWall", "STANDARD", True),
    (13, "IfcWall", "STANDARD", False),
    (14, "IfcWall", "SHEAR", False),
    (15, "IfcWall", "PLUMBINGWALL", False),
    (16, "IfcCurtainWall", "USERDEFINED", True),
    (17, "IfcColumn", "COLUMN", None),
    (18, "IfcColumn", "PILASTER", None),
    (19, "IfcBeam", "BEAM", None),
    (20, "IfcBeam", "LINTEL", None),
    (21, "IfcStair", "STRAIGHT_RUN_STAIR", None),
    (22, "IfcStair", "QUARTER_WINDING_STAIR", None),
    (23, "IfcRamp", "STRAIGHT_RAMP", None),
    (24, "IfcRoof", "FLAT_ROOF", None),
    (25, "IfcRoof", "SHED_ROOF", None),
    (26, "IfcRoof", "GABLE_ROOF", None),
    (27, "IfcCovering", "CEILING", None),
    (28, "IfcCovering", "FLOORING", None),
    (29, "IfcRailing", "HANDRAIL", None),
    (30, "IfcRailing", "GUARDRAIL", None),
]

# --- 2. 初始化 IFC 模型 ---
model = ifcopenshell.file()

# 创建项目层级
project = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcProject", name="Group A Dataset"
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

# 构建空间树
ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=project, products=[site]
)
ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=site, products=[building]
)
ifcopenshell.api.run(
    "aggregate.assign_object", model, relating_object=building, products=[storey]
)


# --- 3. 辅助函数：获取 Pset 名称 ---
def get_pset_common_name(ifc_type):
    type_map = {
        "IfcWall": "Pset_WallCommon",
        "IfcCurtainWall": "Pset_CurtainWallCommon",
        "IfcSlab": "Pset_SlabCommon",
        "IfcSpace": "Pset_SpaceCommon",
        "IfcColumn": "Pset_ColumnCommon",
        "IfcBeam": "Pset_BeamCommon",
        "IfcStair": "Pset_StairCommon",
        "IfcRamp": "Pset_RampCommon",
        "IfcRoof": "Pset_RoofCommon",
        "IfcCovering": "Pset_CoveringCommon",
        "IfcRailing": "Pset_RailingCommon",
    }
    return type_map.get(ifc_type, f"Pset_{ifc_type[3:]}Common")


# --- 4. 循环生成构件 ---
row_length = 5
spacing = 3.0

print("正在生成 IFC 文件...")

for item in dataset:
    no, ifc_type, predefined_type, is_external = item

    # 1. 创建实体
    element = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class=ifc_type,
        name=f"No.{no}_{ifc_type}_{predefined_type}",
    )

    # 2. 设置 PredefinedType
    if predefined_type:
        type_class = f"{ifc_type}Type"
        type_element = ifcopenshell.api.run(
            "root.create_entity", model, ifc_class=type_class, name=f"Type_for_No.{no}"
        )

        try:
            type_element.PredefinedType = predefined_type
        except:
            pass

        ifcopenshell.api.run(
            "type.assign_type",
            model,
            related_objects=[element],
            relating_type=type_element,
        )

        try:
            element.PredefinedType = predefined_type
        except:
            pass

    # 3. 设置 Pset
    if is_external is not None:
        pset_name = get_pset_common_name(ifc_type)
        pset = ifcopenshell.api.run(
            "pset.add_pset", model, product=element, name=pset_name
        )
        ifcopenshell.api.run(
            "pset.edit_pset", model, pset=pset, properties={"IsExternal": is_external}
        )

    # 4. 创建几何形状
    x = (float(no - 1) % row_length) * spacing
    y = (float(no - 1) // row_length) * spacing

    l, w, h = 1.0, 1.0, 2.0
    if ifc_type == "IfcSlab":
        h = 0.3
    if ifc_type == "IfcSpace":
        l, w, h = 2.0, 2.0, 2.5
    if ifc_type == "IfcWall":
        w = 0.2

    # 注意：add_wall_representation 本质是创建拉伸体，对于 Space 来说虽然不语义化，
    # 但为了能在 3D 查看器里看到一个占位块，这样做是可以的。
    representation = ifcopenshell.api.run(
        "geometry.add_wall_representation",
        model,
        context=body,
        length=l,
        height=h,
        thickness=w,
    )

    ifcopenshell.api.run(
        "geometry.edit_object_placement",
        model,
        product=element,
        matrix=[
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
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

    # 5. 分配到楼层 (【修正部分】)
    if ifc_type == "IfcSpace":
        # Space 是空间结构的一部分，使用 aggregate
        ifcopenshell.api.run(
            "aggregate.assign_object", model, relating_object=storey, products=[element]
        )
    else:
        # 物理构件（墙、板等）包含在空间内，使用 spatial container
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            relating_structure=storey,
            products=[element],
        )

# --- 5. 保存文件 ---
filename = "group_a_dataset.ifc"
model.write(filename)
print(f"成功生成文件: {filename}")
print(f"包含 {len(dataset)} 个构件。")
