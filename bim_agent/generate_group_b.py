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
import math

# --- 1. Group B 数据集 (No. 31 - 65) ---
# 格式: (No, IfcType, PredefinedType, Features, Rule, ActualFunction)
dataset = [
    (
        31,
        "IfcSpace",
        "INTERNAL",
        "H<4m, Area<30m², equipment",
        "Normal height+small+equipment",
        "Mechanical room",
    ),
    (
        32,
        "IfcSpace",
        "INTERNAL",
        "H<4m, is_basement=T, refuse",
        "Underground+refuse",
        "Refuse room",
    ),
    (
        33,
        "IfcSpace",
        "INTERNAL",
        "H<4m, electrical equipment",
        "Normal height+electrical",
        "Electrical room",
    ),
    (
        34,
        "IfcSpace",
        "INTERNAL",
        "H<4m, fire equipment",
        "Normal height+fire system",
        "Fire control room",
    ),
    (
        35,
        "IfcSpace",
        "INTERNAL",
        "H<4m, plumbing fixtures",
        "Normal height+fixtures",
        "Toilet room",
    ),
    (
        36,
        "IfcSpace",
        "INTERNAL",
        "H<4m, refrigeration",
        "Normal height+cold storage",
        "Cold storage",
    ),
    (
        37,
        "IfcSpace",
        "INTERNAL",
        "H<4m, server racks",
        "Normal height+IT equipment",
        "Data center",
    ),
    (
        38,
        "IfcSpace",
        "INTERNAL",
        "H<4m, clean rating",
        "Normal height+clean",
        "Clean room",
    ),
    (
        39,
        "IfcSpace",
        "INTERNAL",
        "H>4m, W>20m, Area>200m²",
        "High ceiling+large",
        "Assembly space",
    ),
    (
        40,
        "IfcSpace",
        "INTERNAL",
        "H<4m, Area>100m², open plan",
        "Normal+large+open",
        "Open office",
    ),
    (
        41,
        "IfcSpace",
        "INTERNAL",
        "H<6m, is_basement=T, Area>500m²",
        "Underground+large area",
        "Underground parking",
    ),
    (
        42,
        "IfcSlab",
        "USERDEFINED",
        "is_basement=T, Area>1000m²",
        "Underground+large",
        "Parking deck",
    ),
    (
        43,
        "IfcSlab",
        "USERDEFINED",
        "is_top_floor=T, thickness>500mm",
        "Roof+super thick",
        "Helipad",
    ),
    (
        44,
        "IfcSlab",
        "USERDEFINED",
        "is_external=T, cantilever",
        "External+cantilever",
        "Balcony",
    ),
    (
        45,
        "IfcSlab",
        "USERDEFINED",
        "Mid-floor, partial slab",
        "Partial+mid-level",
        "Mezzanine floor",
    ),
    (
        46,
        "IfcSlab",
        "USERDEFINED",
        "Small area, equipment load",
        "Partial+heavy load",
        "Equipment platform",
    ),
    (
        47,
        "IfcSlab",
        "USERDEFINED",
        "Grade level, loading area",
        "Boundary+loading",
        "Loading dock",
    ),
    (
        48,
        "IfcSlab",
        "USERDEFINED",
        "is_basement=T, on grade",
        "Underground+foundation",
        "Basement slab",
    ),
    (49, "IfcSlab", "FLOOR", "Contains pool", "Floor+pool", "Pool floor slab"),
    (
        50,
        "IfcSlab",
        "USERDEFINED",
        "Transfer level, thick",
        "Transfer+thick",
        "Transfer slab",
    ),
    (51, "IfcSlab", "USERDEFINED", "Exterior, grade", "Exterior+grade", "Patio/plaza"),
    (52, "IfcSlab", "ROOF", "Vegetated surface", "Roof+green", "Green roof"),
    (
        53,
        "IfcSlab",
        "USERDEFINED",
        "Canopy, no enclosure",
        "Canopy+open",
        "Canopy deck",
    ),
    (
        54,
        "IfcSlab",
        "USERDEFINED",
        "Covered walkway",
        "Covered+pedestrian",
        "Covered walkway",
    ),
    (
        55,
        "IfcSlab",
        "USERDEFINED",
        "Connecting buildings, elevated",
        "Bridge+elevated",
        "Pedestrian bridge",
    ),
    (
        56,
        "IfcSpace",
        "USERDEFINED",
        "H<6m, smoke protection",
        "Smoke protection+AdjacentToStair",
        "Refuge area",
    ),
    (57, "IfcSpace", "INTERNAL", "H<4m, corridor", "Linear space", "Corridor"),
    (
        58,
        "IfcSpace",
        "INTERNAL",
        "H<4m, elevator adjacent",
        "Adjacent to elevator",
        "Elevator lobby",
    ),
    (
        59,
        "IfcWall",
        "USERDEFINED",
        "H>6m, independent foundation",
        "Super tall+independent",
        "Fire wall",
    ),
    (60, "IfcWall", "STANDARD", "Unit separation", "Between units", "Demising wall"),
    (
        61,
        "IfcWall",
        "USERDEFINED",
        "is_basement=T, retaining",
        "Underground+retaining",
        "Retaining wall",
    ),
    (62, "IfcWall", "USERDEFINED", "is_top_floor=T, low height", "Roof+low", "Parapet"),
    (63, "IfcWall", "USERDEFINED", "Thin+glass", "Thin+glass", "Curtain wall"),
    (
        64,
        "IfcWall",
        "USERDEFINED",
        "Acoustic requirement",
        "Sound performance",
        "Acoustic wall",
    ),
    (
        65,
        "IfcStair",
        "USERDEFINED",
        "Symmetric layout, shared shaft",
        "Symmetric+shared",
        "Scissor stair",
    ),
]

# --- 2. 初始化 IFC 模型 ---
model = ifcopenshell.file()

# 基础层级
project = ifcopenshell.api.run(
    "root.create_entity", model, ifc_class="IfcProject", name="Group B Dataset"
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


# --- 3. 几何解析器 (模拟函数) ---
def parse_geometry_from_features(ifc_type, features):
    """
    根据文本描述推断大致的 3D 尺寸，以便在 Viewer 中直观区分。
    默认单位: 米
    """
    # 默认值
    l, w, h, thickness = 4.0, 4.0, 3.0, 0.3
    feat_lower = features.lower()

    # 1. 调整高度 (H)
    if "h>6m" in feat_lower or "super tall" in feat_lower:
        h = 7.0
    elif "h>4m" in feat_lower or "high ceiling" in feat_lower:
        h = 5.0
    elif "low height" in feat_lower:
        h = 1.0  # 例如 Parapet

    # 2. 调整面积/尺寸 (Area/W)
    if "area>200" in feat_lower or "large" in feat_lower:
        l, w = 15.0, 15.0
    elif "area>1000" in feat_lower:
        l, w = 30.0, 30.0
    elif "small area" in feat_lower:
        l, w = 2.0, 2.0
    elif "corridor" in feat_lower:
        l, w = 15.0, 2.0  # 细长

    # 3. 调整厚度 (Thickness)
    if "thick" in feat_lower:
        thickness = 0.8
    elif "thin" in feat_lower:
        thickness = 0.05

    # 4. 根据类型微调
    if ifc_type == "IfcSlab":
        h = thickness  # 板的高度就是厚度
    elif ifc_type == "IfcWall":
        w = thickness  # 墙的宽度作为厚度 (在 add_wall_representation 中)
        # 墙通常长而薄
        l = 6.0

    return l, w, h


# --- 4. 生成循环 ---
row_length = 6
spacing_x = 20.0  # 间距大一点，因为有些构件很大
spacing_y = 20.0

print("正在生成 Group B IFC 文件...")

for item in dataset:
    no, ifc_type, predefined_type, features, rule, actual_func = item

    # 1. 创建实体
    name_str = f"No.{no}_{actual_func.replace(' ', '_')}"
    element = ifcopenshell.api.run(
        "root.create_entity", model, ifc_class=ifc_type, name=name_str
    )

    # 2. 类型定义 (PredefinedType)
    if predefined_type:
        type_class = f"{ifc_type}Type"
        type_element = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class=type_class,
            name=f"Type_{predefined_type}",
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

    # 3. 写入自定义属性集 (Pset_SimulatedData)
    pset = ifcopenshell.api.run(
        "pset.add_pset", model, product=element, name="Pset_SimulatedData"
    )
    ifcopenshell.api.run(
        "pset.edit_pset",
        model,
        pset=pset,
        properties={
            "KeyGeometryFeatures": features,
            "InferenceRule": rule,
            "ActualFunction": actual_func,
        },
    )

    # 尝试写入 IsExternal (根据 features 推断)
    is_ext = (
        True
        if "external" in features.lower()
        or "exterior" in features.lower()
        or "roof" in features.lower()
        else False
    )
    # 写入通用 Pset
    common_pset_name = (
        f"Pset_{ifc_type[3:]}Common" if "Ifc" in ifc_type else "Pset_Common"
    )
    # 这里做个简单容错，防止 API 报错找不到对应的 Common
    try:
        c_pset = ifcopenshell.api.run(
            "pset.add_pset", model, product=element, name=common_pset_name
        )
        ifcopenshell.api.run(
            "pset.edit_pset", model, pset=c_pset, properties={"IsExternal": is_ext}
        )
    except:
        pass

    # 4. 生成几何
    l, w, h = parse_geometry_from_features(ifc_type, features)

    # 计算网格坐标
    idx = no - 31  # 从 0 开始
    pos_x = (float(idx) % row_length) * spacing_x
    pos_y = (float(idx) // row_length) * spacing_y

    # 区分 Wall 的绘制方式 (Wall 也是拉伸体，但参数意义略有不同)
    # add_wall_representation: length=长度, height=高度, thickness=厚度
    # 对于 Space/Slab: 我们为了可视化方便，也用这个API生成盒子，把 W 传给 thickness 即可
    thick_val = (
        w if ifc_type != "IfcWall" else 0.3
    )  # 如果是墙，w 已经被 parse 函数设为厚度了

    # 特殊处理 IfcSlab (使其平躺)
    # add_wall_representation 默认是立着的。
    # 为了简单起见，我们生成一个体块即可。

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
    if ifc_type == "IfcSpace":
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

# --- 5. 保存 ---
filename = "group_b_dataset.ifc"
model.write(filename)
print(f"成功生成: {filename}")
print(f"包含 No.31 到 No.65 共 {len(dataset)} 个构件。")
