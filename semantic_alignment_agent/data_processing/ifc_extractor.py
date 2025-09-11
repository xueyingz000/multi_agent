from __future__ import annotations

import ifcopenshell
import ifcopenshell.geom
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from ..utils import (
    log, IfcElementType, IfcElementInfo, GeometricFeatures, 
    Point3D, BoundingBox, config
)


class IfcExtractor:
    """IFC数据提取器，专注于面积计算相关的元素"""
    
    def __init__(self, ifc_file_path: str):
        self.ifc_file_path = Path(ifc_file_path)
        self.ifc_file: Optional[ifcopenshell.file] = None
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)
        
        # 从配置获取目标实体类型
        ifc_config = config.get_ifc_config()
        self.target_entities = ifc_config.get('target_entities', [
            'IfcSlab', 'IfcSpace', 'IfcOpeningElement', 
            'IfcVoid', 'IfcShaft', 'IfcBuildingElementProxy'
        ])
        
        log.info(f"Initialized IFC extractor for: {self.ifc_file_path}")
    
    def load_ifc_file(self) -> ifcopenshell.file:
        """加载IFC文件"""
        if self.ifc_file is None:
            if not self.ifc_file_path.exists():
                raise FileNotFoundError(f"IFC file not found: {self.ifc_file_path}")
            
            log.info(f"Loading IFC file: {self.ifc_file_path}")
            self.ifc_file = ifcopenshell.open(str(self.ifc_file_path))
            log.info(f"IFC file loaded successfully. Schema: {self.ifc_file.schema}")
        
        return self.ifc_file
    
    def extract_all_elements(self) -> List[IfcElementInfo]:
        """提取所有相关元素"""
        ifc_file = self.load_ifc_file()
        elements = []
        
        for entity_type in self.target_entities:
            try:
                entity_elements = ifc_file.by_type(entity_type)
                log.info(f"Found {len(entity_elements)} {entity_type} elements")
                
                for element in entity_elements:
                    element_info = self._extract_element_info(element)
                    if element_info:
                        elements.append(element_info)
                        
            except Exception as e:
                log.warning(f"Error extracting {entity_type}: {e}")
        
        log.info(f"Total extracted elements: {len(elements)}")
        return elements
    
    def extract_slabs(self) -> List[IfcElementInfo]:
        """专门提取IfcSlab元素"""
        ifc_file = self.load_ifc_file()
        slabs = []
        
        for slab in ifc_file.by_type('IfcSlab'):
            slab_info = self._extract_element_info(slab)
            if slab_info:
                # 为slab添加特殊的几何分析
                slab_info = self._enhance_slab_analysis(slab, slab_info)
                slabs.append(slab_info)
        
        log.info(f"Extracted {len(slabs)} slab elements")
        return slabs
    
    def extract_spaces(self) -> List[IfcElementInfo]:
        """专门提取IfcSpace元素"""
        ifc_file = self.load_ifc_file()
        spaces = []
        
        for space in ifc_file.by_type('IfcSpace'):
            space_info = self._extract_element_info(space)
            if space_info:
                # 为space添加特殊的功能分析
                space_info = self._enhance_space_analysis(space, space_info)
                spaces.append(space_info)
        
        log.info(f"Extracted {len(spaces)} space elements")
        return spaces
    
    def extract_vertical_elements(self) -> List[IfcElementInfo]:
        """提取垂直相关元素（用于B类问题）"""
        ifc_file = self.load_ifc_file()
        vertical_elements = []
        
        # 提取可能的垂直空间元素
        vertical_types = ['IfcSpace', 'IfcOpeningElement', 'IfcVoid', 'IfcShaft']
        
        for entity_type in vertical_types:
            try:
                for element in ifc_file.by_type(entity_type):
                    element_info = self._extract_element_info(element)
                    if element_info and self._is_potential_vertical_element(element, element_info):
                        vertical_elements.append(element_info)
            except Exception as e:
                log.warning(f"Error extracting vertical {entity_type}: {e}")
        
        log.info(f"Extracted {len(vertical_elements)} potential vertical elements")
        return vertical_elements
    
    def _extract_element_info(self, element) -> Optional[IfcElementInfo]:
        """提取单个元素的信息"""
        try:
            # 基本信息
            guid = element.GlobalId
            ifc_type = IfcElementType(element.is_a())
            name = getattr(element, 'Name', None)
            description = getattr(element, 'Description', None)
            predefined_type = getattr(element, 'PredefinedType', None)
            object_type = getattr(element, 'ObjectType', None)
            
            # 几何特征
            geometric_features = self._extract_geometric_features(element)
            
            # 属性信息
            properties = self._extract_properties(element)
            
            # 关系信息
            relationships = self._extract_relationships(element)
            
            # 材料信息
            material_info = self._extract_material_info(element)
            
            return IfcElementInfo(
                guid=guid,
                ifc_type=ifc_type,
                name=name,
                description=description,
                predefined_type=predefined_type,
                object_type=object_type,
                geometric_features=geometric_features,
                properties=properties,
                relationships=relationships,
                material_info=material_info
            )
            
        except Exception as e:
            log.warning(f"Error extracting element info for {element}: {e}")
            return None
    
    def _extract_geometric_features(self, element) -> Optional[GeometricFeatures]:
        """提取几何特征"""
        try:
            # 尝试获取几何表示
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            if not shape:
                return None
            
            # 获取边界框
            bbox = shape.geometry.bounding_box
            min_point = Point3D(bbox[0], bbox[1], bbox[2])
            max_point = Point3D(bbox[3], bbox[4], bbox[5])
            bounding_box = BoundingBox(min_point, max_point)
            
            # 计算体积和面积
            volume = shape.geometry.volume
            area = shape.geometry.surface_area
            
            # 计算质心
            centroid_coords = shape.geometry.centroid
            centroid = Point3D(centroid_coords[0], centroid_coords[1], centroid_coords[2])
            
            # 特殊处理：计算厚度（对于slab等）
            thickness = self._calculate_thickness(element, bounding_box)
            
            # 判断是否为垂直元素
            is_vertical = bounding_box.height > max(bounding_box.width, bounding_box.depth)
            
            # 计算楼层信息
            floor_level = self._get_floor_level(element)
            
            return GeometricFeatures(
                bounding_box=bounding_box,
                area=area,
                volume=volume,
                thickness=thickness,
                centroid=centroid,
                floor_level=floor_level,
                is_vertical_element=is_vertical,
                cross_section_area=bounding_box.width * bounding_box.depth if not is_vertical else None
            )
            
        except Exception as e:
            log.warning(f"Error extracting geometric features for {element}: {e}")
            return None
    
    def _calculate_thickness(self, element, bounding_box: BoundingBox) -> Optional[float]:
        """计算元素厚度（主要用于slab）"""
        try:
            if element.is_a('IfcSlab'):
                # 对于slab，厚度通常是Z方向的最小尺寸
                return min(bounding_box.height, 
                          min(bounding_box.width, bounding_box.depth) * 0.1)  # 启发式方法
            return None
        except Exception:
            return None
    
    def _get_floor_level(self, element) -> Optional[float]:
        """获取元素所在楼层标高"""
        try:
            # 通过IfcBuildingStorey关系获取楼层信息
            for rel in getattr(element, 'ContainedInStructure', []):
                if rel.RelatingStructure.is_a('IfcBuildingStorey'):
                    storey = rel.RelatingStructure
                    elevation = getattr(storey, 'Elevation', None)
                    if elevation is not None:
                        return float(elevation)
            
            # 如果没有找到，使用几何中心的Z坐标作为近似
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            if shape:
                return shape.geometry.centroid[2]
                
        except Exception:
            pass
        
        return None
    
    def _extract_properties(self, element) -> Dict[str, Any]:
        """提取属性信息"""
        properties = {}
        
        try:
            # 获取属性集
            for definition in getattr(element, 'IsDefinedBy', []):
                if definition.is_a('IfcRelDefinesByProperties'):
                    property_set = definition.RelatingPropertyDefinition
                    if property_set.is_a('IfcPropertySet'):
                        set_name = property_set.Name
                        properties[set_name] = {}
                        
                        for prop in property_set.HasProperties:
                            if prop.is_a('IfcPropertySingleValue'):
                                prop_name = prop.Name
                                prop_value = prop.NominalValue.wrappedValue if prop.NominalValue else None
                                properties[set_name][prop_name] = prop_value
        
        except Exception as e:
            log.warning(f"Error extracting properties for {element}: {e}")
        
        return properties
    
    def _extract_relationships(self, element) -> Dict[str, List[str]]:
        """提取关系信息"""
        relationships = {}
        
        try:
            # 空间包含关系
            if hasattr(element, 'ContainedInStructure'):
                containers = []
                for rel in element.ContainedInStructure:
                    if hasattr(rel, 'RelatingStructure'):
                        containers.append(rel.RelatingStructure.GlobalId)
                if containers:
                    relationships['contained_in'] = containers
            
            # 空间边界关系（对于IfcSpace）
            if element.is_a('IfcSpace') and hasattr(element, 'BoundedBy'):
                boundaries = []
                for boundary in element.BoundedBy:
                    if hasattr(boundary, 'RelatedBuildingElement'):
                        boundaries.append(boundary.RelatedBuildingElement.GlobalId)
                if boundaries:
                    relationships['bounded_by'] = boundaries
            
            # 开口关系（对于有开口的元素）
            if hasattr(element, 'HasOpenings'):
                openings = []
                for rel in element.HasOpenings:
                    openings.append(rel.RelatedOpeningElement.GlobalId)
                if openings:
                    relationships['has_openings'] = openings
        
        except Exception as e:
            log.warning(f"Error extracting relationships for {element}: {e}")
        
        return relationships
    
    def _extract_material_info(self, element) -> Dict[str, Any]:
        """提取材料信息"""
        material_info = {}
        
        try:
            # 获取材料关联
            for rel in getattr(element, 'HasAssociations', []):
                if rel.is_a('IfcRelAssociatesMaterial'):
                    material = rel.RelatingMaterial
                    if material.is_a('IfcMaterial'):
                        material_info['name'] = material.Name
                        material_info['description'] = getattr(material, 'Description', None)
                    elif material.is_a('IfcMaterialLayerSetUsage'):
                        layer_set = material.ForLayerSet
                        layers = []
                        for layer in layer_set.MaterialLayers:
                            layers.append({
                                'material': layer.Material.Name,
                                'thickness': layer.LayerThickness
                            })
                        material_info['layers'] = layers
        
        except Exception as e:
            log.warning(f"Error extracting material info for {element}: {e}")
        
        return material_info
    
    def _enhance_slab_analysis(self, slab, slab_info: IfcElementInfo) -> IfcElementInfo:
        """增强slab分析"""
        try:
            # 分析slab的特殊属性
            if slab_info.geometric_features:
                # 基于厚度的初步分类提示
                thickness = slab_info.geometric_features.thickness
                if thickness:
                    if thickness < 0.1:
                        slab_info.properties['thickness_category'] = 'thin_decoration'
                    elif thickness < 0.15:
                        slab_info.properties['thickness_category'] = 'medium_equipment'
                    else:
                        slab_info.properties['thickness_category'] = 'thick_structural'
                
                # 位置分析
                if slab_info.geometric_features.floor_level:
                    # 判断是否在屋顶层
                    # 这里需要更复杂的逻辑来判断建筑的最高层
                    slab_info.properties['position_hint'] = 'needs_building_context'
        
        except Exception as e:
            log.warning(f"Error enhancing slab analysis: {e}")
        
        return slab_info
    
    def _enhance_space_analysis(self, space, space_info: IfcElementInfo) -> IfcElementInfo:
        """增强space分析"""
        try:
            # 分析space的功能提示
            predefined_type = space_info.predefined_type
            name = space_info.name or ""
            
            # 基于预定义类型和名称的功能提示
            function_hints = []
            if predefined_type:
                if predefined_type.upper() in ['SHAFT', 'STAIRCASE']:
                    function_hints.append('vertical_circulation')
                elif predefined_type.upper() == 'USERDEFINED':
                    function_hints.append('needs_name_analysis')
            
            if name:
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in ['equipment', 'plant', 'mechanical']):
                    function_hints.append('equipment_room')
                elif any(keyword in name_lower for keyword in ['office', 'room']):
                    function_hints.append('habitable_space')
                elif any(keyword in name_lower for keyword in ['parking', 'garage']):
                    function_hints.append('parking_space')
            
            space_info.properties['function_hints'] = function_hints
        
        except Exception as e:
            log.warning(f"Error enhancing space analysis: {e}")
        
        return space_info
    
    def _is_potential_vertical_element(self, element, element_info: IfcElementInfo) -> bool:
        """判断是否为潜在的垂直元素"""
        try:
            # 基于几何特征判断
            if element_info.geometric_features:
                geom = element_info.geometric_features
                # 高度明显大于宽度和深度
                if geom.is_vertical_element:
                    return True
                
                # 或者高度跨越多个楼层
                if geom.bounding_box.height > 3.0:  # 假设层高大于3米
                    return True
            
            # 基于类型和属性判断
            if element.is_a('IfcSpace'):
                predefined_type = getattr(element, 'PredefinedType', None)
                if predefined_type and predefined_type.upper() in ['SHAFT', 'STAIRCASE']:
                    return True
            
            return False
        
        except Exception:
            return False
    
    def get_building_structure(self) -> Dict[str, Any]:
        """获取建筑结构信息（楼层、建筑等）"""
        ifc_file = self.load_ifc_file()
        structure = {
            'project': None,
            'buildings': [],
            'storeys': [],
            'sites': []
        }
        
        try:
            # 项目信息
            projects = ifc_file.by_type('IfcProject')
            if projects:
                project = projects[0]
                structure['project'] = {
                    'guid': project.GlobalId,
                    'name': getattr(project, 'Name', None),
                    'description': getattr(project, 'Description', None)
                }
            
            # 建筑信息
            for building in ifc_file.by_type('IfcBuilding'):
                structure['buildings'].append({
                    'guid': building.GlobalId,
                    'name': getattr(building, 'Name', None),
                    'description': getattr(building, 'Description', None)
                })
            
            # 楼层信息
            for storey in ifc_file.by_type('IfcBuildingStorey'):
                structure['storeys'].append({
                    'guid': storey.GlobalId,
                    'name': getattr(storey, 'Name', None),
                    'elevation': getattr(storey, 'Elevation', None)
                })
            
            # 场地信息
            for site in ifc_file.by_type('IfcSite'):
                structure['sites'].append({
                    'guid': site.GlobalId,
                    'name': getattr(site, 'Name', None),
                    'description': getattr(site, 'Description', None)
                })
        
        except Exception as e:
            log.warning(f"Error extracting building structure: {e}")
        
        return structure