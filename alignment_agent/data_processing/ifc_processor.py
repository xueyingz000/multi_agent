"""IFC data processor for extracting building information."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    import ifcopenshell
    import ifcopenshell.geom
except ImportError:
    ifcopenshell = None

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IFCEntity:
    """IFC entity data container."""
    global_id: str
    entity_type: str
    name: Optional[str]
    description: Optional[str]
    attributes: Dict[str, Any]
    geometry: Optional[Dict[str, Any]]
    materials: List[str]
    relationships: List[Dict[str, Any]]


@dataclass
class IFCRelationship:
    """IFC relationship data container."""
    relationship_type: str
    relating_object: str
    related_objects: List[str]
    attributes: Dict[str, Any]


class IFCProcessor:
    """IFC file processor for extracting building information model data.
    
    This processor handles IFC files and extracts:
    - Building entities (walls, slabs, columns, etc.)
    - Attributes and properties
    - Spatial relationships
    - Geometric information
    - Material properties
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize IFC processor.
        
        Args:
            config: Optional configuration override
        """
        self.config = get_config()
        if config:
            self.config.update('ifc_processor', config)
        
        # Get supported entity types from config
        self.supported_entity_types = self.config.get('ifc.entity_types', [
            'IfcWall', 'IfcSlab', 'IfcColumn', 'IfcBeam', 'IfcDoor', 'IfcWindow',
            'IfcSpace', 'IfcOpeningElement', 'IfcBuildingStorey', 'IfcBuilding'
        ])
        
        # Check if ifcopenshell is available
        if ifcopenshell is None:
            logger.warning("ifcopenshell not available. IFC processing will be limited.")
        
        logger.info("IFC Processor initialized")
    
    def process_ifc_file(self, file_path: str) -> Dict[str, Any]:
        """Process IFC file and extract building information.
        
        Args:
            file_path: Path to IFC file
            
        Returns:
            Dictionary containing extracted IFC data
        """
        logger.info(f"Processing IFC file: {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"IFC file not found: {file_path}")
        
        try:
            if ifcopenshell is None:
                return self._process_ifc_mock(file_path)
            
            # Load IFC file
            ifc_file = ifcopenshell.open(file_path)
            
            # Extract entities
            entities = self._extract_entities(ifc_file)
            
            # Extract relationships
            relationships = self._extract_relationships(ifc_file)
            
            # Extract spatial hierarchy
            spatial_hierarchy = self._extract_spatial_hierarchy(ifc_file)
            
            # Extract project information
            project_info = self._extract_project_info(ifc_file)
            
            ifc_data = {
                'file_path': file_path,
                'project_info': project_info,
                'entities': entities,
                'relationships': relationships,
                'spatial_hierarchy': spatial_hierarchy,
                'entity_count': len(entities),
                'processing_timestamp': self._get_timestamp()
            }
            
            logger.info(f"IFC processing completed: {len(entities)} entities extracted")
            return ifc_data
            
        except Exception as e:
            logger.error(f"Error processing IFC file: {e}")
            raise
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Alias for process_ifc_file method.
        
        Args:
            file_path: Path to IFC file
            
        Returns:
            Dictionary containing extracted IFC data
        """
        return self.process_ifc_file(file_path)
    
    def _extract_entities(self, ifc_file) -> List[Dict[str, Any]]:
        """Extract entities from IFC file.
        
        Args:
            ifc_file: Loaded IFC file object
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type in self.supported_entity_types:
            try:
                ifc_entities = ifc_file.by_type(entity_type)
                
                for ifc_entity in ifc_entities:
                    entity_data = self._extract_entity_data(ifc_entity)
                    entities.append(entity_data)
                    
            except Exception as e:
                logger.warning(f"Error extracting {entity_type}: {e}")
                continue
        
        return entities
    
    def _extract_entity_data(self, ifc_entity) -> Dict[str, Any]:
        """Extract data from a single IFC entity.
        
        Args:
            ifc_entity: IFC entity object
            
        Returns:
            Dictionary containing entity data
        """
        try:
            # Basic entity information
            entity_data = {
                'global_id': getattr(ifc_entity, 'GlobalId', None),
                'type': ifc_entity.is_a(),
                'name': getattr(ifc_entity, 'Name', None),
                'description': getattr(ifc_entity, 'Description', None),
                'attributes': {},
                'geometry': None,
                'materials': [],
                'relationships': []
            }
            
            # Extract attributes
            entity_data['attributes'] = self._extract_attributes(ifc_entity)
            
            # Extract geometry (if available)
            entity_data['geometry'] = self._extract_geometry(ifc_entity)
            
            # Extract materials
            entity_data['materials'] = self._extract_materials(ifc_entity)
            
            return entity_data
            
        except Exception as e:
            logger.warning(f"Error extracting entity data: {e}")
            return {
                'global_id': None,
                'type': str(type(ifc_entity)),
                'name': None,
                'description': None,
                'attributes': {},
                'geometry': None,
                'materials': [],
                'relationships': []
            }
    
    def _extract_attributes(self, ifc_entity) -> Dict[str, Any]:
        """Extract attributes from IFC entity.
        
        Args:
            ifc_entity: IFC entity object
            
        Returns:
            Dictionary of attributes
        """
        attributes = {}
        
        try:
            # Get property sets
            if hasattr(ifc_entity, 'IsDefinedBy'):
                for definition in ifc_entity.IsDefinedBy:
                    if definition.is_a('IfcRelDefinesByProperties'):
                        prop_set = definition.RelatingPropertyDefinition
                        if prop_set.is_a('IfcPropertySet'):
                            pset_name = prop_set.Name
                            attributes[pset_name] = {}
                            
                            for prop in prop_set.HasProperties:
                                if prop.is_a('IfcPropertySingleValue'):
                                    prop_name = prop.Name
                                    prop_value = prop.NominalValue.wrappedValue if prop.NominalValue else None
                                    attributes[pset_name][prop_name] = prop_value
            
            # Get basic attributes
            basic_attrs = ['Name', 'Description', 'ObjectType', 'Tag']
            for attr in basic_attrs:
                if hasattr(ifc_entity, attr):
                    value = getattr(ifc_entity, attr)
                    if value:
                        attributes[attr] = value
            
        except Exception as e:
            logger.warning(f"Error extracting attributes: {e}")
        
        return attributes
    
    def _extract_geometry(self, ifc_entity) -> Optional[Dict[str, Any]]:
        """Extract geometry information from IFC entity.
        
        Args:
            ifc_entity: IFC entity object
            
        Returns:
            Dictionary containing geometry information or None
        """
        try:
            if not hasattr(ifc_entity, 'Representation'):
                return None
            
            geometry_info = {
                'has_geometry': True,
                'representation_type': None,
                'bounding_box': None,
                'volume': None,
                'area': None
            }
            
            # Try to get geometric representation
            if ifc_entity.Representation:
                representations = ifc_entity.Representation.Representations
                if representations:
                    rep = representations[0]
                    geometry_info['representation_type'] = rep.RepresentationType
            
            # Try to calculate geometric properties using ifcopenshell.geom
            try:
                settings = ifcopenshell.geom.settings()
                shape = ifcopenshell.geom.create_shape(settings, ifc_entity)
                
                if shape:
                    # Get bounding box
                    bbox = shape.geometry.bounding_box
                    geometry_info['bounding_box'] = {
                        'min': list(bbox.min),
                        'max': list(bbox.max)
                    }
                    
                    # Get volume and area if available
                    geometry_info['volume'] = getattr(shape.geometry, 'volume', None)
                    geometry_info['area'] = getattr(shape.geometry, 'surface_area', None)
                    
            except Exception as geom_e:
                logger.debug(f"Could not extract detailed geometry: {geom_e}")
            
            return geometry_info
            
        except Exception as e:
            logger.warning(f"Error extracting geometry: {e}")
            return None
    
    def _extract_materials(self, ifc_entity) -> List[str]:
        """Extract material information from IFC entity.
        
        Args:
            ifc_entity: IFC entity object
            
        Returns:
            List of material names
        """
        materials = []
        
        try:
            if hasattr(ifc_entity, 'HasAssociations'):
                for association in ifc_entity.HasAssociations:
                    if association.is_a('IfcRelAssociatesMaterial'):
                        material = association.RelatingMaterial
                        
                        if material.is_a('IfcMaterial'):
                            materials.append(material.Name)
                        elif material.is_a('IfcMaterialLayerSetUsage'):
                            layer_set = material.ForLayerSet
                            for layer in layer_set.MaterialLayers:
                                if layer.Material:
                                    materials.append(layer.Material.Name)
                        elif material.is_a('IfcMaterialList'):
                            for mat in material.Materials:
                                materials.append(mat.Name)
                                
        except Exception as e:
            logger.warning(f"Error extracting materials: {e}")
        
        return materials
    
    def _extract_relationships(self, ifc_file) -> List[Dict[str, Any]]:
        """Extract relationships from IFC file.
        
        Args:
            ifc_file: Loaded IFC file object
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Common relationship types
        rel_types = [
            'IfcRelContainedInSpatialStructure',
            'IfcRelAggregates',
            'IfcRelConnectsElements',
            'IfcRelFillsElement',
            'IfcRelVoidsElement'
        ]
        
        for rel_type in rel_types:
            try:
                rels = ifc_file.by_type(rel_type)
                for rel in rels:
                    rel_data = self._extract_relationship_data(rel)
                    relationships.append(rel_data)
                    
            except Exception as e:
                logger.warning(f"Error extracting {rel_type}: {e}")
                continue
        
        return relationships
    
    def _extract_relationship_data(self, ifc_relationship) -> Dict[str, Any]:
        """Extract data from IFC relationship.
        
        Args:
            ifc_relationship: IFC relationship object
            
        Returns:
            Dictionary containing relationship data
        """
        try:
            rel_data = {
                'type': ifc_relationship.is_a(),
                'global_id': getattr(ifc_relationship, 'GlobalId', None),
                'name': getattr(ifc_relationship, 'Name', None),
                'relating_object': None,
                'related_objects': [],
                'attributes': {}
            }
            
            # Extract relating object
            if hasattr(ifc_relationship, 'RelatingObject'):
                relating = ifc_relationship.RelatingObject
                rel_data['relating_object'] = getattr(relating, 'GlobalId', str(relating))
            elif hasattr(ifc_relationship, 'RelatingStructure'):
                relating = ifc_relationship.RelatingStructure
                rel_data['relating_object'] = getattr(relating, 'GlobalId', str(relating))
            
            # Extract related objects
            if hasattr(ifc_relationship, 'RelatedObjects'):
                for related in ifc_relationship.RelatedObjects:
                    rel_data['related_objects'].append(getattr(related, 'GlobalId', str(related)))
            elif hasattr(ifc_relationship, 'RelatedElements'):
                for related in ifc_relationship.RelatedElements:
                    rel_data['related_objects'].append(getattr(related, 'GlobalId', str(related)))
            
            return rel_data
            
        except Exception as e:
            logger.warning(f"Error extracting relationship data: {e}")
            return {
                'type': str(type(ifc_relationship)),
                'global_id': None,
                'name': None,
                'relating_object': None,
                'related_objects': [],
                'attributes': {}
            }
    
    def _extract_spatial_hierarchy(self, ifc_file) -> Dict[str, Any]:
        """Extract spatial hierarchy from IFC file.
        
        Args:
            ifc_file: Loaded IFC file object
            
        Returns:
            Dictionary representing spatial hierarchy
        """
        hierarchy = {
            'project': None,
            'sites': [],
            'buildings': [],
            'storeys': [],
            'spaces': []
        }
        
        try:
            # Get project
            projects = ifc_file.by_type('IfcProject')
            if projects:
                project = projects[0]
                hierarchy['project'] = {
                    'global_id': project.GlobalId,
                    'name': project.Name,
                    'description': project.Description
                }
            
            # Get sites
            sites = ifc_file.by_type('IfcSite')
            for site in sites:
                hierarchy['sites'].append({
                    'global_id': site.GlobalId,
                    'name': site.Name,
                    'description': site.Description
                })
            
            # Get buildings
            buildings = ifc_file.by_type('IfcBuilding')
            for building in buildings:
                hierarchy['buildings'].append({
                    'global_id': building.GlobalId,
                    'name': building.Name,
                    'description': building.Description
                })
            
            # Get storeys
            storeys = ifc_file.by_type('IfcBuildingStorey')
            for storey in storeys:
                hierarchy['storeys'].append({
                    'global_id': storey.GlobalId,
                    'name': storey.Name,
                    'description': storey.Description,
                    'elevation': getattr(storey, 'Elevation', None)
                })
            
            # Get spaces
            spaces = ifc_file.by_type('IfcSpace')
            for space in spaces:
                hierarchy['spaces'].append({
                    'global_id': space.GlobalId,
                    'name': space.Name,
                    'description': space.Description,
                    'long_name': getattr(space, 'LongName', None)
                })
                
        except Exception as e:
            logger.warning(f"Error extracting spatial hierarchy: {e}")
        
        return hierarchy
    
    def _extract_project_info(self, ifc_file) -> Dict[str, Any]:
        """Extract project information from IFC file.
        
        Args:
            ifc_file: Loaded IFC file object
            
        Returns:
            Dictionary containing project information
        """
        project_info = {
            'schema': ifc_file.schema,
            'version': None,
            'application': None,
            'timestamp': None
        }
        
        try:
            # Get file header information
            header = ifc_file.header
            if header:
                project_info['version'] = header.file_description.description[0] if header.file_description.description else None
                project_info['application'] = header.file_name.originating_system if header.file_name else None
                project_info['timestamp'] = header.file_name.time_stamp if header.file_name else None
                
        except Exception as e:
            logger.warning(f"Error extracting project info: {e}")
        
        return project_info
    
    def _process_ifc_mock(self, file_path: str) -> Dict[str, Any]:
        """Mock IFC processing when ifcopenshell is not available.
        
        Args:
            file_path: Path to IFC file
            
        Returns:
            Mock IFC data
        """
        logger.warning("Using mock IFC processing - ifcopenshell not available")
        
        # Return mock data structure
        return {
            'file_path': file_path,
            'project_info': {
                'schema': 'IFC4',
                'version': 'Mock Version',
                'application': 'Mock Application',
                'timestamp': self._get_timestamp()
            },
            'entities': [
                {
                    'global_id': 'mock-wall-001',
                    'type': 'IfcWall',
                    'name': 'Mock Wall',
                    'description': 'Mock wall entity',
                    'attributes': {'Height': 3000, 'Width': 200},
                    'geometry': {'has_geometry': True, 'representation_type': 'SweptSolid'},
                    'materials': ['Concrete'],
                    'relationships': []
                },
                {
                    'global_id': 'mock-slab-001',
                    'type': 'IfcSlab',
                    'name': 'Mock Slab',
                    'description': 'Mock slab entity',
                    'attributes': {'Thickness': 200},
                    'geometry': {'has_geometry': True, 'representation_type': 'SweptSolid'},
                    'materials': ['Concrete'],
                    'relationships': []
                }
            ],
            'relationships': [],
            'spatial_hierarchy': {
                'project': {'global_id': 'mock-project', 'name': 'Mock Project', 'description': None},
                'sites': [],
                'buildings': [{'global_id': 'mock-building', 'name': 'Mock Building', 'description': None}],
                'storeys': [],
                'spaces': []
            },
            'entity_count': 2,
            'processing_timestamp': self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_entity_by_id(self, ifc_data: Dict[str, Any], global_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by global ID.
        
        Args:
            ifc_data: Processed IFC data
            global_id: Entity global ID
            
        Returns:
            Entity data or None if not found
        """
        entities = ifc_data.get('entities', [])
        for entity in entities:
            if entity.get('global_id') == global_id:
                return entity
        return None
    
    def get_entities_by_type(self, ifc_data: Dict[str, Any], entity_type: str) -> List[Dict[str, Any]]:
        """Get entities by type.
        
        Args:
            ifc_data: Processed IFC data
            entity_type: IFC entity type
            
        Returns:
            List of entities of specified type
        """
        entities = ifc_data.get('entities', [])
        return [entity for entity in entities if entity.get('type') == entity_type]
    
    def get_related_entities(self, ifc_data: Dict[str, Any], global_id: str) -> List[Dict[str, Any]]:
        """Get entities related to specified entity.
        
        Args:
            ifc_data: Processed IFC data
            global_id: Entity global ID
            
        Returns:
            List of related entities
        """
        related_entities = []
        relationships = ifc_data.get('relationships', [])
        
        for rel in relationships:
            if rel.get('relating_object') == global_id:
                for related_id in rel.get('related_objects', []):
                    entity = self.get_entity_by_id(ifc_data, related_id)
                    if entity:
                        related_entities.append(entity)
            elif global_id in rel.get('related_objects', []):
                relating_entity = self.get_entity_by_id(ifc_data, rel.get('relating_object'))
                if relating_entity:
                    related_entities.append(relating_entity)
        
        return related_entities