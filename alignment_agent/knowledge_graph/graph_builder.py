"""Graph Builder for constructing knowledge graphs from IFC and regulatory data."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from utils import ConfigLoader, get_logger


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any]
    node_type: str  # 'ifc_entity', 'regulatory_term', 'concept', etc.
    source: str  # 'ifc', 'regulatory', 'aligned'
    

@dataclass
class GraphRelation:
    """Represents a relationship in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float = 1.0
    source: str = 'extracted'
    

class GraphBuilder:
    """Builds and manages knowledge graphs for IFC-regulatory semantic alignment."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the graph builder.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)
        
        # Graph backend configuration
        self.backend = self.config.get('knowledge_graph.backend', 'networkx')
        
        # Initialize graph backend
        if self.backend == 'neo4j' and NEO4J_AVAILABLE:
            self._init_neo4j()
        elif self.backend == 'networkx' and NETWORKX_AVAILABLE:
            self._init_networkx()
        else:
            self.logger.warning(f"Backend {self.backend} not available, using mock implementation")
            self._init_mock()
            
        # Node and relation storage
        self.nodes: Dict[str, GraphNode] = {}
        self.relations: List[GraphRelation] = []
        
        # Predefined ontology mappings
        self._load_ontology_mappings()
        
    def _init_neo4j(self):
        """Initialize Neo4j connection."""
        try:
            neo4j_config = self.config.get('knowledge_graph.neo4j', {})
            uri = neo4j_config.get('uri', 'bolt://localhost:7687')
            username = neo4j_config.get('username', 'neo4j')
            password = neo4j_config.get('password', 'password')
            
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.logger.info("Neo4j connection initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j: {e}")
            self._init_mock()
            
    def _init_networkx(self):
        """Initialize NetworkX graph."""
        self.graph = nx.MultiDiGraph()
        self.logger.info("NetworkX graph initialized")
        
    def _init_mock(self):
        """Initialize mock implementation."""
        self.graph = None
        self.driver = None
        self.logger.info("Mock graph implementation initialized")
        
    def _load_ontology_mappings(self):
        """Load predefined ontology mappings."""
        self.ifc_ontology = {
            'IfcWall': {
                'category': 'structural_element',
                'properties': ['thickness', 'height', 'material', 'load_bearing'],
                'relationships': ['contains', 'supports', 'adjacent_to']
            },
            'IfcSlab': {
                'category': 'structural_element', 
                'properties': ['thickness', 'area', 'material', 'structural_type'],
                'relationships': ['supports', 'contains', 'above', 'below']
            },
            'IfcOpeningElement': {
                'category': 'void_element',
                'properties': ['width', 'height', 'depth', 'purpose'],
                'relationships': ['cuts', 'located_in']
            },
            'IfcSpace': {
                'category': 'spatial_element',
                'properties': ['area', 'volume', 'function', 'occupancy'],
                'relationships': ['contains', 'adjacent_to', 'accessed_by']
            }
        }
        
        self.regulatory_ontology = {
            'wall': {
                'synonyms': ['partition', 'barrier', 'enclosure'],
                'types': ['load_bearing', 'non_load_bearing', 'curtain', 'shear'],
                'properties': ['fire_rating', 'acoustic_rating', 'thermal_resistance']
            },
            'slab': {
                'synonyms': ['floor', 'platform', 'deck', 'terrace'],
                'types': ['structural', 'equipment_platform', 'balcony'],
                'properties': ['live_load', 'dead_load', 'deflection_limit']
            },
            'opening': {
                'synonyms': ['shaft', 'void', 'penetration', 'atrium'],
                'types': ['functional_shaft', 'atrium', 'stairwell', 'elevator_shaft'],
                'properties': ['fire_protection', 'ventilation', 'access']
            }
        }
        
    def add_ifc_entities(self, ifc_data: Dict[str, Any]) -> List[str]:
        """Add IFC entities to the knowledge graph.
        
        Args:
            ifc_data: Dictionary containing IFC entities and their properties
            
        Returns:
            List of created node IDs
        """
        created_nodes = []
        
        for entity_id, entity_data in ifc_data.get('entities', {}).items():
            node = GraphNode(
                id=f"ifc_{entity_id}",
                label=entity_data.get('type', 'IfcEntity'),
                properties={
                    'ifc_id': entity_id,
                    'ifc_type': entity_data.get('type'),
                    'attributes': entity_data.get('attributes', {}),
                    'properties': entity_data.get('properties', {}),
                    'geometry': entity_data.get('geometry', {})
                },
                node_type='ifc_entity',
                source='ifc'
            )
            
            self.nodes[node.id] = node
            created_nodes.append(node.id)
            
            # Add to graph backend
            self._add_node_to_backend(node)
            
        # Add spatial relationships
        for rel_data in ifc_data.get('relationships', []):
            relation = GraphRelation(
                source_id=f"ifc_{rel_data['source']}",
                target_id=f"ifc_{rel_data['target']}",
                relation_type=rel_data.get('type', 'related_to'),
                properties=rel_data.get('properties', {}),
                confidence=1.0,
                source='ifc'
            )
            
            self.relations.append(relation)
            self._add_relation_to_backend(relation)
            
        self.logger.info(f"Added {len(created_nodes)} IFC entities to knowledge graph")
        return created_nodes
        
    def add_regulatory_entities(self, regulatory_data: Dict[str, Any]) -> List[str]:
        """Add regulatory entities to the knowledge graph.
        
        Args:
            regulatory_data: Dictionary containing regulatory entities and relationships
            
        Returns:
            List of created node IDs
        """
        created_nodes = []
        
        # Add entities
        for entity in regulatory_data.get('entities', []):
            node = GraphNode(
                id=f"reg_{entity['id']}",
                label=entity.get('label', entity.get('text', 'RegEntity')),
                properties={
                    'text': entity.get('text'),
                    'category': entity.get('category'),
                    'entity_type': entity.get('type'),
                    'context': entity.get('context', ''),
                    'document_source': entity.get('source', ''),
                    'confidence': entity.get('confidence', 1.0)
                },
                node_type='regulatory_term',
                source='regulatory'
            )
            
            self.nodes[node.id] = node
            created_nodes.append(node.id)
            self._add_node_to_backend(node)
            
        # Add relationships
        for rel in regulatory_data.get('relationships', []):
            relation = GraphRelation(
                source_id=f"reg_{rel['source']}",
                target_id=f"reg_{rel['target']}",
                relation_type=rel.get('type', 'related_to'),
                properties={
                    'context': rel.get('context', ''),
                    'strength': rel.get('strength', 1.0)
                },
                confidence=rel.get('confidence', 1.0),
                source='regulatory'
            )
            
            self.relations.append(relation)
            self._add_relation_to_backend(relation)
            
        self.logger.info(f"Added {len(created_nodes)} regulatory entities to knowledge graph")
        return created_nodes
        
    def add_alignment_mappings(self, alignments: List[Dict[str, Any]]) -> List[str]:
        """Add semantic alignment mappings to the knowledge graph.
        
        Args:
            alignments: List of alignment results from semantic alignment module
            
        Returns:
            List of created relation IDs
        """
        created_relations = []
        
        for alignment in alignments:
            # Create alignment relation
            relation = GraphRelation(
                source_id=f"ifc_{alignment['ifc_entity_id']}",
                target_id=f"reg_{alignment['regulatory_entity_id']}",
                relation_type='semantically_aligned',
                properties={
                    'alignment_type': alignment.get('alignment_type', 'entity'),
                    'similarity_score': alignment.get('similarity_score', 0.0),
                    'confidence_score': alignment.get('confidence_score', 0.0),
                    'alignment_method': alignment.get('method', 'unknown'),
                    'context_similarity': alignment.get('context_similarity', 0.0)
                },
                confidence=alignment.get('confidence_score', 0.0),
                source='alignment'
            )
            
            self.relations.append(relation)
            self._add_relation_to_backend(relation)
            created_relations.append(f"{relation.source_id}-{relation.target_id}")
            
        self.logger.info(f"Added {len(created_relations)} alignment mappings to knowledge graph")
        return created_relations
        
    def _add_node_to_backend(self, node: GraphNode):
        """Add node to the graph backend."""
        if self.backend == 'neo4j' and self.driver:
            self._add_node_neo4j(node)
        elif self.backend == 'networkx' and self.graph is not None:
            self._add_node_networkx(node)
            
    def _add_relation_to_backend(self, relation: GraphRelation):
        """Add relation to the graph backend."""
        if self.backend == 'neo4j' and self.driver:
            self._add_relation_neo4j(relation)
        elif self.backend == 'networkx' and self.graph is not None:
            self._add_relation_networkx(relation)
            
    def _add_node_neo4j(self, node: GraphNode):
        """Add node to Neo4j."""
        with self.driver.session() as session:
            query = f"""
            CREATE (n:{node.node_type} {{
                id: $id,
                label: $label,
                properties: $properties,
                source: $source
            }})
            """
            session.run(query, 
                       id=node.id, 
                       label=node.label,
                       properties=json.dumps(node.properties),
                       source=node.source)
                       
    def _add_relation_neo4j(self, relation: GraphRelation):
        """Add relation to Neo4j."""
        with self.driver.session() as session:
            query = f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            CREATE (a)-[r:{relation.relation_type.upper()} {{
                properties: $properties,
                confidence: $confidence,
                source: $source
            }}]->(b)
            """
            session.run(query,
                       source_id=relation.source_id,
                       target_id=relation.target_id,
                       properties=json.dumps(relation.properties),
                       confidence=relation.confidence,
                       source=relation.source)
                       
    def _add_node_networkx(self, node: GraphNode):
        """Add node to NetworkX graph."""
        self.graph.add_node(node.id, **asdict(node))
        
    def _add_relation_networkx(self, relation: GraphRelation):
        """Add relation to NetworkX graph."""
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            **asdict(relation)
        )
        
    def query_neighbors(self, node_id: str, relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Query neighboring nodes.
        
        Args:
            node_id: ID of the central node
            relation_types: Optional list of relation types to filter
            
        Returns:
            List of neighboring nodes with relationship information
        """
        if self.backend == 'neo4j' and self.driver:
            return self._query_neighbors_neo4j(node_id, relation_types)
        elif self.backend == 'networkx' and self.graph is not None:
            return self._query_neighbors_networkx(node_id, relation_types)
        else:
            return self._query_neighbors_mock(node_id, relation_types)
            
    def _query_neighbors_networkx(self, node_id: str, relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Query neighbors using NetworkX."""
        neighbors = []
        
        if node_id not in self.graph:
            return neighbors
            
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            for edge_key, edge_attrs in edge_data.items():
                if relation_types is None or edge_attrs.get('relation_type') in relation_types:
                    neighbor_data = self.graph.nodes[neighbor]
                    neighbors.append({
                        'node_id': neighbor,
                        'node_data': neighbor_data,
                        'relation': edge_attrs
                    })
                    
        return neighbors
        
    def _query_neighbors_mock(self, node_id: str, relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Mock implementation for querying neighbors."""
        neighbors = []
        
        for relation in self.relations:
            if relation.source_id == node_id:
                if relation_types is None or relation.relation_type in relation_types:
                    target_node = self.nodes.get(relation.target_id)
                    if target_node:
                        neighbors.append({
                            'node_id': relation.target_id,
                            'node_data': asdict(target_node),
                            'relation': asdict(relation)
                        })
                        
        return neighbors
        
    def find_semantic_paths(self, source_id: str, target_id: str, max_depth: int = 3) -> List[List[Dict[str, Any]]]:
        """Find semantic paths between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path depth
            
        Returns:
            List of paths, each path is a list of nodes and relations
        """
        if self.backend == 'networkx' and self.graph is not None:
            return self._find_paths_networkx(source_id, target_id, max_depth)
        else:
            return self._find_paths_mock(source_id, target_id, max_depth)
            
    def _find_paths_networkx(self, source_id: str, target_id: str, max_depth: int) -> List[List[Dict[str, Any]]]:
        """Find paths using NetworkX."""
        try:
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_depth))
            semantic_paths = []
            
            for path in paths:
                semantic_path = []
                for i in range(len(path) - 1):
                    current_node = path[i]
                    next_node = path[i + 1]
                    
                    # Get node data
                    node_data = self.graph.nodes[current_node]
                    
                    # Get edge data
                    edge_data = self.graph.get_edge_data(current_node, next_node)
                    
                    semantic_path.append({
                        'node': current_node,
                        'node_data': node_data,
                        'edge_to_next': edge_data
                    })
                    
                # Add final node
                final_node_data = self.graph.nodes[path[-1]]
                semantic_path.append({
                    'node': path[-1],
                    'node_data': final_node_data,
                    'edge_to_next': None
                })
                
                semantic_paths.append(semantic_path)
                
            return semantic_paths
            
        except nx.NetworkXNoPath:
            return []
            
    def _find_paths_mock(self, source_id: str, target_id: str, max_depth: int) -> List[List[Dict[str, Any]]]:
        """Mock implementation for finding paths."""
        # Simple BFS implementation
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
            
        queue = [(source_id, [source_id])]
        visited = set()
        paths = []
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
                
            if current_id == target_id:
                # Convert path to semantic path format
                semantic_path = []
                for i, node_id in enumerate(path):
                    node_data = self.nodes[node_id]
                    edge_to_next = None
                    
                    if i < len(path) - 1:
                        next_node_id = path[i + 1]
                        # Find relation
                        for rel in self.relations:
                            if rel.source_id == node_id and rel.target_id == next_node_id:
                                edge_to_next = asdict(rel)
                                break
                                
                    semantic_path.append({
                        'node': node_id,
                        'node_data': asdict(node_data),
                        'edge_to_next': edge_to_next
                    })
                    
                paths.append(semantic_path)
                continue
                
            if current_id in visited:
                continue
                
            visited.add(current_id)
            
            # Find neighbors
            for rel in self.relations:
                if rel.source_id == current_id and rel.target_id not in path:
                    queue.append((rel.target_id, path + [rel.target_id]))
                    
        return paths
        
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics.
        
        Returns:
            Dictionary containing graph statistics
        """
        stats = {
            'total_nodes': len(self.nodes),
            'total_relations': len(self.relations),
            'node_types': defaultdict(int),
            'relation_types': defaultdict(int),
            'sources': defaultdict(int)
        }
        
        for node in self.nodes.values():
            stats['node_types'][node.node_type] += 1
            stats['sources'][node.source] += 1
            
        for relation in self.relations:
            stats['relation_types'][relation.relation_type] += 1
            
        return dict(stats)
        
    def export_graph(self, format_type: str = 'json', output_path: Optional[str] = None) -> str:
        """Export knowledge graph to various formats.
        
        Args:
            format_type: Export format ('json', 'gexf', 'graphml')
            output_path: Optional output file path
            
        Returns:
            Exported graph data as string or file path
        """
        if format_type == 'json':
            graph_data = {
                'nodes': [asdict(node) for node in self.nodes.values()],
                'relations': [asdict(rel) for rel in self.relations],
                'statistics': self.get_graph_statistics()
            }
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, ensure_ascii=False)
                return output_path
            else:
                return json.dumps(graph_data, indent=2, ensure_ascii=False)
                
        elif format_type in ['gexf', 'graphml'] and self.backend == 'networkx':
            if output_path:
                if format_type == 'gexf':
                    nx.write_gexf(self.graph, output_path)
                elif format_type == 'graphml':
                    nx.write_graphml(self.graph, output_path)
                return output_path
            else:
                self.logger.warning(f"Export format {format_type} requires output_path")
                return ""
        else:
            self.logger.warning(f"Unsupported export format: {format_type}")
            return ""
            
    def close(self):
        """Close graph connections."""
        if self.backend == 'neo4j' and self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")