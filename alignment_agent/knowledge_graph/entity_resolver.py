"""Entity Resolver for disambiguating and resolving entities in the knowledge graph."""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from difflib import SequenceMatcher

from .graph_builder import GraphBuilder, GraphNode
from utils import ConfigLoader, get_logger


@dataclass
class EntityCandidate:
    """Represents a candidate entity for resolution."""
    node_id: str
    entity_text: str
    entity_type: str
    confidence_score: float
    disambiguation_features: Dict[str, Any]
    source_context: str
    

@dataclass
class ResolutionResult:
    """Represents the result of entity resolution."""
    original_text: str
    resolved_entity: Optional[EntityCandidate]
    all_candidates: List[EntityCandidate]
    resolution_method: str
    confidence_score: float
    disambiguation_reasoning: List[str]
    

class EntityResolver:
    """Resolves and disambiguates entities in IFC-regulatory knowledge graph."""
    
    def __init__(self, graph_builder: GraphBuilder, config_path: str = "config.yaml"):
        """Initialize the entity resolver.
        
        Args:
            graph_builder: Initialized GraphBuilder instance
            config_path: Path to configuration file
        """
        self.graph_builder = graph_builder
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)
        
        # Load disambiguation rules and patterns
        self._load_disambiguation_rules()
        
        # Entity type hierarchies
        self._build_entity_hierarchies()
        
        # Context patterns for disambiguation
        self._load_context_patterns()
        
        # Similarity thresholds
        self.similarity_threshold = self.config.get('entity_resolution.similarity_threshold', 0.7)
        self.confidence_threshold = self.config.get('entity_resolution.confidence_threshold', 0.6)
        
    def _load_disambiguation_rules(self):
        """Load disambiguation rules and mappings."""
        self.disambiguation_rules = {
            # IFC entity disambiguation
            'IfcSlab': {
                'structural_indicators': ['load', 'bearing', 'support', 'structural'],
                'platform_indicators': ['equipment', 'platform', 'mechanical', 'service'],
                'floor_indicators': ['floor', 'level', 'story', 'occupancy'],
                'balcony_indicators': ['balcony', 'terrace', 'outdoor', 'external']
            },
            'IfcWall': {
                'load_bearing_indicators': ['load', 'bearing', 'structural', 'support'],
                'partition_indicators': ['partition', 'divider', 'non-load', 'interior'],
                'curtain_indicators': ['curtain', 'facade', 'external', 'glazing'],
                'shear_indicators': ['shear', 'lateral', 'seismic', 'wind']
            },
            'IfcOpeningElement': {
                'shaft_indicators': ['shaft', 'vertical', 'service', 'utility'],
                'atrium_indicators': ['atrium', 'void', 'open', 'multi-story'],
                'door_indicators': ['door', 'entrance', 'access', 'opening'],
                'window_indicators': ['window', 'glazing', 'light', 'view']
            },
            'IfcSpace': {
                'room_indicators': ['room', 'office', 'bedroom', 'kitchen'],
                'corridor_indicators': ['corridor', 'hallway', 'passage', 'circulation'],
                'stair_indicators': ['stair', 'staircase', 'step', 'flight'],
                'elevator_indicators': ['elevator', 'lift', 'vertical', 'transport']
            }
        }
        
        # Regulatory term disambiguation
        self.regulatory_disambiguation = {
            'wall': {
                'fire_wall': ['fire', 'rated', 'separation', 'compartment'],
                'party_wall': ['party', 'shared', 'adjacent', 'property'],
                'retaining_wall': ['retaining', 'earth', 'soil', 'pressure'],
                'foundation_wall': ['foundation', 'basement', 'below', 'grade']
            },
            'slab': {
                'floor_slab': ['floor', 'level', 'occupancy', 'finish'],
                'roof_slab': ['roof', 'top', 'weather', 'drainage'],
                'equipment_slab': ['equipment', 'mechanical', 'platform', 'service']
            },
            'opening': {
                'door_opening': ['door', 'entrance', 'egress', 'access'],
                'window_opening': ['window', 'glazing', 'daylight', 'ventilation'],
                'service_opening': ['service', 'utility', 'penetration', 'duct']
            }
        }
        
    def _build_entity_hierarchies(self):
        """Build entity type hierarchies for better matching."""
        self.ifc_hierarchy = {
            'IfcBuildingElement': {
                'IfcWall': ['IfcWallStandardCase', 'IfcCurtainWall'],
                'IfcSlab': ['IfcSlabStandardCase', 'IfcSlabElementedCase'],
                'IfcBeam': ['IfcBeamStandardCase'],
                'IfcColumn': ['IfcColumnStandardCase'],
                'IfcDoor': ['IfcDoorStandardCase'],
                'IfcWindow': ['IfcWindowStandardCase']
            },
            'IfcSpatialElement': {
                'IfcSpace': ['IfcSpaceType'],
                'IfcBuildingStorey': [],
                'IfcBuilding': [],
                'IfcSite': []
            },
            'IfcFeatureElement': {
                'IfcOpeningElement': ['IfcVoidingFeature'],
                'IfcProjectionElement': []
            }
        }
        
        self.regulatory_hierarchy = {
            'structural_elements': {
                'walls': ['load_bearing_wall', 'shear_wall', 'partition_wall'],
                'slabs': ['floor_slab', 'roof_slab', 'equipment_platform'],
                'beams': ['primary_beam', 'secondary_beam', 'transfer_beam'],
                'columns': ['structural_column', 'architectural_column']
            },
            'spatial_elements': {
                'rooms': ['office', 'bedroom', 'bathroom', 'kitchen'],
                'circulation': ['corridor', 'stairway', 'elevator_shaft'],
                'service': ['mechanical_room', 'electrical_room', 'storage']
            },
            'openings': {
                'doors': ['entrance_door', 'fire_door', 'emergency_exit'],
                'windows': ['fixed_window', 'operable_window', 'curtain_wall'],
                'penetrations': ['duct_opening', 'pipe_opening', 'cable_opening']
            }
        }
        
    def _load_context_patterns(self):
        """Load context patterns for disambiguation."""
        self.context_patterns = {
            'spatial_context': {
                'above': r'\b(above|over|on top of|upper)\b',
                'below': r'\b(below|under|beneath|lower)\b',
                'adjacent': r'\b(adjacent|next to|beside|neighboring)\b',
                'within': r'\b(within|inside|contained in|part of)\b',
                'supports': r'\b(supports|bears|carries|holds)\b'
            },
            'functional_context': {
                'structural': r'\b(structural|load.bearing|support|strength)\b',
                'architectural': r'\b(architectural|aesthetic|finish|appearance)\b',
                'mechanical': r'\b(mechanical|HVAC|equipment|service)\b',
                'electrical': r'\b(electrical|power|lighting|wiring)\b',
                'fire_safety': r'\b(fire|safety|emergency|egress)\b'
            },
            'material_context': {
                'concrete': r'\b(concrete|reinforced|cast.in.place)\b',
                'steel': r'\b(steel|metal|structural.steel)\b',
                'wood': r'\b(wood|timber|lumber)\b',
                'masonry': r'\b(masonry|brick|block|stone)\b'
            }
        }
        
    def resolve_entity(self, entity_text: str, context: str = "", entity_type_hint: Optional[str] = None) -> ResolutionResult:
        """Resolve an entity mention to knowledge graph entities.
        
        Args:
            entity_text: Text mention of the entity
            context: Surrounding context for disambiguation
            entity_type_hint: Optional hint about entity type
            
        Returns:
            Resolution result with best candidate and alternatives
        """
        # Find candidate entities
        candidates = self._find_entity_candidates(entity_text, entity_type_hint)
        
        if not candidates:
            return ResolutionResult(
                original_text=entity_text,
                resolved_entity=None,
                all_candidates=[],
                resolution_method='no_candidates',
                confidence_score=0.0,
                disambiguation_reasoning=["No matching entities found in knowledge graph"]
            )
            
        # Disambiguate candidates using context
        disambiguated_candidates = self._disambiguate_candidates(candidates, context, entity_text)
        
        # Select best candidate
        best_candidate = self._select_best_candidate(disambiguated_candidates)
        
        # Generate reasoning
        reasoning = self._generate_disambiguation_reasoning(entity_text, context, disambiguated_candidates, best_candidate)
        
        return ResolutionResult(
            original_text=entity_text,
            resolved_entity=best_candidate,
            all_candidates=disambiguated_candidates,
            resolution_method='context_disambiguation',
            confidence_score=best_candidate.confidence_score if best_candidate else 0.0,
            disambiguation_reasoning=reasoning
        )
        
    def _find_entity_candidates(self, entity_text: str, entity_type_hint: Optional[str] = None) -> List[EntityCandidate]:
        """Find candidate entities from the knowledge graph."""
        candidates = []
        entity_text_lower = entity_text.lower()
        
        # Search through all nodes in the knowledge graph
        for node_id, node in self.graph_builder.nodes.items():
            # Calculate text similarity
            label_similarity = self._calculate_text_similarity(entity_text_lower, node.label.lower())
            
            # Check properties for additional matches
            property_similarities = []
            for prop_key, prop_value in node.properties.items():
                if isinstance(prop_value, str):
                    prop_similarity = self._calculate_text_similarity(entity_text_lower, prop_value.lower())
                    property_similarities.append(prop_similarity)
                elif isinstance(prop_value, dict):
                    for sub_key, sub_value in prop_value.items():
                        if isinstance(sub_value, str):
                            sub_similarity = self._calculate_text_similarity(entity_text_lower, sub_value.lower())
                            property_similarities.append(sub_similarity)
                            
            # Calculate overall similarity
            max_prop_similarity = max(property_similarities) if property_similarities else 0.0
            overall_similarity = max(label_similarity, max_prop_similarity)
            
            # Filter by similarity threshold
            if overall_similarity >= self.similarity_threshold:
                # Type matching bonus
                type_bonus = 0.0
                if entity_type_hint and self._matches_entity_type(node, entity_type_hint):
                    type_bonus = 0.2
                    
                confidence = min(overall_similarity + type_bonus, 1.0)
                
                candidate = EntityCandidate(
                    node_id=node_id,
                    entity_text=node.label,
                    entity_type=node.node_type,
                    confidence_score=confidence,
                    disambiguation_features=self._extract_disambiguation_features(node),
                    source_context=node.source
                )
                
                candidates.append(candidate)
                
        # Sort by confidence score
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return candidates[:10]  # Return top 10 candidates
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods."""
        # Exact match
        if text1 == text2:
            return 1.0
            
        # Sequence matcher
        seq_similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # Substring matching
        substring_similarity = 0.0
        if text1 in text2 or text2 in text1:
            shorter = min(len(text1), len(text2))
            longer = max(len(text1), len(text2))
            substring_similarity = shorter / longer
            
        # Word overlap
        words1 = set(text1.split())
        words2 = set(text2.split())
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            word_overlap = 0.0
            
        # Combined similarity
        return max(seq_similarity, substring_similarity, word_overlap)
        
    def _matches_entity_type(self, node: GraphNode, entity_type_hint: str) -> bool:
        """Check if node matches the entity type hint."""
        node_type = node.node_type.lower()
        hint_lower = entity_type_hint.lower()
        
        # Direct match
        if hint_lower in node_type:
            return True
            
        # Check IFC type in properties
        ifc_type = node.properties.get('ifc_type', '').lower()
        if hint_lower in ifc_type:
            return True
            
        # Check category in properties
        category = node.properties.get('category', '').lower()
        if hint_lower in category:
            return True
            
        return False
        
    def _extract_disambiguation_features(self, node: GraphNode) -> Dict[str, Any]:
        """Extract features for disambiguation."""
        features = {
            'node_type': node.node_type,
            'source': node.source,
            'label': node.label
        }
        
        # Extract key properties
        properties = node.properties
        
        if node.node_type == 'ifc_entity':
            features.update({
                'ifc_type': properties.get('ifc_type'),
                'attributes': properties.get('attributes', {}),
                'geometry': properties.get('geometry', {})
            })
            
        elif node.node_type == 'regulatory_term':
            features.update({
                'category': properties.get('category'),
                'text': properties.get('text'),
                'context': properties.get('context', '')
            })
            
        return features
        
    def _disambiguate_candidates(self, candidates: List[EntityCandidate], context: str, entity_text: str) -> List[EntityCandidate]:
        """Disambiguate candidates using context."""
        if not context:
            return candidates
            
        context_lower = context.lower()
        disambiguated = []
        
        for candidate in candidates:
            # Calculate context relevance
            context_score = self._calculate_context_relevance(candidate, context_lower, entity_text)
            
            # Adjust confidence score
            adjusted_confidence = (candidate.confidence_score * 0.7) + (context_score * 0.3)
            
            # Create new candidate with adjusted score
            disambiguated_candidate = EntityCandidate(
                node_id=candidate.node_id,
                entity_text=candidate.entity_text,
                entity_type=candidate.entity_type,
                confidence_score=adjusted_confidence,
                disambiguation_features=candidate.disambiguation_features,
                source_context=candidate.source_context
            )
            
            disambiguated.append(disambiguated_candidate)
            
        # Re-sort by adjusted confidence
        disambiguated.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return disambiguated
        
    def _calculate_context_relevance(self, candidate: EntityCandidate, context: str, entity_text: str) -> float:
        """Calculate how relevant a candidate is given the context."""
        relevance_score = 0.0
        
        # Get disambiguation rules for the candidate
        entity_type = candidate.disambiguation_features.get('ifc_type', '')
        if entity_type in self.disambiguation_rules:
            rules = self.disambiguation_rules[entity_type]
            
            for rule_type, indicators in rules.items():
                for indicator in indicators:
                    if indicator.lower() in context:
                        relevance_score += 0.2
                        
        # Check context patterns
        for pattern_category, patterns in self.context_patterns.items():
            for pattern_name, pattern_regex in patterns.items():
                if re.search(pattern_regex, context, re.IGNORECASE):
                    # Check if this pattern is relevant to the candidate
                    if self._is_pattern_relevant(candidate, pattern_category, pattern_name):
                        relevance_score += 0.15
                        
        # Regulatory term specific disambiguation
        if candidate.entity_type == 'regulatory_term':
            candidate_text = candidate.disambiguation_features.get('text', '').lower()
            for reg_type, subtypes in self.regulatory_disambiguation.items():
                if reg_type in candidate_text or reg_type in entity_text.lower():
                    for subtype, indicators in subtypes.items():
                        for indicator in indicators:
                            if indicator in context:
                                relevance_score += 0.25
                                
        return min(relevance_score, 1.0)
        
    def _is_pattern_relevant(self, candidate: EntityCandidate, pattern_category: str, pattern_name: str) -> bool:
        """Check if a context pattern is relevant to a candidate."""
        entity_type = candidate.disambiguation_features.get('ifc_type', '').lower()
        
        # Define relevance mappings
        relevance_map = {
            'spatial_context': {
                'above': ['slab', 'beam', 'roof'],
                'below': ['foundation', 'basement', 'slab'],
                'supports': ['beam', 'column', 'wall'],
                'within': ['space', 'room', 'opening']
            },
            'functional_context': {
                'structural': ['wall', 'beam', 'column', 'slab'],
                'mechanical': ['space', 'room', 'opening'],
                'fire_safety': ['door', 'wall', 'opening']
            },
            'material_context': {
                'concrete': ['wall', 'slab', 'beam', 'column'],
                'steel': ['beam', 'column', 'frame'],
                'wood': ['beam', 'wall', 'frame']
            }
        }
        
        if pattern_category in relevance_map and pattern_name in relevance_map[pattern_category]:
            relevant_types = relevance_map[pattern_category][pattern_name]
            return any(rel_type in entity_type for rel_type in relevant_types)
            
        return False
        
    def _select_best_candidate(self, candidates: List[EntityCandidate]) -> Optional[EntityCandidate]:
        """Select the best candidate from disambiguated list."""
        if not candidates:
            return None
            
        # Return the highest scoring candidate if it meets confidence threshold
        best_candidate = candidates[0]
        
        if best_candidate.confidence_score >= self.confidence_threshold:
            return best_candidate
        else:
            return None
            
    def _generate_disambiguation_reasoning(self, entity_text: str, context: str, candidates: List[EntityCandidate], best_candidate: Optional[EntityCandidate]) -> List[str]:
        """Generate human-readable reasoning for disambiguation."""
        reasoning = []
        
        reasoning.append(f"Resolving entity mention: '{entity_text}'")
        
        if context:
            reasoning.append(f"Using context: '{context[:100]}...'")
            
        reasoning.append(f"Found {len(candidates)} candidate entities")
        
        if best_candidate:
            reasoning.append(f"Selected best match: {best_candidate.entity_text} (confidence: {best_candidate.confidence_score:.2f})")
            reasoning.append(f"Entity type: {best_candidate.entity_type}")
            reasoning.append(f"Source: {best_candidate.source_context}")
        else:
            reasoning.append("No candidate met the confidence threshold")
            if candidates:
                top_candidate = candidates[0]
                reasoning.append(f"Top candidate was: {top_candidate.entity_text} (confidence: {top_candidate.confidence_score:.2f})")
                
        return reasoning
        
    def resolve_entities_batch(self, entity_mentions: List[Dict[str, str]]) -> List[ResolutionResult]:
        """Resolve multiple entity mentions in batch.
        
        Args:
            entity_mentions: List of dicts with 'text', 'context', and optional 'type_hint'
            
        Returns:
            List of resolution results
        """
        results = []
        
        for mention in entity_mentions:
            entity_text = mention.get('text', '')
            context = mention.get('context', '')
            type_hint = mention.get('type_hint')
            
            result = self.resolve_entity(entity_text, context, type_hint)
            results.append(result)
            
        return results
        
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about entities in the knowledge graph."""
        stats = {
            'total_entities': len(self.graph_builder.nodes),
            'entity_types': defaultdict(int),
            'sources': defaultdict(int),
            'ifc_types': defaultdict(int),
            'regulatory_categories': defaultdict(int)
        }
        
        for node in self.graph_builder.nodes.values():
            stats['entity_types'][node.node_type] += 1
            stats['sources'][node.source] += 1
            
            if node.node_type == 'ifc_entity':
                ifc_type = node.properties.get('ifc_type', 'Unknown')
                stats['ifc_types'][ifc_type] += 1
                
            elif node.node_type == 'regulatory_term':
                category = node.properties.get('category', 'Unknown')
                stats['regulatory_categories'][category] += 1
                
        return dict(stats)
        
    def export_disambiguation_rules(self, output_path: str):
        """Export disambiguation rules to JSON file."""
        rules_data = {
            'disambiguation_rules': self.disambiguation_rules,
            'regulatory_disambiguation': self.regulatory_disambiguation,
            'context_patterns': self.context_patterns,
            'ifc_hierarchy': self.ifc_hierarchy,
            'regulatory_hierarchy': self.regulatory_hierarchy
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Disambiguation rules exported to {output_path}")