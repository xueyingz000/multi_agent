"""Cross-modal semantic alignment module for IFC and regulatory text data."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import json

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    cosine_similarity = None
    TfidfVectorizer = None

try:
    import networkx as nx
except ImportError:
    nx = None

from utils.config_loader import get_config
from utils.logger import get_logger
from llm.ner_relation_extractor import Entity, Relation

logger = get_logger(__name__)


@dataclass
class AlignmentResult:
    """Alignment result between IFC and regulatory entities."""
    ifc_entity: str
    regulatory_entity: str
    alignment_type: str  # 'ENTITY', 'ATTRIBUTE', 'RELATIONSHIP'
    confidence_score: float
    semantic_similarity: float
    context_similarity: float
    alignment_evidence: List[str]
    disambiguation_criteria: List[str]
    attributes: Dict[str, Any]


@dataclass
class SemanticMapping:
    """Semantic mapping between IFC and regulatory domains."""
    entity_mappings: Dict[str, List[AlignmentResult]]
    attribute_mappings: Dict[str, List[AlignmentResult]]
    relationship_mappings: Dict[str, List[AlignmentResult]]
    confidence_scores: Dict[str, float]
    mapping_statistics: Dict[str, Any]


class SemanticAlignment:
    """Cross-modal semantic alignment for IFC and regulatory text data.
    
    This module performs:
    - Entity alignment between IFC entities and regulatory terms
    - Attribute alignment between IFC properties and regulatory parameters
    - Relationship alignment between spatial relationships and regulatory logic
    - Context-aware disambiguation
    - Confidence scoring and validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize semantic alignment module.
        
        Args:
            config: Optional configuration override
        """
        self.config = get_config()
        if config:
            self.config.update('semantic_alignment', config)
        
        # Get alignment configuration
        self.similarity_threshold = self.config.get('semantic_alignment.similarity_threshold', 0.3)
        self.confidence_threshold = self.config.get('semantic_alignment.confidence_threshold', 0.4)
        self.alignment_methods = self.config.get('semantic_alignment.methods', ['lexical', 'semantic', 'contextual'])
        
        # Initialize vectorizer for semantic similarity
        self.vectorizer = None
        self._initialize_vectorizer()
        
        # Load predefined mappings and rules
        self.predefined_mappings = self._load_predefined_mappings()
        self.disambiguation_rules = self._load_disambiguation_rules()
        self.context_patterns = self._load_context_patterns()
        
        logger.info("Semantic Alignment module initialized")
    
    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer for semantic similarity."""
        try:
            if TfidfVectorizer is not None:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    lowercase=True
                )
                logger.info("TF-IDF vectorizer initialized")
            else:
                logger.warning("scikit-learn not available, using fallback similarity methods")
        except Exception as e:
            logger.warning(f"Error initializing vectorizer: {e}")
    
    def _load_predefined_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined IFC to regulatory mappings.
        
        Returns:
            Dictionary of predefined mappings
        """
        mappings = {
            # Wall mappings with context disambiguation (English and Chinese)
            'IfcWall': {
                'regulatory_terms': [
                    # English terms
                    'wall', 'partition', 'load-bearing wall', 'curtain wall', 'shear wall', 'exterior wall',
                    # Chinese terms
                    '墙', '墙体', '外墙', '内墙', '承重墙', '隔墙', '幕墙', '剪力墙', '建筑墙体', '墙面'
                ],
                'disambiguation_criteria': {
                    'structural': ['load-bearing', 'structural', 'bearing', 'support', '承重', '结构', '支撑'],
                    'non_structural': ['partition', 'non-bearing', 'interior', '隔墙', '非承重', '内部'],
                    'exterior': ['exterior', 'external', 'facade', 'curtain', '外墙', '外部', '立面', '幕墙'],
                    'fire': ['fire wall', 'fire-rated', 'fire resistance', '防火墙', '防火', '耐火']
                },
                'attributes': {
                    'thickness': ['thickness', 'width', 'depth', '厚度', '宽度', '深度'],
                    'height': ['height', 'elevation', '高度', '标高'],
                    'material': ['material', 'construction', 'composition', '材料', '构造', '组成']
                }
            },
            
            # Slab mappings with functional disambiguation (English and Chinese)
            'IfcSlab': {
                'regulatory_terms': [
                    # English terms
                    'slab', 'floor', 'platform', 'deck', 'terrace', 'balcony',
                    # Chinese terms
                    '楼板', '地板', '平台', '甲板', '露台', '阳台', '板', '楼面', '地面'
                ],
                'disambiguation_criteria': {
                    'floor': ['floor', 'flooring', 'ground', '楼板', '地板', '地面'],
                    'roof': ['roof', 'roofing', 'top', '屋面', '屋顶', '顶部'],
                    'platform': ['platform', 'equipment', 'mechanical', '平台', '设备', '机械'],
                    'structural': ['structural', 'load-bearing', '结构', '承重']
                },
                'attributes': {
                    'thickness': ['thickness', 'depth', '厚度', '深度'],
                    'span': ['span', 'length', 'width', '跨度', '长度', '宽度'],
                    'load_capacity': ['load', 'capacity', 'bearing', '荷载', '承载力', '承重']
                }
            },
            
            # Column mappings
            'IfcColumn': {
                'regulatory_terms': ['column', 'pillar', 'post', 'support', 'pier'],
                'disambiguation_criteria': {
                    'structural': ['structural', 'load-bearing', 'support'],
                    'architectural': ['architectural', 'decorative']
                },
                'attributes': {
                    'cross_section': ['cross section', 'section', 'profile'],
                    'height': ['height', 'length'],
                    'material': ['material', 'construction']
                }
            },
            
            # Beam mappings
            'IfcBeam': {
                'regulatory_terms': ['beam', 'girder', 'joist', 'lintel'],
                'disambiguation_criteria': {
                    'primary': ['main', 'primary', 'girder'],
                    'secondary': ['secondary', 'joist'],
                    'transfer': ['transfer', 'support']
                },
                'attributes': {
                    'span': ['span', 'length'],
                    'depth': ['depth', 'height'],
                    'load_capacity': ['load', 'capacity']
                }
            },
            
            # Opening element mappings with functional context
            'IfcOpeningElement': {
                'regulatory_terms': ['opening', 'aperture', 'void', 'shaft', 'atrium'],
                'disambiguation_criteria': {
                    'shaft': ['shaft', 'vertical', 'elevator', 'stair'],
                    'atrium': ['atrium', 'court', 'open space'],
                    'functional': ['functional', 'service', 'mechanical'],
                    'architectural': ['architectural', 'design', 'aesthetic']
                },
                'attributes': {
                    'area': ['area', 'size'],
                    'height': ['height', 'depth'],
                    'function': ['function', 'purpose', 'use']
                }
            },
            
            # Space mappings
            'IfcSpace': {
                'regulatory_terms': ['room', 'space', 'area', 'zone', 'chamber'],
                'disambiguation_criteria': {
                    'occupiable': ['occupiable', 'habitable', 'occupied'],
                    'service': ['service', 'mechanical', 'utility'],
                    'circulation': ['circulation', 'corridor', 'hallway']
                },
                'attributes': {
                    'area': ['area', 'floor area'],
                    'volume': ['volume', 'cubic'],
                    'occupancy': ['occupancy', 'capacity']
                }
            },
            
            # Door mappings
            'IfcDoor': {
                'regulatory_terms': ['door', 'entrance', 'exit', 'doorway'],
                'disambiguation_criteria': {
                    'fire': ['fire door', 'fire-rated', 'fire exit'],
                    'security': ['security', 'access control'],
                    'emergency': ['emergency', 'exit', 'egress']
                },
                'attributes': {
                    'width': ['width', 'clear width'],
                    'height': ['height', 'clear height'],
                    'fire_rating': ['fire rating', 'fire resistance']
                }
            },
            
            # Window mappings
            'IfcWindow': {
                'regulatory_terms': ['window', 'glazing', 'fenestration', 'skylight'],
                'disambiguation_criteria': {
                    'operable': ['operable', 'openable'],
                    'fixed': ['fixed', 'non-operable'],
                    'emergency': ['emergency', 'egress']
                },
                'attributes': {
                    'area': ['area', 'glazed area'],
                    'u_value': ['u-value', 'thermal transmittance'],
                    'shgc': ['shgc', 'solar heat gain']
                }
            },
            
            # Generic building terms mapping for better Chinese support
            'IfcBuildingElement': {
                'regulatory_terms': [
                    # English terms
                    'building', 'construction', 'structure', 'element', 'component',
                    # Chinese terms
                    '建筑', '建设', '构造', '结构', '构件', '组件', '建筑物', '建筑构件', '建筑元素'
                ],
                'disambiguation_criteria': {
                    'structural': ['structural', 'load-bearing', '结构', '承重'],
                    'architectural': ['architectural', 'design', '建筑', '设计'],
                    'functional': ['functional', 'service', '功能', '服务']
                },
                'attributes': {
                    'area': ['area', 'size', '面积', '尺寸'],
                    'height': ['height', 'elevation', '高度', '标高'],
                    'material': ['material', 'construction', '材料', '构造']
                }
            }
        }
        
        return mappings
    
    def _load_disambiguation_rules(self) -> List[Dict[str, Any]]:
        """Load disambiguation rules for context-dependent mappings.
        
        Returns:
            List of disambiguation rules
        """
        rules = [
            {
                'condition': 'structural_context',
                'keywords': ['load-bearing', 'structural', 'support', 'bearing'],
                'mappings': {
                    'IfcWall': 'load-bearing wall',
                    'IfcSlab': 'structural slab',
                    'IfcColumn': 'structural column'
                }
            },
            {
                'condition': 'fire_safety_context',
                'keywords': ['fire', 'fire-rated', 'fire resistance', 'fire wall'],
                'mappings': {
                    'IfcWall': 'fire wall',
                    'IfcDoor': 'fire door',
                    'IfcSlab': 'fire-rated slab'
                }
            },
            {
                'condition': 'mechanical_context',
                'keywords': ['mechanical', 'equipment', 'service', 'utility'],
                'mappings': {
                    'IfcSpace': 'mechanical room',
                    'IfcSlab': 'equipment platform',
                    'IfcOpeningElement': 'mechanical shaft'
                }
            },
            {
                'condition': 'circulation_context',
                'keywords': ['circulation', 'corridor', 'hallway', 'passage'],
                'mappings': {
                    'IfcSpace': 'circulation space',
                    'IfcSlab': 'walkway',
                    'IfcStair': 'circulation stair'
                }
            },
            {
                'condition': 'exterior_context',
                'keywords': ['exterior', 'external', 'facade', 'outside'],
                'mappings': {
                    'IfcWall': 'exterior wall',
                    'IfcSlab': 'balcony',
                    'IfcWindow': 'exterior window'
                }
            }
        ]
        
        return rules
    
    def _load_context_patterns(self) -> Dict[str, List[str]]:
        """Load context patterns for semantic alignment.
        
        Returns:
            Dictionary of context patterns
        """
        patterns = {
            'spatial_relationships': [
                'located in', 'situated in', 'positioned in',
                'adjacent to', 'next to', 'beside',
                'above', 'below', 'over', 'under',
                'contains', 'encloses', 'houses'
            ],
            'functional_relationships': [
                'supports', 'bears', 'carries',
                'connects to', 'links to', 'joins',
                'separates', 'divides', 'partitions'
            ],
            'material_relationships': [
                'made of', 'constructed of', 'built with',
                'coated with', 'covered with', 'finished with',
                'reinforced with', 'strengthened with'
            ],
            'regulatory_relationships': [
                'complies with', 'meets', 'satisfies',
                'required to', 'shall have', 'must provide',
                'prohibited in', 'not allowed', 'forbidden'
            ]
        }
        
        return patterns
    
    def align_entities(
        self, 
        ifc_entities: List[Dict[str, Any]], 
        regulatory_entities: List[Entity]
    ) -> List[AlignmentResult]:
        """Align IFC entities with regulatory entities.
        
        Args:
            ifc_entities: List of IFC entity information
            regulatory_entities: List of extracted regulatory entities
            
        Returns:
            List of alignment results
        """
        logger.info("Aligning IFC entities with regulatory entities")
        
        alignments = []
        
        for ifc_entity in ifc_entities:
            ifc_type = ifc_entity.get('type', '')
            ifc_name = ifc_entity.get('name', '')
            ifc_context = ifc_entity.get('context', '')
            
            # Find potential regulatory matches
            for reg_entity in regulatory_entities:
                # Handle both Entity objects and dictionary format
                if hasattr(reg_entity, 'label'):
                    # Entity object format
                    reg_label = reg_entity.label
                    reg_text = reg_entity.text
                    reg_context = getattr(reg_entity, 'context', '')
                    reg_attributes = getattr(reg_entity, 'attributes', {})
                else:
                    # Dictionary format
                    reg_label = reg_entity.get('label', '')
                    reg_text = reg_entity.get('text', '')
                    reg_context = reg_entity.get('context', '')
                    reg_attributes = reg_entity.get('attributes', {})
                
                # Check if this is a relevant entity type or try all entities
                if (reg_label in ['BUILDING_COMPONENT', 'SPATIAL_ELEMENT'] or 
                    reg_label in ['ORG', 'PRODUCT', 'FACILITY', 'GPE'] or
                    len(reg_text) > 3):  # Include entities with meaningful text
                    
                    alignment = self._calculate_entity_alignment(
                        ifc_type, ifc_name, ifc_context,
                        reg_text, reg_context, reg_attributes
                    )
                    
                    if alignment and alignment.confidence_score >= self.confidence_threshold:
                        alignments.append(alignment)
        
        # Sort by confidence score
        alignments.sort(key=lambda x: x.confidence_score, reverse=True)
        
        logger.info(f"Generated {len(alignments)} entity alignments")
        return alignments
    
    def align_attributes(
        self, 
        ifc_attributes: List[Dict[str, Any]], 
        regulatory_parameters: List[Dict[str, Any]]
    ) -> List[AlignmentResult]:
        """Align IFC attributes with regulatory parameters.
        
        Args:
            ifc_attributes: List of IFC attribute information
            regulatory_parameters: List of regulatory parameters
            
        Returns:
            List of attribute alignment results
        """
        logger.info("Aligning IFC attributes with regulatory parameters")
        
        alignments = []
        
        for ifc_attr in ifc_attributes:
            ifc_name = ifc_attr.get('name', '')
            ifc_value = ifc_attr.get('value', '')
            ifc_unit = ifc_attr.get('unit', '')
            
            for reg_param in regulatory_parameters:
                reg_name = reg_param.get('name', '')
                reg_value = reg_param.get('value', '')
                reg_context = reg_param.get('context', '')
                
                alignment = self._calculate_attribute_alignment(
                    ifc_name, ifc_value, ifc_unit,
                    reg_name, reg_value, reg_context
                )
                
                if alignment and alignment.confidence_score >= self.confidence_threshold:
                    alignments.append(alignment)
        
        alignments.sort(key=lambda x: x.confidence_score, reverse=True)
        
        logger.info(f"Generated {len(alignments)} attribute alignments")
        return alignments
    
    def align_relationships(
        self, 
        ifc_relationships: List[Dict[str, Any]], 
        regulatory_relations: List[Relation]
    ) -> List[AlignmentResult]:
        """Align IFC relationships with regulatory relationships.
        
        Args:
            ifc_relationships: List of IFC relationship information
            regulatory_relations: List of extracted regulatory relations
            
        Returns:
            List of relationship alignment results
        """
        logger.info("Aligning IFC relationships with regulatory relationships")
        
        alignments = []
        
        for ifc_rel in ifc_relationships:
            ifc_type = ifc_rel.get('type', '')
            ifc_subject = ifc_rel.get('subject', '')
            ifc_object = ifc_rel.get('object', '')
            
            for reg_rel in regulatory_relations:
                alignment = self._calculate_relationship_alignment(
                    ifc_type, ifc_subject, ifc_object,
                    reg_rel.predicate, reg_rel.subject.text, reg_rel.object.text,
                    reg_rel.context
                )
                
                if alignment and alignment.confidence_score >= self.confidence_threshold:
                    alignments.append(alignment)
        
        alignments.sort(key=lambda x: x.confidence_score, reverse=True)
        
        logger.info(f"Generated {len(alignments)} relationship alignments")
        return alignments
    
    def _calculate_entity_alignment(
        self, 
        ifc_type: str, 
        ifc_name: str, 
        ifc_context: str,
        reg_text: str, 
        reg_context: str, 
        reg_attributes: Dict[str, Any]
    ) -> Optional[AlignmentResult]:
        """Calculate alignment between IFC entity and regulatory entity.
        
        Args:
            ifc_type: IFC entity type
            ifc_name: IFC entity name
            ifc_context: IFC entity context
            reg_text: Regulatory entity text
            reg_context: Regulatory entity context
            reg_attributes: Regulatory entity attributes
            
        Returns:
            Alignment result or None
        """
        # Check predefined mappings first
        if ifc_type in self.predefined_mappings:
            mapping_info = self.predefined_mappings[ifc_type]
            regulatory_terms = mapping_info['regulatory_terms']
            
            # Check for direct term match (more flexible matching)
            reg_text_lower = reg_text.lower()
            for term in regulatory_terms:
                term_lower = term.lower()
                # Check for partial matches and related terms
                if (term_lower in reg_text_lower or reg_text_lower in term_lower or
                    any(word in reg_text_lower for word in term_lower.split()) or
                    any(word in term_lower for word in reg_text_lower.split() if len(word) > 2)):
                    # Calculate confidence based on context disambiguation
                    confidence = self._calculate_context_confidence(
                        ifc_context, reg_context, mapping_info.get('disambiguation_criteria', {})
                    )
                    
                    # Calculate semantic similarity
                    semantic_sim = self._calculate_semantic_similarity(ifc_name, reg_text)
                    context_sim = self._calculate_semantic_similarity(ifc_context, reg_context)
                    
                    # Determine disambiguation criteria
                    disambiguation = self._get_disambiguation_criteria(
                        reg_context, mapping_info.get('disambiguation_criteria', {})
                    )
                    
                    return AlignmentResult(
                        ifc_entity=ifc_type,
                        regulatory_entity=reg_text,
                        alignment_type='ENTITY',
                        confidence_score=confidence,
                        semantic_similarity=semantic_sim,
                        context_similarity=context_sim,
                        alignment_evidence=[f"Direct mapping: {term}"],
                        disambiguation_criteria=disambiguation,
                        attributes={
                            'ifc_name': ifc_name,
                            'regulatory_attributes': reg_attributes,
                            'mapping_method': 'predefined'
                        }
                    )
        
        # Check generic building terms if no specific mapping found
        if 'IfcBuildingElement' in self.predefined_mappings:
            generic_mapping = self.predefined_mappings['IfcBuildingElement']
            generic_terms = generic_mapping['regulatory_terms']
            
            reg_text_lower = reg_text.lower()
            for term in generic_terms:
                term_lower = term.lower()
                # Check for partial matches with generic building terms
                if (term_lower in reg_text_lower or reg_text_lower in term_lower or
                    any(word in reg_text_lower for word in term_lower.split()) or
                    any(word in term_lower for word in reg_text_lower.split() if len(word) > 1)):
                    
                    # Calculate confidence for generic match (lower than specific match)
                    confidence = 0.6  # Base confidence for generic matches
                    semantic_sim = self._calculate_semantic_similarity(ifc_name, reg_text)
                    context_sim = self._calculate_semantic_similarity(ifc_context, reg_context)
                    
                    # Boost confidence if semantic similarity is high
                    if semantic_sim > 0.3:
                        confidence = min(0.8, confidence + semantic_sim * 0.3)
                    
                    return AlignmentResult(
                        ifc_entity=ifc_type,
                        regulatory_entity=reg_text,
                        alignment_type='ENTITY',
                        confidence_score=confidence,
                        semantic_similarity=semantic_sim,
                        context_similarity=context_sim,
                        alignment_evidence=[f"Generic building term match: {term}"],
                        disambiguation_criteria=[],
                        attributes={
                            'ifc_name': ifc_name,
                            'regulatory_attributes': reg_attributes,
                            'mapping_method': 'generic_building_term'
                        }
                    )
        
        # Fallback to similarity-based matching
        semantic_sim = self._calculate_semantic_similarity(ifc_name, reg_text)
        context_sim = self._calculate_semantic_similarity(ifc_context, reg_context)
        
        if semantic_sim >= self.similarity_threshold:
            confidence = (semantic_sim + context_sim) / 2
            
            return AlignmentResult(
                ifc_entity=ifc_type,
                regulatory_entity=reg_text,
                alignment_type='ENTITY',
                confidence_score=confidence,
                semantic_similarity=semantic_sim,
                context_similarity=context_sim,
                alignment_evidence=[f"Semantic similarity: {semantic_sim:.3f}"],
                disambiguation_criteria=[],
                attributes={
                    'ifc_name': ifc_name,
                    'regulatory_attributes': reg_attributes,
                    'mapping_method': 'similarity'
                }
            )
        
        return None
    
    def _calculate_attribute_alignment(
        self, 
        ifc_name: str, 
        ifc_value: str, 
        ifc_unit: str,
        reg_name: str, 
        reg_value: str, 
        reg_context: str
    ) -> Optional[AlignmentResult]:
        """Calculate alignment between IFC attribute and regulatory parameter.
        
        Args:
            ifc_name: IFC attribute name
            ifc_value: IFC attribute value
            ifc_unit: IFC attribute unit
            reg_name: Regulatory parameter name
            reg_value: Regulatory parameter value
            reg_context: Regulatory parameter context
            
        Returns:
            Alignment result or None
        """
        # Calculate name similarity
        name_sim = self._calculate_semantic_similarity(ifc_name, reg_name)
        
        # Calculate value similarity if both have values
        value_sim = 0.0
        if ifc_value and reg_value:
            value_sim = self._calculate_value_similarity(ifc_value, reg_value, ifc_unit)
        
        # Combined confidence
        confidence = (name_sim * 0.7 + value_sim * 0.3) if value_sim > 0 else name_sim
        
        if confidence >= self.similarity_threshold:
            return AlignmentResult(
                ifc_entity=ifc_name,
                regulatory_entity=reg_name,
                alignment_type='ATTRIBUTE',
                confidence_score=confidence,
                semantic_similarity=name_sim,
                context_similarity=value_sim,
                alignment_evidence=[f"Name similarity: {name_sim:.3f}"],
                disambiguation_criteria=[],
                attributes={
                    'ifc_value': ifc_value,
                    'ifc_unit': ifc_unit,
                    'regulatory_value': reg_value,
                    'regulatory_context': reg_context
                }
            )
        
        return None
    
    def _calculate_relationship_alignment(
        self, 
        ifc_type: str, 
        ifc_subject: str, 
        ifc_object: str,
        reg_predicate: str, 
        reg_subject: str, 
        reg_object: str, 
        reg_context: str
    ) -> Optional[AlignmentResult]:
        """Calculate alignment between IFC relationship and regulatory relationship.
        
        Args:
            ifc_type: IFC relationship type
            ifc_subject: IFC relationship subject
            ifc_object: IFC relationship object
            reg_predicate: Regulatory relationship predicate
            reg_subject: Regulatory relationship subject
            reg_object: Regulatory relationship object
            reg_context: Regulatory relationship context
            
        Returns:
            Alignment result or None
        """
        # Map IFC relationship types to regulatory predicates
        ifc_to_reg_mapping = {
            'IfcRelContainedInSpatialStructure': ['LOCATED_IN', 'CONTAINED_IN'],
            'IfcRelAggregates': ['CONTAINS', 'COMPOSED_OF'],
            'IfcRelConnectsElements': ['CONNECTS_TO', 'ADJACENT_TO'],
            'IfcRelSpaceBoundary': ['BOUNDS', 'ENCLOSES'],
            'IfcRelVoidsElement': ['VOIDS', 'OPENS']
        }
        
        # Check for direct mapping
        if ifc_type in ifc_to_reg_mapping:
            mapped_predicates = ifc_to_reg_mapping[ifc_type]
            if reg_predicate in mapped_predicates:
                # Calculate entity similarity
                subject_sim = self._calculate_semantic_similarity(ifc_subject, reg_subject)
                object_sim = self._calculate_semantic_similarity(ifc_object, reg_object)
                
                confidence = (subject_sim + object_sim) / 2
                
                if confidence >= self.similarity_threshold:
                    return AlignmentResult(
                        ifc_entity=ifc_type,
                        regulatory_entity=reg_predicate,
                        alignment_type='RELATIONSHIP',
                        confidence_score=confidence,
                        semantic_similarity=confidence,
                        context_similarity=0.8,  # High context similarity for direct mapping
                        alignment_evidence=[f"Direct relationship mapping: {ifc_type} -> {reg_predicate}"],
                        disambiguation_criteria=[],
                        attributes={
                            'ifc_subject': ifc_subject,
                            'ifc_object': ifc_object,
                            'regulatory_subject': reg_subject,
                            'regulatory_object': reg_object,
                            'regulatory_context': reg_context
                        }
                    )
        
        return None
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using LLM.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Try LLM-based similarity first
        llm_sim = self._calculate_llm_semantic_similarity(text1, text2)
        if llm_sim is not None:
            return llm_sim
        
        # Fallback to lexical similarity
        lexical_sim = self._calculate_lexical_similarity(text1, text2)
        
        # TF-IDF similarity as additional fallback if available
        if self.vectorizer is not None and cosine_similarity is not None:
            try:
                vectors = self.vectorizer.fit_transform([text1, text2])
                tfidf_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
                return (lexical_sim + tfidf_sim) / 2
            except Exception as e:
                logger.debug(f"Error calculating TF-IDF similarity: {e}")
        
        return lexical_sim
    
    def _calculate_llm_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using LLM.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1, or None if LLM is unavailable
        """
        try:
            # Import OpenAI client if available
            from openai import OpenAI
            import os
            
            # Check if API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.debug("OpenAI API key not found, falling back to traditional methods")
                return None
            
            client = OpenAI(api_key=api_key)
            
            # Create prompt for semantic similarity evaluation
            prompt = f"""请评估以下两个文本的语义相似度，返回0到1之间的数值，其中1表示完全相同的语义，0表示完全不相关。

文本1: {text1}
文本2: {text2}

请只返回数值，不要解释。考虑以下因素：
1. 概念的语义相关性
2. 上下文的匹配程度
3. 专业术语的对应关系
4. 中英文术语的对应关系

相似度分数（0-1）:"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的语义分析专家，擅长评估建筑和法规领域的文本相似度。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract similarity score from response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract numeric value
            import re
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                score = float(numbers[0])
                # Ensure score is between 0 and 1
                if score > 1:
                    score = score / 100  # Handle percentage format
                return max(0.0, min(1.0, score))
            
            logger.warning(f"Could not parse LLM similarity response: {response_text}")
            return None
            
        except Exception as e:
            logger.debug(f"Error calculating LLM semantic similarity: {e}")
            return None
    
    def _calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """Calculate lexical similarity using Jaccard coefficient.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        # Tokenize and normalize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_value_similarity(self, value1: str, value2: str, unit: str) -> float:
        """Calculate similarity between attribute values.
        
        Args:
            value1: First value
            value2: Second value
            unit: Unit of measurement
            
        Returns:
            Value similarity score
        """
        try:
            # Try to parse as numbers
            num1 = float(value1)
            num2 = float(value2)
            
            # Calculate relative difference
            if max(num1, num2) == 0:
                return 1.0 if num1 == num2 else 0.0
            
            relative_diff = abs(num1 - num2) / max(num1, num2)
            return max(0.0, 1.0 - relative_diff)
            
        except (ValueError, TypeError):
            # Fallback to string similarity
            return self._calculate_lexical_similarity(value1, value2)
    
    def _calculate_context_confidence(
        self, 
        ifc_context: str, 
        reg_context: str, 
        disambiguation_criteria: Dict[str, List[str]]
    ) -> float:
        """Calculate confidence based on context disambiguation.
        
        Args:
            ifc_context: IFC entity context
            reg_context: Regulatory entity context
            disambiguation_criteria: Disambiguation criteria
            
        Returns:
            Context-based confidence score
        """
        base_confidence = 0.7
        
        # Check disambiguation criteria
        for criteria_type, keywords in disambiguation_criteria.items():
            for keyword in keywords:
                if keyword.lower() in reg_context.lower():
                    base_confidence += 0.1
                    break
        
        # Apply disambiguation rules
        for rule in self.disambiguation_rules:
            rule_keywords = rule['keywords']
            if any(keyword.lower() in reg_context.lower() for keyword in rule_keywords):
                base_confidence += 0.15
                break
        
        return min(1.0, base_confidence)
    
    def _get_disambiguation_criteria(
        self, 
        context: str, 
        disambiguation_criteria: Dict[str, List[str]]
    ) -> List[str]:
        """Get applicable disambiguation criteria based on context.
        
        Args:
            context: Context text
            disambiguation_criteria: Available disambiguation criteria
            
        Returns:
            List of applicable criteria
        """
        applicable_criteria = []
        context_lower = context.lower()
        
        for criteria_type, keywords in disambiguation_criteria.items():
            if any(keyword.lower() in context_lower for keyword in keywords):
                applicable_criteria.append(criteria_type)
        
        return applicable_criteria
    
    def create_semantic_mapping(
        self, 
        entity_alignments: List[AlignmentResult],
        attribute_alignments: List[AlignmentResult],
        relationship_alignments: List[AlignmentResult]
    ) -> SemanticMapping:
        """Create comprehensive semantic mapping.
        
        Args:
            entity_alignments: Entity alignment results
            attribute_alignments: Attribute alignment results
            relationship_alignments: Relationship alignment results
            
        Returns:
            Complete semantic mapping
        """
        logger.info("Creating comprehensive semantic mapping")
        
        # Group alignments by type
        entity_mappings = defaultdict(list)
        attribute_mappings = defaultdict(list)
        relationship_mappings = defaultdict(list)
        
        for alignment in entity_alignments:
            entity_mappings[alignment.ifc_entity].append(alignment)
        
        for alignment in attribute_alignments:
            attribute_mappings[alignment.ifc_entity].append(alignment)
        
        for alignment in relationship_alignments:
            relationship_mappings[alignment.ifc_entity].append(alignment)
        
        # Calculate overall confidence scores
        confidence_scores = {
            'entity_alignment': np.mean([a.confidence_score for a in entity_alignments]) if entity_alignments else 0.0,
            'attribute_alignment': np.mean([a.confidence_score for a in attribute_alignments]) if attribute_alignments else 0.0,
            'relationship_alignment': np.mean([a.confidence_score for a in relationship_alignments]) if relationship_alignments else 0.0
        }
        confidence_scores['overall'] = np.mean(list(confidence_scores.values()))
        
        # Generate mapping statistics
        mapping_statistics = {
            'total_entity_mappings': len(entity_alignments),
            'total_attribute_mappings': len(attribute_alignments),
            'total_relationship_mappings': len(relationship_alignments),
            'unique_ifc_entities': len(set(a.ifc_entity for a in entity_alignments)),
            'unique_regulatory_entities': len(set(a.regulatory_entity for a in entity_alignments)),
            'high_confidence_mappings': len([a for a in entity_alignments + attribute_alignments + relationship_alignments if a.confidence_score >= 0.8]),
            'disambiguation_cases': len([a for a in entity_alignments if a.disambiguation_criteria])
        }
        
        semantic_mapping = SemanticMapping(
            entity_mappings=dict(entity_mappings),
            attribute_mappings=dict(attribute_mappings),
            relationship_mappings=dict(relationship_mappings),
            confidence_scores=confidence_scores,
            mapping_statistics=mapping_statistics
        )
        
        logger.info(f"Created semantic mapping with {mapping_statistics['total_entity_mappings']} entity mappings")
        return semantic_mapping
    
    def validate_alignments(self, alignments: List[AlignmentResult]) -> Dict[str, Any]:
        """Validate alignment results and provide quality metrics.
        
        Args:
            alignments: List of alignment results
            
        Returns:
            Validation metrics
        """
        if not alignments:
            return {'valid': False, 'reason': 'No alignments to validate'}
        
        validation_metrics = {
            'total_alignments': len(alignments),
            'average_confidence': np.mean([a.confidence_score for a in alignments]),
            'confidence_distribution': {
                'high': len([a for a in alignments if a.confidence_score >= 0.8]),
                'medium': len([a for a in alignments if 0.6 <= a.confidence_score < 0.8]),
                'low': len([a for a in alignments if a.confidence_score < 0.6])
            },
            'alignment_types': {
                'entity': len([a for a in alignments if a.alignment_type == 'ENTITY']),
                'attribute': len([a for a in alignments if a.alignment_type == 'ATTRIBUTE']),
                'relationship': len([a for a in alignments if a.alignment_type == 'RELATIONSHIP'])
            },
            'disambiguation_coverage': len([a for a in alignments if a.disambiguation_criteria]) / len(alignments),
            'valid': True
        }
        
        return validation_metrics
    
    def export_mappings(self, semantic_mapping: SemanticMapping, format: str = 'json') -> str:
        """Export semantic mappings to specified format.
        
        Args:
            semantic_mapping: Semantic mapping to export
            format: Export format ('json', 'csv', 'rdf')
            
        Returns:
            Exported mapping string
        """
        if format == 'json':
            return self._export_json(semantic_mapping)
        elif format == 'csv':
            return self._export_csv(semantic_mapping)
        elif format == 'rdf':
            return self._export_rdf(semantic_mapping)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, semantic_mapping: SemanticMapping) -> str:
        """Export mappings as JSON."""
        export_data = {
            'entity_mappings': {},
            'attribute_mappings': {},
            'relationship_mappings': {},
            'confidence_scores': semantic_mapping.confidence_scores,
            'statistics': semantic_mapping.mapping_statistics
        }
        
        # Convert alignment results to serializable format
        for ifc_entity, alignments in semantic_mapping.entity_mappings.items():
            export_data['entity_mappings'][ifc_entity] = [
                {
                    'regulatory_entity': a.regulatory_entity,
                    'confidence_score': a.confidence_score,
                    'semantic_similarity': a.semantic_similarity,
                    'context_similarity': a.context_similarity,
                    'alignment_evidence': a.alignment_evidence,
                    'disambiguation_criteria': a.disambiguation_criteria,
                    'attributes': a.attributes
                }
                for a in alignments
            ]
        
        return json.dumps(export_data, indent=2)
    
    def _export_csv(self, semantic_mapping: SemanticMapping) -> str:
        """Export mappings as CSV."""
        lines = ['IFC_Entity,Regulatory_Entity,Alignment_Type,Confidence_Score,Semantic_Similarity,Context_Similarity,Evidence']
        
        all_alignments = []
        for alignments in semantic_mapping.entity_mappings.values():
            all_alignments.extend(alignments)
        for alignments in semantic_mapping.attribute_mappings.values():
            all_alignments.extend(alignments)
        for alignments in semantic_mapping.relationship_mappings.values():
            all_alignments.extend(alignments)
        
        for alignment in all_alignments:
            evidence = '; '.join(alignment.alignment_evidence)
            line = f"{alignment.ifc_entity},{alignment.regulatory_entity},{alignment.alignment_type},{alignment.confidence_score:.3f},{alignment.semantic_similarity:.3f},{alignment.context_similarity:.3f},\"{evidence}\""
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _export_rdf(self, semantic_mapping: SemanticMapping) -> str:
        """Export mappings as RDF/Turtle."""
        rdf_lines = [
            '@prefix ifc: <http://www.buildingsmart-tech.org/ifcOWL/IFC4_ADD2#> .',
            '@prefix reg: <http://example.org/regulatory#> .',
            '@prefix align: <http://example.org/alignment#> .',
            ''
        ]
        
        alignment_id = 0
        for ifc_entity, alignments in semantic_mapping.entity_mappings.items():
            for alignment in alignments:
                alignment_id += 1
                rdf_lines.extend([
                    f'align:alignment_{alignment_id} a align:EntityAlignment ;',
                    f'    align:ifcEntity ifc:{ifc_entity} ;',
                    f'    align:regulatoryEntity reg:{alignment.regulatory_entity.replace(" ", "_")} ;',
                    f'    align:confidenceScore {alignment.confidence_score:.3f} ;',
                    f'    align:semanticSimilarity {alignment.semantic_similarity:.3f} .',
                    ''
                ])
        
        return '\n'.join(rdf_lines)