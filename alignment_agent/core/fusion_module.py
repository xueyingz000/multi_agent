"""Text-Structured Data Fusion Module for IFC Semantic Agent."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from utils.config_loader import get_config
from utils.logger import get_logger
from data_processing.ifc_processor import IFCProcessor
from data_processing.text_processor import TextProcessor

logger = get_logger(__name__)


@dataclass
class MultiModalData:
    """Multi-modal data container."""
    ifc_data: Dict[str, Any]
    text_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str


@dataclass
class FusionResult:
    """Data fusion result container."""
    fused_entities: List[Dict[str, Any]]
    alignment_scores: Dict[str, float]
    semantic_mappings: Dict[str, List[str]]
    confidence_scores: Dict[str, float]


class TextStructuredDataFusion:
    """Text-Structured Data Fusion Module.
    
    This module handles the fusion of IFC structured data with regulatory text data
    to provide multi-perspective semantic understanding.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the fusion module.
        
        Args:
            config: Optional configuration override
        """
        self.config = get_config()
        if config:
            self.config.update('fusion_module', config)
        
        self.ifc_processor = IFCProcessor()
        self.text_processor = TextProcessor()
        
        # Initialize fusion parameters
        self.similarity_threshold = self.config.get('semantic_alignment.similarity_threshold', 0.7)
        self.fusion_weights = {
            'semantic': 0.4,
            'structural': 0.3,
            'contextual': 0.3
        }
        
        logger.info("Text-Structured Data Fusion Module initialized")
    
    def process_multimodal_input(
        self, 
        ifc_file_path: str, 
        text_data_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MultiModalData:
        """Process multi-modal data input.
        
        Step 1: Multi-modal Data Input
        - 3D-Model Data (IFC): Building entities, attributes, spatial relationships
        - Regulatory Text Data: Structured Markdown text content extracted from PDF
        
        Args:
            ifc_file_path: Path to IFC file
            text_data_path: Path to regulatory text data
            metadata: Optional metadata
            
        Returns:
            MultiModalData object containing processed data
        """
        logger.info(f"Processing multi-modal input: IFC={ifc_file_path}, Text={text_data_path}")
        
        try:
            # Process IFC data
            ifc_data = self.ifc_processor.process_ifc_file(ifc_file_path)
            logger.info(f"Processed IFC data: {len(ifc_data.get('entities', []))} entities")
            
            # Process text data
            text_data = self.text_processor.process_text_file(text_data_path)
            logger.info(f"Processed text data: {len(text_data.get('chunks', []))} chunks")
            
            # Create multi-modal data container
            multimodal_data = MultiModalData(
                ifc_data=ifc_data,
                text_data=text_data,
                metadata=metadata or {},
                timestamp=self._get_timestamp()
            )
            
            return multimodal_data
            
        except Exception as e:
            logger.error(f"Error processing multi-modal input: {e}")
            raise
    
    def extract_semantic_features(
        self, 
        multimodal_data: MultiModalData
    ) -> Dict[str, Any]:
        """Extract semantic features from multi-modal data.
        
        Args:
            multimodal_data: Multi-modal data container
            
        Returns:
            Dictionary containing extracted semantic features
        """
        logger.info("Extracting semantic features from multi-modal data")
        
        features = {
            'ifc_features': self._extract_ifc_features(multimodal_data.ifc_data),
            'text_features': self._extract_text_features(multimodal_data.text_data),
            'cross_modal_features': self._extract_cross_modal_features(
                multimodal_data.ifc_data, 
                multimodal_data.text_data
            )
        }
        
        return features
    
    def fuse_data(
        self, 
        multimodal_data: MultiModalData,
        semantic_features: Dict[str, Any]
    ) -> FusionResult:
        """Fuse IFC and text data using semantic alignment.
        
        Args:
            multimodal_data: Multi-modal data container
            semantic_features: Extracted semantic features
            
        Returns:
            FusionResult containing fused data and alignment information
        """
        logger.info("Fusing IFC and text data")
        
        try:
            # Perform entity alignment
            entity_alignments = self._align_entities(
                semantic_features['ifc_features'],
                semantic_features['text_features']
            )
            
            # Perform attribute alignment
            attribute_alignments = self._align_attributes(
                semantic_features['ifc_features'],
                semantic_features['text_features']
            )
            
            # Perform relationship alignment
            relationship_alignments = self._align_relationships(
                semantic_features['ifc_features'],
                semantic_features['text_features']
            )
            
            # Create fused entities
            fused_entities = self._create_fused_entities(
                entity_alignments,
                attribute_alignments,
                relationship_alignments
            )
            
            # Calculate alignment scores
            alignment_scores = self._calculate_alignment_scores(
                entity_alignments,
                attribute_alignments,
                relationship_alignments
            )
            
            # Create semantic mappings
            semantic_mappings = self._create_semantic_mappings(
                entity_alignments,
                attribute_alignments,
                relationship_alignments
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                alignment_scores,
                semantic_features
            )
            
            fusion_result = FusionResult(
                fused_entities=fused_entities,
                alignment_scores=alignment_scores,
                semantic_mappings=semantic_mappings,
                confidence_scores=confidence_scores
            )
            
            logger.info(f"Data fusion completed: {len(fused_entities)} fused entities")
            return fusion_result
            
        except Exception as e:
            logger.error(f"Error during data fusion: {e}")
            raise
    
    def _extract_ifc_features(self, ifc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from IFC data.
        
        Args:
            ifc_data: IFC data dictionary
            
        Returns:
            Dictionary containing IFC features
        """
        entities = ifc_data.get('entities', [])
        relationships = ifc_data.get('relationships', [])
        
        features = {
            'entity_types': [entity.get('type') for entity in entities],
            'entity_attributes': [entity.get('attributes', {}) for entity in entities],
            'spatial_relationships': relationships,
            'geometric_properties': [entity.get('geometry', {}) for entity in entities],
            'material_properties': [entity.get('materials', []) for entity in entities]
        }
        
        return features
    
    def _extract_text_features(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from text data.
        
        Args:
            text_data: Text data dictionary
            
        Returns:
            Dictionary containing text features
        """
        chunks = text_data.get('chunks', [])
        entities = text_data.get('entities', [])
        relationships = text_data.get('relationships', [])
        
        features = {
            'text_chunks': chunks,
            'named_entities': entities,
            'extracted_relationships': relationships,
            'regulatory_terms': text_data.get('regulatory_terms', []),
            'semantic_embeddings': text_data.get('embeddings', [])
        }
        
        return features
    
    def _extract_cross_modal_features(self, ifc_data: Dict[str, Any], text_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cross-modal features.
        
        Args:
            ifc_data: IFC data dictionary
            text_data: Text data dictionary
            
        Returns:
            Dictionary containing cross-modal features
        """
        # This would involve more sophisticated cross-modal analysis
        # For now, we'll implement basic feature extraction
        
        features = {
            'common_terms': self._find_common_terms(ifc_data, text_data),
            'semantic_overlap': self._calculate_semantic_overlap(ifc_data, text_data),
            'contextual_similarity': self._calculate_contextual_similarity(ifc_data, text_data)
        }
        
        return features
    
    def _align_entities(self, ifc_features: Dict[str, Any], text_features: Dict[str, Any]) -> Dict[str, Any]:
        """Align IFC entities with regulatory terms.
        
        Args:
            ifc_features: IFC features
            text_features: Text features
            
        Returns:
            Entity alignment mappings
        """
        alignments = {}
        
        ifc_entities = ifc_features.get('entity_types', [])
        regulatory_terms = text_features.get('regulatory_terms', [])
        
        for ifc_entity in ifc_entities:
            if ifc_entity:
                aligned_terms = self._find_aligned_terms(ifc_entity, regulatory_terms)
                if aligned_terms:
                    alignments[ifc_entity] = aligned_terms
        
        return alignments
    
    def _align_attributes(self, ifc_features: Dict[str, Any], text_features: Dict[str, Any]) -> Dict[str, Any]:
        """Align IFC attributes with regulatory parameters.
        
        Args:
            ifc_features: IFC features
            text_features: Text features
            
        Returns:
            Attribute alignment mappings
        """
        alignments = {}
        
        # Implementation would involve sophisticated attribute matching
        # This is a simplified version
        
        return alignments
    
    def _align_relationships(self, ifc_features: Dict[str, Any], text_features: Dict[str, Any]) -> Dict[str, Any]:
        """Align spatial relationships with regulatory logic.
        
        Args:
            ifc_features: IFC features
            text_features: Text features
            
        Returns:
            Relationship alignment mappings
        """
        alignments = {}
        
        # Implementation would involve relationship pattern matching
        # This is a simplified version
        
        return alignments
    
    def _find_aligned_terms(self, ifc_entity: str, regulatory_terms: List[str]) -> List[str]:
        """Find regulatory terms aligned with IFC entity.
        
        Args:
            ifc_entity: IFC entity type
            regulatory_terms: List of regulatory terms
            
        Returns:
            List of aligned terms
        """
        # Predefined mappings for common IFC entities
        entity_mappings = {
            'IfcWall': ['Wall', 'Partition Wall', 'Load-bearing Wall', 'Structural Wall'],
            'IfcSlab': ['Slab', 'Platform', 'Terrace', 'Balcony', 'Floor'],
            'IfcColumn': ['Column', 'Pillar', 'Support', 'Structural Column'],
            'IfcBeam': ['Beam', 'Girder', 'Structural Beam'],
            'IfcDoor': ['Door', 'Entrance', 'Exit', 'Opening'],
            'IfcWindow': ['Window', 'Opening', 'Glazing'],
            'IfcSpace': ['Space', 'Room', 'Area', 'Zone'],
            'IfcOpeningElement': ['Opening', 'Shaft', 'Atrium', 'Void'],
            'IfcStair': ['Stair', 'Staircase', 'Steps'],
            'IfcRoof': ['Roof', 'Roofing', 'Cover']
        }
        
        aligned_terms = []
        predefined_terms = entity_mappings.get(ifc_entity, [])
        
        for term in regulatory_terms:
            if any(pred_term.lower() in term.lower() for pred_term in predefined_terms):
                aligned_terms.append(term)
        
        return aligned_terms
    
    def _create_fused_entities(self, entity_alignments, attribute_alignments, relationship_alignments) -> List[Dict[str, Any]]:
        """Create fused entities from alignment results."""
        fused_entities = []
        
        for ifc_entity, aligned_terms in entity_alignments.items():
            fused_entity = {
                'ifc_type': ifc_entity,
                'regulatory_terms': aligned_terms,
                'attributes': attribute_alignments.get(ifc_entity, {}),
                'relationships': relationship_alignments.get(ifc_entity, {}),
                'fusion_timestamp': self._get_timestamp()
            }
            fused_entities.append(fused_entity)
        
        return fused_entities
    
    def _calculate_alignment_scores(self, entity_alignments, attribute_alignments, relationship_alignments) -> Dict[str, float]:
        """Calculate alignment scores."""
        scores = {}
        
        # Calculate entity alignment score
        entity_score = len(entity_alignments) / max(len(entity_alignments) + 1, 1)
        scores['entity_alignment'] = entity_score
        
        # Calculate attribute alignment score
        attr_score = len(attribute_alignments) / max(len(attribute_alignments) + 1, 1)
        scores['attribute_alignment'] = attr_score
        
        # Calculate relationship alignment score
        rel_score = len(relationship_alignments) / max(len(relationship_alignments) + 1, 1)
        scores['relationship_alignment'] = rel_score
        
        # Calculate overall score
        scores['overall'] = (
            entity_score * self.fusion_weights['semantic'] +
            attr_score * self.fusion_weights['structural'] +
            rel_score * self.fusion_weights['contextual']
        )
        
        return scores
    
    def _create_semantic_mappings(self, entity_alignments, attribute_alignments, relationship_alignments) -> Dict[str, List[str]]:
        """Create semantic mappings."""
        mappings = {
            'entities': entity_alignments,
            'attributes': attribute_alignments,
            'relationships': relationship_alignments
        }
        
        return mappings
    
    def _calculate_confidence_scores(self, alignment_scores, semantic_features) -> Dict[str, float]:
        """Calculate confidence scores."""
        confidence = {}
        
        # Base confidence on alignment scores
        confidence['alignment_confidence'] = alignment_scores.get('overall', 0.0)
        
        # Factor in feature quality
        feature_quality = self._assess_feature_quality(semantic_features)
        confidence['feature_quality'] = feature_quality
        
        # Overall confidence
        confidence['overall'] = (confidence['alignment_confidence'] + confidence['feature_quality']) / 2
        
        return confidence
    
    def _assess_feature_quality(self, semantic_features) -> float:
        """Assess the quality of extracted features."""
        # Simplified feature quality assessment
        ifc_quality = len(semantic_features.get('ifc_features', {}).get('entity_types', [])) > 0
        text_quality = len(semantic_features.get('text_features', {}).get('named_entities', [])) > 0
        
        return float(ifc_quality and text_quality)
    
    def _find_common_terms(self, ifc_data, text_data) -> List[str]:
        """Find common terms between IFC and text data."""
        # Simplified implementation
        return []
    
    def _calculate_semantic_overlap(self, ifc_data, text_data) -> float:
        """Calculate semantic overlap between IFC and text data."""
        # Simplified implementation
        return 0.5
    
    def _calculate_contextual_similarity(self, ifc_data, text_data) -> float:
        """Calculate contextual similarity between IFC and text data."""
        # Simplified implementation
        return 0.5
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()