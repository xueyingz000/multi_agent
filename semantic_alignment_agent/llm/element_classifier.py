"""LLM-based Element Classifier

This module provides intelligent element classification using Large Language Models
to analyze IFC elements and determine their functional semantics.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .llm_client import LLMClient
from .prompt_templates import PromptTemplates
from ..utils.data_structures import ClassificationResult

logger = logging.getLogger(__name__)

class LLMElementClassifier:
    """LLM-powered element classifier for semantic analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the LLM element classifier.
        
        Args:
            config_path: Path to configuration file
        """
        self.llm_client = LLMClient(config_path)
        self.prompt_templates = PromptTemplates()
        
        # Fallback classification rules for when LLM is unavailable
        self.fallback_rules = self._init_fallback_rules()
    
    def _init_fallback_rules(self) -> Dict[str, Dict]:
        """Initialize fallback classification rules."""
        return {
            'IfcSpace': {
                'office': {'keywords': ['office', 'work'], 'min_area': 5.0},
                'meeting_room': {'keywords': ['meeting', 'conference'], 'min_area': 8.0},
                'corridor': {'keywords': ['corridor', 'hallway'], 'aspect_ratio': 3.0},
                'storage': {'keywords': ['storage', 'closet'], 'max_area': 20.0},
                'mechanical': {'keywords': ['mechanical', 'equipment'], 'max_area': 50.0}
            },
            'IfcSlab': {
                'floor_slab': {'thickness_range': (0.1, 0.5), 'position': 'horizontal'},
                'roof_slab': {'thickness_range': (0.15, 0.6), 'position': 'top'},
                'foundation_slab': {'thickness_range': (0.2, 1.0), 'position': 'bottom'}
            }
        }
    
    def classify_element(
        self,
        ifc_element: Dict[str, Any],
        geometric_features: Dict[str, Any],
        spatial_context: Dict[str, Any],
        building_context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        """Classify an IFC element using LLM analysis.
        
        Args:
            ifc_element: IFC element data
            geometric_features: Geometric analysis results
            spatial_context: Spatial context information
            building_context: Building-level context
            
        Returns:
            Classification result with confidence scores
        """
        element_type = ifc_element.get('ifc_type', 'Unknown')
        element_guid = ifc_element.get('guid', 'Unknown')
        
        logger.info(f"Classifying element {element_guid} of type {element_type}")
        
        # Try LLM-based classification first
        if self.llm_client.is_available():
            try:
                return self._llm_classify_element(
                    ifc_element, geometric_features, spatial_context, building_context
                )
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, falling back to rule-based")
        
        # Fallback to rule-based classification
        return self._fallback_classify_element(
            ifc_element, geometric_features, spatial_context
        )
    
    def _llm_classify_element(
        self,
        ifc_element: Dict[str, Any],
        geometric_features: Dict[str, Any],
        spatial_context: Dict[str, Any],
        building_context: Optional[Dict[str, Any]]
    ) -> ClassificationResult:
        """Perform LLM-based element classification."""
        # Prepare context for LLM
        context = {
            'ifc_type': ifc_element.get('ifc_type', 'Unknown'),
            'guid': ifc_element.get('guid', 'Unknown'),
            'geometric_features': json.dumps(geometric_features, indent=2),
            'property_sets': json.dumps(ifc_element.get('properties', {}), indent=2),
            'spatial_context': json.dumps(spatial_context, indent=2),
            'building_context': json.dumps(building_context or {}, indent=2),
            'region_info': 'International',
            'project_phase': 'Design Development'
        }
        
        # Generate classification prompt
        prompt = self.prompt_templates.get_element_classification_prompt(**context)
        
        # Get LLM analysis with confidence scoring
        llm_result = self.llm_client.analyze_with_confidence(
            prompt, context, temperature=0.1
        )
        
        # Parse LLM response
        return self._parse_llm_classification_result(llm_result, ifc_element)
    
    def _parse_llm_classification_result(
        self,
        llm_result: Dict[str, Any],
        ifc_element: Dict[str, Any]
    ) -> ClassificationResult:
        """Parse LLM classification result into structured format."""
        try:
            # Try to parse structured JSON response from LLM analysis
            analysis_text = llm_result.get('analysis', '')
            
            # Extract classification from analysis or conclusion
            conclusion = llm_result.get('conclusion', '')
            confidence = llm_result.get('confidence_score', 0.5)
            
            # Try to extract structured data from analysis
            classification_data = self._extract_classification_from_text(
                analysis_text + ' ' + conclusion
            )
            
            return ClassificationResult(
                element_guid=ifc_element.get('guid', 'Unknown'),
                element_type=ifc_element.get('ifc_type', 'Unknown'),
                primary_function=classification_data.get('primary_function', 'unknown'),
                sub_category=classification_data.get('sub_category', 'general'),
                usage_intensity=classification_data.get('usage_intensity', 'medium'),
                confidence_score=confidence,
                reasoning=llm_result.get('analysis', 'LLM analysis'),
                alternative_classifications=self._extract_alternatives(llm_result),
                regulatory_hints=classification_data.get('regulatory_hints', []),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM classification result: {e}")
            return self._create_default_classification_result(ifc_element, 0.3)
    
    def _extract_classification_from_text(self, text: str) -> Dict[str, Any]:
        """Extract classification information from LLM text response."""
        text_lower = text.lower()
        
        # Simple keyword-based extraction as fallback
        classification = {
            'primary_function': 'unknown',
            'sub_category': 'general',
            'usage_intensity': 'medium',
            'regulatory_hints': []
        }
        
        # Extract primary function
        function_keywords = {
            'office': ['office', 'workspace', 'work'],
            'meeting_room': ['meeting', 'conference', 'boardroom'],
            'corridor': ['corridor', 'hallway', 'circulation'],
            'storage': ['storage', 'closet', 'warehouse'],
            'mechanical': ['mechanical', 'equipment', 'utility'],
            'floor_slab': ['floor', 'slab', 'deck'],
            'roof_slab': ['roof', 'ceiling'],
            'residential': ['residential', 'living', 'bedroom', 'kitchen']
        }
        
        for function, keywords in function_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                classification['primary_function'] = function
                break
        
        # Extract usage intensity
        if any(word in text_lower for word in ['high', 'frequent', 'busy']):
            classification['usage_intensity'] = 'high'
        elif any(word in text_lower for word in ['low', 'rare', 'minimal']):
            classification['usage_intensity'] = 'low'
        
        return classification
    
    def _extract_alternatives(self, llm_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract alternative classifications from LLM result."""
        alternatives = []
        
        # Look for uncertainty areas that might suggest alternatives
        uncertainty_areas = llm_result.get('uncertainty_areas', [])
        for area in uncertainty_areas:
            if 'could be' in area.lower() or 'might be' in area.lower():
                alternatives.append({
                    'function_category': 'alternative',
                    'confidence_score': 0.3,
                    'reasoning': area
                })
        
        return alternatives[:3]  # Limit to top 3 alternatives
    
    def _fallback_classify_element(
        self,
        ifc_element: Dict[str, Any],
        geometric_features: Dict[str, Any],
        spatial_context: Dict[str, Any]
    ) -> ClassificationResult:
        """Perform rule-based classification as fallback."""
        element_type = ifc_element.get('ifc_type', 'Unknown')
        
        # Apply rule-based classification
        if element_type in self.fallback_rules:
            classification = self._apply_fallback_rules(
                element_type, ifc_element, geometric_features
            )
        else:
            classification = {
                'primary_function': 'unknown',
                'confidence': 0.2
            }
        
        return ClassificationResult(
            element_guid=ifc_element.get('guid', 'Unknown'),
            element_type=element_type,
            primary_function=classification.get('primary_function', 'unknown'),
            sub_category='general',
            usage_intensity='medium',
            confidence_score=classification.get('confidence', 0.2),
            reasoning='Rule-based fallback classification',
            alternative_classifications=[],
            regulatory_hints=[],
            timestamp=datetime.now().isoformat()
        )
    
    def _apply_fallback_rules(
        self,
        element_type: str,
        ifc_element: Dict[str, Any],
        geometric_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply rule-based classification logic."""
        rules = self.fallback_rules.get(element_type, {})
        
        # Get element properties
        name = ifc_element.get('name', '').lower()
        area = geometric_features.get('area', 0.0)
        
        # Check each rule
        for function, criteria in rules.items():
            confidence = 0.5
            
            # Check keyword matches
            if 'keywords' in criteria:
                if any(keyword in name for keyword in criteria['keywords']):
                    confidence += 0.3
            
            # Check area constraints
            if 'min_area' in criteria and area >= criteria['min_area']:
                confidence += 0.1
            if 'max_area' in criteria and area <= criteria['max_area']:
                confidence += 0.1
            
            # If confidence is high enough, return this classification
            if confidence > 0.6:
                return {
                    'primary_function': function,
                    'confidence': min(confidence, 0.8)  # Cap at 0.8 for rule-based
                }
        
        # Default classification
        return {
            'primary_function': 'unknown',
            'confidence': 0.2
        }
    
    def _create_default_classification_result(
        self,
        ifc_element: Dict[str, Any],
        confidence: float = 0.1
    ) -> ClassificationResult:
        """Create a default classification result for error cases."""
        return ClassificationResult(
            element_guid=ifc_element.get('guid', 'Unknown'),
            element_type=ifc_element.get('ifc_type', 'Unknown'),
            primary_function='unknown',
            sub_category='general',
            usage_intensity='medium',
            confidence_score=confidence,
            reasoning='Classification failed, using default',
            alternative_classifications=[],
            regulatory_hints=[],
            timestamp=datetime.now().isoformat()
        )
    
    def batch_classify_elements(
        self,
        elements_data: List[Dict[str, Any]]
    ) -> List[ClassificationResult]:
        """Classify multiple elements in batch.
        
        Args:
            elements_data: List of element data dictionaries
            
        Returns:
            List of classification results
        """
        results = []
        
        for element_data in elements_data:
            try:
                result = self.classify_element(
                    element_data.get('ifc_element', {}),
                    element_data.get('geometric_features', {}),
                    element_data.get('spatial_context', {}),
                    element_data.get('building_context', {})
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify element: {e}")
                results.append(
                    self._create_default_classification_result(
                        element_data.get('ifc_element', {})
                    )
                )
        
        return results