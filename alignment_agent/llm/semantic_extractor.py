"""LLM-based semantic extractor for regulatory texts."""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticExtractionResult:
    """Semantic extraction result container."""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    key_concepts: List[Dict[str, Any]]
    regulatory_requirements: List[Dict[str, Any]]
    semantic_mappings: Dict[str, List[str]]
    confidence_scores: Dict[str, float]


class SemanticExtractor:
    """LLM-based semantic extractor for building regulatory texts.
    
    This extractor uses Large Language Models to perform:
    - Advanced Named Entity Recognition (NER)
    - Relationship extraction between building components
    - Key concept identification
    - Regulatory requirement extraction
    - Semantic mapping generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize semantic extractor.
        
        Args:
            config: Optional configuration override
        """
        self.config = get_config()
        if config:
            self.config.update('semantic_extractor', config)
        
        # Get LLM configuration
        self.model_name = self.config.get('llm.model_name', 'gpt-3.5-turbo')
        self.temperature = self.config.get('llm.temperature', 0.1)
        self.max_tokens = self.config.get('llm.max_tokens', 2048)
        
        # Initialize OpenAI client
        self.client = None
        self._initialize_llm_client()
        
        # Load extraction prompts
        self.prompts = self._load_extraction_prompts()
        
        logger.info("Semantic Extractor initialized")
    
    def _initialize_llm_client(self):
        """Initialize LLM client."""
        try:
            if openai is not None:
                api_key = self.config.get('llm.api_key')
                base_url = self.config.get('llm.base_url')
                
                if api_key:
                    self.client = openai.OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )
                    logger.info("OpenAI client initialized")
                else:
                    logger.warning("OpenAI API key not found in configuration")
            else:
                logger.warning("OpenAI library not available")
                
        except Exception as e:
            logger.warning(f"Error initializing LLM client: {e}")
    
    def _load_extraction_prompts(self) -> Dict[str, str]:
        """Load extraction prompts for different tasks.
        
        Returns:
            Dictionary of prompts by task type
        """
        prompts = {
            'entity_extraction': """
You are an expert in building codes and construction regulations. Extract named entities from the following regulatory text.

Focus on identifying:
1. Building components (walls, slabs, columns, beams, doors, windows, etc.)
2. Spatial elements (rooms, spaces, areas, zones, etc.)
3. Materials (concrete, steel, wood, etc.)
4. Dimensions and measurements
5. Regulatory terms and requirements

Text: {text}

Return the results in JSON format with the following structure:
{
  "entities": [
    {
      "text": "entity text",
      "category": "BUILDING_COMPONENT|SPATIAL_ELEMENT|MATERIAL|DIMENSION|REGULATORY_TERM",
      "subcategory": "specific subcategory",
      "confidence": 0.95,
      "context": "surrounding context"
    }
  ]
}
""",
            
            'relationship_extraction': """
You are an expert in building codes and construction regulations. Extract relationships between building components and regulatory concepts from the following text.

Focus on identifying:
1. Spatial relationships (located in, adjacent to, above, below, etc.)
2. Functional relationships (supports, contains, connects to, etc.)
3. Regulatory relationships (complies with, requires, specifies, etc.)
4. Material relationships (made of, coated with, reinforced with, etc.)

Text: {text}

Return the results in JSON format with the following structure:
{
  "relationships": [
    {
      "subject": "subject entity",
      "predicate": "relationship type",
      "object": "object entity",
      "relationship_category": "SPATIAL|FUNCTIONAL|REGULATORY|MATERIAL",
      "confidence": 0.90,
      "context": "sentence or phrase containing the relationship"
    }
  ]
}
""",
            
            'concept_extraction': """
You are an expert in building codes and construction regulations. Extract key concepts and their definitions from the following regulatory text.

Focus on identifying:
1. Technical definitions of building terms
2. Regulatory requirements and specifications
3. Safety and performance criteria
4. Design standards and guidelines

Text: {text}

Return the results in JSON format with the following structure:
{
  "concepts": [
    {
      "term": "concept term",
      "definition": "concept definition or description",
      "category": "DEFINITION|REQUIREMENT|CRITERIA|STANDARD",
      "importance": "HIGH|MEDIUM|LOW",
      "related_terms": ["related term 1", "related term 2"],
      "context": "full context where concept appears"
    }
  ]
}
""",
            
            'requirement_extraction': """
You are an expert in building codes and construction regulations. Extract specific regulatory requirements from the following text.

Focus on identifying:
1. Mandatory requirements (shall, must, required)
2. Dimensional specifications (minimum, maximum dimensions)
3. Performance criteria (strength, fire resistance, etc.)
4. Compliance conditions (when, where, how requirements apply)

Text: {text}

Return the results in JSON format with the following structure:
{
  "requirements": [
    {
      "requirement_text": "full requirement statement",
      "requirement_type": "MANDATORY|DIMENSIONAL|PERFORMANCE|CONDITIONAL",
      "applies_to": ["building component or situation"],
      "conditions": ["conditions when requirement applies"],
      "parameters": {
        "parameter_name": "parameter_value"
      },
      "compliance_level": "SHALL|SHOULD|MAY",
      "context": "surrounding regulatory context"
    }
  ]
}
""",
            
            'semantic_mapping': """
You are an expert in building information modeling (BIM) and construction regulations. Create semantic mappings between IFC building components and regulatory terminology.

Given the following regulatory text, identify how IFC entities should be mapped to regulatory terms:

Text: {text}

Common IFC entities to consider:
- IfcWall, IfcSlab, IfcColumn, IfcBeam
- IfcDoor, IfcWindow, IfcSpace
- IfcOpeningElement, IfcBuildingStorey
- IfcStair, IfcRoof, IfcFoundation

Return the results in JSON format with the following structure:
{
  "mappings": [
    {
      "ifc_entity": "IfcWall",
      "regulatory_terms": ["wall", "partition", "load-bearing wall"],
      "context_dependent": true,
      "disambiguation_criteria": ["structural function", "location", "material"],
      "confidence": 0.85
    }
  ]
}
"""
        }
        
        return prompts
    
    def extract_semantic_information(
        self, 
        text: str, 
        extraction_types: Optional[List[str]] = None
    ) -> SemanticExtractionResult:
        """Extract semantic information from regulatory text using LLM.
        
        Args:
            text: Input regulatory text
            extraction_types: Types of extraction to perform. 
                            If None, performs all types.
            
        Returns:
            SemanticExtractionResult containing extracted information
        """
        logger.info("Extracting semantic information using LLM")
        
        if extraction_types is None:
            extraction_types = ['entities', 'relationships', 'concepts', 'requirements', 'mappings']
        
        try:
            # Extract entities
            entities = []
            if 'entities' in extraction_types:
                entities = self._extract_entities_llm(text)
            
            # Extract relationships
            relationships = []
            if 'relationships' in extraction_types:
                relationships = self._extract_relationships_llm(text)
            
            # Extract key concepts
            key_concepts = []
            if 'concepts' in extraction_types:
                key_concepts = self._extract_concepts_llm(text)
            
            # Extract regulatory requirements
            regulatory_requirements = []
            if 'requirements' in extraction_types:
                regulatory_requirements = self._extract_requirements_llm(text)
            
            # Generate semantic mappings
            semantic_mappings = {}
            if 'mappings' in extraction_types:
                semantic_mappings = self._generate_semantic_mappings_llm(text)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                entities, relationships, key_concepts, regulatory_requirements
            )
            
            result = SemanticExtractionResult(
                entities=entities,
                relationships=relationships,
                key_concepts=key_concepts,
                regulatory_requirements=regulatory_requirements,
                semantic_mappings=semantic_mappings,
                confidence_scores=confidence_scores
            )
            
            logger.info(f"Semantic extraction completed: {len(entities)} entities, {len(relationships)} relationships")
            return result
            
        except Exception as e:
            logger.error(f"Error in semantic extraction: {e}")
            raise
    
    def _extract_entities_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using LLM.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        if self.client is None:
            logger.warning("LLM client not available, using fallback entity extraction")
            return self._extract_entities_fallback(text)
        
        try:
            prompt = self.prompts['entity_extraction'].format(text=text)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in building codes and construction regulations. Always return valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_text = response.choices[0].message.content
            if not result_text or len(result_text.strip()) < 10:
                logger.warning(f"LLM returned empty or too short response: '{result_text}'")
                return self._extract_entities_fallback(text)
                
            result_json = self._parse_json_response(result_text)
            
            return result_json.get('entities', [])
            
        except Exception as e:
            logger.warning(f"Error in LLM entity extraction: {e}")
            return self._extract_entities_fallback(text)
    
    def _extract_relationships_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships using LLM.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted relationships
        """
        if self.client is None:
            logger.warning("LLM client not available, using fallback relationship extraction")
            return self._extract_relationships_fallback(text)
        
        try:
            prompt = self.prompts['relationship_extraction'].format(text=text)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in building codes and construction regulations. Always return valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_text = response.choices[0].message.content
            if not result_text or len(result_text.strip()) < 10:
                logger.warning(f"LLM returned empty or too short response: '{result_text}'")
                return self._extract_relationships_fallback(text)
                
            result_json = self._parse_json_response(result_text)
            
            return result_json.get('relationships', [])
            
        except Exception as e:
            logger.warning(f"Error in LLM relationship extraction: {e}")
            return self._extract_relationships_fallback(text)
    
    def _extract_concepts_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract key concepts using LLM.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted concepts
        """
        if self.client is None:
            logger.warning("LLM client not available, using fallback concept extraction")
            return self._extract_concepts_fallback(text)
        
        try:
            prompt = self.prompts['concept_extraction'].format(text=text)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in building codes and construction regulations. Always return valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_text = response.choices[0].message.content
            if not result_text or len(result_text.strip()) < 10:
                logger.warning(f"LLM returned empty or too short response: '{result_text}'")
                return self._extract_concepts_fallback(text)
                
            result_json = self._parse_json_response(result_text)
            
            return result_json.get('concepts', [])
            
        except Exception as e:
            logger.warning(f"Error in LLM concept extraction: {e}")
            return self._extract_concepts_fallback(text)
    
    def _extract_requirements_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract regulatory requirements using LLM.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted requirements
        """
        if self.client is None:
            logger.warning("LLM client not available, using fallback requirement extraction")
            return self._extract_requirements_fallback(text)
        
        try:
            prompt = self.prompts['requirement_extraction'].format(text=text)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in building codes and construction regulations. Always return valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_text = response.choices[0].message.content
            if not result_text or len(result_text.strip()) < 10:
                logger.warning(f"LLM returned empty or too short response: '{result_text}'")
                return self._extract_requirements_fallback(text)
                
            result_json = self._parse_json_response(result_text)
            
            return result_json.get('requirements', [])
            
        except Exception as e:
            logger.warning(f"Error in LLM requirement extraction: {e}")
            return self._extract_requirements_fallback(text)
    
    def _generate_semantic_mappings_llm(self, text: str) -> Dict[str, List[str]]:
        """Generate semantic mappings using LLM.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of semantic mappings
        """
        if self.client is None:
            logger.warning("LLM client not available, using fallback semantic mapping")
            return self._generate_semantic_mappings_fallback(text)
        
        try:
            prompt = self.prompts['semantic_mapping'].format(text=text)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in building information modeling (BIM) and construction regulations. Always return valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_text = response.choices[0].message.content
            if not result_text or len(result_text.strip()) < 10:
                logger.warning(f"LLM returned empty or too short response: '{result_text}'")
                return []
                
            result_json = self._parse_json_response(result_text)
            
            # Convert mappings to dictionary format
            mappings = {}
            for mapping in result_json.get('mappings', []):
                ifc_entity = mapping.get('ifc_entity')
                regulatory_terms = mapping.get('regulatory_terms', [])
                if ifc_entity:
                    mappings[ifc_entity] = regulatory_terms
            
            return mappings
            
        except Exception as e:
            logger.warning(f"Error in LLM semantic mapping: {e}")
            return self._generate_semantic_mappings_fallback(text)
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Check if response is too short or incomplete
            if len(cleaned_text) < 10:
                logger.warning(f"Response too short: '{cleaned_text}'")
                return {}
            
            # Try to extract JSON from response using multiple patterns
            json_patterns = [
                r'```json\s*({.*?})\s*```',  # JSON in code blocks
                r'```\s*({.*?})\s*```',  # JSON in generic code blocks
                r'({\s*"[^"]+"\s*:\s*\[[^\]]*\]\s*})',  # Simple key-array pattern
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON pattern
                r'\{.*?\}',  # Simple JSON pattern
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
                if json_match:
                    json_text = json_match.group(1) if json_match.groups() else json_match.group()
                    try:
                        # Try to fix common JSON issues
                        json_text = self._fix_json_format(json_text)
                        parsed_json = json.loads(json_text)
                        # Validate that the JSON has expected structure
                        if isinstance(parsed_json, dict) and len(parsed_json) > 0:
                            return parsed_json
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON decode error for pattern: {e}")
                        continue
            
            # Try to construct JSON from partial response
            constructed_json = self._construct_json_from_partial(cleaned_text)
            if constructed_json:
                return constructed_json
            
            # If no JSON pattern matches, try parsing the entire response
            try:
                fixed_text = self._fix_json_format(cleaned_text)
                parsed_json = json.loads(fixed_text)
                if isinstance(parsed_json, dict):
                    return parsed_json
            except json.JSONDecodeError:
                pass
                
        except Exception as e:
            logger.warning(f"Unexpected error parsing JSON response: {e}")
            
        # Log the problematic response for debugging
        logger.warning(f"Failed to parse JSON response. Raw response: '{response_text[:200]}...'")
        return {}
    
    def _fix_json_format(self, json_text: str) -> str:
        """Fix common JSON formatting issues.
        
        Args:
            json_text: Raw JSON text
            
        Returns:
            Fixed JSON text
        """
        # Remove leading/trailing whitespace and newlines
        json_text = json_text.strip()
        
        # Fix missing quotes around keys
        json_text = re.sub(r'(\w+)\s*:', r'"\1":', json_text)
        
        # Fix trailing commas
        json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
        
        # Fix single quotes to double quotes
        json_text = json_text.replace("'", '"')
        
        # Ensure proper JSON structure
        if not json_text.startswith('{'):
            json_text = '{' + json_text
        if not json_text.endswith('}'):
            json_text = json_text + '}'
            
        return json_text
    
    def _construct_json_from_partial(self, text: str) -> Dict[str, Any]:
        """Construct JSON from partial response text.
        
        Args:
            text: Partial response text
            
        Returns:
            Constructed JSON dictionary or empty dict
        """
        try:
            # Look for key patterns in the text
            result = {}
            
            # Extract entities pattern
            entities_match = re.search(r'"entities"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if entities_match:
                result['entities'] = []
            
            # Extract relationships pattern
            relationships_match = re.search(r'"relationships"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if relationships_match:
                result['relationships'] = []
            
            # Extract concepts pattern
            concepts_match = re.search(r'"concepts"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if concepts_match:
                result['concepts'] = []
            
            # Extract requirements pattern
            requirements_match = re.search(r'"requirements"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if requirements_match:
                result['requirements'] = []
            
            # Extract mappings pattern
            mappings_match = re.search(r'"mappings"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if mappings_match:
                result['mappings'] = []
            
            return result if result else {}
            
        except Exception as e:
            logger.debug(f"Error constructing JSON from partial text: {e}")
            return {}
    
    def _calculate_confidence_scores(
        self, 
        entities: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]], 
        concepts: List[Dict[str, Any]], 
        requirements: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for extraction results.
        
        Args:
            entities: Extracted entities
            relationships: Extracted relationships
            concepts: Extracted concepts
            requirements: Extracted requirements
            
        Returns:
            Dictionary of confidence scores
        """
        scores = {}
        
        # Entity extraction confidence
        if entities:
            entity_confidences = [e.get('confidence', 0.5) for e in entities]
            scores['entity_extraction'] = sum(entity_confidences) / len(entity_confidences)
        else:
            scores['entity_extraction'] = 0.0
        
        # Relationship extraction confidence
        if relationships:
            rel_confidences = [r.get('confidence', 0.5) for r in relationships]
            scores['relationship_extraction'] = sum(rel_confidences) / len(rel_confidences)
        else:
            scores['relationship_extraction'] = 0.0
        
        # Overall confidence
        scores['overall'] = (scores['entity_extraction'] + scores['relationship_extraction']) / 2
        
        return scores
    
    # Fallback methods when LLM is not available
    def _extract_entities_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity extraction using regex patterns."""
        entities = []
        
        # Basic building component patterns
        patterns = {
            'BUILDING_COMPONENT': [
                r'\b(?:wall|partition|load[\-\s]bearing\s+wall)\b',
                r'\b(?:slab|floor|platform)\b',
                r'\b(?:column|pillar|support)\b',
                r'\b(?:beam|girder)\b',
                r'\b(?:door|entrance|exit)\b',
                r'\b(?:window|glazing)\b'
            ],
            'MATERIAL': [
                r'\b(?:concrete|steel|wood|timber|brick)\b'
            ],
            'DIMENSION': [
                r'\d+(?:\.\d+)?\s*(?:mm|cm|m|ft|in)\b'
            ]
        }
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'category': category,
                        'confidence': 0.6,
                        'context': text[max(0, match.start()-50):match.end()+50]
                    })
        
        return entities
    
    def _extract_relationships_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback relationship extraction using regex patterns."""
        relationships = []
        
        # Basic relationship patterns
        patterns = [
            (r'(\w+)\s+(?:located|situated)\s+(?:in|within)\s+(\w+)', 'LOCATED_IN'),
            (r'(\w+)\s+(?:adjacent|next)\s+to\s+(\w+)', 'ADJACENT_TO'),
            (r'(\w+)\s+(?:above|over)\s+(\w+)', 'ABOVE')
        ]
        
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    'subject': match.group(1),
                    'predicate': rel_type,
                    'object': match.group(2),
                    'confidence': 0.5,
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return relationships
    
    def _extract_concepts_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback concept extraction using predefined building concepts."""
        concepts = []
        
        # Define building concept patterns
        concept_patterns = {
            'STRUCTURAL': [
                r'\b(?:structural|load[\-\s]bearing|support|foundation)\b',
                r'\b(?:reinforcement|concrete|steel|frame)\b'
            ],
            'SAFETY': [
                r'\b(?:fire[\-\s]safety|emergency|evacuation|exit)\b',
                r'\b(?:smoke|alarm|sprinkler|detection)\b'
            ],
            'ACCESSIBILITY': [
                r'\b(?:accessible|disability|wheelchair|ramp)\b',
                r'\b(?:barrier[\-\s]free|universal[\-\s]design)\b'
            ],
            'ENERGY': [
                r'\b(?:energy[\-\s]efficient|insulation|thermal)\b',
                r'\b(?:ventilation|hvac|heating|cooling)\b'
            ],
            'COMPLIANCE': [
                r'\b(?:building[\-\s]code|regulation|standard|requirement)\b',
                r'\b(?:permit|approval|inspection|compliance)\b'
            ]
        }
        
        for category, pattern_list in concept_patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    concepts.append({
                        'concept': match.group(),
                        'category': category,
                        'confidence': 0.7,
                        'context': text[max(0, match.start()-50):match.end()+50]
                    })
        
        return concepts
    
    def _extract_requirements_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback requirement extraction."""
        requirements = []
        
        # Look for requirement keywords
        requirement_patterns = [
            r'shall\s+([^.]+)',
            r'must\s+([^.]+)',
            r'required\s+to\s+([^.]+)'
        ]
        
        for pattern in requirement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                requirements.append({
                    'requirement_text': match.group(),
                    'requirement_type': 'MANDATORY',
                    'compliance_level': 'SHALL',
                    'context': text[max(0, match.start()-100):match.end()+100]
                })
        
        return requirements
    
    def _generate_semantic_mappings_fallback(self, text: str) -> Dict[str, List[str]]:
        """Fallback semantic mapping generation."""
        # Basic predefined mappings
        mappings = {
            'IfcWall': ['wall', 'partition'],
            'IfcSlab': ['slab', 'floor'],
            'IfcColumn': ['column', 'pillar'],
            'IfcBeam': ['beam', 'girder'],
            'IfcDoor': ['door', 'entrance'],
            'IfcWindow': ['window', 'glazing']
        }
        
        return mappings