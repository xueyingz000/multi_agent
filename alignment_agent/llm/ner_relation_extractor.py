"""Named Entity Recognition and Relation Extraction module."""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

try:
    import spacy
    from spacy.matcher import Matcher
except ImportError:
    spacy = None
    Matcher = None

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    """Named entity representation."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    attributes: Dict[str, Any]
    context: str


@dataclass
class Relation:
    """Relation representation."""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float
    relation_type: str
    context: str
    attributes: Dict[str, Any]


class NERRelationExtractor:
    """Named Entity Recognition and Relation Extraction for building regulatory texts.
    
    This extractor specializes in:
    - Building component entity recognition
    - Spatial relationship extraction
    - Regulatory term identification
    - IFC-specific entity mapping
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize NER and relation extractor.
        
        Args:
            config: Optional configuration override
        """
        self.config = get_config()
        if config:
            self.config.update('ner_relation_extractor', config)
        
        # Initialize spaCy model
        self.nlp = None
        self.matcher = None
        self._initialize_spacy()
        
        # Load entity patterns and rules
        self.entity_patterns = self._load_entity_patterns()
        self.relation_patterns = self._load_relation_patterns()
        self.ifc_mappings = self._load_ifc_mappings()
        
        logger.info("NER Relation Extractor initialized")
    
    def _initialize_spacy(self):
        """Initialize spaCy NLP pipeline."""
        try:
            if spacy is not None:
                model_name = self.config.get('nlp.spacy_model', 'en_core_web_sm')
                
                # Try to load the model
                try:
                    self.nlp = spacy.load(model_name)
                    self.matcher = Matcher(self.nlp.vocab)
                    logger.info(f"Loaded spaCy model: {model_name}")
                except OSError:
                    logger.warning(f"spaCy model '{model_name}' not found, using fallback methods")
                    self.nlp = None
            else:
                logger.warning("spaCy not available, using regex-based extraction")
                
        except Exception as e:
            logger.warning(f"Error initializing spaCy: {e}")
    
    def _load_entity_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load entity recognition patterns.
        
        Returns:
            Dictionary of entity patterns by category
        """
        patterns = {
            'BUILDING_COMPONENT': [
                # Walls
                {'pattern': r'\b(?:wall|partition|load[\-\s]bearing\s+wall|curtain\s+wall|shear\s+wall)\b', 'subcategory': 'wall'},
                {'pattern': r'\b(?:exterior\s+wall|interior\s+wall|bearing\s+wall|non[\-\s]bearing\s+wall)\b', 'subcategory': 'wall'},
                
                # Slabs and floors
                {'pattern': r'\b(?:slab|floor|platform|deck|terrace|balcony)\b', 'subcategory': 'slab'},
                {'pattern': r'\b(?:concrete\s+slab|floor\s+slab|roof\s+slab|foundation\s+slab)\b', 'subcategory': 'slab'},
                
                # Columns and supports
                {'pattern': r'\b(?:column|pillar|post|support|pier)\b', 'subcategory': 'column'},
                {'pattern': r'\b(?:steel\s+column|concrete\s+column|composite\s+column)\b', 'subcategory': 'column'},
                
                # Beams
                {'pattern': r'\b(?:beam|girder|joist|lintel)\b', 'subcategory': 'beam'},
                {'pattern': r'\b(?:steel\s+beam|concrete\s+beam|wood\s+beam|transfer\s+beam)\b', 'subcategory': 'beam'},
                
                # Openings
                {'pattern': r'\b(?:door|entrance|exit|doorway|portal)\b', 'subcategory': 'door'},
                {'pattern': r'\b(?:window|glazing|fenestration|skylight)\b', 'subcategory': 'window'},
                {'pattern': r'\b(?:opening|aperture|void|shaft)\b', 'subcategory': 'opening'},
                
                # Stairs and vertical circulation
                {'pattern': r'\b(?:stair|staircase|stairway|step)\b', 'subcategory': 'stair'},
                {'pattern': r'\b(?:elevator|lift|escalator)\b', 'subcategory': 'vertical_transport'},
                
                # Roof elements
                {'pattern': r'\b(?:roof|roofing|ceiling)\b', 'subcategory': 'roof'},
                {'pattern': r'\b(?:parapet|eave|ridge|gutter)\b', 'subcategory': 'roof_element'},
                
                # Foundation elements
                {'pattern': r'\b(?:foundation|footing|basement|crawl\s+space)\b', 'subcategory': 'foundation'}
            ],
            
            'SPATIAL_ELEMENT': [
                # Rooms and spaces
                {'pattern': r'\b(?:room|space|area|zone|chamber)\b', 'subcategory': 'space'},
                {'pattern': r'\b(?:corridor|hallway|passage|aisle)\b', 'subcategory': 'circulation'},
                {'pattern': r'\b(?:lobby|foyer|entrance\s+hall|vestibule)\b', 'subcategory': 'entrance_space'},
                
                # Building levels
                {'pattern': r'\b(?:floor|level|storey|story|basement|attic)\b', 'subcategory': 'level'},
                {'pattern': r'\b(?:ground\s+floor|first\s+floor|upper\s+floor|mezzanine)\b', 'subcategory': 'level'},
                
                # Functional spaces
                {'pattern': r'\b(?:office|classroom|laboratory|workshop)\b', 'subcategory': 'functional_space'},
                {'pattern': r'\b(?:bathroom|restroom|toilet|washroom)\b', 'subcategory': 'service_space'},
                {'pattern': r'\b(?:kitchen|cafeteria|dining\s+room)\b', 'subcategory': 'food_service'},
                
                # Shafts and voids
                {'pattern': r'\b(?:shaft|duct|chase|void|atrium)\b', 'subcategory': 'shaft'},
                {'pattern': r'\b(?:elevator\s+shaft|stair\s+shaft|mechanical\s+shaft)\b', 'subcategory': 'shaft'}
            ],
            
            'MATERIAL': [
                # Structural materials
                {'pattern': r'\b(?:concrete|reinforced\s+concrete|precast\s+concrete)\b', 'subcategory': 'concrete'},
                {'pattern': r'\b(?:steel|structural\s+steel|carbon\s+steel|stainless\s+steel)\b', 'subcategory': 'steel'},
                {'pattern': r'\b(?:wood|timber|lumber|plywood)\b', 'subcategory': 'wood'},
                
                # Masonry materials
                {'pattern': r'\b(?:brick|masonry|stone|block)\b', 'subcategory': 'masonry'},
                {'pattern': r'\b(?:concrete\s+block|cinder\s+block|clay\s+brick)\b', 'subcategory': 'masonry'},
                
                # Composite materials
                {'pattern': r'\b(?:composite|fiber\s+reinforced|laminated)\b', 'subcategory': 'composite'},
                
                # Finishing materials
                {'pattern': r'\b(?:drywall|gypsum|plaster|stucco)\b', 'subcategory': 'finish'},
                {'pattern': r'\b(?:tile|ceramic|marble|granite)\b', 'subcategory': 'finish'}
            ],
            
            'DIMENSION': [
                # Linear dimensions
                {'pattern': r'\d+(?:\.\d+)?\s*(?:mm|millimeter|cm|centimeter|m|meter|ft|foot|feet|in|inch|inches)\b', 'subcategory': 'length'},
                
                # Area dimensions
                {'pattern': r'\d+(?:\.\d+)?\s*(?:sq\s*m|m²|sq\s*ft|ft²)\b', 'subcategory': 'area'},
                
                # Volume dimensions
                {'pattern': r'\d+(?:\.\d+)?\s*(?:cu\s*m|m³|cu\s*ft|ft³)\b', 'subcategory': 'volume'},
                
                # Thickness and width
                {'pattern': r'\b(?:thickness|width|height|depth|diameter)\s*:?\s*\d+(?:\.\d+)?\s*(?:mm|cm|m|ft|in)\b', 'subcategory': 'dimension_spec'}
            ],
            
            'REGULATORY_TERM': [
                # Code references
                {'pattern': r'\b(?:code|standard|regulation|requirement|specification)\b', 'subcategory': 'code_reference'},
                {'pattern': r'\b(?:IBC|UBC|NBC|ASCE|ACI|AISC)\b', 'subcategory': 'code_standard'},
                
                # Compliance terms
                {'pattern': r'\b(?:shall|must|required|mandatory|prohibited)\b', 'subcategory': 'compliance'},
                {'pattern': r'\b(?:minimum|maximum|not\s+less\s+than|not\s+more\s+than)\b', 'subcategory': 'limit'},
                
                # Safety and performance
                {'pattern': r'\b(?:fire\s+resistance|fire\s+rating|flame\s+spread)\b', 'subcategory': 'fire_safety'},
                {'pattern': r'\b(?:structural\s+integrity|load\s+bearing|seismic)\b', 'subcategory': 'structural'},
                {'pattern': r'\b(?:accessibility|ADA|barrier[\-\s]free)\b', 'subcategory': 'accessibility'}
            ]
        }
        
        return patterns
    
    def _load_relation_patterns(self) -> List[Dict[str, Any]]:
        """Load relation extraction patterns.
        
        Returns:
            List of relation patterns
        """
        patterns = [
            # Spatial relationships
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:located|situated|positioned)\s+(?:in|within|inside)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'SPATIAL',
                'predicate': 'LOCATED_IN'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:adjacent|next)\s+to\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'SPATIAL',
                'predicate': 'ADJACENT_TO'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:above|over|on\s+top\s+of)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'SPATIAL',
                'predicate': 'ABOVE'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:below|under|beneath)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'SPATIAL',
                'predicate': 'BELOW'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:contains|encloses|houses)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'SPATIAL',
                'predicate': 'CONTAINS'
            },
            
            # Functional relationships
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:supports|bears|carries)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'FUNCTIONAL',
                'predicate': 'SUPPORTS'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:connects|links|joins)\s+(?:to\s+)?(\w+(?:\s+\w+)*)',
                'relation_type': 'FUNCTIONAL',
                'predicate': 'CONNECTS_TO'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:separates|divides)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'FUNCTIONAL',
                'predicate': 'SEPARATES'
            },
            
            # Material relationships
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:made\s+of|constructed\s+of|built\s+with)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'MATERIAL',
                'predicate': 'MADE_OF'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:coated\s+with|covered\s+with|finished\s+with)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'MATERIAL',
                'predicate': 'COATED_WITH'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:reinforced\s+with|strengthened\s+with)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'MATERIAL',
                'predicate': 'REINFORCED_WITH'
            },
            
            # Regulatory relationships
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:shall|must)\s+(?:comply\s+with|meet|satisfy)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'REGULATORY',
                'predicate': 'COMPLIES_WITH'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:required\s+to|shall)\s+(?:have|provide|maintain)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'REGULATORY',
                'predicate': 'REQUIRES'
            },
            {
                'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:prohibited|not\s+allowed|forbidden)\s+(?:in|from)\s+(\w+(?:\s+\w+)*)',
                'relation_type': 'REGULATORY',
                'predicate': 'PROHIBITED_IN'
            }
        ]
        
        return patterns
    
    def _load_ifc_mappings(self) -> Dict[str, List[str]]:
        """Load IFC entity mappings to regulatory terms.
        
        Returns:
            Dictionary mapping IFC entities to regulatory terms
        """
        mappings = {
            'IfcWall': ['wall', 'partition', 'load-bearing wall', 'curtain wall', 'shear wall', 'exterior wall', 'interior wall'],
            'IfcSlab': ['slab', 'floor', 'platform', 'deck', 'terrace', 'balcony', 'roof slab'],
            'IfcColumn': ['column', 'pillar', 'post', 'support', 'pier'],
            'IfcBeam': ['beam', 'girder', 'joist', 'lintel', 'transfer beam'],
            'IfcDoor': ['door', 'entrance', 'exit', 'doorway', 'portal'],
            'IfcWindow': ['window', 'glazing', 'fenestration', 'skylight'],
            'IfcStair': ['stair', 'staircase', 'stairway', 'step'],
            'IfcRoof': ['roof', 'roofing', 'ceiling'],
            'IfcSpace': ['room', 'space', 'area', 'zone', 'chamber'],
            'IfcBuildingStorey': ['floor', 'level', 'storey', 'story'],
            'IfcOpeningElement': ['opening', 'aperture', 'void', 'shaft', 'atrium'],
            'IfcFoundation': ['foundation', 'footing', 'basement'],
            'IfcCurtainWall': ['curtain wall', 'glazed wall', 'facade'],
            'IfcRailing': ['railing', 'handrail', 'guardrail', 'balustrade'],
            'IfcStairFlight': ['stair flight', 'flight of stairs'],
            'IfcRamp': ['ramp', 'inclined walkway'],
            'IfcCovering': ['covering', 'finish', 'cladding', 'flooring'],
            'IfcFurnishingElement': ['furniture', 'furnishing', 'fixture']
        }
        
        return mappings
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        logger.info("Extracting entities from text")
        
        entities = []
        
        # Use spaCy if available
        if self.nlp is not None:
            entities.extend(self._extract_entities_spacy(text))
        
        # Use regex patterns
        entities.extend(self._extract_entities_regex(text))
        
        # Remove duplicates and merge overlapping entities
        entities = self._merge_entities(entities)
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    
    def extract_relations(self, text: str, entities: Optional[List[Entity]] = None) -> List[Relation]:
        """Extract relations from text.
        
        Args:
            text: Input text
            entities: Optional pre-extracted entities
            
        Returns:
            List of extracted relations
        """
        logger.info("Extracting relations from text")
        
        if entities is None:
            entities = self.extract_entities(text)
        
        relations = []
        
        # Extract relations using patterns
        relations.extend(self._extract_relations_regex(text, entities))
        
        # Extract dependency-based relations if spaCy is available
        if self.nlp is not None:
            relations.extend(self._extract_relations_dependency(text, entities))
        
        logger.info(f"Extracted {len(relations)} relations")
        return relations
    
    def _extract_entities_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            List of entities
        """
        entities = []
        
        try:
            doc = self.nlp(text)
            
            # Extract standard NER entities
            for ent in doc.ents:
                # Map spaCy labels to our categories
                label_mapping = {
                    'ORG': 'ORGANIZATION',
                    'PERSON': 'PERSON',
                    'GPE': 'LOCATION',
                    'QUANTITY': 'DIMENSION',
                    'CARDINAL': 'NUMBER'
                }
                
                mapped_label = label_mapping.get(ent.label_, ent.label_)
                
                entity = Entity(
                    text=ent.text,
                    label=mapped_label,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8,  # Default confidence for spaCy entities
                    attributes={'spacy_label': ent.label_},
                    context=text[max(0, ent.start_char-50):ent.end_char+50]
                )
                entities.append(entity)
            
            # Use matcher for custom patterns
            if self.matcher is not None:
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    span = doc[start:end]
                    entity = Entity(
                        text=span.text,
                        label='CUSTOM',
                        start=span.start_char,
                        end=span.end_char,
                        confidence=0.7,
                        attributes={'matcher_id': match_id},
                        context=text[max(0, span.start_char-50):span.end_char+50]
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.warning(f"Error in spaCy entity extraction: {e}")
        
        return entities
    
    def _extract_entities_regex(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of entities
        """
        entities = []
        
        for category, patterns in self.entity_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                subcategory = pattern_info.get('subcategory', 'general')
                
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        label=category,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.6,  # Default confidence for regex matches
                        attributes={'subcategory': subcategory, 'pattern': pattern},
                        context=text[max(0, match.start()-50):match.end()+50]
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_relations_regex(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using regex patterns.
        
        Args:
            text: Input text
            entities: List of entities
            
        Returns:
            List of relations
        """
        relations = []
        
        for pattern_info in self.relation_patterns:
            pattern = pattern_info['pattern']
            relation_type = pattern_info['relation_type']
            predicate = pattern_info['predicate']
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject_text = match.group(1).strip()
                object_text = match.group(2).strip()
                
                # Find corresponding entities
                subject_entity = self._find_entity_by_text(subject_text, entities)
                object_entity = self._find_entity_by_text(object_text, entities)
                
                if subject_entity and object_entity:
                    relation = Relation(
                        subject=subject_entity,
                        predicate=predicate,
                        object=object_entity,
                        confidence=0.6,
                        relation_type=relation_type,
                        context=text[max(0, match.start()-100):match.end()+100],
                        attributes={'pattern': pattern}
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_relations_dependency(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using dependency parsing.
        
        Args:
            text: Input text
            entities: List of entities
            
        Returns:
            List of relations
        """
        relations = []
        
        try:
            doc = self.nlp(text)
            
            # Extract relations based on dependency patterns
            for token in doc:
                # Look for specific dependency patterns
                if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                    head = token.head
                    
                    # Find entities corresponding to tokens
                    subject_entity = self._find_entity_by_position(token.idx, token.idx + len(token.text), entities)
                    object_entity = self._find_entity_by_position(head.idx, head.idx + len(head.text), entities)
                    
                    if subject_entity and object_entity and subject_entity != object_entity:
                        relation = Relation(
                            subject=subject_entity,
                            predicate=head.lemma_.upper(),
                            object=object_entity,
                            confidence=0.5,
                            relation_type='DEPENDENCY',
                            context=text[max(0, min(token.idx, head.idx)-50):max(token.idx + len(token.text), head.idx + len(head.text))+50],
                            attributes={'dependency': token.dep_, 'head_pos': head.pos_}
                        )
                        relations.append(relation)
                        
        except Exception as e:
            logger.warning(f"Error in dependency relation extraction: {e}")
        
        return relations
    
    def _find_entity_by_text(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """Find entity by text match.
        
        Args:
            text: Text to search for
            entities: List of entities
            
        Returns:
            Matching entity or None
        """
        text_lower = text.lower().strip()
        
        for entity in entities:
            if entity.text.lower().strip() == text_lower:
                return entity
            
            # Check if text is contained in entity text
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity
        
        return None
    
    def _find_entity_by_position(self, start: int, end: int, entities: List[Entity]) -> Optional[Entity]:
        """Find entity by position overlap.
        
        Args:
            start: Start position
            end: End position
            entities: List of entities
            
        Returns:
            Overlapping entity or None
        """
        for entity in entities:
            # Check for overlap
            if (start <= entity.start < end) or (start < entity.end <= end) or (entity.start <= start < entity.end):
                return entity
        
        return None
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities and remove duplicates.
        
        Args:
            entities: List of entities
            
        Returns:
            Merged list of entities
        """
        if not entities:
            return entities
        
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda e: (e.start, e.end))
        merged = []
        
        for entity in sorted_entities:
            # Check if this entity overlaps with the last merged entity
            if merged and self._entities_overlap(merged[-1], entity):
                # Merge entities - keep the one with higher confidence
                if entity.confidence > merged[-1].confidence:
                    merged[-1] = entity
            else:
                merged.append(entity)
        
        return merged
    
    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities overlap.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities overlap
        """
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)
    
    def map_to_ifc_entities(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """Map extracted entities to IFC entity types.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Dictionary mapping IFC entities to regulatory entities
        """
        ifc_mappings = defaultdict(list)
        
        for entity in entities:
            entity_text_lower = entity.text.lower()
            
            # Find matching IFC entities
            for ifc_entity, regulatory_terms in self.ifc_mappings.items():
                for term in regulatory_terms:
                    if term.lower() in entity_text_lower or entity_text_lower in term.lower():
                        ifc_mappings[ifc_entity].append(entity)
                        break
        
        return dict(ifc_mappings)
    
    def get_entity_statistics(self, entities: List[Entity]) -> Dict[str, Any]:
        """Get statistics about extracted entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_entities': len(entities),
            'by_category': defaultdict(int),
            'by_subcategory': defaultdict(int),
            'average_confidence': 0.0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        if not entities:
            return stats
        
        total_confidence = 0
        for entity in entities:
            stats['by_category'][entity.label] += 1
            
            subcategory = entity.attributes.get('subcategory', 'unknown')
            stats['by_subcategory'][subcategory] += 1
            
            total_confidence += entity.confidence
            
            # Confidence distribution
            if entity.confidence >= 0.8:
                stats['confidence_distribution']['high'] += 1
            elif entity.confidence >= 0.6:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        stats['average_confidence'] = total_confidence / len(entities)
        
        return dict(stats)
    
    def get_relation_statistics(self, relations: List[Relation]) -> Dict[str, Any]:
        """Get statistics about extracted relations.
        
        Args:
            relations: List of relations
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_relations': len(relations),
            'by_type': defaultdict(int),
            'by_predicate': defaultdict(int),
            'average_confidence': 0.0
        }
        
        if not relations:
            return stats
        
        total_confidence = 0
        for relation in relations:
            stats['by_type'][relation.relation_type] += 1
            stats['by_predicate'][relation.predicate] += 1
            total_confidence += relation.confidence
        
        stats['average_confidence'] = total_confidence / len(relations)
        
        return dict(stats)