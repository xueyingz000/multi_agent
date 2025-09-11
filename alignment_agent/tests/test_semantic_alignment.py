#!/usr/bin/env python3
"""Test cases for semantic alignment functionality."""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.semantic_alignment import SemanticAlignment, AlignmentResult, SemanticMapping
from core.ifc_semantic_agent import IFCSemanticAgent, ActionType
from llm.semantic_extractor import SemanticExtractor
from llm.ner_relation_extractor import NERRelationExtractor, Entity, Relation


class TestSemanticAlignment(unittest.TestCase):
    """Test cases for SemanticAlignment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.alignment = SemanticAlignment()
        
        # Mock IFC entity
        self.mock_ifc_entity = {
            "id": "wall_001",
            "type": "IfcWall",
            "name": "ExteriorWall",
            "properties": {
                "Width": 200,
                "Height": 3000,
                "Material": "Concrete",
                "FireResistance": 120
            }
        }
        
        # Mock regulatory entity
        self.mock_regulatory_entity = Entity(
            text="承重墙",
            label="STRUCTURAL_ELEMENT",
            start_pos=0,
            end_pos=3,
            confidence=0.9,
            properties={"thickness_requirement": "180mm", "fire_resistance": "120min"}
        )
    
    def test_initialization(self):
        """Test SemanticAlignment initialization."""
        self.assertIsNotNone(self.alignment.tfidf_vectorizer)
        self.assertIsInstance(self.alignment.predefined_mappings, dict)
        self.assertIsInstance(self.alignment.disambiguation_rules, dict)
    
    def test_semantic_similarity_calculation(self):
        """Test semantic similarity calculation."""
        text1 = "exterior wall"
        text2 = "external wall"
        
        similarity = self.alignment._calculate_semantic_similarity(text1, text2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        self.assertGreater(similarity, 0.5)  # Should be relatively high for similar terms
    
    def test_lexical_similarity_calculation(self):
        """Test lexical similarity calculation."""
        text1 = "IfcWall"
        text2 = "Wall"
        
        similarity = self.alignment._calculate_lexical_similarity(text1, text2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_value_similarity_calculation(self):
        """Test value similarity calculation."""
        # Test numeric values
        similarity1 = self.alignment._calculate_value_similarity(200, 180)
        self.assertGreater(similarity1, 0.8)  # Should be high for close values
        
        # Test string values
        similarity2 = self.alignment._calculate_value_similarity("Concrete", "concrete")
        self.assertGreater(similarity2, 0.9)  # Should be high for case-insensitive match
        
        # Test different types
        similarity3 = self.alignment._calculate_value_similarity(200, "200mm")
        self.assertGreater(similarity3, 0.7)  # Should handle unit conversion
    
    def test_context_similarity_calculation(self):
        """Test context similarity calculation."""
        context1 = "building structural element load bearing"
        context2 = "construction wall support structure"
        
        similarity = self.alignment._calculate_context_similarity(context1, context2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_entity_alignment(self):
        """Test entity alignment functionality."""
        result = self.alignment.align_entities(
            self.mock_ifc_entity,
            self.mock_regulatory_entity
        )
        
        self.assertIsInstance(result, AlignmentResult)
        self.assertIsInstance(result.similarity_score, float)
        self.assertGreaterEqual(result.similarity_score, 0.0)
        self.assertLessEqual(result.similarity_score, 1.0)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.alignment_type, str)
    
    def test_attribute_alignment(self):
        """Test attribute alignment functionality."""
        ifc_attributes = self.mock_ifc_entity["properties"]
        regulatory_attributes = self.mock_regulatory_entity.properties
        
        results = self.alignment.align_attributes(ifc_attributes, regulatory_attributes)
        
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, AlignmentResult)
    
    def test_relationship_alignment(self):
        """Test relationship alignment functionality."""
        ifc_relation = {
            "type": "IfcRelContainedInSpatialStructure",
            "relating_structure": "building_floor",
            "related_elements": ["wall_001"]
        }
        
        regulatory_relation = Relation(
            subject="承重墙",
            predicate="位于",
            object="建筑楼层",
            confidence=0.8,
            relation_type="SPATIAL"
        )
        
        result = self.alignment.align_relationships(ifc_relation, regulatory_relation)
        
        self.assertIsInstance(result, AlignmentResult)
        self.assertIsInstance(result.similarity_score, float)
    
    def test_semantic_mapping_creation(self):
        """Test semantic mapping creation."""
        alignment_results = [
            AlignmentResult(
                ifc_element="IfcWall",
                regulatory_element="承重墙",
                similarity_score=0.85,
                confidence=0.9,
                alignment_type="ENTITY",
                reasoning="High semantic similarity between wall concepts"
            )
        ]
        
        mapping = self.alignment.create_semantic_mapping(alignment_results)
        
        self.assertIsInstance(mapping, SemanticMapping)
        self.assertEqual(len(mapping.entity_mappings), 1)
        self.assertGreater(mapping.overall_confidence, 0.0)
    
    def test_alignment_validation(self):
        """Test alignment validation."""
        mapping = SemanticMapping(
            entity_mappings={"IfcWall": "承重墙"},
            attribute_mappings={"Width": "厚度"},
            relationship_mappings={"contains": "包含"},
            overall_confidence=0.85,
            validation_status="pending"
        )
        
        is_valid = self.alignment.validate_alignment(mapping)
        
        self.assertIsInstance(is_valid, bool)
    
    def test_mapping_export_json(self):
        """Test mapping export to JSON format."""
        mapping = SemanticMapping(
            entity_mappings={"IfcWall": "承重墙"},
            attribute_mappings={"Width": "厚度"},
            relationship_mappings={"contains": "包含"},
            overall_confidence=0.85,
            validation_status="validated"
        )
        
        json_output = self.alignment.export_mapping(mapping, format="json")
        
        self.assertIsInstance(json_output, str)
        self.assertIn("entity_mappings", json_output)
        self.assertIn("IfcWall", json_output)


class TestNERRelationExtractor(unittest.TestCase):
    """Test cases for NERRelationExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = NERRelationExtractor()
        self.sample_text = """
        建筑设计规范要求承重墙厚度不应小于180mm，
        楼板厚度应根据跨度确定，一般不小于120mm。
        外墙应具备保温、防水功能。
        """
    
    @patch('spacy.load')
    def test_initialization_with_spacy(self, mock_spacy_load):
        """Test initialization with spaCy model."""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        extractor = NERRelationExtractor(use_spacy=True)
        
        self.assertEqual(extractor.nlp, mock_nlp)
        mock_spacy_load.assert_called_once()
    
    def test_regex_pattern_loading(self):
        """Test regex pattern loading."""
        patterns = self.extractor._load_regex_patterns()
        
        self.assertIsInstance(patterns, dict)
        self.assertIn('building_components', patterns)
        self.assertIn('materials', patterns)
        self.assertIn('dimensions', patterns)
    
    def test_entity_extraction_regex(self):
        """Test entity extraction using regex."""
        entities = self.extractor._extract_entities_regex(self.sample_text)
        
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        for entity in entities:
            self.assertIsInstance(entity, Entity)
            self.assertIsInstance(entity.text, str)
            self.assertIsInstance(entity.label, str)
    
    def test_relation_extraction_regex(self):
        """Test relation extraction using regex."""
        entities = self.extractor._extract_entities_regex(self.sample_text)
        relations = self.extractor._extract_relations_regex(self.sample_text, entities)
        
        self.assertIsInstance(relations, list)
        
        for relation in relations:
            self.assertIsInstance(relation, Relation)
            self.assertIsInstance(relation.subject, str)
            self.assertIsInstance(relation.predicate, str)
            self.assertIsInstance(relation.object, str)
    
    def test_entity_merging(self):
        """Test entity merging functionality."""
        entities = [
            Entity("承重墙", "STRUCTURAL", 0, 3, 0.9),
            Entity("墙", "STRUCTURAL", 2, 3, 0.7),  # Overlapping
            Entity("楼板", "STRUCTURAL", 10, 12, 0.8)
        ]
        
        merged = self.extractor._merge_entities(entities)
        
        self.assertLessEqual(len(merged), len(entities))
        
        for entity in merged:
            self.assertIsInstance(entity, Entity)
    
    def test_ifc_mapping(self):
        """Test IFC entity mapping."""
        entities = [
            Entity("墙", "STRUCTURAL", 0, 1, 0.9),
            Entity("楼板", "STRUCTURAL", 5, 7, 0.8)
        ]
        
        mapped = self.extractor._map_to_ifc_types(entities)
        
        self.assertIsInstance(mapped, list)
        
        for entity in mapped:
            self.assertIsInstance(entity, Entity)
            if hasattr(entity, 'ifc_type'):
                self.assertIsInstance(entity.ifc_type, str)
    
    def test_full_extraction_pipeline(self):
        """Test the complete extraction pipeline."""
        entities, relations = self.extractor.extract_entities_and_relations(self.sample_text)
        
        self.assertIsInstance(entities, list)
        self.assertIsInstance(relations, list)
        
        # Check that we got some results
        self.assertGreater(len(entities), 0)
        
        # Verify entity structure
        for entity in entities:
            self.assertIsInstance(entity, Entity)
            self.assertIsNotNone(entity.text)
            self.assertIsNotNone(entity.label)
        
        # Verify relation structure
        for relation in relations:
            self.assertIsInstance(relation, Relation)
            self.assertIsNotNone(relation.subject)
            self.assertIsNotNone(relation.predicate)
            self.assertIsNotNone(relation.object)
    
    def test_statistics_generation(self):
        """Test statistics generation."""
        entities, relations = self.extractor.extract_entities_and_relations(self.sample_text)
        stats = self.extractor.get_extraction_statistics(entities, relations)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_entities', stats)
        self.assertIn('total_relations', stats)
        self.assertIn('entity_types', stats)
        self.assertIn('relation_types', stats)
        
        self.assertEqual(stats['total_entities'], len(entities))
        self.assertEqual(stats['total_relations'], len(relations))


class TestIFCSemanticAgent(unittest.TestCase):
    """Test cases for IFCSemanticAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration
        self.mock_config = {
            'react_agent': {
                'max_steps': 10,
                'confidence_threshold': 0.7
            },
            'llm': {
                'provider': 'mock',
                'model': 'test-model'
            }
        }
        
        # Create agent with mocked dependencies
        with patch('core.ifc_semantic_agent.load_config') as mock_load_config:
            mock_load_config.return_value = self.mock_config
            with patch.multiple(
                'core.ifc_semantic_agent',
                IFCProcessor=Mock(),
                TextProcessor=Mock(),
                SemanticExtractor=Mock(),
                NERRelationExtractor=Mock(),
                GraphBuilder=Mock(),
                RAGSystem=Mock(),
                EntityResolver=Mock(),
                TextStructuredDataFusion=Mock(),
                SemanticAlignment=Mock()
            ):
                self.agent = IFCSemanticAgent("mock_config.yaml")
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent.config)
        self.assertEqual(self.agent.max_steps, 10)
        self.assertEqual(self.agent.confidence_threshold, 0.7)
        self.assertIsInstance(self.agent.available_actions, list)
    
    def test_action_type_enum(self):
        """Test ActionType enum."""
        self.assertEqual(ActionType.ANALYZE_IFC.value, "analyze_ifc")
        self.assertEqual(ActionType.PROCESS_TEXT.value, "process_text")
        self.assertEqual(ActionType.EXTRACT_SEMANTICS.value, "extract_semantics")
    
    def test_query_processing_structure(self):
        """Test query processing structure."""
        # Mock the process_query method to return a structured response
        with patch.object(self.agent, '_execute_react_loop') as mock_react:
            mock_react.return_value = {
                'final_answer': 'Test answer',
                'confidence_score': 0.85,
                'total_steps': 3,
                'react_steps': [],
                'execution_time': 1.5
            }
            
            response = self.agent.process_query(
                query="Test query",
                ifc_data=None,
                regulatory_text=None
            )
            
            self.assertIsNotNone(response)
            self.assertEqual(response.final_answer, 'Test answer')
            self.assertEqual(response.confidence_score, 0.85)
            self.assertEqual(response.total_steps, 3)
    
    def test_agent_state_management(self):
        """Test agent state management."""
        state = self.agent.get_agent_state()
        
        self.assertIsInstance(state, dict)
        self.assertIn('current_step', state)
        self.assertIn('react_history_length', state)
        self.assertIn('overall_confidence', state)
        self.assertIn('available_actions', state)
    
    def test_agent_reset(self):
        """Test agent reset functionality."""
        # Modify agent state
        self.agent.current_step = 5
        self.agent.react_history = [Mock(), Mock()]
        
        # Reset agent
        self.agent.reset_agent()
        
        # Verify reset
        self.assertEqual(self.agent.current_step, 0)
        self.assertEqual(len(self.agent.react_history), 0)


class TestSemanticExtractor(unittest.TestCase):
    """Test cases for SemanticExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = SemanticExtractor()
        self.sample_text = """
        建筑设计防火规范要求：承重墙厚度不应小于180mm，
        耐火极限不应低于3小时。楼板厚度应根据跨度确定。
        """
    
    def test_initialization(self):
        """Test SemanticExtractor initialization."""
        self.assertIsNotNone(self.extractor)
        # Test fallback initialization when LLM is not available
        self.assertIsNotNone(self.extractor.fallback_patterns)
    
    def test_key_concept_identification(self):
        """Test key concept identification."""
        concepts = self.extractor.identify_key_concepts(self.sample_text)
        
        self.assertIsInstance(concepts, list)
        self.assertGreater(len(concepts), 0)
        
        for concept in concepts:
            self.assertIsInstance(concept, dict)
            self.assertIn('concept', concept)
            self.assertIn('importance', concept)
    
    def test_regulatory_requirement_extraction(self):
        """Test regulatory requirement extraction."""
        requirements = self.extractor.extract_regulatory_requirements(self.sample_text)
        
        self.assertIsInstance(requirements, list)
        
        for requirement in requirements:
            self.assertIsInstance(requirement, dict)
            self.assertIn('requirement', requirement)
            self.assertIn('type', requirement)
    
    def test_semantic_mapping_generation(self):
        """Test semantic mapping generation."""
        ifc_elements = ["IfcWall", "IfcSlab"]
        
        mappings = self.extractor.generate_semantic_mappings(
            self.sample_text, 
            ifc_elements
        )
        
        self.assertIsInstance(mappings, list)
        
        for mapping in mappings:
            self.assertIsInstance(mapping, dict)
            self.assertIn('ifc_element', mapping)
            self.assertIn('regulatory_concept', mapping)
            self.assertIn('confidence', mapping)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSemanticAlignment))
    test_suite.addTest(unittest.makeSuite(TestNERRelationExtractor))
    test_suite.addTest(unittest.makeSuite(TestIFCSemanticAgent))
    test_suite.addTest(unittest.makeSuite(TestSemanticExtractor))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"测试总结")
    print(f"{'='*60}")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}")