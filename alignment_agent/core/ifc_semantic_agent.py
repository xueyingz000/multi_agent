"""IFC Semantic Agent using ReAct (Reasoning + Acting) framework."""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .fusion_module import TextStructuredDataFusion
from .semantic_alignment import SemanticAlignment
from knowledge_graph import GraphBuilder, RAGSystem, EntityResolver
from llm import SemanticExtractor, NERRelationExtractor
from data_processing import IFCProcessor, TextProcessor
from utils import ConfigLoader, get_logger


class ActionType(Enum):
    """Types of actions the agent can perform."""
    ANALYZE_IFC = "analyze_ifc"
    PROCESS_REGULATORY_TEXT = "process_regulatory_text"
    EXTRACT_SEMANTICS = "extract_semantics"
    ALIGN_ENTITIES = "align_entities"
    QUERY_KNOWLEDGE_GRAPH = "query_knowledge_graph"
    RESOLVE_ENTITY = "resolve_entity"
    GENERATE_MAPPING = "generate_mapping"
    VALIDATE_ALIGNMENT = "validate_alignment"
    REFLECT_ON_RESULTS = "reflect_on_results"


@dataclass
class Observation:
    """Represents an observation from the environment."""
    content: Any
    observation_type: str
    timestamp: float
    metadata: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class Thought:
    """Represents a reasoning step."""
    content: str
    reasoning_type: str  # 'analysis', 'planning', 'reflection', 'conclusion'
    confidence: float
    supporting_evidence: List[str]
    timestamp: float


@dataclass
class Action:
    """Represents an action to be performed."""
    action_type: ActionType
    parameters: Dict[str, Any]
    reasoning: str
    expected_outcome: str
    timestamp: float


@dataclass
class ReActStep:
    """Represents one complete ReAct step (Thought -> Action -> Observation)."""
    step_id: int
    thought: Thought
    action: Action
    observation: Observation
    success: bool
    error_message: Optional[str] = None


@dataclass
class AgentResponse:
    """Complete agent response with reasoning trace."""
    query: str
    final_answer: str
    confidence_score: float
    react_steps: List[ReActStep]
    total_steps: int
    execution_time: float
    knowledge_sources: List[str]
    semantic_mappings: List[Dict[str, Any]]


class IFCSemanticAgent:
    """ReAct-based agent for IFC-regulatory semantic understanding and alignment."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the IFC Semantic Agent.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)
        
        # Initialize core components
        self._initialize_components()
        
        # ReAct configuration
        self.max_iterations = self.config.get('react_agent.max_iterations', 10)
        self.reflection_threshold = self.config.get('react_agent.reflection_threshold', 0.3)
        self.confidence_threshold = self.config.get('react_agent.confidence_threshold', 0.7)
        
        # Agent state
        self.current_step = 0
        self.react_history: List[ReActStep] = []
        self.working_memory: Dict[str, Any] = {}
        
        # Available tools/actions
        self.available_actions = {
            ActionType.ANALYZE_IFC: self._analyze_ifc_data,
            ActionType.PROCESS_REGULATORY_TEXT: self._process_regulatory_text,
            ActionType.EXTRACT_SEMANTICS: self._extract_semantics,
            ActionType.ALIGN_ENTITIES: self._align_entities,
            ActionType.QUERY_KNOWLEDGE_GRAPH: self._query_knowledge_graph,
            ActionType.RESOLVE_ENTITY: self._resolve_entity,
            ActionType.GENERATE_MAPPING: self._generate_mapping,
            ActionType.VALIDATE_ALIGNMENT: self._validate_alignment,
            ActionType.REFLECT_ON_RESULTS: self._reflect_on_results
        }
        
    def _initialize_components(self):
        """Initialize all agent components."""
        try:
            # Core processing modules
            self.ifc_processor = IFCProcessor()
            self.text_processor = TextProcessor()
            
            # LLM-based semantic extraction
            self.semantic_extractor = SemanticExtractor()
            self.ner_extractor = NERRelationExtractor()
            
            # Knowledge graph and RAG system
            self.graph_builder = GraphBuilder()
            self.rag_system = RAGSystem(self.graph_builder)
            self.entity_resolver = EntityResolver(self.graph_builder)
            
            # Core fusion and alignment modules
            self.fusion_module = TextStructuredDataFusion()
            self.semantic_alignment = SemanticAlignment()
            
            self.logger.info("All agent components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
            
    def process_query(self, query: str, ifc_data: Optional[Dict[str, Any]] = None, regulatory_text: Optional[str] = None) -> AgentResponse:
        """Process a semantic alignment query using ReAct framework.
        
        Args:
            query: Natural language query about IFC-regulatory alignment
            ifc_data: Optional IFC data to analyze
            regulatory_text: Optional regulatory text to process
            
        Returns:
            Complete agent response with reasoning trace
        """
        start_time = time.time()
        
        # Reset agent state
        self.current_step = 0
        self.react_history = []
        self.working_memory = {
            'query': query,
            'ifc_data': ifc_data,
            'regulatory_text': regulatory_text,
            'extracted_entities': {},
            'alignments': [],
            'confidence_scores': []
        }
        
        self.logger.info(f"Starting ReAct processing for query: {query}")
        
        # Main ReAct loop
        final_answer = ""
        while self.current_step < self.max_iterations:
            try:
                # Reasoning step (Thought)
                thought = self._generate_thought()
                
                # Planning step (Action)
                action = self._plan_action(thought)
                
                # Execution step (Observation)
                observation = self._execute_action(action)
                
                # Create ReAct step
                react_step = ReActStep(
                    step_id=self.current_step,
                    thought=thought,
                    action=action,
                    observation=observation,
                    success=observation.confidence > 0.5
                )
                
                self.react_history.append(react_step)
                
                # Check if we have a satisfactory answer
                if self._should_terminate():
                    final_answer = self._generate_final_answer()
                    break
                    
                self.current_step += 1
                
            except Exception as e:
                self.logger.error(f"Error in ReAct step {self.current_step}: {e}")
                error_step = ReActStep(
                    step_id=self.current_step,
                    thought=Thought(
                        content=f"Error occurred: {str(e)}",
                        reasoning_type="error",
                        confidence=0.0,
                        supporting_evidence=[],
                        timestamp=time.time()
                    ),
                    action=Action(
                        action_type=ActionType.REFLECT_ON_RESULTS,
                        parameters={},
                        reasoning="Handling error",
                        expected_outcome="Error recovery",
                        timestamp=time.time()
                    ),
                    observation=Observation(
                        content={"error": str(e)},
                        observation_type="error",
                        timestamp=time.time(),
                        metadata={},
                        confidence=0.0
                    ),
                    success=False,
                    error_message=str(e)
                )
                self.react_history.append(error_step)
                break
                
        # Generate final response
        execution_time = time.time() - start_time
        confidence_score = self._calculate_overall_confidence()
        
        response = AgentResponse(
            query=query,
            final_answer=final_answer or self._generate_fallback_answer(),
            confidence_score=confidence_score,
            react_steps=self.react_history,
            total_steps=len(self.react_history),
            execution_time=execution_time,
            knowledge_sources=self._get_knowledge_sources(),
            semantic_mappings=self.working_memory.get('alignments', [])
        )
        
        self.logger.info(f"ReAct processing completed in {execution_time:.2f}s with {len(self.react_history)} steps")
        return response
        
    def _generate_thought(self) -> Thought:
        """Generate a reasoning thought based on current state."""
        # Analyze current situation
        if self.current_step == 0:
            # Initial analysis
            content = f"Starting analysis of query: '{self.working_memory['query']}'. "
            
            if self.working_memory.get('ifc_data'):
                content += "IFC data is available for analysis. "
            if self.working_memory.get('regulatory_text'):
                content += "Regulatory text is available for processing. "
                
            content += "Need to determine the best approach for semantic alignment."
            reasoning_type = "analysis"
            confidence = 0.9
            
        elif len(self.working_memory.get('extracted_entities', {})) == 0:
            # Need to extract entities
            content = "No entities have been extracted yet. Need to process available data to identify IFC entities and regulatory terms."
            reasoning_type = "planning"
            confidence = 0.8
            
        elif len(self.working_memory.get('alignments', [])) == 0:
            # Need to perform alignment
            content = "Entities have been extracted. Now need to perform semantic alignment between IFC entities and regulatory terms."
            reasoning_type = "planning"
            confidence = 0.8
            
        else:
            # Reflection or refinement
            avg_confidence = sum(self.working_memory.get('confidence_scores', [0])) / max(len(self.working_memory.get('confidence_scores', [1])), 1)
            
            if avg_confidence < self.reflection_threshold:
                content = f"Current alignment confidence ({avg_confidence:.2f}) is below threshold. Need to refine the analysis or gather more context."
                reasoning_type = "reflection"
                confidence = 0.6
            else:
                content = "Alignment results look satisfactory. Ready to generate final mapping and conclusions."
                reasoning_type = "conclusion"
                confidence = 0.9
                
        return Thought(
            content=content,
            reasoning_type=reasoning_type,
            confidence=confidence,
            supporting_evidence=self._get_supporting_evidence(),
            timestamp=time.time()
        )
        
    def _plan_action(self, thought: Thought) -> Action:
        """Plan the next action based on the current thought."""
        if thought.reasoning_type == "analysis" and self.current_step == 0:
            # Decide what to analyze first
            if self.working_memory.get('ifc_data'):
                action_type = ActionType.ANALYZE_IFC
                parameters = {'data': self.working_memory['ifc_data']}
                reasoning = "Starting with IFC data analysis to understand available entities"
                expected_outcome = "Extracted IFC entities with properties and relationships"
            elif self.working_memory.get('regulatory_text'):
                action_type = ActionType.PROCESS_REGULATORY_TEXT
                parameters = {'text': self.working_memory['regulatory_text']}
                reasoning = "Starting with regulatory text processing to extract terms"
                expected_outcome = "Extracted regulatory entities and relationships"
            else:
                action_type = ActionType.QUERY_KNOWLEDGE_GRAPH
                parameters = {'query': self.working_memory['query']}
                reasoning = "No direct data provided, querying existing knowledge graph"
                expected_outcome = "Relevant entities and relationships from knowledge graph"
                
        elif thought.reasoning_type == "planning":
            if len(self.working_memory.get('extracted_entities', {})) == 0:
                # Need to extract entities
                if self.working_memory.get('ifc_data') and 'ifc_entities' not in self.working_memory.get('extracted_entities', {}):
                    action_type = ActionType.ANALYZE_IFC
                    parameters = {'data': self.working_memory['ifc_data']}
                elif self.working_memory.get('regulatory_text') and 'regulatory_entities' not in self.working_memory.get('extracted_entities', {}):
                    action_type = ActionType.PROCESS_REGULATORY_TEXT
                    parameters = {'text': self.working_memory['regulatory_text']}
                else:
                    action_type = ActionType.EXTRACT_SEMANTICS
                    parameters = {'query': self.working_memory['query']}
                    
                reasoning = "Extracting entities from available data sources"
                expected_outcome = "Comprehensive entity extraction results"
                
            else:
                # Perform alignment
                action_type = ActionType.ALIGN_ENTITIES
                parameters = {
                    'ifc_entities': self.working_memory.get('extracted_entities', {}).get('ifc_entities', []),
                    'regulatory_entities': self.working_memory.get('extracted_entities', {}).get('regulatory_entities', [])
                }
                reasoning = "Performing semantic alignment between extracted entities"
                expected_outcome = "Semantic alignment mappings with confidence scores"
                
        elif thought.reasoning_type == "reflection":
            action_type = ActionType.QUERY_KNOWLEDGE_GRAPH
            parameters = {
                'query': self.working_memory['query'],
                'strategy': 'contextual_path',
                'top_k': 10
            }
            reasoning = "Gathering additional context from knowledge graph to improve alignment"
            expected_outcome = "Additional contextual information for better alignment"
            
        else:  # conclusion
            action_type = ActionType.GENERATE_MAPPING
            parameters = {
                'alignments': self.working_memory.get('alignments', []),
                'query': self.working_memory['query']
            }
            reasoning = "Generating final semantic mapping based on alignment results"
            expected_outcome = "Complete semantic mapping with explanations"
            
        return Action(
            action_type=action_type,
            parameters=parameters,
            reasoning=reasoning,
            expected_outcome=expected_outcome,
            timestamp=time.time()
        )
        
    def _execute_action(self, action: Action) -> Observation:
        """Execute the planned action and return observation."""
        try:
            if action.action_type in self.available_actions:
                result = self.available_actions[action.action_type](action.parameters)
                
                return Observation(
                    content=result,
                    observation_type=action.action_type.value,
                    timestamp=time.time(),
                    metadata={'action_parameters': action.parameters},
                    confidence=self._evaluate_result_confidence(result)
                )
            else:
                return Observation(
                    content={'error': f'Unknown action type: {action.action_type}'},
                    observation_type='error',
                    timestamp=time.time(),
                    metadata={},
                    confidence=0.0
                )
                
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            return Observation(
                content={'error': str(e)},
                observation_type='error',
                timestamp=time.time(),
                metadata={'action_type': action.action_type.value},
                confidence=0.0
            )
            
    def _analyze_ifc_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze IFC data and extract entities."""
        ifc_data = parameters.get('data')
        
        if isinstance(ifc_data, str):  # File path
            processed_data = self.ifc_processor.process_ifc_file(ifc_data)
        else:  # Already processed data
            processed_data = ifc_data
            
        # Add to knowledge graph
        node_ids = self.graph_builder.add_ifc_entities(processed_data)
        
        # Update working memory
        if 'extracted_entities' not in self.working_memory:
            self.working_memory['extracted_entities'] = {}
        self.working_memory['extracted_entities']['ifc_entities'] = processed_data.get('entities', {})
        
        return {
            'entities_count': len(processed_data.get('entities', {})),
            'relationships_count': len(processed_data.get('relationships', [])),
            'node_ids': node_ids,
            'entity_types': list(set([e.get('type') for e in processed_data.get('entities', {}).values()]))
        }
        
    def _process_regulatory_text(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process regulatory text and extract entities."""
        text = parameters.get('text')
        
        # Extract entities using NER
        entities = self.ner_extractor.extract_entities(text)
        relations = self.ner_extractor.extract_relations(text)
        
        # Extract semantic information using LLM
        semantic_info = self.semantic_extractor.extract_semantic_information(text)
        
        # Combine results
        regulatory_data = {
            'entities': [asdict(entity) for entity in entities],
            'relationships': [asdict(relation) for relation in relations],
            'semantic_info': semantic_info
        }
        
        # Add to knowledge graph
        node_ids = self.graph_builder.add_regulatory_entities(regulatory_data)
        
        # Update working memory
        if 'extracted_entities' not in self.working_memory:
            self.working_memory['extracted_entities'] = {}
        self.working_memory['extracted_entities']['regulatory_entities'] = entities
        
        return {
            'entities_count': len(entities),
            'relationships_count': len(relations),
            'node_ids': node_ids,
            'categories': list(set([e.category for e in entities if e.category]))
        }
        
    def _extract_semantics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic information from query or context."""
        query = parameters.get('query')
        
        # Use semantic extractor to understand the query
        semantic_info = self.semantic_extractor.extract_semantic_information(query)
        
        # Extract entities mentioned in the query
        query_entities = self.ner_extractor.extract_entities(query)
        
        return {
            'semantic_info': semantic_info,
            'query_entities': [asdict(entity) for entity in query_entities],
            'key_concepts': semantic_info.key_concepts,
            'requirements': semantic_info.regulatory_requirements
        }
        
    def _align_entities(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic alignment between IFC and regulatory entities."""
        ifc_entities = parameters.get('ifc_entities', [])
        regulatory_entities = parameters.get('regulatory_entities', [])
        
        # Perform alignment
        alignments = []
        
        for ifc_entity in ifc_entities:
            for reg_entity in regulatory_entities:
                alignment_result = self.semantic_alignment.align_entities(
                    ifc_entity, reg_entity
                )
                
                if alignment_result.confidence_score > 0.3:  # Threshold for considering alignment
                    alignments.append(asdict(alignment_result))
                    
        # Add alignments to knowledge graph
        if alignments:
            self.graph_builder.add_alignment_mappings(alignments)
            
        # Update working memory
        self.working_memory['alignments'] = alignments
        confidence_scores = [a['confidence_score'] for a in alignments]
        self.working_memory['confidence_scores'] = confidence_scores
        
        return {
            'alignments_count': len(alignments),
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            'high_confidence_count': len([c for c in confidence_scores if c > 0.7]),
            'alignments': alignments
        }
        
    def _query_knowledge_graph(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge graph for relevant information."""
        query = parameters.get('query')
        strategy = parameters.get('strategy', 'hybrid')
        top_k = parameters.get('top_k', 5)
        
        # Perform RAG query
        rag_response = self.rag_system.query(
            query=query,
            strategy=strategy,
            top_k=top_k,
            generate_response=True
        )
        
        return {
            'retrieved_nodes': [asdict(node) for node in rag_response.retrieved_nodes],
            'generated_response': rag_response.generated_response,
            'confidence_score': rag_response.confidence_score,
            'reasoning_path': rag_response.reasoning_path,
            'supporting_evidence': rag_response.supporting_evidence
        }
        
    def _resolve_entity(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve entity mentions to knowledge graph entities."""
        entity_text = parameters.get('entity_text')
        context = parameters.get('context', '')
        
        resolution_result = self.entity_resolver.resolve_entity(
            entity_text=entity_text,
            context=context
        )
        
        return asdict(resolution_result)
        
    def _generate_mapping(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final semantic mapping."""
        alignments = parameters.get('alignments', [])
        query = parameters.get('query')
        
        # Create comprehensive mapping
        mapping = self.semantic_alignment.create_semantic_mapping(
            alignments, context=query
        )
        
        return {
            'semantic_mapping': asdict(mapping),
            'total_mappings': len(mapping.entity_mappings),
            'high_confidence_mappings': len([m for m in mapping.entity_mappings if m.confidence_score > 0.7])
        }
        
    def _validate_alignment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate alignment results."""
        alignments = parameters.get('alignments', self.working_memory.get('alignments', []))
        
        validation_results = []
        for alignment in alignments:
            is_valid = self.semantic_alignment.validate_alignment(alignment)
            validation_results.append({
                'alignment_id': alignment.get('id', 'unknown'),
                'is_valid': is_valid,
                'confidence': alignment.get('confidence_score', 0.0)
            })
            
        valid_count = len([r for r in validation_results if r['is_valid']])
        
        return {
            'total_alignments': len(validation_results),
            'valid_alignments': valid_count,
            'validation_rate': valid_count / len(validation_results) if validation_results else 0.0,
            'results': validation_results
        }
        
    def _reflect_on_results(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on current results and suggest improvements."""
        current_confidence = self._calculate_overall_confidence()
        
        reflections = []
        
        if current_confidence < 0.5:
            reflections.append("Low overall confidence suggests need for more context or different approach")
            
        if len(self.working_memory.get('alignments', [])) == 0:
            reflections.append("No alignments found - may need to broaden search criteria")
            
        if len(self.working_memory.get('extracted_entities', {})) < 2:
            reflections.append("Limited entity extraction - may need additional data sources")
            
        return {
            'current_confidence': current_confidence,
            'reflections': reflections,
            'suggested_actions': self._suggest_next_actions()
        }
        
    def _should_terminate(self) -> bool:
        """Determine if the ReAct loop should terminate."""
        # Check if we have a satisfactory answer
        overall_confidence = self._calculate_overall_confidence()
        
        if overall_confidence >= self.confidence_threshold:
            return True
            
        # Check if we have alignments with reasonable confidence
        alignments = self.working_memory.get('alignments', [])
        if alignments:
            high_conf_alignments = [a for a in alignments if a.get('confidence_score', 0) > 0.6]
            if len(high_conf_alignments) >= 3:  # At least 3 good alignments
                return True
                
        # Check if we've made progress in recent steps
        if len(self.react_history) >= 3:
            recent_steps = self.react_history[-3:]
            if all(step.observation.confidence < 0.3 for step in recent_steps):
                return True  # No progress, terminate
                
        return False
        
    def _generate_final_answer(self) -> str:
        """Generate the final answer based on all collected information."""
        query = self.working_memory['query']
        alignments = self.working_memory.get('alignments', [])
        
        if not alignments:
            return f"Unable to find semantic alignments for the query: '{query}'. No matching entities were identified between IFC and regulatory domains."
            
        # Summarize findings
        answer_parts = []
        answer_parts.append(f"Analysis of '{query}' reveals the following semantic alignments:")
        
        # High confidence alignments
        high_conf = [a for a in alignments if a.get('confidence_score', 0) > 0.7]
        if high_conf:
            answer_parts.append(f"\nHigh confidence alignments ({len(high_conf)}):")
            for alignment in high_conf[:5]:  # Top 5
                ifc_entity = alignment.get('ifc_entity', 'Unknown')
                reg_entity = alignment.get('regulatory_entity', 'Unknown')
                confidence = alignment.get('confidence_score', 0)
                answer_parts.append(f"- {ifc_entity} ↔ {reg_entity} (confidence: {confidence:.2f})")
                
        # Medium confidence alignments
        med_conf = [a for a in alignments if 0.4 <= a.get('confidence_score', 0) <= 0.7]
        if med_conf:
            answer_parts.append(f"\nMedium confidence alignments ({len(med_conf)}):")
            for alignment in med_conf[:3]:  # Top 3
                ifc_entity = alignment.get('ifc_entity', 'Unknown')
                reg_entity = alignment.get('regulatory_entity', 'Unknown')
                confidence = alignment.get('confidence_score', 0)
                answer_parts.append(f"- {ifc_entity} ↔ {reg_entity} (confidence: {confidence:.2f})")
                
        # Overall assessment
        overall_confidence = self._calculate_overall_confidence()
        answer_parts.append(f"\nOverall alignment confidence: {overall_confidence:.2f}")
        
        if overall_confidence > 0.7:
            answer_parts.append("The semantic alignment analysis shows strong correspondence between IFC and regulatory terminology.")
        elif overall_confidence > 0.4:
            answer_parts.append("The semantic alignment analysis shows moderate correspondence. Additional context may improve accuracy.")
        else:
            answer_parts.append("The semantic alignment analysis shows limited correspondence. The query may require domain-specific knowledge or different data sources.")
            
        return "\n".join(answer_parts)
        
    def _generate_fallback_answer(self) -> str:
        """Generate a fallback answer when normal processing fails."""
        return f"Unable to complete semantic alignment analysis for the query: '{self.working_memory.get('query', 'Unknown query')}'. The system encountered issues during processing. Please check the input data and try again."
        
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score for the current analysis."""
        confidence_scores = self.working_memory.get('confidence_scores', [])
        
        if not confidence_scores:
            return 0.0
            
        # Weighted average with higher weight for recent scores
        weights = [1.0 + (i * 0.1) for i in range(len(confidence_scores))]
        weighted_sum = sum(score * weight for score, weight in zip(confidence_scores, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
    def _get_supporting_evidence(self) -> List[str]:
        """Get supporting evidence for current reasoning."""
        evidence = []
        
        if self.working_memory.get('extracted_entities'):
            ifc_count = len(self.working_memory['extracted_entities'].get('ifc_entities', []))
            reg_count = len(self.working_memory['extracted_entities'].get('regulatory_entities', []))
            evidence.append(f"Extracted {ifc_count} IFC entities and {reg_count} regulatory entities")
            
        if self.working_memory.get('alignments'):
            alignment_count = len(self.working_memory['alignments'])
            evidence.append(f"Generated {alignment_count} semantic alignments")
            
        if self.react_history:
            successful_steps = len([step for step in self.react_history if step.success])
            evidence.append(f"Completed {successful_steps} successful reasoning steps")
            
        return evidence
        
    def _get_knowledge_sources(self) -> List[str]:
        """Get list of knowledge sources used in the analysis."""
        sources = set()
        
        if self.working_memory.get('ifc_data'):
            sources.add('IFC Data')
            
        if self.working_memory.get('regulatory_text'):
            sources.add('Regulatory Text')
            
        sources.add('Knowledge Graph')
        sources.add('Semantic Alignment Rules')
        
        return list(sources)
        
    def _evaluate_result_confidence(self, result: Dict[str, Any]) -> float:
        """Evaluate confidence of an action result."""
        if 'error' in result:
            return 0.0
            
        # Different confidence evaluation based on result type
        if 'confidence_score' in result:
            return result['confidence_score']
            
        if 'average_confidence' in result:
            return result['average_confidence']
            
        if 'entities_count' in result:
            # More entities generally means higher confidence
            count = result['entities_count']
            return min(count / 10.0, 1.0)  # Normalize to 0-1
            
        if 'alignments_count' in result:
            # More alignments generally means higher confidence
            count = result['alignments_count']
            return min(count / 5.0, 1.0)  # Normalize to 0-1
            
        return 0.5  # Default moderate confidence
        
    def _suggest_next_actions(self) -> List[str]:
        """Suggest next actions based on current state."""
        suggestions = []
        
        if not self.working_memory.get('extracted_entities'):
            suggestions.append("Extract entities from available data sources")
            
        if not self.working_memory.get('alignments'):
            suggestions.append("Perform semantic alignment between entities")
            
        current_confidence = self._calculate_overall_confidence()
        if current_confidence < 0.5:
            suggestions.append("Query knowledge graph for additional context")
            suggestions.append("Refine entity extraction with different parameters")
            
        return suggestions
        
    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state for debugging or monitoring."""
        return {
            'current_step': self.current_step,
            'working_memory': self.working_memory,
            'react_history_length': len(self.react_history),
            'overall_confidence': self._calculate_overall_confidence(),
            'available_actions': list(self.available_actions.keys())
        }
        
    def reset_agent(self):
        """Reset agent state for new query processing."""
        self.current_step = 0
        self.react_history = []
        self.working_memory = {}
        self.logger.info("Agent state reset")