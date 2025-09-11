"""IFC Semantic Agent using ReAct (Reasoning + Acting) framework - No Knowledge Graph Version."""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .fusion_module import TextStructuredDataFusion
from .semantic_alignment import SemanticAlignment

# Removed knowledge graph imports: GraphBuilder, RAGSystem, EntityResolver
from llm import SemanticExtractor, NERRelationExtractor
from data_processing import IFCProcessor, TextProcessor
from utils import ConfigLoader, get_logger


class ActionType(Enum):
    """Types of actions the agent can perform."""

    ANALYZE_IFC = "analyze_ifc"
    PROCESS_REGULATORY_TEXT = "process_regulatory_text"
    EXTRACT_SEMANTICS = "extract_semantics"
    ALIGN_ENTITIES = "align_entities"
    # Removed knowledge graph actions: QUERY_KNOWLEDGE_GRAPH, RESOLVE_ENTITY
    GENERATE_MAPPING = "generate_mapping"
    VALIDATE_ALIGNMENT = "validate_alignment"
    REFLECT_ON_RESULTS = "reflect_on_results"


@dataclass
class Observation:
    """Observation from environment after action execution."""

    content: Any
    observation_type: str
    timestamp: float
    metadata: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class Thought:
    """Agent's reasoning step."""

    content: str
    reasoning_type: str  # 'analysis', 'planning', 'reflection', 'conclusion'
    confidence: float
    supporting_evidence: List[str]
    timestamp: float


@dataclass
class Action:
    """Action to be executed by the agent."""

    action_type: ActionType
    parameters: Dict[str, Any]
    reasoning: str
    expected_outcome: str
    timestamp: float


@dataclass
class ReActStep:
    """Single step in ReAct reasoning chain."""

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


class IFCSemanticAgentNoKG:
    """ReAct-based agent for IFC-regulatory semantic understanding and alignment - No Knowledge Graph Version."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the IFC Semantic Agent without Knowledge Graph.

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)

        # Initialize core components (without knowledge graph)
        self._initialize_components()

        # ReAct configuration
        self.max_iterations = self.config.get("react_agent.max_iterations", 10)
        self.reflection_threshold = self.config.get("react_agent.reflection_threshold", 0.3)
        self.confidence_threshold = self.config.get("react_agent.confidence_threshold", 0.7)

        # Agent state
        self.current_step = 0
        self.react_history: List[ReActStep] = []
        self.working_memory: Dict[str, Any] = {}

        # Available tools/actions (without knowledge graph actions)
        self.available_actions = {
            ActionType.ANALYZE_IFC: self._analyze_ifc_data,
            ActionType.PROCESS_REGULATORY_TEXT: self._process_regulatory_text,
            ActionType.EXTRACT_SEMANTICS: self._extract_semantics,
            ActionType.ALIGN_ENTITIES: self._align_entities,
            # Removed: ActionType.QUERY_KNOWLEDGE_GRAPH, ActionType.RESOLVE_ENTITY
            ActionType.GENERATE_MAPPING: self._generate_mapping,
            ActionType.VALIDATE_ALIGNMENT: self._validate_alignment,
            ActionType.REFLECT_ON_RESULTS: self._reflect_on_results,
        }

    def _initialize_components(self):
        """Initialize all agent components (without knowledge graph)."""
        try:
            # Core processing modules
            self.ifc_processor = IFCProcessor()
            self.text_processor = TextProcessor()

            # LLM-based semantic extraction
            self.semantic_extractor = SemanticExtractor()
            self.ner_extractor = NERRelationExtractor()

            # Removed knowledge graph components:
            # self.graph_builder = GraphBuilder()
            # self.rag_system = RAGSystem(self.graph_builder)
            # self.entity_resolver = EntityResolver(self.graph_builder)

            # Core fusion and alignment modules
            self.fusion_module = TextStructuredDataFusion()
            self.semantic_alignment = SemanticAlignment()

            self.logger.info(
                "All agent components initialized successfully (without knowledge graph)"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def process_query(
        self,
        query: str,
        ifc_data: Optional[Dict[str, Any]] = None,
        regulatory_text: Optional[str] = None,
    ) -> AgentResponse:
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
            "query": query,
            "ifc_data": ifc_data,
            "regulatory_text": regulatory_text,
            "extracted_entities": {},
            "alignments": [],
            "confidence_scores": [],
        }

        try:
            # Main ReAct loop
            while self.current_step < self.max_iterations:
                # Think
                thought = self._generate_thought()

                # Act
                action = self._plan_action(thought)

                # Execute action and observe
                observation = self._execute_action(action)

                # Create ReAct step
                react_step = ReActStep(
                    step_id=self.current_step,
                    thought=thought,
                    action=action,
                    observation=observation,
                    success=observation.confidence > 0.5,
                )

                self.react_history.append(react_step)

                # Check if we should stop
                if self._should_stop(react_step):
                    break

                self.current_step += 1

            # Generate final response
            final_answer = self._generate_final_answer()
            confidence_score = self._calculate_overall_confidence()

            return AgentResponse(
                query=query,
                final_answer=final_answer,
                confidence_score=confidence_score,
                react_steps=self.react_history,
                total_steps=len(self.react_history),
                execution_time=time.time() - start_time,
                knowledge_sources=self._get_knowledge_sources(),
                semantic_mappings=self.working_memory.get("alignments", []),
            )

        except Exception as e:
            self.logger.error(f"Error in process_query: {e}")
            return AgentResponse(
                query=query,
                final_answer=f"Error processing query: {str(e)}",
                confidence_score=0.0,
                react_steps=self.react_history,
                total_steps=len(self.react_history),
                execution_time=time.time() - start_time,
                knowledge_sources=[],
                semantic_mappings=[],
            )

    def _generate_thought(self) -> Thought:
        """Generate the next reasoning step."""
        if self.current_step == 0:
            # Initial analysis
            content = f"I need to analyze the query: '{self.working_memory['query']}' and determine what data sources are available."
            reasoning_type = "analysis"
            confidence = 0.9
            evidence = ["Starting new query processing"]

        elif self.current_step < 3:
            # Planning phase
            content = "I need to extract entities from the available data sources and prepare for semantic alignment."
            reasoning_type = "planning"
            confidence = 0.8
            evidence = [f"Step {self.current_step} of entity extraction and processing"]

        elif len(self.working_memory.get("alignments", [])) == 0:
            # Need alignment
            content = "I have extracted entities and now need to perform semantic alignment between IFC and regulatory entities."
            reasoning_type = "planning"
            confidence = 0.7
            evidence = ["Entities extracted, ready for alignment"]

        else:
            # Conclusion phase
            content = "I have performed semantic alignment and should generate the final mapping and conclusions."
            reasoning_type = "conclusion"
            confidence = 0.8
            evidence = ["Alignment completed, ready for final answer"]

        return Thought(
            content=content,
            reasoning_type=reasoning_type,
            confidence=confidence,
            supporting_evidence=evidence,
            timestamp=time.time(),
        )

    def _plan_action(self, thought: Thought) -> Action:
        """Plan the next action based on the current thought."""
        if thought.reasoning_type == "analysis" and self.current_step == 0:
            # Decide what to analyze first
            if self.working_memory.get("ifc_data"):
                action_type = ActionType.ANALYZE_IFC
                parameters = {"data": self.working_memory["ifc_data"]}
                reasoning = "Starting with IFC data analysis to understand available entities"
                expected_outcome = "Extracted IFC entities with properties and relationships"
            elif self.working_memory.get("regulatory_text"):
                action_type = ActionType.PROCESS_REGULATORY_TEXT
                parameters = {"text": self.working_memory["regulatory_text"]}
                reasoning = "Starting with regulatory text processing to extract terms"
                expected_outcome = "Extracted regulatory entities and relationships"
            else:
                # No direct data, use semantic extraction on query
                action_type = ActionType.EXTRACT_SEMANTICS
                parameters = {"query": self.working_memory["query"]}
                reasoning = "No direct data provided, extracting semantics from query"
                expected_outcome = "Semantic entities and concepts from query"

        elif thought.reasoning_type == "planning":
            if len(self.working_memory.get("extracted_entities", {})) == 0:
                # Need to extract entities
                if self.working_memory.get(
                    "ifc_data"
                ) and "ifc_entities" not in self.working_memory.get("extracted_entities", {}):
                    action_type = ActionType.ANALYZE_IFC
                    parameters = {"data": self.working_memory["ifc_data"]}
                elif self.working_memory.get(
                    "regulatory_text"
                ) and "regulatory_entities" not in self.working_memory.get(
                    "extracted_entities", {}
                ):
                    action_type = ActionType.PROCESS_REGULATORY_TEXT
                    parameters = {"text": self.working_memory["regulatory_text"]}
                else:
                    action_type = ActionType.EXTRACT_SEMANTICS
                    parameters = {"query": self.working_memory["query"]}

                reasoning = "Extracting entities from available data sources"
                expected_outcome = "Comprehensive entity extraction results"

            else:
                # Perform alignment
                ifc_entities = self.working_memory.get("extracted_entities", {}).get("ifc_entities", [])
                regulatory_entities = self.working_memory.get("extracted_entities", {}).get("regulatory_entities", [])
                
                # Convert ifc_entities from dict to list if needed
                if isinstance(ifc_entities, dict):
                    ifc_entities = list(ifc_entities.values())
                
                action_type = ActionType.ALIGN_ENTITIES
                parameters = {
                    "ifc_entities": ifc_entities,
                    "regulatory_entities": regulatory_entities,
                }
                reasoning = "Performing semantic alignment between extracted entities"
                expected_outcome = "Semantic alignment mappings with confidence scores"

        else:  # conclusion
            action_type = ActionType.GENERATE_MAPPING
            parameters = {
                "alignments": self.working_memory.get("alignments", []),
                "query": self.working_memory["query"],
            }
            reasoning = "Generating final semantic mapping based on alignment results"
            expected_outcome = "Complete semantic mapping with explanations"

        return Action(
            action_type=action_type,
            parameters=parameters,
            reasoning=reasoning,
            expected_outcome=expected_outcome,
            timestamp=time.time(),
        )

    def _execute_action(self, action: Action) -> Observation:
        """Execute the planned action and return observation."""
        try:
            # Get the action function
            action_func = self.available_actions.get(action.action_type)
            if not action_func:
                raise ValueError(f"Unknown action type: {action.action_type}")

            # Execute the action
            result = action_func(action.parameters)

            return Observation(
                content=result,
                observation_type=action.action_type.value,
                timestamp=time.time(),
                metadata={"action_parameters": action.parameters},
                confidence=result.get("confidence_score", 0.8) if isinstance(result, dict) else 0.8,
            )

        except Exception as e:
            self.logger.error(f"Error executing action {action.action_type}: {e}")
            return Observation(
                content={"error": str(e)},
                observation_type="error",
                timestamp=time.time(),
                metadata={"action_parameters": action.parameters},
                confidence=0.0,
            )

    def _analyze_ifc_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze IFC data and extract entities (without knowledge graph)."""
        ifc_data = parameters.get("data")

        if isinstance(ifc_data, str):  # File path
            processed_data = self.ifc_processor.process_ifc_file(ifc_data)
        else:  # Already processed data
            processed_data = ifc_data

        # Get entities - handle both list and dict formats
        entities = processed_data.get("entities", [])
        if isinstance(entities, list):
            # Convert list to dict for consistency
            entities_dict = {f"entity_{i}": entity for i, entity in enumerate(entities)}
        else:
            entities_dict = entities

        # Store entities in working memory (without adding to knowledge graph)
        if "extracted_entities" not in self.working_memory:
            self.working_memory["extracted_entities"] = {}
        self.working_memory["extracted_entities"]["ifc_entities"] = entities_dict

        return {
            "entities_count": len(entities) if isinstance(entities, list) else len(entities_dict),
            "relationships_count": len(processed_data.get("relationships", [])),
            "entity_types": list(
                set(
                    [
                        e.get("type")
                        for e in (
                            entities if isinstance(entities, list) else entities_dict.values()
                        )
                    ]
                )
            ),
            "confidence_score": 0.9,
        }

    def _process_regulatory_text(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process regulatory text and extract entities."""
        text = parameters.get("text")

        # Process text
        processed_text = self.text_processor.process_text(text)

        # Extract entities using NER
        ner_result = self.ner_extractor.extract_entities_and_relations(text)

        # Store in working memory
        if "extracted_entities" not in self.working_memory:
            self.working_memory["extracted_entities"] = {}
        self.working_memory["extracted_entities"]["regulatory_entities"] = ner_result.entities
        self.working_memory["extracted_entities"]["regulatory_relations"] = ner_result.relations

        return {
            "entities_count": len(ner_result.entities),
            "relations_count": len(ner_result.relations),
            "entity_types": list(set([e.entity_type for e in ner_result.entities])),
            "confidence_score": ner_result.confidence_score,
        }

    def _extract_semantics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic information from query or text."""
        query = parameters.get("query", "")

        # Use semantic extractor
        semantic_result = self.semantic_extractor.extract_semantic_information(query)

        # Store in working memory
        if "extracted_entities" not in self.working_memory:
            self.working_memory["extracted_entities"] = {}
        self.working_memory["extracted_entities"]["semantic_entities"] = semantic_result.entities
        self.working_memory["extracted_entities"][
            "semantic_concepts"
        ] = semantic_result.key_concepts

        # Calculate overall confidence from confidence_scores dict
        overall_confidence = semantic_result.confidence_scores.get("overall", 0.0)

        return {
            "entities_count": len(semantic_result.entities),
            "concepts_count": len(semantic_result.key_concepts),
            "confidence_score": overall_confidence,
        }

    def _align_entities(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic alignment between entities."""
        ifc_entities = parameters.get("ifc_entities", [])
        regulatory_entities = parameters.get("regulatory_entities", [])

        # Perform alignment using semantic alignment module
        alignments = self.semantic_alignment.align_entities(
            ifc_entities=ifc_entities, regulatory_entities=regulatory_entities
        )

        # Store alignments in working memory
        self.working_memory["alignments"] = alignments
        confidence_scores = [a.confidence_score for a in alignments]
        self.working_memory["confidence_scores"] = confidence_scores

        # Calculate overall confidence
        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        )

        return {
            "alignments_count": len(alignments),
            "average_confidence": overall_confidence,
            "confidence_score": overall_confidence,
        }

    def _generate_mapping(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final semantic mapping."""
        alignments = parameters.get("alignments", [])
        query = parameters.get("query", "")

        # Generate comprehensive mapping
        mapping = {
            "query": query,
            "alignments": alignments,
            "summary": self._generate_alignment_summary(alignments),
            "recommendations": self._generate_recommendations(alignments),
        }

        self.working_memory["final_mapping"] = mapping

        return {"mapping": mapping, "confidence_score": self._calculate_overall_confidence()}

    def _validate_alignment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the semantic alignment results."""
        alignments = self.working_memory.get("alignments", [])

        validation_results = []
        for alignment in alignments:
            # Simple validation based on confidence scores
            # Handle both AlignmentResult objects and dictionaries
            if hasattr(alignment, "confidence_score"):
                confidence = alignment.confidence_score
            else:
                confidence = alignment.get("confidence_score", 0)

            is_valid = confidence > 0.5
            validation_results.append(
                {"alignment": alignment, "is_valid": is_valid, "validation_score": confidence}
            )

        return {
            "validation_results": validation_results,
            "valid_count": sum(1 for r in validation_results if r["is_valid"]),
            "total_count": len(validation_results),
            "confidence_score": 0.8,
        }

    def _reflect_on_results(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on the current results and suggest improvements."""
        current_confidence = self._calculate_overall_confidence()

        reflections = []
        if current_confidence < 0.5:
            reflections.append(
                "Low confidence in alignments - may need more context or different approach"
            )
        if len(self.working_memory.get("alignments", [])) == 0:
            reflections.append(
                "No alignments found - may need to adjust entity extraction parameters"
            )

        suggestions = self._suggest_next_actions()

        return {
            "reflections": reflections,
            "suggestions": suggestions,
            "current_confidence": current_confidence,
            "confidence_score": 0.7,
        }

    def _should_stop(self, react_step: ReActStep) -> bool:
        """Determine if the agent should stop reasoning."""
        # Stop if we have a final mapping with good confidence
        if "final_mapping" in self.working_memory:
            confidence = self._calculate_overall_confidence()
            return confidence > self.confidence_threshold

        # Stop if we've reached max iterations
        if self.current_step >= self.max_iterations - 1:
            return True

        # Check for consistent progress - only stop if multiple consecutive steps fail
        if len(self.react_history) >= 3:
            recent_steps = self.react_history[-3:]
            recent_confidences = [step.observation.confidence for step in recent_steps]

            # Stop only if all recent steps have very low confidence AND no progress
            if all(conf < 0.2 for conf in recent_confidences):
                # Check if we have any useful results despite low step confidence
                alignments = self.working_memory.get("alignments", [])
                if not alignments or all(
                    getattr(a, "confidence_score", a.get("confidence_score", 0)) < 0.3
                    for a in alignments
                ):
                    return True

        # Don't stop based on single step failure - allow recovery
        return False

    def _generate_final_answer(self) -> str:
        """Generate the final answer based on the reasoning process."""
        if "final_mapping" in self.working_memory:
            mapping = self.working_memory["final_mapping"]
            return mapping.get("summary", "Semantic alignment completed")

        # Enhanced fallback answer with detailed analysis
        alignments = self.working_memory.get("alignments", [])
        extracted_entities = self.working_memory.get("extracted_entities", {})
        query = self.working_memory.get("query", "")

        answer_parts = []

        # Query analysis summary
        if query:
            answer_parts.append(f"Analysis for query: '{query}'")
            answer_parts.append("=" * 50)

        # Entity extraction results
        if extracted_entities:
            ifc_entities = extracted_entities.get("ifc_entities", [])
            reg_entities = extracted_entities.get("regulatory_entities", [])
            semantic_entities = extracted_entities.get("semantic_entities", [])

            if ifc_entities or reg_entities or semantic_entities:
                answer_parts.append("\n📊 Entity Extraction Results:")
                if ifc_entities:
                    answer_parts.append(f"  • IFC entities found: {len(ifc_entities)}")
                if reg_entities:
                    answer_parts.append(f"  • Regulatory entities found: {len(reg_entities)}")
                if semantic_entities:
                    answer_parts.append(f"  • Semantic entities found: {len(semantic_entities)}")

        # Alignment results
        if alignments:
            answer_parts.append("\n🔗 Semantic Alignment Results:")

            # Categorize alignments by confidence
            high_conf = [a for a in alignments if self._get_alignment_confidence(a) > 0.7]
            med_conf = [a for a in alignments if 0.4 <= self._get_alignment_confidence(a) <= 0.7]
            low_conf = [a for a in alignments if self._get_alignment_confidence(a) < 0.4]

            answer_parts.append(f"  • Total alignments: {len(alignments)}")
            answer_parts.append(f"  • High confidence (>0.7): {len(high_conf)}")
            answer_parts.append(f"  • Medium confidence (0.4-0.7): {len(med_conf)}")
            answer_parts.append(f"  • Low confidence (<0.4): {len(low_conf)}")

            # Show top alignments
            top_alignments = sorted(alignments, key=self._get_alignment_confidence, reverse=True)[
                :5
            ]
            if top_alignments:
                answer_parts.append("\n🏆 Top Alignments:")
                for i, alignment in enumerate(top_alignments, 1):
                    ifc_entity = self._get_alignment_field(alignment, "ifc_entity")
                    reg_entity = self._get_alignment_field(alignment, "regulatory_entity")
                    confidence = self._get_alignment_confidence(alignment)
                    answer_parts.append(
                        f"  {i}. {ifc_entity} ↔ {reg_entity} (confidence: {confidence:.2f})"
                    )

            overall_confidence = self._calculate_overall_confidence()
            answer_parts.append(f"\n📈 Overall alignment confidence: {overall_confidence:.2f}")

            # Provide interpretation
            if overall_confidence > 0.7:
                answer_parts.append(
                    "✅ Strong semantic correspondence found between IFC and regulatory terminology."
                )
            elif overall_confidence > 0.4:
                answer_parts.append(
                    "⚠️ Moderate semantic correspondence found. Additional context may improve accuracy."
                )
            else:
                answer_parts.append(
                    "❌ Limited semantic correspondence found. Consider providing more specific data."
                )
        else:
            answer_parts.append("\n❌ No semantic alignments were found.")

        # Reasoning process summary
        if self.react_history:
            successful_steps = [step for step in self.react_history if step.success]
            answer_parts.append(
                f"\n🔄 Reasoning Process: {len(self.react_history)} steps ({len(successful_steps)} successful)"
            )

        # Recommendations
        recommendations = self._generate_enhanced_recommendations()
        if recommendations:
            answer_parts.append("\n💡 Recommendations:")
            for rec in recommendations:
                answer_parts.append(f"  • {rec}")

        return (
            "\n".join(answer_parts)
            if answer_parts
            else "Unable to complete semantic alignment analysis with the provided data."
        )

    def _get_alignment_confidence(self, alignment) -> float:
        """Get confidence score from alignment object."""
        if hasattr(alignment, "confidence_score"):
            return alignment.confidence_score
        return alignment.get("confidence_score", 0)

    def _get_alignment_field(self, alignment, field_name: str) -> str:
        """Get field value from alignment object."""
        if hasattr(alignment, field_name):
            return getattr(alignment, field_name, "Unknown")
        return alignment.get(field_name, "Unknown")

    def _generate_enhanced_recommendations(self) -> List[str]:
        """Generate enhanced recommendations based on analysis results."""
        recommendations = []

        alignments = self.working_memory.get("alignments", [])
        extracted_entities = self.working_memory.get("extracted_entities", {})
        overall_confidence = self._calculate_overall_confidence()

        # Data quality recommendations
        if not alignments:
            recommendations.append(
                "Provide more specific IFC data or regulatory text for better alignment"
            )
            recommendations.append("Consider using domain-specific terminology in your query")

        # Confidence-based recommendations
        if overall_confidence < 0.3:
            recommendations.append("Low confidence suggests need for more contextual information")
            recommendations.append("Try rephrasing the query with more specific building elements")
        elif overall_confidence < 0.6:
            recommendations.append(
                "Moderate confidence - consider adding more detailed entity descriptions"
            )
            recommendations.append("Review alignment results and validate with domain expertise")
        else:
            recommendations.append("High confidence results - suitable for further processing")
            recommendations.append("Consider using these alignments for automated mapping systems")

        # Entity extraction recommendations
        if not extracted_entities or all(not entities for entities in extracted_entities.values()):
            recommendations.append("Improve entity extraction by providing structured input data")
            recommendations.append(
                "Ensure IFC and regulatory texts contain relevant building terminology"
            )

        # Process-based recommendations
        if len(self.react_history) >= self.max_iterations - 1:
            recommendations.append(
                "Analysis reached maximum iterations - consider increasing iteration limit"
            )
            recommendations.append(
                "Complex queries may benefit from breaking down into smaller parts"
            )

        return recommendations

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score."""
        confidence_scores = self.working_memory.get("confidence_scores", [])
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        return 0.5  # Default moderate confidence

    def _get_knowledge_sources(self) -> List[str]:
        """Get list of knowledge sources used in the analysis."""
        sources = set()

        if self.working_memory.get("ifc_data"):
            sources.add("IFC Data")

        if self.working_memory.get("regulatory_text"):
            sources.add("Regulatory Text")

        # Removed: sources.add('Knowledge Graph')
        sources.add("Semantic Alignment Rules")
        sources.add("LLM Semantic Extraction")

        return list(sources)

    def _generate_alignment_summary(self, alignments: List[Any]) -> str:
        """Generate a summary of the alignment results."""
        if not alignments:
            return "No semantic alignments found."

        def get_confidence(alignment):
            if hasattr(alignment, "confidence_score"):
                return alignment.confidence_score
            else:
                return alignment.get("confidence_score", 0)

        high_conf = [a for a in alignments if get_confidence(a) > 0.7]
        med_conf = [a for a in alignments if 0.4 <= get_confidence(a) <= 0.7]
        low_conf = [a for a in alignments if get_confidence(a) < 0.4]

        summary = f"Found {len(alignments)} semantic alignments:\n"
        summary += f"- High confidence: {len(high_conf)}\n"
        summary += f"- Medium confidence: {len(med_conf)}\n"
        summary += f"- Low confidence: {len(low_conf)}"

        return summary

    def _generate_recommendations(self, alignments: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on alignment results."""
        recommendations = []

        if not alignments:
            recommendations.append("Consider providing more specific IFC data or regulatory text")
            recommendations.append("Try rephrasing the query with more domain-specific terms")
        else:

            def get_confidence(alignment):
                if hasattr(alignment, "confidence_score"):
                    return alignment.confidence_score
                else:
                    return alignment.get("confidence_score", 0)

            avg_confidence = sum(get_confidence(a) for a in alignments) / len(alignments)
            if avg_confidence < 0.5:
                recommendations.append(
                    "Consider adding more context to improve alignment confidence"
                )
                recommendations.append("Review entity extraction parameters for better results")
            else:
                recommendations.append("Alignment results show good semantic correspondence")
                recommendations.append("Consider validating results with domain experts")

        return recommendations

    def _suggest_next_actions(self) -> List[str]:
        """Suggest next actions based on current state."""
        suggestions = []

        if not self.working_memory.get("extracted_entities"):
            suggestions.append("Extract entities from available data sources")

        if not self.working_memory.get("alignments"):
            suggestions.append("Perform semantic alignment between entities")

        current_confidence = self._calculate_overall_confidence()
        if current_confidence < 0.5:
            suggestions.append("Refine entity extraction with different parameters")
            suggestions.append("Consider using additional semantic similarity methods")

        return suggestions

    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state for debugging or monitoring."""
        return {
            "current_step": self.current_step,
            "working_memory": self.working_memory,
            "react_history_length": len(self.react_history),
            "overall_confidence": self._calculate_overall_confidence(),
            "available_actions": list(self.available_actions.keys()),
        }

    def reset_agent(self):
        """Reset agent state for new query processing."""
        self.current_step = 0
        self.react_history = []
        self.working_memory = {}
        self.logger.info("Agent state reset")
