# IFC Semantic Agent

An intelligent agent based on the ReAct framework for semantic understanding and alignment between IFC structured data and regulatory text data.

## Project Overview

This project aims to solve the problem of lacking adaptive mapping between IFC spatial semantics (such as IfcSlab as structural slab vs. equipment platform, or IfcOpeningElement as functional shaft vs. atrium) and regulatory terminology. By integrating knowledge graph RAG methods, the system can understand the relationships between entities to achieve better semantic alignment.

## Core Features

### 🏗️ Multi-modal Data Fusion
- **IFC Data Processing**: Supports IFC file parsing, extracting building entities, properties, spatial relationships, etc.
- **Regulatory Text Processing**: Processes regulatory documents in PDF, DOCX, Markdown and other formats
- **Semantic Feature Extraction**: Uses LLM for deep semantic understanding

### 🧠 Intelligent Semantic Alignment
- **Entity Alignment**: IFC entities ↔ regulatory terms (e.g., IfcWall ↔ "wall"/"partition"/"load-bearing wall")
- **Attribute Alignment**: IFC properties ↔ regulatory parameters (e.g., Width ↔ "thickness"/"width")
- **Relationship Alignment**: Spatial relationships ↔ regulatory logic (e.g., ContainedIn ↔ "located in"/"contains")

### 🕸️ Knowledge Graph RAG System
- **Graph Construction**: Automatically builds IFC-regulatory knowledge graphs
- **Intelligent Retrieval**: Multi-strategy retrieval (semantic similarity, graph traversal, contextual paths, hybrid strategies)
- **Entity Disambiguation**: Intelligent entity resolution and disambiguation

### 🤖 ReAct Intelligent Agent
- **Reasoning-Action Loop**: Iterative process of Think→Act→Observe
- **Adaptive Decision Making**: Dynamically adjusts strategies based on context
- **Explainability**: Complete reasoning traces and decision processes

## Project Structure

```
alignment_agent/
├── core/                          # Core modules
│   ├── __init__.py
│   ├── fusion_module.py           # Text-structured data fusion module
│   ├── semantic_alignment.py      # Cross-modal semantic alignment module
│   └── ifc_semantic_agent.py      # ReAct framework intelligent agent
├── data_processing/               # Data processing modules
│   ├── __init__.py
│   ├── ifc_processor.py           # IFC data processor
│   └── text_processor.py          # Text processor
├── llm/                          # LLM integration modules
│   ├── __init__.py
│   ├── semantic_extractor.py      # Semantic extractor
│   └── ner_relation_extractor.py  # NER and relation extractor
├── knowledge_graph/              # Knowledge graph modules
│   ├── __init__.py
│   ├── graph_builder.py           # Graph builder
│   ├── rag_system.py              # RAG system
│   └── entity_resolver.py         # Entity resolver
├── utils/                        # Utility modules
│   ├── __init__.py
│   ├── config_loader.py           # Configuration loader
│   ├── logger.py                  # Logging utilities
│   └── file_utils.py              # File utilities
├── config.yaml                   # Main configuration file
├── requirements.txt              # Dependency management
└── README.md                     # Project documentation
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the project
git clone <repository-url>
cd alignment_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
```

### 2. Configuration Setup

Create a `.env` file and set necessary environment variables:

```bash
# LLM API configuration
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Neo4j configuration (optional)
NEO4J_PASSWORD=your_neo4j_password
```

### 3. Basic Usage

```python
from core import IFCSemanticAgent

# Initialize the intelligent agent
agent = IFCSemanticAgent("config.yaml")

# Process semantic alignment query
response = agent.process_query(
    query="How to map IfcSlab to platform concept in regulatory documents?",
    ifc_data="path/to/building.ifc",
    regulatory_text="Regulatory document content..."
)

# View results
print(f"Final answer: {response.final_answer}")
print(f"Confidence: {response.confidence_score}")
print(f"Reasoning steps: {response.total_steps}")
```

## Core Algorithms

### ReAct Framework Process

1. **Thought Phase**: Analyze current state and determine next action
2. **Action Phase**: Execute specific operations (data processing, semantic extraction, alignment, etc.)
3. **Observation Phase**: Evaluate action results and update working memory
4. **Iterative Optimization**: Decide whether to continue based on confidence and reflection mechanisms

### Semantic Alignment Strategies

- **Lexical Similarity**: Based on string matching and edit distance
- **Semantic Similarity**: Calculate vector similarity using pre-trained embedding models
- **Contextual Similarity**: Consider surrounding text and structural information
- **Structural Similarity**: Based on entity hierarchies and relationship patterns

### Knowledge Graph Retrieval

- **Semantic Retrieval**: Similarity search based on embedding vectors
- **Graph Traversal**: Explore neighbor nodes along relationship edges
- **Path Finding**: Find semantic connection paths between entities
- **Hybrid Strategy**: Comprehensive scoring combining multiple retrieval methods

## Configuration

Main configuration items in `config.yaml`:

- `react_agent`: ReAct agent parameters (max iterations, confidence threshold, etc.)
- `llm`: LLM configuration (provider, model, API keys, etc.)
- `embeddings`: Embedding model configuration
- `knowledge_graph`: Knowledge graph backend configuration
- `semantic_alignment`: Semantic alignment parameters
- `rag_system`: RAG system configuration

## API Interfaces

### Core Classes

#### IFCSemanticAgent
Main intelligent agent class providing complete ReAct framework implementation.

```python
class IFCSemanticAgent:
    def process_query(self, query: str, ifc_data: Optional[Dict], regulatory_text: Optional[str]) -> AgentResponse
    def get_agent_state(self) -> Dict[str, Any]
    def reset_agent(self)
```

#### TextStructuredDataFusion
Multi-modal data fusion module.

```python
class TextStructuredDataFusion:
    def process_multimodal_input(self, ifc_data: Dict, text_data: str) -> Dict[str, Any]
    def extract_semantic_features(self, data: Dict) -> Dict[str, Any]
    def fuse_data(self, ifc_features: Dict, text_features: Dict) -> Dict[str, Any]
```

#### SemanticAlignment
Cross-modal semantic alignment module.

```python
class SemanticAlignment:
    def align_entities(self, ifc_entity: Dict, regulatory_entity: Dict) -> AlignmentResult
    def align_attributes(self, ifc_attrs: List, regulatory_attrs: List) -> List[AlignmentResult]
    def create_semantic_mapping(self, alignments: List) -> SemanticMapping
```

## Extension Development

### Adding New Data Processors

```python
from data_processing.base_processor import BaseProcessor

class CustomProcessor(BaseProcessor):
    def process(self, data: Any) -> Dict[str, Any]:
        # Implement custom processing logic
        pass
```

### Adding New Alignment Strategies

```python
from core.semantic_alignment import AlignmentStrategy

class CustomAlignmentStrategy(AlignmentStrategy):
    def calculate_similarity(self, entity1: Dict, entity2: Dict) -> float:
        # Implement custom similarity calculation
        pass
```

### Adding New Retrieval Strategies

```python
from knowledge_graph.rag_system import RetrievalStrategy

class CustomRetrievalStrategy(RetrievalStrategy):
    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        # Implement custom retrieval logic
        pass
```

## Testing

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_semantic_alignment.py

# Generate coverage report
pytest --cov=core --cov=data_processing --cov=llm --cov=knowledge_graph
```

## Performance Optimization

- **Caching Mechanism**: Automatically cache embedding vectors and processing results
- **Batch Processing**: Support batch data processing
- **Asynchronous Processing**: Support asynchronous I/O operations
- **Memory Management**: Intelligent memory usage and cleanup

## Troubleshooting

### Common Issues

1. **LLM API Call Failures**
   - Check API key configuration
   - Verify network connection
   - Review rate limits

2. **IFC File Parsing Errors**
   - Confirm correct file format
   - Check file size limits
   - Verify ifcopenshell installation

3. **Memory Issues**
   - Adjust batch processing size
   - Enable cache cleanup
   - Reduce concurrent processing count

### Log Analysis

Log files are located at `logs/agent.log`, containing detailed execution information and error stacks.

## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Project Maintainer: [Your Name]
- Email: [your.email@example.com]
- Project Link: [https://github.com/yourusername/alignment_agent](https://github.com/yourusername/alignment_agent)

## Acknowledgments

- OpenAI GPT models
- spaCy natural language processing library
- NetworkX graph processing library
- ChromaDB vector database
- IFCOpenShell IFC processing library