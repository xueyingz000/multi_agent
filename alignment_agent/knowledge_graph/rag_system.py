"""RAG System for knowledge graph-based retrieval and generation."""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .graph_builder import GraphBuilder, GraphNode, GraphRelation
from utils import ConfigLoader, get_logger


@dataclass
class RetrievalResult:
    """Represents a retrieval result from the knowledge graph."""
    node_id: str
    node_data: Dict[str, Any]
    similarity_score: float
    context_path: List[Dict[str, Any]]
    retrieval_method: str
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Represents a complete RAG response."""
    query: str
    retrieved_nodes: List[RetrievalResult]
    generated_response: str
    confidence_score: float
    reasoning_path: List[str]
    supporting_evidence: List[Dict[str, Any]]


class RAGSystem:
    """Retrieval-Augmented Generation system for IFC-regulatory knowledge graph."""
    
    def __init__(self, graph_builder: GraphBuilder, config_path: str = "config.yaml"):
        """Initialize the RAG system.
        
        Args:
            graph_builder: Initialized GraphBuilder instance
            config_path: Path to configuration file
        """
        self.graph_builder = graph_builder
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)
        
        # Vector database configuration
        self.vector_db_type = self.config.get('vector_database.type', 'chroma')
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize vector database
        self._init_vector_database()
        
        # Node embeddings cache
        self.node_embeddings: Dict[str, np.ndarray] = {}
        
        # Retrieval strategies
        self.retrieval_strategies = {
            'semantic_similarity': self._retrieve_by_semantic_similarity,
            'graph_traversal': self._retrieve_by_graph_traversal,
            'hybrid': self._retrieve_hybrid,
            'contextual_path': self._retrieve_by_contextual_path
        }
        
    def _init_embedding_model(self):
        """Initialize the embedding model."""
        model_name = self.config.get('embedding.model_name', 'all-MiniLM-L6-v2')
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                self.logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                self._init_mock_embedding()
        else:
            self._init_mock_embedding()
            
    def _init_mock_embedding(self):
        """Initialize mock embedding model."""
        self.embedding_model = None
        self.embedding_dim = 384  # Default dimension
        self.logger.warning("Using mock embedding model")
        
    def _init_vector_database(self):
        """Initialize vector database."""
        if self.vector_db_type == 'chroma' and CHROMA_AVAILABLE:
            self._init_chroma()
        elif self.vector_db_type == 'faiss' and FAISS_AVAILABLE:
            self._init_faiss()
        else:
            self._init_mock_vector_db()
            
    def _init_chroma(self):
        """Initialize ChromaDB."""
        try:
            chroma_config = self.config.get('vector_database.chroma', {})
            persist_directory = chroma_config.get('persist_directory', './chroma_db')
            
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name="ifc_regulatory_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.logger.info("ChromaDB initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            self._init_mock_vector_db()
            
    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_id_map = {}  # Map FAISS indices to node IDs
            self.logger.info("FAISS index initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS: {e}")
            self._init_mock_vector_db()
            
    def _init_mock_vector_db(self):
        """Initialize mock vector database."""
        self.vector_db = None
        self.logger.warning("Using mock vector database")
        
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.embedding_model:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        else:
            # Mock embedding - simple hash-based vector
            hash_val = hash(text) % 1000000
            return np.random.RandomState(hash_val).rand(self.embedding_dim).astype(np.float32)
            
    def index_knowledge_graph(self):
        """Index all nodes in the knowledge graph for retrieval."""
        self.logger.info("Indexing knowledge graph nodes...")
        
        nodes_to_index = []
        embeddings_to_index = []
        
        for node_id, node in self.graph_builder.nodes.items():
            # Create text representation of the node
            node_text = self._create_node_text_representation(node)
            
            # Generate embedding
            embedding = self.embed_text(node_text)
            self.node_embeddings[node_id] = embedding
            
            nodes_to_index.append({
                'id': node_id,
                'text': node_text,
                'metadata': {
                    'node_type': node.node_type,
                    'source': node.source,
                    'label': node.label
                }
            })
            embeddings_to_index.append(embedding)
            
        # Index in vector database
        if self.vector_db_type == 'chroma' and hasattr(self, 'collection'):
            self._index_chroma(nodes_to_index, embeddings_to_index)
        elif self.vector_db_type == 'faiss' and hasattr(self, 'faiss_index'):
            self._index_faiss(nodes_to_index, embeddings_to_index)
            
        self.logger.info(f"Indexed {len(nodes_to_index)} nodes")
        
    def _create_node_text_representation(self, node: GraphNode) -> str:
        """Create text representation of a graph node.
        
        Args:
            node: Graph node
            
        Returns:
            Text representation
        """
        text_parts = []
        
        # Add label and type
        text_parts.append(f"Label: {node.label}")
        text_parts.append(f"Type: {node.node_type}")
        
        # Add properties
        for key, value in node.properties.items():
            if isinstance(value, (str, int, float)):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (str, int, float)):
                        text_parts.append(f"{key}.{sub_key}: {sub_value}")
                        
        return " | ".join(text_parts)
        
    def _index_chroma(self, nodes: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Index nodes in ChromaDB."""
        try:
            ids = [node['id'] for node in nodes]
            documents = [node['text'] for node in nodes]
            metadatas = [node['metadata'] for node in nodes]
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception as e:
            self.logger.error(f"Failed to index in ChromaDB: {e}")
            
    def _index_faiss(self, nodes: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Index nodes in FAISS."""
        try:
            embeddings_array = np.vstack(embeddings)
            self.faiss_index.add(embeddings_array)
            
            # Update ID mapping
            for i, node in enumerate(nodes):
                self.faiss_id_map[i] = node['id']
        except Exception as e:
            self.logger.error(f"Failed to index in FAISS: {e}")
            
    def retrieve(self, query: str, strategy: str = 'hybrid', top_k: int = 5, **kwargs) -> List[RetrievalResult]:
        """Retrieve relevant nodes from the knowledge graph.
        
        Args:
            query: Search query
            strategy: Retrieval strategy
            top_k: Number of results to return
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of retrieval results
        """
        if strategy not in self.retrieval_strategies:
            self.logger.warning(f"Unknown strategy {strategy}, using hybrid")
            strategy = 'hybrid'
            
        return self.retrieval_strategies[strategy](query, top_k, **kwargs)
        
    def _retrieve_by_semantic_similarity(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """Retrieve nodes by semantic similarity."""
        query_embedding = self.embed_text(query)
        results = []
        
        if self.vector_db_type == 'chroma' and hasattr(self, 'collection'):
            try:
                chroma_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                
                for i, node_id in enumerate(chroma_results['ids'][0]):
                    if node_id in self.graph_builder.nodes:
                        node = self.graph_builder.nodes[node_id]
                        similarity = 1 - chroma_results['distances'][0][i]  # Convert distance to similarity
                        
                        results.append(RetrievalResult(
                            node_id=node_id,
                            node_data=node.__dict__,
                            similarity_score=similarity,
                            context_path=[],
                            retrieval_method='semantic_similarity',
                            metadata={'chroma_distance': chroma_results['distances'][0][i]}
                        ))
            except Exception as e:
                self.logger.error(f"ChromaDB query failed: {e}")
                
        elif self.vector_db_type == 'faiss' and hasattr(self, 'faiss_index'):
            try:
                similarities, indices = self.faiss_index.search(
                    query_embedding.reshape(1, -1), top_k
                )
                
                for i, idx in enumerate(indices[0]):
                    if idx in self.faiss_id_map:
                        node_id = self.faiss_id_map[idx]
                        if node_id in self.graph_builder.nodes:
                            node = self.graph_builder.nodes[node_id]
                            
                            results.append(RetrievalResult(
                                node_id=node_id,
                                node_data=node.__dict__,
                                similarity_score=similarities[0][i],
                                context_path=[],
                                retrieval_method='semantic_similarity',
                                metadata={'faiss_similarity': similarities[0][i]}
                            ))
            except Exception as e:
                self.logger.error(f"FAISS query failed: {e}")
        else:
            # Fallback: compute similarities manually
            results = self._retrieve_similarity_fallback(query, query_embedding, top_k)
            
        return results
        
    def _retrieve_similarity_fallback(self, query: str, query_embedding: np.ndarray, top_k: int) -> List[RetrievalResult]:
        """Fallback similarity retrieval."""
        similarities = []
        
        for node_id, node_embedding in self.node_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, node_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
            )
            similarities.append((node_id, similarity))
            
        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for node_id, similarity in similarities[:top_k]:
            if node_id in self.graph_builder.nodes:
                node = self.graph_builder.nodes[node_id]
                results.append(RetrievalResult(
                    node_id=node_id,
                    node_data=node.__dict__,
                    similarity_score=similarity,
                    context_path=[],
                    retrieval_method='semantic_similarity_fallback',
                    metadata={'cosine_similarity': similarity}
                ))
                
        return results
        
    def _retrieve_by_graph_traversal(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """Retrieve nodes by graph traversal from seed nodes."""
        # First find seed nodes using semantic similarity
        seed_results = self._retrieve_by_semantic_similarity(query, min(3, top_k), **kwargs)
        
        if not seed_results:
            return []
            
        # Expand from seed nodes
        expanded_nodes = set()
        results = []
        
        for seed_result in seed_results:
            # Add seed node
            expanded_nodes.add(seed_result.node_id)
            results.append(seed_result)
            
            # Get neighbors
            neighbors = self.graph_builder.query_neighbors(
                seed_result.node_id,
                relation_types=kwargs.get('relation_types')
            )
            
            for neighbor in neighbors:
                if neighbor['node_id'] not in expanded_nodes and len(results) < top_k:
                    # Calculate relevance score based on relation confidence and semantic similarity
                    relation_confidence = neighbor['relation'].get('confidence', 0.5)
                    semantic_score = seed_result.similarity_score * 0.8  # Decay factor
                    
                    combined_score = (relation_confidence + semantic_score) / 2
                    
                    results.append(RetrievalResult(
                        node_id=neighbor['node_id'],
                        node_data=neighbor['node_data'],
                        similarity_score=combined_score,
                        context_path=[seed_result.node_id],
                        retrieval_method='graph_traversal',
                        metadata={
                            'seed_node': seed_result.node_id,
                            'relation_type': neighbor['relation'].get('relation_type'),
                            'relation_confidence': relation_confidence
                        }
                    ))
                    
                    expanded_nodes.add(neighbor['node_id'])
                    
        return results[:top_k]
        
    def _retrieve_by_contextual_path(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """Retrieve nodes considering contextual paths."""
        # Find initial candidates
        candidates = self._retrieve_by_semantic_similarity(query, top_k * 2, **kwargs)
        
        enhanced_results = []
        
        for candidate in candidates:
            # Find paths to other relevant nodes
            context_paths = []
            
            for other_candidate in candidates:
                if candidate.node_id != other_candidate.node_id:
                    paths = self.graph_builder.find_semantic_paths(
                        candidate.node_id,
                        other_candidate.node_id,
                        max_depth=kwargs.get('max_path_depth', 3)
                    )
                    
                    if paths:
                        # Take the shortest path
                        shortest_path = min(paths, key=len)
                        context_paths.append({
                            'target_node': other_candidate.node_id,
                            'path': shortest_path,
                            'path_length': len(shortest_path)
                        })
                        
            # Calculate enhanced score based on context
            context_bonus = len(context_paths) * 0.1  # Bonus for being well-connected
            enhanced_score = candidate.similarity_score + context_bonus
            
            enhanced_results.append(RetrievalResult(
                node_id=candidate.node_id,
                node_data=candidate.node_data,
                similarity_score=enhanced_score,
                context_path=context_paths,
                retrieval_method='contextual_path',
                metadata={
                    'original_score': candidate.similarity_score,
                    'context_bonus': context_bonus,
                    'num_context_paths': len(context_paths)
                }
            ))
            
        # Sort by enhanced score and return top_k
        enhanced_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return enhanced_results[:top_k]
        
    def _retrieve_hybrid(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """Hybrid retrieval combining multiple strategies."""
        # Get results from different strategies
        semantic_results = self._retrieve_by_semantic_similarity(query, top_k, **kwargs)
        traversal_results = self._retrieve_by_graph_traversal(query, top_k, **kwargs)
        
        # Combine and deduplicate
        combined_results = {}
        
        # Add semantic results with weight
        for result in semantic_results:
            combined_results[result.node_id] = RetrievalResult(
                node_id=result.node_id,
                node_data=result.node_data,
                similarity_score=result.similarity_score * 0.7,  # Weight for semantic
                context_path=result.context_path,
                retrieval_method='hybrid_semantic',
                metadata=result.metadata
            )
            
        # Add traversal results with weight
        for result in traversal_results:
            if result.node_id in combined_results:
                # Combine scores
                existing = combined_results[result.node_id]
                combined_score = existing.similarity_score + (result.similarity_score * 0.3)
                
                combined_results[result.node_id] = RetrievalResult(
                    node_id=result.node_id,
                    node_data=result.node_data,
                    similarity_score=combined_score,
                    context_path=existing.context_path + result.context_path,
                    retrieval_method='hybrid_combined',
                    metadata={**existing.metadata, **result.metadata}
                )
            else:
                combined_results[result.node_id] = RetrievalResult(
                    node_id=result.node_id,
                    node_data=result.node_data,
                    similarity_score=result.similarity_score * 0.3,  # Weight for traversal
                    context_path=result.context_path,
                    retrieval_method='hybrid_traversal',
                    metadata=result.metadata
                )
                
        # Sort by combined score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return final_results[:top_k]
        
    def generate_response(self, query: str, retrieved_nodes: List[RetrievalResult], **kwargs) -> RAGResponse:
        """Generate response based on retrieved knowledge.
        
        Args:
            query: Original query
            retrieved_nodes: Retrieved relevant nodes
            **kwargs: Additional generation parameters
            
        Returns:
            Complete RAG response
        """
        # Extract relevant information from retrieved nodes
        context_info = self._extract_context_information(retrieved_nodes)
        
        # Generate reasoning path
        reasoning_path = self._generate_reasoning_path(query, retrieved_nodes, context_info)
        
        # Generate response text
        response_text = self._generate_response_text(query, context_info, reasoning_path)
        
        # Calculate confidence score
        confidence_score = self._calculate_response_confidence(retrieved_nodes, context_info)
        
        # Prepare supporting evidence
        supporting_evidence = self._prepare_supporting_evidence(retrieved_nodes)
        
        return RAGResponse(
            query=query,
            retrieved_nodes=retrieved_nodes,
            generated_response=response_text,
            confidence_score=confidence_score,
            reasoning_path=reasoning_path,
            supporting_evidence=supporting_evidence
        )
        
    def _extract_context_information(self, retrieved_nodes: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract structured information from retrieved nodes."""
        context = {
            'ifc_entities': [],
            'regulatory_terms': [],
            'alignments': [],
            'relationships': [],
            'properties': defaultdict(list)
        }
        
        for result in retrieved_nodes:
            node_data = result.node_data
            
            if node_data.get('node_type') == 'ifc_entity':
                context['ifc_entities'].append({
                    'id': result.node_id,
                    'type': node_data.get('label'),
                    'properties': node_data.get('properties', {}),
                    'similarity': result.similarity_score
                })
                
            elif node_data.get('node_type') == 'regulatory_term':
                context['regulatory_terms'].append({
                    'id': result.node_id,
                    'text': node_data.get('properties', {}).get('text'),
                    'category': node_data.get('properties', {}).get('category'),
                    'similarity': result.similarity_score
                })
                
            # Extract relationships from context paths
            for path_info in result.context_path:
                if isinstance(path_info, dict) and 'path' in path_info:
                    for path_step in path_info['path']:
                        if path_step.get('edge_to_next'):
                            context['relationships'].append(path_step['edge_to_next'])
                            
        return context
        
    def _generate_reasoning_path(self, query: str, retrieved_nodes: List[RetrievalResult], context_info: Dict[str, Any]) -> List[str]:
        """Generate step-by-step reasoning path."""
        reasoning_steps = []
        
        # Step 1: Query analysis
        reasoning_steps.append(f"Analyzing query: '{query}'")
        
        # Step 2: Retrieved entities
        if context_info['ifc_entities']:
            ifc_types = [entity['type'] for entity in context_info['ifc_entities']]
            reasoning_steps.append(f"Found relevant IFC entities: {', '.join(set(ifc_types))}")
            
        if context_info['regulatory_terms']:
            reg_categories = [term['category'] for term in context_info['regulatory_terms'] if term['category']]
            reasoning_steps.append(f"Found relevant regulatory terms in categories: {', '.join(set(reg_categories))}")
            
        # Step 3: Semantic alignments
        alignment_count = len([node for node in retrieved_nodes if 'alignment' in node.retrieval_method])
        if alignment_count > 0:
            reasoning_steps.append(f"Identified {alignment_count} semantic alignments between IFC and regulatory domains")
            
        # Step 4: Relationship analysis
        if context_info['relationships']:
            rel_types = [rel.get('relation_type') for rel in context_info['relationships']]
            reasoning_steps.append(f"Analyzed relationships: {', '.join(set(rel_types))}")
            
        return reasoning_steps
        
    def _generate_response_text(self, query: str, context_info: Dict[str, Any], reasoning_path: List[str]) -> str:
        """Generate natural language response."""
        response_parts = []
        
        # Introduction
        response_parts.append(f"Based on the knowledge graph analysis for '{query}':")
        
        # IFC entities information
        if context_info['ifc_entities']:
            response_parts.append("\n**IFC Entities:**")
            for entity in context_info['ifc_entities'][:3]:  # Top 3
                response_parts.append(f"- {entity['type']} (similarity: {entity['similarity']:.2f})")
                
        # Regulatory terms information
        if context_info['regulatory_terms']:
            response_parts.append("\n**Regulatory Terms:**")
            for term in context_info['regulatory_terms'][:3]:  # Top 3
                response_parts.append(f"- {term['text']} (category: {term['category']}, similarity: {term['similarity']:.2f})")
                
        # Semantic alignments
        alignments = context_info.get('alignments', [])
        if alignments:
            response_parts.append("\n**Semantic Alignments:**")
            for alignment in alignments[:3]:
                response_parts.append(f"- {alignment}")
                
        # Relationships
        if context_info['relationships']:
            response_parts.append("\n**Key Relationships:**")
            unique_relations = list(set([rel.get('relation_type') for rel in context_info['relationships']]))
            for rel_type in unique_relations[:3]:
                response_parts.append(f"- {rel_type}")
                
        return "\n".join(response_parts)
        
    def _calculate_response_confidence(self, retrieved_nodes: List[RetrievalResult], context_info: Dict[str, Any]) -> float:
        """Calculate confidence score for the response."""
        if not retrieved_nodes:
            return 0.0
            
        # Average similarity score
        avg_similarity = sum(node.similarity_score for node in retrieved_nodes) / len(retrieved_nodes)
        
        # Diversity bonus (different node types)
        node_types = set(node.node_data.get('node_type') for node in retrieved_nodes)
        diversity_bonus = min(len(node_types) * 0.1, 0.3)
        
        # Relationship bonus
        relationship_bonus = min(len(context_info['relationships']) * 0.05, 0.2)
        
        confidence = avg_similarity + diversity_bonus + relationship_bonus
        return min(confidence, 1.0)
        
    def _prepare_supporting_evidence(self, retrieved_nodes: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Prepare supporting evidence for the response."""
        evidence = []
        
        for node in retrieved_nodes:
            evidence.append({
                'node_id': node.node_id,
                'node_type': node.node_data.get('node_type'),
                'label': node.node_data.get('label'),
                'similarity_score': node.similarity_score,
                'retrieval_method': node.retrieval_method,
                'key_properties': self._extract_key_properties(node.node_data)
            })
            
        return evidence
        
    def _extract_key_properties(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key properties from node data."""
        properties = node_data.get('properties', {})
        key_props = {}
        
        # Extract important properties based on node type
        if node_data.get('node_type') == 'ifc_entity':
            for key in ['ifc_type', 'attributes', 'geometry']:
                if key in properties:
                    key_props[key] = properties[key]
                    
        elif node_data.get('node_type') == 'regulatory_term':
            for key in ['text', 'category', 'context']:
                if key in properties:
                    key_props[key] = properties[key]
                    
        return key_props
        
    def query(self, query: str, strategy: str = 'hybrid', top_k: int = 5, generate_response: bool = True, **kwargs) -> RAGResponse:
        """Complete RAG query pipeline.
        
        Args:
            query: Search query
            strategy: Retrieval strategy
            top_k: Number of nodes to retrieve
            generate_response: Whether to generate natural language response
            **kwargs: Additional parameters
            
        Returns:
            Complete RAG response
        """
        # Retrieve relevant nodes
        retrieved_nodes = self.retrieve(query, strategy, top_k, **kwargs)
        
        if generate_response:
            # Generate complete response
            return self.generate_response(query, retrieved_nodes, **kwargs)
        else:
            # Return minimal response with just retrieval results
            return RAGResponse(
                query=query,
                retrieved_nodes=retrieved_nodes,
                generated_response="",
                confidence_score=0.0,
                reasoning_path=[],
                supporting_evidence=[]
            )