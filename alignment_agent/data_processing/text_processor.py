"""Text data processor for regulatory documents."""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    import spacy
    from sentence_transformers import SentenceTransformer
except ImportError:
    spacy = None
    SentenceTransformer = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from utils.config_loader import get_config
from utils.logger import get_logger
from utils.file_utils import FileUtils

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """Text chunk data container."""
    id: str
    content: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class NamedEntity:
    """Named entity data container."""
    text: str
    label: str
    start: int
    end: int
    confidence: float


@dataclass
class Relationship:
    """Relationship data container."""
    subject: str
    predicate: str
    object: str
    confidence: float
    context: str


class TextProcessor:
    """Text processor for regulatory documents and building codes.
    
    This processor handles various text formats and extracts:
    - Text chunks for semantic processing
    - Named entities (building components, regulations, etc.)
    - Relationships between entities
    - Regulatory terms and definitions
    - Semantic embeddings
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize text processor.
        
        Args:
            config: Optional configuration override
        """
        self.config = get_config()
        if config:
            self.config.update('text_processor', config)
        
        # Get configuration
        self.chunk_size = self.config.get('text_processing.chunk_size', 512)
        self.chunk_overlap = self.config.get('text_processing.chunk_overlap', 50)
        self.supported_formats = self.config.get('text_processing.supported_formats', ['pdf', 'txt', 'md', 'docx'])
        
        # Initialize NLP models
        self.nlp_model = None
        self.embedding_model = None
        self._initialize_models()
        
        # Regulatory term patterns
        self.regulatory_patterns = self._load_regulatory_patterns()
        
        logger.info("Text Processor initialized")
    
    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            if spacy is not None:
                # Try to load English model
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("English spaCy model not found. NER will be limited.")
                    self.nlp_model = None
            
            if SentenceTransformer is not None:
                model_name = self.config.get('embedding.model_name', 'sentence-transformers/all-MiniLM-L6-v2')
                try:
                    self.embedding_model = SentenceTransformer(model_name)
                    logger.info(f"Loaded embedding model: {model_name}")
                except Exception as e:
                    logger.warning(f"Could not load embedding model: {e}")
                    self.embedding_model = None
            
        except Exception as e:
            logger.warning(f"Error initializing NLP models: {e}")
    
    def _load_regulatory_patterns(self) -> Dict[str, List[str]]:
        """Load regulatory term patterns.
        
        Returns:
            Dictionary of regulatory patterns by category
        """
        patterns = {
            'building_components': [
                r'\b(?:wall|partition|load[\-\s]bearing\s+wall)\b',
                r'\b(?:slab|floor|platform|terrace|balcony)\b',
                r'\b(?:column|pillar|support|structural\s+column)\b',
                r'\b(?:beam|girder|structural\s+beam)\b',
                r'\b(?:door|entrance|exit|opening)\b',
                r'\b(?:window|glazing|fenestration)\b',
                r'\b(?:space|room|area|zone|compartment)\b',
                r'\b(?:shaft|atrium|void|opening\s+element)\b',
                r'\b(?:stair|staircase|steps|stairway)\b',
                r'\b(?:roof|roofing|cover|ceiling)\b'
            ],
            'dimensions': [
                r'\b(?:height|width|length|thickness|depth)\b',
                r'\b(?:area|volume|capacity)\b',
                r'\b(?:minimum|maximum|min|max)\s+(?:height|width|length|thickness|depth|area|volume)\b',
                r'\d+(?:\.\d+)?\s*(?:mm|cm|m|ft|in)\b'
            ],
            'materials': [
                r'\b(?:concrete|steel|wood|timber|brick|stone)\b',
                r'\b(?:reinforced\s+concrete|structural\s+steel)\b',
                r'\b(?:insulation|thermal\s+insulation|acoustic\s+insulation)\b',
                r'\b(?:fire\s+resistant|fire\s+rated|fireproof)\b'
            ],
            'regulations': [
                r'\b(?:building\s+code|fire\s+code|safety\s+code)\b',
                r'\b(?:requirement|regulation|standard|specification)\b',
                r'\b(?:shall|must|should|required|mandatory)\b',
                r'\b(?:comply|compliance|conformance|accordance)\b'
            ],
            'spatial_relationships': [
                r'\b(?:located\s+in|contained\s+in|within|inside)\b',
                r'\b(?:adjacent\s+to|next\s+to|beside|neighboring)\b',
                r'\b(?:above|below|over|under|beneath)\b',
                r'\b(?:connected\s+to|attached\s+to|joined\s+to)\b'
            ]
        }
        
        return patterns
    
    def process_text_file(self, file_path: str) -> Dict[str, Any]:
        """Process text file and extract semantic information.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Dictionary containing processed text data
        """
        logger.info(f"Processing text file: {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Handle JSON format regulatory text
            if file_ext == '.json':
                return self._process_json_regulatory_text(file_path)
            
            # Read text content for other formats
            text_content = self._read_text_file(file_path)
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(text_content)
            
            # Create text chunks
            chunks = self._create_text_chunks(cleaned_text)
            
            # Extract named entities
            entities = self._extract_named_entities(cleaned_text)
            
            # Extract relationships
            relationships = self._extract_relationships(cleaned_text)
            
            # Extract regulatory terms
            regulatory_terms = self._extract_regulatory_terms(cleaned_text)
            
            # Generate embeddings
            embeddings = self._generate_embeddings(chunks)
            
            text_data = {
                'file_path': file_path,
                'original_text': text_content,
                'cleaned_text': cleaned_text,
                'chunks': [chunk.__dict__ for chunk in chunks],
                'entities': [entity.__dict__ for entity in entities],
                'relationships': [rel.__dict__ for rel in relationships],
                'regulatory_terms': regulatory_terms,
                'embeddings': embeddings,
                'chunk_count': len(chunks),
                'entity_count': len(entities),
                'processing_timestamp': self._get_timestamp()
            }
            
            logger.info(f"Text processing completed: {len(chunks)} chunks, {len(entities)} entities")
            return text_data
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise
    
    def _read_text_file(self, file_path: str) -> str:
        """Read text content from file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Text content as string
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.txt':
            return FileUtils.read_text(file_path)
        elif file_ext == '.md':
            return FileUtils.read_text(file_path)
        elif file_ext == '.pdf':
            return self._read_pdf(file_path)
        elif file_ext == '.docx':
            return self._read_docx(file_path)
        else:
            # Try to read as plain text
            return FileUtils.read_text(file_path)
    
    def _process_json_regulatory_text(self, file_path: str) -> Dict[str, Any]:
        """Process JSON format regulatory text.
        
        Args:
            file_path: Path to JSON file containing regulatory text
            
        Returns:
            Dictionary containing processed regulatory data
        """
        import json
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Extract text content from JSON structure
            text_content = self._extract_text_from_json(json_data)
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(text_content)
            
            # Create text chunks
            chunks = self._create_text_chunks(cleaned_text)
            
            # Extract named entities
            entities = self._extract_named_entities(cleaned_text)
            
            # Extract relationships
            relationships = self._extract_relationships(cleaned_text)
            
            # Extract regulatory terms
            regulatory_terms = self._extract_regulatory_terms(cleaned_text)
            
            # Generate embeddings
            embeddings = self._generate_embeddings(chunks)
            
            # Extract structured regulatory data from JSON
            structured_regulations = self._extract_structured_regulations(json_data)
            
            text_data = {
                'file_path': file_path,
                'format': 'json',
                'original_json': json_data,
                'original_text': text_content,
                'cleaned_text': cleaned_text,
                'chunks': [chunk.__dict__ for chunk in chunks],
                'entities': [entity.__dict__ for entity in entities],
                'relationships': [rel.__dict__ for rel in relationships],
                'regulatory_terms': regulatory_terms,
                'structured_regulations': structured_regulations,
                'embeddings': embeddings,
                'chunk_count': len(chunks),
                'entity_count': len(entities),
                'regulation_count': len(structured_regulations),
                'processing_timestamp': self._get_timestamp()
            }
            
            logger.info(f"JSON regulatory text processing completed: {len(structured_regulations)} regulations, {len(chunks)} chunks, {len(entities)} entities")
            return text_data
            
        except Exception as e:
            logger.error(f"Error processing JSON regulatory file: {e}")
            raise
    
    def _extract_text_from_json(self, json_data: Dict[str, Any]) -> str:
        """Extract text content from JSON regulatory data.
        
        Args:
            json_data: JSON data containing regulatory information
            
        Returns:
            Concatenated text content
        """
        text_parts = []
        
        def extract_text_recursive(obj, prefix="", context=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Handle specific keys that contain text content
                    if key in ['title', 'description', 'content', 'text', 'requirement', 'regulation', 'evidence']:
                        if isinstance(value, str) and value.strip():
                            text_parts.append(f"{context}{key}: {value.strip()}")
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(f"{context}{key}: {item['text']}")
                                elif isinstance(item, str) and item.strip():
                                    text_parts.append(f"{context}{key}: {item.strip()}")
                    # Handle height rules and other structured data
                    elif key == 'height_rules' and isinstance(value, list):
                        for rule in value:
                            if isinstance(rule, dict):
                                label = rule.get('label', 'rule')
                                comparator = rule.get('comparator', '')
                                threshold_min = rule.get('threshold_min_m', '')
                                threshold_max = rule.get('threshold_max_m', '')
                                unit = rule.get('unit_normalized', 'm')
                                
                                rule_text = f"Height rule ({label}): {comparator} {threshold_min}-{threshold_max} {unit}"
                                text_parts.append(f"{context}height_rule: {rule_text}")
                                
                                # Extract evidence text
                                if 'evidence' in rule and isinstance(rule['evidence'], list):
                                    for evidence in rule['evidence']:
                                        if isinstance(evidence, dict) and 'text' in evidence:
                                            text_parts.append(f"{context}evidence: {evidence['text']}")
                    # Handle area calculation rules
                    elif key in ['area_calculation_rules', 'rules'] and isinstance(value, list):
                        for rule in value:
                            if isinstance(rule, dict):
                                rule_type = rule.get('type', rule.get('label', 'rule'))
                                description = rule.get('description', rule.get('text', ''))
                                if description:
                                    text_parts.append(f"{context}rule ({rule_type}): {description}")
                    # Handle region-specific data
                    elif key == 'per_region' and isinstance(value, dict):
                        for region, region_data in value.items():
                            region_context = f"{context}region_{region} - "
                            extract_text_recursive(region_data, f"{prefix}{key}.", region_context)
                    # Handle other nested structures
                    elif isinstance(value, (dict, list)):
                        new_context = f"{context}{key} - " if context else f"{key} - "
                        extract_text_recursive(value, f"{prefix}{key}.", new_context)
                    # Handle simple string values
                    elif isinstance(value, str) and value.strip() and key not in ['region', 'source_name', 'source_path', 'unit_normalized', 'comparator']:
                        text_parts.append(f"{context}{key}: {value.strip()}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_text_recursive(item, f"{prefix}[{i}]", f"{context}[{i}] ")
            elif isinstance(obj, str) and obj.strip():
                text_parts.append(f"{context}{obj.strip()}")
        
        extract_text_recursive(json_data)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in text_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)
        
        return "\n".join(unique_parts)
    
    def _extract_structured_regulations(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract structured regulatory information from JSON.
        
        Args:
            json_data: JSON data containing regulatory information
            
        Returns:
            List of structured regulation objects
        """
        regulations = []
        
        def extract_regulations_recursive(obj, path=""):
            if isinstance(obj, dict):
                # Check if this is a regulation object
                if any(key in obj for key in ['regulation_id', 'code', 'section', 'requirement']):
                    regulation = {
                        'id': obj.get('regulation_id', obj.get('id', f"reg_{len(regulations)}")),
                        'title': obj.get('title', obj.get('name', '')),
                        'content': obj.get('content', obj.get('text', obj.get('requirement', ''))),
                        'section': obj.get('section', obj.get('code', '')),
                        'category': obj.get('category', obj.get('type', 'general')),
                        'mandatory': obj.get('mandatory', obj.get('required', True)),
                        'path': path,
                        'metadata': {k: v for k, v in obj.items() if k not in ['regulation_id', 'id', 'title', 'name', 'content', 'text', 'requirement', 'section', 'code', 'category', 'type', 'mandatory', 'required']}
                    }
                    regulations.append(regulation)
                else:
                    # Recursively search in nested objects
                    for key, value in obj.items():
                        new_path = f"{path}.{key}" if path else key
                        extract_regulations_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    extract_regulations_recursive(item, new_path)
        
        extract_regulations_recursive(json_data)
        return regulations
    
    def _read_pdf(self, file_path: str) -> str:
        """Read text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                return text_content
                
        except ImportError:
            logger.warning("PyPDF2 not available. Cannot read PDF files.")
            return f"PDF file: {file_path} (content not extracted)"
        except Exception as e:
            logger.warning(f"Error reading PDF file: {e}")
            return f"PDF file: {file_path} (error reading content)"
    
    def _read_docx(self, file_path: str) -> str:
        """Read text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text content
        """
        try:
            import docx
            
            doc = docx.Document(file_path)
            text_content = ""
            
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            return text_content
            
        except ImportError:
            logger.warning("python-docx not available. Cannot read DOCX files.")
            return f"DOCX file: {file_path} (content not extracted)"
        except Exception as e:
            logger.warning(f"Error reading DOCX file: {e}")
            return f"DOCX file: {file_path} (error reading content)"
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]"\']', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _create_text_chunks(self, text: str) -> List[TextChunk]:
        """Create text chunks for processing.
        
        Args:
            text: Cleaned text content
            
        Returns:
            List of text chunks
        """
        chunks = []
        text_length = len(text)
        
        start = 0
        chunk_id = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence ending within overlap range
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size - self.chunk_overlap:
                    end = sentence_end + 1
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunk = TextChunk(
                    id=f"chunk_{chunk_id:04d}",
                    content=chunk_content,
                    start_pos=start,
                    end_pos=end,
                    metadata={
                        'length': len(chunk_content),
                        'word_count': len(chunk_content.split())
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        return chunks
    
    def _extract_named_entities(self, text: str) -> List[NamedEntity]:
        """Extract named entities from text.
        
        Args:
            text: Text content
            
        Returns:
            List of named entities
        """
        entities = []
        
        if self.nlp_model is not None:
            try:
                doc = self.nlp_model(text)
                
                for ent in doc.ents:
                    entity = NamedEntity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=1.0  # spaCy doesn't provide confidence scores by default
                    )
                    entities.append(entity)
                    
            except Exception as e:
                logger.warning(f"Error extracting named entities: {e}")
        
        # Extract entities using regex patterns
        regex_entities = self._extract_entities_with_regex(text)
        entities.extend(regex_entities)
        
        return entities
    
    def _extract_entities_with_regex(self, text: str) -> List[NamedEntity]:
        """Extract entities using regex patterns.
        
        Args:
            text: Text content
            
        Returns:
            List of named entities
        """
        entities = []
        
        for category, patterns in self.regulatory_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity = NamedEntity(
                        text=match.group(),
                        label=category.upper(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8  # Default confidence for regex matches
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_relationships(self, text: str) -> List[Relationship]:
        """Extract relationships from text.
        
        Args:
            text: Text content
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Simple relationship extraction using patterns
        relationship_patterns = [
            (r'(\w+)\s+(?:is|are)\s+(?:located|situated|positioned)\s+(?:in|within|inside)\s+(\w+)', 'LOCATED_IN'),
            (r'(\w+)\s+(?:contains|includes|comprises)\s+(\w+)', 'CONTAINS'),
            (r'(\w+)\s+(?:adjacent|next)\s+to\s+(\w+)', 'ADJACENT_TO'),
            (r'(\w+)\s+(?:above|over)\s+(\w+)', 'ABOVE'),
            (r'(\w+)\s+(?:below|under|beneath)\s+(\w+)', 'BELOW'),
            (r'(\w+)\s+(?:connected|attached|joined)\s+to\s+(\w+)', 'CONNECTED_TO')
        ]
        
        for pattern, relation_type in relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                subject = match.group(1)
                obj = match.group(2)
                context = text[max(0, match.start()-50):match.end()+50]
                
                relationship = Relationship(
                    subject=subject,
                    predicate=relation_type,
                    object=obj,
                    confidence=0.7,
                    context=context.strip()
                )
                relationships.append(relationship)
        
        return relationships
    
    def _extract_regulatory_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract regulatory terms and definitions.
        
        Args:
            text: Text content
            
        Returns:
            List of regulatory terms with metadata
        """
        regulatory_terms = []
        
        # Extract terms by category
        for category, patterns in self.regulatory_patterns.items():
            category_terms = set()
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    term = match.group().strip()
                    if len(term) > 2:  # Filter out very short matches
                        category_terms.add(term.lower())
            
            for term in category_terms:
                regulatory_terms.append({
                    'term': term,
                    'category': category,
                    'frequency': text.lower().count(term),
                    'contexts': self._find_term_contexts(text, term)
                })
        
        # Sort by frequency
        regulatory_terms.sort(key=lambda x: x['frequency'], reverse=True)
        
        return regulatory_terms
    
    def _find_term_contexts(self, text: str, term: str, context_window: int = 100) -> List[str]:
        """Find contexts where a term appears.
        
        Args:
            text: Text content
            term: Term to find
            context_window: Context window size
            
        Returns:
            List of context strings
        """
        contexts = []
        text_lower = text.lower()
        term_lower = term.lower()
        
        start = 0
        while True:
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            
            context_start = max(0, pos - context_window)
            context_end = min(len(text), pos + len(term) + context_window)
            context = text[context_start:context_end].strip()
            
            contexts.append(context)
            start = pos + 1
        
        return contexts[:5]  # Limit to first 5 contexts
    
    def _generate_embeddings(self, chunks: List[TextChunk]) -> List[np.ndarray]:
        """Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        if self.embedding_model is not None:
            try:
                texts = [chunk.content for chunk in chunks]
                chunk_embeddings = self.embedding_model.encode(texts)
                
                for i, embedding in enumerate(chunk_embeddings):
                    chunks[i].embedding = embedding
                    embeddings.append(embedding)
                    
            except Exception as e:
                logger.warning(f"Error generating embeddings: {e}")
        
        return embeddings
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def search_similar_chunks(self, text_data: Dict[str, Any], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar text chunks using embeddings.
        
        Args:
            text_data: Processed text data
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks with similarity scores
        """
        if self.embedding_model is None:
            logger.warning("Embedding model not available for similarity search")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Get chunk embeddings
            chunks = text_data.get('chunks', [])
            similarities = []
            
            for chunk in chunks:
                if 'embedding' in chunk and chunk['embedding'] is not None:
                    chunk_embedding = np.array(chunk['embedding'])
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    similarities.append({
                        'chunk': chunk,
                        'similarity': float(similarity)
                    })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_entities_by_category(self, text_data: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
        """Get entities by category.
        
        Args:
            text_data: Processed text data
            category: Entity category
            
        Returns:
            List of entities in specified category
        """
        entities = text_data.get('entities', [])
        return [entity for entity in entities if entity.get('label', '').upper() == category.upper()]
    
    def get_relationships_by_type(self, text_data: Dict[str, Any], relation_type: str) -> List[Dict[str, Any]]:
        """Get relationships by type.
        
        Args:
            text_data: Processed text data
            relation_type: Relationship type
            
        Returns:
            List of relationships of specified type
        """
        relationships = text_data.get('relationships', [])
        return [rel for rel in relationships if rel.get('predicate', '').upper() == relation_type.upper()]
    
    def process_json_regulations(self, regulations_data: Dict[str, Any]) -> str:
        """Convert JSON regulations data to text format for processing.
        
        Args:
            regulations_data: JSON data containing regulatory information
            
        Returns:
            Formatted text string containing regulatory content
        """
        text_parts = []
        
        # Handle per_region format (like our output-1.json)
        if 'per_region' in regulations_data:
            text_parts.append("Building Regulations by Region")
            text_parts.append("=" * 40)
            text_parts.append("")
            
            for region_code, region_data in regulations_data['per_region'].items():
                text_parts.append(f"Region: {region_code}")
                text_parts.append(f"Source: {region_data.get('source_name', 'Unknown')}")
                text_parts.append("")
                
                # Process height rules
                if 'height_rules' in region_data:
                    text_parts.append("Height Rules:")
                    for rule in region_data['height_rules']:
                        label = rule.get('label', 'rule')
                        comparator = rule.get('comparator', '')
                        threshold_min = rule.get('threshold_min_m', '')
                        threshold_max = rule.get('threshold_max_m', '')
                        unit = rule.get('unit_normalized', 'm')
                        
                        # Create rule description
                        if comparator == '>=':
                            rule_desc = f"Height >= {threshold_min}{unit} ({label})"
                        elif comparator == '<':
                            rule_desc = f"Height < {threshold_max}{unit} ({label})"
                        elif comparator == 'between':
                            rule_desc = f"Height between {threshold_min}{unit} and {threshold_max}{unit} ({label})"
                        else:
                            rule_desc = f"Height rule ({label}): {comparator} {threshold_min}-{threshold_max} {unit}"
                        
                        text_parts.append(f"  {rule_desc}")
                        
                        # Add evidence text
                        if 'evidence' in rule and isinstance(rule['evidence'], list):
                            for evidence in rule['evidence']:
                                if isinstance(evidence, dict) and 'text' in evidence:
                                    text_parts.append(f"    Evidence: {evidence['text']}")
                    text_parts.append("")
                
                # Process area calculation rules
                if 'area_calculation_rules' in region_data:
                    text_parts.append("Area Calculation Rules:")
                    for rule in region_data['area_calculation_rules']:
                        rule_type = rule.get('type', rule.get('label', 'rule'))
                        description = rule.get('description', rule.get('text', ''))
                        if description:
                            text_parts.append(f"  {rule_type}: {description}")
                    text_parts.append("")
        
        # Handle standard document format
        else:
            # Add document info
            if 'document_info' in regulations_data:
                doc_info = regulations_data['document_info']
                text_parts.append(f"Document: {doc_info.get('title', 'Unknown')}")
                text_parts.append(f"Version: {doc_info.get('version', 'Unknown')}")
                text_parts.append(f"Jurisdiction: {doc_info.get('jurisdiction', 'Unknown')}")
                text_parts.append("")
            
            # Process sections
            if 'sections' in regulations_data:
                for section in regulations_data['sections']:
                    section_id = section.get('section_id', '')
                    title = section.get('title', '')
                    content = section.get('content', '')
                    
                    text_parts.append(f"Section {section_id}: {title}")
                    text_parts.append(content)
                    
                    # Add requirements
                    if 'requirements' in section:
                        for req in section['requirements']:
                            req_id = req.get('requirement_id', '')
                            description = req.get('description', '')
                            specification = req.get('specification', '')
                            
                            text_parts.append(f"  Requirement {req_id}: {description}")
                            text_parts.append(f"  Specification: {specification}")
                    
                    text_parts.append("")
            
            # Add definitions
            if 'definitions' in regulations_data:
                text_parts.append("Definitions:")
                for term, definition in regulations_data['definitions'].items():
                    text_parts.append(f"  {term}: {definition}")
                text_parts.append("")
        
        return "\n".join(text_parts)