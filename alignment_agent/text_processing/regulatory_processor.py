#!/usr/bin/env python3
"""Regulatory text processing module for IFC Semantic Agent."""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import defaultdict


@dataclass
class RegulatorySection:
    """Represents a section of regulatory text."""
    id: str
    title: str
    content: str
    section_type: str  # 'article', 'clause', 'paragraph', 'table', 'figure'
    level: int  # Hierarchical level (1, 2, 3, etc.)
    parent_id: Optional[str] = None
    subsections: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulatoryRequirement:
    """Represents a specific regulatory requirement."""
    id: str
    text: str
    requirement_type: str  # 'mandatory', 'recommended', 'prohibited', 'conditional'
    category: str  # 'structural', 'fire_safety', 'accessibility', 'energy', etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source_section: Optional[str] = None


@dataclass
class RegulatoryDocument:
    """Represents a complete regulatory document."""
    id: str
    title: str
    version: str
    effective_date: Optional[str] = None
    authority: Optional[str] = None
    scope: Optional[str] = None
    sections: List[RegulatorySection] = field(default_factory=list)
    requirements: List[RegulatoryRequirement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegulatoryProcessor:
    """Advanced processor for regulatory documents and texts."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the regulatory processor."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load processing patterns and rules
        self.section_patterns = self._load_section_patterns()
        self.requirement_patterns = self._load_requirement_patterns()
        self.keyword_patterns = self._load_keyword_patterns()
        self.reference_patterns = self._load_reference_patterns()
        
        # Initialize document cache
        self.document_cache: Dict[str, RegulatoryDocument] = {}
        
        # Load predefined regulatory categories
        self.regulatory_categories = self._load_regulatory_categories()
        
        self.logger.info("RegulatoryProcessor initialized")
    
    def _load_section_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying document sections."""
        return {
            'article': [
                r'第\s*([\d一二三四五六七八九十]+)\s*条',
                r'Article\s+(\d+)',
                r'条文\s*([\d.]+)'
            ],
            'clause': [
                r'第\s*([\d.]+)\s*款',
                r'Clause\s+([\d.]+)',
                r'款项\s*([\d.]+)'
            ],
            'paragraph': [
                r'第\s*([\d.]+)\s*项',
                r'\((\d+)\)',
                r'([\d.]+)\s*、'
            ],
            'chapter': [
                r'第\s*([\d一二三四五六七八九十]+)\s*章',
                r'Chapter\s+(\d+)',
                r'章节\s*([\d.]+)'
            ],
            'section': [
                r'第\s*([\d.]+)\s*节',
                r'Section\s+([\d.]+)',
                r'节\s*([\d.]+)'
            ]
        }
    
    def _load_requirement_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying regulatory requirements."""
        return {
            'mandatory': [
                r'应当|必须|应该|须|不得|禁止|严禁',
                r'shall|must|required|mandatory',
                r'不应|不宜|不允许'
            ],
            'recommended': [
                r'宜|建议|推荐|可以|最好',
                r'should|recommended|advisable|preferable',
                r'可采用|可选用'
            ],
            'prohibited': [
                r'不得|禁止|严禁|不允许|不应',
                r'shall not|must not|prohibited|forbidden',
                r'不可|不能'
            ],
            'conditional': [
                r'当.*时|如果.*则|在.*情况下',
                r'when|if|where|provided that',
                r'除.*外|但.*时'
            ]
        }
    
    def _load_keyword_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying key regulatory concepts."""
        return {
            'structural': [
                r'承重墙|剪力墙|框架|梁|柱|楼板|基础',
                r'结构|荷载|抗震|强度|刚度|稳定性',
                r'钢筋|混凝土|钢结构|木结构'
            ],
            'fire_safety': [
                r'防火|耐火|阻燃|防烟|排烟|疏散',
                r'消防|灭火|火灾|燃烧|耐火极限',
                r'防火墙|防火门|安全出口|疏散楼梯'
            ],
            'accessibility': [
                r'无障碍|残疾人|轮椅|坡道|扶手',
                r'通道宽度|门槛|台阶|电梯',
                r'盲道|语音提示|标识'
            ],
            'energy': [
                r'节能|保温|隔热|传热系数|热桥',
                r'能耗|绿色建筑|可再生能源',
                r'外墙|屋面|门窗|遮阳'
            ],
            'dimensions': [
                r'\d+(?:\.\d+)?\s*(?:mm|cm|m|米|毫米|厘米)',
                r'不小于|不大于|不超过|不少于',
                r'宽度|高度|厚度|长度|面积|体积'
            ],
            'materials': [
                r'混凝土|钢材|木材|砖|石材|玻璃',
                r'防水材料|保温材料|装饰材料',
                r'等级|强度|性能|质量'
            ]
        }
    
    def _load_reference_patterns(self) -> List[str]:
        """Load patterns for identifying document references."""
        return [
            r'GB\s*/?\s*T?\s*\d+(?:[.-]\d+)*',  # Chinese national standards
            r'JGJ\s*\d+(?:[.-]\d+)*',  # Construction industry standards
            r'CJJ\s*\d+(?:[.-]\d+)*',  # Urban construction standards
            r'DB\s*\d+(?:[/-]\d+)*',  # Local standards
            r'ISO\s*\d+(?:[:-]\d+)*',  # International standards
            r'EN\s*\d+(?:[:-]\d+)*',   # European standards
            r'ASTM\s*[A-Z]\d+(?:[.-]\d+)*',  # ASTM standards
            r'第\s*[\d一二三四五六七八九十]+\s*条',  # Article references
            r'附录\s*[A-Z]',  # Appendix references
        ]
    
    def _load_regulatory_categories(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined regulatory categories and their characteristics."""
        return {
            'building_code': {
                'keywords': ['建筑设计', '建筑规范', '设计标准'],
                'typical_requirements': ['structural', 'fire_safety', 'accessibility'],
                'common_sections': ['general', 'structural', 'fire', 'accessibility']
            },
            'fire_code': {
                'keywords': ['防火规范', '消防', '火灾'],
                'typical_requirements': ['fire_safety', 'evacuation', 'fire_resistance'],
                'common_sections': ['fire_prevention', 'evacuation', 'fire_systems']
            },
            'structural_code': {
                'keywords': ['结构设计', '荷载规范', '抗震'],
                'typical_requirements': ['structural', 'seismic', 'load_bearing'],
                'common_sections': ['loads', 'materials', 'design_methods']
            },
            'energy_code': {
                'keywords': ['节能', '绿色建筑', '能耗'],
                'typical_requirements': ['energy', 'thermal', 'renewable'],
                'common_sections': ['envelope', 'systems', 'renewable_energy']
            },
            'accessibility_code': {
                'keywords': ['无障碍', '残疾人', '通用设计'],
                'typical_requirements': ['accessibility', 'universal_design'],
                'common_sections': ['access_routes', 'facilities', 'signage']
            }
        }
    
    def process_document(self, text: str, document_id: str = None, metadata: Dict[str, Any] = None) -> RegulatoryDocument:
        """Process a complete regulatory document."""
        self.logger.info(f"Processing regulatory document: {document_id}")
        
        # Extract document metadata
        doc_metadata = self._extract_document_metadata(text)
        if metadata:
            doc_metadata.update(metadata)
        
        # Create document object
        document = RegulatoryDocument(
            id=document_id or f"doc_{hash(text[:100])}"[:8],
            title=doc_metadata.get('title', 'Unknown Document'),
            version=doc_metadata.get('version', '1.0'),
            effective_date=doc_metadata.get('effective_date'),
            authority=doc_metadata.get('authority'),
            scope=doc_metadata.get('scope'),
            metadata=doc_metadata
        )
        
        # Parse document structure
        sections = self._parse_document_structure(text)
        document.sections = sections
        
        # Extract requirements from all sections
        all_requirements = []
        for section in sections:
            requirements = self._extract_requirements_from_section(section)
            all_requirements.extend(requirements)
        
        document.requirements = all_requirements
        
        # Cache the document
        self.document_cache[document.id] = document
        
        self.logger.info(f"Document processed: {len(sections)} sections, {len(all_requirements)} requirements")
        return document
    
    def _extract_document_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from document text."""
        metadata = {}
        
        # Extract title (usually first significant line)
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and not re.match(r'^\d+', line):
                metadata['title'] = line
                break
        
        # Extract standard numbers
        standard_matches = re.findall(r'GB\s*/?\s*T?\s*\d+(?:[.-]\d+)*', text)
        if standard_matches:
            metadata['standard_number'] = standard_matches[0]
        
        # Extract version/year
        year_matches = re.findall(r'(19|20)\d{2}', text)
        if year_matches:
            metadata['year'] = year_matches[-1]  # Use the last year found
        
        # Determine document category
        text_lower = text.lower()
        for category, info in self.regulatory_categories.items():
            for keyword in info['keywords']:
                if keyword in text:
                    metadata['category'] = category
                    break
            if 'category' in metadata:
                break
        
        return metadata
    
    def _parse_document_structure(self, text: str) -> List[RegulatorySection]:
        """Parse document into hierarchical sections."""
        sections = []
        lines = text.split('\n')
        
        current_section = None
        section_counter = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            section_info = self._identify_section_header(line)
            
            if section_info:
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)
                
                # Create new section
                section_counter += 1
                current_section = RegulatorySection(
                    id=f"section_{section_counter:03d}",
                    title=section_info['title'],
                    content="",
                    section_type=section_info['type'],
                    level=section_info['level'],
                    metadata={'line_number': i + 1}
                )
            elif current_section:
                # Add content to current section
                if current_section.content:
                    current_section.content += "\n"
                current_section.content += line
            else:
                # Create a default section for content without headers
                if not sections:
                    current_section = RegulatorySection(
                        id="section_001",
                        title="Preamble",
                        content=line,
                        section_type="paragraph",
                        level=1
                    )
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        # Post-process sections
        self._post_process_sections(sections)
        
        return sections
    
    def _identify_section_header(self, line: str) -> Optional[Dict[str, Any]]:
        """Identify if a line is a section header and extract information."""
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    # Determine level based on section type
                    level_map = {
                        'chapter': 1,
                        'section': 2,
                        'article': 3,
                        'clause': 4,
                        'paragraph': 5
                    }
                    
                    return {
                        'type': section_type,
                        'level': level_map.get(section_type, 3),
                        'title': line,
                        'number': match.group(1) if match.groups() else None
                    }
        
        return None
    
    def _post_process_sections(self, sections: List[RegulatorySection]):
        """Post-process sections to establish hierarchy and extract additional info."""
        for i, section in enumerate(sections):
            # Extract keywords
            section.keywords = self._extract_keywords(section.content)
            
            # Extract references
            section.references = self._extract_references(section.content)
            
            # Establish parent-child relationships
            if i > 0:
                for j in range(i - 1, -1, -1):
                    if sections[j].level < section.level:
                        section.parent_id = sections[j].id
                        sections[j].subsections.append(section.id)
                        break
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text based on predefined patterns."""
        keywords = []
        
        for category, patterns in self.keyword_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                keywords.extend(matches)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract document references from text."""
        references = []
        
        for pattern in self.reference_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    def _extract_requirements_from_section(self, section: RegulatorySection) -> List[RegulatoryRequirement]:
        """Extract regulatory requirements from a section."""
        requirements = []
        
        # Split section content into sentences
        sentences = re.split(r'[。；;]', section.content)
        
        req_counter = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Determine requirement type
            req_type = self._classify_requirement_type(sentence)
            if not req_type:
                continue  # Not a requirement
            
            # Determine category
            category = self._classify_requirement_category(sentence)
            
            # Extract parameters (numbers, dimensions, etc.)
            parameters = self._extract_requirement_parameters(sentence)
            
            # Extract conditions
            conditions = self._extract_conditions(sentence)
            
            req_counter += 1
            requirement = RegulatoryRequirement(
                id=f"{section.id}_req_{req_counter:03d}",
                text=sentence,
                requirement_type=req_type,
                category=category,
                parameters=parameters,
                conditions=conditions,
                source_section=section.id,
                confidence=self._calculate_requirement_confidence(sentence, req_type)
            )
            
            requirements.append(requirement)
        
        return requirements
    
    def _classify_requirement_type(self, text: str) -> Optional[str]:
        """Classify the type of requirement based on text patterns."""
        for req_type, patterns in self.requirement_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return req_type
        return None
    
    def _classify_requirement_category(self, text: str) -> str:
        """Classify the category of requirement."""
        text_lower = text.lower()
        
        # Score each category
        category_scores = defaultdict(int)
        
        for category, patterns in self.keyword_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                category_scores[category] += matches
        
        # Return category with highest score, or 'general' if no matches
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def _extract_requirement_parameters(self, text: str) -> Dict[str, Any]:
        """Extract numerical parameters from requirement text."""
        parameters = {}
        
        # Extract dimensions
        dimension_pattern = r'(\d+(?:\.\d+)?)\s*(mm|cm|m|米|毫米|厘米)'
        dimensions = re.findall(dimension_pattern, text)
        if dimensions:
            parameters['dimensions'] = [{'value': float(d[0]), 'unit': d[1]} for d in dimensions]
        
        # Extract percentages
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(percentage_pattern, text)
        if percentages:
            parameters['percentages'] = [float(p) for p in percentages]
        
        # Extract ratios
        ratio_pattern = r'(\d+(?:\.\d+)?)\s*[:：]\s*(\d+(?:\.\d+)?)'
        ratios = re.findall(ratio_pattern, text)
        if ratios:
            parameters['ratios'] = [{'numerator': float(r[0]), 'denominator': float(r[1])} for r in ratios]
        
        # Extract temperature values
        temp_pattern = r'(\d+(?:\.\d+)?)\s*[°℃]C?'
        temperatures = re.findall(temp_pattern, text)
        if temperatures:
            parameters['temperatures'] = [float(t) for t in temperatures]
        
        return parameters
    
    def _extract_conditions(self, text: str) -> List[str]:
        """Extract conditional clauses from requirement text."""
        conditions = []
        
        # Common conditional patterns
        conditional_patterns = [
            r'当(.+?)时',
            r'如果(.+?)则',
            r'在(.+?)情况下',
            r'除(.+?)外',
            r'但(.+?)时'
        ]
        
        for pattern in conditional_patterns:
            matches = re.findall(pattern, text)
            conditions.extend(matches)
        
        return conditions
    
    def _calculate_requirement_confidence(self, text: str, req_type: str) -> float:
        """Calculate confidence score for requirement extraction."""
        base_confidence = 0.7
        
        # Boost confidence for clear requirement indicators
        strong_indicators = ['必须', '应当', '不得', 'shall', 'must']
        for indicator in strong_indicators:
            if indicator in text.lower():
                base_confidence += 0.1
                break
        
        # Boost confidence for specific parameters
        if re.search(r'\d+(?:\.\d+)?\s*(?:mm|cm|m|%)', text):
            base_confidence += 0.1
        
        # Reduce confidence for vague language
        vague_terms = ['可能', '一般', '通常', '大约', 'approximately']
        for term in vague_terms:
            if term in text.lower():
                base_confidence -= 0.1
                break
        
        return min(1.0, max(0.1, base_confidence))
    
    def search_requirements(self, query: str, category: str = None, req_type: str = None) -> List[RegulatoryRequirement]:
        """Search for requirements matching query criteria."""
        results = []
        
        for document in self.document_cache.values():
            for requirement in document.requirements:
                # Filter by category if specified
                if category and requirement.category != category:
                    continue
                
                # Filter by requirement type if specified
                if req_type and requirement.requirement_type != req_type:
                    continue
                
                # Check if query matches requirement text
                if query.lower() in requirement.text.lower():
                    results.append(requirement)
        
        # Sort by confidence score
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def get_document_statistics(self, document_id: str) -> Dict[str, Any]:
        """Get statistics for a processed document."""
        if document_id not in self.document_cache:
            return {}
        
        document = self.document_cache[document_id]
        
        # Count requirements by type
        req_type_counts = defaultdict(int)
        for req in document.requirements:
            req_type_counts[req.requirement_type] += 1
        
        # Count requirements by category
        category_counts = defaultdict(int)
        for req in document.requirements:
            category_counts[req.category] += 1
        
        # Calculate average confidence
        avg_confidence = sum(req.confidence for req in document.requirements) / len(document.requirements) if document.requirements else 0
        
        return {
            'document_id': document_id,
            'title': document.title,
            'total_sections': len(document.sections),
            'total_requirements': len(document.requirements),
            'requirement_types': dict(req_type_counts),
            'requirement_categories': dict(category_counts),
            'average_confidence': avg_confidence,
            'processing_metadata': document.metadata
        }
    
    def export_document(self, document_id: str, format_type: str = 'json') -> str:
        """Export processed document in specified format."""
        if document_id not in self.document_cache:
            raise ValueError(f"Document {document_id} not found in cache")
        
        document = self.document_cache[document_id]
        
        if format_type == 'json':
            return json.dumps({
                'document': {
                    'id': document.id,
                    'title': document.title,
                    'version': document.version,
                    'metadata': document.metadata
                },
                'sections': [{
                    'id': section.id,
                    'title': section.title,
                    'type': section.section_type,
                    'level': section.level,
                    'content': section.content,
                    'keywords': section.keywords,
                    'references': section.references
                } for section in document.sections],
                'requirements': [{
                    'id': req.id,
                    'text': req.text,
                    'type': req.requirement_type,
                    'category': req.category,
                    'parameters': req.parameters,
                    'conditions': req.conditions,
                    'confidence': req.confidence,
                    'source_section': req.source_section
                } for req in document.requirements]
            }, ensure_ascii=False, indent=2)
        
        elif format_type == 'xml':
            # Basic XML export (can be enhanced)
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<regulatory_document id="{document.id}">
    <title>{document.title}</title>
    <version>{document.version}</version>
    <sections>
"""
            for section in document.sections:
                xml_content += f"""        <section id="{section.id}" type="{section.section_type}" level="{section.level}">
            <title>{section.title}</title>
            <content><![CDATA[{section.content}]]></content>
        </section>
"""
            xml_content += "    </sections>\n</regulatory_document>"
            return xml_content
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")