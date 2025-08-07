"""Enhanced Knowledge Graph Service with Advanced Entity Relationships and Clause Matching"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
import networkx as nx
import re
from dataclasses import dataclass
from collections import defaultdict
import pickle
import os
from datetime import datetime

# Optional dependencies with graceful fallback
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    name: str
    entity_type: str
    mentions: List[str]
    context: str
    confidence: float = 1.0

@dataclass
class Relationship:
    subject: str
    predicate: str
    object: str
    context: str
    confidence: float = 1.0

@dataclass
class ClauseMatch:
    query_clause: str
    document_clause: str
    similarity_score: float
    entity_overlap: List[str]
    relationship_overlap: List[str]
    explanation: str

class KnowledgeGraphService:
    def __init__(self):
        # Core graph structures
        self.graph = nx.MultiDiGraph()
        self.entity_index = {}
        self.relationship_index = defaultdict(list)
        
        # Domain-specific entities for insurance/medical domain
        self.domain_entities = {
            'medical': ['surgery', 'treatment', 'diagnosis', 'procedure', 'therapy', 'consultation', 
                       'hospital', 'doctor', 'patient', 'disease', 'condition', 'symptom',
                       'operation', 'examination', 'test', 'screening', 'biopsy', 'endoscopy',
                       'angioplasty', 'transplant', 'chemotherapy', 'radiotherapy', 'dialysis'],
            'insurance': ['policy', 'premium', 'coverage', 'claim', 'deductible', 'copay', 'benefit',
                         'exclusion', 'waiting period', 'sum insured', 'policyholder', 'nominee',
                         'cashless', 'reimbursement', 'network hospital', 'pre-existing'],
            'financial': ['amount', 'cost', 'price', 'limit', 'percentage', 'discount', 'fee', 'charge',
                         'rupees', 'rs', 'inr', 'maximum', 'minimum', 'lakh', 'crore'],
            'temporal': ['year', 'month', 'day', 'period', 'duration', 'before', 'after', 'during',
                        'days', 'months', 'years', 'hours', 'weeks', 'immediately', 'continuous']
        }
        
        # spaCy model for advanced NLP
        self.nlp_model = None
        if SPACY_AVAILABLE:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("SpaCy model loaded for advanced entity extraction")
            except OSError:
                logger.warning("SpaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        
        # Neo4j integration (optional)
        self.neo4j_driver = None
        self.use_neo4j = False
        
        # Data persistence
        self.data_path = "data/"
        os.makedirs(self.data_path, exist_ok=True)
        
        # Advanced clause matching patterns
        self.clause_patterns = {
            'coverage': [
                r'covers?\s+(.+?)(?=\.|,|;|$)',
                r'includes?\s+(.+?)(?=\.|,|;|$)', 
                r'entitled\s+to\s+(.+?)(?=\.|,|;|$)',
                r'eligible\s+for\s+(.+?)(?=\.|,|;|$)',
                r'benefits?\s+(?:for|of)\s+(.+?)(?=\.|,|;|$)',
                r'reimbursement\s+(?:for|of)\s+(.+?)(?=\.|,|;|$)'
            ],
            'exclusion': [
                r'excludes?\s+(.+?)(?=\.|,|;|$)',
                r'not\s+cover(?:ed)?\s+(.+?)(?=\.|,|;|$)',
                r'except\s+(.+?)(?=\.|,|;|$)',
                r'does\s+not\s+apply\s+to\s+(.+?)(?=\.|,|;|$)',
                r'excluded\s+(?:from|are)\s+(.+?)(?=\.|,|;|$)',
                r'not\s+eligible\s+(?:for|are)\s+(.+?)(?=\.|,|;|$)'
            ],
            'condition': [
                r'subject\s+to\s+(.+?)(?=\.|,|;|$)',
                r'provided\s+that\s+(.+?)(?=\.|,|;|$)',
                r'on\s+condition\s+(?:that\s+)?(.+?)(?=\.|,|;|$)',
                r'if\s+(.+?)(?=\.|,|;|$)',
                r'unless\s+(.+?)(?=\.|,|;|$)',
                r'contingent\s+(?:on|upon)\s+(.+?)(?=\.|,|;|$)'
            ],
            'amount': [
                r'up\s+to\s+(?:Rs\.?\s*|INR\s*)?(\d+(?:,\d+)*(?:\.\d+)?)',
                r'maximum\s+(?:of\s+)?(?:Rs\.?\s*|INR\s*)?(\d+(?:,\d+)*(?:\.\d+)?)',
                r'limit\s+(?:of\s+)?(?:Rs\.?\s*|INR\s*)?(\d+(?:,\d+)*(?:\.\d+)?)',
                r'(?:Rs\.?\s*|INR\s*)(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:or\s+)?(?:above|below|maximum|minimum)',
                r'sum\s+insured\s+(?:of\s+)?(?:Rs\.?\s*|INR\s*)?(\d+(?:,\d+)*(?:\.\d+)?)',
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|crore|thousand)'
            ],
            'time_period': [
                r'(\d+)\s*(?:days?|months?|years?|hours?|weeks?)',
                r'waiting\s+period\s+of\s+(\d+)\s*(?:days?|months?|years?)',
                r'after\s+(\d+)\s*(?:days?|months?|years?)',
                r'within\s+(\d+)\s*(?:days?|months?|years?)',
                r'continuous\s+(?:coverage\s+)?(?:for\s+)?(\d+)\s*(?:days?|months?|years?)',
                r'(\d+)\s*(?:days?|months?|years?)\s+(?:of\s+)?(?:continuous\s+)?(?:coverage|waiting)'
            ]
        }
        
    async def initialize(self):
        """Initialize the Knowledge Graph Service"""
        try:
            # Load existing knowledge graph if available
            await self._load_knowledge_graph()
            
            logger.info("Knowledge Graph Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Graph Service: {str(e)}")
            raise

    async def build_from_document(self, document_content: str, document_url: str):
        """Build knowledge graph from document content"""
        try:
            logger.info("Building knowledge graph from document")
            
            # Extract entities and relationships
            entities = await self._extract_entities(document_content, document_url)
            relationships = await self._extract_relationships(document_content, entities)
            
            # Add to knowledge graph
            for entity in entities:
                self._add_entity_to_graph(entity)
            
            for relationship in relationships:
                self._add_relationship_to_graph(relationship)
            
            # Build domain-specific indices
            await self._build_domain_indices()
            
            # Save knowledge graph
            await self._save_knowledge_graph()
            
            logger.info(f"Knowledge graph built: {len(entities)} entities, {len(relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")

    async def _extract_entities(self, text: str, document_url: str) -> List[Entity]:
        """Extract entities using multiple approaches"""
        entities = []
        
        # Use spaCy if available
        if self.nlp_model:
            entities.extend(await self._spacy_entity_extraction(text, document_url))
        
        # Rule-based entity extraction (always available)
        entities.extend(await self._rule_based_entity_extraction(text, document_url))
        
        # Pattern-based entity extraction
        entities.extend(await self._pattern_based_entity_extraction(text, document_url))
        
        # Deduplicate and merge similar entities
        unique_entities = self._deduplicate_entities(entities)
        
        logger.info(f"Extracted {len(unique_entities)} unique entities from document")
        return unique_entities

    async def _spacy_entity_extraction(self, text: str, document_url: str) -> List[Entity]:
        """Extract entities using spaCy NER"""
        entities = []
        
        try:
            # Process text in chunks for large documents
            max_chars = 1000000  # 1MB limit for spaCy
            text_chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
            
            for chunk in text_chunks:
                doc = self.nlp_model(chunk)
                
                for ent in doc.ents:
                    entity_type = self._map_spacy_label(ent.label_)
                    if entity_type != 'unknown':  # Only keep meaningful entities
                        entity = Entity(
                            name=ent.text.strip(),
                            entity_type=entity_type,
                            mentions=[ent.text.strip()],
                            context=ent.sent.text[:200] if ent.sent else "",
                            confidence=0.8  # SpaCy confidence
                        )
                        entities.append(entity)
                        
        except Exception as e:
            logger.warning(f"SpaCy entity extraction failed: {str(e)}")
            
        return entities

    def _map_spacy_label(self, spacy_label: str) -> str:
        """Map spaCy entity labels to domain types"""
        label_mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'MONEY': 'financial',
            'DATE': 'temporal',
            'TIME': 'temporal',
            'PERCENT': 'financial',
            'CARDINAL': 'numeric',
            'ORDINAL': 'numeric',
            'GPE': 'location',
            'EVENT': 'medical',
            'PRODUCT': 'insurance',
            'LAW': 'insurance',
            'FACILITY': 'medical'
        }
        return label_mapping.get(spacy_label, 'unknown')

    async def _rule_based_entity_extraction(self, text: str, document_url: str) -> List[Entity]:
        """Extract entities using domain-specific rules"""
        entities = []
        text_lower = text.lower()
        
        # Extract domain-specific entities
        for domain, keywords in self.domain_entities.items():
            for keyword in keywords:
                pattern = rf'\b{re.escape(keyword)}\b'
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                if matches:
                    mentions = [match.group() for match in matches]
                    # Get context around first mention
                    first_match = matches[0]
                    start = max(0, first_match.start() - 100)
                    end = min(len(text), first_match.end() + 100)
                    context = text[start:end].strip()
                    
                    entity = Entity(
                        name=keyword,
                        entity_type=domain,
                        mentions=mentions,
                        context=context,
                        confidence=0.9  # High confidence for domain keywords
                    )
                    entities.append(entity)
        
        return entities

    async def _pattern_based_entity_extraction(self, text: str, document_url: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        # Extract numerical entities (amounts, percentages, time periods)
        number_patterns = {
            'financial': [
                r'(?:Rs\.?\s*|INR\s*|rupees?\s*)[\d,]+(?:\.\d+)?',
                r'\d+(?:\.\d+)?\s*%',
                r'sum\s+insured\s+of\s+(?:Rs\.?\s*)?[\d,]+(?:\.\d+)?',
                r'\d+\s*(?:lakh|crore|thousand)(?:\s+rupees?)?'
            ],
            'temporal': [
                r'\d+\s*(?:days?|months?|years?|hrs?|hours?|weeks?)',
                r'waiting\s+period\s+of\s+\d+\s*(?:days?|months?|years?)',
                r'\d+\s*-\s*\d+\s*(?:days?|months?|years?)',
                r'continuous\s+coverage\s+for\s+\d+\s*(?:days?|months?|years?)'
            ],
            'medical': [
                r'(?:diagnosis|treatment|surgery|operation|procedure)\s+(?:of|for)\s+([a-zA-Z\s]+)',
                r'([a-zA-Z\s]+)\s+(?:surgery|operation|treatment|procedure)',
                r'medical\s+(?:treatment|procedure)\s+for\s+([a-zA-Z\s]+)'
            ]
        }
        
        for entity_type, patterns in number_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    entity_text = match.group(1) if match.groups() else match.group().strip()
                    
                    entity = Entity(
                        name=entity_text,
                        entity_type=entity_type,
                        mentions=[entity_text],
                        context=context,
                        confidence=0.95  # High confidence for pattern matches
                    )
                    entities.append(entity)
        
        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate and merge similar entities"""
        unique_entities = {}
        
        for entity in entities:
            # Create key based on normalized name and type
            key = (entity.name.lower().strip(), entity.entity_type)
            
            if key in unique_entities:
                # Merge with existing entity
                existing = unique_entities[key]
                existing.mentions.extend(entity.mentions)
                existing.mentions = list(set(existing.mentions))  # Remove duplicates
                existing.confidence = max(existing.confidence, entity.confidence)
                # Update context if new one is longer
                if len(entity.context) > len(existing.context):
                    existing.context = entity.context
            else:
                unique_entities[key] = entity
        
        return list(unique_entities.values())

    async def _extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities"""
        relationships = []
        
        # Enhanced relationship patterns for insurance domain
        relationship_patterns = [
            # Coverage relationships
            (r'(.+?)\s+(?:is\s+|are\s+)?covered\s+(?:under|by)\s+(.+?)(?=\.|,|;|$)', 'covered_by'),
            (r'(.+?)\s+covers\s+(.+?)(?=\.|,|;|$)', 'covers'),
            (r'policy\s+covers\s+(.+?)(?=\.|,|;|$)', 'policy_covers'),
            (r'coverage\s+(?:includes?|for)\s+(.+?)(?=\.|,|;|$)', 'coverage_includes'),
            
            # Exclusion relationships
            (r'(.+?)\s+(?:is\s+|are\s+)?excluded\s+from\s+(.+?)(?=\.|,|;|$)', 'excluded_from'),
            (r'(.+?)\s+excludes\s+(.+?)(?=\.|,|;|$)', 'excludes'),
            (r'not\s+covered\s+(?:under|by)\s+(.+?)(?=\.|,|;|$)', 'not_covered_by'),
            
            # Condition relationships
            (r'(.+?)\s+(?:is\s+)?subject\s+to\s+(.+?)(?=\.|,|;|$)', 'subject_to'),
            (r'(.+?)\s+requires\s+(.+?)(?=\.|,|;|$)', 'requires'),
            (r'(.+?)\s+(?:is\s+)?applicable\s+(?:to|for)\s+(.+?)(?=\.|,|;|$)', 'applicable_to'),
            (r'(.+?)\s+(?:is\s+)?conditional\s+(?:on|upon)\s+(.+?)(?=\.|,|;|$)', 'conditional_on'),
            
            # Amount relationships
            (r'(.+?)\s+(?:is\s+)?limited\s+to\s+(.+?)(?=\.|,|;|$)', 'limited_to'),
            (r'(.+?)\s+(?:has\s+)?maximum\s+(?:of\s+)?(.+?)(?=\.|,|;|$)', 'has_maximum'),
            (r'(.+?)\s+costs?\s+(.+?)(?=\.|,|;|$)', 'costs'),
            (r'(.+?)\s+(?:is\s+)?capped\s+at\s+(.+?)(?=\.|,|;|$)', 'capped_at'),
            
            # Time relationships
            (r'(.+?)\s+(?:has\s+)?waiting\s+period\s+(?:of\s+)?(.+?)(?=\.|,|;|$)', 'has_waiting_period'),
            (r'(.+?)\s+(?:is\s+)?effective\s+after\s+(.+?)(?=\.|,|;|$)', 'effective_after'),
            (r'(.+?)\s+(?:is\s+)?valid\s+for\s+(.+?)(?=\.|,|;|$)', 'valid_for'),
            (r'(.+?)\s+(?:requires\s+)?continuous\s+coverage\s+(?:for\s+)?(.+?)(?=\.|,|;|$)', 'requires_continuous'),
            
            # Medical relationships
            (r'(.+?)\s+(?:is\s+)?treated\s+(?:at|in)\s+(.+?)(?=\.|,|;|$)', 'treated_at'),
            (r'(.+?)\s+(?:is\s+)?available\s+(?:at|in)\s+(.+?)(?=\.|,|;|$)', 'available_at'),
            (r'(.+?)\s+(?:procedure|surgery|treatment)\s+(?:for|of)\s+(.+?)(?=\.|,|;|$)', 'procedure_for')
        ]
        
        for pattern, relation_type in relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    subject = match.group(1).strip()
                    object_val = match.group(2).strip()
                    
                    # Filter out very short or meaningless subjects/objects
                    if len(subject) > 2 and len(object_val) > 2:
                        # Get context around the match
                        start = max(0, match.start() - 100)
                        end = min(len(text), match.end() + 100)
                        context = text[start:end].strip()
                        
                        relationship = Relationship(
                            subject=subject,
                            predicate=relation_type,
                            object=object_val,
                            context=context,
                            confidence=0.8
                        )
                        relationships.append(relationship)
        
        return relationships

    def _add_entity_to_graph(self, entity: Entity):
        """Add entity to the knowledge graph"""
        node_id = f"{entity.entity_type}:{entity.name.lower()}"
        
        if node_id in self.graph:
            # Update existing entity
            existing_mentions = self.graph.nodes[node_id].get('mentions', [])
            updated_mentions = list(set(existing_mentions + entity.mentions))
            self.graph.nodes[node_id]['mentions'] = updated_mentions
            self.graph.nodes[node_id]['confidence'] = max(
                self.graph.nodes[node_id].get('confidence', 0), 
                entity.confidence
            )
        else:
            # Add new entity
            self.graph.add_node(node_id, **{
                'name': entity.name,
                'type': entity.entity_type,
                'mentions': entity.mentions,
                'context': entity.context,
                'confidence': entity.confidence
            })
        
        # Update entity index
        if entity.entity_type not in self.entity_index:
            self.entity_index[entity.entity_type] = []
        if node_id not in self.entity_index[entity.entity_type]:
            self.entity_index[entity.entity_type].append(node_id)

    def _add_relationship_to_graph(self, relationship: Relationship):
        """Add relationship to the knowledge graph"""
        # Create simplified node IDs for subject and object
        subject_id = f"entity:{relationship.subject.lower()}"
        object_id = f"entity:{relationship.object.lower()}"
        
        # Add nodes if they don't exist
        if subject_id not in self.graph:
            self.graph.add_node(subject_id, name=relationship.subject, type='entity')
        if object_id not in self.graph:
            self.graph.add_node(object_id, name=relationship.object, type='entity')
        
        # Add edge with relationship info
        self.graph.add_edge(subject_id, object_id, **{
            'predicate': relationship.predicate,
            'context': relationship.context,
            'confidence': relationship.confidence
        })
        
        # Update relationship index
        self.relationship_index[relationship.predicate].append({
            'subject': subject_id,
            'object': object_id,
            'context': relationship.context,
            'confidence': relationship.confidence
        })

    async def _build_domain_indices(self):
        """Build specialized indices for different domains"""
        try:
            # Medical procedure index
            medical_entities = self.entity_index.get('medical', [])
            for entity_id in medical_entities:
                if entity_id in self.graph.nodes:
                    entity_data = self.graph.nodes[entity_id]
                    # Find relevant relationships
                    medical_relations = []
                    for edge in self.graph.edges(entity_id, data=True):
                        if edge[2].get('predicate') in ['covered_by', 'excluded_from', 'requires', 'has_waiting_period']:
                            medical_relations.append(edge)
                    
                    self.graph.nodes[entity_id]['medical_relations'] = medical_relations
            
            # Insurance term index
            insurance_entities = self.entity_index.get('insurance', [])
            for entity_id in insurance_entities:
                if entity_id in self.graph.nodes:
                    # Build coverage relationships
                    coverage_relations = []
                    for edge in self.graph.edges(entity_id, data=True):
                        if edge[2].get('predicate') in ['covers', 'excludes', 'limited_to']:
                            coverage_relations.append(edge)
                    
                    self.graph.nodes[entity_id]['coverage_relations'] = coverage_relations
            
            logger.info("Domain indices built successfully")
            
        except Exception as e:
            logger.warning(f"Failed to build domain indices: {str(e)}")

    async def enhance_retrieval_with_entities(self, query: str, retrieval_results: List[Any]) -> List[Any]:
        """Enhance retrieval results using knowledge graph entities"""
        try:
            # Extract entities from query
            query_entities = await self._extract_query_entities(query)
            
            if not query_entities:
                return retrieval_results
            
            # Score and enhance results based on entity matches
            enhanced_results = []
            for result in retrieval_results:
                # Get result text
                result_text = self._extract_text_from_result(result)
                
                # Extract entities from result
                result_entities = await self._extract_query_entities(result_text)
                
                # Calculate entity overlap
                entity_overlap = self._calculate_entity_overlap(query_entities, result_entities)
                
                # Find relationship matches
                relationship_matches = self._find_relationship_matches(query_entities, result_entities)
                
                # Enhance result with knowledge graph info
                if hasattr(result, 'score'):
                    original_score = result.score
                    # Boost score based on entity overlap and relationships
                    entity_boost = min(entity_overlap * 0.2, 0.25)  # Max 25% boost
                    relationship_boost = min(len(relationship_matches) * 0.1, 0.15)  # Max 15% boost
                    result.score = original_score * (1 + entity_boost + relationship_boost)
                    
                    # Update explanation
                    if hasattr(result, 'explanation'):
                        if entity_overlap > 0:
                            result.explanation += f" | Entity overlap: {entity_overlap:.2f}"
                        if relationship_matches:
                            result.explanation += f" | Relationships: {len(relationship_matches)}"
                
                enhanced_results.append(result)
            
            # Sort by enhanced scores
            if enhanced_results and hasattr(enhanced_results[0], 'score'):
                enhanced_results.sort(key=lambda x: x.score, reverse=True)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error enhancing retrieval with entities: {str(e)}")
            return retrieval_results

    def _extract_text_from_result(self, result) -> str:
        """Extract text from different result formats"""
        if hasattr(result, 'chunk') and hasattr(result.chunk, 'text'):
            return result.chunk.text
        elif isinstance(result, dict):
            return result.get('text', str(result))
        else:
            return str(result)

    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query text"""
        entities = set()
        query_lower = query.lower()
        
        # Check against known domain entities
        for domain, keywords in self.domain_entities.items():
            for keyword in keywords:
                if keyword in query_lower:
                    entities.add(keyword)
        
        # Extract numerical entities
        number_patterns = [
            r'(?:Rs\.?\s*|INR\s*|rupees?\s*)[\d,]+(?:\.\d+)?',
            r'\d+(?:\.\d+)?\s*%',
            r'\d+\s*(?:days?|months?|years?|hrs?|hours?|weeks?)',
            r'\d+\s*(?:lakh|crore|thousand)'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.update([match.strip() for match in matches])
        
        return list(entities)

    def _calculate_entity_overlap(self, entities1: List[str], entities2: List[str]) -> float:
        """Calculate overlap between two entity lists"""
        if not entities1 or not entities2:
            return 0.0
        
        set1 = set(e.lower().strip() for e in entities1)
        set2 = set(e.lower().strip() for e in entities2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0

    def _find_relationship_matches(self, query_entities: List[str], result_entities: List[str]) -> List[str]:
        """Find matching relationships between query and result entities"""
        matches = []
        
        for q_entity in query_entities:
            for r_entity in result_entities:
                # Check if there are any relationships between these entities in the graph
                q_node = f"entity:{q_entity.lower()}"
                r_node = f"entity:{r_entity.lower()}"
                
                if q_node in self.graph and r_node in self.graph:
                    if self.graph.has_edge(q_node, r_node) or self.graph.has_edge(r_node, q_node):
                        matches.append(f"{q_entity}-{r_entity}")
        
        return matches

    async def find_similar_clauses(self, query: str, document_text: str, top_k: int = 5) -> List[ClauseMatch]:
        """Find clauses in document similar to query using knowledge graph"""
        try:
            clause_matches = []
            
            # Extract clauses from query and document
            query_clauses = self._extract_clauses(query)
            document_clauses = self._extract_clauses(document_text)
            
            # Find matches between query and document clauses
            for query_clause in query_clauses[:3]:  # Limit query clauses
                for doc_clause in document_clauses:
                    similarity = await self._calculate_clause_similarity(query_clause, doc_clause)
                    
                    if similarity > 0.3:  # Minimum similarity threshold
                        # Extract entities from both clauses
                        query_entities = await self._extract_query_entities(query_clause)
                        doc_entities = await self._extract_query_entities(doc_clause)
                        
                        entity_overlap = [e for e in query_entities if e.lower() in [d.lower() for d in doc_entities]]
                        
                        # Find relationship overlaps
                        relationship_overlap = self._find_relationship_overlap(query_clause, doc_clause)
                        
                        # Generate explanation
                        explanation = self._generate_clause_match_explanation(
                            similarity, entity_overlap, relationship_overlap
                        )
                        
                        clause_match = ClauseMatch(
                            query_clause=query_clause,
                            document_clause=doc_clause,
                            similarity_score=similarity,
                            entity_overlap=entity_overlap,
                            relationship_overlap=relationship_overlap,
                            explanation=explanation
                        )
                        clause_matches.append(clause_match)
            
            # Sort by similarity and return top-k
            clause_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            return clause_matches[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar clauses: {str(e)}")
            return []

    def _extract_clauses(self, text: str) -> List[str]:
        """Extract meaningful clauses from text"""
        # Split by sentences and filter out short ones
        sentences = re.split(r'[.!?]+', text)
        clauses = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum clause length
                # Further split by commas and semicolons for sub-clauses
                sub_clauses = []
                for delimiter in [',', ';']:
                    if delimiter in sentence:
                        parts = [s.strip() for s in sentence.split(delimiter) if len(s.strip()) > 15]
                        sub_clauses.extend(parts[:2])  # Limit sub-clauses per delimiter
                
                if sub_clauses:
                    clauses.extend(sub_clauses[:3])  # Limit sub-clauses per sentence
                else:
                    clauses.append(sentence)
        
        return clauses[:15]  # Limit total clauses

    async def _calculate_clause_similarity(self, clause1: str, clause2: str) -> float:
        """Calculate similarity between two clauses"""
        try:
            # Simple word overlap similarity
            words1 = set(re.findall(r'\w+', clause1.lower()))
            words2 = set(re.findall(r'\w+', clause2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            word_similarity = intersection / union if union > 0 else 0.0
            
            # Entity-based similarity boost
            entities1 = await self._extract_query_entities(clause1)
            entities2 = await self._extract_query_entities(clause2)
            entity_similarity = self._calculate_entity_overlap(entities1, entities2)
            
            # Domain-specific term similarity
            domain_similarity = self._calculate_domain_term_similarity(clause1, clause2)
            
            # Combined similarity (weighted)
            combined_similarity = (
                0.5 * word_similarity + 
                0.3 * entity_similarity + 
                0.2 * domain_similarity
            )
            
            return combined_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating clause similarity: {str(e)}")
            return 0.0

    def _calculate_domain_term_similarity(self, clause1: str, clause2: str) -> float:
        """Calculate similarity based on domain-specific terms"""
        clause1_lower = clause1.lower()
        clause2_lower = clause2.lower()
        
        domain_terms = []
        for domain_list in self.domain_entities.values():
            domain_terms.extend(domain_list)
        
        terms1 = set(term for term in domain_terms if term in clause1_lower)
        terms2 = set(term for term in domain_terms if term in clause2_lower)
        
        if not terms1 or not terms2:
            return 0.0
        
        intersection = len(terms1 & terms2)
        union = len(terms1 | terms2)
        
        return intersection / union if union > 0 else 0.0

    def _find_relationship_overlap(self, clause1: str, clause2: str) -> List[str]:
        """Find overlapping relationships between clauses"""
        overlapping_relations = []
        
        # Check for common relationship patterns
        relation_patterns = ['covers', 'excludes', 'requires', 'applicable', 'limited', 'waiting', 
                           'subject', 'conditional', 'effective', 'maximum', 'minimum']
        
        clause1_lower = clause1.lower()
        clause2_lower = clause2.lower()
        
        for pattern in relation_patterns:
            if pattern in clause1_lower and pattern in clause2_lower:
                overlapping_relations.append(pattern)
        
        return overlapping_relations

    def _generate_clause_match_explanation(
        self, 
        similarity: float, 
        entity_overlap: List[str], 
        relationship_overlap: List[str]
    ) -> str:
        """Generate explanation for clause match"""
        explanations = []
        
        if similarity > 0.8:
            explanations.append("High textual similarity")
        elif similarity > 0.6:
            explanations.append("Good textual similarity")
        elif similarity > 0.4:
            explanations.append("Moderate textual similarity")
        else:
            explanations.append("Low textual similarity")
        
        if entity_overlap:
            entities_str = ', '.join(entity_overlap[:3])
            explanations.append(f"Shared entities: {entities_str}")
        
        if relationship_overlap:
            relations_str = ', '.join(relationship_overlap[:2])
            explanations.append(f"Similar relationships: {relations_str}")
        
        return "; ".join(explanations)

    async def _save_knowledge_graph(self):
        """Save knowledge graph to disk"""
        try:
            graph_path = os.path.join(self.data_path, "knowledge_graph.pkl")
            with open(graph_path, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'entity_index': self.entity_index,
                    'relationship_index': dict(self.relationship_index),
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            logger.info("Knowledge graph saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")

    async def _load_knowledge_graph(self):
        """Load knowledge graph from disk"""
        try:
            graph_path = os.path.join(self.data_path, "knowledge_graph.pkl")
            if os.path.exists(graph_path):
                with open(graph_path, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data.get('graph', nx.MultiDiGraph())
                    self.entity_index = data.get('entity_index', {})
                    self.relationship_index = defaultdict(list, data.get('relationship_index', {}))
                
                logger.info(f"Knowledge graph loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            
        except Exception as e:
            logger.warning(f"Could not load existing knowledge graph: {str(e)}")

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'nodes': len(self.graph.nodes),
            'edges': len(self.graph.edges),
            'entity_types': list(self.entity_index.keys()),
            'relationship_types': list(self.relationship_index.keys()),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes) if self.graph.nodes else 0,
            'spacy_available': self.nlp_model is not None,
            'neo4j_available': NEO4J_AVAILABLE
        }

    def is_healthy(self) -> bool:
        """Check if knowledge graph service is healthy"""
        return self.graph is not None and len(self.graph.nodes) >= 0

    # Legacy compatibility method
    def process_document_for_kg(self, text: str, document_source: str):
        """Process document and extract knowledge for graph (legacy compatibility)"""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.build_from_document(text, document_source))
            loop.close()
        except Exception as e:
            logger.error(f"Error in legacy document processing: {str(e)}")

# Create service instance for import compatibility
KnowledgeGraphService = KnowledgeGraphService