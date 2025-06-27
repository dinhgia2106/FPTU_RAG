"""
Graph Database Module - Tích hợp Neo4j cho GraphRAG
Triển khai Gradual Migration: Vector Search + Graph Traversal
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Đại diện cho một node trong knowledge graph"""
    id: str
    type: str  # Course, Topic, Semester, Combo, CLO, Material
    properties: Dict[str, Any]

@dataclass
class GraphRelationship:
    """Đại diện cho một relationship trong knowledge graph"""
    source_id: str
    target_id: str
    type: str  # HAS_PREREQUISITE, TAUGHT_IN, BELONGS_TO_COMBO, etc.
    properties: Dict[str, Any]

@dataclass
class GraphPath:
    """Đại diện cho một path trong graph cho multi-hop reasoning"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    length: int
    path_type: str  # learning_path, prerequisite_chain, related_courses

class GraphDatabase:
    """
    Neo4j Graph Database integration cho FPTU RAG
    Hỗ trợ gradual migration từ vector-only sang hybrid GraphRAG
    """
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        # Neo4j connection settings
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = None
        self.is_connected = False
        
        logger.info(f"GraphDatabase initialized - URI: {self.uri}")
    
    def connect(self) -> bool:
        """Kết nối tới Neo4j database"""
        try:
            from neo4j import GraphDatabase as Neo4jDriver
            from neo4j.exceptions import ServiceUnavailable, AuthError
            
            self.driver = Neo4jDriver.driver(self.uri, auth=(self.username, self.password))
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
            if test_value == 1:
                self.is_connected = True
                logger.info("✓ Kết nối Neo4j thành công")
                return True
            else:
                logger.error("✗ Test connection failed")
                return False
                
        except (ServiceUnavailable, AuthError) as e:
            logger.warning(f"⚠ Neo4j không khả dụng: {e}")
            logger.warning("  Hệ thống sẽ fallback về vector-only mode")
            self.is_connected = False
            return False
        except ImportError:
            logger.warning("⚠ Neo4j driver chưa được cài đặt - pip install neo4j")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"✗ Lỗi kết nối Neo4j: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Đóng kết nối Neo4j"""
        if self.driver:
            self.driver.close()
            self.is_connected = False
            logger.info("Đóng kết nối Neo4j")
    
    def extract_entities_from_curriculum_data(self, curriculum_data: List[Dict]) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        Extract entities và relationships từ dữ liệu curriculum hiện tại
        Transform JSON data thành graph structure
        """
        nodes = []
        relationships = []
        
        logger.info(f"Extracting entities từ {len(curriculum_data)} curriculum items...")
        
        # DEBUG: Log first few items to understand structure
        if curriculum_data:
            logger.info(f"DEBUG: Sample data structure:")
            for i, item in enumerate(curriculum_data[:2]):
                logger.info(f"  Item {i}: keys = {list(item.keys())}")
                if 'syllabuses' in item:
                    logger.info(f"    Found syllabuses: {len(item['syllabuses'])} items")
                elif 'metadata' in item:
                    metadata = item['metadata']
                    logger.info(f"    Found metadata with keys: {list(metadata.keys())}")
                    if 'course_id' in metadata:
                        logger.info(f"    Course ID in metadata: {metadata['course_id']}")
                elif 'course_id' in item:
                    logger.info(f"    Direct course_id: {item.get('course_id', 'N/A')}")
                else:
                    logger.info(f"    Unknown structure, sample keys: {list(item.keys())[:5]}")
        
        # Determine the data format and process accordingly
        syllabuses_to_process = []
        major_code = 'AI'  # Default
        
        if curriculum_data:
            first_item = curriculum_data[0]
            
            # Case 1: curriculum_data is array of whole curriculum objects
            if 'syllabuses' in first_item:
                logger.info("Format: Array of curriculum objects with syllabuses")
                for item in curriculum_data:
                    syllabuses_to_process.extend(item.get('syllabuses', []))
                    major_code = item.get('major_code_input', major_code)
                    
            # Case 2: curriculum_data is already array of syllabuses  
            elif 'metadata' in first_item:
                logger.info("Format: Array of syllabus objects")
                syllabuses_to_process = curriculum_data
                
            # Case 3: Unknown format - try to extract what we can
            else:
                logger.info("Format: Unknown - trying fallback extraction")
                syllabuses_to_process = curriculum_data
        
        logger.info(f"Processing {len(syllabuses_to_process)} syllabuses for major {major_code}")
        
        # Process each syllabus
        for i, syllabus in enumerate(syllabuses_to_process):
            try:
                metadata = syllabus.get('metadata', {})
                
                # Extract course node
                course_node = self._extract_course_node_from_syllabus(metadata, major_code)
                if course_node:
                    nodes.append(course_node)
                    
                    # Extract relationships từ course
                    course_relationships = self._extract_course_relationships_from_syllabus(metadata)
                    relationships.extend(course_relationships)
                    
                    logger.info(f"  Extracted course: {course_node.id}")
                
                # Extract CLO entities từ learning outcomes
                clo_entities = self._extract_clo_entities_from_syllabus(syllabus, metadata.get('course_id', ''))
                nodes.extend(clo_entities)
                
                # Extract material entities
                material_entities = self._extract_material_entities_from_syllabus(syllabus, metadata.get('course_id', ''))
                nodes.extend(material_entities)
                
            except Exception as e:
                logger.error(f"Lỗi extract entity từ syllabus {i}: {e}")
                continue
        
        # Remove duplicates
        unique_nodes = self._deduplicate_nodes(nodes)
        unique_relationships = self._deduplicate_relationships(relationships)
        
        logger.info(f"✓ Extracted {len(unique_nodes)} nodes, {len(unique_relationships)} relationships")
        if unique_nodes:
            logger.info(f"  Node types: {[node.type for node in unique_nodes[:5]]}")
            logger.info(f"  Sample node IDs: {[node.id for node in unique_nodes[:5]]}")
        
        return unique_nodes, unique_relationships
    
    def find_learning_path(self, start_course: str, end_course: str, max_length: int = 5) -> List[GraphPath]:
        """
        Tìm learning paths giữa 2 courses - Multi-hop reasoning thực sự
        """
        if not self.is_connected:
            logger.warning("Neo4j không kết nối - không thể tìm learning path")
            return []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH path = allShortestPaths((start:Course {id: $start_course})-[*1..%d]-(end:Course {id: $end_course}))
                WHERE ALL(r IN relationships(path) WHERE type(r) IN ['HAS_PREREQUISITE', 'TAUGHT_IN', 'BELONGS_TO_COMBO'])
                RETURN path
                LIMIT 10
                """ % max_length
                
                result = session.run(query, start_course=start_course, end_course=end_course)
                
                paths = []
                for record in result:
                    path_data = record['path']
                    graph_path = self._parse_neo4j_path(path_data, 'learning_path')
                    paths.append(graph_path)
                
                logger.info(f"Tìm thấy {len(paths)} learning paths từ {start_course} đến {end_course}")
                return paths
                
        except Exception as e:
            logger.error(f"Lỗi tìm learning path: {e}")
            return []
    
    def find_prerequisites(self, course_code: str, depth: int = 3) -> List[GraphPath]:
        """Tìm prerequisite chain cho một course"""
        if not self.is_connected:
            return []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH path = (course:Course {id: $course_code})-[:HAS_PREREQUISITE*1..%d]->(prereq:Course)
                RETURN path
                ORDER BY length(path)
                """ % depth
                
                result = session.run(query, course_code=course_code)
                
                paths = []
                for record in result:
                    path_data = record['path']
                    graph_path = self._parse_neo4j_path(path_data, 'prerequisite_chain')
                    paths.append(graph_path)
                
                return paths
                
        except Exception as e:
            logger.error(f"Lỗi tìm prerequisites: {e}")
            return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về graph database"""
        if not self.is_connected:
            return {'connected': False}
        
        try:
            with self.driver.session() as session:
                # Count nodes by type
                node_counts = {}
                result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
                for record in result:
                    label = record['labels'][0] if record['labels'] else 'Unknown'
                    node_counts[label] = record['count']
                
                # Count relationships by type
                rel_counts = {}
                result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
                for record in result:
                    rel_counts[record['rel_type']] = record['count']
                
                return {
                    'connected': True,
                    'node_counts': node_counts,
                    'relationship_counts': rel_counts,
                    'total_nodes': sum(node_counts.values()),
                    'total_relationships': sum(rel_counts.values())
                }
                
        except Exception as e:
            logger.error(f"Lỗi lấy graph stats: {e}")
            return {'connected': False, 'error': str(e)}
    
    def _extract_course_node(self, item: Dict) -> Optional[GraphNode]:
        """Extract course node từ curriculum item"""
        metadata = item.get('metadata', {})
        subject_code = item.get('subject_code', '')
        
        if not subject_code:
            return None
        
        # Clean course properties
        properties = {
            'name': metadata.get('course_name_from_curriculum', metadata.get('title', '')),
            'english_title': metadata.get('english_title', ''),
            'credits': self._safe_int(metadata.get('credits', 0)),
            'semester': self._safe_int(metadata.get('semester_from_curriculum', 0)),
            'description': metadata.get('description', ''),
            'course_type': metadata.get('course_type_guess', 'unknown'),
            'major_code': item.get('major_code', 'AI')
        }
        
        return GraphNode(
            id=subject_code,
            type='Course',
            properties=properties
        )
    
    def _extract_course_relationships(self, item: Dict) -> List[GraphRelationship]:
        """Extract relationships từ course item"""
        relationships = []
        metadata = item.get('metadata', {})
        subject_code = item.get('subject_code', '')
        
        if not subject_code:
            return relationships
        
        # TAUGHT_IN relationship (Course -> Semester)
        semester = self._safe_int(metadata.get('semester_from_curriculum', 0))
        if semester > 0:
            relationships.append(GraphRelationship(
                source_id=subject_code,
                target_id=f"Semester_{semester}",
                type='TAUGHT_IN',
                properties={'semester_number': semester}
            ))
        
        return relationships
    
    def _extract_combo_entities(self, item: Dict) -> List[GraphNode]:
        """Extract combo/specialization entities"""
        return []  # Simplified for now
    
    def _extract_clo_entities(self, item: Dict) -> List[GraphNode]:
        """Extract CLO entities"""
        return []  # Simplified for now
    
    def _safe_int(self, value, default=0) -> int:
        """Safely convert value to int"""
        try:
            if isinstance(value, str):
                cleaned = re.sub(r'[^\d.]', '', value)
                return int(float(cleaned)) if cleaned else default
            return int(value) if value else default
        except (ValueError, TypeError):
            return default
    
    def _deduplicate_nodes(self, nodes: List[GraphNode]) -> List[GraphNode]:
        """Remove duplicate nodes by ID"""
        seen_ids = set()
        unique_nodes = []
        
        for node in nodes:
            if node.id not in seen_ids:
                seen_ids.add(node.id)
                unique_nodes.append(node)
        
        return unique_nodes
    
    def _deduplicate_relationships(self, relationships: List[GraphRelationship]) -> List[GraphRelationship]:
        """Remove duplicate relationships"""
        seen_rels = set()
        unique_rels = []
        
        for rel in relationships:
            rel_key = (rel.source_id, rel.target_id, rel.type)
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_rels.append(rel)
        
        return unique_rels
    
    def _parse_neo4j_path(self, neo4j_path, path_type: str) -> GraphPath:
        """Convert Neo4j path object thành GraphPath"""
        nodes = []
        relationships = []
        
        # Parse nodes
        for neo4j_node in neo4j_path.nodes:
            node = GraphNode(
                id=neo4j_node['id'],
                type=list(neo4j_node.labels)[0],
                properties=dict(neo4j_node)
            )
            nodes.append(node)
        
        # Parse relationships
        for neo4j_rel in neo4j_path.relationships:
            rel = GraphRelationship(
                source_id=neo4j_rel.start_node['id'],
                target_id=neo4j_rel.end_node['id'],
                type=neo4j_rel.type,
                properties=dict(neo4j_rel)
            )
            relationships.append(rel)
        
        return GraphPath(
            nodes=nodes,
            relationships=relationships,
            length=len(neo4j_path.relationships),
            path_type=path_type
        )
    
    def _extract_course_node_from_syllabus(self, metadata: Dict, major_code: str) -> Optional[GraphNode]:
        """Extract course node từ syllabus metadata"""
        course_id = metadata.get('course_id', metadata.get('subject_code_on_page', ''))
        
        if not course_id:
            return None
        
        # Clean course properties
        properties = {
            'name': metadata.get('course_name_from_curriculum', metadata.get('title', '')),
            'english_title': metadata.get('english_title', ''),
            'credits': self._safe_int(metadata.get('credits', 0)),
            'semester': self._safe_int(metadata.get('semester_from_curriculum', 0)),
            'description': metadata.get('description', ''),
            'course_type': metadata.get('course_type_guess', 'unknown'),
            'major_code': major_code,
            'prerequisites': metadata.get('prerequisites', ''),
            'syllabus_id': metadata.get('syllabus_id', ''),
            'degree_level': metadata.get('degree_level', '')
        }
        
        return GraphNode(
            id=course_id,
            type='Course',
            properties=properties
        )
    
    def _extract_course_relationships_from_syllabus(self, metadata: Dict) -> List[GraphRelationship]:
        """Extract relationships từ syllabus metadata"""
        relationships = []
        course_id = metadata.get('course_id', metadata.get('subject_code_on_page', ''))
        
        if not course_id:
            return relationships
        
        # TAUGHT_IN relationship (Course -> Semester)
        semester = self._safe_int(metadata.get('semester_from_curriculum', 0))
        if semester > 0:
            relationships.append(GraphRelationship(
                source_id=course_id,
                target_id=f"Semester_{semester}",
                type='TAUGHT_IN',
                properties={'semester_number': semester}
            ))
        
        # HAS_PREREQUISITE relationships
        prerequisites = metadata.get('prerequisites', '')
        if prerequisites and prerequisites.strip():
            # Parse prerequisite string to extract course codes
            prereq_codes = self._parse_prerequisite_string(prerequisites)
            for prereq_code in prereq_codes:
                relationships.append(GraphRelationship(
                    source_id=course_id,
                    target_id=prereq_code,
                    type='HAS_PREREQUISITE',
                    properties={'prerequisite_text': prerequisites}
                ))
        
        return relationships
    
    def _extract_clo_entities_from_syllabus(self, syllabus: Dict, course_id: str) -> List[GraphNode]:
        """Extract CLO (Course Learning Outcomes) entities"""
        nodes = []
        learning_outcomes = syllabus.get('learning_outcomes', [])
        
        for outcome in learning_outcomes:
            clo_id = outcome.get('id', '')
            details = outcome.get('details', '')
            
            if clo_id and details:
                node = GraphNode(
                    id=f"{course_id}_{clo_id}",
                    type='CLO',
                    properties={
                        'clo_id': clo_id,
                        'details': details,
                        'course_id': course_id
                    }
                )
                nodes.append(node)
        
        return nodes
    
    def _extract_material_entities_from_syllabus(self, syllabus: Dict, course_id: str) -> List[GraphNode]:
        """Extract material entities"""
        nodes = []
        materials = syllabus.get('materials', [])
        
        for i, material in enumerate(materials):
            description = material.get('description', '')
            author = material.get('author', '')
            
            if description:
                material_id = f"{course_id}_Material_{i+1}"
                node = GraphNode(
                    id=material_id,
                    type='Material',
                    properties={
                        'description': description,
                        'author': author,
                        'publisher': material.get('publisher', ''),
                        'published_date': material.get('published_date', ''),
                        'edition': material.get('edition', ''),
                        'isbn': material.get('isbn', ''),
                        'is_main_material': material.get('is_main_material', False),
                        'is_hard_copy': material.get('is_hard_copy', False),
                        'is_online': material.get('is_online', False),
                        'course_id': course_id
                    }
                )
                nodes.append(node)
        
        return nodes
    
    def _parse_prerequisite_string(self, prereq_text: str) -> List[str]:
        """Parse prerequisite string để extract course codes"""
        if not prereq_text or prereq_text.strip() == "":
            return []
        
        # Simple regex to find course codes (letters + numbers)
        import re
        course_codes = re.findall(r'[A-Z]{2,4}\d{3}[a-z]?', prereq_text.upper())
        
        # Remove duplicates and return
        return list(set(course_codes))
    
    def _extract_course_node_from_metadata(self, metadata: Dict, major_code: str) -> Optional[GraphNode]:
        """Extract course node từ metadata ở top level (processed format)"""
        course_id = metadata.get('subject_code', '')
        
        if not course_id:
            return None
        
        # Clean course properties
        properties = {
            'name': metadata.get('course_name', metadata.get('title', '')),
            'english_title': metadata.get('english_title', ''),
            'credits': self._safe_int(metadata.get('credits', 0)),
            'semester': self._safe_int(metadata.get('semester', 0)),
            'description': metadata.get('description', ''),
            'course_type': metadata.get('course_type', 'unknown'),
            'major_code': major_code,
            'prerequisites': metadata.get('prerequisites', ''),
            'syllabus_id': metadata.get('syllabus_id', ''),
            'degree_level': metadata.get('degree_level', '')
        }
        
        return GraphNode(
            id=course_id,
            type='Course',
            properties=properties
        ) 