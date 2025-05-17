"""
Contextual Query Processor - Module xử lý truy vấn ngữ cảnh cho Syllabus Search Engine

Module này cung cấp các chức năng để phân tích và xử lý truy vấn người dùng,
xác định loại truy vấn, thực thể liên quan, và chuyển đổi truy vấn tự nhiên
thành truy vấn có cấu trúc để tìm kiếm hiệu quả.
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sentence_transformers import SentenceTransformer

class QueryType:
    """Định nghĩa các loại truy vấn được hỗ trợ."""
    SIMPLE_INFO = "simple_info"  # Truy vấn thông tin đơn giản
    ATTRIBUTE = "attribute"      # Truy vấn thuộc tính
    RELATIONSHIP = "relationship"  # Truy vấn quan hệ
    AGGREGATION = "aggregation"  # Truy vấn tổng hợp
    CLASSIFICATION = "classification"  # Truy vấn phân loại
    LINKING = "linking"          # Truy vấn liên kết

class EntityType:
    """Định nghĩa các loại thực thể được hỗ trợ."""
    COURSE = "course"
    SESSION = "session"
    CLO = "clo"
    ASSESSMENT = "assessment"
    MATERIAL = "material"

class ContextualQueryProcessor:
    """
    Lớp xử lý truy vấn ngữ cảnh, phân tích truy vấn người dùng và
    chuyển đổi thành truy vấn có cấu trúc.
    """
    
    def __init__(self, embedding_model: SentenceTransformer, entity_data: Dict[str, Any] = None):
        """
        Khởi tạo processor với mô hình embedding và dữ liệu thực thể.
        
        Args:
            embedding_model: Mô hình SentenceTransformer để tạo embedding
            entity_data: Dữ liệu thực thể đã được xử lý (nếu có)
        """
        self.embedding_model = embedding_model
        self.entity_data = entity_data or {}
        
        # Từ điển các pattern để nhận diện loại truy vấn
        self.query_patterns = {
            QueryType.SIMPLE_INFO: [
                r"(.*?) là (môn học|môn|khóa học) gì",
                r"(.*?) (dạy|học) về gì",
                r"mô tả (về )?(môn học|môn|khóa học) (.*?)",
                r"thông tin (về )?(môn học|môn|khóa học) (.*?)",
                r"(.*?) (là|thuộc) (môn|môn học|khóa học) (gì|nào)"
            ],
            QueryType.ATTRIBUTE: [
                r"(.*?) có (bao nhiêu|mấy) (tín chỉ|credit)",
                r"(số|mấy|bao nhiêu) (tín chỉ|credit) (của )?(môn |môn học |khóa học )?(.*?)",
                r"(điểm|điểm số|điểm đánh giá|điểm thi|điểm kiểm tra) (của )?(môn |môn học |khóa học )?(.*?)",
                r"(cách|phương pháp) (tính|đánh giá) điểm (của )?(môn |môn học |khóa học )?(.*?)",
                r"(.*?) (có|gồm) (bao nhiêu|mấy) (phần trăm|%) (.*?)",
                r"(thời lượng|thời gian) (của )?(môn |môn học |khóa học )?(.*?)"
            ],
            QueryType.RELATIONSHIP: [
                r"(session|buổi|buổi học) (\d+) (của )?(môn |môn học |khóa học )?(.*?) (học|dạy|nói về|bao gồm) (gì|những gì|điều gì)",
                r"(session|buổi|buổi học) (\d+) (học|dạy|nói về|bao gồm) (gì|những gì|điều gì)",
                r"(nội dung|chủ đề) (của )?(session|buổi|buổi học) (\d+) (của )?(môn |môn học |khóa học )?(.*?)",
                r"(clo|chuẩn đầu ra) (\d+) (của )?(môn |môn học |khóa học )?(.*?) (là|bao gồm|nói về) (gì|những gì|điều gì)",
                r"(clo|chuẩn đầu ra) (\d+) (là|bao gồm|nói về) (gì|những gì|điều gì)",
                r"(tài liệu|material|giáo trình) (của )?(môn |môn học |khóa học )?(.*?)"
            ],
            QueryType.AGGREGATION: [
                r"(có|gồm) (bao nhiêu|mấy) (session|buổi|buổi học) (trong|của) (môn |môn học |khóa học )?(.*?)",
                r"(có|gồm) (bao nhiêu|mấy) (clo|chuẩn đầu ra) (trong|của) (môn |môn học |khóa học )?(.*?)",
                r"(có|gồm) (bao nhiêu|mấy) (bài kiểm tra|bài thi|đánh giá|assessment) (trong|của) (môn |môn học |khóa học )?(.*?)",
                r"(có|gồm) (bao nhiêu|mấy) (tài liệu|material|giáo trình) (trong|của) (môn |môn học |khóa học )?(.*?)",
                r"(có|gồm) (bao nhiêu|mấy) (môn|môn học|khóa học) (toán|lập trình|marketing|kinh tế|quản trị|tiếng anh)",
                r"(có|gồm) (bao nhiêu|mấy) (môn|môn học|khóa học) (.*?)"
            ],
            QueryType.CLASSIFICATION: [
                r"(những|các|tất cả|toàn bộ) (môn|môn học|khóa học) (kết thúc|bắt đầu) (bằng|với) (chữ|ký tự|kí tự) ['\"](.*?)['\"]",
                r"(những|các|tất cả|toàn bộ) (môn|môn học|khóa học) (toán|lập trình|marketing|kinh tế|quản trị|tiếng anh)",
                r"(những|các|tất cả|toàn bộ) (môn|môn học|khóa học) (có) (.*?) (tín chỉ|credit)",
                r"(những|các|tất cả|toàn bộ) (môn|môn học|khóa học) (liên quan|về) (.*?)"
            ],
            QueryType.LINKING: [
                r"(đưa|liệt kê|cho xem|hiển thị) (tất cả|toàn bộ|các) (link|đường dẫn|url) (.*?)",
                r"(đưa|liệt kê|cho xem|hiển thị) (tất cả|toàn bộ|các) (tài liệu|material|giáo trình) (.*?)",
                r"(đưa|liệt kê|cho xem|hiển thị) (tất cả|toàn bộ|các) (môn|môn học|khóa học) (.*?)"
            ]
        }
        
        # Từ điển các pattern để trích xuất thực thể
        self.entity_patterns = {
            EntityType.COURSE: [
                r"môn học ([A-Z]{2,}[0-9]{3}[a-z]*)",
                r"môn ([A-Z]{2,}[0-9]{3}[a-z]*)",
                r"khóa học ([A-Z]{2,}[0-9]{3}[a-z]*)",
                r"([A-Z]{2,}[0-9]{3}[a-z]*)",
                r"môn (.*?)(là|có|gồm|bao gồm|dạy|học)"
            ],
            EntityType.SESSION: [
                r"(session|buổi|buổi học) (\d+)",
                r"(session|buổi|buổi học) (số )?(\d+)"
            ],
            EntityType.CLO: [
                r"(clo|chuẩn đầu ra) (\d+)",
                r"(clo|chuẩn đầu ra) (số )?(\d+)"
            ],
            EntityType.ASSESSMENT: [
                r"(bài kiểm tra|bài thi|đánh giá|assessment) (\d+)",
                r"(bài kiểm tra|bài thi|đánh giá|assessment) (số )?(\d+)"
            ],
            EntityType.MATERIAL: [
                r"(tài liệu|material|giáo trình) (.*?)"
            ]
        }
        
        # Từ điển các từ khóa để nhận diện thuộc tính
        self.attribute_keywords = {
            "tín chỉ": ["tín chỉ", "credit", "số tín", "credits"],
            "mô tả": ["mô tả", "description", "giới thiệu", "tổng quan"],
            "điểm": ["điểm", "điểm số", "điểm đánh giá", "điểm thi", "điểm kiểm tra", "grade", "grading"],
            "thời lượng": ["thời lượng", "thời gian", "duration", "time"],
            "đánh giá": ["đánh giá", "assessment", "kiểm tra", "thi", "exam", "test"],
            "tài liệu": ["tài liệu", "material", "giáo trình", "sách", "book"],
            "tiên quyết": ["tiên quyết", "prerequisite", "điều kiện", "yêu cầu"],
            "giảng viên": ["giảng viên", "giáo viên", "lecturer", "teacher", "instructor"],
            "coursera": ["coursera", "khóa học online", "online course"],
            "link": ["link", "đường dẫn", "url", "website"]
        }
        
        # Từ điển các từ khóa để nhận diện mối quan hệ
        self.relationship_keywords = {
            "course_session": ["session của", "buổi học của", "buổi của"],
            "course_clo": ["clo của", "chuẩn đầu ra của"],
            "course_assessment": ["đánh giá của", "kiểm tra của", "thi của"],
            "course_material": ["tài liệu của", "giáo trình của", "sách của"],
            "session_clo": ["clo trong session", "chuẩn đầu ra trong buổi"],
            "clo_assessment": ["đánh giá cho clo", "kiểm tra cho chuẩn đầu ra"]
        }
        
        # Từ điển các từ khóa để nhận diện truy vấn tổng hợp
        self.aggregation_keywords = {
            "count": ["bao nhiêu", "mấy", "số lượng", "đếm", "count", "how many"],
            "sum": ["tổng", "sum", "total"],
            "average": ["trung bình", "average", "mean"],
            "max": ["lớn nhất", "tối đa", "max", "maximum"],
            "min": ["nhỏ nhất", "tối thiểu", "min", "minimum"]
        }
        
        # Từ điển các từ khóa để nhận diện truy vấn phân loại
        self.classification_keywords = {
            "subject_area": ["toán", "lập trình", "marketing", "kinh tế", "quản trị", "tiếng anh"],
            "credit_count": ["tín chỉ", "credit"],
            "alphabet": ["chữ", "ký tự", "kí tự", "bắt đầu bằng", "kết thúc bằng"]
        }

    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """
        Phân tích truy vấn người dùng và trả về thông tin chi tiết.
        
        Args:
            query_text: Câu truy vấn của người dùng
            
        Returns:
            Dict chứa thông tin phân tích truy vấn:
            - query_type: Loại truy vấn
            - entities: Danh sách các thực thể được nhận diện
            - attributes: Danh sách các thuộc tính được nhận diện
            - relationships: Danh sách các mối quan hệ được nhận diện
            - aggregations: Danh sách các phép tổng hợp được nhận diện
            - classifications: Danh sách các phân loại được nhận diện
            - structured_query: Truy vấn có cấu trúc
        """
        # Chuẩn hóa truy vấn
        normalized_query = self._normalize_query(query_text)
        
        # Xác định loại truy vấn
        query_type = self._identify_query_type(normalized_query)
        
        # Trích xuất thực thể
        entities = self._extract_entities(normalized_query)
        
        # Trích xuất thuộc tính
        attributes = self._extract_attributes(normalized_query)
        
        # Trích xuất mối quan hệ
        relationships = self._extract_relationships(normalized_query)
        
        # Trích xuất phép tổng hợp
        aggregations = self._extract_aggregations(normalized_query)
        
        # Trích xuất phân loại
        classifications = self._extract_classifications(normalized_query)
        
        # Tạo truy vấn có cấu trúc
        structured_query = self._create_structured_query(
            query_type, entities, attributes, relationships, aggregations, classifications
        )
        
        return {
            "original_query": query_text,
            "normalized_query": normalized_query,
            "query_type": query_type,
            "entities": entities,
            "attributes": attributes,
            "relationships": relationships,
            "aggregations": aggregations,
            "classifications": classifications,
            "structured_query": structured_query
        }
    
    def _normalize_query(self, query_text: str) -> str:
        """
        Chuẩn hóa truy vấn: chuyển về chữ thường, loại bỏ dấu câu thừa.
        
        Args:
            query_text: Câu truy vấn gốc
            
        Returns:
            Câu truy vấn đã chuẩn hóa
        """
        # Chuyển về chữ thường
        query = query_text.lower()
        
        # Loại bỏ dấu câu thừa
        query = re.sub(r'[^\w\s\d]', ' ', query)
        
        # Loại bỏ khoảng trắng thừa
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def _identify_query_type(self, normalized_query: str) -> str:
        """
        Xác định loại truy vấn dựa trên các pattern đã định nghĩa.
        
        Args:
            normalized_query: Câu truy vấn đã chuẩn hóa
            
        Returns:
            Loại truy vấn (một trong các giá trị của QueryType)
        """
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, normalized_query):
                    return query_type
        
        # Mặc định là truy vấn thông tin đơn giản
        return QueryType.SIMPLE_INFO
    
    def _extract_entities(self, normalized_query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Trích xuất các thực thể từ truy vấn.
        
        Args:
            normalized_query: Câu truy vấn đã chuẩn hóa
            
        Returns:
            Dict chứa danh sách các thực thể theo loại
        """
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            entities[entity_type] = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, normalized_query)
                
                for match in matches:
                    if entity_type == EntityType.COURSE:
                        # Xử lý đặc biệt cho mã môn học
                        if len(match.groups()) >= 1:
                            course_code = match.group(1)
                            # Kiểm tra xem có phải mã môn học hợp lệ không
                            if re.match(r'^[A-Za-z]{2,}[0-9]{3}[a-z]*$', course_code):
                                entities[entity_type].append({
                                    "value": course_code.upper(),
                                    "confidence": 0.9
                                })
                            else:
                                # Có thể là tên môn học
                                entities[entity_type].append({
                                    "value": course_code,
                                    "confidence": 0.6
                                })
                    elif entity_type == EntityType.SESSION:
                        # Xử lý đặc biệt cho session
                        if len(match.groups()) >= 2:
                            session_number = match.group(2)
                            entities[entity_type].append({
                                "value": int(session_number) if session_number is not None else None,
                                "confidence": 0.9
                            })
                    elif entity_type == EntityType.CLO:
                        # Xử lý đặc biệt cho CLO
                        if len(match.groups()) >= 2:
                            clo_number = match.group(2)
                            entities[entity_type].append({
                                "value": int(clo_number),
                                "confidence": 0.9
                            })
                    else:
                        # Xử lý chung cho các loại thực thể khác
                        if len(match.groups()) >= 1:
                            entity_value = match.group(1)
                            entities[entity_type].append({
                                "value": entity_value,
                                "confidence": 0.8
                            })
        
        return entities
    
    def _extract_attributes(self, normalized_query: str) -> List[Dict[str, Any]]:
        """
        Trích xuất các thuộc tính từ truy vấn.
        
        Args:
            normalized_query: Câu truy vấn đã chuẩn hóa
            
        Returns:
            Danh sách các thuộc tính được nhận diện
        """
        attributes = []
        
        for attr_name, keywords in self.attribute_keywords.items():
            for keyword in keywords:
                if keyword in normalized_query:
                    attributes.append({
                        "name": attr_name,
                        "value": keyword,
                        "confidence": 0.8
                    })
                    break  # Tìm thấy một keyword là đủ
        
        return attributes
    
    def _extract_relationships(self, normalized_query: str) -> List[Dict[str, Any]]:
        """
        Trích xuất các mối quan hệ từ truy vấn.
        
        Args:
            normalized_query: Câu truy vấn đã chuẩn hóa
            
        Returns:
            Danh sách các mối quan hệ được nhận diện
        """
        relationships = []
        
        for rel_name, keywords in self.relationship_keywords.items():
            for keyword in keywords:
                if keyword in normalized_query:
                    relationships.append({
                        "name": rel_name,
                        "value": keyword,
                        "confidence": 0.8
                    })
                    break  # Tìm thấy một keyword là đủ
        
        return relationships
    
    def _extract_aggregations(self, normalized_query: str) -> List[Dict[str, Any]]:
        """
        Trích xuất các phép tổng hợp từ truy vấn.
        
        Args:
            normalized_query: Câu truy vấn đã chuẩn hóa
            
        Returns:
            Danh sách các phép tổng hợp được nhận diện
        """
        aggregations = []
        
        for agg_name, keywords in self.aggregation_keywords.items():
            for keyword in keywords:
                if keyword in normalized_query:
                    aggregations.append({
                        "name": agg_name,
                        "value": keyword,
                        "confidence": 0.8
                    })
                    break  # Tìm thấy một keyword là đủ
        
        return aggregations
    
    def _extract_classifications(self, normalized_query: str) -> List[Dict[str, Any]]:
        """
        Trích xuất các phân loại từ truy vấn.
        
        Args:
            normalized_query: Câu truy vấn đã chuẩn hóa
            
        Returns:
            Danh sách các phân loại được nhận diện
        """
        classifications = []
        
        for class_name, keywords in self.classification_keywords.items():
            for keyword in keywords:
                if keyword in normalized_query:
                    classifications.append({
                        "name": class_name,
                        "value": keyword,
                        "confidence": 0.8
                    })
                    break  # Tìm thấy một keyword là đủ
        
        # Xử lý đặc biệt cho phân loại theo chữ cái
        alphabet_match = re.search(r'(kết thúc|bắt đầu) (bằng|với) (chữ|ký tự|kí tự) [\'"]([a-zA-Z])[\'"]', normalized_query)
        if alphabet_match:
            position = alphabet_match.group(1)  # "kết thúc" hoặc "bắt đầu"
            character = alphabet_match.group(4)  # Ký tự
            
            classifications.append({
                "name": "alphabet",
                "value": character,
                "position": position,
                "confidence": 0.9
            })
        
        return classifications
    
    def _create_structured_query(
        self, 
        query_type: str, 
        entities: Dict[str, List[Dict[str, Any]]], 
        attributes: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]], 
        aggregations: List[Dict[str, Any]], 
        classifications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Tạo truy vấn có cấu trúc từ các thành phần đã phân tích.
        
        Args:
            query_type: Loại truy vấn
            entities: Danh sách các thực thể
            attributes: Danh sách các thuộc tính
            relationships: Danh sách các mối quan hệ
            aggregations: Danh sách các phép tổng hợp
            classifications: Danh sách các phân loại
            
        Returns:
            Truy vấn có cấu trúc
        """
        structured_query = {
            "type": query_type,
            "target": {}
        }
        
        # Xử lý theo loại truy vấn
        if query_type == QueryType.SIMPLE_INFO:
            # Truy vấn thông tin đơn giản
            if entities.get(EntityType.COURSE):
                structured_query["target"] = {
                    "entity_type": EntityType.COURSE,
                    "entity_value": entities[EntityType.COURSE][0]["value"]
                }
            
        elif query_type == QueryType.ATTRIBUTE:
            # Truy vấn thuộc tính
            if entities.get(EntityType.COURSE) and attributes:
                structured_query["target"] = {
                    "entity_type": EntityType.COURSE,
                    "entity_value": entities[EntityType.COURSE][0]["value"],
                    "attribute": attributes[0]["name"]
                }
            
        elif query_type == QueryType.RELATIONSHIP:
            # Truy vấn quan hệ
            if entities.get(EntityType.SESSION) and entities.get(EntityType.COURSE):
                structured_query["target"] = {
                    "source_type": EntityType.COURSE,
                    "source_value": entities[EntityType.COURSE][0]["value"],
                    "target_type": EntityType.SESSION,
                    "target_value": entities[EntityType.SESSION][0]["value"],
                    "relationship": "has_session"
                }
            elif entities.get(EntityType.SESSION):
                structured_query["target"] = {
                    "target_type": EntityType.SESSION,
                    "target_value": entities[EntityType.SESSION][0]["value"],
                    "relationship": "has_session"
                }
            elif entities.get(EntityType.CLO) and entities.get(EntityType.COURSE):
                structured_query["target"] = {
                    "source_type": EntityType.COURSE,
                    "source_value": entities[EntityType.COURSE][0]["value"],
                    "target_type": EntityType.CLO,
                    "target_value": entities[EntityType.CLO][0]["value"],
                    "relationship": "has_clo"
                }
            elif entities.get(EntityType.CLO):
                structured_query["target"] = {
                    "target_type": EntityType.CLO,
                    "target_value": entities[EntityType.CLO][0]["value"],
                    "relationship": "has_clo"
                }
            
        elif query_type == QueryType.AGGREGATION:
            # Truy vấn tổng hợp
            if aggregations and aggregations[0]["name"] == "count":
                if "session" in entities:
                    structured_query["target"] = {
                        "operation": "count",
                        "entity_type": EntityType.SESSION
                    }
                    if entities.get(EntityType.COURSE):
                        structured_query["target"]["filter"] = {
                            "entity_type": EntityType.COURSE,
                            "entity_value": entities[EntityType.COURSE][0]["value"]
                        }
                elif "clo" in entities:
                    structured_query["target"] = {
                        "operation": "count",
                        "entity_type": EntityType.CLO
                    }
                    if entities.get(EntityType.COURSE):
                        structured_query["target"]["filter"] = {
                            "entity_type": EntityType.COURSE,
                            "entity_value": entities[EntityType.COURSE][0]["value"]
                        }
                elif "assessment" in entities:
                    structured_query["target"] = {
                        "operation": "count",
                        "entity_type": EntityType.ASSESSMENT
                    }
                    if entities.get(EntityType.COURSE):
                        structured_query["target"]["filter"] = {
                            "entity_type": EntityType.COURSE,
                            "entity_value": entities[EntityType.COURSE][0]["value"]
                        }
                elif "material" in entities:
                    structured_query["target"] = {
                        "operation": "count",
                        "entity_type": EntityType.MATERIAL
                    }
                    if entities.get(EntityType.COURSE):
                        structured_query["target"]["filter"] = {
                            "entity_type": EntityType.COURSE,
                            "entity_value": entities[EntityType.COURSE][0]["value"]
                        }
                elif "course" in entities:
                    structured_query["target"] = {
                        "operation": "count",
                        "entity_type": EntityType.COURSE
                    }
                    if classifications:
                        structured_query["target"]["filter"] = {
                            "classification": classifications[0]["name"],
                            "value": classifications[0]["value"]
                        }
            
        elif query_type == QueryType.CLASSIFICATION:
            # Truy vấn phân loại
            if classifications:
                structured_query["target"] = {
                    "operation": "filter",
                    "entity_type": EntityType.COURSE,
                    "classification": classifications[0]["name"],
                    "value": classifications[0]["value"]
                }
                if classifications[0]["name"] == "alphabet" and "position" in classifications[0]:
                    structured_query["target"]["position"] = classifications[0]["position"]
            
        elif query_type == QueryType.LINKING:
            # Truy vấn liên kết
            if "link" in attributes or "coursera" in attributes:
                structured_query["target"] = {
                    "operation": "filter",
                    "entity_type": EntityType.MATERIAL
                }
                if "coursera" in attributes:
                    structured_query["target"]["filter"] = {
                        "attribute": "is_coursera",
                        "value": True
                    }
                elif "link" in attributes:
                    structured_query["target"]["filter"] = {
                        "attribute": "url",
                        "exists": True
                    }
        
        return structured_query
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Tạo embedding cho văn bản.
        
        Args:
            text: Văn bản cần tạo embedding
            
        Returns:
            Vector embedding
        """
        return self.embedding_model.encode(text)
    
    def find_similar_entities(self, query_text: str, entity_type: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm các thực thể tương tự với truy vấn.
        
        Args:
            query_text: Câu truy vấn
            entity_type: Loại thực thể cần tìm (nếu None thì tìm tất cả)
            top_k: Số lượng kết quả trả về
            
        Returns:
            Danh sách các thực thể tương tự
        """
        if not self.entity_data:
            return []
        
        query_embedding = self.get_embedding(query_text)
        
        results = []
        for entity_id, entity in self.entity_data.items():
            if entity_type and entity.get("entity_type") != entity_type:
                continue
                
            if "embedding" in entity:
                similarity = np.dot(query_embedding, entity["embedding"])
                results.append({
                    "entity_id": entity_id,
                    "entity": entity,
                    "similarity": float(similarity)
                })
        
        # Sắp xếp theo độ tương tự giảm dần
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]


class EntityLinker:
    """
    Lớp liên kết thực thể, xác định và liên kết các thực thể trong truy vấn
    với các thực thể trong cơ sở dữ liệu.
    """
    
    def __init__(self, entity_data: Dict[str, Any] = None):
        """
        Khởi tạo entity linker với dữ liệu thực thể.
        
        Args:
            entity_data: Dữ liệu thực thể đã được xử lý (nếu có)
        """
        self.entity_data = entity_data or {}
        
        # Tạo các chỉ mục để tìm kiếm nhanh
        self._build_indices()
    
    def _build_indices(self):
        """Xây dựng các chỉ mục để tìm kiếm nhanh."""
        # Chỉ mục theo loại thực thể
        self.entity_type_index = {}
        
        # Chỉ mục theo mã môn học
        self.subject_code_index = {}
        
        # Chỉ mục theo session
        self.session_index = {}
        
        # Chỉ mục theo CLO
        self.clo_index = {}
        
        # Chỉ mục theo tài liệu
        self.material_index = {}
        
        # Xây dựng các chỉ mục
        for entity_id, entity in self.entity_data.items():
            entity_type = entity.get("entity_type")
            
            # Thêm vào chỉ mục theo loại thực thể
            if entity_type not in self.entity_type_index:
                self.entity_type_index[entity_type] = []
            self.entity_type_index[entity_type].append(entity_id)
            
            # Thêm vào chỉ mục theo mã môn học
            if entity_type == EntityType.COURSE:
                subject_code = entity.get("subject_code")
                if subject_code:
                    self.subject_code_index[subject_code] = entity_id
            
            # Thêm vào chỉ mục theo session
            elif entity_type == EntityType.SESSION:
                subject_code = entity.get("subject_code")
                session_number = entity.get("session_number")
                if subject_code and session_number:
                    key = f"{subject_code}_{session_number}"
                    self.session_index[key] = entity_id
            
            # Thêm vào chỉ mục theo CLO
            elif entity_type == EntityType.CLO:
                subject_code = entity.get("subject_code")
                clo_name = entity.get("clo_name")
                if subject_code and clo_name:
                    key = f"{subject_code}_{clo_name}"
                    self.clo_index[key] = entity_id
            
            # Thêm vào chỉ mục theo tài liệu
            elif entity_type == EntityType.MATERIAL:
                subject_code = entity.get("subject_code")
                if subject_code:
                    if subject_code not in self.material_index:
                        self.material_index[subject_code] = []
                    self.material_index[subject_code].append(entity_id)
    
    def link_entities(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Liên kết các thực thể trong phân tích truy vấn với các thực thể trong cơ sở dữ liệu.
        
        Args:
            query_analysis: Kết quả phân tích truy vấn từ ContextualQueryProcessor
            
        Returns:
            Phân tích truy vấn đã được bổ sung thông tin liên kết thực thể
        """
        linked_analysis = query_analysis.copy()
        
        # Liên kết các thực thể
        linked_entities = {}
        
        for entity_type, entities in query_analysis.get("entities", {}).items():
            linked_entities[entity_type] = []
            
            for entity in entities:
                linked_entity = entity.copy()
                
                # Liên kết thực thể dựa trên loại
                if entity_type == EntityType.COURSE:
                    entity_value = entity.get("value")
                    if entity_value in self.subject_code_index:
                        entity_id = self.subject_code_index[entity_value]
                        linked_entity["entity_id"] = entity_id
                        linked_entity["linked"] = True
                        linked_entity["entity_data"] = self.entity_data.get(entity_id)
                
                elif entity_type == EntityType.SESSION:
                    # Cần có thông tin môn học để liên kết session
                    course_entities = query_analysis.get("entities", {}).get(EntityType.COURSE, [])
                    if course_entities:
                        subject_code = course_entities[0].get("value")
                        session_number = entity.get("value")
                        key = f"{subject_code}_{session_number}"
                        
                        if key in self.session_index:
                            entity_id = self.session_index[key]
                            linked_entity["entity_id"] = entity_id
                            linked_entity["linked"] = True
                            linked_entity["entity_data"] = self.entity_data.get(entity_id)
                
                elif entity_type == EntityType.CLO:
                    # Cần có thông tin môn học để liên kết CLO
                    course_entities = query_analysis.get("entities", {}).get(EntityType.COURSE, [])
                    if course_entities:
                        subject_code = course_entities[0].get("value")
                        clo_name = f"CLO{entity.get('value')}"
                        key = f"{subject_code}_{clo_name}"
                        
                        if key in self.clo_index:
                            entity_id = self.clo_index[key]
                            linked_entity["entity_id"] = entity_id
                            linked_entity["linked"] = True
                            linked_entity["entity_data"] = self.entity_data.get(entity_id)
                
                linked_entities[entity_type].append(linked_entity)
        
        linked_analysis["linked_entities"] = linked_entities
        
        # Cập nhật truy vấn có cấu trúc với thông tin liên kết
        structured_query = query_analysis.get("structured_query", {})
        
        if "target" in structured_query:
            target = structured_query["target"]
            
            # Cập nhật thông tin thực thể trong target
            if "entity_type" in target and target["entity_type"] == EntityType.COURSE:
                entity_value = target.get("entity_value")
                if entity_value in self.subject_code_index:
                    target["entity_id"] = self.subject_code_index[entity_value]
            
            # Cập nhật thông tin quan hệ trong target
            if "source_type" in target and "target_type" in target:
                if target["source_type"] == EntityType.COURSE and target["target_type"] == EntityType.SESSION:
                    subject_code = target.get("source_value")
                    session_number = target.get("target_value")
                    key = f"{subject_code}_{session_number}"
                    
                    if key in self.session_index:
                        target["target_id"] = self.session_index[key]
                
                elif target["source_type"] == EntityType.COURSE and target["target_type"] == EntityType.CLO:
                    subject_code = target.get("source_value")
                    clo_name = f"CLO{target.get('target_value')}"
                    key = f"{subject_code}_{clo_name}"
                    
                    if key in self.clo_index:
                        target["target_id"] = self.clo_index[key]
        
        linked_analysis["structured_query"] = structured_query
        
        return linked_analysis


class QueryExecutor:
    """
    Lớp thực thi truy vấn, thực hiện các truy vấn có cấu trúc và trả về kết quả.
    """
    
    def __init__(self, entity_data: Dict[str, Any] = None, faiss_index = None, chunks_data: List[Dict[str, Any]] = None):
        """
        Khởi tạo query executor với dữ liệu thực thể và index.
        
        Args:
            entity_data: Dữ liệu thực thể đã được xử lý (nếu có)
            faiss_index: FAISS index để tìm kiếm vector
            chunks_data: Dữ liệu chunks
        """
        self.entity_data = entity_data or {}
        self.faiss_index = faiss_index
        self.chunks_data = chunks_data or []
        
        # Tạo các chỉ mục để tìm kiếm nhanh
        self._build_indices()
    
    def _build_indices(self):
        """Xây dựng các chỉ mục để tìm kiếm nhanh."""
        # Chỉ mục theo loại thực thể
        self.entity_type_index = {}
        
        # Chỉ mục theo mã môn học
        self.subject_code_index = {}
        
        # Chỉ mục theo session
        self.session_index = {}
        
        # Chỉ mục theo CLO
        self.clo_index = {}
        
        # Chỉ mục theo tài liệu
        self.material_index = {}
        
        # Chỉ mục theo đánh giá
        self.assessment_index = {}
        
        # Xây dựng các chỉ mục từ entity_data
        if self.entity_data:
            for entity_id, entity in self.entity_data.items():
                entity_type = entity.get("entity_type")
                
                # Thêm vào chỉ mục theo loại thực thể
                if entity_type not in self.entity_type_index:
                    self.entity_type_index[entity_type] = []
                self.entity_type_index[entity_type].append(entity_id)
                
                # Thêm vào chỉ mục theo mã môn học
                if entity_type == EntityType.COURSE:
                    subject_code = entity.get("subject_code")
                    if subject_code:
                        self.subject_code_index[subject_code] = entity_id
                
                # Thêm vào chỉ mục theo session
                elif entity_type == EntityType.SESSION:
                    subject_code = entity.get("subject_code")
                    session_number = entity.get("session_number")
                    if subject_code and session_number:
                        key = f"{subject_code}_{session_number}"
                        self.session_index[key] = entity_id
                
                # Thêm vào chỉ mục theo CLO
                elif entity_type == EntityType.CLO:
                    subject_code = entity.get("subject_code")
                    clo_name = entity.get("clo_name")
                    if subject_code and clo_name:
                        key = f"{subject_code}_{clo_name}"
                        self.clo_index[key] = entity_id
                
                # Thêm vào chỉ mục theo tài liệu
                elif entity_type == EntityType.MATERIAL:
                    subject_code = entity.get("subject_code")
                    if subject_code:
                        if subject_code not in self.material_index:
                            self.material_index[subject_code] = []
                        self.material_index[subject_code].append(entity_id)
                
                # Thêm vào chỉ mục theo đánh giá
                elif entity_type == EntityType.ASSESSMENT:
                    subject_code = entity.get("subject_code")
                    if subject_code:
                        if subject_code not in self.assessment_index:
                            self.assessment_index[subject_code] = []
                        self.assessment_index[subject_code].append(entity_id)
        
        # Xây dựng các chỉ mục từ chunks_data
        if self.chunks_data:
            # Chỉ mục chunk theo loại
            self.chunk_type_index = {}
            
            # Chỉ mục chunk theo mã môn học
            self.chunk_subject_index = {}
            
            # Chỉ mục chunk theo session
            self.chunk_session_index = {}
            
            # Chỉ mục chunk theo CLO
            self.chunk_clo_index = {}
            
            for i, chunk in enumerate(self.chunks_data):
                chunk_type = chunk.get("type")
                
                # Thêm vào chỉ mục theo loại chunk
                if chunk_type not in self.chunk_type_index:
                    self.chunk_type_index[chunk_type] = []
                self.chunk_type_index[chunk_type].append(i)
                
                # Thêm vào chỉ mục theo mã môn học
                if "metadata" in chunk and "subject_code" in chunk["metadata"]:
                    subject_code = chunk["metadata"]["subject_code"]
                    if subject_code not in self.chunk_subject_index:
                        self.chunk_subject_index[subject_code] = []
                    self.chunk_subject_index[subject_code].append(i)
                
                # Thêm vào chỉ mục theo session
                if chunk_type == "session" and "metadata" in chunk and "session_number" in chunk["metadata"]:
                    subject_code = chunk["metadata"].get("subject_code")
                    session_number = chunk["metadata"]["session_number"]
                    if subject_code:
                        key = f"{subject_code}_{session_number}"
                        self.chunk_session_index[key] = i
                
                # Thêm vào chỉ mục theo CLO
                if chunk_type == "clo" and "metadata" in chunk and "clo_id" in chunk["metadata"]:
                    subject_code = chunk["metadata"].get("subject_code")
                    clo_id = chunk["metadata"]["clo_id"]
                    if subject_code:
                        key = f"{subject_code}_{clo_id}"
                        self.chunk_clo_index[key] = i
    
    def execute_query(self, structured_query: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Thực thi truy vấn có cấu trúc và trả về kết quả.
        
        Args:
            structured_query: Truy vấn có cấu trúc từ ContextualQueryProcessor
            top_k: Số lượng kết quả trả về
            
        Returns:
            Kết quả truy vấn
        """
        query_type = structured_query.get("type")
        target = structured_query.get("target", {})
        
        if query_type == QueryType.SIMPLE_INFO:
            return self._execute_simple_info_query(target)
            
        elif query_type == QueryType.ATTRIBUTE:
            return self._execute_attribute_query(target)
            
        elif query_type == QueryType.RELATIONSHIP:
            return self._execute_relationship_query(target)
            
        elif query_type == QueryType.AGGREGATION:
            return self._execute_aggregation_query(target)
            
        elif query_type == QueryType.CLASSIFICATION:
            return self._execute_classification_query(target)
            
        elif query_type == QueryType.LINKING:
            return self._execute_linking_query(target)
            
        else:
            return {
                "success": False,
                "error": "Unsupported query type",
                "results": []
            }
    
    def _execute_simple_info_query(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi truy vấn thông tin đơn giản.
        
        Args:
            target: Thông tin mục tiêu truy vấn
            
        Returns:
            Kết quả truy vấn
        """
        entity_type = target.get("entity_type")
        entity_value = target.get("entity_value")
        entity_id = target.get("entity_id")
        
        if entity_type == EntityType.COURSE:
            # Tìm kiếm thông tin môn học
            if entity_id and entity_id in self.entity_data:
                return {
                    "success": True,
                    "entity_type": entity_type,
                    "entity_value": entity_value,
                    "entity_id": entity_id,
                    "results": [self.entity_data[entity_id]]
                }
            elif entity_value in self.subject_code_index:
                entity_id = self.subject_code_index[entity_value]
                return {
                    "success": True,
                    "entity_type": entity_type,
                    "entity_value": entity_value,
                    "entity_id": entity_id,
                    "results": [self.entity_data[entity_id]]
                }
            
            # Tìm kiếm trong chunks
            if entity_value in self.chunk_subject_index:
                chunk_indices = self.chunk_subject_index[entity_value]
                chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] in ["general_info", "overview"]]
                
                if chunks:
                    return {
                        "success": True,
                        "entity_type": entity_type,
                        "entity_value": entity_value,
                        "results": chunks
                    }
        
        return {
            "success": False,
            "error": "Entity not found",
            "results": []
        }
    
    def _execute_attribute_query(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi truy vấn thuộc tính.
        
        Args:
            target: Thông tin mục tiêu truy vấn
            
        Returns:
            Kết quả truy vấn
        """
        entity_type = target.get("entity_type")
        entity_value = target.get("entity_value")
        entity_id = target.get("entity_id")
        attribute = target.get("attribute")
        
        if entity_type == EntityType.COURSE and attribute:
            # Tìm kiếm thuộc tính của môn học
            if entity_id and entity_id in self.entity_data:
                entity = self.entity_data[entity_id]
                if attribute in entity:
                    return {
                        "success": True,
                        "entity_type": entity_type,
                        "entity_value": entity_value,
                        "entity_id": entity_id,
                        "attribute": attribute,
                        "results": [{
                            "entity": entity,
                            "attribute_value": entity[attribute]
                        }]
                    }
            elif entity_value in self.subject_code_index:
                entity_id = self.subject_code_index[entity_value]
                entity = self.entity_data[entity_id]
                if attribute in entity:
                    return {
                        "success": True,
                        "entity_type": entity_type,
                        "entity_value": entity_value,
                        "entity_id": entity_id,
                        "attribute": attribute,
                        "results": [{
                            "entity": entity,
                            "attribute_value": entity[attribute]
                        }]
                    }
            
            # Tìm kiếm trong chunks
            if entity_value in self.chunk_subject_index:
                chunk_indices = self.chunk_subject_index[entity_value]
                
                # Lọc chunks theo thuộc tính
                attribute_chunks = []
                for i in chunk_indices:
                    chunk = self.chunks_data[i]
                    content = chunk.get("content", "").lower()
                    
                    # Kiểm tra xem nội dung chunk có chứa thông tin về thuộc tính không
                    if attribute == "tín chỉ" and any(keyword in content for keyword in ["tín chỉ", "credit", "số tín"]):
                        attribute_chunks.append(chunk)
                    elif attribute == "mô tả" and any(keyword in content for keyword in ["mô tả", "description", "giới thiệu"]):
                        attribute_chunks.append(chunk)
                    elif attribute == "điểm" and any(keyword in content for keyword in ["điểm", "đánh giá", "grading", "grade"]):
                        attribute_chunks.append(chunk)
                    elif attribute == "thời lượng" and any(keyword in content for keyword in ["thời lượng", "thời gian", "duration"]):
                        attribute_chunks.append(chunk)
                    elif attribute == "đánh giá" and any(keyword in content for keyword in ["đánh giá", "assessment", "kiểm tra", "thi"]):
                        attribute_chunks.append(chunk)
                    elif attribute == "tài liệu" and any(keyword in content for keyword in ["tài liệu", "material", "giáo trình"]):
                        attribute_chunks.append(chunk)
                    elif attribute == "tiên quyết" and any(keyword in content for keyword in ["tiên quyết", "prerequisite", "điều kiện"]):
                        attribute_chunks.append(chunk)
                
                if attribute_chunks:
                    return {
                        "success": True,
                        "entity_type": entity_type,
                        "entity_value": entity_value,
                        "attribute": attribute,
                        "results": attribute_chunks
                    }
        
        return {
            "success": False,
            "error": "Attribute not found",
            "results": []
        }
    
    def _execute_relationship_query(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi truy vấn quan hệ.
        
        Args:
            target: Thông tin mục tiêu truy vấn
            
        Returns:
            Kết quả truy vấn
        """
        source_type = target.get("source_type")
        source_value = target.get("source_value")
        target_type = target.get("target_type")
        target_value = target.get("target_value")
        relationship = target.get("relationship")
        
        if source_type == EntityType.COURSE and target_type == EntityType.SESSION:
            # Tìm kiếm thông tin session của môn học
            key = f"{source_value}_{target_value}"
            
            # Tìm trong entity_data
            if key in self.session_index:
                entity_id = self.session_index[key]
                return {
                    "success": True,
                    "source_type": source_type,
                    "source_value": source_value,
                    "target_type": target_type,
                    "target_value": target_value,
                    "relationship": relationship,
                    "results": [self.entity_data[entity_id]]
                }
            
            # Tìm trong chunks
            if key in self.chunk_session_index:
                chunk_index = self.chunk_session_index[key]
                return {
                    "success": True,
                    "source_type": source_type,
                    "source_value": source_value,
                    "target_type": target_type,
                    "target_value": target_value,
                    "relationship": relationship,
                    "results": [self.chunks_data[chunk_index]]
                }
        
        elif source_type == EntityType.COURSE and target_type == EntityType.CLO:
            # Tìm kiếm thông tin CLO của môn học
            clo_name = f"CLO{target_value}"
            key = f"{source_value}_{clo_name}"
            
            # Tìm trong entity_data
            if key in self.clo_index:
                entity_id = self.clo_index[key]
                return {
                    "success": True,
                    "source_type": source_type,
                    "source_value": source_value,
                    "target_type": target_type,
                    "target_value": target_value,
                    "relationship": relationship,
                    "results": [self.entity_data[entity_id]]
                }
            
            # Tìm trong chunks
            if key in self.chunk_clo_index:
                chunk_index = self.chunk_clo_index[key]
                return {
                    "success": True,
                    "source_type": source_type,
                    "source_value": source_value,
                    "target_type": target_type,
                    "target_value": target_value,
                    "relationship": relationship,
                    "results": [self.chunks_data[chunk_index]]
                }
        
        elif target_type == EntityType.SESSION:
            # Tìm kiếm thông tin session mà không có thông tin môn học
            # Cần tìm kiếm trong tất cả các môn học
            session_chunks = []
            
            for subject_code in self.chunk_subject_index:
                key = f"{subject_code}_{target_value}"
                if key in self.chunk_session_index:
                    chunk_index = self.chunk_session_index[key]
                    session_chunks.append(self.chunks_data[chunk_index])
            
            if session_chunks:
                return {
                    "success": True,
                    "target_type": target_type,
                    "target_value": target_value,
                    "relationship": relationship,
                    "results": session_chunks
                }
        
        elif target_type == EntityType.CLO:
            # Tìm kiếm thông tin CLO mà không có thông tin môn học
            # Cần tìm kiếm trong tất cả các môn học
            clo_name = f"CLO{target_value}"
            clo_chunks = []
            
            for subject_code in self.chunk_subject_index:
                key = f"{subject_code}_{clo_name}"
                if key in self.chunk_clo_index:
                    chunk_index = self.chunk_clo_index[key]
                    clo_chunks.append(self.chunks_data[chunk_index])
            
            if clo_chunks:
                return {
                    "success": True,
                    "target_type": target_type,
                    "target_value": target_value,
                    "relationship": relationship,
                    "results": clo_chunks
                }
        
        return {
            "success": False,
            "error": "Relationship not found",
            "results": []
        }
    
    def _execute_aggregation_query(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi truy vấn tổng hợp.
        
        Args:
            target: Thông tin mục tiêu truy vấn
            
        Returns:
            Kết quả truy vấn
        """
        operation = target.get("operation")
        entity_type = target.get("entity_type")
        filter_info = target.get("filter", {})
        
        if operation == "count":
            if entity_type == EntityType.SESSION:
                # Đếm số lượng session
                filter_entity_type = filter_info.get("entity_type")
                filter_entity_value = filter_info.get("entity_value")
                
                if filter_entity_type == EntityType.COURSE:
                    # Đếm số lượng session của một môn học cụ thể
                    session_count = 0
                    
                    # Tìm trong entity_data
                    if filter_entity_value in self.subject_code_index:
                        entity_id = self.subject_code_index[filter_entity_value]
                        entity = self.entity_data[entity_id]
                        if "related_entities" in entity and "sessions" in entity["related_entities"]:
                            session_count = len(entity["related_entities"]["sessions"])
                    
                    # Tìm trong chunks
                    if filter_entity_value in self.chunk_subject_index:
                        chunk_indices = self.chunk_subject_index[filter_entity_value]
                        session_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] == "session"]
                        
                        if session_count == 0:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                            session_count = len(session_chunks)
                        
                        # Tìm trong metadata của chunk tổng quan
                        overview_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] in ["overview", "structure"]]
                        for chunk in overview_chunks:
                            if "metadata" in chunk and "total_sessions" in chunk["metadata"]:
                                session_count = chunk["metadata"]["total_sessions"]
                                break
                    
                    return {
                        "success": True,
                        "operation": operation,
                        "entity_type": entity_type,
                        "filter": filter_info,
                        "count": session_count,
                        "results": [{"count": session_count}]
                    }
            
            elif entity_type == EntityType.CLO:
                # Đếm số lượng CLO
                filter_entity_type = filter_info.get("entity_type")
                filter_entity_value = filter_info.get("entity_value")
                
                if filter_entity_type == EntityType.COURSE:
                    # Đếm số lượng CLO của một môn học cụ thể
                    clo_count = 0
                    
                    # Tìm trong entity_data
                    if filter_entity_value in self.subject_code_index:
                        entity_id = self.subject_code_index[filter_entity_value]
                        entity = self.entity_data[entity_id]
                        if "related_entities" in entity and "clos" in entity["related_entities"]:
                            clo_count = len(entity["related_entities"]["clos"])
                    
                    # Tìm trong chunks
                    if filter_entity_value in self.chunk_subject_index:
                        chunk_indices = self.chunk_subject_index[filter_entity_value]
                        clo_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] == "clo"]
                        
                        if clo_count == 0:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                            clo_count = len(clo_chunks)
                        
                        # Tìm trong metadata của chunk tổng quan
                        overview_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] in ["overview", "structure"]]
                        for chunk in overview_chunks:
                            if "metadata" in chunk and "total_clos" in chunk["metadata"]:
                                clo_count = chunk["metadata"]["total_clos"]
                                break
                    
                    return {
                        "success": True,
                        "operation": operation,
                        "entity_type": entity_type,
                        "filter": filter_info,
                        "count": clo_count,
                        "results": [{"count": clo_count}]
                    }
            
            elif entity_type == EntityType.ASSESSMENT:
                # Đếm số lượng assessment
                filter_entity_type = filter_info.get("entity_type")
                filter_entity_value = filter_info.get("entity_value")
                
                if filter_entity_type == EntityType.COURSE:
                    # Đếm số lượng assessment của một môn học cụ thể
                    assessment_count = 0
                    
                    # Tìm trong entity_data
                    if filter_entity_value in self.subject_code_index:
                        entity_id = self.subject_code_index[filter_entity_value]
                        entity = self.entity_data[entity_id]
                        if "related_entities" in entity and "assessments" in entity["related_entities"]:
                            assessment_count = len(entity["related_entities"]["assessments"])
                    
                    # Tìm trong chunks
                    if filter_entity_value in self.chunk_subject_index:
                        chunk_indices = self.chunk_subject_index[filter_entity_value]
                        assessment_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] == "assessment"]
                        
                        if assessment_count == 0:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                            assessment_count = len(assessment_chunks)
                    
                    return {
                        "success": True,
                        "operation": operation,
                        "entity_type": entity_type,
                        "filter": filter_info,
                        "count": assessment_count,
                        "results": [{"count": assessment_count}]
                    }
            
            elif entity_type == EntityType.MATERIAL:
                # Đếm số lượng material
                filter_entity_type = filter_info.get("entity_type")
                filter_entity_value = filter_info.get("entity_value")
                
                if filter_entity_type == EntityType.COURSE:
                    # Đếm số lượng material của một môn học cụ thể
                    material_count = 0
                    
                    # Tìm trong entity_data
                    if filter_entity_value in self.subject_code_index:
                        entity_id = self.subject_code_index[filter_entity_value]
                        entity = self.entity_data[entity_id]
                        if "related_entities" in entity and "materials" in entity["related_entities"]:
                            material_count = len(entity["related_entities"]["materials"])
                    
                    # Tìm trong chunks
                    if filter_entity_value in self.chunk_subject_index:
                        chunk_indices = self.chunk_subject_index[filter_entity_value]
                        material_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] == "material"]
                        
                        if material_count == 0:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                            material_count = len(material_chunks)
                        
                        # Tìm trong metadata của chunk tổng quan
                        overview_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] in ["overview", "structure"]]
                        for chunk in overview_chunks:
                            if "metadata" in chunk and "total_materials" in chunk["metadata"]:
                                material_count = chunk["metadata"]["total_materials"]
                                break
                    
                    return {
                        "success": True,
                        "operation": operation,
                        "entity_type": entity_type,
                        "filter": filter_info,
                        "count": material_count,
                        "results": [{"count": material_count}]
                    }
            
            elif entity_type == EntityType.COURSE:
                # Đếm số lượng môn học
                classification = filter_info.get("classification")
                value = filter_info.get("value")
                
                if classification == "subject_area":
                    # Đếm số lượng môn học thuộc một lĩnh vực cụ thể
                    subject_area_courses = []
                    
                    # Tìm trong entity_data
                    for entity_id, entity in self.entity_data.items():
                        if entity.get("entity_type") == EntityType.COURSE:
                            # Kiểm tra xem môn học có thuộc lĩnh vực không
                            description = entity.get("description", "").lower()
                            syllabus_name = entity.get("syllabus_name", "").lower()
                            
                            if value in description or value in syllabus_name:
                                subject_area_courses.append(entity)
                    
                    # Tìm trong chunks
                    if not subject_area_courses:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                        for subject_code in self.chunk_subject_index:
                            chunk_indices = self.chunk_subject_index[subject_code]
                            overview_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] in ["overview", "general_info"]]
                            
                            for chunk in overview_chunks:
                                content = chunk.get("content", "").lower()
                                if value in content:
                                    subject_area_courses.append({
                                        "subject_code": subject_code,
                                        "content": chunk.get("content")
                                    })
                                    break
                    
                    return {
                        "success": True,
                        "operation": operation,
                        "entity_type": entity_type,
                        "filter": filter_info,
                        "count": len(subject_area_courses),
                        "results": subject_area_courses
                    }
        
        return {
            "success": False,
            "error": "Aggregation not supported",
            "results": []
        }
    
    def _execute_classification_query(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi truy vấn phân loại.
        
        Args:
            target: Thông tin mục tiêu truy vấn
            
        Returns:
            Kết quả truy vấn
        """
        operation = target.get("operation")
        entity_type = target.get("entity_type")
        classification = target.get("classification")
        value = target.get("value")
        position = target.get("position")
        
        if operation == "filter" and entity_type == EntityType.COURSE:
            if classification == "alphabet":
                # Lọc môn học theo chữ cái
                filtered_courses = []
                
                # Tìm trong entity_data
                for entity_id, entity in self.entity_data.items():
                    if entity.get("entity_type") == EntityType.COURSE:
                        subject_code = entity.get("subject_code", "")
                        
                        if position == "kết thúc" and subject_code.endswith(value):
                            filtered_courses.append(entity)
                        elif position == "bắt đầu" and subject_code.startswith(value):
                            filtered_courses.append(entity)
                
                # Tìm trong chunks
                if not filtered_courses:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                    for subject_code in self.chunk_subject_index:
                        if position == "kết thúc" and subject_code.endswith(value):
                            chunk_indices = self.chunk_subject_index[subject_code]
                            overview_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] in ["overview", "general_info"]]
                            
                            if overview_chunks:
                                filtered_courses.append({
                                    "subject_code": subject_code,
                                    "content": overview_chunks[0].get("content")
                                })
                        elif position == "bắt đầu" and subject_code.startswith(value):
                            chunk_indices = self.chunk_subject_index[subject_code]
                            overview_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] in ["overview", "general_info"]]
                            
                            if overview_chunks:
                                filtered_courses.append({
                                    "subject_code": subject_code,
                                    "content": overview_chunks[0].get("content")
                                })
                
                return {
                    "success": True,
                    "operation": operation,
                    "entity_type": entity_type,
                    "classification": classification,
                    "value": value,
                    "position": position,
                    "count": len(filtered_courses),
                    "results": filtered_courses
                }
            
            elif classification == "subject_area":
                # Lọc môn học theo lĩnh vực
                filtered_courses = []
                
                # Tìm trong entity_data
                for entity_id, entity in self.entity_data.items():
                    if entity.get("entity_type") == EntityType.COURSE:
                        # Kiểm tra xem môn học có thuộc lĩnh vực không
                        description = entity.get("description", "").lower()
                        syllabus_name = entity.get("syllabus_name", "").lower()
                        
                        if value in description or value in syllabus_name:
                            filtered_courses.append(entity)
                
                # Tìm trong chunks
                if not filtered_courses:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                    for subject_code in self.chunk_subject_index:
                        chunk_indices = self.chunk_subject_index[subject_code]
                        overview_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] in ["overview", "general_info"]]
                        
                        for chunk in overview_chunks:
                            content = chunk.get("content", "").lower()
                            if value in content:
                                filtered_courses.append({
                                    "subject_code": subject_code,
                                    "content": chunk.get("content")
                                })
                                break
                
                return {
                    "success": True,
                    "operation": operation,
                    "entity_type": entity_type,
                    "classification": classification,
                    "value": value,
                    "count": len(filtered_courses),
                    "results": filtered_courses
                }
            
            elif classification == "credit_count":
                # Lọc môn học theo số tín chỉ
                filtered_courses = []
                
                # Tìm trong entity_data
                for entity_id, entity in self.entity_data.items():
                    if entity.get("entity_type") == EntityType.COURSE:
                        credits = entity.get("credits")
                        
                        if credits and str(credits) == str(value):
                            filtered_courses.append(entity)
                
                # Tìm trong chunks
                if not filtered_courses:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                    for subject_code in self.chunk_subject_index:
                        chunk_indices = self.chunk_subject_index[subject_code]
                        overview_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] in ["overview", "general_info"]]
                        
                        for chunk in overview_chunks:
                            content = chunk.get("content", "").lower()
                            if f"tín chỉ: {value}" in content or f"credit: {value}" in content:
                                filtered_courses.append({
                                    "subject_code": subject_code,
                                    "content": chunk.get("content")
                                })
                                break
                
                return {
                    "success": True,
                    "operation": operation,
                    "entity_type": entity_type,
                    "classification": classification,
                    "value": value,
                    "count": len(filtered_courses),
                    "results": filtered_courses
                }
        
        return {
            "success": False,
            "error": "Classification not supported",
            "results": []
        }
    
    def _execute_linking_query(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi truy vấn liên kết.
        
        Args:
            target: Thông tin mục tiêu truy vấn
            
        Returns:
            Kết quả truy vấn
        """
        operation = target.get("operation")
        entity_type = target.get("entity_type")
        filter_info = target.get("filter", {})
        
        if operation == "filter" and entity_type == EntityType.MATERIAL:
            attribute = filter_info.get("attribute")
            value = filter_info.get("value")
            exists = filter_info.get("exists")
            
            if attribute == "is_coursera" and value:
                # Lọc tài liệu Coursera
                coursera_materials = []
                
                # Tìm trong entity_data
                for entity_id, entity in self.entity_data.items():
                    if entity.get("entity_type") == EntityType.MATERIAL:
                        is_coursera = entity.get("is_coursera")
                        
                        if is_coursera:
                            coursera_materials.append(entity)
                
                # Tìm trong chunks
                if not coursera_materials:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                    for subject_code in self.chunk_subject_index:
                        chunk_indices = self.chunk_subject_index[subject_code]
                        material_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] == "material"]
                        
                        for chunk in material_chunks:
                            content = chunk.get("content", "").lower()
                            if "coursera" in content:
                                coursera_materials.append({
                                    "subject_code": subject_code,
                                    "content": chunk.get("content")
                                })
                
                return {
                    "success": True,
                    "operation": operation,
                    "entity_type": entity_type,
                    "filter": filter_info,
                    "count": len(coursera_materials),
                    "results": coursera_materials
                }
            
            elif attribute == "url" and exists:
                # Lọc tài liệu có URL
                url_materials = []
                
                # Tìm trong entity_data
                for entity_id, entity in self.entity_data.items():
                    if entity.get("entity_type") == EntityType.MATERIAL:
                        url = entity.get("url")
                        
                        if url:
                            url_materials.append(entity)
                
                # Tìm trong chunks
                if not url_materials:  # Chỉ sử dụng nếu không tìm thấy trong entity_data
                    for subject_code in self.chunk_subject_index:
                        chunk_indices = self.chunk_subject_index[subject_code]
                        material_chunks = [self.chunks_data[i] for i in chunk_indices if self.chunks_data[i]["type"] == "material"]
                        
                        for chunk in material_chunks:
                            content = chunk.get("content", "").lower()
                            # Tìm URL trong nội dung
                            url_match = re.search(r'https?://\S+', content)
                            if url_match:
                                url_materials.append({
                                    "subject_code": subject_code,
                                    "content": chunk.get("content"),
                                    "url": url_match.group(0)
                                })
                
                return {
                    "success": True,
                    "operation": operation,
                    "entity_type": entity_type,
                    "filter": filter_info,
                    "count": len(url_materials),
                    "results": url_materials
                }
        
        return {
            "success": False,
            "error": "Linking query not supported",
            "results": []
        }


class ContextualSearchEngine:
    """
    Lớp search engine ngữ cảnh, tích hợp các thành phần để xử lý truy vấn ngữ cảnh.
    """
    
    def __init__(self, embedding_model: SentenceTransformer, faiss_index = None, chunks_data: List[Dict[str, Any]] = None, entity_data: Dict[str, Any] = None):
        """
        Khởi tạo search engine với các thành phần cần thiết.
        
        Args:
            embedding_model: Mô hình SentenceTransformer để tạo embedding
            faiss_index: FAISS index để tìm kiếm vector
            chunks_data: Dữ liệu chunks
            entity_data: Dữ liệu thực thể đã được xử lý (nếu có)
        """
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.chunks_data = chunks_data or []
        self.entity_data = entity_data or {}
        
        # Khởi tạo các thành phần
        self.query_processor = ContextualQueryProcessor(embedding_model, entity_data)
        self.entity_linker = EntityLinker(entity_data)
        self.query_executor = QueryExecutor(entity_data, faiss_index, chunks_data)
    
    def search(self, query_text: str, top_k: int = 5, structured_query: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Thực hiện tìm kiếm với truy vấn ngữ cảnh.
        
        Args:
            query_text: Câu truy vấn của người dùng
            top_k: Số lượng kết quả trả về
            structured_query: Truy vấn có cấu trúc được chỉ định (nếu có)
            
        Returns:
            Kết quả tìm kiếm
        """
        # Phân tích truy vấn
        query_analysis = self.query_processor.analyze_query(query_text)
        
        # Liên kết thực thể
        linked_analysis = self.entity_linker.link_entities(query_analysis)
        
        # Thực thi truy vấn
        if structured_query is not None:
            # Sử dụng truy vấn có cấu trúc được chỉ định
            query_results = self.query_executor.execute_query(structured_query, top_k)
        else:
            # Sử dụng truy vấn từ phân tích
            query_results = self.query_executor.execute_query(linked_analysis["structured_query"], top_k)
        
        # Tìm kiếm vector nếu cần
        if not query_results["success"] or not query_results["results"]:
            # Thực hiện tìm kiếm vector
            vector_results = self._search_vector(query_text, top_k)
            
            # Kết hợp kết quả
            if vector_results["success"]:
                if not query_results["success"]:
                    query_results = vector_results
                else:
                    query_results["results"].extend(vector_results["results"])
        
        # Thêm thông tin phân tích vào kết quả
        query_results["query_analysis"] = linked_analysis
        
        return query_results
    
    def _search_vector(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Thực hiện tìm kiếm vector.
        
        Args:
            query_text: Câu truy vấn của người dùng
            top_k: Số lượng kết quả trả về
            
        Returns:
            Kết quả tìm kiếm
        """
        if not self.faiss_index or not self.chunks_data:
            return {
                "success": False,
                "error": "Vector search not available",
                "results": []
            }
        
        # Tạo embedding cho truy vấn
        query_embedding = self.embedding_model.encode([query_text])
        
        # Tìm kiếm trong FAISS index
        distances, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        # Lấy kết quả
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks_data):
                chunk = self.chunks_data[idx]
                results.append({
                    "chunk": chunk,
                    "distance": float(distances[0][i]),
                    "score": float(1 / (1 + distances[0][i]))
                })
        
        return {
            "success": True,
            "query_type": "vector",
            "results": results
        }
    
    def build_entity_data(self) -> Dict[str, Any]:
        """
        Xây dựng dữ liệu thực thể từ chunks_data.
        
        Returns:
            Dữ liệu thực thể
        """
        entity_data = {}
        
        # Xây dựng thực thể Course
        course_entities = {}
        
        # Tìm tất cả các môn học
        subject_codes = set()
        for chunk in self.chunks_data:
            if "metadata" in chunk and "subject_code" in chunk["metadata"]:
                subject_codes.add(chunk["metadata"]["subject_code"])
        
        # Xây dựng thực thể Course cho mỗi môn học
        for subject_code in subject_codes:
            course_id = f"course_{subject_code}"
            course_entity = {
                "entity_id": course_id,
                "entity_type": EntityType.COURSE,
                "subject_code": subject_code,
                "related_entities": {
                    "sessions": [],
                    "clos": [],
                    "assessments": [],
                    "materials": []
                }
            }
            
            # Tìm thông tin chi tiết về môn học
            for chunk in self.chunks_data:
                if "metadata" in chunk and chunk["metadata"].get("subject_code") == subject_code:
                    if chunk["type"] in ["general_info", "overview"]:
                        # Trích xuất thông tin từ chunk
                        content = chunk.get("content", "")
                        
                        # Trích xuất tên môn học
                        name_match = re.search(r"Môn học .+ - (.+?)[\.\(]", content)
                        if name_match:
                            course_entity["syllabus_name"] = name_match.group(1).strip()
                        
                        # Trích xuất tên tiếng Anh
                        english_match = re.search(r"Tên tiếng Anh: (.+?)\.", content)
                        if english_match:
                            course_entity["syllabus_english"] = english_match.group(1).strip()
                        
                        # Trích xuất số tín chỉ
                        credit_match = re.search(r"Số tín chỉ: (\d+)", content)
                        if credit_match:
                            course_entity["credits"] = int(credit_match.group(1))
                        
                        # Trích xuất mô tả
                        desc_match = re.search(r"Mô tả: (.+?)(?:\.|$)", content)
                        if desc_match:
                            course_entity["description"] = desc_match.group(1).strip()
                    
                    # Lấy thông tin từ metadata
                    if "metadata" in chunk:
                        metadata = chunk["metadata"]
                        if "total_sessions" in metadata:
                            course_entity["total_sessions"] = metadata["total_sessions"]
                        if "total_clos" in metadata:
                            course_entity["total_clos"] = metadata["total_clos"]
                        if "total_materials" in metadata:
                            course_entity["total_materials"] = metadata["total_materials"]
            
            # Tạo embedding cho môn học
            if "syllabus_name" in course_entity:
                course_text = f"{subject_code} {course_entity.get('syllabus_name', '')} {course_entity.get('syllabus_english', '')} {course_entity.get('description', '')}"
                course_entity["embedding"] = self.embedding_model.encode(course_text).tolist()
            
            entity_data[course_id] = course_entity
            course_entities[subject_code] = course_id
        
        # Xây dựng thực thể Session
        for chunk in self.chunks_data:
            if chunk["type"] == "session" and "metadata" in chunk:
                metadata = chunk["metadata"]
                subject_code = metadata.get("subject_code")
                session_number = metadata.get("session_number")
                
                if subject_code and session_number:
                    session_id = f"session_{subject_code}_{session_number}"
                    
                    # Trích xuất thông tin từ chunk
                    content = chunk.get("content", "")
                    
                    # Trích xuất chủ đề
                    topic_match = re.search(r"Buổi học số \d+: Chủ đề - (.+?)\.", content)
                    topic = topic_match.group(1).strip() if topic_match else ""
                    
                    session_entity = {
                        "entity_id": session_id,
                        "entity_type": EntityType.SESSION,
                        "subject_code": subject_code,
                        "session_number": session_number,
                        "topic": topic,
                        "content": content,
                        "related_entities": {
                            "course": f"course_{subject_code}"
                        }
                    }
                    
                    # Tạo embedding cho session
                    session_entity["embedding"] = self.embedding_model.encode(content).tolist()
                    
                    entity_data[session_id] = session_entity
                    
                    # Thêm session vào related_entities của course
                    course_id = f"course_{subject_code}"
                    if course_id in entity_data:
                        entity_data[course_id]["related_entities"]["sessions"].append(session_id)
        
        # Xây dựng thực thể CLO
        for chunk in self.chunks_data:
            if chunk["type"] == "clo" and "metadata" in chunk:
                metadata = chunk["metadata"]
                subject_code = metadata.get("subject_code")
                clo_id_val = metadata.get("clo_id")
                
                if subject_code and clo_id_val:
                    clo_id = f"clo_{subject_code}_{clo_id_val}"
                    
                    # Trích xuất thông tin từ chunk
                    content = chunk.get("content", "")
                    
                    clo_entity = {
                        "entity_id": clo_id,
                        "entity_type": EntityType.CLO,
                        "subject_code": subject_code,
                        "clo_name": clo_id_val,
                        "content": content,
                        "related_entities": {
                            "course": f"course_{subject_code}"
                        }
                    }
                    
                    # Tạo embedding cho CLO
                    clo_entity["embedding"] = self.embedding_model.encode(content).tolist()
                    
                    entity_data[clo_id] = clo_entity
                    
                    # Thêm CLO vào related_entities của course
                    course_id = f"course_{subject_code}"
                    if course_id in entity_data:
                        entity_data[course_id]["related_entities"]["clos"].append(clo_id)
        
        return entity_data
    
    def update_entity_data(self, entity_data: Dict[str, Any]):
        """
        Cập nhật dữ liệu thực thể.
        
        Args:
            entity_data: Dữ liệu thực thể mới
        """
        self.entity_data = entity_data
        self.query_processor = ContextualQueryProcessor(self.embedding_model, entity_data)
        self.entity_linker = EntityLinker(entity_data)
        self.query_executor = QueryExecutor(entity_data, self.faiss_index, self.chunks_data)


# Hàm tiện ích để tạo entity data từ chunks
def build_entity_data_from_chunks(chunks_data: List[Dict[str, Any]], embedding_model: SentenceTransformer) -> Dict[str, Any]:
    """
    Xây dựng dữ liệu thực thể từ chunks_data.
    
    Args:
        chunks_data: Dữ liệu chunks
        embedding_model: Mô hình embedding
        
    Returns:
        Dữ liệu thực thể
    """
    entity_data = {}
    
    # Xây dựng thực thể Course
    course_entities = {}
    
    # Tìm tất cả các môn học
    subject_codes = set()
    for chunk in chunks_data:
        if "metadata" in chunk and "subject_code" in chunk["metadata"]:
            subject_codes.add(chunk["metadata"]["subject_code"])
    
    # Xây dựng thực thể Course cho mỗi môn học
    for subject_code in subject_codes:
        course_id = f"course_{subject_code}"
        course_entity = {
            "entity_id": course_id,
            "entity_type": EntityType.COURSE,
            "subject_code": subject_code,
            "related_entities": {
                "sessions": [],
                "clos": [],
                "assessments": [],
                "materials": []
            }
        }
        
        # Tìm thông tin chi tiết về môn học
        for chunk in chunks_data:
            if "metadata" in chunk and chunk["metadata"].get("subject_code") == subject_code:
                if chunk["type"] in ["general_info", "overview"]:
                    # Trích xuất thông tin từ chunk
                    content = chunk.get("content", "")
                    
                    # Trích xuất tên môn học
                    name_match = re.search(r"Môn học .+ - (.+?)[\.\(]", content)
                    if name_match:
                        course_entity["syllabus_name"] = name_match.group(1).strip()
                    
                    # Trích xuất tên tiếng Anh
                    english_match = re.search(r"Tên tiếng Anh: (.+?)\.", content)
                    if english_match:
                        course_entity["syllabus_english"] = english_match.group(1).strip()
                    
                    # Trích xuất số tín chỉ
                    credit_match = re.search(r"Số tín chỉ: (\d+)", content)
                    if credit_match:
                        course_entity["credits"] = int(credit_match.group(1))
                    
                    # Trích xuất mô tả
                    desc_match = re.search(r"Mô tả: (.+?)(?:\.|$)", content)
                    if desc_match:
                        course_entity["description"] = desc_match.group(1).strip()
                
                # Lấy thông tin từ metadata
                if "metadata" in chunk:
                    metadata = chunk["metadata"]
                    if "total_sessions" in metadata:
                        course_entity["total_sessions"] = metadata["total_sessions"]
                    if "total_clos" in metadata:
                        course_entity["total_clos"] = metadata["total_clos"]
                    if "total_materials" in metadata:
                        course_entity["total_materials"] = metadata["total_materials"]
        
        # Tạo embedding cho môn học
        if "syllabus_name" in course_entity:
            course_text = f"{subject_code} {course_entity.get('syllabus_name', '')} {course_entity.get('syllabus_english', '')} {course_entity.get('description', '')}"
            course_entity["embedding"] = embedding_model.encode(course_text).tolist()
        
        entity_data[course_id] = course_entity
        course_entities[subject_code] = course_id
    
    # Xây dựng thực thể Session
    for chunk in chunks_data:
        if chunk["type"] == "session" and "metadata" in chunk:
            metadata = chunk["metadata"]
            subject_code = metadata.get("subject_code")
            session_number = metadata.get("session_number")
            
            if subject_code and session_number:
                session_id = f"session_{subject_code}_{session_number}"
                
                # Trích xuất thông tin từ chunk
                content = chunk.get("content", "")
                
                # Trích xuất chủ đề
                topic_match = re.search(r"Buổi học số \d+: Chủ đề - (.+?)\.", content)
                topic = topic_match.group(1).strip() if topic_match else ""
                
                session_entity = {
                    "entity_id": session_id,
                    "entity_type": EntityType.SESSION,
                    "subject_code": subject_code,
                    "session_number": session_number,
                    "topic": topic,
                    "content": content,
                    "related_entities": {
                        "course": f"course_{subject_code}"
                    }
                }
                
                # Tạo embedding cho session
                session_entity["embedding"] = embedding_model.encode(content).tolist()
                
                entity_data[session_id] = session_entity
                
                # Thêm session vào related_entities của course
                course_id = f"course_{subject_code}"
                if course_id in entity_data:
                    entity_data[course_id]["related_entities"]["sessions"].append(session_id)
    
    # Xây dựng thực thể CLO
    for chunk in chunks_data:
        if chunk["type"] == "clo" and "metadata" in chunk:
            metadata = chunk["metadata"]
            subject_code = metadata.get("subject_code")
            clo_id_val = metadata.get("clo_id")
            
            if subject_code and clo_id_val:
                clo_id = f"clo_{subject_code}_{clo_id_val}"
                
                # Trích xuất thông tin từ chunk
                content = chunk.get("content", "")
                
                clo_entity = {
                    "entity_id": clo_id,
                    "entity_type": EntityType.CLO,
                    "subject_code": subject_code,
                    "clo_name": clo_id_val,
                    "content": content,
                    "related_entities": {
                        "course": f"course_{subject_code}"
                    }
                }
                
                # Tạo embedding cho CLO
                clo_entity["embedding"] = embedding_model.encode(content).tolist()
                
                entity_data[clo_id] = clo_entity
                
                # Thêm CLO vào related_entities của course
                course_id = f"course_{subject_code}"
                if course_id in entity_data:
                    entity_data[course_id]["related_entities"]["clos"].append(clo_id)
    
    return entity_data


# Hàm tiện ích để lưu và tải entity data
def save_entity_data(entity_data: Dict[str, Any], file_path: str):
    """
    Lưu dữ liệu thực thể vào file.
    
    Args:
        entity_data: Dữ liệu thực thể
        file_path: Đường dẫn file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(entity_data, f, ensure_ascii=False, indent=2)

def load_entity_data(file_path: str) -> Dict[str, Any]:
    """
    Tải dữ liệu thực thể từ file.
    
    Args:
        file_path: Đường dẫn file
        
    Returns:
        Dữ liệu thực thể
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
# 