"""
Advanced RAG Engine - Hệ thống RAG tiên tiến cho FPTU
Tích hợp các kỹ thuật tiên tiến: Hierarchical Indexing, Multi-stage Retrieval, Query Routing, Document Summarization, Multi-hop Query
ENHANCED với GraphRAG: Vector Search + Knowledge Graph Traversal
"""

import json
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dataclasses import dataclass
import re
import os
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from enum import Enum
import time
import hashlib
import pickle

# Import GraphRAG components
try:
    from graph_database import GraphDatabase, GraphPath, GraphNode, GraphRelationship
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠ GraphDatabase module không khả dụng - chạy trong vector-only mode")

# Cấu hình logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

class APIKeyManager:
    """Quản lý và xoay vòng API keys để tránh quota exceeded"""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = [key for key in api_keys if key]  # Filter out empty keys
        self.current_index = 0
        self.failed_keys = set()  # Track keys that are temporarily failed
        self.last_error_time = {}  # Track when each key last failed
        self.retry_delay = 60  # Wait 60 seconds before retrying a failed key
        
        if not self.api_keys:
            raise ValueError("Cần ít nhất một API key hợp lệ")
        
        logger.info(f"✓ APIKeyManager khởi tạo với {len(self.api_keys)} API keys")
        
    def get_current_key(self) -> str:
        """Lấy API key hiện tại"""
        return self.api_keys[self.current_index]
        
    def get_current_model(self):
        """Lấy Gemini model với API key hiện tại"""
        try:
            current_key = self.get_current_key()
            genai.configure(api_key=current_key)
            return genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            logger.error(f"Lỗi tạo model với key {self.current_index + 1}: {e}")
            return None
    
    def rotate_key(self, error_message: str = None) -> bool:
        """
        Xoay sang API key tiếp theo
        Returns: True nếu còn key khả dụng, False nếu hết key
        """
        # Mark current key as failed
        current_key_index = self.current_index
        self.failed_keys.add(current_key_index)
        self.last_error_time[current_key_index] = time.time()
        
        logger.warning(f"API Key {current_key_index + 1} failed: {error_message}")
        
        # Try to find next available key
        attempts = 0
        while attempts < len(self.api_keys):
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            
            # Check if this key is available
            if self._is_key_available(self.current_index):
                logger.info(f"Chuyển sang API Key {self.current_index + 1}")
                return True
                
            attempts += 1
        
        # No available keys
        logger.error("Tất cả API keys đều không khả dụng")
        return False
    
    def _is_key_available(self, key_index: int) -> bool:
        """Kiểm tra xem API key có khả dụng không"""
        if key_index not in self.failed_keys:
            return True
            
        # Check if enough time has passed since last failure
        if key_index in self.last_error_time:
            time_since_failure = time.time() - self.last_error_time[key_index]
            if time_since_failure > self.retry_delay:
                # Remove from failed set to retry
                self.failed_keys.discard(key_index)
                logger.info(f"API Key {key_index + 1} sẵn sàng thử lại sau {time_since_failure:.1f}s")
                return True
        
        return False
    
    def reset_failed_keys(self):
        """Reset tất cả failed keys - dùng để force retry"""
        self.failed_keys.clear()
        self.last_error_time.clear()
        logger.info("Đã reset tất cả failed API keys")
    
    def get_status(self) -> Dict[str, Any]:
        """Lấy trạng thái của API key manager"""
        return {
            'total_keys': len(self.api_keys),
            'current_index': self.current_index,
            'current_key_suffix': self.get_current_key()[-10:] if self.api_keys else 'N/A',
            'failed_keys': list(self.failed_keys),
            'available_keys': [i for i in range(len(self.api_keys)) if self._is_key_available(i)]
        }
    
    def call_with_rotation(self, func, *args, **kwargs):
        """
        Gọi function với auto rotation khi gặp quota error
        """
        max_attempts = len(self.api_keys)
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Get current model and try the function
                model = self.get_current_model()
                if not model:
                    if not self.rotate_key("Failed to create model"):
                        break
                    attempt += 1
                    continue
                
                # Call function with current model
                if hasattr(func, '__self__'):
                    # Method call - update the model reference
                    func.__self__.gemini_model = model
                
                return func(*args, **kwargs)
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a quota error
                if '429' in error_str or 'quota' in error_str.lower() or 'exceeded' in error_str.lower():
                    logger.warning(f"Quota exceeded with API Key {self.current_index + 1}, attempting rotation...")
                    
                    if not self.rotate_key(error_str):
                        # No more keys available
                        logger.error("Tất cả API keys đã hết quota")
                        raise e
                    
                    attempt += 1
                    continue
                else:
                    # Non-quota error, don't rotate
                    raise e
        
        # If we get here, all attempts failed
        raise Exception("Tất cả API keys đều không khả dụng")

@dataclass
class SearchResult:
    """Kết quả tìm kiếm với metadata phong phú"""
    content: str
    score: float
    subject_code: str
    document_type: str
    level: str  # 'summary', 'chunk', 'detail'
    metadata: Dict[str, Any]

@dataclass
class QueryIntent:
    """Phân tích ý định truy vấn"""
    query_type: str  # 'factual', 'analytical', 'comparative', 'listing'
    subject_scope: str  # 'single', 'multiple', 'all'
    complexity: str  # 'simple', 'medium', 'complex'
    requires_summarization: bool
    target_subjects: List[str]

@dataclass
class FollowupQuery:
    """Truy vấn tiếp theo được phát hiện từ câu trả lời"""
    query: str
    confidence: float
    query_type: str  # 'prerequisite', 'related_subject', 'detail_expansion'
    source_info: str  # Thông tin nguồn gốc từ câu trả lời trước

@dataclass
class QueryChainResult:
    """Kết quả của chuỗi truy vấn đa cấp"""
    original_query: str
    original_answer: str
    followup_queries: List[FollowupQuery]
    followup_results: List[Dict[str, Any]]
    final_integrated_answer: str
    execution_path: List[str]

@dataclass
class ProcessedQuery:
    """Query đã được xử lý bởi LLM"""
    original_query: str
    processed_query: str
    intent_description: str
    suggested_keywords: List[str]
    confidence: float
    needs_data_search: bool

class QueryPreprocessor:
    """Xử lý query bằng LLM trước khi đưa vào hệ thống chính"""
    
    def __init__(self, gemini_model, rag_engine=None):
        self.gemini_model = gemini_model
        self.rag_engine = rag_engine  # Reference to parent engine for rotation
        
        # Domain-specific knowledge về FPTU
        self.fptu_domain_knowledge = {
            'combo_terms': [
                'combo', 'combo chuyên ngành', 'chuyên ngành hẹp', 'specialization', 
                'track', 'specialization track', 'major track'
            ],
            'semester_terms': [
                'kì', 'ky', 'kỳ', 'ki', 'semester', 'học kì', 'hoc ky', 'term'
            ],
            'quantity_terms': [
                'bao nhiêu', 'có mấy', 'số lượng', 'tổng cộng', 'how many', 'count'
            ],
            'definition_terms': [
                'là gì', 'la gi', 'what is', 'what are', 'định nghĩa', 'dinh nghia', 'nghĩa là'
            ]
        }
        
    def preprocess_query(self, query: str, conversation_context: str = "") -> ProcessedQuery:
        """Xử lý query bằng LLM để cải thiện hiểu ý định"""
        
        try:
            # Tạo prompt cho LLM
            prompt = self._build_preprocessing_prompt(query, conversation_context)
            
            # Gọi LLM với rotation nếu có reference tới engine
            if self.rag_engine and hasattr(self.rag_engine, '_call_gemini_with_rotation'):
                result_text = self.rag_engine._call_gemini_with_rotation(prompt)
            else:
                # Fallback to direct call
                response = self.gemini_model.generate_content(prompt)
                result_text = response.text.strip()
            
            # Parse kết quả
            parsed_result = self._parse_llm_response(result_text, query)
            
            logger.info(f"QUERY PREPROCESSING:")
            logger.info(f"  Original: '{query}'")
            logger.info(f"  Processed: '{parsed_result.processed_query}'")
            logger.info(f"  Intent: {parsed_result.intent_description}")
            logger.info(f"  Keywords: {parsed_result.suggested_keywords}")
            logger.info(f"  Needs data search: {parsed_result.needs_data_search}")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Lỗi preprocessing query: {e}")
            # Fallback - trả về query gốc
            return ProcessedQuery(
                original_query=query,
                processed_query=query,
                intent_description="unknown",
                suggested_keywords=[],
                confidence=0.5,
                needs_data_search=True
            )
    
    def _build_preprocessing_prompt(self, query: str, conversation_context: str) -> str:
        """Tạo prompt cho LLM preprocessing"""
        
        context_part = ""
        if conversation_context:
            context_part = f"\nLịch sử cuộc trò chuyện:\n{conversation_context}\n"
        
        return f"""Bạn là chuyên gia phân tích query cho hệ thống thông tin học tập FPT University. 
Nhiệm vụ: Phân tích và cải thiện query của người dùng để tìm kiếm hiệu quả hơn.

DOMAIN KNOWLEDGE - FPT UNIVERSITY:
- Ngành AI có 45 môn học, phân bố theo 8 kì
- Có 3 combo chuyên ngành hẹp: AI17_COM1 (Data Science), AI17_COM2.1 (Text Mining), AI17_COM3 (AI Healthcare)
- Mỗi môn có mã (vd: CSI106, AIG202c), tên tiếng Việt/Anh, số tín chỉ, kì học
- Thuật ngữ "combo chuyên ngành hẹp" = "specialization track"

{context_part}

QUERY CẦN PHÂN TÍCH: "{query}"

Hãy phân tích và trả về kết quả theo định dạng JSON:
{{
  "processed_query": "[Query được cải thiện để tìm kiếm tốt hơn]",
  "intent_description": "[Mô tả ý định: factual/listing/definition/counting/etc.]",
  "suggested_keywords": ["keyword1", "keyword2", "keyword3"],
  "confidence": [0.0-1.0],
  "needs_data_search": [true/false],
  "reasoning": "[Giải thích logic phân tích]"
}}

HƯỚNG DẪN CỤTHỂ:
1. Nếu hỏi "bao nhiêu combo/chuyên ngành" → processed_query: "liệt kê tất cả combo chuyên ngành hẹp AI"
2. Nếu hỏi "combo/chuyên ngành là gì" → processed_query: "thông tin về combo chuyên ngành hẹp ngành AI"
3. Nếu hỏi về kì học cụ thể → thêm keywords về semester
4. Nếu chỉ chào hỏi/cảm ơn → needs_data_search: false

Trả về JSON hợp lệ:"""

    def _parse_llm_response(self, response_text: str, original_query: str) -> ProcessedQuery:
        """Parse phản hồi từ LLM"""
        
        try:
            # Tìm JSON trong response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                return ProcessedQuery(
                    original_query=original_query,
                    processed_query=parsed.get('processed_query', original_query),
                    intent_description=parsed.get('intent_description', 'unknown'),
                    suggested_keywords=parsed.get('suggested_keywords', []),
                    confidence=float(parsed.get('confidence', 0.7)),
                    needs_data_search=bool(parsed.get('needs_data_search', True))
                )
            else:
                # Không parse được JSON, fallback
                logger.warning("Không parse được JSON từ LLM response")
                return self._fallback_processing(original_query)
                
        except Exception as e:
            logger.error(f"Lỗi parse LLM response: {e}")
            return self._fallback_processing(original_query)
    
    def _fallback_processing(self, query: str) -> ProcessedQuery:
        """Fallback processing khi LLM thất bại - Enhanced version"""
        
        query_lower = query.lower()
        
        # ENHANCED RULE-BASED PROCESSING
        
        # 1. SEMESTER RELATIONSHIP QUERIES
        if ('kì' in query_lower or 'ky' in query_lower or 'kỳ' in query_lower) and ('liên quan' in query_lower or 'lien quan' in query_lower):
            return ProcessedQuery(
                original_query=query,
                processed_query=f"môn học kì 4 kì 5 mối liên hệ tiên quyết phụ thuộc {query}",
                intent_description="semester_relationship_analysis",
                suggested_keywords=['kì 4', 'kì 5', 'môn học', 'liên quan', 'tiên quyết', 'prerequisite'],
                confidence=0.85,
                needs_data_search=True
            )
        
        # 2. COMBO COUNTING QUERIES
        if any(term in query_lower for term in ['bao nhiêu', 'có mấy', 'số lượng']):
            if any(term in query_lower for term in ['combo', 'chuyên ngành']):
                return ProcessedQuery(
                    original_query=query,
                    processed_query="liệt kê tất cả combo chuyên ngành hẹp AI specialization track",
                    intent_description="counting_combo",
                    suggested_keywords=['combo', 'chuyên ngành hẹp', 'AI', 'specialization', 'track'],
                    confidence=0.9,
                    needs_data_search=True
                )
        
        # 3. COMBO DEFINITION QUERIES  
        if any(term in query_lower for term in ['là gì', 'what is', 'định nghĩa']):
            if any(term in query_lower for term in ['combo', 'chuyên ngành']):
                return ProcessedQuery(
                    original_query=query,
                    processed_query="thông tin định nghĩa combo chuyên ngành hẹp ngành AI specialization track",
                    intent_description="definition_combo",
                    suggested_keywords=['combo', 'chuyên ngành hẹp', 'thông tin', 'specialization'],
                    confidence=0.9,
                    needs_data_search=True
                )
        
        # 4. SEMESTER LISTING QUERIES
        semester_patterns = ['kì 1', 'kì 2', 'kì 3', 'kì 4', 'kì 5', 'kì 6', 'kì 7', 'kì 8',
                           'ky 1', 'ky 2', 'ky 3', 'ky 4', 'ky 5', 'ky 6', 'ky 7', 'ky 8']
        if any(pattern in query_lower for pattern in semester_patterns):
            # Extract semester numbers
            semesters = []
            for i in range(1, 9):
                if f'kì {i}' in query_lower or f'ky {i}' in query_lower or f'kỳ {i}' in query_lower:
                    semesters.append(str(i))
            
            if semesters:
                semester_text = ' '.join([f'kì {s}' for s in semesters])
                return ProcessedQuery(
                    original_query=query,
                    processed_query=f"môn học {semester_text} semester {' '.join(semesters)} curriculum subjects",
                    intent_description="semester_listing",
                    suggested_keywords=['môn học'] + [f'kì {s}' for s in semesters] + ['semester', 'curriculum'],
                    confidence=0.85,
                    needs_data_search=True
                )
        
        # 5. GREETINGS AND THANKS - SHOULD BE DIRECT CHAT
        greeting_patterns = [
            'xin chào', 'hello', 'hi', 'chào bạn', 'chao ban',
            'cảm ơn', 'cam on', 'thank you', 'thanks', 'cám ơn'
        ]
        if any(pattern in query_lower for pattern in greeting_patterns):
            return ProcessedQuery(
                original_query=query,
                processed_query=query,
                intent_description="greeting_or_thanks",
                suggested_keywords=[],
                confidence=0.95,
                needs_data_search=False  # IMPORTANT: Direct chat
            )
        
        # 6. SUBJECT CODE QUERIES
        subject_codes = re.findall(r'[A-Za-z]{2,4}\d{3}[a-zA-Z]*', query)
        if subject_codes:
            return ProcessedQuery(
                original_query=query,
                processed_query=f"{query} subject code course information",
                intent_description="subject_specific",
                suggested_keywords=subject_codes + ['môn học', 'course', 'subject'],
                confidence=0.8,
                needs_data_search=True
            )
        
        # 7. GENERAL ACADEMIC QUERIES
        academic_terms = ['môn học', 'mon hoc', 'subject', 'course', 'curriculum', 'chương trình', 'chuong trinh']
        if any(term in query_lower for term in academic_terms):
            return ProcessedQuery(
                original_query=query,
                processed_query=f"{query} academic curriculum course information",
                intent_description="general_academic",
                suggested_keywords=['môn học', 'curriculum', 'academic'],
                confidence=0.7,
                needs_data_search=True
            )
        
        # 8. DEFAULT FALLBACK
        return ProcessedQuery(
            original_query=query,
            processed_query=query,
            intent_description="unknown",
            suggested_keywords=[],
            confidence=0.5,
            needs_data_search=True  # Default to data search unless clearly conversational
        )

class QueryRouter:
    """Router thông minh để định tuyến query"""
    
    def __init__(self):
        # Quick response patterns for basic queries
        self.quick_response_patterns = {
            'greeting': {
                'patterns': [
                    r'xin chào', r'hello', r'hi', r'chào bạn', r'chào',
                    r'good morning', r'good afternoon', r'good evening'
                ],
                'response': "Xin chào! Tôi là AI Assistant của FPTU. Tôi có thể giúp bạn tìm kiếm thông tin về các môn học, syllabus và chương trình đào tạo. Hãy đặt câu hỏi cho tôi!"
            },
            'identity': {
                'patterns': [
                    r'bạn là ai', r'who are you', r'bạn là gì', r'what are you',
                    r'giới thiệu về bạn', r'tell me about yourself'
                ],
                'response': "Tôi là AI Assistant của FPTU - hệ thống hỗ trợ tìm kiếm thông tin về chương trình đào tạo và môn học tại FPT University. Tôi có thể giúp bạn:\n\n• Tìm kiếm thông tin môn học theo mã (VD: CSI106, SEG301)\n• Liệt kê các môn học theo ngành\n• Tìm hiểu về syllabus, CLO, tài liệu học tập\n• Tra cứu thông tin về môn tiên quyết\n\nHãy thử hỏi tôi về bất kỳ môn học nào bạn quan tâm!"
            },
            'help': {
                'patterns': [
                    r'help', r'giúp đỡ', r'hướng dẫn', r'làm gì', r'what can you do',
                    r'bạn có thể làm gì', r'tôi có thể hỏi gì'
                ],
                'response': "Tôi có thể giúp bạn:\n\n**Tìm kiếm môn học:**\n• Theo mã môn: 'CSI106 là môn gì?'\n• Theo tên môn: 'Machine Learning là môn gì?'\n• Theo ngành: 'Liệt kê các môn học ngành AI'\n\n**Thông tin chi tiết:**\n• Syllabus và CLO\n• Tài liệu học tập\n• Phương thức đánh giá\n• Môn tiên quyết\n\n**Câu hỏi mẫu:**\n• 'Các môn có 3 tín chỉ'\n• 'Danh sách môn học kỳ 1'\n• 'SEG301 và các môn tiên quyết'"
            },
            'thanks': {
                'patterns': [
                    r'cảm ơn', r'thank you', r'thanks', r'cám ơn', r'cảm ơn bạn'
                ],
                'response': "Rất vui được giúp đỡ bạn! Nếu bạn có thêm câu hỏi nào về chương trình đào tạo hoặc môn học tại FPTU, đừng ngại hỏi tôi nhé!"
            }
        }
        
        self.factual_patterns = [
            r'(.*?) là gì',
            r'định nghĩa (.*?)',
            r'(.*?) có (bao nhiêu|mấy) tín chỉ',
            r'ai là giảng viên (.*?)',
            r'(.*?) thuộc kỳ (.*?)'
        ]
        
        self.listing_patterns = [
            r'liệt kê (.*?)',
            r'tất cả (.*?) môn',
            r'các môn (.*?)',
            r'danh sách (.*?)',
            r'cho tôi biết (.*?) môn'
        ]
        
        self.comparative_patterns = [
            r'so sánh (.*?) và (.*?)',
            r'khác nhau giữa (.*?) và (.*?)',
            r'(.*?) hay (.*?) tốt hơn'
        ]
        
        self.analytical_patterns = [
            r'tại sao (.*?)',
            r'phân tích (.*?)',
            r'đánh giá (.*?)',
            r'ưu nhược điểm (.*?)'
        ]

    def check_quick_response(self, query: str) -> Optional[str]:
        """Check if query matches quick response patterns"""
        query_lower = query.lower().strip()
        
        # Skip if query looks like subject search (contains technical terms)
        # But be more careful with short terms that might be in normal conversation
        technical_indicators = [
            'machine learning', 'artificial intelligence', 'data science',
            'programming', 'algorithm', 'database', 'network', 'security',
            'software', 'hardware', 'computer science', 'engineering',
            'mathematics', 'statistics', 'physics', 'chemistry'
        ]
        
        # Check for longer technical terms first
        for indicator in technical_indicators:
            if indicator in query_lower:
                return None
        
        # For short terms, be more restrictive - only block if they appear as isolated words
        # BUT exclude common conversational contexts
        short_technical_terms = ['ml', 'cs', 'se', 'it']  # Remove 'ai' as it's too common in Vietnamese
        words = query_lower.split()
        
        # Special handling for 'ai' - only block if it's clearly technical context
        if 'ai' in words:
            # Don't block if it's in conversational context
            conversational_ai_contexts = [
                'bạn là ai', 'ai là', 'ai đó', 'ai đang', 'ai sẽ', 
                'cho ai', 'với ai', 'của ai', 'ai có', 'ai cần'
            ]
            is_conversational = any(ctx in query_lower for ctx in conversational_ai_contexts)
            
            if not is_conversational:
                # Check if it's technical AI context
                if any(tech in query_lower for tech in ['artificial', 'intelligence', 'machine', 'learning', 'deep']):
                    return None
        
        for term in short_technical_terms:
            if term in words:  # Only if it's a separate word, not part of another word
                return None
        
        # Skip if query contains subject codes (full or partial)
        if re.search(r'[A-Z]{2,4}\d{3}[a-z]*', query):
            return None
            
        # Also skip if query contains likely subject code patterns
        # Look for common subject code prefixes that appear in academic context
        academic_context_indicators = [
            'môn', 'mon', 'học', 'hoc', 'tín chỉ', 'tin chi', 'tiên quyết', 'tien quyet',
            'syllabus', 'course', 'subject', 'credit', 'prerequisite', 'thông tin', 'thong tin',
            'chi tiết', 'chi tiet', 'nội dung', 'noi dung',
            # COMBO/SPECIALIZATION INDICATORS
            'combo', 'chuyên ngành', 'chuyen nganh', 'specialization', 'track',
            'chuyên ngành hẹp', 'chuyen nganh hep', 'combo chuyên ngành', 'combo chuyen nganh',
            # COUNTING/QUANTITY INDICATORS
            'bao nhiêu', 'bao nhieu', 'có mấy', 'co may', 'số lượng', 'so luong',
            'tổng cộng', 'tong cong', 'how many', 'count', 'list', 'liệt kê', 'liet ke'
        ]
        
        # Only skip if query contains both academic context AND potential subject codes
        has_academic_context = any(indicator in query_lower for indicator in academic_context_indicators)
        
        if has_academic_context:
            words = query.upper().split()
            excluded_words = {
                'MÔN', 'MON', 'HỌC', 'HOC', 'CÓ', 'CO', 'KHÔNG', 'KHONG', 
                'GÌ', 'GI', 'LÀ', 'LA', 'NÀO', 'NAO', 'TIÊN', 'TIEN', 
                'QUYẾT', 'QUYET', 'ĐIỀU', 'DIEU', 'KIỆN', 'KIEN',
                'COURSE', 'SUBJECT', 'WHAT', 'IS', 'ARE', 'THE', 'HAVE', 'HAS',
                'VÀ', 'VA', 'CỦA', 'CUA', 'THÔNG', 'THONG', 'TIN', 'CHI', 'TIẾT', 'TIET',
                'CÁC', 'CAC', 'NỘI', 'NOI', 'DUNG', 'DỤNG', 'BẰNG', 'BANG'
            }
            
            for word in words:
                if (len(word) >= 3 and len(word) <= 6 and word.isalpha() 
                    and word not in excluded_words):
                    # This looks like a subject code in academic context
                    return None
        
        for category, data in self.quick_response_patterns.items():
            for pattern in data['patterns']:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return data['response']
        
        return None

    def analyze_query(self, query: str) -> QueryIntent:
        """Phân tích ý định và phạm vi của query"""
        query_lower = query.lower()
        
        # Extract subject codes and combo codes for later use (CASE INSENSITIVE)
        subject_codes = re.findall(r'[A-Za-z]{2,4}\d{3}[a-zA-Z]*', query, re.IGNORECASE)
        # Normalize to uppercase
        subject_codes = [code.upper() for code in subject_codes]
        
        # ENHANCED: Extract combo codes (like AI17_COM1, AI17_COM2.1)
        combo_codes = re.findall(r'[A-Za-z]{2,4}\d{2}_[A-Za-z]{3}\d+(?:\.\d+)?', query, re.IGNORECASE)
        # Normalize to uppercase
        combo_codes = [code.upper() for code in combo_codes]
        
        # Combine subject and combo codes
        all_codes = subject_codes + combo_codes
        
        # ENHANCED: Extract partial subject codes (like "DPL" -> "DPL302m")
        if not all_codes:
            # Look for partial subject codes (3+ characters)
            words = query.upper().split()
            for word in words:
                # Only consider words that could be subject codes
                # - 3+ characters, alphabetic
                # - Not common Vietnamese/English words
                excluded_words = {
                    'MÔN', 'MON', 'HỌC', 'HOC', 'CÓ', 'CO', 'KHÔNG', 'KHONG', 
                    'GÌ', 'GI', 'LÀ', 'LA', 'NÀO', 'NAO', 'TIÊN', 'TIEN', 
                    'QUYẾT', 'QUYET', 'ĐIỀU', 'DIEU', 'KIỆN', 'KIEN',
                    'COURSE', 'SUBJECT', 'WHAT', 'IS', 'ARE', 'THE', 'HAVE', 'HAS',
                    'VÀ', 'VA', 'CỦA', 'CUA', 'THÔNG', 'THONG', 'TIN', 'CHI', 'TIẾT', 'TIET',
                    'CÁC', 'CAC', 'NỘI', 'NOI', 'DUNG', 'DỤNG', 'BẰNG', 'BANG'
                }
                
                # Only add words that look like actual subject codes (3-4 chars, often with consonants)
                if (len(word) >= 3 and len(word) <= 4 and word.isalpha() 
                    and word not in excluded_words
                    and not word.endswith('NG')):  # Avoid words like "THONG", "DUNG"
                    # This could be a partial subject code
                    all_codes.append(word)
        
        # IMPROVEMENT: Also extract partial codes from full subject codes to enable broader search
        # This helps when specific subject code search fails but partial search succeeds
        partial_codes_from_full = []
        for full_code in subject_codes:
            # Extract the prefix (e.g., "DPL" from "DPL302m")
            prefix_match = re.match(r'([A-Z]{2,4})', full_code)
            if prefix_match:
                prefix = prefix_match.group(1)
                if prefix not in all_codes and len(prefix) >= 3:
                    partial_codes_from_full.append(prefix)
        
        # Add partial codes as fallback option
        all_codes.extend(partial_codes_from_full)
        
        # PRIORITY 1: SEMESTER/TERM QUERIES - Always treat as LISTING
        if any(term in query_lower for term in ['kỳ', 'ky', 'kì', 'ki', 'semester', 'học kỳ', 'hoc ky']):
            # Check if asking for subjects in a semester
            if any(term in query_lower for term in ['môn', 'mon', 'subject', 'có gì', 'co gi', 'gồm có', 'gom co']):
                return QueryIntent(
                    query_type='listing',
                    subject_scope='multiple',
                    complexity='medium',
                    requires_summarization=True,
                    target_subjects=all_codes
                )
        
        # PRIORITY 2: STUDENT QUERIES (Higher priority than general listing)
        student_indicators = [
            'sinh viên', 'sinh vien', 'student', 'học sinh', 'hoc sinh', 
            'danh sách sinh viên', 'mã sinh viên', 'ma sinh vien',
            'danh sách sinh vien', 'ma sinh vien',
            'hoc sinh', 'sv ', ' sv', 'students'
        ]
        
        # Also check for combined patterns like "sinh viên ngành", "sinh viên AI"
        if (any(term in query_lower for term in student_indicators) or 
            ('sinh' in query_lower and 'viên' in query_lower) or
            ('sinh' in query_lower and 'vien' in query_lower)):
            # Determine if listing all students or specific student
            if any(pattern in query_lower for pattern in ['danh sách', 'danh sach', 'list', 'tất cả', 'tat ca', 'các sinh viên', 'cac sinh vien']):
                return QueryIntent(
                    query_type='listing',
                    subject_scope='multiple',
                    complexity='medium',
                    requires_summarization=True,
                    target_subjects=[]  # No specific subjects for student listing
                )
            else:
                # Specific student query (usually contains roll number)
                # Extract roll numbers if present
                roll_numbers = re.findall(r'[A-Z]{2}\d{6}', query.upper())
                return QueryIntent(
                    query_type='factual',
                    subject_scope='single',
                    complexity='simple',
                    requires_summarization=False,
                    target_subjects=roll_numbers  # Use roll numbers instead of partial words
                )

        # PRIORITY 3: EXPLICIT LISTING QUERIES (for non-student content)
        if any(pattern in query_lower for pattern in ['liệt kê', 'liet ke', 'danh sách', 'danh sach', 'list', 'tất cả', 'tat ca', 'các môn', 'cac mon']):
            complexity = 'complex' if any(term in query_lower for term in ['phân tích', 'so sánh', 'compare']) else 'medium'
            return QueryIntent(
                query_type='listing',
                subject_scope='multiple' if not all_codes else 'single',
                complexity=complexity,
                requires_summarization=True,
                target_subjects=all_codes
            )
        
        # PRIORITY 4: COMPARATIVE QUERIES
        if any(pattern in query_lower for pattern in ['so sánh', 'so sanh', 'compare', 'khác nhau', 'khac nhau', 'giống', 'giong']):
            return QueryIntent(
                query_type='comparative',
                subject_scope='multiple',
                complexity='complex',
                requires_summarization=True,
                target_subjects=all_codes
            )
        
        # PRIORITY 5: ANALYTICAL QUERIES
        if any(pattern in query_lower for pattern in ['phân tích', 'phan tich', 'analyze', 'đánh giá', 'danh gia', 'evaluate', 'lộ trình', 'lo trinh', 'roadmap']):
            return QueryIntent(
                query_type='analytical',
                subject_scope='multiple',
                complexity='complex',
                requires_summarization=True,
                target_subjects=all_codes
            )
        
        # DEFAULT: FACTUAL QUERIES
        if all_codes:
            scope = 'single'
            complexity = 'simple'
        elif any(term in query_lower for term in ['ngành', 'nganh', 'major', 'chương trình', 'chuong trinh']):
            scope = 'multiple'
            complexity = 'medium'
        else:
            scope = 'single'
            complexity = 'simple'
        
        return QueryIntent(
            query_type='factual',
            subject_scope=scope,
            complexity=complexity,
            requires_summarization=False,
            target_subjects=all_codes
        )

class HierarchicalIndex:
    """Hệ thống indexing phân cấp"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.summary_index: Optional[faiss.Index] = None
        self.chunk_index: Optional[faiss.Index] = None
        self.detail_index: Optional[faiss.Index] = None
        
        self.summary_data: List[Dict] = []
        self.chunk_data: List[Dict] = []
        self.detail_data: List[Dict] = []
        
        # Metadata mapping
        self.subject_summary: Dict[str, Dict] = {}
        self.document_hierarchy: Dict[str, List[str]] = {}

    def build_hierarchy(self, data: List[Dict], gemini_model):
        """Xây dựng cấu trúc phân cấp"""
        logger.info("Bắt đầu xây dựng hierarchical index...")
        
        # Level 1: Subject Summaries
        self._build_subject_summaries(data, gemini_model)
        
        # Level 2: Document Chunks
        self._build_document_chunks(data)
        
        # Level 3: Detail Chunks
        self._build_detail_chunks(data)
        
        # Build FAISS indices
        self._build_faiss_indices()
        
        logger.info("Hoàn thành xây dựng hierarchical index")

    def _build_subject_summaries(self, data: List[Dict], gemini_model):
        """Xây dựng tóm tắt cấp môn học"""
        subjects = defaultdict(list)
        
        # Nhóm theo môn học
        for item in data:
            subject_code = item.get('metadata', {}).get('subject_code', 'UNKNOWN')
            subjects[subject_code].append(item)
        
        for subject_code, items in subjects.items():
            # Tạo tóm tắt toàn diện cho môn học
            combined_content = self._combine_subject_content(items)
            summary = self._generate_summary(combined_content, gemini_model, 'subject')
            
            subject_data = {
                'content': summary,
                'subject_code': subject_code,
                'level': 'summary',
                'item_count': len(items),
                'metadata': self._extract_subject_metadata(items)
            }
            
            self.summary_data.append(subject_data)
            self.subject_summary[subject_code] = subject_data

    def _build_document_chunks(self, data: List[Dict]):
        """Xây dựng chunks cấp tài liệu"""
        documents = defaultdict(list)
        
        # Nhóm theo loại tài liệu
        for item in data:
            doc_type = item.get('type', 'general')
            subject_code = item.get('metadata', {}).get('subject_code', 'UNKNOWN')
            key = f"{subject_code}_{doc_type}"
            documents[key].append(item)
        
        for doc_key, items in documents.items():
            if len(items) > 1:  # Chỉ tạo chunk nếu có nhiều items
                combined_content = self._combine_document_content(items)
                
                chunk_data = {
                    'content': combined_content[:2000],  # Limit length
                    'document_key': doc_key,
                    'level': 'chunk',
                    'item_count': len(items),
                    'metadata': items[0].get('metadata', {})
                }
                
                self.chunk_data.append(chunk_data)

    def _build_detail_chunks(self, data: List[Dict]):
        """Xây dựng chunks chi tiết"""
        for item in data:
            detail_data = {
                'content': item.get('content', ''),
                'level': 'detail',
                'original_data': item,
                'metadata': item.get('metadata', {})
            }
            self.detail_data.append(detail_data)

    def _combine_subject_content(self, items: List[Dict]) -> str:
        """Kết hợp nội dung của một môn học"""
        sections = []
        
        # Thông tin chung
        general_info = [item for item in items if item.get('type') == 'general_info']
        if general_info:
            sections.append(f"Thông tin chung: {general_info[0].get('content', '')}")
        
        # CLOs
        clos = [item for item in items if item.get('type') == 'clo']
        if clos:
            clo_content = ' '.join([item.get('content', '') for item in clos])
            sections.append(f"Chuẩn đầu ra: {clo_content}")
        
        # Sessions
        sessions = [item for item in items if item.get('type') == 'session']
        if sessions:
            session_content = ' '.join([item.get('content', '') for item in sessions[:5]])  # Limit
            sections.append(f"Nội dung buổi học: {session_content}")
        
        return ' '.join(sections)

    def _combine_document_content(self, items: List[Dict]) -> str:
        """Kết hợp nội dung của một loại tài liệu"""
        return ' '.join([item.get('content', '') for item in items])

    def _generate_summary(self, content: str, gemini_model, level: str) -> str:
        """Tạo tóm tắt bằng Gemini"""
        if not content.strip():
            return "Không có thông tin"
        
        if level == 'subject':
            prompt = f"""
            Tạo một tóm tắt ngắn gọn và toàn diện về môn học dựa trên thông tin sau:
            
            {content[:4000]}
            
            Tóm tắt cần bao gồm:
            - Tên môn học và mã môn
            - Số tín chỉ
            - Mô tả ngắn gọn
            - Các chuẩn đầu ra chính
            - Nội dung học chính
            
            Giới hạn trong 200 từ.
            """
        else:
            prompt = f"Tóm tắt ngắn gọn nội dung sau trong 100 từ:\n\n{content[:2000]}"
        
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Lỗi tạo tóm tắt: {e}")
            return content[:500]  # Fallback

    def _extract_subject_metadata(self, items: List[Dict]) -> Dict:
        """Trích xuất metadata cho môn học"""
        metadata = {}
        
        for item in items:
            item_meta = item.get('metadata', {})
            if not metadata.get('syllabus_name') and item_meta.get('syllabus_name'):
                metadata['syllabus_name'] = item_meta['syllabus_name']
            if not metadata.get('credits') and item_meta.get('credits'):
                metadata['credits'] = item_meta['credits']
            if not metadata.get('total_sessions') and item_meta.get('total_sessions'):
                metadata['total_sessions'] = item_meta['total_sessions']
        
        return metadata

    def _build_faiss_indices(self):
        """Xây dựng FAISS indices cho các cấp"""
        
        # Summary index
        if self.summary_data:
            summary_embeddings = []
            for item in self.summary_data:
                embedding = self.embedding_model.encode(item['content'])
                summary_embeddings.append(embedding)
            
            summary_embeddings = np.array(summary_embeddings).astype('float32')
            self.summary_index = faiss.IndexFlatIP(summary_embeddings.shape[1])
            self.summary_index.add(summary_embeddings)
        
        # Chunk index
        if self.chunk_data:
            chunk_embeddings = []
            for item in self.chunk_data:
                embedding = self.embedding_model.encode(item['content'])
                chunk_embeddings.append(embedding)
            
            chunk_embeddings = np.array(chunk_embeddings).astype('float32')
            self.chunk_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
            self.chunk_index.add(chunk_embeddings)
        
        # Detail index
        if self.detail_data:
            detail_embeddings = []
            for item in self.detail_data:
                embedding = self.embedding_model.encode(item['content'])
                detail_embeddings.append(embedding)
            
            detail_embeddings = np.array(detail_embeddings).astype('float32')
            self.detail_index = faiss.IndexFlatIP(detail_embeddings.shape[1])
            self.detail_index.add(detail_embeddings)

    def search_hierarchical(self, query: str, intent: QueryIntent, top_k: int = 10) -> List[SearchResult]:
        """Tìm kiếm phân cấp thông minh"""
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        results = []
        
        if intent.query_type == 'listing' or intent.subject_scope == 'all':
            # Tìm kiếm ở level summary trước
            if self.summary_index:
                distances, indices = self.summary_index.search(query_embedding, top_k)
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(self.summary_data):
                        item = self.summary_data[idx]
                        results.append(SearchResult(
                            content=item['content'],
                            score=float(distance),
                            subject_code=item['subject_code'],
                            document_type='summary',
                            level='summary',
                            metadata=item['metadata']
                        ))
        
        elif intent.query_type == 'factual' and intent.complexity == 'simple':
            # Tìm kiếm chi tiết cho câu hỏi đơn giản
            if self.detail_index:
                distances, indices = self.detail_index.search(query_embedding, top_k)
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(self.detail_data):
                        item = self.detail_data[idx]
                        results.append(SearchResult(
                            content=item['content'],
                            score=float(distance),
                            subject_code=item['metadata'].get('subject_code', 'UNKNOWN'),
                            document_type=item['metadata'].get('type', 'unknown'),
                            level='detail',
                            metadata=item['metadata']
                        ))
        
        else:
            # Tìm kiếm multi-level cho các trường hợp phức tạp
            results.extend(self._multi_level_search(query_embedding, top_k // 2))
        
        return results[:top_k]

    def _multi_level_search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """Tìm kiếm đa cấp"""
        results = []
        
        # Tìm ở level chunk
        if self.chunk_index and len(self.chunk_data) > 0:
            distances, indices = self.chunk_index.search(query_embedding, top_k)
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunk_data):
                    item = self.chunk_data[idx]
                    results.append(SearchResult(
                        content=item['content'],
                        score=float(distance),
                        subject_code=item['metadata'].get('subject_code', 'UNKNOWN'),
                        document_type=item['document_key'],
                        level='chunk',
                        metadata=item['metadata']
                    ))
        
        # Tìm ở level detail
        if self.detail_index and len(self.detail_data) > 0:
            distances, indices = self.detail_index.search(query_embedding, top_k)
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.detail_data):
                    item = self.detail_data[idx]
                    results.append(SearchResult(
                        content=item['content'],
                        score=float(distance),
                        subject_code=item['metadata'].get('subject_code', 'UNKNOWN'),
                        document_type=item['metadata'].get('type', 'unknown'),
                        level='detail',
                        metadata=item['metadata']
                    ))
        
        return results

class AdvancedRAGEngine:
    """
    Engine RAG tiên tiến với đầy đủ tính năng  
    ENHANCED với GraphRAG: Hybrid Vector + Graph Database Architecture
    """
    
    def __init__(self, api_keys: Union[str, List[str]], enable_graph: bool = True):
        """Khởi tạo RAG Engine với API key rotation và GraphRAG capabilities"""
        
        # Xử lý API keys input
        if isinstance(api_keys, str):
            api_keys = [api_keys]  # Convert single key to list
        
        # Initialize API Key Manager
        self.api_key_manager = APIKeyManager(api_keys)
        
        # Configure Gemini với key đầu tiên
        self.model = self.api_key_manager.get_current_model()
        
        # Initialize core components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.data = None
        self.index = None
        
        # Enhanced components
        self.hierarchical_index = None
        self.subject_mapping = {}
        self.query_router = QueryRouter()
        self.multihop_engine = None
        
        # NEW: Query Preprocessor với API key rotation support
        self.query_preprocessor = QueryPreprocessor(self.model, self)
        
        # GraphRAG components
        self.graph_db = None
        self.graph_enabled = enable_graph and GRAPH_AVAILABLE
        
        if self.graph_enabled:
            logger.info("🔄 Khởi tạo GraphRAG components...")
            try:
                self.graph_db = GraphDatabase()
                graph_connected = self.graph_db.connect()
                if graph_connected:
                    logger.info("✅ GraphRAG mode: Vector + Graph Hybrid")
                else:
                    logger.warning("⚠ Graph DB không kết nối được - fallback to Vector-only")
                    self.graph_enabled = False
            except Exception as e:
                logger.warning(f"⚠ Lỗi khởi tạo GraphDatabase: {e} - fallback to Vector-only")
                self.graph_enabled = False
        else:
            logger.info("📋 Vector-only mode enabled")
        
        # Conversation memory for chatbot functionality
        self.conversation_memory = {}  # session_id -> [{'user': query, 'bot': response}]
        
        logger.info(f"✓ AdvancedRAGEngine được khởi tạo với {len(self.api_key_manager.api_keys)} API keys")
        logger.info(f"✓ GraphRAG enabled: {self.graph_enabled}")

    def _call_gemini_with_rotation(self, prompt: str, max_retries: int = None) -> str:
        """
        Gọi Gemini với auto rotation khi gặp quota error
        """
        if max_retries is None:
            max_retries = len(self.api_key_manager.api_keys)
        
        for attempt in range(max_retries):
            try:
                # Lấy model hiện tại
                current_model = self.api_key_manager.get_current_model()
                if not current_model:
                    raise Exception("Không thể tạo Gemini model")
                
                # Update model reference
                self.model = current_model
                self.query_preprocessor.gemini_model = current_model
                
                # Gọi API
                response = current_model.generate_content(prompt)
                return response.text.strip()
                
            except Exception as e:
                error_str = str(e)
                
                # Kiểm tra quota error
                if ('429' in error_str or 
                    'quota' in error_str.lower() or 
                    'exceeded' in error_str.lower() or
                    'resource exhausted' in error_str.lower()):
                    
                    logger.warning(f"API Key {self.api_key_manager.current_index + 1} quota exceeded, rotating...")
                    
                    # Thử rotate key
                    if not self.api_key_manager.rotate_key(error_str):
                        logger.error("Tất cả API keys đã hết quota")
                        break
                    
                    continue  # Thử với key mới
                else:
                    # Lỗi khác (không phải quota) -> throw ngay
                    raise e
        
        # Nếu tới đây = hết quota tất cả keys
        raise Exception("Tất cả API keys đã hết quota, vui lòng thử lại sau")

    def initialize(self, data_path: str):
        """Khởi tạo engine với dữ liệu từ file JSON và populate graph database"""
        logger.info(f"Đang khởi tạo Advanced RAG Engine với dữ liệu từ {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.data = self._process_data(raw_data)
        self._create_embeddings()
        self._build_index()
        
        # Initialize hierarchical index
        self.hierarchical_index = HierarchicalIndex(self.embedding_model)
        
        # Initialize query chain for multi-hop queries
        self.query_chain = QueryChain(self)
        
        # GraphRAG: Populate graph database
        if self.graph_enabled and self.graph_db:
            logger.info("🔄 Populating Graph Database...")
            try:
                # Extract curriculum data for graph
                curriculum_data = []
                if isinstance(raw_data, dict) and 'syllabuses' in raw_data:
                    curriculum_data = raw_data['syllabuses']
                elif isinstance(raw_data, list):
                    curriculum_data = raw_data
                else:
                    logger.warning("Unknown data format for graph extraction")
                
                if curriculum_data:
                    nodes, relationships = self.graph_db.extract_entities_from_curriculum_data(curriculum_data)
                    
                    # Create schema and populate (simplified approach)
                    if nodes and relationships:
                        logger.info(f"📊 Graph entities: {len(nodes)} nodes, {len(relationships)} relationships")
                        # Note: In production, you'd want to populate the actual Neo4j database here
                        # For now, we store the extracted entities for later use
                        self.graph_entities = {'nodes': nodes, 'relationships': relationships}
                        logger.info("✅ Graph entities extracted successfully")
                    else:
                        logger.warning("⚠ No graph entities extracted")
                else:
                    logger.warning("⚠ No curriculum data found for graph extraction")
                    
            except Exception as e:
                logger.error(f"❌ Lỗi populate graph database: {e}")
                self.graph_enabled = False
        
        self.is_initialized = True
        logger.info("✅ Khởi tạo hoàn tất")
    
    def add_to_conversation(self, session_id: str, user_message: str, bot_response: str):
        """Thêm tin nhắn vào lịch sử conversation"""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        
        self.conversation_memory[session_id].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': time.time()
        })
        
        # Giới hạn lịch sử conversation (chỉ giữ 10 tin nhắn gần nhất)
        if len(self.conversation_memory[session_id]) > 10:
            self.conversation_memory[session_id] = self.conversation_memory[session_id][-10:]
    
    def get_conversation_context(self, session_id: str) -> str:
        """Lấy context từ lịch sử conversation"""
        if session_id not in self.conversation_memory:
            return ""
        
        context = "Lịch sử cuộc trò chuyện:\n"
        for msg in self.conversation_memory[session_id][-5:]:  # Chỉ lấy 5 tin nhắn gần nhất
            context += f"Người dùng: {msg['user']}\n"
            context += f"AI: {msg['bot']}\n\n"
        
        return context
    
    def should_use_data_search(self, query: str, session_id: str = None) -> bool:
        """Phân biệt câu hỏi cần tìm kiếm data hay chỉ cần LLM với conversation context"""
        query_lower = query.lower().strip()
        
        # ENHANCED: Kiểm tra context conversation history để xác định câu hỏi follow-up về data
        has_conversation_history = (session_id and 
                                   session_id in self.conversation_memory and 
                                   len(self.conversation_memory[session_id]) > 0)
        
        if has_conversation_history:
            # Lấy câu trả lời bot gần nhất để hiểu context
            last_bot_response = self.conversation_memory[session_id][-1].get('bot', '')
            
            # SMART CONTEXT DETECTION: Nếu câu trả lời trước có chứa thông tin về FPTU/môn học
            # và câu hỏi hiện tại là follow-up về học kỳ khác -> cần data search
            fptu_context_indicators = [
                'môn học', 'mon hoc', 'subject', 'course', 'tín chỉ', 'tin chi', 'credit',
                'kỳ', 'ky', 'semester', 'ngành ai', 'nganh ai', 'fptu', 'fpt university',
                'csi', 'mad', 'csd', 'dbi', 'aig', 'cea', 'jpd', 'ady', 'ite'  # Common subject prefixes
            ]
            
            has_fptu_context = any(indicator in last_bot_response.lower() for indicator in fptu_context_indicators)
            
            # ENHANCED: Phát hiện câu hỏi follow-up về kì học khác
            semester_followup_patterns = [
                r'kì\s*(\d+|hai|ba|bốn|năm|sáu|bảy|tám)\s*(thì\s*sao|như\s*thế\s*nào|ra\s*sao)',
                r'ky\s*(\d+|hai|ba|bốn|năm|sáu|bảy|tám)\s*(thì\s*sao|như\s*thế\s*nào|ra\s*sao)',
                r'semester\s*(\d+|two|three|four|five|six|seven|eight)\s*(how|what)',
                r'(còn|con)\s*kì\s*(\d+|hai|ba|bốn|năm|sáu|bảy|tám)',
                r'(còn|con)\s*ky\s*(\d+|hai|ba|bốn|năm|sáu|bảy|tám)',
                r'(kì|ky)\s*(\d+|hai|ba|bốn|năm|sáu|bảy|tám)\s*(có|co)\s*(gì|gi|những\s*gì)',
                r'(học|hoc)\s*(kì|ky)\s*(\d+|hai|ba|bốn|năm|sáu|bảy|tám)',
                r'(thì|thi)\s*(kì|ky)\s*(\d+|hai|ba|bốn|năm|sáu|bảy|tám)',
                # Thêm các pattern khác
                r'kì\s*tiếp\s*theo', r'ky\s*tiep\s*theo',
                r'kì\s*sau', r'ky\s*sau',
                r'kì\s*khác', r'ky\s*khac'
            ]
            
            is_semester_followup = any(re.search(pattern, query_lower) for pattern in semester_followup_patterns)
            
            if has_fptu_context and is_semester_followup:
                logger.info("SMART DETECTION: Semester follow-up question with FPTU context - using DATA SEARCH")
                return True
        
        # Kiểm tra các pattern chỉ cần conversation context (KHÔNG cần data search)
        conversation_only_patterns = [
            # Tham chiếu đến câu trả lời trước (nhưng KHÔNG phải về kì học)
            r'(trong|ở|từ)?\s*(danh sách|bảng|kết quả|thông tin)\s*(trên|này|đó|vừa|ở trên)',
            r'(môn|item|mục)\s*(đầu tiên|cuối cùng|thứ \d+|số \d+)',
            r'(theo|dựa vào|từ)\s*(thông tin|dữ liệu|kết quả)\s*(trên|vừa|đã)',
            r'(cái|thứ|món)\s*(đầu|cuối|nào)\s*(trong|ở)',
            r'(first|last|which)\s*(one|item)',
            
            # Câu hỏi về conversation trước
            r'(bạn|em|mình)\s*(vừa|đã|vửa)\s*(nói|trả lời|đưa ra)',
            r'(từ|theo)\s*(câu trả lời|thông tin)\s*(trước|phía trên)',
            r'(ý nghĩa|hiểu|giải thích).*(?!môn|ngành)',  # Trừ khi hỏi về môn/ngành
            
            # Pure greetings/chit-chat (KHÔNG liên quan FPTU)
            r'^(xin chào|chào|hi|hello)$',
            r'^(cảm ơn|thank|thanks).*',
            r'^(tạm biệt|bye|goodbye).*',
            r'(bạn|em)\s*(có thể|có thể)\s*(giúp|hỗ trợ|làm)\s*(gì|được gì)',
        ]
        
        # Kiểm tra conversation context patterns trước (NHƯNG BỎ QUA pattern follow-up về học kì)
        for pattern in conversation_only_patterns:
            if re.search(pattern, query_lower):
                # ĐẶC BIỆT: Nếu pattern match nhưng có chứa từ khóa về kì học -> vẫn cần data search
                if any(semester_word in query_lower for semester_word in ['kì', 'ky', 'semester']):
                    logger.info(f"CONVERSATION PATTERN '{pattern}' matched but contains semester keyword - still using DATA SEARCH")
                    return True
                logger.info(f"CONVERSATION CONTEXT DETECTED: Pattern '{pattern}' matched")
                return False
        
        # ENHANCED: DOMAIN-SPECIFIC PATTERNS - FPTU AI context mạnh mẽ hơn
        fptu_domain_patterns = [
            # Ngành AI context (ngầm định về FPTU)
            r'(môn|mon).*ngành.*ai(?!.*nào)',  # "môn ngành AI" 
            r'ngành.*ai(?!.*nào).*môn',
            r'toàn bộ.*môn.*ai',
            r'danh sách.*môn.*ai(?!.*nào)',
            
            # Kì học patterns (ENHANCED)
            r'kì\s*(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám)',
            r'ky\s*(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám)', 
            r'semester\s*(\d+|one|two|three|four|five|six|seven|eight)',
            r'học\s*kì\s*(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám)',
            r'hoc\s*ky\s*(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám)',
            
            # FPTU-specific terms
            r'combo.*chuyên ngành',
            r'chuyên ngành.*hẹp',
            r'combo.*hẹp',
            r'combo.*ai',  # "combo AI"
            r'fptu?.*ai',
            r'fpt.*university.*ai',
            
            # Course-related với context AI
            r'(môn|mon).*(ai|artificial intelligence)',
            r'curriculum.*ai',
            r'chương trình.*ai',
            
            # Khi hỏi chung về FPTU domain
            r'môn.*(?:nào|gì).*(?:thuộc|trong).*ai',
            r'ai.*có.*môn.*nào',
        ]
        
        # Kiểm tra FPTU domain patterns
        for pattern in fptu_domain_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"FPTU DOMAIN CONTEXT DETECTED: Pattern '{pattern}' matched")
                return True
        
        # Các pattern CẦN tìm kiếm data (traditional patterns)
        data_search_patterns = [
            # Tìm kiếm môn học cụ thể (case insensitive)
            r'[a-zA-Z]{2,4}\d{3}[a-z]*\s+(?:là|gì|thông tin|chi tiết|môn)',  # Mã môn học với context câu hỏi
            r'[a-zA-Z]{2,4}\d{3}[a-z]*$',  # Mã môn học đơn thuần
            r'[a-zA-Z]{2,4}\d{3}[a-z]*',  # Mã môn học general
            r'môn học.*(?:nào|gì|là)', r'mon hoc', r'subject', r'course',
            r'tín chỉ', r'tin chi', r'credit',
            r'syllabus', r'curriculum', r'chương trình',
            r'ngành.*(?:nào|gì|có)', r'nganh', r'major',
            r'tiên quyết', r'tien quyet', r'prerequisite',
            r'CLO', r'learning outcome',
            
            # Từ khóa tìm kiếm TOÀN BỘ data (không phải follow-up)
            r'^(liệt kê|liet ke)', r'^(danh sách|danh sach)(?!.*trên)',
            r'^(tất cả|tat ca)', r'^(các môn|cac mon)',
            r'thông tin.*(?:về|của|môn)', r'chi tiết.*(?:về|của|môn)',
            r'bao nhiêu.*(?:môn|tín chỉ|credit)',
            
            # Tên môn học phổ biến
            r'machine learning', r'artificial intelligence', r'data science',
            r'programming', r'algorithm', r'database', r'network',
            r'mathematics', r'toán(?:.*học)?', r'toan', r'physics', r'vật lý',
        ]
        
        # Kiểm tra xem có pattern data search nào match không
        for pattern in data_search_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"DATA SEARCH DETECTED: Pattern '{pattern}' matched")
                return True
        
        logger.info("NO DATA SEARCH PATTERNS - using conversation/direct chat")
        return False
    
    def chat_direct(self, question: str, session_id: str = None) -> str:
        """Chat trực tiếp với LLM không cần tìm kiếm data"""
        try:
            # Lấy context từ conversation history nếu có session_id
            conversation_context = ""
            if session_id:
                conversation_context = self.get_conversation_context(session_id)
            
            # Kiểm tra xem có phải câu hỏi follow-up về dữ liệu vừa trả lời không
            is_followup_about_data = (
                conversation_context and 
                any(indicator in question.lower() for indicator in [
                    'danh sách trên', 'bảng trên', 'thông tin trên',
                    'đầu tiên', 'cuối cùng', 'thứ', 'trong danh sách'
                ])
            )
            
            # Kiểm tra có phải câu hỏi về FPTU domain mà được misclassified không
            fptu_related_but_misclassified = any(indicator in question.lower() for indicator in [
                'môn ngành ai', 'ngành ai', 'combo chuyên ngành', 'chuyên ngành hẹp',
                'fptu', 'fpt university', 'toàn bộ môn ai', 'danh sách môn ai'
            ])
            
            if is_followup_about_data:
                prompt = f"""Bạn là AI Assistant của FPTU. Hãy trả lời câu hỏi dựa trên CHÍNH XÁC thông tin đã cung cấp trong cuộc trò chuyện trước.

{conversation_context}

QUAN TRỌNG: 
- Dựa vào CHÍNH XÁC thông tin trong lịch sử cuộc trò chuyện
- Nếu hỏi về "đầu tiên", hãy tìm item đầu tiên trong danh sách/bảng đã trả lời
- Nếu hỏi về "cuối cùng", hãy tìm item cuối cùng trong danh sách/bảng đã trả lời  
- Không tự tạo ra thông tin mới
- Trả lời ngắn gọn và chính xác

Câu hỏi của người dùng: {question}

Trả lời:"""
            elif fptu_related_but_misclassified:
                # Gợi ý người dùng hỏi cụ thể hơn về FPTU
                prompt = f"""Bạn là AI Assistant của FPTU - chuyên hỗ trợ thông tin về trường FPT University.

{conversation_context}

Tôi hiểu bạn đang hỏi về thông tin liên quan đến FPT University và ngành AI. Để tôi có thể cung cấp thông tin chính xác nhất, bạn có thể hỏi cụ thể hơn như:

- "Liệt kê các môn học ngành AI theo kỳ"
- "Danh sách combo chuyên ngành AI" 
- "Các môn học trong chương trình AI FPTU"
- "Thông tin chi tiết về môn [tên môn]"

Hướng dẫn trả lời:
- Trả lời thân thiện và gợi ý cách hỏi hiệu quả
- Nhấn mạnh đây là hệ thống hỗ trợ thông tin FPTU
- Khuyến khích hỏi cụ thể để có kết quả tốt nhất
- Không sử dụng biểu tượng cảm xúc hay icon

Câu hỏi của người dùng: {question}

Trả lời:"""
            else:
                prompt = f"""Bạn là AI Assistant của FPTU - trợ lý thông minh hỗ trợ sinh viên.

{conversation_context}

Hướng dẫn trả lời:
- Trả lời bằng tiếng Việt một cách tự nhiên và thân thiện
- Nếu câu hỏi liên quan đến môn học cụ thể, gợi ý người dùng hỏi cụ thể hơn
- Giữ phong cách trò chuyện tự nhiên và hữu ích
- Không sử dụng biểu tượng cảm xúc hay icon

Câu hỏi của người dùng: {question}

Trả lời:"""
            
            answer = self._call_gemini_with_rotation(prompt)
            
            # Lưu vào conversation memory nếu có session_id
            if session_id:
                self.add_to_conversation(session_id, question, answer)
            
            return answer
            
        except Exception as e:
            logger.error(f"Lỗi chat_direct: {e}")
            return "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi của bạn. Bạn có thể thử lại không?"
    
    def chatbot_query(self, question: str, session_id: str, enable_multihop: bool = False) -> Dict[str, Any]:
        """
        Main chatbot method - kết hợp Query Preprocessing, RAG search và conversation management
        
        Args:
            question: Câu hỏi của người dùng
            session_id: ID phiên để quản lý conversation
            enable_multihop: Có cho phép multi-hop query không
            
        Returns:
            Dict chứa answer, search_results, metadata và conversation info
        """
        logger.info(f"========== CHATBOT QUERY ==========")
        logger.info(f"SESSION: {session_id}")
        logger.info(f"USER QUERY: '{question}'")
        logger.info(f"MULTIHOP ENABLED: {enable_multihop}")
        
        start_time = time.time()
        
        try:
            # BƯỚC 1: Query Preprocessing với LLM
            conversation_context = self.get_conversation_context(session_id)
            conversation_summary = ""
            if conversation_context:
                # Tạo summary ngắn gọn cho context
                last_exchanges = conversation_context.split('\n')[-4:]  # 2 lượt cuối
                conversation_summary = '\n'.join(last_exchanges)
            
            preprocessed = self.query_preprocessor.preprocess_query(question, conversation_summary)
            
            # BƯỚC 2: Quyết định strategy dựa trên kết quả preprocessing
            use_data_search = preprocessed.needs_data_search
            final_query = preprocessed.processed_query if preprocessed.confidence > 0.6 else question
            
            logger.info(f"PREPROCESSING DECISION:")
            logger.info(f"  Use preprocessed query: {preprocessed.confidence > 0.6}")
            logger.info(f"  Final query: '{final_query}'")
            logger.info(f"  Strategy: {'DATA SEARCH' if use_data_search else 'DIRECT CHAT'}")
            
            if use_data_search:
                # Sử dụng RAG với query đã được preprocessing
                logger.info("Executing RAG search with preprocessed query...")
                
                if enable_multihop:
                    result = self.query_with_multihop(final_query, enable_multihop)
                    query_type = 'data_search_multihop'
                    # For multihop, the answer is in 'final_answer' key
                    answer = result.get('final_answer', result.get('answer', ''))
                else:
                    result = self.query(final_query)
                    query_type = 'data_search'
                    answer = result.get('answer', '')
                
                # Lưu vào conversation memory (dùng question gốc)
                self.add_to_conversation(session_id, question, answer)
                
                # Build response structure for multihop vs regular
                if enable_multihop:
                    # For multihop, build consistent response structure
                    response_result = {
                        'answer': answer,
                        'search_results': result.get('followup_results', []),  # Use followup results as search results
                        'metadata': {
                            'query_type': query_type,
                            'is_data_search': True,
                            'is_direct_chat': False,
                            'conversation_enhanced': bool(conversation_context),
                            'subjects_covered': 0,  # Will be calculated below
                            'preprocessing_info': {
                                'original_query': preprocessed.original_query,
                                'processed_query': preprocessed.processed_query,
                                'intent': preprocessed.intent_description,
                                'confidence': preprocessed.confidence,
                                'keywords': preprocessed.suggested_keywords,
                                'query_improved': preprocessed.confidence > 0.6
                            }
                        },
                        'multihop_info': {
                            'has_followup': result.get('has_followup', False),
                            'followup_queries': result.get('followup_queries', []),
                            'execution_path': result.get('execution_path', []),
                            'original_answer': result.get('original_answer', ''),
                            'final_answer': result.get('final_answer', answer)
                        }
                    }
                    result = response_result
                else:
                    # Set metadata với preprocessing info
                    if 'metadata' not in result:
                        result['metadata'] = {}
                    
                    result['metadata']['query_type'] = query_type
                    result['metadata']['is_data_search'] = True
                    result['metadata']['is_direct_chat'] = False
                    result['metadata']['conversation_enhanced'] = bool(conversation_context)
                    result['metadata']['preprocessing_info'] = {
                        'original_query': preprocessed.original_query,
                        'processed_query': preprocessed.processed_query,
                        'intent': preprocessed.intent_description,
                        'confidence': preprocessed.confidence,
                        'keywords': preprocessed.suggested_keywords,
                        'query_improved': preprocessed.confidence > 0.6
                    }
                    
                    # Add multihop info if not present
                    if 'multihop_info' not in result:
                        result['multihop_info'] = {'has_followup': False}
                
                logger.info(f"CHATBOT RESPONSE:")
                logger.info(f"  - Query type: {result['metadata']['query_type']}")
                logger.info(f"  - Preprocessing confidence: {preprocessed.confidence:.2f}")
                logger.info(f"  - Used improved query: {preprocessed.confidence > 0.6}")
                logger.info(f"  - Response length: {len(answer)} chars")
                logger.info(f"  - Search results: {len(result.get('search_results', []))}")
                logger.info(f"  - Processing time: {time.time() - start_time:.2f}s")
                
                return result
                
            else:
                # Sử dụng direct chat
                logger.info("Executing direct chat...")
                answer = self.chat_direct(question, session_id)
                
                # Build result structure consistent với RAG response
                result = {
                    'answer': answer,
                    'search_results': [],
                    'metadata': {
                        'query_type': 'direct_chat',
                        'is_data_search': False,
                        'is_direct_chat': True,
                        'conversation_enhanced': True,
                        'subjects_covered': 0,
                        'preprocessing_info': {
                            'original_query': preprocessed.original_query,
                            'processed_query': preprocessed.processed_query,
                            'intent': preprocessed.intent_description,
                            'confidence': preprocessed.confidence,
                            'keywords': preprocessed.suggested_keywords,
                            'query_improved': False
                        }
                    },
                    'multihop_info': {
                        'has_followup': False
                    }
                }
                
                logger.info(f"CHATBOT RESPONSE:")
                logger.info(f"  - Query type: {result['metadata']['query_type']}")
                logger.info(f"  - Preprocessing confidence: {preprocessed.confidence:.2f}")
                logger.info(f"  - Response length: {len(answer)} chars")
                logger.info(f"  - Search results: {len(result.get('search_results', []))}")
                logger.info(f"  - Processing time: {time.time() - start_time:.2f}s")
                
                return result
                
        except Exception as e:
            logger.error(f"Lỗi trong chatbot_query: {e}")
            
            # Fallback response
            fallback_answer = "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi của bạn. Bạn có thể thử lại không?"
            
            result = {
                'answer': fallback_answer,
                'search_results': [],
                'metadata': {
                    'query_type': 'error',
                    'is_data_search': False,
                    'is_direct_chat': False,
                    'conversation_enhanced': False,
                    'subjects_covered': 0,
                    'error': str(e),
                    'preprocessing_info': None
                },
                'multihop_info': {
                    'has_followup': False
                }
            }
            
            return result
            
        finally:
            total_time = time.time() - start_time
            logger.info(f"========== CHATBOT QUERY COMPLETE ==========")

    def _process_data(self, raw_data):
        processed_data = []
        
        if isinstance(raw_data, dict) and 'syllabuses' in raw_data:
            # Extract major information
            major_code = raw_data.get('major_code_input', 'UNKNOWN')
            curriculum_title = raw_data.get('curriculum_title_on_page', 'N/A')
            
            # 0. MAJOR/NGÀNH OVERVIEW ENTRY
            major_overview = f"""
Nganh hoc: {major_code}
Ten chuong trinh: {curriculum_title}
Tong so mon hoc: {len(raw_data['syllabuses'])} mon
Cac mon hoc trong nganh {major_code}:
"""
            
            for syllabus in raw_data['syllabuses']:
                metadata = syllabus.get('metadata', {})
                subject_code = metadata.get('subject_code_on_page', 'UNKNOWN')
                course_name = metadata.get('title', 'N/A')
                credits = metadata.get('credits', 'N/A')
                semester = metadata.get('semester_from_curriculum', 'N/A')
                major_overview += f"- {subject_code}: {course_name} ({credits} tin chi, Ky {semester})\n"
            
            processed_data.append({
                'content': major_overview,
                'subject_code': 'MAJOR_OVERVIEW',
                'type': 'major_overview',
                'major_code': major_code,
                'metadata': {
                    'major_code': major_code,
                    'curriculum_title': curriculum_title,
                    'total_subjects': len(raw_data['syllabuses']),
                    'type': 'major_overview',
                    'search_keywords': f"nganh {major_code} chuong trinh hoc tat ca mon hoc liet ke danh sach"
                }
            })
            
            # Create combo specialization content
            combo_groups = {}
            for syllabus in raw_data['syllabuses']:
                metadata = syllabus.get('metadata', {})
                combo = metadata.get('combo_short_name_from_curriculum', '').strip()
                
                if combo:  # Only process subjects with actual combo names
                    if combo not in combo_groups:
                        combo_groups[combo] = []
                    combo_groups[combo].append({
                        'subject_code': metadata.get('subject_code_on_page', ''),
                        'title': metadata.get('title', ''),
                        'credits': metadata.get('credits', ''),
                        'semester': metadata.get('semester_from_curriculum', ''),
                        'description': metadata.get('description', '')[:200] + '...' if metadata.get('description') else ''
                    })
            
            # Create combo overview content
            for combo_name, subjects in combo_groups.items():
                combo_content = f"""
Combo chuyen nganh hep: {combo_name}
Nganh: {major_code}
So luong mon hoc: {len(subjects)} mon
Cac mon hoc trong combo:
"""
                for subject in subjects:
                    combo_content += f"- {subject['subject_code']}: {subject['title']} ({subject['credits']} tin chi, Ky {subject['semester']})\n"
                    if subject['description']:
                        combo_content += f"  Mo ta: {subject['description']}\n"
                
                # Add combo description based on name
                if 'COM1' in combo_name:
                    combo_content += "\nChuyen nganh: Data Science va Big Data Analytics"
                    combo_content += "\nMo ta: Tap trung vao khoa hoc du lieu, khai pha du lieu va phan tich du lieu lon"
                elif 'COM3' in combo_name:
                    combo_content += "\nChuyen nganh: AI for Healthcare va Research"
                    combo_content += "\nMo ta: Ung dung AI trong y te, nghien cuu khoa hoc va giao dich tai chinh"
                elif 'COM2' in combo_name:
                    combo_content += "\nChuyen nganh: Text Mining va Search Engineering"
                    combo_content += "\nMo ta: Khai thac van ban, xu ly ngon ngu tu nhien va cong cu tim kiem"
                
                processed_data.append({
                    'content': combo_content,
                    'subject_code': f'COMBO_{combo_name}',
                    'type': 'combo_specialization',
                    'major_code': major_code,
                    'combo_name': combo_name,
                    'subject_count': len(subjects),
                    'metadata': {
                        'major_code': major_code,
                        'combo_name': combo_name,
                        'type': 'combo_specialization',
                        'subject_count': len(subjects),
                        'search_keywords': f"combo chuyen nganh hep {combo_name} {major_code} specialization track"
                    }
                })
            
            for syllabus in raw_data['syllabuses']:
                metadata = syllabus.get('metadata', {})
                subject_code = metadata.get('subject_code_on_page', 'UNKNOWN')
                course_name = metadata.get('title', 'N/A')
                credits = metadata.get('credits', 'N/A')
                semester = metadata.get('semester_from_curriculum', 'N/A')
                
                # 1. COMPREHENSIVE GENERAL INFO
                general_content = f"""
Nganh: {major_code}
Mon hoc: {course_name}
Ma mon: {subject_code}
So tin chi: {credits}
Ky hoc: {semester}
Mon tien quyet: {metadata.get('prerequisites', 'Khong co')}
Mo ta: {metadata.get('description', 'N/A')}
Nhiem vu sinh vien: {metadata.get('student_tasks', 'N/A')}
"""
                
                processed_data.append({
                    'content': general_content,
                    'subject_code': subject_code,
                    'type': 'general_info',
                    'major_code': major_code,
                    'metadata': {
                        'major_code': major_code,
                        'subject_code': subject_code,
                        'course_name': course_name,
                        'credits': credits,
                        'semester': semester,
                        'semester_from_curriculum': metadata.get('semester_from_curriculum'),  # For filtering
                        'prerequisites': metadata.get('prerequisites', ''),
                        'course_type_guess': metadata.get('course_type_guess', ''),  # CRITICAL: For Coursera boost
                        'search_keywords': f"nganh {major_code} {subject_code} {course_name} {credits} tin chi ky {semester}"
                    }
                })
                
                # 2. DETAILED CLO PROCESSING
                if 'learning_outcomes' in syllabus:
                    clos = syllabus['learning_outcomes']
                    clo_count = len(clos)
                    
                    # Summary CLO info
                    clo_summary = f"""
Nganh {major_code} - Mon {subject_code} ({course_name}) co {clo_count} CLO (Course Learning Outcomes):
"""
                    for i, clo in enumerate(clos, 1):
                        clo_summary += f"{clo.get('id', f'CLO{i}')}: {clo.get('details', '')[:100]}...\n"
                    
                    processed_data.append({
                        'content': clo_summary,
                        'subject_code': subject_code,
                        'type': 'learning_outcomes_summary',
                        'major_code': major_code,
                        'clo_count': clo_count,
                        'metadata': {
                            'major_code': major_code,
                            'subject_code': subject_code,
                            'type': 'CLO_summary',
                            'count': clo_count,
                            'search_keywords': f"nganh {major_code} {subject_code} CLO learning outcomes {clo_count} CLO"
                        }
                    })
                    
                    # Individual CLO details
                    for clo in clos:
                        clo_detail = f"""
Nganh {major_code} - Mon {subject_code} - {clo.get('id', 'CLO')}: {clo.get('details', 'Khong co mo ta')}
"""
                        processed_data.append({
                            'content': clo_detail,
                            'subject_code': subject_code,
                            'type': 'learning_outcome_detail',
                            'major_code': major_code,
                            'clo_id': clo.get('id', ''),
                            'metadata': {
                                'major_code': major_code,
                                'subject_code': subject_code,
                                'clo_id': clo.get('id', ''),
                                'type': 'CLO_detail',
                                'search_keywords': f"nganh {major_code} {subject_code} {clo.get('id', '')} learning outcome"
                            }
                        })
                
                # 3. MATERIALS PROCESSING
                if 'materials' in syllabus:
                    materials = syllabus['materials']
                    main_materials = [m for m in materials if m.get('is_main_material', False)]
                    
                    materials_content = f"""
Tai lieu mon {subject_code} - Nganh {major_code} ({len(materials)} tai lieu):
Tai lieu chinh: {len(main_materials)} tai lieu
"""
                    for mat in materials:
                        mat_type = "Chinh" if mat.get('is_main_material') else "Phu"
                        materials_content += f"- [{mat_type}] {mat.get('description', '')}"
                        if mat.get('author'):
                            materials_content += f" - {mat.get('author', '')}"
                        materials_content += "\n"
                    
                    processed_data.append({
                        'content': materials_content,
                        'subject_code': subject_code,
                        'type': 'materials',
                        'major_code': major_code,
                        'material_count': len(materials),
                        'metadata': {
                            'major_code': major_code,
                            'subject_code': subject_code,
                            'type': 'materials',
                            'count': len(materials),
                            'main_count': len(main_materials),
                            'search_keywords': f"nganh {major_code} {subject_code} tai lieu materials textbook"
                        }
                    })
                
                # 4. ASSESSMENTS PROCESSING với Special Notes Enhancement
                if 'assessments' in syllabus:
                    assessments = syllabus['assessments']
                    
                    assessment_content = f"""
Danh gia mon {subject_code} - Nganh {major_code} ({len(assessments)} loai):
"""
                    total_weight = 0
                    for assess in assessments:
                        weight = float(assess.get('weight', 0))
                        total_weight += weight
                        assessment_content += f"- {assess.get('category', '')}: {weight}% ({assess.get('type', '')})\n"
                        if assess.get('clos'):
                            assessment_content += f"  CLO lien quan: {', '.join(assess.get('clos', []))}\n"
                    
                    assessment_content += f"Tong trong so: {total_weight}%"
                    
                    processed_data.append({
                        'content': assessment_content,
                        'subject_code': subject_code,
                        'type': 'assessments',
                        'major_code': major_code,
                        'assessment_count': len(assessments),
                        'total_weight': total_weight,
                        'metadata': {
                            'major_code': major_code,
                            'subject_code': subject_code,
                            'type': 'assessments',
                            'count': len(assessments),
                            'total_weight': total_weight,
                            'search_keywords': f"nganh {major_code} {subject_code} danh gia assessment exam test weight"
                        }
                    })
                    
                    # SPECIAL NOTES PROCESSING - Extract bonus paper và các note đặc biệt
                    # Check both syllabus-level và assessment-level completion criteria và notes
                    completion_criteria = syllabus.get('completion_criteria', '')
                    note_content = syllabus.get('note', '')
                    
                    # Also check individual assessments for completion_criteria (especially final exam)
                    assessment_criteria = ""
                    for assess in assessments:
                        assess_criteria = assess.get('completion_criteria', '')
                        if assess_criteria:
                            assessment_criteria += f" {assess_criteria}"
                    
                    # Combine all sources để tìm special features
                    combined_special_text = f"{completion_criteria}\n{note_content}\n{assessment_criteria}".lower()
                    
                    special_features = []
                    special_content = ""
                    
                    # Detect Bonus Paper Scoring
                    if 'bonus' in combined_special_text and 'paper' in combined_special_text:
                        if 'scopus' in combined_special_text or 'isi' in combined_special_text:
                            special_features.append('bonus_paper_scoring')
                            
                            # Extract the bonus scoring section từ tất cả nguồn
                            bonus_section = ""
                            if 'bonus score for accepted paper' in combined_special_text:
                                # Check trong completion_criteria
                                all_text_sources = [completion_criteria, note_content, assessment_criteria]
                                for text_source in all_text_sources:
                                    if 'bonus score for accepted paper' in text_source.lower():
                                        lines = text_source.split('\n')
                                        in_bonus_section = False
                                        for line in lines:
                                            if 'bonus score for accepted paper' in line.lower():
                                                in_bonus_section = True
                                            if in_bonus_section:
                                                bonus_section += line + "\n"
                                                if 'source to check' in line.lower():
                                                    break
                                        if bonus_section.strip():
                                            break
                            
                            special_content += f"""
MON {subject_code} CO DIEM THUONG PAPER KHOA HOC:

{bonus_section}
"""
                    
                    # Detect Project-based Assessment
                    if 'capstone' in combined_special_text or 'project' in combined_special_text:
                        if 'oral presentation' in combined_special_text or 'final presentation' in combined_special_text:
                            special_features.append('capstone_project_assessment')
                            special_content += f"Mon {subject_code}: Co capstone project voi oral presentation\n"
                    
                    # Detect MOOC Requirements
                    if 'mooc' in combined_special_text or 'specialization' in combined_special_text:
                        special_features.append('mooc_required')
                        special_content += f"Mon {subject_code}: Yeu cau hoan thanh MOOC/certification\n"
                    
                    # Detect Special Language Requirements
                    if any(lang in combined_special_text for lang in ['korean', 'japanese', 'kor101', 'jpd']):
                        special_features.append('language_requirement')
                        special_content += f"Mon {subject_code}: Co yeu cau ngoai ngu dac biet\n"
                    
                    # Create special features chunk if any special features found
                    if special_features and special_content.strip():
                        processed_data.append({
                            'content': special_content,
                            'subject_code': subject_code,
                            'type': 'special_features',
                            'major_code': major_code,
                            'special_features': special_features,
                            'has_bonus_paper': 'bonus_paper_scoring' in special_features,
                            'has_capstone': 'capstone_project_assessment' in special_features,
                            'has_mooc': 'mooc_required' in special_features,
                            'has_language_req': 'language_requirement' in special_features,
                            'metadata': {
                                'major_code': major_code,
                                'subject_code': subject_code,
                                'type': 'special_features',
                                'special_features': special_features,
                                'has_bonus_paper': 'bonus_paper_scoring' in special_features,
                                'has_capstone': 'capstone_project_assessment' in special_features,
                                'has_mooc': 'mooc_required' in special_features,
                                'has_language_req': 'language_requirement' in special_features,
                                'search_keywords': f"nganh {major_code} {subject_code} diem thuong bonus paper scopus isi special features {' '.join(special_features)}"
                            }
                        })
                
                # 5. SCHEDULE SUMMARY (sample sessions)
                if 'schedule' in syllabus:
                    schedule = syllabus['schedule']
                    schedule_sample = schedule[:3]  # First 3 sessions as sample
                    
                    schedule_content = f"""
Lich hoc mon {subject_code} - Nganh {major_code} (Tong {len(schedule)} buoi hoc):
Cac buoi hoc dau:
"""
                    for i, session in enumerate(schedule_sample, 1):
                        schedule_content += f"Buoi {i}: {session.get('session', '')[:100]}...\n"
                        schedule_content += f"  CLO: {session.get('teaching_type', '')}\n"
                    
                    if len(schedule) > 3:
                        schedule_content += f"... va {len(schedule)-3} buoi hoc khac"
                    
                    processed_data.append({
                        'content': schedule_content,
                        'subject_code': subject_code,
                        'type': 'schedule',
                        'major_code': major_code,
                        'session_count': len(schedule),
                        'metadata': {
                            'major_code': major_code,
                            'subject_code': subject_code,
                            'type': 'schedule',
                            'session_count': len(schedule),
                            'search_keywords': f"nganh {major_code} {subject_code} lich hoc schedule session buoi hoc"
                        }
                    })
        
        # STUDENT DATA PROCESSING
        if 'students' in raw_data and raw_data['students']:
            students = raw_data['students']
            major_code = raw_data.get('major_code_input', 'UNKNOWN')
            
            # Create student overview
            student_overview = f"""
Danh sach sinh vien nganh {major_code}:
Tong so sinh vien: {len(students)} sinh vien
"""
            
            # Group students by major for overview
            major_groups = {}
            for student in students:
                student_major = student.get('Major', 'Unknown')
                if student_major not in major_groups:
                    major_groups[student_major] = []
                major_groups[student_major].append(student)
            
            # Add major distribution to overview
            for major, student_list in major_groups.items():
                student_overview += f"Nganh {major}: {len(student_list)} sinh vien\n"
            
            # Add sample student list
            student_overview += "\nDanh sach sinh vien:\n"
            for student in students[:10]:  # First 10 students as sample
                student_overview += f"- {student.get('RollNumber', '')}: {student.get('Fullname', '')} ({student.get('Major', '')})\n"
            
            if len(students) > 10:
                student_overview += f"... va {len(students)-10} sinh vien khac"
            
            processed_data.append({
                'content': student_overview,
                'subject_code': 'STUDENT_LIST',
                'type': 'student_overview',
                'major_code': major_code,
                'student_count': len(students),
                'metadata': {
                    'major_code': major_code,
                    'type': 'student_overview',
                    'student_count': len(students),
                    'majors': list(major_groups.keys()),
                    'search_keywords': f"sinh vien {major_code} danh sach student list roll number"
                }
            })
            
            # Individual student records
            for student in students:
                student_detail = f"""
Thong tin sinh vien:
Ma sinh vien: {student.get('RollNumber', '')}
Ho ten: {student.get('Fullname', '')}
Email: {student.get('Email', '')}
Nganh: {student.get('Major', '')}
Ho: {student.get('LastName', '')}
Ten dem: {student.get('MiddleName', '')}
Ten: {student.get('FirstName', '')}
"""
                
                processed_data.append({
                    'content': student_detail,
                    'subject_code': 'STUDENT_DETAIL',
                    'type': 'student_detail',
                    'major_code': student.get('Major', ''),
                    'roll_number': student.get('RollNumber', ''),
                    'student_name': student.get('Fullname', ''),
                    'metadata': {
                        'major_code': student.get('Major', ''),
                        'roll_number': student.get('RollNumber', ''),
                        'student_name': student.get('Fullname', ''),
                        'email': student.get('Email', ''),
                        'type': 'student_detail',
                        'search_keywords': f"sinh vien {student.get('RollNumber', '')} {student.get('Fullname', '')} {student.get('Major', '')} student"
                    }
                })
        
        return processed_data

    def _create_embeddings(self):
        logger.info("Đang tạo embeddings...")
        contents = [item['content'] for item in self.data]
        self.embeddings = self.embedding_model.encode(contents)

    def _build_index(self):
        logger.info("Đang xây dựng FAISS index...")
        embeddings_np = np.array(self.embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(embeddings_np.shape[1])
        self.index.add(embeddings_np)

    def query(self, question: str, max_results: int = 10) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"========== BẮT ĐẦU XỬ LÝ TRUY VẤN ==========")
        logger.info(f"USER QUERY: '{question}'")
        
        if not self.is_initialized:
            logger.error("Engine chưa được khởi tạo")
            raise RuntimeError("Engine chưa được khởi tạo")
        
        # Check for quick response first
        logger.info("Kiểm tra quick response patterns...")
        quick_response = self.query_router.check_quick_response(question)
        if quick_response:
            logger.info("✓ Tìm thấy quick response pattern")
            logger.info(f"QUICK RESPONSE: {quick_response[:100]}...")
            processing_time = time.time() - start_time
            logger.info(f"========== KẾT THÚC XỬ LÝ (Quick Response) - Thời gian: {processing_time:.2f}s ==========")
            return {
                'question': question,
                'answer': quick_response,
                'search_results': [],
                'is_quick_response': True
            }
        
        logger.info("Không tìm thấy quick response, tiếp tục xử lý...")
        
        # Expand query
        logger.info("Mở rộng truy vấn...")
        expanded_query = self._expand_query(question)
        if expanded_query != question:
            logger.info(f"EXPANDED QUERY: '{expanded_query}'")
        
        # Analyze query intent
        logger.info("Phân tích ý định truy vấn...")
        intent = self.query_router.analyze_query(expanded_query)
        logger.info(f"QUERY INTENT:")
        logger.info(f"  - Type: {intent.query_type}")
        logger.info(f"  - Scope: {intent.subject_scope}")
        logger.info(f"  - Complexity: {intent.complexity}")
        logger.info(f"  - Requires summarization: {intent.requires_summarization}")
        if intent.target_subjects:
            logger.info(f"  - Target subjects: {intent.target_subjects}")
        else:
            logger.info(f"  - Target subjects: None")
        
        # Search strategy
        logger.info("Thực hiện search strategy...")
        search_results = self._search_strategy(expanded_query, intent)
        logger.info(f"SEARCH RESULTS: Tìm thấy {len(search_results)} kết quả")
        
        # Log chi tiết từng kết quả
        for i, result in enumerate(search_results[:5]):
            final_score = result.get('final_score', result.get('score', 0))
            search_method = result.get('search_method', 'unknown')
            content_preview = result.get('content', '')[:100].replace('\n', ' ')
            
            logger.info(f"  [{i+1}] {result.get('subject_code', 'N/A')}:")
            logger.info(f"      Type: {result.get('type', 'N/A')}")
            logger.info(f"      Score: {final_score:.4f}")
            logger.info(f"      Method: {search_method}")
            logger.info(f"      Content: {content_preview}...")
            
            # Log metadata nếu có
            metadata = result.get('metadata', {})
            if metadata:
                logger.info(f"      Metadata: {dict(list(metadata.items())[:3])}...")  # Chỉ hiển thị 3 fields đầu
        
        # Prepare context
        logger.info("Chuẩn bị context...")
        context = self._prepare_context(search_results)
        context_length = len(context)
        logger.info(f"CONTEXT PREPARATION:")
        logger.info(f"  - Total length: {context_length} ký tự")
        
        # Count sections (cannot use backslash in f-string)
        context_lines = context.split('\n')
        section_count = len([line for line in context_lines if line.startswith('**')])
        logger.info(f"  - Number of sections: {section_count}")
        
        # Log preview của context
        context_lines = context.split('\n')
        logger.info(f"CONTEXT PREVIEW (first 10 lines):")
        for i, line in enumerate(context_lines[:10]):
            if line.strip():
                logger.info(f"  {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
        
        if len(context_lines) > 10:
            logger.info(f"  ... và {len(context_lines) - 10} dòng khác")
        
        # Generate response
        logger.info("Gọi Gemini để tạo phản hồi...")
        response = self._generate_response(question, context)
        response_length = len(response)
        logger.info(f"GEMINI RESPONSE: Độ dài {response_length} ký tự")
        logger.info(f"RESPONSE PREVIEW: {response[:200]}...")
        
        processing_time = time.time() - start_time
        logger.info(f"========== KẾT THÚC XỬ LÝ - Thời gian: {processing_time:.2f}s ==========")
        
        return {
            'question': question,
            'answer': response,
            'search_results': search_results[:5],
            'is_quick_response': False
        }

    def query_with_multihop(self, question: str, enable_multihop: bool = True, max_results: int = 10) -> Dict[str, Any]:
        """
        Thực hiện truy vấn với khả năng multi-hop (truy vấn kép)
        
        Args:
            question: Câu hỏi gốc
            enable_multihop: Bật/tắt tính năng multi-hop
            max_results: Số kết quả tối đa
            
        Returns:
            Dict chứa kết quả chi tiết của chuỗi truy vấn
        """
        start_time = time.time()
        logger.info(f"========== BẮT ĐẦU XỬ LÝ MULTI-HOP QUERY ==========")
        logger.info(f"USER QUERY: '{question}'")
        logger.info(f"MULTI-HOP ENABLED: {enable_multihop}")
        
        if not self.is_initialized:
            logger.error("Engine chưa được khởi tạo")
            raise RuntimeError("Engine chưa được khởi tạo")
        
        # Check for quick response first (no need for multi-hop)
        logger.info("Kiểm tra quick response patterns...")
        quick_response = self.query_router.check_quick_response(question)
        if quick_response:
            logger.info("✓ Tìm thấy quick response pattern - bỏ qua multi-hop")
            logger.info(f"QUICK RESPONSE: {quick_response[:100]}...")
            processing_time = time.time() - start_time
            logger.info(f"========== KẾT THÚC XỬ LÝ (Quick Response) - Thời gian: {processing_time:.2f}s ==========")
            return {
                'question': question,
                'original_answer': quick_response,
                'final_answer': quick_response,
                'followup_queries': [],
                'followup_results': [],
                'execution_path': ['Quick response - không cần tìm kiếm database'],
                'multihop_enabled': False,
                'has_followup': False,
                'is_quick_response': True
            }
        
        if not self.query_chain:
            logger.warning("QueryChain không khả dụng - fallback to normal query")
            # Fallback to normal query if QueryChain not available
            normal_result = self.query(question, max_results)
            processing_time = time.time() - start_time
            logger.info(f"========== KẾT THÚC XỬ LÝ (Fallback) - Thời gian: {processing_time:.2f}s ==========")
            return {
                'question': question,
                'original_answer': normal_result['answer'],
                'final_answer': normal_result['answer'],
                'followup_queries': [],
                'followup_results': [],
                'execution_path': ['Fallback to normal query'],
                'multihop_enabled': False,
                'has_followup': False,
                'is_quick_response': normal_result.get('is_quick_response', False)
            }
        
        # Thực hiện truy vấn chuỗi
        logger.info("Bắt đầu thực hiện query chain...")
        chain_result = self.query_chain.execute_query_chain(question, enable_multihop)
        
        logger.info(f"FOLLOWUP QUERIES: {len(chain_result.followup_queries)} truy vấn")
        for i, fq in enumerate(chain_result.followup_queries):
            logger.info(f"  Followup {i+1}: '{fq.query}' (confidence: {fq.confidence:.2f})")
        
        processing_time = time.time() - start_time
        logger.info(f"FINAL ANSWER LENGTH: {len(chain_result.final_integrated_answer)} ký tự")
        logger.info(f"FINAL ANSWER PREVIEW: {chain_result.final_integrated_answer[:200]}...")
        logger.info(f"========== KẾT THÚC XỬ LÝ MULTI-HOP - Thời gian: {processing_time:.2f}s ==========")
        
        return {
            'question': question,
            'original_answer': chain_result.original_answer,
            'final_answer': chain_result.final_integrated_answer,
            'followup_queries': [
                {
                    'query': fq.query,
                    'confidence': fq.confidence,
                    'type': fq.query_type,
                    'source': fq.source_info
                } for fq in chain_result.followup_queries
            ],
            'followup_results': chain_result.followup_results,
            'execution_path': chain_result.execution_path,
            'multihop_enabled': enable_multihop,
            'has_followup': len(chain_result.followup_queries) > 0,
            'is_quick_response': False
        }

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and multilingual terms for better matching"""
        
        # Define common synonym mappings
        query_synonyms = {
            # Vietnamese-English mappings  
            'tín chỉ': ['credit', 'credits'],
            'tin chi': ['credit', 'credits'], 
            'bao nhiêu': ['how many', 'số lượng'],
            'có bao nhiêu': ['how many', 'số lượng'],
            'môn học': ['subject', 'course'],
            'mon hoc': ['subject', 'course'],
            'ngành': ['major', 'program'],
            'nganh': ['major', 'program'],
            'kỳ học': ['semester', 'term'],
            'ky hoc': ['semester', 'term'],
            'đánh giá': ['assessment', 'evaluation'],
            'danh gia': ['assessment', 'evaluation'],
            'tài liệu': ['material', 'resource'],
            'tai lieu': ['material', 'resource'],
            'lịch học': ['schedule', 'timetable'],
            'lich hoc': ['schedule', 'timetable']
        }
        
        expanded_query = query
        query_lower = query.lower()
        
        # Add synonyms to query
        for vn_term, en_terms in query_synonyms.items():
            if vn_term in query_lower:
                # Add English equivalents
                for en_term in en_terms:
                    if en_term not in query_lower:
                        expanded_query += f" {en_term}"
        
        return expanded_query

    def _search_strategy(self, query: str, intent: QueryIntent) -> List[Dict[str, Any]]:
        """Enhanced search strategy với intelligent routing"""
        search_start_time = time.time()
        logger.info(f"SEARCH STRATEGY: Bắt đầu tìm kiếm")
        logger.info(f"  Query: '{query}'")
        logger.info(f"  Intent: {intent.query_type} / {intent.subject_scope} / {intent.complexity}")
        logger.info(f"  Target subjects: {intent.target_subjects}")
        
        # Log search configuration
        config = self._get_search_config(intent, query.lower())
        logger.info(f"SEARCH CONFIG:")
        logger.info(f"  - Max results: {config['max_results']}")
        logger.info(f"  - Content types: {config['content_types']}")
        logger.info(f"  - Boost factors: {config['boost_factors']}")
        
        # Log special configurations
        if config.get('force_all_combos'):
            logger.info(f"  - SPECIAL: Force all combos enabled")
        if config.get('coursera_boost'):
            logger.info(f"  - SPECIAL: Coursera boost enabled")
        if config.get('smart_filter_semester'):
            logger.info(f"  - SMART FILTER: semester={config['smart_filter_semester']}, suffix='{config.get('smart_filter_suffix', '')}'")
        if config.get('force_specific_student'):
            logger.info(f"  - STUDENT FILTER: targeting {config['force_specific_student']}")
        if config.get('force_student_overview'):
            logger.info(f"  - STUDENT FILTER: prioritizing overview")
        
        results = []
        
        # PRIORITY CHECK: For semester/major queries, ensure major_overview is found
        query_lower = query.lower()
        is_semester_query = any(term in query_lower for term in ['kỳ', 'ky', 'kì', 'ki', 'semester', 'học kỳ', 'hoc ky'])
        is_major_query = any(term in query_lower for term in ['ngành', 'nganh', 'major', 'chương trình'])
        
        # Get search configuration
        config = self._get_search_config(intent, query_lower)
        
        # STEP 0: FORCE ALL COMBOS for listing combo queries
        if config.get('force_all_combos', False):
            all_combo_items = [item for item in self.data if item.get('type') == 'combo_specialization']
            
            if all_combo_items:
                print(f"FORCED all combos: {len(all_combo_items)} items")
                query_embedding = self.embedding_model.encode([query])
                
                for item in all_combo_items:
                    # Find original index
                    original_idx = None
                    for i, orig_item in enumerate(self.data):
                        if orig_item is item:
                            original_idx = i
                            break
                    
                    if original_idx is not None:
                        item_embedding = self.embeddings[original_idx:original_idx+1]
                        
                        # Calculate inner product (to match FAISS)
                        score = float(np.dot(query_embedding[0], item_embedding[0]))
                        
                        # Force very high scores for all combos
                        score *= config['boost_factors'].get('combo_specialization', 15.0)
                        score += 50.0  # Base boost to ensure all combos appear
                        
                        results.append({
                            'content': item['content'],
                            'subject_code': item['subject_code'],
                            'type': item['type'],
                            'score': score,
                            'metadata': item.get('metadata', {}),
                            'search_method': 'forced_all_combos'
                        })
                        
                        print(f"FORCED combo {item['subject_code']} with score: {score:.4f}")
        
        # STEP 1: FORCE MAJOR_OVERVIEW for semester/major queries
        if (is_semester_query or is_major_query) and 'major_overview' in config['content_types']:
            major_overview_items = [item for item in self.data if item.get('type') == 'major_overview']
            
            if major_overview_items:
                # Calculate similarity for major_overview
                query_embedding = self.embedding_model.encode([query])
                
                for item in major_overview_items:
                    # Find original index
                    original_idx = None
                    for i, orig_item in enumerate(self.data):
                        if orig_item is item:
                            original_idx = i
                            break
                    
                    if original_idx is not None:
                        item_embedding = self.embeddings[original_idx:original_idx+1]
                        
                        # Calculate inner product (to match FAISS)
                        score = float(np.dot(query_embedding[0], item_embedding[0]))
                        
                        # Apply boost factor
                        if 'major_overview' in config['boost_factors']:
                            score *= config['boost_factors']['major_overview']
                        
                        # Force high priority for semester queries
                        if is_semester_query:
                            score *= 2.0  # Extra boost for semester queries
                        
                        results.append({
                            'content': item['content'],
                            'subject_code': item['subject_code'],
                            'type': item['type'],
                            'score': score,
                            'metadata': item.get('metadata', {}),
                            'search_method': 'forced_major_overview'
                        })
                        
                        print(f"FORCED major_overview with score: {score:.4f}")
        
        # STEP 2: Subject-specific search (if subjects detected)
        if intent.target_subjects:
            logger.info(f"STEP 2: Subject-specific search")
            logger.info(f"  Raw target subjects: {intent.target_subjects}")
            
            # Enhanced subject code resolution - handle partial codes
            resolved_subjects = []
            
            for subject_code in intent.target_subjects:
                # Check if it's a full subject code (has numbers)
                if re.match(r'[A-Z]{2,4}\d{3}[a-z]*', subject_code):
                    resolved_subjects.append(subject_code)
                # Check if it's a combo code (like AI17_COM2.1)
                elif re.match(r'[A-Z]{2,4}\d{2}_[A-Z]{3}\d+(?:\.\d+)?', subject_code):
                    # Convert to internal combo format
                    combo_subject = f"COMBO_{subject_code}"
                    resolved_subjects.append(combo_subject)
                    print(f"Resolved combo '{subject_code}' to: {combo_subject}")
                else:
                    # Partial subject code - find matches in data
                    partial_matches = []
                    for item in self.data:
                        item_subject = item.get('subject_code', '')
                        if subject_code.upper() in item_subject.upper():
                            partial_matches.append(item_subject)
                    
                    # Add unique matches
                    unique_matches = list(set(partial_matches))
                    resolved_subjects.extend(unique_matches)
                    
                    if unique_matches:
                        print(f"Resolved '{subject_code}' to: {unique_matches}")
            
            logger.info(f"  Resolved subjects: {resolved_subjects}")
            
            if resolved_subjects:
                subject_results = self._search_by_subject(resolved_subjects, config)
                logger.info(f"  Subject search returned: {len(subject_results)} results")
                results.extend(subject_results)
            else:
                logger.info(f"  No subjects could be resolved")
        
        # STEP 3: Content type search
        logger.info(f"STEP 3: Content type search")
        content_type_results = self._search_by_content_type(query, config)
        logger.info(f"  Content type search returned: {len(content_type_results)} results")
        results.extend(content_type_results)
        
        # STEP 4: General semantic search as fallback
        if len(results) < config['max_results']:
            logger.info(f"STEP 4: General semantic search (fallback)")
            remaining_slots = config['max_results'] - len(results)
            logger.info(f"  Need {remaining_slots} more results")
            fallback_config = config.copy()
            fallback_config['max_results'] = remaining_slots
            
            semantic_results = self._semantic_search(query, fallback_config)
            logger.info(f"  Semantic search returned: {len(semantic_results)} results")
            results.extend(semantic_results)
        else:
            logger.info(f"STEP 4: Skipped (already have {len(results)} results)")
        
        # STEP 5: Remove duplicates
        logger.info(f"STEP 5: Remove duplicates")
        original_count = len(results)
        results = self._deduplicate_results(results)
        logger.info(f"  Before: {original_count} results, After: {len(results)} results")
        logger.info(f"  Removed {original_count - len(results)} duplicates")
        
        # STEP 5.5: SMART PATTERN FILTERING - Add all matching courses if pattern detected
        if config.get('smart_filter_semester') and config.get('smart_filter_suffix'):
            target_semester = config['smart_filter_semester']
            target_suffix = config['smart_filter_suffix']
            print(f"STRATEGY PATTERN FILTER: semester={target_semester} + suffix='{target_suffix}'")
            
            # Find ALL courses matching the pattern in the data
            pattern_matches = []
            for item in self.data:
                if item.get('type') == 'general_info':  # Only general_info for courses
                    metadata = item.get('metadata', {})
                    subject_code = item.get('subject_code', '')
                    
                    is_target_semester = metadata.get('semester_from_curriculum') == target_semester
                    is_coursera = metadata.get('course_type_guess', '').startswith('coursera')
                    has_target_suffix = subject_code.endswith(target_suffix) and len(subject_code) > 1
                    
                    if is_target_semester and is_coursera and has_target_suffix:
                        # Check if already in results
                        already_included = any(r.get('subject_code') == subject_code for r in results)
                        if not already_included:
                            pattern_matches.append({
                                'content': item['content'],
                                'subject_code': item['subject_code'],
                                'type': item['type'],
                                'score': 30.0,  # High score for pattern match
                                'metadata': item.get('metadata', {}),
                                'search_method': 'strategy_pattern_matched'
                            })
                            print(f"STRATEGY PATTERN MATCHED & ADDED: {subject_code} (semester {target_semester} + coursera + đuôi '{target_suffix}')")
            
            # Add pattern matches to results
            results.extend(pattern_matches)
        
        # STEP 6: Advanced ranking
        logger.info(f"STEP 6: Advanced ranking")
        pre_ranking_scores = [r.get('score', 0) for r in results[:5]]
        logger.info(f"  Pre-ranking top 5 scores: {[round(s, 2) for s in pre_ranking_scores]}")
        
        results = self._rank_results(results, query, intent)
        
        post_ranking_scores = [r.get('final_score', r.get('score', 0)) for r in results[:5]]
        logger.info(f"  Post-ranking top 5 scores: {[round(s, 2) for s in post_ranking_scores]}")
        
        search_time = time.time() - search_start_time
        logger.info(f"SEARCH STRATEGY: Hoàn thành trong {search_time:.2f}s")
        logger.info(f"  Final results: {len(results)} kết quả")
        logger.info(f"  Top 5 scores: {[round(r.get('final_score', r.get('score', 0)), 2) for r in results[:5]]}")
        
        # Log search method distribution
        method_counts = {}
        for r in results:
            method = r.get('search_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        logger.info(f"  Search methods used: {method_counts}")
        
        # Log content type distribution
        type_counts = {}
        for r in results:
            content_type = r.get('type', 'unknown')
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        logger.info(f"  Content types found: {type_counts}")
        
        final_results = results[:config['max_results']]
        logger.info(f"  Returning top {len(final_results)} results")
        
        return final_results
    
    def hybrid_graph_query(self, question: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Hybrid GraphRAG query: Kết hợp Vector Search và Graph Traversal
        Implements the GraphRAG architecture from the research paper
        """
        if not self.graph_enabled:
            logger.info("Graph không enabled - fallback to regular query")
            return self.query(question, max_results)
        
        start_time = time.time()
        logger.info(f"========== HYBRID GRAPHRAG QUERY ==========")
        logger.info(f"USER QUERY: '{question}'")
        
        try:
            # STEP 1: Regular vector search for semantic similarity
            logger.info("STEP 1: Vector Search for semantic similarity...")
            vector_results = self.query(question, max_results)
            vector_search_results = vector_results.get('search_results', [])
            
            # STEP 2: Extract entities from query for graph traversal
            logger.info("STEP 2: Entity extraction for graph traversal...")
            extracted_entities = self._extract_entities_from_query(question)
            
            # STEP 3: Graph traversal for relationship discovery
            graph_results = []
            if extracted_entities and hasattr(self, 'graph_entities'):
                logger.info("STEP 3: Graph traversal...")
                graph_results = self._perform_graph_traversal(extracted_entities, question)
            
            # STEP 4: Hybrid result integration
            logger.info("STEP 4: Integrating vector and graph results...")
            integrated_results = self._integrate_vector_graph_results(
                vector_search_results, graph_results, question
            )
            
            # STEP 5: Generate enhanced answer with both semantic and relational context
            logger.info("STEP 5: Generating enhanced answer...")
            enhanced_context = self._prepare_hybrid_context(integrated_results)
            enhanced_answer = self._generate_hybrid_response(question, enhanced_context, vector_results.get('answer', ''))
            
            processing_time = time.time() - start_time
            
            result = {
                'answer': enhanced_answer,
                'search_results': integrated_results,
                'metadata': {
                    'query_type': 'hybrid_graphrag',
                    'vector_results_count': len(vector_search_results),
                    'graph_results_count': len(graph_results),
                    'total_results': len(integrated_results),
                    'processing_time': processing_time,
                    'graph_enabled': True,
                    'entities_extracted': extracted_entities
                },
                'graph_info': {
                    'entities_found': extracted_entities,
                    'graph_traversal_performed': len(graph_results) > 0,
                    'relationship_paths': len(graph_results)
                }
            }
            
            logger.info(f"HYBRID GraphRAG SUMMARY:")
            logger.info(f"  - Vector results: {len(vector_search_results)}")
            logger.info(f"  - Graph results: {len(graph_results)}")
            logger.info(f"  - Integrated results: {len(integrated_results)}")
            logger.info(f"  - Entities extracted: {extracted_entities}")
            logger.info(f"  - Processing time: {processing_time:.2f}s")
            logger.info(f"========== HYBRID GRAPHRAG COMPLETED ==========")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Lỗi hybrid graph query: {e}")
            # Fallback to regular vector search
            logger.info("Fallback to regular vector search...")
            return self.query(question, max_results)
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract course codes và entities từ user query"""
        entities = []
        
        # Extract course codes (e.g., CSI106, MAD101, etc.)
        course_pattern = r'\b([A-Z]{2,4}\d{3}[a-z]*)\b'
        course_matches = re.findall(course_pattern, query, re.IGNORECASE)
        entities.extend([code.upper() for code in course_matches])
        
        # Extract semester numbers
        semester_pattern = r'kì\s*(\d+)|ky\s*(\d+)|semester\s*(\d+)'
        semester_matches = re.findall(semester_pattern, query, re.IGNORECASE)
        for match in semester_matches:
            semester_num = next((num for num in match if num), None)
            if semester_num:
                entities.append(f"Semester_{semester_num}")
        
        # Extract combo patterns
        if 'combo' in query.lower() or 'chuyên ngành' in query.lower():
            entities.append("COMBO_ENTITY")
        
        logger.info(f"Extracted entities from query: {entities}")
        return entities
    
    def _perform_graph_traversal(self, entities: List[str], query: str) -> List[Dict[str, Any]]:
        """Perform graph traversal to find relationships"""
        if not hasattr(self, 'graph_entities'):
            return []
        
        graph_results = []
        nodes = self.graph_entities.get('nodes', [])
        relationships = self.graph_entities.get('relationships', [])
        
        logger.info(f"Graph traversal với {len(entities)} entities...")
        
        # Simple graph traversal - find related nodes
        for entity in entities:
            # Find direct matches in nodes
            matching_nodes = [node for node in nodes if node.id == entity]
            
            # Find relationships involving this entity
            related_relationships = [
                rel for rel in relationships 
                if rel.source_id == entity or rel.target_id == entity
            ]
            
            # Build graph results
            for rel in related_relationships:
                # Find the related course/entity
                related_entity_id = rel.target_id if rel.source_id == entity else rel.source_id
                related_node = next((node for node in nodes if node.id == related_entity_id), None)
                
                if related_node:
                    graph_result = {
                        'content': f"Graph relationship: {entity} {rel.type} {related_entity_id}",
                        'subject_code': related_entity_id,
                        'type': 'graph_relationship',
                        'score': 5.0,  # High score for direct relationships
                        'metadata': {
                            'relationship_type': rel.type,
                            'source_entity': entity,
                            'target_entity': related_entity_id,
                            'graph_traversal': True,
                            'node_properties': related_node.properties
                        },
                        'search_method': 'graph_traversal'
                    }
                    graph_results.append(graph_result)
        
        logger.info(f"Graph traversal found {len(graph_results)} relationship results")
        return graph_results
    
    def _integrate_vector_graph_results(self, vector_results: List[Dict], graph_results: List[Dict], query: str) -> List[Dict]:
        """Integrate vector search results with graph traversal results"""
        integrated = []
        
        # Add vector results with marking
        for result in vector_results:
            result['result_source'] = 'vector_search'
            integrated.append(result)
        
        # Add graph results with marking and boost scores
        for result in graph_results:
            result['result_source'] = 'graph_traversal'
            # Boost graph result scores since they represent explicit relationships
            if 'score' in result:
                result['score'] = result['score'] * 1.5  # Boost graph results
            integrated.append(result)
        
        # Remove duplicates based on subject_code and content similarity
        unique_results = []
        seen_combinations = set()
        
        for result in integrated:
            key = (result.get('subject_code', ''), result.get('type', ''))
            if key not in seen_combinations:
                seen_combinations.add(key)
                unique_results.append(result)
        
        # Sort by score (descending)
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"Integrated results: {len(vector_results)} vector + {len(graph_results)} graph = {len(unique_results)} unique")
        return unique_results
    
    def _prepare_hybrid_context(self, integrated_results: List[Dict]) -> str:
        """Prepare context từ integrated vector + graph results"""
        if not integrated_results:
            return "Không tìm thấy thông tin liên quan."
        
        context = "=== THÔNG TIN TÌM KIẾM ===\n\n"
        
        # Group by result source
        vector_results = [r for r in integrated_results if r.get('result_source') == 'vector_search']
        graph_results = [r for r in integrated_results if r.get('result_source') == 'graph_traversal']
        
        # Add vector search results
        if vector_results:
            context += "📚 THÔNG TIN NỘI DUNG:\n"
            for i, result in enumerate(vector_results[:5], 1):  # Top 5 vector results
                context += f"{i}. {result.get('content', '')}\n\n"
        
        # Add graph relationship results
        if graph_results:
            context += "🔗 MỐI QUAN HỆ LIÊN KẾT:\n"
            for i, result in enumerate(graph_results[:3], 1):  # Top 3 graph results
                rel_type = result.get('metadata', {}).get('relationship_type', 'RELATED')
                source = result.get('metadata', {}).get('source_entity', '')
                target = result.get('metadata', {}).get('target_entity', '')
                context += f"{i}. {source} --[{rel_type}]--> {target}\n"
                context += f"   Chi tiết: {result.get('content', '')}\n\n"
        
        return context
    
    def _generate_hybrid_response(self, question: str, hybrid_context: str, original_answer: str) -> str:
        """Generate enhanced answer sử dụng cả vector và graph context"""
        try:
            prompt = f"""Bạn là AI Assistant thông minh của FPTU, chuyên phân tích thông tin giáo dục với khả năng hiểu mối quan hệ phức tạp.

NGỮ CẢNH THÔNG TIN (Vector Search + Graph Relationships):
{hybrid_context}

CÂU HỎI: {question}

CÂU TRẢ LỜI GỐC (Vector-only): {original_answer}

HƯỚNG DẪN TRẢ LỜI ENHANCED:
1. Kết hợp thông tin từ cả nội dung (vector search) và mối quan hệ (graph traversal)
2. Ưu tiên thông tin có mối quan hệ rõ ràng từ graph analysis
3. Giải thích các mối liên kết và phụ thuộc nếu có
4. Đưa ra câu trả lời toàn diện và có cấu trúc
5. Không sử dụng biểu tượng cảm xúc

Trả lời bằng tiếng Việt một cách tự nhiên và chuyên nghiệp:"""

            enhanced_answer = self._call_gemini_with_rotation(prompt)
            return enhanced_answer
            
        except Exception as e:
            logger.error(f"Lỗi generate hybrid response: {e}")
            return original_answer  # Fallback to original answer
    
    def _get_search_config(self, intent: QueryIntent, query_lower: str) -> Dict[str, Any]:
        """Get search configuration based on query intent and content"""
        
        # Base configuration
        config = {
            'max_results': 5,
            'content_types': ['general_info', 'learning_outcomes_summary'],
            'boost_factors': {
                'general_info': 2.0,
                'learning_outcomes_summary': 1.5
            }
        }
        
        # Adjust based on query type
        if intent.query_type == 'listing':
            config['max_results'] = 15
            config['content_types'] = ['major_overview', 'combo_specialization', 'general_info', 'learning_outcomes_summary']
            config['boost_factors'] = {
                'major_overview': 10.0,  # Highest priority for listing queries
                'combo_specialization': 8.0,  # High priority for combo queries
                'general_info': 3.0,
                'learning_outcomes_summary': 2.0
            }
        
        elif intent.query_type == 'factual':
            config['content_types'] = ['combo_specialization', 'general_info', 'learning_outcomes_summary', 'materials', 'assessments']
            config['boost_factors'] = {
                'combo_specialization': 10.0,  # High priority for combo queries
                'general_info': 3.0,
                'learning_outcomes_summary': 2.0,
                'materials': 1.8,
                'assessments': 1.5
            }
        
        elif intent.query_type == 'comparative':
            config['max_results'] = 20
            config['content_types'] = ['major_overview', 'general_info', 'learning_outcomes_summary']
            config['boost_factors'] = {
                'major_overview': 5.0,
                'general_info': 3.0,
                'learning_outcomes_summary': 2.0
            }
        
        elif intent.query_type == 'analytical':
            config['max_results'] = 25
            config['content_types'] = ['major_overview', 'general_info', 'learning_outcomes_summary', 'materials', 'assessments', 'schedule']
            config['boost_factors'] = {
                'major_overview': 6.0,
                'general_info': 4.0,
                'learning_outcomes_summary': 3.0,
                'materials': 2.0,
                'assessments': 2.0,
                'schedule': 1.5
            }
        
        # Special cases based on query content
        
        # Special Features Detection - Bonus Paper, Special Assessment, etc.
        special_features_keywords = [
            # Bonus paper keywords
            'điểm thưởng', 'diem thuong', 'bonus', 'paper', 'bài báo', 'bai bao', 'scopus', 'isi',
            'nghiên cứu', 'nghien cuu', 'research', 'publication', 'xuất bản', 'xuat ban',
            # Project-based keywords
            'capstone', 'dự án', 'du an', 'project', 'thuyết trình', 'thuyet trinh', 'presentation',
            # MOOC keywords
            'mooc', 'coursera', 'certification', 'chứng chỉ', 'chung chi', 'online',
            # Language requirements
            'tiếng nhật', 'tieng nhat', 'tiếng hàn', 'tieng han', 'japanese', 'korean',
            # Special assessment
            'đặc biệt', 'dac biet', 'special', 'unique', 'riêng', 'rieng'
        ]
        
        if any(keyword in query_lower for keyword in special_features_keywords):
            # Add special_features to content types with high priority
            if 'special_features' not in config['content_types']:
                config['content_types'].insert(0, 'special_features')
            config['boost_factors']['special_features'] = 20.0  # Very high priority
            config['max_results'] = max(config['max_results'], 10)
            config['include_special_features'] = True
        
        # Combo/specialization queries get highest priority
        if any(keyword in query_lower for keyword in ['combo', 'chuyên ngành', 'chuyen nganh', 'specialization', 'track', 'hẹp', 'hep']):
            config['content_types'].insert(0, 'combo_specialization')
            config['boost_factors']['combo_specialization'] = 15.0
            config['max_results'] = max(config['max_results'], 10)
            
            # Special handling for listing all combos
            if any(list_term in query_lower for list_term in ['các', 'cac', 'tất cả', 'tat ca', 'all', 'list']):
                config['force_all_combos'] = True
        
        if 'ngành' in query_lower:
            config['content_types'].insert(0, 'major_overview')
            config['boost_factors']['major_overview'] = 8.0
            config['max_results'] = max(config['max_results'], 20)
            
        # Semester/term specific queries - IMPORTANT FIX
        if any(term in query_lower for term in ['kỳ', 'ky', 'kì', 'ki', 'semester', 'học kỳ', 'hoc ky']):
            config['content_types'].insert(0, 'major_overview')
            config['boost_factors']['major_overview'] = 15.0  # Very high priority for semester queries
            config['max_results'] = max(config['max_results'], 15)
            
        if any(term in query_lower for term in ['liệt kê', 'danh sách', 'list', 'tất cả', 'tat ca']):
            config['content_types'].insert(0, 'major_overview')
            config['boost_factors']['major_overview'] = 12.0
            config['max_results'] = max(config['max_results'], 15)
        
        if any(term in query_lower for term in ['tài liệu', 'materials', 'sách', 'book']):
            config['content_types'].extend(['materials'])
            config['boost_factors']['materials'] = 4.0
        
        if any(term in query_lower for term in ['đánh giá', 'assessment', 'kiểm tra', 'bài tập']):
            config['content_types'].extend(['assessments'])
            config['boost_factors']['assessments'] = 4.0
        
        if any(term in query_lower for term in ['lịch học', 'schedule', 'tuần', 'week']):
            config['content_types'].extend(['schedule'])
            config['boost_factors']['schedule'] = 4.0
        
        # Coursera-specific queries - CRITICAL FIX for missing DWP301c
        if any(term in query_lower for term in ['coursera', 'mooc', 'online course']):
            # Add special handling to prioritize Coursera courses in FAISS search
            config['coursera_boost'] = True  # Flag to boost coursera courses in semantic search
            config['boost_factors']['general_info'] = 5.0  # Higher boost for coursera course info
            config['max_results'] = max(config['max_results'], 15)
            
            # PATTERN-BASED FILTERING: If both coursera + semester mentioned
            semester_match = None
            for i in range(1, 9):
                if any(sem_term in query_lower for sem_term in [f'kỳ {i}', f'ky {i}', f'kì {i}', f'ki {i}', f'semester {i}']):
                    semester_match = i
                    break
            
            if semester_match:
                config['force_semester_coursera_filter'] = semester_match
                print(f"SMART FILTER: Forcing semester {semester_match} + Coursera courses (đuôi 'c')")
                config['max_results'] = max(config['max_results'], 20)
                
                # SMART FILTERING: No hardcode, just filter by pattern
                config['smart_filter_semester'] = semester_match
                config['smart_filter_suffix'] = 'c'  # Filter courses ending with 'c'
                print(f"SMART PATTERN: semester={semester_match} + suffix='c' + coursera_type")
        
        # Student queries - SMART PATTERN RECOGNITION
        if any(term in query_lower for term in ['sinh viên', 'sinh vien', 'student', 'học sinh', 'hoc sinh', 'danh sách sinh viên', 'mã sinh viên', 'ma sinh vien']):
            config['content_types'].extend(['student_overview', 'student_detail'])
            config['boost_factors']['student_overview'] = 20.0  # Higher than major_overview boost
            config['boost_factors']['student_detail'] = 15.0   # High priority for student details
            config['max_results'] = max(config['max_results'], 15)
            
            # PATTERN DETECTION: Student ID format (DE + digits)
            import re
            student_id_pattern = re.search(r'[Dd][Ee]\d{6}', query_lower)
            if student_id_pattern:
                config['force_specific_student'] = student_id_pattern.group().upper()
                print(f"SMART FILTER: Targeting specific student {config['force_specific_student']}")
                config['boost_factors']['student_detail'] = 50.0  # MASSIVE boost for exact student match
            
            # For general student queries, check if asking for list/overview
            elif any(list_term in query_lower for list_term in ['danh sách', 'list', 'tất cả', 'tat ca']):
                config['force_student_overview'] = True
                print(f"SMART FILTER: Prioritizing student overview/listing")
                config['boost_factors']['student_overview'] = 30.0
            
            # Legacy: For specific student queries (with roll numbers), prioritize detail
            elif any(code in query_lower for code in ['de', 'DE']) or any(char.isdigit() for char in query_lower):
                config['boost_factors']['student_detail'] = 25.0  # Highest priority for specific students
                config['boost_factors']['student_overview'] = 10.0
        
        # Remove duplicates while preserving order
        config['content_types'] = list(dict.fromkeys(config['content_types']))
        
        return config
    
    def _search_by_subject(self, subject_codes: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search specifically by subject codes with enhanced fallback"""
        results = []
        
        for subject_code in subject_codes:
            # First, try exact match
            subject_items = [item for item in self.data if item['subject_code'] == subject_code]
            print(f"Found {len(subject_items)} items for exact match '{subject_code}'")
            
            # If no exact match and this looks like a full subject code, try partial match
            if len(subject_items) == 0 and len(subject_code) > 4:
                # Extract prefix for fallback search (e.g., "DPL" from "DPL302m")
                prefix_match = re.match(r'([A-Z]{2,4})', subject_code)
                if prefix_match:
                    prefix = prefix_match.group(1)
                    print(f"No exact match for '{subject_code}', trying partial match with prefix '{prefix}'")
                    
                    # Search for items containing the prefix or mentioning the full code
                    partial_items = []
                    for item in self.data:
                        item_content = item.get('content', '').upper()
                        item_subject = item.get('subject_code', '').upper()
                        
                        # Check if item mentions the subject code in content or has similar prefix
                        if (subject_code.upper() in item_content or 
                            (item_subject.startswith(prefix) and item_subject != subject_code.upper()) or
                            (prefix in item_content and 'TIÊN QUYẾT' in item_content.upper()) or
                            (prefix in item_content and 'PREREQUISITE' in item_content.upper())):
                            partial_items.append(item)
                    
                    subject_items = partial_items
                    print(f"Found {len(subject_items)} items for partial/content match")
            
            # For subject-specific queries, we want comprehensive info
            # Override config to include all relevant content types
            if len(subject_items) > 0:
                # Check if this is a combo item
                is_combo = subject_items[0].get('type') == 'combo_specialization'
                
                if is_combo:
                    # For combo queries, prioritize combo content
                    relevant_content_types = ['combo_specialization']
                else:
                    relevant_content_types = ['general_info', 'learning_outcomes_summary', 'materials', 'assessments', 'schedule']
                
                # Filter by available content types in config, but prioritize comprehensive coverage
                for content_type in relevant_content_types:
                    filtered_items = [
                        item for item in subject_items 
                        if item['type'] == content_type
                    ]
                    
                    for item in filtered_items:
                        score = 2.0  # Base high score for exact subject match
                        
                        # Lower score if this was a fallback search
                        if item['subject_code'] != subject_code:
                            score *= 0.8  # Slightly lower score for partial matches
                        
                        # Apply content type priorities
                        if content_type == 'combo_specialization':
                            score *= 5.0  # Highest priority for combo content
                        elif content_type == 'general_info':
                            score *= 3.0  # High priority for general info
                        elif content_type == 'learning_outcomes_summary':
                            score *= 2.0
                        elif content_type in ['materials', 'assessments']:
                            score *= 1.5
                        
                        # Apply boost factors from config if available
                        if content_type in config.get('boost_factors', {}):
                            score *= config['boost_factors'][content_type]
                        
                        search_method = 'subject_specific' if item['subject_code'] == subject_code else 'subject_partial_match'
                        
                        result_item = {
                            'content': item['content'],
                            'subject_code': item['subject_code'],
                            'type': item['type'],
                            'score': score,
                            'metadata': item.get('metadata', {}),
                            'search_method': search_method
                        }
                        
                        # Preserve special fields for special_features
                        if item['type'] == 'special_features':
                            result_item['has_bonus_paper'] = item.get('has_bonus_paper', False)
                            result_item['has_mooc'] = item.get('has_mooc', False)
                            result_item['has_special_assessment'] = item.get('has_special_assessment', False)
                        
                        results.append(result_item)
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        print(f"Subject search returning {len(results)} results (top types: {[r['type'] for r in results[:5]]})")
        return results
    
    def _search_by_content_type(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search by content type with semantic similarity"""
        results = []
        query_embedding = self.embedding_model.encode([query])
        
        # Filter data by content types
        filtered_data = [
            item for item in self.data 
            if item['type'] in config['content_types']
        ]
        
        if not filtered_data:
            return results
        
        # Get embeddings for filtered data
        filtered_indices = [i for i, item in enumerate(self.data) if item['type'] in config['content_types']]
        filtered_embeddings = self.embeddings[filtered_indices]
        
        # Build temporary index
        temp_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings.astype('float32'))
        
        # Search
        distances, indices = temp_index.search(query_embedding.astype('float32'), min(20, len(filtered_data)))
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(filtered_data):
                original_idx = filtered_indices[idx]
                item = self.data[original_idx]
                
                score = float(dist)
                # Apply boost factors
                if item['type'] in config['boost_factors']:
                    score *= config['boost_factors'][item['type']]
                
                result_item = {
                    'content': item['content'],
                    'subject_code': item['subject_code'],
                    'type': item['type'],
                    'score': score,
                    'metadata': item.get('metadata', {}),
                    'search_method': 'content_type_semantic'
                }
                
                # Preserve special fields for special_features
                if item['type'] == 'special_features':
                    result_item['has_bonus_paper'] = item.get('has_bonus_paper', False)
                    result_item['has_mooc'] = item.get('has_mooc', False)
                    result_item['has_special_assessment'] = item.get('has_special_assessment', False)
                
                results.append(result_item)
        
        return results
    
    def _semantic_search(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """General semantic search as fallback"""
        results = []
        query_embedding = self.embedding_model.encode([query])
        
        # Search in main index with more results if coursera query
        search_k = config['max_results'] * 10 if config.get('coursera_boost') else config['max_results']
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.data):
                item = self.data[idx]
                
                # Apply content type filter
                if item['type'] not in config['content_types']:
                    continue
                
                score = float(dist)
                
                # SMART SEMESTER + COURSERA FILTER: Force specific patterns
                if config.get('force_semester_coursera_filter'):
                    target_semester = config['force_semester_coursera_filter']
                    metadata = item.get('metadata', {})
                    subject_code = item.get('subject_code', '')
                    
                    # Check if this matches semester + coursera pattern
                    is_target_semester = metadata.get('semester_from_curriculum') == target_semester
                    is_coursera = metadata.get('course_type_guess', '').startswith('coursera')
                    has_c_suffix = subject_code.endswith('c') and len(subject_code) > 1
                    
                    if is_target_semester and is_coursera and has_c_suffix:
                        score *= 20.0  # MASSIVE boost for perfect pattern match
                        print(f"PERFECT PATTERN MATCH: {subject_code} (semester {target_semester} + coursera + đuôi 'c')")
                    elif is_target_semester and is_coursera:
                        score *= 10.0  # High boost for semester + coursera
                        print(f"SEMESTER + COURSERA MATCH: {subject_code}")
                    elif is_coursera and has_c_suffix:
                        score *= 8.0  # Good boost for coursera + suffix
                        print(f"COURSERA + SUFFIX MATCH: {subject_code}")
                
                # SMART STUDENT PATTERN FILTER
                elif config.get('force_specific_student'):
                    target_student = config['force_specific_student']
                    content = item.get('content', '')
                    if target_student in content:
                        score *= 100.0  # MASSIVE boost for exact student match
                        print(f"EXACT STUDENT MATCH: {target_student} found in content")
                
                elif config.get('force_student_overview'):
                    if item.get('type') == 'student_overview':
                        score *= 15.0  # High boost for student overview
                        print(f"STUDENT OVERVIEW BOOST applied")
                
                # COURSERA BOOST: Check if this is a Coursera course
                elif config.get('coursera_boost') and item.get('metadata', {}).get('course_type_guess', '').startswith('coursera'):
                    score *= 5.0  # 5x boost for Coursera courses when coursera query detected
                    print(f"COURSERA BOOST applied to {item.get('subject_code', 'unknown')}: {item.get('metadata', {}).get('course_type_guess', '')}")
                    
                    # Extra boost if also semester 5
                    if item.get('metadata', {}).get('semester_from_curriculum') == 5:
                        score *= 2.0  # Additional 2x boost for semester 5 Coursera courses
                        print(f"SEMESTER 5 + COURSERA BOOST applied to {item.get('subject_code', 'unknown')}")
                
                result_item = {
                    'content': item['content'],
                    'subject_code': item['subject_code'],
                    'type': item['type'],
                    'score': score,
                    'metadata': item.get('metadata', {}),
                    'search_method': 'general_semantic'
                }
                
                # Preserve special fields for special_features
                if item['type'] == 'special_features':
                    result_item['has_bonus_paper'] = item.get('has_bonus_paper', False)
                    result_item['has_mooc'] = item.get('has_mooc', False)
                    result_item['has_special_assessment'] = item.get('has_special_assessment', False)
                
                results.append(result_item)
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # SMART PATTERN FILTERING: Add all matching courses if pattern detected
        if config.get('smart_filter_semester') and config.get('smart_filter_suffix'):
            target_semester = config['smart_filter_semester']
            target_suffix = config['smart_filter_suffix']
            
            # Find ALL courses matching the pattern in the data
            pattern_matches = []
            for item in self.data:
                if item.get('type') == 'general_info':  # Only general_info for courses
                    metadata = item.get('metadata', {})
                    subject_code = item.get('subject_code', '')
                    
                    is_target_semester = metadata.get('semester_from_curriculum') == target_semester
                    is_coursera = metadata.get('course_type_guess', '').startswith('coursera')
                    has_target_suffix = subject_code.endswith(target_suffix) and len(subject_code) > 1
                    
                    if is_target_semester and is_coursera and has_target_suffix:
                        # Check if already in results
                        already_included = any(r.get('subject_code') == subject_code for r in results)
                        if not already_included:
                            pattern_matches.append({
                                'content': item['content'],
                                'subject_code': item['subject_code'],
                                'type': item['type'],
                                'score': 25.0,  # High score for pattern match
                                'metadata': item.get('metadata', {}),
                                'search_method': 'pattern_matched'
                            })
                            print(f"PATTERN MATCHED & ADDED: {subject_code} (semester {target_semester} + coursera + đuôi '{target_suffix}')")
            
            # Add pattern matches to results
            results.extend(pattern_matches)
            results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:config['max_results']]
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity"""
        if not results:
            return results
        
        unique_results = []
        seen_contents = set()
        
        for result in results:
            # Create content hash for deduplication
            content_hash = hash(result['content'][:200])  # Hash first 200 chars
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[Dict[str, Any]], query: str, intent: QueryIntent) -> List[Dict[str, Any]]:
        """Advanced ranking based on multiple factors"""
        
        for result in results:
            final_score = result['score']
            
            # 1. Search method boost
            if result['search_method'] == 'subject_specific':
                final_score *= 1.5
            elif result['search_method'] == 'content_type_semantic':
                final_score *= 1.2
            
            # 2. Keyword matching boost
            query_words = set(query.lower().split())
            content_words = set(result['content'].lower().split())
            keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
            final_score *= (1 + keyword_overlap * 0.3)
            
            # 3. Metadata relevance boost
            metadata = result.get('metadata', {})
            if 'search_keywords' in metadata:
                keyword_text = metadata['search_keywords'].lower()
                if any(word in keyword_text for word in query.lower().split()):
                    final_score *= 1.3
            
            # 4. Intent-specific boosts
            if intent.query_type == 'listing' and result['type'] == 'general_info':
                final_score *= 1.4
            elif intent.requires_summarization and result['type'].endswith('_summary'):
                final_score *= 1.3
            
            result['final_score'] = final_score
        
        # Sort by final score
        return sorted(results, key=lambda x: x['final_score'], reverse=True)

    def _prepare_context(self, results: List[Dict]) -> str:
        """Intelligent context preparation with compression and summarization"""
        if not results:
            return ""
        
        # 1. GROUP RESULTS BY TYPE AND SUBJECT
        grouped_results = self._group_results(results)
        
        # 2. APPLY CONTEXT COMPRESSION
        compressed_context = self._compress_context(grouped_results)
        
        # 3. ENSURE CONTEXT LENGTH LIMIT
        final_context = self._limit_context_length(compressed_context)
        
        return final_context
    
    def _group_results(self, results: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
        """Group results by subject code and content type"""
        grouped = {}
        
        for result in results:
            subject = result.get('subject_code', 'UNKNOWN')
            content_type = result.get('type', 'general')
            
            if subject not in grouped:
                grouped[subject] = {}
            if content_type not in grouped[subject]:
                grouped[subject][content_type] = []
            
            grouped[subject][content_type].append(result)
        
        return grouped
    
    def _compress_context(self, grouped_results: Dict[str, Dict[str, List[Dict]]]) -> str:
        """Compress context based on content type and importance"""
        context_parts = []
        
        # PRIORITY 1: Special Features with Bonus Paper
        special_features_results = []
        bonus_paper_results = []
        for subject_code, content_types in grouped_results.items():
            if 'special_features' in content_types:
                for result in content_types['special_features']:
                    special_features_results.append(result)
                    if result.get('has_bonus_paper', False):
                        bonus_paper_results.append(result)
        
        # If we have bonus paper results, show them prominently
        if bonus_paper_results:
            context_parts.append(f"\n** MON CO DIEM THUONG PAPER KHOA HOC **:")
            for bonus_result in bonus_paper_results:
                subject_code = bonus_result.get('subject_code', '')
                content = bonus_result.get('content', '')
                context_parts.append(f"\n** {subject_code} **: {content}")
            context_parts.append("")  # Empty line
        
        # PRIORITY 2: Check if we have combo specialization (highest priority for combo queries)
        combo_results = []
        for subject_code, content_types in grouped_results.items():
            if 'combo_specialization' in content_types:
                combo_results.extend(content_types['combo_specialization'])
        
        # If we have combo results, prioritize them
        if combo_results:
            context_parts.append(f"\n** COMBO CHUYEN NGANH HEP **:")
            for combo_result in combo_results:
                context_parts.append(f"\n{combo_result['content']}")
        
        # Check if we have major overview
        major_overview_results = []
        for subject_code, content_types in grouped_results.items():
            if 'major_overview' in content_types:
                major_overview_results.extend(content_types['major_overview'])
        
        # If we have major overview, prioritize it
        if major_overview_results:
            best_overview = max(major_overview_results, key=lambda x: x.get('final_score', x.get('score', 0)))
            context_parts.append(f"\n** MAJOR_OVERVIEW **:\n{best_overview['content']}")
            
            # For major queries, we might want to add some supporting general info
            general_info_count = 0
            for subject_code, content_types in grouped_results.items():
                if subject_code != 'MAJOR_OVERVIEW' and 'general_info' in content_types and general_info_count < 3:
                    results = content_types['general_info']
                    best_result = max(results, key=lambda x: x.get('final_score', x.get('score', 0)))
                    context_parts.append(f"\n** {subject_code} **:\n{self._format_general_info(best_result)}")
                    general_info_count += 1
        
        else:
            # Regular processing for non-major queries
            for subject_code, content_types in grouped_results.items():
                if subject_code == 'MAJOR_OVERVIEW':
                    continue
                    
                subject_context = f"\n** {subject_code} **:\n"
                
                # Prioritize content types
                priority_order = [
                    'general_info',
                    'learning_outcomes_summary', 
                    'learning_outcome_detail',
                    'assessments',
                    'materials',
                    'schedule'
                ]
                
                for content_type in priority_order:
                    if content_type in content_types:
                        results = content_types[content_type]
                        
                        if content_type == 'general_info':
                            # For general info, take the best one
                            best_result = max(results, key=lambda x: x.get('final_score', x.get('score', 0)))
                            subject_context += self._format_general_info(best_result)
                            
                        elif content_type == 'learning_outcomes_summary':
                            # Combine CLO summaries
                            best_result = max(results, key=lambda x: x.get('final_score', x.get('score', 0)))
                            subject_context += f"\n** CLO tom tat **: {best_result['content'].strip()}\n"
                            
                        elif content_type == 'learning_outcome_detail':
                            # Summarize individual CLOs
                            clo_details = [r['content'].strip() for r in results[:5]]  # Top 5 CLOs
                            if clo_details:
                                subject_context += f"\n** CLO chi tiet **:\n" + "\n".join(clo_details[:3]) + "\n"
                            
                        elif content_type == 'assessments':
                            best_result = max(results, key=lambda x: x.get('final_score', x.get('score', 0)))
                            subject_context += f"\n** Danh gia **: {best_result['content'].strip()}\n"
                            
                        elif content_type == 'materials':
                            best_result = max(results, key=lambda x: x.get('final_score', x.get('score', 0)))
                            subject_context += f"\n** Tai lieu **: {best_result['content'].strip()}\n"
                            
                        elif content_type == 'schedule':
                            best_result = max(results, key=lambda x: x.get('final_score', x.get('score', 0)))
                            subject_context += f"\n** Lich hoc **: {best_result['content'].strip()}\n"
                
                context_parts.append(subject_context)
        
        return "\n".join(context_parts)
    
    def _format_general_info(self, result: Dict) -> str:
        """Format general info in a structured way"""
        content = result['content'].strip()
        metadata = result.get('metadata', {})
        
        formatted = f"\n** Thong tin chung **:\n{content}"
        
        # Add metadata if available
        if metadata.get('credits'):
            formatted += f"\nTin chi: {metadata['credits']}"
        if metadata.get('semester'):
            formatted += f"\nKy hoc: {metadata['semester']}"
        if metadata.get('prerequisites'):
            formatted += f"\nTien quyet: {metadata['prerequisites']}"
        
        return formatted + "\n"
    
    def _limit_context_length(self, context: str, max_length: int = 8000) -> str:
        """Limit context length to prevent overwhelming the LLM"""
        if len(context) <= max_length:
            return context
        
        # Intelligent truncation - keep the most important parts
        lines = context.split('\n')
        important_lines = []
        current_length = 0
        
        # For major overview, prioritize keeping the full list
        if '** MAJOR_OVERVIEW **' in context:
            # Keep major overview section fully
            for line in lines:
                if current_length + len(line) > max_length:
                    break
                important_lines.append(line)
                current_length += len(line)
        else:
            # Priority: general_info > assessments > learning_outcomes > materials > schedule
            for line in lines:
                if current_length + len(line) > max_length:
                    break
                
                # Always keep subject headers and general info
                if ('** ' in line and ' **' in line):
                    important_lines.append(line)
                    current_length += len(line)
                elif current_length < max_length * 0.8:  # Use 80% for detailed content
                    important_lines.append(line)
                    current_length += len(line)
        
        if current_length >= max_length:
            important_lines.append("\n... (da rut gon de toi uu phan hoi)")
        
        return '\n'.join(important_lines)

    def _generate_response(self, question: str, context: str) -> str:
        """Tạo phản hồi sử dụng Gemini với context được cung cấp"""
        try:
            # ENHANCED: Phân tích câu hỏi để tạo prompt phù hợp
            question_lower = question.lower()
            
            # Detect question type for better prompt
            is_semester_question = any(term in question_lower for term in ['kỳ', 'ky', 'kì', 'ki', 'semester'])
            is_listing_question = any(term in question_lower for term in ['liệt kê', 'danh sách', 'list', 'tất cả', 'các môn'])
            is_specific_subject = bool(re.search(r'[A-Za-z]{2,4}\d{3}[a-zA-Z]*', question))
            is_followup_question = any(term in question_lower for term in ['thì sao', 'ra sao', 'như thế nào', 'còn', 'con'])
            
            # Detect special features question
            is_special_features_question = any(keyword in question_lower for keyword in [
                'điểm thưởng', 'diem thuong', 'bonus', 'paper', 'bài báo', 'bai bao', 'scopus', 'isi',
                'đặc biệt', 'dac biet', 'special', 'mooc', 'coursera'
            ])
            has_bonus_paper_in_context = '** MON CO DIEM THUONG PAPER KHOA HOC **' in context
            
            # Build enhanced prompt based on question type
            if is_special_features_question and has_bonus_paper_in_context:
                # Special features questions (bonus paper, MOOC, etc.)
                prompt = f"""Bạn là AI Assistant chuyên môn về thông tin học tập tại FPT University. Hãy trả lời câu hỏi dựa trên CHÍNH XÁC thông tin được cung cấp.

NGUYÊN TẮC QUAN TRỌNG CHO SPECIAL FEATURES:
- Trả lời CHÍNH XÁC dựa trên dữ liệu được cung cấp về các môn học có đặc điểm đặc biệt
- Khi có thông tin về điểm thưởng paper, hãy liệt kê rõ ràng các môn học và mức điểm thưởng
- Tạo bảng thông tin với các cột: Mã môn, Điểm thưởng ISI/Scopus, Điều kiện
- CHÚ Ý: Thông tin về bonus paper nằm trong phần "** MON CO DIEM THUONG PAPER KHOA HOC **"

DỮ LIỆU:
{context}

TÍNH NĂNG QUAN TRỌNG:
- Nếu câu hỏi hỏi về "môn có điểm thưởng paper", hãy tập trung vào phần "** MON CO DIEM THUONG PAPER KHOA HOC **"
- Trả lời bằng format bảng rõ ràng về mức điểm thưởng cho từng loại journal
- Bao gồm điều kiện và yêu cầu cho việc nhận điểm thưởng

CÂU HỎI: {question}"""
            
            elif is_semester_question and (is_listing_question or is_followup_question):
                # Semester-focused questions
                prompt = f"""Bạn là AI Assistant chuyên môn về thông tin học tập tại FPT University. Hãy trả lời câu hỏi dựa trên CHÍNH XÁC thông tin được cung cấp.

NGUYÊN TẮC QUAN TRỌNG:
- Sử dụng CHÍNH XÁC thông tin từ dữ liệu được cung cấp
- Khi trả lời về môn học trong một kỳ cụ thể, hãy liệt kê TẤT CẢ các môn của kỳ đó
- Tạo bảng thông tin rõ ràng với các cột: Mã môn, Tên môn, Số tín chỉ
- Đảm bảo thông tin về tín chỉ, kỳ học được hiển thị chính xác
- Tính tổng số tín chỉ của kỳ học
- Không tự tạo ra thông tin không có trong dữ liệu

ĐỊNH DẠNG TRẢ LỜI CHO CÂUHỎI VỀ KỲ HỌC:

**Kỳ [số] ngành AI tại FPT University gồm [số] môn học:**

| Mã môn | Tên môn học | Số tín chỉ |
|--------|-------------|------------|
| [mã]   | [tên]       | [tín chỉ]  |

**Tổng số tín chỉ:** [tổng] tín chỉ

**Lưu ý bổ sung:** [nếu có thông tin đặc biệt về môn nào]

DỮ LIỆU:
{context}

CÂUHỎI: {question}

TRẢ LỜI:"""

            elif is_specific_subject:
                # Subject-specific questions
                prompt = f"""Bạn là AI Assistant chuyên môn về thông tin học tập tại FPT University. Hãy trả lời câu hỏi về môn học cụ thể dựa trên thông tin được cung cấp.

NGUYÊN TẮC:
- Cung cấp thông tin CHI TIẾT và CHÍNH XÁC về môn học
- Bao gồm: mã môn, tên đầy đủ, số tín chỉ, kỳ học, mô tả, CLO nếu có
- Định dạng thông tin rõ ràng, dễ đọc
- Nếu có nhiều môn tương tự, so sánh và phân biệt

DỮ LIỆU:
{context}

CÂUHỎI: {question}

TRẢ LỜI:"""

            else:
                # General questions
                prompt = f"""Bạn là AI Assistant chuyên môn về thông tin học tập tại FPT University (FPTU). Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

HƯỚNG DẪN TRẢ LỜI:
1. Sử dụng thông tin chính xác từ dữ liệu được cung cấp
2. Trả lời bằng tiếng Việt, rõ ràng và chuyên nghiệp  
3. Cấu trúc thông tin logic, dễ hiểu
4. Nếu cần liệt kê nhiều môn học, sử dụng bảng markdown
5. Tính toán tổng số tín chỉ khi cần thiết
6. Thêm lưu ý hữu ích cho sinh viên

QUAN TRỌNG:
- KHÔNG sử dụng biểu tượng cảm xúc hay icon
- KHÔNG tự tạo thông tin không có trong dữ liệu
- Nếu thiếu thông tin, nêu rõ và gợi ý cách tìm hiểu thêm

DỮ LIỆU:
{context}

CÂUHỎI: {question}

TRẢ LỜI:"""
            
            # Generate response with enhanced context awareness and API rotation
            answer = self._call_gemini_with_rotation(prompt)
            
            # ENHANCED: Post-process to ensure quality
            # Ensure no emojis or icons (as per user rules)
            answer = re.sub(r'[🎯🚀✅❌📚📝💡⭐🔍📊🌟✨🎪🎨🎭🎪🔥💥🎉🎊🎈🎁🎀🎃🎄🎆🎇✨🌈⚡💎🌟]', '', answer)
            
            # Ensure proper formatting for tables if detected
            if '|' in answer and 'Mã môn' in answer:
                # This looks like a table - ensure proper markdown formatting
                lines = answer.split('\n')
                formatted_lines = []
                in_table = False
                
                for line in lines:
                    if '|' in line and ('Mã môn' in line or 'Tên môn' in line):
                        in_table = True
                        formatted_lines.append(line)
                        # Add separator line if not already present
                        if not any('---' in next_line for next_line in lines[lines.index(line)+1:lines.index(line)+3]):
                            formatted_lines.append('|--------|-------------|------------|')
                    elif '|' in line and in_table:
                        formatted_lines.append(line)
                    elif in_table and line.strip() == '':
                        in_table = False
                        formatted_lines.append(line)
                    else:
                        formatted_lines.append(line)
                
                answer = '\n'.join(formatted_lines)
            
            return answer
            
        except Exception as e:
            logger.error(f"Lỗi tạo phản hồi: {e}")
            return self._fallback_response(question, context)
    
    def _fallback_response(self, question: str, context: str) -> str:
        """Enhanced fallback response khi không thể gọi Gemini"""
        
        # ENHANCED: Try to extract useful information from context even without LLM
        question_lower = question.lower()
        
        # 1. COMBO/SPECIALIZATION QUERIES
        if any(term in question_lower for term in ['combo', 'chuyên ngành']):
            if any(term in question_lower for term in ['bao nhiêu', 'có mấy', 'số lượng']):
                # Extract combo information from context
                combo_count = context.count('COMBO_')
                combo_names = []
                if 'AI17_COM1' in context:
                    combo_names.append('AI17_COM1 (Data Science và Big Data Analytics)')
                if 'AI17_COM3' in context:
                    combo_names.append('AI17_COM3 (AI for Healthcare và Research)')
                if 'AI17_COM2.1' in context:
                    combo_names.append('AI17_COM2.1 (Text Mining và Search Engineering)')
                
                if combo_names:
                    response = f"Ngành AI tại FPT University có **{len(combo_names)} combo chuyên ngành hẹp**:\n\n"
                    for i, combo in enumerate(combo_names, 1):
                        response += f"{i}. {combo}\n"
                    response += "\nMỗi combo bao gồm các môn học chuyên sâu trong lĩnh vực đó."
                    return response
            
            elif any(term in question_lower for term in ['là gì', 'định nghĩa']):
                return """**Combo chuyên ngành hẹp** là các nhóm môn học chuyên sâu trong ngành AI tại FPT University.

Ngành AI có **3 combo chuyên ngành hẹp**:

1. **AI17_COM1**: Data Science và Big Data Analytics
   - Tập trung vào khoa học dữ liệu và phân tích dữ liệu lớn

2. **AI17_COM2.1**: Text Mining và Search Engineering  
   - Khai thác văn bản và công cụ tìm kiếm

3. **AI17_COM3**: AI for Healthcare và Research
   - Ứng dụng AI trong y tế và nghiên cứu khoa học

Mỗi combo giúp sinh viên chuyên sâu vào một lĩnh vực cụ thể của AI."""
        
        # 2. SEMESTER QUERIES
        if any(term in question_lower for term in ['kì', 'ky', 'kỳ']):
            # Extract semester information from context
            semester_info = {}
            lines = context.split('\n')
            for line in lines:
                if 'Ky ' in line or 'ký ' in line:
                    # Try to extract semester info
                    if ' - ' in line and ':' in line:
                        parts = line.split(' - ')
                        if len(parts) >= 2:
                            subject_part = parts[0].strip()
                            desc_part = parts[1].split('(')[0].strip()
                            # Extract semester number
                            ky_match = re.search(r'Ky (\d+)', line)
                            if ky_match:
                                semester = ky_match.group(1)
                                if semester not in semester_info:
                                    semester_info[semester] = []
                                semester_info[semester].append(f"- {subject_part}: {desc_part}")
            
            if semester_info:
                response = "**Thông tin môn học theo kì:**\n\n"
                for semester in sorted(semester_info.keys()):
                    if len(semester_info[semester]) > 0:
                        response += f"**Kì {semester}:**\n"
                        response += '\n'.join(semester_info[semester][:5])  # Limit to 5 subjects
                        if len(semester_info[semester]) > 5:
                            response += f"\n... và {len(semester_info[semester]) - 5} môn khác"
                        response += '\n\n'
                return response
        
        # 3. RELATIONSHIP QUERIES
        if 'liên quan' in question_lower and ('kì' in question_lower or 'ky' in question_lower):
            return """Để phân tích mối liên hệ giữa các môn học ở các kì khác nhau, cần xem xét:

**1. Môn tiên quyết (Prerequisites)**:
- Một số môn ở kì sau yêu cầu hoàn thành môn ở kì trước
- Ví dụ: Toán cao cấp (kì 1) → Xác suất thống kê (kì sau)

**2. Chuỗi kiến thức**:
- Môn cơ bản → Môn nâng cao → Môn chuyên sâu
- Ví dụ: Lập trình cơ bản → Cấu trúc dữ liệu → Thuật toán AI

**3. Nhóm môn cùng lĩnh vực**:
- Toán học: MAD101, MAE101, PRO192...
- Lập trình: PFP191, PRO192, OOP...
- AI chuyên sâu: AIG, AIE, AIH...

Để biết chi tiết mối liên hệ cụ thể, bạn có thể hỏi về từng cặp môn học."""
        
        # 4. EXTRACT SUBJECT CODES FROM CONTEXT
        subjects = set()
        for line in context.split('\n'):
            # Extract from various patterns
            if '** ' in line:
                match = re.search(r'\*\*([A-Z]{2,4}\d{3}[a-z]*)\*\*', line)
                if match:
                    subjects.add(match.group(1))
            # Extract from listing patterns
            subject_codes = re.findall(r'([A-Z]{2,4}\d{3}[a-z]*)', line)
            subjects.update(subject_codes)
        
        if subjects:
            subjects_list = sorted(list(subjects))
            return f"""Dựa trên dữ liệu tìm được, tôi có thông tin về **{len(subjects_list)} môn học**: {', '.join(subjects_list[:10])}{', ...' if len(subjects_list) > 10 else ''}.

**Câu hỏi của bạn**: {question}

Tuy nhiên, hệ thống gặp sự cố kỹ thuật khi tạo câu trả lời chi tiết. Bạn có thể:
- Hỏi cụ thể về một môn học: "CSI106 là môn gì?"
- Hỏi về kì học: "Kì 1 có những môn gì?"
- Thử lại sau khi hệ thống phục hồi"""
        
        # 5. DEFAULT FALLBACK
        return f"""Xin lỗi, hệ thống đang gặp sự cố kỹ thuật khi xử lý câu hỏi.

**Câu hỏi của bạn**: {question}

**Gợi ý các câu hỏi bạn có thể thử**:
- "Liệt kê các môn học kì 1"
- "CSI106 là môn gì?"
- "Có bao nhiêu combo chuyên ngành hẹp?"
- "Chuyên ngành hẹp là gì?"
- "Các môn có 3 tín chỉ"

**Hoặc thử lại sau khi hệ thống phục hồi.**"""

    def get_subject_overview(self, subject_code: str = None) -> Dict[str, Any]:
        """Lấy tổng quan về môn học"""
        if subject_code:
            # Find subject in data
            subject_items = [item for item in self.data if item.get('subject_code') == subject_code]
            if subject_items:
                general_info = next((item for item in subject_items if item.get('type') == 'general_info'), None)
                if general_info:
                    return {
                        'subject_code': subject_code,
                        'summary': general_info['content'],
                        'metadata': general_info.get('metadata', {})
                    }
            return {'error': f'Không tìm thấy môn {subject_code}'}
        else:
            # Trả về tất cả môn
            subjects = []
            seen_subjects = set()
            
            for item in self.data:
                if item.get('type') == 'general_info' and item.get('subject_code') not in seen_subjects:
                    subject_code = item.get('subject_code')
                    seen_subjects.add(subject_code)
                    metadata = item.get('metadata', {})
                    
                    subjects.append({
                        'subject_code': subject_code,
                        'name': metadata.get('course_name', metadata.get('title', subject_code)),
                        'credits': metadata.get('credits', 'N/A'),
                        'summary': item['content'][:200] + '...'
                    })
            
            return {'subjects': subjects, 'total': len(subjects)}

class QueryChain:
    """Xử lý chuỗi truy vấn đa cấp (multi-hop query)"""
    
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.max_hops = 3  # Giới hạn số lần truy vấn kép để tránh vòng lặp
        
        # Patterns để nhận diện thông tin có thể truy vấn tiếp
        self.prerequisite_patterns = [
            r'môn tiên quyết.*?([A-Z]{2,4}\d{3}[a-z]*)',
            r'prerequisite.*?([A-Z]{2,4}\d{3}[a-z]*)',
            r'cần học trước.*?([A-Z]{2,4}\d{3}[a-z]*)',
            r'phải hoàn thành.*?([A-Z]{2,4}\d{3}[a-z]*)',
        ]
        
        self.related_subject_patterns = [
            r'liên quan đến.*?([A-Z]{2,4}\d{3}[a-z]*)',
            r'kết hợp với.*?([A-Z]{2,4}\d{3}[a-z]*)',
            r'tiếp theo.*?([A-Z]{2,4}\d{3}[a-z]*)',
            r'nâng cao.*?([A-Z]{2,4}\d{3}[a-z]*)',
        ]
        
        self.detail_expansion_keywords = [
            'chi tiết hơn', 'thông tin đầy đủ', 'mô tả cụ thể', 'tài liệu',
            'giáo trình', 'syllabus', 'CLO', 'learning outcomes'
        ]

    def detect_followup_queries(self, answer: str, original_query: str) -> List[FollowupQuery]:
        """Phát hiện các truy vấn tiếp theo từ câu trả lời - Optimized to be less aggressive"""
        followup_queries = []
        answer_lower = answer.lower()
        original_query_lower = original_query.lower()
        
        # Early exit for simple queries that don't need followup
        simple_query_indicators = [
            'là gì', 'là môn gì', 'bao nhiêu tín chỉ', 'kỳ nào', 'kỳ mấy',
            'giảng viên là ai', 'học phí', 'đánh giá như thế nào'
        ]
        
        if any(indicator in original_query_lower for indicator in simple_query_indicators):
            # Only proceed if explicitly asking for prerequisites
            if not any(keyword in original_query_lower for keyword in ['tiên quyết', 'liên quan', 'chi tiết']):
                return []
        
        # Only detect prerequisites if explicitly mentioned in original query OR answer is very detailed
        if ('tiên quyết' in original_query_lower or 'và các môn' in original_query_lower or 
            len(answer.split()) > 200):  # Only for detailed answers
            
            # 1. Phát hiện môn tiên quyết (more selective)
            for pattern in self.prerequisite_patterns:
                matches = re.findall(pattern, answer, re.IGNORECASE)
                for subject_code in matches:
                    if (subject_code not in original_query and 
                        subject_code not in [m[0] for m in followup_queries if hasattr(m, 'query')]):
                        query = f"Thông tin chi tiết về môn {subject_code}"
                        followup_queries.append(FollowupQuery(
                            query=query,
                            confidence=0.9,
                            query_type='prerequisite',
                            source_info=f"Môn tiên quyết được nhắc đến trong câu trả lời"
                        ))
                        # Limit to 1 prerequisite query per answer
                        break
        
        # 2. Only detect related subjects if explicitly requested
        if any(keyword in original_query_lower for keyword in ['liên quan', 'kết hợp', 'tương tự']):
            for pattern in self.related_subject_patterns:
                matches = re.findall(pattern, answer, re.IGNORECASE)
                for subject_code in matches[:1]:  # Limit to 1
                    if subject_code not in original_query:
                        query = f"Thông tin về môn {subject_code}"
                        followup_queries.append(FollowupQuery(
                            query=query,
                            confidence=0.7,
                            query_type='related_subject',
                            source_info=f"Môn liên quan được nhắc đến"
                        ))
                        break
        
        # 3. Only expand details if explicitly requested
        if any(keyword in original_query_lower for keyword in ['chi tiết', 'đầy đủ', 'syllabus', 'mở rộng']):
            for keyword in self.detail_expansion_keywords:
                if keyword in answer_lower:
                    # Extract subject codes từ câu trả lời
                    subject_codes = re.findall(r'([A-Z]{2,4}\d{3}[a-z]*)', answer)
                    for subject_code in subject_codes[:1]:  # Only 1 expansion
                        if subject_code not in original_query:
                            query = f"Thông tin đầy đủ về {subject_code} bao gồm syllabus và CLO"
                            followup_queries.append(FollowupQuery(
                                query=query,
                                confidence=0.6,
                                query_type='detail_expansion',
                                source_info=f"Cần thông tin chi tiết hơn về {subject_code}"
                            ))
                            break
        
        # 4. Skip incomplete information detection for basic queries
        # Only proceed for complex/detailed queries
        if (len(answer.split()) > 150 and 
            any(phrase in answer_lower for phrase in [
                'không có thông tin đầy đủ', 'cần tìm thêm thông tin', 
                'thông tin chi tiết cần được tìm hiểu thêm'
            ])):
            # Extract subject codes từ câu trả lời
            subject_codes = re.findall(r'([A-Z]{2,4}\d{3}[a-z]*)', answer)
            for subject_code in subject_codes[:1]:  # Limit to 1
                if subject_code not in original_query:
                    query = f"Thông tin chi tiết về {subject_code}"
                    followup_queries.append(FollowupQuery(
                        query=query,
                        confidence=0.8,
                        query_type='detail_expansion',
                        source_info=f"Câu trả lời thiếu thông tin về {subject_code}"
                    ))
                    break
        
        # Sắp xếp theo confidence và loại bỏ trùng lặp
        unique_queries = {}
        for fq in followup_queries:
            if fq.query not in unique_queries or unique_queries[fq.query].confidence < fq.confidence:
                unique_queries[fq.query] = fq
        
        # Giới hạn số lượng followup queries - more restrictive
        sorted_queries = sorted(unique_queries.values(), key=lambda x: x.confidence, reverse=True)
        return sorted_queries[:2]  # Tối đa 2 followup queries (giảm từ 3)

    def execute_query_chain(self, original_query: str, enable_multihop: bool = True) -> QueryChainResult:
        """Thực hiện chuỗi truy vấn đa cấp"""
        execution_path = [f"Truy vấn gốc: {original_query}"]
        
        # Bước 1: Thực hiện truy vấn gốc
        original_result = self.rag_engine.query(original_query)
        original_answer = original_result['answer']
        
        if not enable_multihop:
            return QueryChainResult(
                original_query=original_query,
                original_answer=original_answer,
                followup_queries=[],
                followup_results=[],
                final_integrated_answer=original_answer,
                execution_path=execution_path
            )
        
        # Bước 2: Phát hiện truy vấn tiếp theo
        followup_queries = self.detect_followup_queries(original_answer, original_query)
        
        if not followup_queries:
            execution_path.append("Không phát hiện truy vấn tiếp theo")
            return QueryChainResult(
                original_query=original_query,
                original_answer=original_answer,
                followup_queries=[],
                followup_results=[],
                final_integrated_answer=original_answer,
                execution_path=execution_path
            )
        
        # Bước 3: Thực hiện các truy vấn tiếp theo
        followup_results = []
        for i, fq in enumerate(followup_queries):
            if i >= self.max_hops:
                break
                
            execution_path.append(f"Truy vấn tiếp theo {i+1}: {fq.query} (confidence: {fq.confidence:.2f})")
            
            try:
                result = self.rag_engine.query(fq.query)
                result['followup_query'] = fq
                followup_results.append(result)
                execution_path.append(f"  -> Hoàn thành truy vấn {i+1}")
            except Exception as e:
                execution_path.append(f"  -> Lỗi truy vấn {i+1}: {str(e)}")
                continue
        
        # Bước 4: Tích hợp kết quả
        final_answer = self._integrate_results(original_result, followup_results, original_query)
        execution_path.append("Tích hợp kết quả hoàn tất")
        
        return QueryChainResult(
            original_query=original_query,
            original_answer=original_answer,
            followup_queries=followup_queries,
            followup_results=followup_results,
            final_integrated_answer=final_answer,
            execution_path=execution_path
        )

    def _integrate_results(self, original_result: Dict, followup_results: List[Dict], original_query: str) -> str:
        """Tích hợp kết quả từ truy vấn gốc và các truy vấn tiếp theo"""
        if not followup_results:
            return original_result['answer']
        
        # Chuẩn bị context tích hợp
        integration_context = f"""
TRUY VẤN GỐC: {original_query}
THÔNG TIN SỐ 1 (Câu trả lời chính):
{original_result['answer']}

"""
        
        # Thêm thông tin từ các truy vấn tiếp theo
        for i, result in enumerate(followup_results):
            fq = result.get('followup_query')
            integration_context += f"""THÔNG TIN BỔ SUNG {i+2} (Từ truy vấn: {fq.query if fq else 'N/A'}):
{result['answer']}

"""
        
        # Prompt tích hợp thông minh
        integration_prompt = f"""
Dựa trên thông tin được cung cấp, hãy tạo ra một câu trả lời tích hợp và hoàn chỉnh cho truy vấn gốc.

QUY TẮC TÍCH HỢP:
1. Bắt đầu với câu trả lời chính cho truy vấn gốc
2. Bổ sung thông tin chi tiết từ các truy vấn tiếp theo một cách logic
3. Tránh lặp lại thông tin
4. Sắp xếp thông tin theo thứ tự quan trọng và logic
5. Giữ nguyên các thông tin quan trọng như mã môn học, tín chỉ, kỳ học
6. Nếu có thông tin mâu thuẫn, ưu tiên thông tin từ câu trả lời chính

THÔNG TIN ĐẦU VÀO:
{integration_context}

Hãy tạo ra một câu trả lời tích hợp, đầy đủ và có cấu trúc rõ ràng:
"""
        
        try:
            # Sử dụng Gemini để tích hợp với rotation
            if hasattr(self.rag_engine, '_call_gemini_with_rotation'):
                result_text = self.rag_engine._call_gemini_with_rotation(integration_prompt)
                return result_text
            else:
                # Fallback: direct call
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(integration_prompt)
                if response.text:
                    return response.text.strip()
                else:
                    return self._simple_integration(original_result['answer'], followup_results)
                
        except Exception as e:
            logger.error(f"Lỗi tích hợp kết quả với Gemini: {e}")
            return self._simple_integration(original_result['answer'], followup_results)

    def _simple_integration(self, original_answer: str, followup_results: List[Dict]) -> str:
        """Tích hợp đơn giản khi Gemini không khả dụng"""
        integrated = original_answer + "\n\n"
        
        for i, result in enumerate(followup_results):
            fq = result.get('followup_query')
            if fq:
                integrated += f"THÔNG TIN BỔ SUNG VỀ {fq.query.upper()}:\n"
                integrated += result['answer'] + "\n\n"
        
        return integrated.strip() 