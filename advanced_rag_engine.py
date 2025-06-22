"""
Advanced RAG Engine - Hệ thống RAG tiên tiến cho FPTU
Tích hợp các kỹ thuật tiên tiến: Hierarchical Indexing, Multi-stage Retrieval, Query Routing, Document Summarization, Multi-hop Query
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

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            'chi tiết', 'chi tiet', 'nội dung', 'noi dung'
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
        
        # Extract subject codes and combo codes for later use
        subject_codes = re.findall(r'[A-Z]{2,4}\d{3}[a-z]*', query)
        
        # ENHANCED: Extract combo codes (like AI17_COM1, AI17_COM2.1)
        combo_codes = re.findall(r'[A-Z]{2,4}\d{2}_[A-Z]{3}\d+(?:\.\d+)?', query)
        
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
    """Engine RAG tiên tiến với đầy đủ tính năng"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize the Advanced RAG Engine"""
        self.gemini_api_key = gemini_api_key
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.genai = None
        self.data = []
        self.embeddings = None
        self.index = None
        self.hierarchical_index = None
        self.query_router = QueryRouter()
        self.query_chain = None  # Sẽ được khởi tạo sau khi engine sẵn sàng
        self.is_initialized = False
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.genai = genai.GenerativeModel('gemini-1.5-flash')
        
        logger.info("Khởi tạo Advanced RAG Engine")

    def initialize(self, data_path: str):
        """Khởi tạo engine với dữ liệu từ file JSON"""
        logger.info(f"Đang khởi tạo Advanced RAG Engine với dữ liệu từ {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.data = self._process_data(raw_data)
        self._create_embeddings()
        self._build_index()
        
        # Initialize hierarchical index
        self.hierarchical_index = HierarchicalIndex(self.model)
        
        # Initialize query chain for multi-hop queries
        self.query_chain = QueryChain(self)
        
        self.is_initialized = True
        logger.info("Khởi tạo hoàn tất")

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
                
                # 4. ASSESSMENTS PROCESSING
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
        self.embeddings = self.model.encode(contents)

    def _build_index(self):
        logger.info("Đang xây dựng FAISS index...")
        embeddings_np = np.array(self.embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(embeddings_np.shape[1])
        self.index.add(embeddings_np)

    def query(self, question: str, max_results: int = 10) -> Dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("Engine chưa được khởi tạo")
        
        # Check for quick response first
        quick_response = self.query_router.check_quick_response(question)
        if quick_response:
            return {
                'question': question,
                'answer': quick_response,
                'search_results': [],
                'is_quick_response': True
            }
        
        expanded_query = self._expand_query(question)
        search_results = self._search_strategy(expanded_query, self.query_router.analyze_query(expanded_query))
        context = self._prepare_context(search_results)
        response = self._generate_response(question, context)
        
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
        if not self.is_initialized:
            raise RuntimeError("Engine chưa được khởi tạo")
        
        # Check for quick response first (no need for multi-hop)
        quick_response = self.query_router.check_quick_response(question)
        if quick_response:
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
            # Fallback to normal query if QueryChain not available
            normal_result = self.query(question, max_results)
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
        chain_result = self.query_chain.execute_query_chain(question, enable_multihop)
        
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
                query_embedding = self.model.encode([query])
                
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
                query_embedding = self.model.encode([query])
                
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
            
            if resolved_subjects:
                subject_results = self._search_by_subject(resolved_subjects, config)
                results.extend(subject_results)
        
        # STEP 3: Content type search
        content_type_results = self._search_by_content_type(query, config)
        results.extend(content_type_results)
        
        # STEP 4: General semantic search as fallback
        if len(results) < config['max_results']:
            remaining_slots = config['max_results'] - len(results)
            fallback_config = config.copy()
            fallback_config['max_results'] = remaining_slots
            
            semantic_results = self._semantic_search(query, fallback_config)
            results.extend(semantic_results)
        
        # STEP 5: Remove duplicates
        results = self._deduplicate_results(results)
        
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
        results = self._rank_results(results, query, intent)
        
        print(f"Final ranked results: {len(results)}")
        return results[:config['max_results']]
    
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
        """Search specifically by subject codes"""
        results = []
        
        for subject_code in subject_codes:
            subject_items = [item for item in self.data if item['subject_code'] == subject_code]
            print(f"Found {len(subject_items)} items for {subject_code}")
            
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
                        
                        results.append({
                            'content': item['content'],
                            'subject_code': item['subject_code'],
                            'type': item['type'],
                            'score': score,
                            'metadata': item.get('metadata', {}),
                            'search_method': 'subject_specific'
                        })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        print(f"Subject search returning {len(results)} results (top types: {[r['type'] for r in results[:5]]})")
        return results
    
    def _search_by_content_type(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search by content type with semantic similarity"""
        results = []
        query_embedding = self.model.encode([query])
        
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
                
                results.append({
                    'content': item['content'],
                    'subject_code': item['subject_code'],
                    'type': item['type'],
                    'score': score,
                    'metadata': item.get('metadata', {}),
                    'search_method': 'content_type_semantic'
                })
        
        return results
    
    def _semantic_search(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """General semantic search as fallback"""
        results = []
        query_embedding = self.model.encode([query])
        
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
                
                results.append({
                    'content': item['content'],
                    'subject_code': item['subject_code'],
                    'type': item['type'],
                    'score': score,
                    'metadata': item.get('metadata', {}),
                    'search_method': 'general_semantic'
                })
        
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
        
        # Check if we have combo specialization first (highest priority for combo queries)
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
        """Generate response using Gemini với enhanced prompting"""
        
        # Enhanced prompt construction based on question type
        question_lower = question.lower()
        
        # Detect specific query types for enhanced prompting
        is_coursera_query = any(term in question_lower for term in ['coursera', 'course', 'khóa học', 'trực tuyến'])
        is_listing_query = any(term in question_lower for term in ['liệt kê', 'danh sách', 'những môn', 'các môn', 'có gì'])
        is_semester_query = any(term in question_lower for term in ['kỳ', 'kì', 'ky', 'ki', 'semester'])
        
        # Base prompt
        base_prompt = f"""
Bạn là chuyên gia tư vấn học tập tại FPT University, chuyên về ngành AI.
Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

THÔNG TIN QUAN TRỌNG VỀ MÃ MÔN HỌC:
- Môn có đuôi 'c' (ví dụ: BDI302c, DWP301c): Môn học hoàn toàn trên Coursera, không cần lên lớp
- Môn có đuôi 'm' (ví dụ: DPL302m, AIH301m): Học trên lớp kết hợp với Coursera tự học
- Môn không có đuôi đặc biệt: Học trên lớp bình thường

"""

        # Enhanced prompting for specific query types
        if is_coursera_query and is_listing_query:
            enhanced_prompt = base_prompt + """
NHIỆM VỤ ĐặC BIỆT: Tìm và liệt kê TẤT CẢ các môn học Coursera (có đuôi 'c')
- Đọc kỹ toàn bộ thông tin được cung cấp
- Tìm tất cả môn học có mã kết thúc bằng 'c' 
- Liệt kê đầy đủ, không bỏ sót môn nào
- Giải thích rõ ràng rằng đây là các môn học hoàn toàn trên Coursera

"""
        elif is_listing_query and is_semester_query:
            enhanced_prompt = base_prompt + """
NHIỆM VỤ ĐặC BIỆT: Liệt kê ĐẦY ĐỦ TẤT CẢ các môn học trong kỳ
- ĐỌC KỸ VÀ DUYỆT TOÀN BỘ thông tin được cung cấp
- Tìm TẤT CẢ môn học thuộc kỳ được hỏi (ví dụ: "Ky 5", "Ky 6")
- KHÔNG ĐƯỢC BỎ SÓT bất kỳ môn học nào
- Liệt kê theo thứ tự: mã môn, tên môn đầy đủ, số tín chỉ
- Phân loại rõ ràng theo loại môn học:
  * Môn có đuôi 'c': Học hoàn toàn trên Coursera
  * Môn có đuôi 'm': Học trên lớp kết hợp với Coursera
  * Môn không có đuôi đặc biệt: Học trên lớp thường
- ĐẾMV SỐ LƯỢNG môn trong dữ liệu và đảm bảo liệt kê đủ số đó
- TUYỆT ĐỐI KHÔNG ĐƯỢC thiếu môn nào

"""
        else:
            enhanced_prompt = base_prompt + """
Hãy trả lời chính xác và đầy đủ dựa trên thông tin được cung cấp.

"""

        # Final prompt assembly
        prompt = enhanced_prompt + f"""
THÔNG TIN ĐƯỢC CUNG CẤP:
{context}

CÂU HỎI: {question}

Hãy trả lời một cách chính xác, đầy đủ và có cấu trúc rõ ràng. Nếu thông tin không đủ để trả lời, hãy nói rõ và gợi ý cách tìm thêm thông tin."""

        try:
            response = self.genai.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Lỗi khi generate response: {e}")
            return self._fallback_response(question, context)
    
    def _fallback_response(self, question: str, context: str) -> str:
        """Fallback response when main generation fails"""
        if not context:
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn. Bạn có thể thử hỏi về một môn học cụ thể bằng mã môn (ví dụ: CSI106)?"
        
        # Simple context summary
        subjects = set()
        for line in context.split('\n'):
            if '** ' in line:
                match = re.search(r'\*\*([A-Z]{2,4}\d{3}[a-z]*)\*\*', line)
                if match:
                    subjects.add(match.group(1))
        
        if subjects:
            return f"Tôi tìm thấy thông tin về {len(subjects)} môn học: {', '.join(sorted(subjects))}. Tuy nhiên, có lỗi khi xử lý câu hỏi. Bạn có thể hỏi cụ thể hơn về môn học nào?"
        
        return "Có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại với câu hỏi khác."

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
            # Sử dụng Gemini để tích hợp
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(integration_prompt)
            
            if response.text:
                return response.text.strip()
            else:
                # Fallback: ghép thông tin đơn giản
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