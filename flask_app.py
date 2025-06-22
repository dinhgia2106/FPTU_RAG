"""
Flask App - Giao diện Chat hiện đại cho FPTU RAG
"""

from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
from advanced_rag_engine import AdvancedRAGEngine
import uuid
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG Engine
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY không được tìm thấy trong .env file")

rag_engine = AdvancedRAGEngine(GEMINI_API_KEY)

# Initialize with data
try:
    if os.path.exists("Data/combined_data.json"):
        rag_engine.initialize("Data/combined_data.json")
        logger.info("Đã khởi tạo RAG engine với combined_data.json")
    elif os.path.exists("Data/reduced_data.json"):
        rag_engine.initialize("Data/reduced_data.json")
        logger.info("Đã khởi tạo RAG engine với reduced_data.json")
    else:
        logger.error("Không tìm thấy file dữ liệu")
except Exception as e:
    logger.error(f"Lỗi khởi tạo RAG engine: {e}")

@app.route('/')
def index():
    """Trang chính với giao diện chat"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint cho chat với hỗ trợ multi-hop query"""
    try:
        data = request.get_json()
        question = data.get('message', '').strip()
        enable_multihop = data.get('multihop', False)  # Mặc định TẮT multi-hop
        
        if not question:
            return jsonify({'error': 'Vui lòng nhập câu hỏi'}), 400
        
        # Check if multi-hop is supported
        if enable_multihop and hasattr(rag_engine, 'query_with_multihop'):
            # Use multi-hop query
            result = rag_engine.query_with_multihop(question, enable_multihop=True)
            
            # Handle multi-hop result format
            if isinstance(result, dict):
                answer = result.get('final_answer', result.get('original_answer', ''))
                search_results = result.get('search_results', [])
                is_quick = result.get('is_quick_response', False)
                
                response = {
                    'answer': answer,
                    'search_results': search_results,
                    'multihop_info': {
                        'has_followup': result.get('has_followup', False),
                        'followup_queries': result.get('followup_queries', []),
                        'execution_path': result.get('execution_path', [])
                    },
                    'metadata': {
                        'subjects_covered': len(set([r.get('subject_code', '') for r in search_results if isinstance(r, dict)])),
                        'query_type': 'quick' if is_quick else ('multihop' if result.get('has_followup', False) else 'single'),
                        'followup_count': len(result.get('followup_queries', [])),
                        'is_quick_response': is_quick
                    }
                }
            else:
                # Fallback format
                response = {
                    'answer': str(result),
                    'search_results': [],
                    'multihop_info': {'has_followup': False},
                    'metadata': {'subjects_covered': 0, 'query_type': 'single'}
                }
        else:
            # Use normal query as fallback
            result = rag_engine.query(question)
            
            # Handle normal result format
            if isinstance(result, dict) and 'answer' in result:
                answer = result['answer']
                search_results = result.get('search_results', [])
            else:
                answer = str(result)
                search_results = []
            
            response = {
                'answer': answer,
                'search_results': search_results,
                'multihop_info': {'has_followup': False},
                'metadata': {
                    'subjects_covered': len(set([r.get('subject_code', '') for r in search_results if isinstance(r, dict)])),
                    'query_type': 'single'
                }
            }
        
        logger.info(f"Processed query: {question[:50]}... (multihop: {enable_multihop})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Lỗi xử lý chat: {e}", exc_info=True)
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'}), 500

@app.route('/api/subjects')
def get_subjects():
    """API để lấy danh sách môn học"""
    try:
        # Check if rag_engine is initialized and has data
        if not hasattr(rag_engine, 'data') or not rag_engine.data:
            return jsonify({'error': 'Dữ liệu chưa được khởi tạo'}), 500
        
        # Tạo danh sách môn học từ data
        subjects = []
        subject_codes = set()
        
        for item in rag_engine.data:
            if isinstance(item, dict) and item.get('subject_code') not in subject_codes:
                subject_code = item.get('subject_code', 'UNKNOWN')
                subject_codes.add(subject_code)
                
                # Get metadata from item
                metadata = item.get('metadata', {})
                
                subjects.append({
                    'code': subject_code,
                    'name': metadata.get('course_name', metadata.get('title', subject_code)),
                    'credits': metadata.get('credits', 'N/A'),
                    'semester': metadata.get('semester', 'N/A')
                })
        
        # Sort by subject code
        subjects.sort(key=lambda x: x['code'])
        
        return jsonify({'subjects': subjects, 'total': len(subjects)})
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách môn học: {e}", exc_info=True)
        return jsonify({'error': f'Không thể lấy danh sách môn học: {str(e)}'}), 500

@app.route('/api/examples')
def get_examples():
    """API để lấy câu hỏi mẫu"""
    examples = [
        "Liệt kê các môn học ngành AI",
        "CSI106 là môn gì?",
        "Các môn có 3 tín chỉ",
        "MAD101 có bao nhiêu tín chỉ?",
        "Tất cả môn học toán",
        "Danh sách môn học kỳ 1"
    ]
    return jsonify({'examples': examples})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Trang không tìm thấy'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Lỗi server nội bộ'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 