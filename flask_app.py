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

# Cấu hình logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

# Initialize RAG Engine
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY không được tìm thấy trong .env file")

rag_engine = AdvancedRAGEngine(GEMINI_API_KEY)

# Initialize with data
logger.info("==================== BẮT ĐẦU KHỞI TẠO FPTU RAG ENGINE ====================")
try:
    if os.path.exists("Data/combined_data.json"):
        logger.info("Tìm thấy combined_data.json, đang khởi tạo...")
        rag_engine.initialize("Data/combined_data.json")
        logger.info("✓ Đã khởi tạo RAG engine với combined_data.json")
        logger.info(f"✓ Dữ liệu: {len(rag_engine.data)} items")
    elif os.path.exists("Data/reduced_data.json"):
        logger.info("Tìm thấy reduced_data.json, đang khởi tạo...")
        rag_engine.initialize("Data/reduced_data.json")
        logger.info("✓ Đã khởi tạo RAG engine với reduced_data.json")
        logger.info(f"✓ Dữ liệu: {len(rag_engine.data)} items")
    else:
        logger.error("✗ Không tìm thấy file dữ liệu")
        logger.error("Cần có Data/combined_data.json hoặc Data/reduced_data.json")
except Exception as e:
    logger.error(f"✗ Lỗi khởi tạo RAG engine: {e}")
    
logger.info("==================== FPTU RAG ENGINE SẴN SÀNG ====================")
logger.info("Flask app đang khởi động...")
logger.info("Truy cập http://localhost:5000 để sử dụng")

@app.route('/')
def index():
    """Trang chính với giao diện chat"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint cho chat với hỗ trợ multi-hop query"""
    import time
    start_time = time.time()
    
    try:
        data = request.get_json()
        question = data.get('message', '').strip()
        enable_multihop = data.get('multihop', False)  # Mặc định TẮT multi-hop
        
        logger.info(f"==================== FLASK: NHẬN REQUEST ====================")
        logger.info(f"USER IP: {request.remote_addr}")
        logger.info(f"SESSION ID: {session.get('session_id', 'N/A')}")
        logger.info(f"USER QUESTION: '{question}'")
        logger.info(f"MULTIHOP ENABLED: {enable_multihop}")
        
        if not question:
            logger.warning("Câu hỏi rỗng")
            return jsonify({'error': 'Vui lòng nhập câu hỏi'}), 400
        
        # Check if multi-hop is supported
        if enable_multihop and hasattr(rag_engine, 'query_with_multihop'):
            logger.info("Sử dụng multi-hop query engine...")
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
            logger.info("Sử dụng normal query engine...")
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
        
        # Tính thời gian xử lý tổng
        total_time = time.time() - start_time
        
        logger.info(f"FLASK RESPONSE SUMMARY:")
        logger.info(f"  Query: '{question[:50]}{'...' if len(question) > 50 else ''}'")
        logger.info(f"  Multihop: {enable_multihop}")
        logger.info(f"  Response type: {response.get('metadata', {}).get('query_type', 'unknown')}")
        logger.info(f"  Is quick response: {response.get('metadata', {}).get('is_quick_response', False)}")
        
        # Chi tiết về phản hồi
        answer_text = response.get('answer', '')
        logger.info(f"RESPONSE DETAILS:")
        logger.info(f"  - Length: {len(answer_text)} ký tự")
        logger.info(f"  - Word count: ~{len(answer_text.split())} từ")
        answer_line_count = len(answer_text.split('\n'))
        logger.info(f"  - Line count: {answer_line_count}")
        logger.info(f"  - Contains markdown tables: {'|' in answer_text}")
        logger.info(f"  - Contains headers: {'#' in answer_text}")
        
        # Chi tiết về search results
        search_results = response.get('search_results', [])
        logger.info(f"SEARCH RESULTS DETAILS:")
        logger.info(f"  - Count: {len(search_results)}")
        if search_results:
            subject_codes = [r.get('subject_code', 'N/A') for r in search_results]
            content_types = [r.get('type', 'N/A') for r in search_results]
            scores = [r.get('score', 0) for r in search_results]
            
            logger.info(f"  - Subject codes: {subject_codes}")
            logger.info(f"  - Content types: {list(set(content_types))}")
            logger.info(f"  - Score range: {min(scores):.2f} - {max(scores):.2f}")
        
        # Multihop details nếu có
        if 'multihop_info' in response and response['multihop_info'].get('has_followup'):
            followup_count = len(response['multihop_info'].get('followup_queries', []))
            logger.info(f"MULTIHOP DETAILS:")
            logger.info(f"  - Followup queries: {followup_count}")
            for i, fq in enumerate(response['multihop_info'].get('followup_queries', [])[:3]):
                logger.info(f"    {i+1}. {fq.get('query', '')[:60]}... (confidence: {fq.get('confidence', 0):.2f})")
        
        logger.info(f"PERFORMANCE:")
        logger.info(f"  - Total processing time: {total_time:.2f}s")
        logger.info(f"  - Subjects covered: {response.get('metadata', {}).get('subjects_covered', 0)}")
        logger.info(f"==================== FLASK: HOÀN THÀNH REQUEST ====================")
        
        return jsonify(response)
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"✗ LỖI XỬ LÝ CHAT sau {total_time:.2f}s: {e}", exc_info=True)
        logger.error(f"==================== FLASK: LỖI REQUEST ====================")
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
    logger.info("==================== KHỞI ĐỘNG FLASK SERVER ====================")
    logger.info("Server đang chạy tại: http://0.0.0.0:5000")
    logger.info("Debug mode: ON")
    logger.info("Ấn Ctrl+C để tắt server")
    logger.info("==================================================================")
    app.run(debug=True, host='0.0.0.0', port=5000) 