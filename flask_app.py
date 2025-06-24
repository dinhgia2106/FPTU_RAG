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

# Initialize RAG Engine với multiple API keys
api_keys = []
for i in range(1, 4):  # Check for GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
        api_keys.append(key)

# Fallback to original GEMINI_API_KEY nếu không có numbered keys
if not api_keys:
    original_key = os.getenv("GEMINI_API_KEY")
    if original_key:
        api_keys.append(original_key)

if not api_keys:
    raise ValueError("Không tìm thấy API key nào. Cần có ít nhất GEMINI_API_KEY hoặc GEMINI_API_KEY_1")

logger.info(f"✓ Tìm thấy {len(api_keys)} API keys")
for i, key in enumerate(api_keys, 1):
    logger.info(f"  API Key {i}: ...{key[-10:]}")

rag_engine = AdvancedRAGEngine(api_keys)

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
        
        # Sử dụng chatbot_query method mới
        session_id = session.get('session_id', 'anonymous')
        logger.info("Sử dụng chatbot engine với conversation memory...")
        
        result = rag_engine.chatbot_query(question, session_id, enable_multihop)
        
        # Handle result format
        answer = result.get('answer', '')
        search_results = result.get('search_results', [])
        query_type = result.get('query_type', 'unknown')
        
        # Build response object
        response = {
            'answer': answer,
            'search_results': search_results,
            'metadata': {
                'subjects_covered': len(set([r.get('subject_code', '') for r in search_results if isinstance(r, dict)])),
                'query_type': query_type,
                'conversation_enhanced': result.get('conversation_enhanced', False),
                'is_data_search': query_type == 'data_search',
                'is_direct_chat': query_type == 'direct_chat'
            }
        }
        
        # Add multihop info if available
        if 'multihop_info' in result:
            response['multihop_info'] = result['multihop_info']
        else:
            response['multihop_info'] = {'has_followup': False}
        
        # Tính thời gian xử lý tổng
        total_time = time.time() - start_time
        
        logger.info(f"FLASK RESPONSE SUMMARY:")
        logger.info(f"  Query: '{question}'")
        logger.info(f"  Session ID: {session_id}")
        logger.info(f"  Multihop: {enable_multihop}")
        logger.info(f"  Query type: {response.get('metadata', {}).get('query_type', 'unknown')}")
        logger.info(f"  Is data search: {response.get('metadata', {}).get('is_data_search', False)}")
        logger.info(f"  Is direct chat: {response.get('metadata', {}).get('is_direct_chat', False)}")
        logger.info(f"  Conversation enhanced: {response.get('metadata', {}).get('conversation_enhanced', False)}")
        
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
            followup_queries = response['multihop_info'].get('followup_queries', [])
            followup_count = len(followup_queries)
            logger.info(f"MULTIHOP DETAILS:")
            logger.info(f"  - Followup queries: {followup_count}")
            for i, fq in enumerate(followup_queries):
                logger.info(f"    {i+1}. {fq.get('query', '')} (confidence: {fq.get('confidence', 0):.2f})")
        
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
        "Xin chào",
        "Bạn có thể giúp gì?",
        "Liệt kê các môn học ngành AI",
        "CSI106 là môn gì?",
        "Các môn có 3 tín chỉ",
        "MAD101 có bao nhiêu tín chỉ?",
        "Tất cả môn học toán",
        "Danh sách môn học kỳ 1",
        "Cảm ơn bạn"
    ]
    return jsonify({'examples': examples})

@app.route('/api/conversation')
def get_conversation():
    """API để lấy lịch sử conversation"""
    try:
        session_id = session.get('session_id', 'anonymous')
        
        if session_id in rag_engine.conversation_memory:
            history = rag_engine.conversation_memory[session_id]
            return jsonify({
                'session_id': session_id,
                'conversation_count': len(history),
                'conversation': history[-10:]  # Chỉ trả về 10 tin nhắn gần nhất
            })
        else:
            return jsonify({
                'session_id': session_id,
                'conversation_count': 0,
                'conversation': []
            })
    except Exception as e:
        logger.error(f"Lỗi lấy conversation: {e}")
        return jsonify({'error': f'Không thể lấy lịch sử conversation: {str(e)}'}), 500

@app.route('/api/conversation/clear', methods=['POST'])
def clear_conversation():
    """API để xóa lịch sử conversation"""
    try:
        session_id = session.get('session_id', 'anonymous')
        
        if session_id in rag_engine.conversation_memory:
            del rag_engine.conversation_memory[session_id]
        
        return jsonify({
            'success': True,
            'message': 'Đã xóa lịch sử conversation',
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Lỗi xóa conversation: {e}")
        return jsonify({'error': f'Không thể xóa lịch sử conversation: {str(e)}'}), 500

@app.route('/api/status')
def get_system_status():
    """API để lấy trạng thái hệ thống và API keys"""
    try:
        api_status = rag_engine.api_key_manager.get_status()
        
        # Test current API key
        current_key_working = False
        try:
            current_model = rag_engine.api_key_manager.get_current_model()
            if current_model:
                test_response = current_model.generate_content("Test")
                current_key_working = True
        except:
            current_key_working = False
        
        return jsonify({
            'api_keys': {
                'total': api_status['total_keys'],
                'current_index': api_status['current_index'],
                'current_key_suffix': api_status['current_key_suffix'],
                'failed_keys': api_status['failed_keys'],
                'available_keys': api_status['available_keys'],
                'current_key_working': current_key_working
            },
            'system': {
                'data_loaded': hasattr(rag_engine, 'data') and rag_engine.data is not None,
                'data_count': len(rag_engine.data) if hasattr(rag_engine, 'data') and rag_engine.data else 0,
                'conversations_active': len(rag_engine.conversation_memory)
            }
        })
    except Exception as e:
        logger.error(f"Lỗi lấy system status: {e}")
        return jsonify({'error': f'Không thể lấy trạng thái hệ thống: {str(e)}'}), 500

@app.route('/api/reset-keys', methods=['POST'])
def reset_api_keys():
    """API để reset failed API keys"""
    try:
        rag_engine.api_key_manager.reset_failed_keys()
        
        return jsonify({
            'success': True,
            'message': 'Đã reset tất cả failed API keys',
            'status': rag_engine.api_key_manager.get_status()
        })
    except Exception as e:
        logger.error(f"Lỗi reset API keys: {e}")
        return jsonify({'error': f'Không thể reset API keys: {str(e)}'}), 500

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