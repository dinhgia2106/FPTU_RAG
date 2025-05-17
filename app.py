import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
import asyncio
import re

# Xử lý event loop để tránh xung đột giữa Streamlit và PyTorch
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Tải biến môi trường
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Đảm bảo API key được cấu hình
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY không được cấu hình. Vui lòng kiểm tra file .env")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Lỗi khi cấu hình Gemini API: {e}")
    st.stop()

# Cache dữ liệu để tránh tải lại
@st.cache_resource
def load_resources():
    """Tải FAISS index, dữ liệu chunks và các mô hình cần thiết."""
    
    # Đường dẫn đến file
    faiss_index_path = "Faiss/all_syllabi_faiss.index"
    chunks_json_path = "Embedded/all_embeddings.json"
    
    # Tải FAISS index
    try:
        index = faiss.read_index(faiss_index_path)
        st.sidebar.success(f"✅ Đã tải FAISS index với {index.ntotal} vectors")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải FAISS index: {e}")
        return None, None, None
    
    # Tải dữ liệu chunks
    try:
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        st.sidebar.success(f"✅ Đã tải dữ liệu chunks: {len(chunks_data)} chunks")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải dữ liệu chunks: {e}")
        return index, None, None
    
    # Tải mô hình embedding
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    try:
        embedding_model = SentenceTransformer(embedding_model_name)
        st.sidebar.success(f"✅ Đã tải mô hình embedding")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải mô hình embedding: {e}")
        return index, chunks_data, None
    
    # Tạo bảng tra cứu subject_code -> syllabus_name và các metadata khác
    subject_metadata = {}
    for chunk in chunks_data:
        if 'metadata' in chunk and 'subject_code' in chunk['metadata']:
            subject_code = chunk['metadata']['subject_code']
            if subject_code not in subject_metadata and chunk['type'] == 'general_info':
                # Tìm tên môn học từ nội dung chunk
                content = chunk.get('content', '')
                name_match = re.search(r"Tên môn học: ([^(]+)", content)
                if name_match:
                    subject_name = name_match.group(1).strip()
                    subject_metadata[subject_code] = {
                        'name': subject_name,
                        'keywords': [subject_code, subject_name]
                    }
                    
                    # Extract number of credits
                    credit_match = re.search(r"Số tín chỉ: ([0-9]+)", content)
                    if credit_match:
                        subject_metadata[subject_code]['credits'] = credit_match.group(1)
                        
    # Lấy danh sách các mã môn học
    subject_codes = sorted(list(subject_metadata.keys()))
    
    return index, chunks_data, embedding_model, subject_codes, subject_metadata

def identify_subject_from_query(query, subject_metadata, embedding_model, faiss_index, all_chunks_data, top_k=3):
    """Xác định môn học mà câu hỏi có thể đang đề cập đến."""
    # Tạo embedding cho câu hỏi
    query_embedding = embedding_model.encode([query])
    
    # Tìm kiếm top_k chunks gần nhất
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k*5)
    
    # Đếm số lần xuất hiện của mỗi mã môn học
    subject_counts = {}
    subject_scores = {}  # Điểm số cho mỗi môn học dựa trên khoảng cách embedding
    
    for idx, i in enumerate(indices[0]):
        if 0 <= i < len(all_chunks_data):
            chunk = all_chunks_data[i]
            if 'metadata' in chunk and 'subject_code' in chunk['metadata']:
                subject_code = chunk['metadata']['subject_code']
                
                # Tăng số lần xuất hiện
                if subject_code not in subject_counts:
                    subject_counts[subject_code] = 0
                    subject_scores[subject_code] = 0
                
                subject_counts[subject_code] += 1
                # Tính điểm: 1/khoảng cách (càng gần càng cao điểm)
                # Cộng thêm điểm nếu chunk là general_info
                score_boost = 2 if chunk['type'] == 'general_info' else 1
                subject_scores[subject_code] += (1 / (1 + distances[0][idx])) * score_boost
    
    # Kiểm tra xem có trực tiếp đề cập đến mã môn học trong câu hỏi không
    direct_mention = None
    for subject_code in subject_metadata:
        # Kiểm tra mã môn học đầy đủ
        if subject_code.lower() in query.lower():
            direct_mention = subject_code
            break
        # Kiểm tra prefix của mã môn học (e.g., "DPL" cho "DPL302m")
        code_prefix = ''.join([c for c in subject_code if c.isalpha()])
        if code_prefix.lower() in query.lower() and len(code_prefix) >= 2:  # Chỉ xét prefix có ít nhất 2 ký tự
            # Tìm tất cả môn có prefix này
            possible_subjects = [code for code in subject_metadata if code.startswith(code_prefix)]
            if len(possible_subjects) == 1:
                direct_mention = possible_subjects[0]
                break
            elif len(possible_subjects) > 1:
                # Nếu có nhiều môn cùng prefix, chọn môn có điểm cao nhất
                highest_score = 0
                for subj in possible_subjects:
                    if subj in subject_scores and subject_scores[subj] > highest_score:
                        highest_score = subject_scores[subj]
                        direct_mention = subj
                break
    
    if direct_mention:
        return direct_mention, subject_metadata.get(direct_mention, {}).get('name', '')
    
    # Nếu không có đề cập trực tiếp, tìm môn có điểm cao nhất
    if subject_scores:
        best_subject = max(subject_scores, key=subject_scores.get)
        return best_subject, subject_metadata.get(best_subject, {}).get('name', '')
    
    return None, None

def get_answer_from_rag(query_text, faiss_index, all_chunks_data, embedding_model, gemini_model, 
                       subject_metadata, subject_filter=None, top_k=5, temperature=0.2,
                       auto_detect_subject=True):
    """Thực hiện pipeline RAG với Gemini để lấy câu trả lời."""
    start_time = time.time()  # Move this to the beginning of the function
    
    if not query_text:
        return "Vui lòng cung cấp câu hỏi.", [], None, None, 0  # Added 0 as processing_time
    
    detected_subject_code = None
    detected_subject_name = None
    
    # 1. Xử lý Câu hỏi và xác định môn học liên quan (nếu được bật)
    with st.status("Đang xử lý câu hỏi...") as status:
        if auto_detect_subject and not subject_filter:
            status.update(label="Đang xác định môn học từ câu hỏi...")
            detected_subject_code, detected_subject_name = identify_subject_from_query(
                query_text, subject_metadata, embedding_model, faiss_index, all_chunks_data
            )
            
            if detected_subject_code:
                status.update(label=f"Đã xác định câu hỏi có thể liên quan đến môn {detected_subject_code} ({detected_subject_name})")
                # Sử dụng môn học được phát hiện làm bộ lọc
                subject_filter = detected_subject_code
            else:
                status.update(label="Không xác định được môn học cụ thể, tìm kiếm trong tất cả các môn")
        
        try:
            status.update(label="Đang tạo embedding cho câu hỏi...")
            query_embedding = embedding_model.encode([query_text])
            status.write("✅ Đã tạo embedding cho câu hỏi")
        except Exception as e:
            status.error(f"❌ Lỗi khi tạo embedding cho câu hỏi: {e}")
            processing_time = time.time() - start_time  # Calculate time even for errors
            return "Lỗi khi xử lý câu hỏi.", [], None, None, processing_time

        # 2. Truy xuất Thông tin (Information Retrieval)
        try:
            status.update(label="Đang tìm kiếm thông tin liên quan...")
            distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k*3)  # Lấy nhiều hơn để lọc
            status.write("✅ Đã tìm kiếm trong FAISS index")
        except Exception as e:
            status.error(f"❌ Lỗi khi tìm kiếm trong FAISS: {e}")
            processing_time = time.time() - start_time
            return "Lỗi khi truy xuất thông tin.", [], None, None, processing_time

        # 3. Chuẩn bị Ngữ cảnh (Context Preparation)
        retrieved_chunks = []
        retrieved_chunks_content = []
        retrieved_indices = []

        for idx, i in enumerate(indices[0]):
            if 0 <= i < len(all_chunks_data):
                chunk = all_chunks_data[i]
                # Lọc theo môn học nếu có yêu cầu
                if subject_filter and 'metadata' in chunk and 'subject_code' in chunk['metadata']:
                    chunk_subject = chunk['metadata']['subject_code']
                    # Kiểm tra xem mã môn học có khớp với bộ lọc không (khớp đầy đủ hoặc là tiền tố)
                    if not (chunk_subject == subject_filter or 
                           (subject_filter in chunk_subject and 
                            chunk_subject.startswith(subject_filter))):
                        continue
                
                # Thêm vào danh sách kết quả
                retrieved_chunks.append(chunk)
                retrieved_chunks_content.append(chunk.get("content", ""))
                retrieved_indices.append(i)
                
                # Dừng khi đủ top_k chunks
                if len(retrieved_chunks) >= top_k:
                    break
            else:
                status.warning(f"Cảnh báo: Index {i} nằm ngoài phạm vi của all_chunks_data.")
        
        if not retrieved_chunks_content:
            processing_time = time.time() - start_time
            if subject_filter:
                return f"Không tìm thấy thông tin liên quan trong syllabus môn {subject_filter} để trả lời câu hỏi này.", [], detected_subject_code, detected_subject_name, processing_time
            else:
                return "Không tìm thấy thông tin liên quan trong các syllabus để trả lời câu hỏi này.", [], None, None, processing_time

        status.write(f"✅ Đã truy xuất {len(retrieved_chunks_content)} chunks liên quan")
        
        # Nếu phát hiện môn học, thêm thông tin vào ngữ cảnh
        if detected_subject_code or subject_filter:
            subject_code_to_use = detected_subject_code if detected_subject_code else subject_filter
            subject_info = subject_metadata.get(subject_code_to_use, {})
            subject_context = f"Thông tin về môn học {subject_code_to_use}"
            if 'name' in subject_info:
                subject_context += f" ({subject_info['name']})"
            if 'credits' in subject_info:
                subject_context += f", số tín chỉ: {subject_info['credits']}"
            
            context_for_llm = subject_context + "\n\n" + "\n\n".join(retrieved_chunks_content)
        else:
            context_for_llm = "\n\n".join(retrieved_chunks_content)
        
        # Tạo prompt cho Gemini
        prompt = f"""Dựa vào các thông tin sau đây từ syllabus của các môn học tại trường Đại học FPT:

-- BẮT ĐẦU NGỮ CẢNH SYLLABUS --
{context_for_llm}
-- KẾT THÚC NGỮ CẢNH SYLLABUS --

Hãy trả lời câu hỏi sau một cách ngắn gọn và chính xác, CHỈ dựa vào thông tin được cung cấp trong ngữ cảnh syllabus ở trên. 
Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không tìm thấy thông tin đó trong tài liệu được cung cấp.

Câu hỏi: {query_text}

Nếu câu hỏi là về một môn học cụ thể, luôn đề cập đến mã môn học (như DPL302m) trong câu trả lời để làm rõ bạn đang cung cấp thông tin về môn học nào.

Trả lời:"""

        # 4. Sinh Câu trả lời (Answer Generation) với Gemini
        status.update(label="Đang sinh câu trả lời bằng Gemini...")
        try:
            model_options = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            gemini_model_name = "gemini-1.5-flash-latest"
            model = genai.GenerativeModel(gemini_model_name, generation_config=model_options)
            response = model.generate_content(prompt)
            answer = response.text
            status.write("✅ Đã nhận phản hồi từ Gemini")
        except Exception as e:
            status.error(f"❌ Lỗi khi Gemini sinh câu trả lời: {e}")
            if hasattr(e, 'response') and e.response:
                status.error(f"Gemini API Response Error: {e.response}")
            if hasattr(e, 'message'):
                status.error(f"Error message: {e.message}")
            processing_time = time.time() - start_time
            return "Lỗi khi tạo câu trả lời bằng Gemini.", [], detected_subject_code, detected_subject_name, processing_time
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return answer, retrieved_chunks, detected_subject_code, detected_subject_name, processing_time

# UI của ứng dụng Streamlit
def main():
    st.set_page_config(
        page_title="RAG Syllabus Query - FPTU",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Hệ thống Truy vấn Syllabus FPT University")
    st.markdown("Hỏi đáp thông tin về các môn học dựa trên syllabus của trường Đại học FPT.")
    
    # Tải resources
    with st.spinner("Đang tải dữ liệu..."):
        faiss_index, chunks_data, embedding_model, subject_codes, subject_metadata = load_resources()
    
    if not faiss_index or not chunks_data or not embedding_model:
        st.error("Không thể tải các thành phần cần thiết. Vui lòng kiểm tra lỗi.")
        return
    
    # Sidebar cho cấu hình
    st.sidebar.title("Cấu hình")
    
    selected_subject = st.sidebar.selectbox(
        "Chọn môn học cụ thể (để trống để tìm trong tất cả các môn):",
        ["Tất cả các môn"] + subject_codes
    )
    
    subject_filter = None if selected_subject == "Tất cả các môn" else selected_subject
    
    auto_detect_subject = st.sidebar.checkbox("Tự động phát hiện môn học từ câu hỏi", value=True)
    
    top_k = st.sidebar.slider("Số lượng chunks để truy xuất:", 1, 100, 20)
    
    temperature = st.sidebar.slider(
        "Nhiệt độ (độ sáng tạo) của Gemini:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.2, 
        step=0.1
    )
    
    show_sources = st.sidebar.checkbox("Hiển thị nguồn tham khảo", value=True)
    
    # Khởi tạo Gemini model 
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    # Ô input cho câu hỏi
    query_text = st.text_area("Nhập câu hỏi của bạn về syllabus:", height=100)
    
    # Gợi ý câu hỏi
    st.caption("""
    **Gợi ý câu hỏi:** 
    - "DPL302m là môn học gì?"
    - "Môn Deep Learning có bao nhiêu tín chỉ?"
    - "Chuẩn đầu ra của môn DPL là gì?"
    - "Cho tôi biết tài liệu học tập của môn deep learning"
    """)
    
    # Nút tìm kiếm
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)
    
    # Xử lý tìm kiếm khi nhấn nút
    if search_button and query_text:
        # Thực hiện RAG và hiển thị kết quả
        answer, retrieved_chunks, detected_subject, detected_subject_name, processing_time = get_answer_from_rag(
            query_text, faiss_index, chunks_data, embedding_model, gemini_model,
            subject_metadata, subject_filter, top_k, temperature, auto_detect_subject
        )
        
        # Hiển thị môn học được phát hiện (nếu có)
        if detected_subject and auto_detect_subject:
            st.info(f"📘 Hệ thống phát hiện câu hỏi liên quan đến môn: **{detected_subject}** ({detected_subject_name})")
        
        # Hiển thị thời gian xử lý
        st.caption(f"Thời gian xử lý: {processing_time:.2f} giây")
        
        # Hiển thị câu trả lời
        st.markdown("### Câu trả lời:")
        st.markdown(answer)
        
        # Hiển thị nguồn tham khảo nếu được chọn
        if show_sources and retrieved_chunks:
            st.markdown("### Nguồn tham khảo:")
            
            for i, chunk in enumerate(retrieved_chunks):
                with st.expander(f"Nguồn {i+1}: {chunk.get('metadata', {}).get('title', f'Chunk {i+1}')}"):
                    st.markdown(f"**Môn học:** {chunk.get('metadata', {}).get('subject_code', 'N/A')}")
                    st.markdown(f"**Loại nội dung:** {chunk.get('type', 'N/A')}")
                    st.markdown(f"**Nội dung:**\n{chunk.get('content', 'N/A')}")

if __name__ == "__main__":
    main()