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
    
    # Lấy danh sách các mã môn học
    subject_codes = set()
    for chunk in chunks_data:
        if 'metadata' in chunk and 'subject_code' in chunk['metadata']:
            subject_codes.add(chunk['metadata']['subject_code'])
    
    return index, chunks_data, embedding_model, sorted(list(subject_codes))

def get_answer_from_rag(query_text, faiss_index, all_chunks_data, embedding_model, gemini_model, 
                       subject_filter=None, top_k=5, temperature=0.2):
    """Thực hiện pipeline RAG với Gemini để lấy câu trả lời."""
    if not query_text:
        return "Vui lòng cung cấp câu hỏi.", []
    
    start_time = time.time()
    
    # 1. Xử lý Câu hỏi (Query Processing)
    with st.status("Đang xử lý câu hỏi..."):
        try:
            query_embedding = embedding_model.encode([query_text])
            st.write("✅ Đã tạo embedding cho câu hỏi")
        except Exception as e:
            st.error(f"❌ Lỗi khi tạo embedding cho câu hỏi: {e}")
            return "Lỗi khi xử lý câu hỏi.", []

        # 2. Truy xuất Thông tin (Information Retrieval)
        try:
            distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k*3)  # Lấy nhiều hơn để lọc theo môn học
            st.write("✅ Đã tìm kiếm trong FAISS index")
        except Exception as e:
            st.error(f"❌ Lỗi khi tìm kiếm trong FAISS: {e}")
            return "Lỗi khi truy xuất thông tin.", []

        # 3. Chuẩn bị Ngữ cảnh (Context Preparation)
        retrieved_chunks = []
        retrieved_chunks_content = []
        retrieved_indices = []

        for idx, i in enumerate(indices[0]):
            if 0 <= i < len(all_chunks_data):
                chunk = all_chunks_data[i]
                # Lọc theo môn học nếu có yêu cầu
                if subject_filter and 'metadata' in chunk and 'subject_code' in chunk['metadata']:
                    if chunk['metadata']['subject_code'] != subject_filter:
                        continue
                
                # Thêm vào danh sách kết quả
                retrieved_chunks.append(chunk)
                retrieved_chunks_content.append(chunk.get("content", ""))
                retrieved_indices.append(i)
                
                # Dừng khi đủ top_k chunks
                if len(retrieved_chunks) >= top_k:
                    break
            else:
                st.warning(f"Cảnh báo: Index {i} nằm ngoài phạm vi của all_chunks_data.")
        
        if not retrieved_chunks_content:
            if subject_filter:
                return f"Không tìm thấy thông tin liên quan trong syllabus môn {subject_filter} để trả lời câu hỏi này.", []
            else:
                return "Không tìm thấy thông tin liên quan trong các syllabus để trả lời câu hỏi này.", []

        st.write(f"✅ Đã truy xuất {len(retrieved_chunks_content)} chunks liên quan")
        context_for_llm = "\n\n".join(retrieved_chunks_content)
        
        # Tạo prompt cho Gemini
        prompt = f"""Dựa vào các thông tin sau đây từ syllabus của các môn học tại trường Đại học FPT:

-- BẮT ĐẦU NGỮ CẢNH SYLLABUS --
{context_for_llm}
-- KẾT THÚC NGỮ CẢNH SYLLABUS --

Hãy trả lời câu hỏi sau một cách ngắn gọn và chính xác, CHỈ dựa vào thông tin được cung cấp trong ngữ cảnh syllabus ở trên. 
Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không tìm thấy thông tin đó trong tài liệu được cung cấp.

Câu hỏi: {query_text}

Trả lời:"""

        # 4. Sinh Câu trả lời (Answer Generation) với Gemini
        st.write("Đang sinh câu trả lời bằng Gemini...")
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
            st.write("✅ Đã nhận phản hồi từ Gemini")
        except Exception as e:
            st.error(f"❌ Lỗi khi Gemini sinh câu trả lời: {e}")
            if hasattr(e, 'response') and e.response:
                st.error(f"Gemini API Response Error: {e.response}")
            if hasattr(e, 'message'):
                st.error(f"Error message: {e.message}")
            return "Lỗi khi tạo câu trả lời bằng Gemini.", []
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return answer, retrieved_chunks, processing_time

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
        faiss_index, chunks_data, embedding_model, subject_codes = load_resources()
    
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
    
    top_k = st.sidebar.slider("Số lượng chunks để truy xuất:", 1, 10, 5)
    
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
    
    # Nút tìm kiếm
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)
    
    # Xử lý tìm kiếm khi nhấn nút
    if search_button and query_text:
        # Thực hiện RAG và hiển thị kết quả
        answer, retrieved_chunks, processing_time = get_answer_from_rag(
            query_text, faiss_index, chunks_data, embedding_model, gemini_model,
            subject_filter, top_k, temperature
        )
        
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