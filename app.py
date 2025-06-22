"""
Improved Syllabus Search Engine App - Cải tiến ứng dụng tìm kiếm Syllabus FPT

Ứng dụng Streamlit tích hợp các tính năng tìm kiếm nâng cao, hiểu ngữ cảnh,
và truy vấn phức tạp cho dữ liệu syllabus.
"""

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
import sys
from typing import Dict, List, Any, Optional, Union

# Thêm đường dẫn để import các module tùy chỉnh
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các module tùy chỉnh
from contextual_query_processor import (
    ContextualQueryProcessor, 
    EntityLinker, 
    QueryExecutor, 
    ContextualSearchEngine,
    build_entity_data_from_chunks,
    save_entity_data,
    load_entity_data,
    EntityType,
    QueryType
)

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
    entity_data_path = "Entity/entity_data.json"
    
    # Tải FAISS index
    try:
        index = faiss.read_index(faiss_index_path)
        st.sidebar.success(f"✅ Đã tải FAISS index với {index.ntotal} vectors")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải FAISS index: {e}")
        return None, None, None, None, None, None
    
    # Tải dữ liệu chunks
    try:
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        st.sidebar.success(f"✅ Đã tải dữ liệu chunks: {len(chunks_data)} chunks")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải dữ liệu chunks: {e}")
        return index, None, None, None, None, None
    
    # Tải mô hình embedding
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    try:
        embedding_model = SentenceTransformer(embedding_model_name)
        st.sidebar.success(f"✅ Đã tải mô hình embedding")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải mô hình embedding: {e}")
        return index, chunks_data, None, None, None, None
    
    # Tải hoặc xây dựng dữ liệu thực thể
    entity_data = {}
    try:
        if os.path.exists(entity_data_path):
            entity_data = load_entity_data(entity_data_path)
            st.sidebar.success(f"✅ Đã tải dữ liệu thực thể: {len(entity_data)} thực thể")
        else:
            st.sidebar.info("Đang xây dựng dữ liệu thực thể từ chunks...")
            entity_data = build_entity_data_from_chunks(chunks_data, embedding_model)
            os.makedirs(os.path.dirname(entity_data_path), exist_ok=True)
            save_entity_data(entity_data, entity_data_path)
            st.sidebar.success(f"✅ Đã xây dựng và lưu dữ liệu thực thể: {len(entity_data)} thực thể")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Lỗi khi xử lý dữ liệu thực thể: {e}. Sẽ tiếp tục với dữ liệu trống.")
    
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
    
    # Tạo contextual search engine
    try:
        search_engine = ContextualSearchEngine(
            embedding_model=embedding_model,
            faiss_index=index,
            chunks_data=chunks_data,
            entity_data=entity_data
        )
        st.sidebar.success(f"✅ Đã khởi tạo Contextual Search Engine")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi khởi tạo Contextual Search Engine: {e}")
        search_engine = None
    
    return index, chunks_data, embedding_model, subject_codes, subject_metadata, search_engine

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

def get_answer_from_contextual_rag(query_text, search_engine, gemini_model, 
                                  subject_metadata, subject_filter=None, top_k=5, temperature=0.2,
                                  auto_detect_subject=True):
    """Thực hiện pipeline RAG với Contextual Search Engine và Gemini để lấy câu trả lời."""
    start_time = time.time()
    
    if not query_text:
        return "Vui lòng cung cấp câu hỏi.", [], None, None, 0
    
    detected_subject_code = None
    detected_subject_name = None
    
    with st.status("Đang xử lý câu hỏi...") as status:
        # 1. Xác định môn học liên quan (nếu được bật)
        if auto_detect_subject and not subject_filter:
            status.update(label="Đang phân tích truy vấn và xác định môn học...")
            
            # Sử dụng phân tích truy vấn từ ContextualQueryProcessor
            query_analysis = search_engine.query_processor.analyze_query(query_text)
            
            # Kiểm tra xem có thực thể Course trong phân tích không
            course_entities = query_analysis.get("entities", {}).get(EntityType.COURSE, [])
            if course_entities:
                subject_code = course_entities[0].get("value")
                detected_subject_code = subject_code
                detected_subject_name = subject_metadata.get(subject_code, {}).get('name', '')
                status.update(label=f"Đã xác định câu hỏi liên quan đến môn {detected_subject_code} ({detected_subject_name})")
                subject_filter = detected_subject_code
            else:
                # Sử dụng phương pháp dự phòng dựa trên vector similarity
                detected_subject_code, detected_subject_name = identify_subject_from_query(
                    query_text, subject_metadata, search_engine.embedding_model, 
                    search_engine.faiss_index, search_engine.chunks_data
                )
                
                if detected_subject_code:
                    status.update(label=f"Đã xác định câu hỏi có thể liên quan đến môn {detected_subject_code} ({detected_subject_name})")
                    subject_filter = detected_subject_code
                else:
                    status.update(label="Không xác định được môn học cụ thể, tìm kiếm trong tất cả các môn")
        
        # 2. Thực hiện tìm kiếm ngữ cảnh
        try:
            status.update(label="Đang thực hiện tìm kiếm ngữ cảnh...")
            
            # Áp dụng bộ lọc môn học nếu có
            if subject_filter:
                # Tạo truy vấn có cấu trúc với bộ lọc môn học
                structured_query = {
                    "type": "filter",
                    "target": {
                        "entity_type": EntityType.COURSE,
                        "entity_value": subject_filter
                    }
                }
                search_results = search_engine.search(query_text, top_k, structured_query)
            else:
                search_results = search_engine.search(query_text, top_k)
            
            status.write("✅ Đã thực hiện tìm kiếm ngữ cảnh")
            
            # Kiểm tra kết quả tìm kiếm
            if not search_results.get("success") or not search_results.get("results"):
                status.warning("⚠️ Không tìm thấy kết quả phù hợp, sẽ thử tìm kiếm vector")
                
                # Thực hiện tìm kiếm vector như phương pháp dự phòng
                query_embedding = search_engine.embedding_model.encode([query_text])
                distances, indices = search_engine.faiss_index.search(query_embedding.astype(np.float32), top_k)
                
                retrieved_chunks = []
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(search_engine.chunks_data):
                        chunk = search_engine.chunks_data[idx]
                        
                        # Lọc theo môn học nếu có yêu cầu
                        if subject_filter and 'metadata' in chunk and 'subject_code' in chunk['metadata']:
                            chunk_subject = chunk['metadata']['subject_code']
                            if chunk_subject != subject_filter:
                                continue
                        
                        retrieved_chunks.append({
                            "chunk": chunk,
                            "distance": float(distances[0][i]),
                            "score": float(1 / (1 + distances[0][i]))
                        })
                
                if not retrieved_chunks:
                    processing_time = time.time() - start_time
                    if subject_filter:
                        return f"Không tìm thấy thông tin liên quan trong syllabus môn {subject_filter} để trả lời câu hỏi này.", [], detected_subject_code, detected_subject_name, processing_time
                    else:
                        return "Không tìm thấy thông tin liên quan trong các syllabus để trả lời câu hỏi này.", [], None, None, processing_time
                
                search_results = {
                    "success": True,
                    "query_type": "vector",
                    "results": retrieved_chunks
                }
            
        except Exception as e:
            status.error(f"❌ Lỗi khi thực hiện tìm kiếm: {e}")
            processing_time = time.time() - start_time
            return f"Lỗi khi tìm kiếm thông tin: {e}", [], detected_subject_code, detected_subject_name, processing_time
        
        # 3. Chuẩn bị ngữ cảnh cho LLM
        status.update(label="Đang chuẩn bị ngữ cảnh cho Gemini...")
        
        context_chunks = []
        if search_results.get("query_type") == "vector":
            # Kết quả từ tìm kiếm vector
            for result in search_results.get("results", []):
                chunk = result.get("chunk", {})
                if "content" in chunk:
                    context_chunks.append(chunk.get("content", ""))
        else:
            # Kết quả từ tìm kiếm ngữ cảnh
            for result in search_results.get("results", []):
                if isinstance(result, dict):
                    if "chunk" in result and "content" in result["chunk"]:
                        context_chunks.append(result["chunk"].get("content", ""))
                    elif "content" in result:
                        context_chunks.append(result.get("content", ""))
                    elif "entity_data" in result and "content" in result["entity_data"]:
                        context_chunks.append(result["entity_data"].get("content", ""))
        
        if not context_chunks:
            processing_time = time.time() - start_time
            if subject_filter:
                return f"Không tìm thấy thông tin liên quan trong syllabus môn {subject_filter} để trả lời câu hỏi này.", [], detected_subject_code, detected_subject_name, processing_time
            else:
                return "Không tìm thấy thông tin liên quan trong các syllabus để trả lời câu hỏi này.", [], None, None, processing_time
        
        status.write(f"✅ Đã chuẩn bị {len(context_chunks)} đoạn ngữ cảnh")
        
        # Nếu phát hiện môn học, thêm thông tin vào ngữ cảnh
        if detected_subject_code or subject_filter:
            subject_code_to_use = detected_subject_code if detected_subject_code else subject_filter
            subject_info = subject_metadata.get(subject_code_to_use, {})
            subject_context = f"Thông tin về môn học {subject_code_to_use}"
            if 'name' in subject_info:
                subject_context += f" ({subject_info['name']})"
            if 'credits' in subject_info:
                subject_context += f", số tín chỉ: {subject_info['credits']}"
            
            context_for_llm = subject_context + "\n\n" + "\n\n".join(context_chunks)
        else:
            context_for_llm = "\n\n".join(context_chunks)
        
        # Thêm thông tin phân tích truy vấn vào prompt nếu có
        query_analysis_info = ""
        if "query_analysis" in search_results:
            analysis = search_results["query_analysis"]
            query_type = analysis.get("query_type", "")
            
            if query_type == QueryType.SIMPLE_INFO:
                query_analysis_info = "Đây là truy vấn thông tin đơn giản về một môn học."
            elif query_type == QueryType.ATTRIBUTE:
                attributes = analysis.get("attributes", [])
                if attributes:
                    attr_names = [attr.get("name") for attr in attributes]
                    query_analysis_info = f"Đây là truy vấn về thuộc tính {', '.join(attr_names)} của một môn học."
            elif query_type == QueryType.RELATIONSHIP:
                query_analysis_info = "Đây là truy vấn về mối quan hệ giữa các thực thể (ví dụ: session, CLO)."
            elif query_type == QueryType.AGGREGATION:
                aggregations = analysis.get("aggregations", [])
                if aggregations:
                    agg_names = [agg.get("name") for agg in aggregations]
                    query_analysis_info = f"Đây là truy vấn tổng hợp {', '.join(agg_names)}."
            elif query_type == QueryType.CLASSIFICATION:
                classifications = analysis.get("classifications", [])
                if classifications:
                    class_names = [cls.get("name") for cls in classifications]
                    query_analysis_info = f"Đây là truy vấn phân loại theo {', '.join(class_names)}."
            elif query_type == QueryType.LINKING:
                query_analysis_info = "Đây là truy vấn liên kết giữa các thực thể."
        
        # Tạo prompt cho Gemini
        prompt = f"""Dựa vào các thông tin sau đây từ syllabus của các môn học tại trường Đại học FPT:

-- BẮT ĐẦU NGỮ CẢNH SYLLABUS --
{context_for_llm}
-- KẾT THÚC NGỮ CẢNH SYLLABUS --

{query_analysis_info}

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
            
            gemini_model_name = "gemini-1.5-flash"
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
    
    return answer, search_results.get("results", []), detected_subject_code, detected_subject_name, processing_time

# UI của ứng dụng Streamlit
def main():
    st.set_page_config(
        page_title="Enhanced Syllabus Query - FPTU",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Hệ thống Truy vấn Syllabus FPT University (Cải tiến)")
    st.markdown("Hỏi đáp thông tin về các môn học dựa trên syllabus của trường Đại học FPT với khả năng hiểu ngữ cảnh nâng cao.")
    
    # Tải resources
    with st.spinner("Đang tải dữ liệu..."):
        faiss_index, chunks_data, embedding_model, subject_codes, subject_metadata, search_engine = load_resources()
    
    if not faiss_index or not chunks_data or not embedding_model or not search_engine:
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
    
    top_k = st.sidebar.slider("Số lượng chunks để truy xuất:", 1, 5000, 100)
    
    temperature = st.sidebar.slider(
        "Nhiệt độ (độ sáng tạo) của Gemini:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.2, 
        step=0.1
    )
    
    show_sources = st.sidebar.checkbox("Hiển thị nguồn tham khảo", value=True)
    show_analysis = st.sidebar.checkbox("Hiển thị phân tích truy vấn", value=False)
    
    # Khởi tạo Gemini model 
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Ô input cho câu hỏi
    query_text = st.text_area("Nhập câu hỏi của bạn về syllabus:", height=100)
    
    # Gợi ý câu hỏi
    st.caption("""
    **Gợi ý câu hỏi nâng cao:** 
    - "DPL302m là môn học gì?"
    - "Môn Deep Learning có bao nhiêu tín chỉ?"
    - "Chuẩn đầu ra của môn DPL là gì?"
    - "Môn DPL302m có bao nhiêu CLO?"
    - "Có bao nhiêu môn toán?"
    - "Môn DPL302m có bao nhiêu bài kiểm tra, cách tính điểm thế nào?"
    """)
    
    # Nút tìm kiếm
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)
    
    # Xử lý tìm kiếm khi nhấn nút
    if search_button and query_text:
        # Thực hiện RAG và hiển thị kết quả
        answer, retrieved_results, detected_subject, detected_subject_name, processing_time = get_answer_from_contextual_rag(
            query_text, search_engine, gemini_model,
            subject_metadata, subject_filter, top_k, temperature, auto_detect_subject
        )
        
        # Hiển thị thông tin môn học được phát hiện
        if detected_subject and auto_detect_subject and not subject_filter:
            st.info(f"📌 Câu hỏi được xác định liên quan đến môn: **{detected_subject}** ({detected_subject_name})")
        
        # Hiển thị câu trả lời
        st.markdown("### Câu trả lời:")
        st.markdown(answer)
        
        # Hiển thị thời gian xử lý
        st.caption(f"⏱️ Thời gian xử lý: {processing_time:.2f} giây")
        
        # Hiển thị phân tích truy vấn nếu được yêu cầu
        if show_analysis:
            st.markdown("### Phân tích truy vấn:")
            
            # Phân tích truy vấn
            query_analysis = search_engine.query_processor.analyze_query(query_text)
            
            # Hiển thị loại truy vấn
            query_type = query_analysis.get("query_type", "")
            query_type_names = {
                QueryType.SIMPLE_INFO: "Truy vấn thông tin đơn giản",
                QueryType.ATTRIBUTE: "Truy vấn thuộc tính",
                QueryType.RELATIONSHIP: "Truy vấn quan hệ",
                QueryType.AGGREGATION: "Truy vấn tổng hợp",
                QueryType.CLASSIFICATION: "Truy vấn phân loại",
                QueryType.LINKING: "Truy vấn liên kết"
            }
            st.markdown(f"**Loại truy vấn:** {query_type_names.get(query_type, query_type)}")
            
            # Hiển thị các thực thể được nhận diện
            entities = query_analysis.get("entities", {})
            if entities:
                st.markdown("**Thực thể được nhận diện:**")
                for entity_type, entity_list in entities.items():
                    entity_type_names = {
                        EntityType.COURSE: "Môn học",
                        EntityType.SESSION: "Buổi học",
                        EntityType.CLO: "Chuẩn đầu ra",
                        EntityType.ASSESSMENT: "Đánh giá",
                        EntityType.MATERIAL: "Tài liệu"
                    }
                    st.markdown(f"- {entity_type_names.get(entity_type, entity_type)}: {', '.join([e.get('value', '') for e in entity_list])}")
            
            # Hiển thị các thuộc tính được nhận diện
            attributes = query_analysis.get("attributes", [])
            if attributes:
                st.markdown("**Thuộc tính được nhận diện:**")
                for attr in attributes:
                    st.markdown(f"- {attr.get('name', '')}")
            
            # Hiển thị truy vấn có cấu trúc
            structured_query = query_analysis.get("structured_query", {})
            if structured_query:
                st.markdown("**Truy vấn có cấu trúc:**")
                st.json(structured_query)
        
        # Hiển thị nguồn tham khảo nếu được yêu cầu
        if show_sources and retrieved_results:
            st.markdown("### Nguồn tham khảo:")
            
            for i, result in enumerate(retrieved_results[:5]):  # Giới hạn hiển thị 5 nguồn
                with st.expander(f"Nguồn #{i+1}"):
                    if isinstance(result, dict):
                        if "chunk" in result and "content" in result["chunk"]:
                            st.markdown(result["chunk"].get("content", ""))
                            if "metadata" in result["chunk"]:
                                metadata = result["chunk"]["metadata"]
                                st.caption(f"Loại: {result['chunk'].get('type', 'N/A')} | Môn học: {metadata.get('subject_code', 'N/A')} | Tiêu đề: {metadata.get('title', 'N/A')}")
                        elif "content" in result:
                            st.markdown(result.get("content", ""))
                            if "metadata" in result:
                                metadata = result["metadata"]
                                st.caption(f"Loại: {result.get('type', 'N/A')} | Môn học: {metadata.get('subject_code', 'N/A')} | Tiêu đề: {metadata.get('title', 'N/A')}")
                        elif "entity_data" in result and "content" in result["entity_data"]:
                            st.markdown(result["entity_data"].get("content", ""))
                            entity_data = result["entity_data"]
                            st.caption(f"Loại thực thể: {entity_data.get('entity_type', 'N/A')} | ID: {entity_data.get('entity_id', 'N/A')}")

if __name__ == "__main__":
    main()
