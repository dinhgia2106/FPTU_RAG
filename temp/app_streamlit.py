import streamlit as st
import json
import os
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import time
import re

# Xử lý event loop cho asyncio (đặt ở đầu)
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():  # Kiểm tra nếu loop đã bị đóng
        raise RuntimeError("Event loop is closed")
except RuntimeError:
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)

# --- Constants ---
# Sử dụng các đường dẫn file từ pipeline của chúng ta
FAISS_INDEX_FILE = "all_data.faiss"
FAISS_MAPPING_FILE = "all_data_faiss_mapping.json"
# File này chứa nội dung đầy đủ và metadata của các chunk
ALL_CHUNKS_DATA_FILE = "embedded_all_chunks_with_students.json"

CHROMA_PERSIST_DIRECTORY = "all_data_chroma_db_store"
CHROMA_COLLECTION_NAME = "all_syllabus_and_students_collection"

# Model embedding được sử dụng (phải khớp với embedder.py và các vector stores)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Gemini API Key and Model
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY không được cấu hình trong file .env. Vui lòng kiểm tra.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Lỗi khi cấu hình Gemini API: {e}")
    st.stop()

GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'  # Hoặc gemini-1.5-pro-latest

# --- Helper Functions for Translation ---


def is_vietnamese(text):
    """Kiểm tra sơ bộ xem văn bản có chứa ký tự tiếng Việt không."""
    vietnamese_chars = r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    return bool(re.search(vietnamese_chars, text))


def translate_to_english_if_vietnamese(text_to_translate, gemini_model_instance, original_user_question_for_log=""):
    """Dịch văn bản sang tiếng Anh nếu phát hiện là tiếng Việt, sử dụng Gemini."""
    if is_vietnamese(text_to_translate):
        log_query_display = original_user_question_for_log if original_user_question_for_log else text_to_translate
        print(
            f"App: Phát hiện câu hỏi tiếng Việt: '{log_query_display}'. Đang dịch sang tiếng Anh...")
        st.info(
            f"Đang dịch câu hỏi '{log_query_display[:50]}...' sang tiếng Anh...")
        try:
            prompt = f'''Translate the following Vietnamese text to English. Provide only the English translation, without any introductory phrases, explanations, or quotation marks. Vietnamese text: "{text_to_translate}"'''
            response = gemini_model_instance.generate_content(prompt)
            translated_text = response.text.strip()
            # Xóa dấu ngoặc kép bao quanh nếu có
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]
            if translated_text.startswith("'") and translated_text.endswith("'"):
                translated_text = translated_text[1:-1]

            print(f'''App: Đã dịch sang tiếng Anh: "{translated_text}"''')
            st.success(
                f'''Đã dịch câu hỏi sang tiếng Anh: "{translated_text[:100]}..."''')
            return translated_text
        except Exception as e:
            st.warning(
                f"Lỗi khi dịch sang tiếng Anh: {e}. Sử dụng câu hỏi gốc.")
            print(
                f"App: Dịch thuật sang tiếng Anh thất bại cho: '{text_to_translate}'. Lỗi: {e}")
            return text_to_translate  # Trả về văn bản gốc nếu lỗi
    return text_to_translate  # Trả về văn bản gốc nếu không phải tiếng Việt

# --- Functions ---


@st.cache_resource
def get_gemini_model():
    """Tải và cache mô hình Gemini."""
    print("App: Đang khởi tạo mô hình Gemini...")
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print("App: Khởi tạo mô hình Gemini thành công.")
        # Thêm trạng thái cho Gemini LLM
        st.sidebar.success("✅ Mô hình Gemini (LLM)")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi tải Gemini LLM: {e}")
        st.error(
            f"Không thể tải mô hình Gemini: {e}. Ứng dụng không thể tiếp tục.")
        st.stop()
        return None


@st.cache_resource
def load_all_resources():
    """Tải tất cả các tài nguyên cần thiết: mô hình embedding, FAISS, ChromaDB, dữ liệu chunks và mô hình Gemini LLM."""
    resources = {
        "embedding_model": None,
        "faiss_index": None,
        "faiss_mapping": None,
        "all_chunks_data": None,
        "chroma_collection": None,
        "gemini_llm_model": None  # Thêm key cho mô hình Gemini
    }

    st.sidebar.write("--- Trạng Thái Tải Tài Nguyên ---")

    # 0. Tải mô hình Gemini LLM (cho dịch và sinh câu trả lời)
    resources["gemini_llm_model"] = get_gemini_model()
    if not resources["gemini_llm_model"]:  # get_gemini_model đã st.stop() nếu lỗi
        return resources  # Sẽ không đến đây nếu get_gemini_model bị lỗi và stop

    # 1. Tải mô hình Embedding
    try:
        print("App: Đang tải mô hình Sentence Transformer...")
        resources["embedding_model"] = SentenceTransformer(
            EMBEDDING_MODEL_NAME)
        print("App: Tải mô hình Sentence Transformer thành công.")
        st.sidebar.success("✅ Mô hình Embedding")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi tải Embedding Model: {e}")
        st.error(
            f"Không thể tải mô hình embedding: {e}. Ứng dụng không thể tiếp tục.")
        st.stop()

    # 2. Tải dữ liệu chunks đầy đủ (cần cho cả FAISS để lấy content)
    if not os.path.exists(ALL_CHUNKS_DATA_FILE):
        st.sidebar.error(
            f"❌ File chunks chính '{ALL_CHUNKS_DATA_FILE}' không tìm thấy.")
        st.error(
            f"File dữ liệu chunks '{ALL_CHUNKS_DATA_FILE}' không tồn tại. Hãy đảm bảo nó được tạo bởi script embedder.py.")
    else:
        try:
            with open(ALL_CHUNKS_DATA_FILE, 'r', encoding='utf-8') as f:
                resources["all_chunks_data"] = json.load(f)
            st.sidebar.success(
                f"✅ Dữ liệu Chunks ({len(resources['all_chunks_data'])} items)")
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi tải file '{ALL_CHUNKS_DATA_FILE}': {e}")

    # 3. Tải FAISS index và mapping
    if not os.path.exists(FAISS_INDEX_FILE):
        st.sidebar.warning(
            f"ℹ️ File FAISS index '{FAISS_INDEX_FILE}' không tìm thấy.")
    elif not os.path.exists(FAISS_MAPPING_FILE):
        st.sidebar.warning(
            f"ℹ️ File FAISS mapping '{FAISS_MAPPING_FILE}' không tìm thấy.")
    elif resources["all_chunks_data"] is None:  # FAISS cần all_chunks_data để hoạt động
        st.sidebar.error(
            "❌ Không thể tải FAISS do thiếu dữ liệu chunks chính.")
    else:
        try:
            resources["faiss_index"] = faiss.read_index(FAISS_INDEX_FILE)
            with open(FAISS_MAPPING_FILE, 'r', encoding='utf-8') as f:
                faiss_mapping_raw = json.load(f)
            # Chuyển key của id_to_chunk_info từ string sang int
            resources["faiss_mapping"] = {
                int(k): v for k, v in faiss_mapping_raw.items()}
            st.sidebar.success(
                f"✅ FAISS Index ({resources['faiss_index'].ntotal} vectors)")
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi tải FAISS: {e}")

    # 4. Kết nối ChromaDB
    if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        st.sidebar.warning(
            f"ℹ️ Thư mục ChromaDB '{CHROMA_PERSIST_DIRECTORY}' không tìm thấy.")
    else:
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            resources["chroma_collection"] = client.get_collection(
                name=CHROMA_COLLECTION_NAME)
            st.sidebar.success(
                f"✅ ChromaDB Collection ({resources['chroma_collection'].count()} items)")
        except Exception as e:
            # Có thể collection chưa tồn tại, không nên coi là lỗi nghiêm trọng ở bước tải
            st.sidebar.warning(f"ℹ️ ChromaDB: {e}")
            # Nếu lỗi không phải do collection không tồn tại, có thể là lỗi kết nối
            # st.error(f"Lỗi nghiêm trọng khi kết nối ChromaDB: {e}")

    st.sidebar.write("--------------------------------")
    return resources


def search_faiss(index, id_to_chunk_info, all_chunks_data, query_embedding, k=5):
    if not index or not id_to_chunk_info or not all_chunks_data:
        st.warning("FAISS chưa sẵn sàng hoặc thiếu dữ liệu.")
        return []

    print(f"FAISS: Đang tìm kiếm với k={k}...")
    distances, indices = index.search(
        np.array([query_embedding]).astype('float32'), k)
    results = []
    if indices.size > 0:
        for i in range(len(indices[0])):
            faiss_id = indices[0][i]
            if faiss_id < 0:  # Invalid index from FAISS
                continue

            mapped_info = id_to_chunk_info.get(faiss_id)
            if mapped_info:
                original_chunk_index = mapped_info.get('original_chunk_index')
                if original_chunk_index is not None and 0 <= original_chunk_index < len(all_chunks_data):
                    chunk_data = all_chunks_data[original_chunk_index]
                    # Sử dụng ID gốc từ all_chunks_data nếu có, nếu không thì tạo ID FAISS
                    result_id = chunk_data.get(
                        'id', f"faiss_idx_{faiss_id}_orig_idx_{original_chunk_index}")
                    results.append({
                        'id': result_id,
                        'content': chunk_data.get('content', 'N/A'),
                        'score': float(distances[0][i]),
                        'metadata': chunk_data.get('metadata', {}),
                        # Đảm bảo type, syllabus_id, course_id được lấy từ chunk_data.metadata nếu có
                        'type': chunk_data.get('metadata', {}).get('type', chunk_data.get('type', 'N/A')),
                        'syllabus_id': chunk_data.get('metadata', {}).get('syllabus_id', mapped_info.get('syllabus_id', 'N/A')),
                        'course_id': chunk_data.get('metadata', {}).get('course_id', mapped_info.get('course_id', 'N/A'))
                    })
                else:
                    print(
                        f"FAISS: original_chunk_index không hợp lệ ({original_chunk_index}) cho FAISS ID {faiss_id}.")
            else:
                print(
                    f"FAISS: Không tìm thấy mapping cho FAISS ID {faiss_id}.")
    print(f"FAISS: Tìm thấy {len(results)} kết quả.")
    return results


def search_chroma(collection, query_embedding, k=5):
    if not collection:
        st.warning("ChromaDB collection chưa sẵn sàng.")
        return []

    print(f"ChromaDB: Đang tìm kiếm với k={k}...")
    try:
        query_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
    except Exception as e:
        st.error(f"Lỗi khi query ChromaDB: {e}")
        return []

    results = []
    if query_results and query_results['ids'] and query_results['ids'][0]:
        for i in range(len(query_results['ids'][0])):
            results.append({
                'id': query_results['ids'][0][i],
                'content': query_results['documents'][0][i] if query_results['documents'] else 'N/A',
                'score': float(query_results['distances'][0][i]) if query_results['distances'] else float('inf'),
                'metadata': query_results['metadatas'][0][i] if query_results['metadatas'] else {}
            })
    print(f"ChromaDB: Tìm thấy {len(results)} kết quả.")
    return results


def get_answer_from_gemini(question, context_chunks, gemini_model_instance, temperature=0.2, queried_student_name=None):
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY chưa được cấu hình.")
        return "Lỗi: API key của Gemini chưa được cấu hình."
    if not gemini_model_instance:
        st.error("Mô hình Gemini (LLM) chưa được tải.")
        return "Lỗi: Mô hình Gemini (LLM) chưa được tải."

    context_str = "Không có thông tin tham khảo nào được tìm thấy."  # Default if no chunks
    if context_chunks:
        context_str = "\n\n---\n\n".join(
            [f"Nguồn {i+1} (ID: {chunk.get('id', 'N/A')}, Loại: {chunk.get('type', chunk.get('metadata', {}).get('type', 'N/A'))}, Course: {chunk.get('course_id', chunk.get('metadata', {}).get('course_id', 'N/A'))}, Score: {chunk.get('score', -1):.4f})\nNội dung: {chunk['content']}"
             for i, chunk in enumerate(context_chunks)]
        )

    specific_student_instruction = ""
    if queried_student_name and any(chunk.get('type') == 'student_list_by_major' for chunk in context_chunks):
        specific_student_instruction = f"""
Đặc biệt quan trọng: Câu hỏi này liên quan đến sinh viên cụ thể '{queried_student_name}'. 
Dựa vào 'Thông tin tham khảo', hãy đảm bảo bạn trích xuất và liệt kê đầy đủ các thông tin sau của sinh viên này nếu có:
- Mã số sinh viên (MSSV)
- Họ và tên đầy đủ
- Địa chỉ email
Nếu các thông tin này có trong một dòng hoặc mục danh sách, hãy trình bày rõ ràng từng thông tin."""

    prompt = f"""Dựa vào các thông tin được cung cấp dưới đây từ tài liệu syllabus và danh sách sinh viên, hãy trả lời câu hỏi sau đây một cách chi tiết và chính xác.

Câu hỏi: {question}

Thông tin tham khảo:
{context_str}

Hướng dẫn trả lời:
- Trả lời bằng tiếng Việt.
- Chỉ dựa vào "Thông tin tham khảo" được cung cấp. Không sử dụng kiến thức bên ngoài trừ khi được yêu cầu rõ ràng.
- Nếu thông tin không có trong ngữ cảnh được cung cấp, hãy nói rõ là "Thông tin không có trong tài liệu tham khảo."
- Nếu câu hỏi liên quan đến nhiều thực thể (ví dụ: nhiều môn học), hãy cố gắng tìm và giải thích mối quan hệ giữa chúng (ví dụ: môn tiên quyết, nội dung liên quan, so sánh) dựa trên thông tin được cung cấp. Nếu không có mối quan hệ nào được tìm thấy trong tài liệu, hãy nêu rõ điều đó.
- Khi liệt kê Learning Outcomes (CLO), hãy đảm bảo liệt kê đầy đủ nếu có thông tin.
- Ưu tiên trích xuất thông tin trực tiếp từ nội dung chunk nếu có thể và chỉ rõ nguồn (ví dụ: "Theo Nguồn X...").
{specific_student_instruction}

Câu trả lời của bạn:"""

    try:
        print("App: Gửi prompt tới Gemini...")
        response = gemini_model_instance.generate_content(prompt)
        print("App: Nhận được phản hồi từ Gemini.")
        return response.text
    except Exception as e:
        st.error(f"Lỗi khi gọi Gemini API: {e}")
        return f"Đã xảy ra lỗi khi cố gắng tạo câu trả lời từ Gemini: {str(e)}"

# --- Streamlit UI ---


def main():
    st.set_page_config(
        layout="wide", page_title="Hỏi Đáp Syllabus FPTU (Gemini)")
    st.title("📚 Hệ Thống Hỏi Đáp Thông Tin Syllabus FPTU")
    st.markdown(
        "Sử dụng AI (Gemini) và Vector Search (FAISS/ChromaDB) để tìm kiếm và giải đáp thắc mắc về chương trình học.")

    loaded_resources = load_all_resources()

    embedding_model = loaded_resources["embedding_model"]
    faiss_index = loaded_resources["faiss_index"]
    faiss_mapping = loaded_resources["faiss_mapping"]
    all_chunks_data = loaded_resources["all_chunks_data"]
    chroma_collection = loaded_resources["chroma_collection"]
    # Lấy mô hình Gemini LLM
    gemini_llm_model = loaded_resources["gemini_llm_model"]

    if not embedding_model or not all_chunks_data or not gemini_llm_model:
        st.error(
            "Không thể tải một hoặc nhiều tài nguyên cần thiết (Embedding model, Chunks data, Gemini LLM). Vui lòng kiểm tra lại.")
        return

    st.sidebar.title("⚙️ Cấu Hình Tìm Kiếm")
    db_option = st.sidebar.radio(
        "Chọn Vector Database:",
        ('FAISS', 'ChromaDB'),
        help="FAISS: Nhanh, tìm kiếm trên vector. ChromaDB: Persistent, có thể query metadata (chưa implement trong UI này)."
    )

    k_results = st.sidebar.slider(
        "Số lượng kết quả tìm kiếm (chunks) mỗi truy vấn:", 1, 1000, 100)
    gemini_temp = st.sidebar.slider(
        "Nhiệt độ Gemini (sáng tạo):", 0.0, 1.0, 0.2, 0.05)

    st.sidebar.markdown("---")

    # Input câu hỏi
    user_question = st.text_area(
        "Nhập câu hỏi của bạn tại đây:",
        height=100,
        placeholder="Ví dụ: Session 1 của môn XYZ có chuẩn đầu ra nào? Môn ABC là tiên quyết của môn nào? Danh sách sinh viên chuyên ngành SE? Thông tin của Nguyễn Văn A?"
    )

    submit_button = st.button("🔍 Gửi Câu Hỏi")

    if submit_button and user_question:
        st.markdown("---")
        st.subheader(f'💬 Câu hỏi của bạn: "{user_question}"')

        original_user_question = user_question  # Giữ lại câu hỏi gốc
        query_for_search_embedding = translate_to_english_if_vietnamese(
            user_question, gemini_llm_model, original_user_question_for_log=user_question)

        retrieved_chunks = []
        search_method_message = ""
        extracted_name_for_gemini_prompt = None  # Khởi tạo để truyền cho Gemini
        # search_logic_flow_debug = [] # Optional: for detailed debug path

        # --- Định nghĩa các pattern Regex ---
        # Pattern cho tên sinh viên (2-4 từ tiếng Việt có viết hoa chữ cái đầu)
        specific_name_pattern = re.compile(
            r"\b([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+){1,3})\b")
        # Pattern cho từ khóa danh sách sinh viên theo ngành
        student_list_major_pattern = re.compile(
            r"(danh sách sinh viên|sinh viên ngành|list of students|students in major|student list for)\s*([A-Z]{2,4})\b", re.IGNORECASE)
        # Pattern cho mã môn học
        course_code_pattern = re.compile(
            r"\b([A-Z]{3,4}[0-9]{3}[a-z]?)\b", re.IGNORECASE)
        # Từ khóa thường đi với câu hỏi về môn học (để phân biệt với tên người)
        course_related_keywords = re.compile(
            r"(môn học|học phần|tiên quyết|syllabus|course code|mã môn|CLO|học gì|nội dung|đề cương|thông tin môn)", re.IGNORECASE)

        # --- Logic tìm kiếm ưu tiên ---
        with st.spinner(f"Đang tìm kiếm thông tin trong {db_option}..."):
            # 1. Ưu tiên tìm kiếm thông tin sinh viên cụ thể bằng tên (scan trực tiếp)
            potential_name_match_in_query = specific_name_pattern.search(
                original_user_question)
            if potential_name_match_in_query and not student_list_major_pattern.search(original_user_question) and not course_related_keywords.search(original_user_question) and not course_code_pattern.search(original_user_question):
                extracted_name = potential_name_match_in_query.group(1).strip()
                print(
                    f"App: Phát hiện truy vấn có thể là tên SV cụ thể: '{extracted_name}'. Thử tìm trực tiếp.")
                # search_logic_flow_debug.append(f"Trying direct student name search for: {extracted_name}")
                if all_chunks_data:
                    for idx, chunk_item in enumerate(all_chunks_data):
                        if chunk_item.get('type') == 'student_list_by_major' and \
                           extracted_name.lower() in chunk_item.get('content', '').lower():
                            retrieved_chunks = [{
                                'id': f"direct_student_name_idx_{idx}",
                                'content': chunk_item.get('content', 'N/A'),
                                'score': 0.0,
                                'metadata': chunk_item.get('metadata', {}),
                                'type': chunk_item.get('type', 'N/A'),
                                'course_id': chunk_item.get('metadata', {}).get('course_id'),
                                'syllabus_id': chunk_item.get('metadata', {}).get('syllabus_id')
                            }]
                            major_name_for_msg = retrieved_chunks[0]['metadata'].get(
                                'major_name', 'N/A')
                            search_method_message = f"Tìm thấy thông tin trực tiếp cho sinh viên '{extracted_name}' trong danh sách của chuyên ngành {major_name_for_msg}."
                            # Lưu tên để dùng cho prompt Gemini
                            extracted_name_for_gemini_prompt = extracted_name
                            # search_logic_flow_debug.append(f"SUCCESS: Found '{extracted_name}' in student list chunk.")
                            print(search_method_message)
                            break
                    if not retrieved_chunks:
                        # search_logic_flow_debug.append(f"FAILED: Direct student name search for '{extracted_name}' found no chunk.")
                        print(
                            f"App: Không tìm thấy '{extracted_name}' trực tiếp trong chunk danh sách sinh viên nào.")

            # 2. Nếu không phải truy vấn SV cụ thể (hoặc tìm không thấy), kiểm tra truy vấn danh sách SV theo ngành
            if not retrieved_chunks:
                student_major_match_result = student_list_major_pattern.search(
                    original_user_question)
                if student_major_match_result:
                    # Cần flag này để không bị nhầm với multi-course/general
                    is_student_list_query_flag = True
                    target_major_code = student_major_match_result.group(
                        2).upper()
                    print(
                        f"App: Phát hiện câu hỏi danh sách sinh viên cho ngành: {target_major_code}. Ưu tiên tìm trực tiếp.")
                    # search_logic_flow_debug.append(f"Trying student list by major (direct scan first): {target_major_code}")

                    # Thử tìm trực tiếp trước
                    if all_chunks_data:
                        for idx, chunk_item in enumerate(all_chunks_data):
                            meta = chunk_item.get('metadata', {})
                            if chunk_item.get('type') == 'student_list_by_major' and meta.get('major_code') == target_major_code:
                                retrieved_chunks = [{
                                    'id': f"direct_major_list_idx_{idx}",
                                    'content': chunk_item.get('content', 'N/A'),
                                    'score': 0.0,
                                    'metadata': meta,
                                    'type': chunk_item.get('type', 'N/A'),
                                    'course_id': meta.get('course_id'),
                                    'syllabus_id': meta.get('syllabus_id')
                                }]
                                search_method_message = f"Đã tìm thấy trực tiếp chunk danh sách sinh viên cho ngành {target_major_code}."
                                # search_logic_flow_debug.append("SUCCESS: Found student list by major directly.")
                                print(search_method_message)
                                break

                    if not retrieved_chunks:  # Nếu tìm trực tiếp không thấy, dùng vector search làm fallback
                        print(
                            f"App: Không tìm thấy DS SV cho ngành {target_major_code} trực tiếp. Dùng vector search...")
                        # search_logic_flow_debug.append(f"FALLBACK: Vector search for student list by major: {target_major_code}")
                        query_embedding_for_list = embedding_model.encode(
                            query_for_search_embedding)  # Encode câu hỏi đã dịch
                        candidate_chunks_vector = []
                        # ... (logic vector search lấy k_results*5 chunks như đã code ở bước trước)
                        # Lấy nhiều ứng viên hơn để lọc sau
                        num_candidates_to_fetch = k_results * 5
                        if db_option == 'FAISS':
                            if faiss_index and faiss_mapping and all_chunks_data:
                                if num_candidates_to_fetch > faiss_index.ntotal:
                                    num_candidates_to_fetch = faiss_index.ntotal
                                distances, indices = faiss_index.search(np.array(
                                    [query_embedding_for_list]).astype('float32'), num_candidates_to_fetch)
                                # ... (xử lý kết quả faiss để điền vào candidate_chunks_vector) ...
                                if indices.size > 0:
                                    for i in range(len(indices[0])):
                                        faiss_id = indices[0][i]
                                        mapped_info = faiss_mapping.get(
                                            faiss_id)
                                        if faiss_id < 0 or not mapped_info:
                                            continue
                                        original_chunk_index = mapped_info.get(
                                            'original_chunk_index')
                                        if original_chunk_index is not None and 0 <= original_chunk_index < len(all_chunks_data):
                                            chunk_data = all_chunks_data[original_chunk_index]
                                            candidate_chunks_vector.append({'id': f"faiss_id_{faiss_id}_orig_idx_{original_chunk_index}", 'content': chunk_data.get('content', 'N/A'), 'score': float(distances[0][i]), 'metadata': chunk_data.get(
                                                'metadata', {}), 'type': chunk_data.get('metadata', {}).get('type', chunk_data.get('type', 'N/A')), 'course_id': chunk_data.get('metadata', {}).get('course_id', mapped_info.get('course_id', 'N/A'))})
                        elif db_option == 'ChromaDB':
                            if chroma_collection:
                                if num_candidates_to_fetch > chroma_collection.count():
                                    num_candidates_to_fetch = chroma_collection.count()
                                query_res_chroma = chroma_collection.query(query_embeddings=[query_embedding_for_list.tolist(
                                )], n_results=num_candidates_to_fetch, include=['documents', 'metadatas', 'distances'])
                                # ... (xử lý kết quả chroma để điền vào candidate_chunks_vector) ...
                                if query_res_chroma and query_res_chroma['ids'] and query_res_chroma['ids'][0]:
                                    for i in range(len(query_res_chroma['ids'][0])):
                                        candidate_chunks_vector.append({'id': query_res_chroma['ids'][0][i], 'content': query_res_chroma['documents'][0][i] if query_res_chroma['documents'] else 'N/A', 'score': float(
                                            query_res_chroma['distances'][0][i]) if query_res_chroma['distances'] else float('inf'), 'metadata': query_res_chroma['metadatas'][0][i] if query_res_chroma['metadatas'] else {}})

                        # Lọc lại từ candidate_chunks_vector
                        student_list_chunks_from_vector = []
                        other_vector_chunks = []
                        for chunk in candidate_chunks_vector:
                            meta_vec = chunk.get('metadata', {})
                            if chunk.get('type') == 'student_list_by_major' and meta_vec.get('major_code') == target_major_code:
                                student_list_chunks_from_vector.append(chunk)
                            else:
                                other_vector_chunks.append(chunk)
                        student_list_chunks_from_vector.sort(
                            key=lambda x: x.get('score', float('inf')))
                        other_vector_chunks.sort(
                            key=lambda x: x.get('score', float('inf')))
                        retrieved_chunks = (
                            student_list_chunks_from_vector + other_vector_chunks)[:k_results]
                        search_method_message = f"Tìm DS SV cho ngành {target_major_code} bằng vector search. {len(student_list_chunks_from_vector)} chunk DS SV khớp. Tổng {len(retrieved_chunks)} chunks."
                        if not student_list_chunks_from_vector:
                            search_method_message += " Không tìm thấy chunk DS SV khớp trực tiếp từ vector search."
                        # search_logic_flow_debug.append(f"Vector search for student list yielded {len(student_list_chunks_from_vector)} specific chunks.")
                else:
                    is_student_list_query_flag = False  # Reset flag if not this type of query

            # 3. Nếu không phải các loại trên, kiểm tra truy vấn đa môn hoặc đơn môn (dùng vector search)
            if not retrieved_chunks:
                query_embedding_for_courses = embedding_model.encode(
                    query_for_search_embedding)  # Đảm bảo đã encode câu hỏi đã dịch

                course_codes_found_in_query = list(
                    # Tìm mã môn trên câu hỏi đã dịch
                    set([match.upper() for match in course_code_pattern.findall(query_for_search_embedding)]))

                # search_logic_flow_debug.append(f"Course codes found in translated query: {course_codes_found_in_query}")

                # is_multi_course_query_flag chỉ True nếu >1 mã môn VÀ KHÔNG PHẢI là truy vấn DS Sinh viên đã được xử lý
                is_multi_course_query_flag = len(
                    course_codes_found_in_query) > 1 and not is_student_list_query_flag

                if is_multi_course_query_flag:
                    # search_logic_flow_debug.append("Handling as multi-course query.")
                    search_method_message = f"Tìm kiếm thông tin cho các môn: {', '.join(course_codes_found_in_query)}.\n"
                    sub_k_fetch = max(k_results * 2, 15)

                    all_sub_query_chunks_collected = []
                    # ... (Toàn bộ logic của is_multi_course_query như cũ, nhưng dùng course_codes_found_in_query và query_embedding_for_courses nếu cần, hoặc sub_query_text_en cho sub-embeddings)
                    # Đảm bảo rằng sub_query_text_en được encode và tìm kiếm
                    # Và kết quả cuối cùng được gán cho retrieved_chunks
                    # ... (Copy và điều chỉnh logic is_multi_course_query từ phiên bản trước vào đây) ...
                    # Ví dụ:
                    processed_sub_ids = set()
                    for course_code_item in course_codes_found_in_query:
                        sub_query_text_en = f"Detailed information about the course {course_code_item}, including its overview, learning objectives, CLOs, reference materials, and most importantly, its prerequisites or any directly related courses to {course_code_item}."
                        sub_query_emb = embedding_model.encode(
                            sub_query_text_en)
                        current_sub_res = []
                        if db_option == 'FAISS':
                            if faiss_index and faiss_mapping and all_chunks_data:
                                current_sub_res = search_faiss(
                                    faiss_index, faiss_mapping, all_chunks_data, sub_query_emb, k=sub_k_fetch)
                        elif db_option == 'ChromaDB':
                            if chroma_collection:
                                current_sub_res = search_chroma(
                                    chroma_collection, sub_query_emb, k=sub_k_fetch)
                        all_sub_query_chunks_collected.extend(current_sub_res)

                    unique_chunks_by_id_dict = {
                        chunk['id']: chunk for chunk in all_sub_query_chunks_collected}
                    prioritized_for_gemini = []
                    temp_processed_ids_multi = set()

                    for code_mc in course_codes_found_in_query:
                        ov_chunk = next((c for c_id, c in unique_chunks_by_id_dict.items() if c.get('metadata', {}).get(
                            'course_id') == code_mc and c.get('type') == 'overview' and c_id not in temp_processed_ids_multi), None)
                        if ov_chunk:
                            prioritized_for_gemini.append(ov_chunk)
                            temp_processed_ids_multi.add(ov_chunk['id'])
                        pr_chunk = next((c for c_id, c in unique_chunks_by_id_dict.items() if c.get('metadata', {}).get(
                            'course_id') == code_mc and c.get('type') == 'prerequisites' and c_id not in temp_processed_ids_multi), None)
                        if pr_chunk:
                            prioritized_for_gemini.append(pr_chunk)
                            temp_processed_ids_multi.add(pr_chunk['id'])

                    remaining_sorted = sorted([c for c_id, c in unique_chunks_by_id_dict.items(
                    ) if c_id not in temp_processed_ids_multi], key=lambda x: x.get('score', float('inf')))
                    for chunk_rem in remaining_sorted:
                        if len(prioritized_for_gemini) >= k_results:
                            break
                        if chunk_rem['id'] not in temp_processed_ids_multi:
                            prioritized_for_gemini.append(chunk_rem)
                            temp_processed_ids_multi.add(chunk_rem['id'])

                    retrieved_chunks = prioritized_for_gemini[:k_results]
                    retrieved_chunks.sort(
                        key=lambda x: x.get('score', float('inf')))
                    search_method_message += f"Đã xử lý {len(course_codes_found_in_query)} sub-queries. {len(retrieved_chunks)} chunks được chọn (giới hạn k={k_results})."

                elif not is_student_list_query_flag:  # Truy vấn đơn lẻ, không phải SV, không phải đa môn
                    # search_logic_flow_debug.append("Handling as general single vector search.")
                    print(
                        f"App: Thực hiện general vector search cho: {query_for_search_embedding}")

                    # Kiểm tra xem có phải là truy vấn cho một môn học duy nhất không
                    if len(course_codes_found_in_query) == 1 and not is_student_list_query_flag and not is_multi_course_query_flag:
                        single_course_code = course_codes_found_in_query[0]
                        # search_logic_flow_debug.append(f"Identified as single course query for: {single_course_code}")
                        search_method_message = f"Tìm kiếm thông tin chi tiết cho môn học: {single_course_code}.\n"
                        print(
                            f"App: Ưu tiên lấy trực tiếp chunks overview, prerequisites, learning_outcomes cho {single_course_code}")

                        directly_fetched_single_course_chunks = []
                        processed_ids_single_course = set()

                        if all_chunks_data:
                            # 1. Lấy overview
                            overview_chunk = next((c for c in all_chunks_data if c.get('metadata', {}).get(
                                'course_id') == single_course_code and c.get('type') == 'overview'), None)
                            if overview_chunk and overview_chunk.get('id') not in processed_ids_single_course:
                                directly_fetched_single_course_chunks.append(
                                    # Ưu tiên cao
                                    {**overview_chunk, 'score': 0.01})
                                processed_ids_single_course.add(
                                    overview_chunk.get('id'))

                            # 2. Lấy prerequisites
                            prereq_chunk = next((c for c in all_chunks_data if c.get('metadata', {}).get(
                                'course_id') == single_course_code and c.get('type') == 'prerequisites'), None)
                            if prereq_chunk and prereq_chunk.get('id') not in processed_ids_single_course:
                                directly_fetched_single_course_chunks.append(
                                    # Ưu tiên cao
                                    {**prereq_chunk, 'score': 0.02})
                                processed_ids_single_course.add(
                                    prereq_chunk.get('id'))

                            # 3. Lấy tất cả learning_outcomes
                            for clo_chunk in all_chunks_data:
                                if clo_chunk.get('metadata', {}).get('course_id') == single_course_code and clo_chunk.get('type') == 'learning_outcome':
                                    if clo_chunk.get('id') not in processed_ids_single_course:
                                        directly_fetched_single_course_chunks.append(
                                            # Ưu tiên cao
                                            {**clo_chunk, 'score': 0.03})
                                        processed_ids_single_course.add(
                                            clo_chunk.get('id'))

                        search_method_message += f"Đã lấy trực tiếp {len(directly_fetched_single_course_chunks)} chunks (overview, prereqs, CLOs). "

                        # 4. Bổ sung bằng vector search nếu cần thêm chunks
                        remaining_k = k_results - \
                            len(directly_fetched_single_course_chunks)
                        if remaining_k > 0:
                            search_method_message += f"Tìm thêm {remaining_k} chunks bằng vector search. "
                            vector_search_results_single_course = []
                            if db_option == 'FAISS':
                                if faiss_index and faiss_mapping and all_chunks_data:
                                    vector_search_results_single_course = search_faiss(
                                        # Lấy k_results ứng viên
                                        faiss_index, faiss_mapping, all_chunks_data, query_embedding_for_courses, k=k_results)
                            elif db_option == 'ChromaDB':
                                if chroma_collection:
                                    vector_search_results_single_course = search_chroma(
                                        chroma_collection, query_embedding_for_courses, k=k_results)  # Lấy k_results ứng viên

                            # Thêm các chunks từ vector search mà chưa có, cho đến khi đủ k_results
                            for v_chunk in vector_search_results_single_course:
                                if len(directly_fetched_single_course_chunks) >= k_results:
                                    break
                                if v_chunk.get('id') not in processed_ids_single_course:
                                    directly_fetched_single_course_chunks.append(
                                        v_chunk)
                                    processed_ids_single_course.add(
                                        v_chunk.get('id'))

                        retrieved_chunks = directly_fetched_single_course_chunks[:k_results]
                        # Sắp xếp lại lần cuối, ưu tiên các chunk lấy trực tiếp, sau đó theo score từ vector search
                        retrieved_chunks.sort(
                            key=lambda x: x.get('score', float('inf')))
                        search_method_message += f"Tổng cộng {len(retrieved_chunks)} chunks được chọn cho Gemini."

                    # General vector search (không phải specific single course, không phải student list, không phải multi-course)
                    else:
                        # search_logic_flow_debug.append("Fallback to general vector search as no specific logic matched.")
                        print(
                            f"App: Thực hiện general vector search (fallback) cho: {query_for_search_embedding}")
                        if db_option == 'FAISS':
                            if faiss_index and faiss_mapping and all_chunks_data:
                                retrieved_chunks = search_faiss(
                                    faiss_index, faiss_mapping, all_chunks_data, query_embedding_for_courses, k=k_results)
                                search_method_message = f"Đã tìm thấy {len(retrieved_chunks)} chunks liên quan bằng FAISS (general search)."
                            else:
                                st.error(
                                    "FAISS chưa được tải hoặc cấu hình đúng.")
                        elif db_option == 'ChromaDB':
                            if chroma_collection:
                                retrieved_chunks = search_chroma(
                                    chroma_collection, query_embedding_for_courses, k=k_results)
                                search_method_message = f"Đã tìm thấy {len(retrieved_chunks)} chunks liên quan bằng ChromaDB (general search)."
                            else:
                                st.error(
                                    "ChromaDB collection chưa được tải đúng.")

            # print(f"App: Search Logic Flow: {' -> '.join(search_logic_flow_debug)}") # Optional debug
            # Nếu không có thông báo nào được đặt và không có chunk
            if not search_method_message and not retrieved_chunks:
                search_method_message = "Không có logic tìm kiếm nào được kích hoạt hoặc không tìm thấy kết quả."
            # Nếu có chunk nhưng không có msg (trường hợp hiếm)
            elif not search_method_message and retrieved_chunks:
                search_method_message = f"Đã tìm thấy {len(retrieved_chunks)} chunks."

        st.info(search_method_message)

        if retrieved_chunks:
            with st.expander("Xem các chunks thông tin được tìm thấy", expanded=False):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(
                        f"**Chunk {i+1} (Score: {chunk.get('score', 'N/A'):.4f})** - ID: `{chunk.get('id', 'N/A')}`")

                    # Lấy thông tin type và course_id một cách nhất quán từ metadata nếu có
                    chunk_type = chunk.get('type', 'N/A')
                    chunk_course_id = 'N/A'
                    if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                        # Ưu tiên type từ metadata nếu có, nếu không thì dùng type ở ngoài
                        chunk_type = chunk['metadata'].get(
                            'type', chunk_type)
                        chunk_course_id = chunk['metadata'].get(
                            'course_id', 'N/A')

                    st.markdown(
                        f"*Loại:* `{chunk_type}` - *Course ID:* `{chunk_course_id}`")
                    st.text_area(
                        f"Nội dung chunk {i+1}", chunk['content'], height=150, key=f"chunk_exp_{i}")
                    st.markdown("---")

        st.markdown("---")
        st.subheader("🤖 Câu Trả Lời từ Gemini:")
        with st.spinner("Gemini đang xử lý và tạo câu trả lời..."):
            start_time = time.time()
            gemini_answer = get_answer_from_gemini(
                # Truyền gemini_llm_model
                user_question,
                retrieved_chunks,
                gemini_llm_model,
                temperature=gemini_temp,
                queried_student_name=extracted_name_for_gemini_prompt  # Truyền tên SV đã trích xuất
            )
            processing_time = time.time() - start_time
            st.markdown(gemini_answer)
            st.caption(
                f"Thời gian Gemini xử lý: {processing_time:.2f} giây")

    elif submit_button and not user_question:
        st.warning("Vui lòng nhập câu hỏi của bạn.")

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"Đang sử dụng file chunks: {ALL_CHUNKS_DATA_FILE}. Model: {EMBEDDING_MODEL_NAME}. Gemini: {GEMINI_MODEL_NAME}.")


if __name__ == "__main__":
    main()
