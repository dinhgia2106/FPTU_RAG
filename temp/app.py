from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
import json
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
import time
from collections import defaultdict

app = Flask(__name__)

# --- Constants (Chuyển từ app_streamlit.py) ---
FAISS_INDEX_FILE = "all_data.faiss"
FAISS_MAPPING_FILE = "all_data_faiss_mapping.json"
ALL_CHUNKS_DATA_FILE = "embedded_all_chunks_with_students.json"  # Quan trọng
CHROMA_PERSIST_DIRECTORY = "all_data_chroma_db_store"
CHROMA_COLLECTION_NAME = "all_syllabus_and_students_collection"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GEMINI_LLM_FOR_RAG_NAME = 'gemini-1.5-flash-latest'  # Model dùng cho RAG và dịch

# --- Global Variables for Loaded Resources ---
# Global dictionary to store loaded resources
INITIALIZATION_ATTEMPTS = 0  # Biến toàn cục để đếm số lần gọi initialize_resources
loaded_resources = {
    "embedding_model": None,
    "faiss_index": None,
    "faiss_mapping": None,
    "all_chunks_data": None,  # Rất quan trọng
    "chroma_collection": None,
    "gemini_llm_model": None,  # Model chính, sẽ được cập nhật bởi logic xoay key
    "gemini_api_keys": [],    # Danh sách các API key của Gemini
    "current_gemini_key_index": 0  # Index của key đang được ưu tiên sử dụng
}
chat_history = []  # Lưu trữ lịch sử chat

# --- Environment and API Key Configuration ---
# Xóa khối cấu hình GEMINI_API_KEY cũ ở đây, việc load key sẽ thực hiện trong initialize_resources

# --- Helper Functions (Ported and Adapted from app_streamlit.py) ---


def is_vietnamese(text):
    """Kiểm tra sơ bộ xem văn bản có chứa ký tự tiếng Việt không."""
    vietnamese_chars = r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    return bool(re.search(vietnamese_chars, text))

# --- Helper function to build structured curriculum context ---


def _build_structured_curriculum_for_major(major_query_term, all_course_chunks):
    """
    Xây dựng thông tin chương trình học có cấu trúc cho một ngành cụ thể từ tất cả các chunk.
    Trả về một chuỗi định dạng, số lượng môn học tìm thấy, và số lượng học kỳ.
    """
    courses_by_semester = defaultdict(set)
    major_query_term_lower = major_query_term.lower()

    print(
        f"[DEBUG CURRICULUM] === Building curriculum for major: '{major_query_term}' ===")
    if not all_course_chunks:
        print("[DEBUG CURRICULUM] No course chunks provided.")
        return "Không có dữ liệu môn học để xử lý.", 0, 0

    all_chunks_count = len(all_course_chunks)
    print(f"[DEBUG CURRICULUM] Processing {all_chunks_count} total chunks.")

    # {course_id: (course_name, semester_str)}
    unique_courses_for_curriculum = {}

    # Log sample of chunk metadata for diagnostics
    # sample_chunks_logged = 0 # Not needed anymore
    for i, chunk_to_sample in enumerate(all_course_chunks):
        if i < 5 or (all_chunks_count - i) <= 5:  # Log first 5 and last 5
            meta_sample = chunk_to_sample.get(
                "meta") or chunk_to_sample.get("metadata", {})
            print(f"[DEBUG CURRICULUM] Sample chunk {i} meta: course_id='{meta_sample.get('course_id', 'N/A')}', course_name='{meta_sample.get('course_name', 'N/A')}', major='{meta_sample.get('major_name', 'N/A')}', semester='{meta_sample.get('semester_from_curriculum', 'N/A')}'")

    for chunk_index, chunk in enumerate(all_course_chunks):
        chunk_meta = chunk.get("meta") or chunk.get("metadata", {})
        course_id = chunk_meta.get('course_id')
        course_name = chunk_meta.get('course_name')

        if not course_id or not course_name:
            # if chunk_index < 20 : # Log if early chunks are missing critical info
            #     print(f"[DEBUG CURRICULUM] Chunk {chunk_index} skipped: missing course_id or course_name. Meta: {chunk_meta}")
            continue

        is_major_match_for_semester_map = False
        major_name_meta_lc = chunk_meta.get('major_name', '').lower()
        course_id_meta_lc = course_id.lower()
        course_name_meta_lc = course_name.lower()

        # Điều kiện để một chunk được coi là thuộc ngành đang truy vấn
        if major_query_term_lower in major_name_meta_lc:
            is_major_match_for_semester_map = True
        elif major_query_term_lower in course_id_meta_lc:  # ví dụ: "ai" trong "AIG202c"
            is_major_match_for_semester_map = True
        elif major_query_term_lower in course_name_meta_lc:  # ví dụ: "ai" trong "Trí tuệ AI"
            is_major_match_for_semester_map = True
        # Heuristic cho trường hợp cụ thể "ai" - có thể mở rộng hoặc điều chỉnh
        # Ví dụ: nếu major_name là "Kỹ thuật phần mềm - CN AI", major_query_term_lower ("ai") sẽ khớp
        elif major_query_term_lower == "ai" and \
            ("trí tuệ nhân tạo" in major_name_meta_lc or
             "artificial intelligence" in major_name_meta_lc or
             major_name_meta_lc.startswith("ai") or
             major_name_meta_lc.endswith("ai")):
            is_major_match_for_semester_map = True

        if is_major_match_for_semester_map:
            semester_from_meta = chunk_meta.get('semester_from_curriculum')
            # Chuẩn hóa giá trị học kỳ
            if semester_from_meta is None or str(semester_from_meta).strip() == '':
                semester_str = "Chưa rõ học kỳ"
            else:
                semester_str = str(semester_from_meta).strip()

            # Lưu trữ thông tin môn học, đảm bảo mỗi môn chỉ xuất hiện một lần với học kỳ ưu tiên (nếu có nhiều chunk)
            current_course_name, current_semester_str = unique_courses_for_curriculum.get(
                course_id, (None, None))

            if not current_course_name:  # Môn này chưa được thêm
                unique_courses_for_curriculum[course_id] = (
                    course_name, semester_str)
            # Ưu tiên học kỳ có số rõ ràng hơn là "Chưa rõ học kỳ"
            elif current_semester_str == "Chưa rõ học kỳ" and semester_str != "Chưa rõ học kỳ":
                unique_courses_for_curriculum[course_id] = (
                    course_name, semester_str)
            # Optional: Ghi log nếu một course_id có nhiều thông tin học kỳ khác nhau (cảnh báo tiềm ẩn)
            # elif current_semester_str != semester_str and semester_str != "Chưa rõ học kỳ":
            # print(f"[DEBUG CURRICULUM] Warning: CourseID '{course_id}' has conflicting semesters: '{current_semester_str}' vs '{semester_str}'")

    # Ghi log các môn học duy nhất đã được xác định cho ngành
    print(
        f"[DEBUG CURRICULUM] Identified {len(unique_courses_for_curriculum)} unique courses for major '{major_query_term}'. Listing up to 20:")
    count = 0
    for c_id, (c_name, sem) in unique_courses_for_curriculum.items():
        if count < 20:
            print(
                f"[DEBUG CURRICULUM]   - CourseID: {c_id}, Name: {c_name}, Semester: {sem}")
            count += 1
        else:
            break
    if len(unique_courses_for_curriculum) > 20:
        print(
            f"[DEBUG CURRICULUM]   ... and {len(unique_courses_for_curriculum) - 20} more courses.")

    # Bây giờ mới gom nhóm các môn học duy nhất theo học kỳ
    for course_id_val, (course_name_val, semester_str_val) in unique_courses_for_curriculum.items():
        courses_by_semester[semester_str_val].add(
            (course_id_val, course_name_val))

    if not courses_by_semester:
        print(
            f"[DEBUG CURRICULUM] No courses found and grouped by semester for major '{major_query_term}'.")
        return f"Xin lỗi, mình không tìm thấy thông tin chi tiết về các môn học theo từng học kỳ cho ngành {major_query_term}.", 0, 0

    # Sắp xếp các học kỳ
    # Chuyển "Học kỳ 1", "Học kỳ 2",... thành số để sort, "Chưa rõ học kỳ" ra sau cùng
    def semester_sort_key(s_key):
        if isinstance(s_key, str):
            numeric_part = ''.join(filter(str.isdigit, s_key))
            if numeric_part:
                # (type, value) - numeric semesters first
                return (0, int(numeric_part))
        if s_key == "Chưa rõ học kỳ":
            return (2, 0)  # "Chưa rõ học kỳ" last
        return (1, str(s_key))  # Other strings in between

    try:
        # Sắp xếp keys của courses_by_semester (các học kỳ)
        # Ví dụ: "1", "2", "N/A", "3" -> sort thành "1", "2", "3", "N/A"
        # Cần đảm bảo "semester_from_curriculum" là số hoặc chuỗi số.

        # Custom sort key:
        # 1. Ưu tiên những key là số nguyên (hoặc chuỗi biểu diễn số nguyên)
        # 2. Các chuỗi khác (không phải "Chưa rõ học kỳ")
        # 3. "Chưa rõ học kỳ" ở cuối cùng

        sorted_semesters_keys = sorted(
            courses_by_semester.keys(), key=semester_sort_key)

    except ValueError as e:
        print(
            f"[DEBUG CURRICULUM] Error sorting semesters: {e}. Semesters found: {list(courses_by_semester.keys())}")
        # Fallback: sắp xếp theo chuỗi thông thường nếu có lỗi
        sorted_semesters_keys = sorted(courses_by_semester.keys())

    curriculum_parts = []
    total_courses_in_curriculum = 0
    for semester_key in sorted_semesters_keys:
        # Sắp xếp môn học trong từng kỳ theo ID
        courses_in_semester = sorted(list(courses_by_semester[semester_key]))
        if courses_in_semester:
            # Đảm bảo semester_key được hiển thị đúng là "Học kỳ X" nếu là số.
            display_semester_key = semester_key
            if semester_key.isdigit():  # Kiểm tra nếu key là một chuỗi số
                display_semester_key = f"Học kỳ {semester_key}"

            curriculum_parts.append(f"**{display_semester_key}**:")
            for course_id_item, course_name_item in courses_in_semester:
                curriculum_parts.append(
                    f"  - {course_id_item}: {course_name_item}")
                total_courses_in_curriculum += 1
            curriculum_parts.append("")  # Thêm dòng trống cho dễ đọc

    if not curriculum_parts:
        print(
            f"[DEBUG CURRICULUM] No curriculum parts generated for major '{major_query_term}'.")
        return f"Dường như không có thông tin cụ thể về các môn học cho ngành {major_query_term} vào lúc này.", 0, 0

    # Sử dụng \n cho join, Gemini sẽ xử lý Markdown
    final_curriculum_str = "\n".join(curriculum_parts)
    num_semesters = len(courses_by_semester)
    print(
        f"[DEBUG CURRICULUM] Built curriculum for '{major_query_term}': {total_courses_in_curriculum} courses across {num_semesters} semesters.")
    # Log \n as \\n for clarity
    print(
        f"[DEBUG CURRICULUM] Final curriculum string for Gemini (first 200 chars): {final_curriculum_str[:200].replace('\n', '\\n')}")
    print(
        f"[DEBUG CURRICULUM] === Finished building curriculum for major: '{major_query_term}' ===")
    return final_curriculum_str, total_courses_in_curriculum, num_semesters

# --- Helper function for Gemini API calls with key rotation ---


def _call_gemini_api_with_rotation(prompt_text, generation_config_dict=None, safety_settings_list=None):
    """
    Calls the Gemini API with the given prompt and handles key rotation on quota errors.
    Returns the response text or an error message string starting with "Lỗi:" or "Rất tiếc,".
    """
    global loaded_resources

    api_keys = loaded_resources.get("gemini_api_keys")
    if not api_keys:
        print("App: _call_gemini_api_with_rotation - Không có API key nào của Gemini được cấu hình.")
        return "Lỗi: Không có API key nào của Gemini khả dụng."

    current_start_index = loaded_resources.get("current_gemini_key_index", 0)
    num_keys = len(api_keys)

    for i in range(num_keys):
        key_index_to_try = (current_start_index + i) % num_keys
        current_api_key = api_keys[key_index_to_try]

        print(
            f"App: Đang thử gọi Gemini API với key index {key_index_to_try}...")
        try:
            genai.configure(api_key=current_api_key)
            model = genai.GenerativeModel(GEMINI_LLM_FOR_RAG_NAME)

            gen_config_obj = None
            if generation_config_dict:
                gen_config_obj = genai.types.GenerationConfig(
                    **generation_config_dict)

            response = model.generate_content(
                prompt_text,
                generation_config=gen_config_obj,
                safety_settings=safety_settings_list
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_str = str(response.prompt_feedback.block_reason)
                print(
                    f"App: Gemini đã chặn phản hồi với key index {key_index_to_try}. Lý do: {block_reason_str}")
                print(
                    f"Safety ratings: {response.prompt_feedback.safety_ratings}")
                return f"Rất tiếc, yêu cầu của bạn không thể được xử lý do các hạn chế về nội dung (key index {key_index_to_try}, lý do: {block_reason_str})."

            loaded_resources["current_gemini_key_index"] = key_index_to_try
            print(
                f"App: Gọi Gemini API thành công với key index {key_index_to_try}.")
            return response.text

        except Exception as e:
            error_str = str(e).lower()
            is_quota_error = (
                "429" in error_str or
                "quota" in error_str or
                "resourceexhausted" in error_str or
                "free_tier_input_token_count" in error_str or
                "rate_limit" in error_str
            )

            print(
                f"App: Lỗi với Gemini API key index {key_index_to_try}: {type(e).__name__} - {e}")
            if is_quota_error:
                if i < num_keys - 1:
                    print(
                        f"App: Lỗi quota với key index {key_index_to_try}. Thử key tiếp theo...")
                    continue
                else:
                    print("App: Tất cả các key đều gặp lỗi quota cho yêu cầu này.")
                    loaded_resources["current_gemini_key_index"] = (
                        key_index_to_try + 1) % num_keys
                    return f"Lỗi: Đã xảy ra lỗi với tất cả API keys do vấn đề quota hoặc rate limit. Chi tiết: {e}"
            else:
                print(
                    f"App: Lỗi không phải quota với key index {key_index_to_try}. Ngừng thử các key khác cho yêu cầu này.")
                return f"Lỗi: Đã xảy ra lỗi (không phải quota/rate limit) khi gọi Gemini API với key index {key_index_to_try}. Chi tiết: {e}"

    print("App: _call_gemini_api_with_rotation - Đã thử tất cả các API key nhưng không thành công.")
    return "Lỗi: Đã thử tất cả các API key của Gemini nhưng không thành công."


def translate_to_english_if_vietnamese(text_to_translate, original_user_question_for_log=""):
    """Dịch văn bản sang tiếng Anh nếu phát hiện là tiếng Việt, sử dụng Gemini với key rotation."""
    # global loaded_resources # Không cần global ở đây nếu chỉ đọc

    if not loaded_resources.get("gemini_api_keys"):
        print("App (Dịch): Lỗi dịch - Không có API key của Gemini được cấu hình.")
        return text_to_translate

    if is_vietnamese(text_to_translate):
        log_query_display = original_user_question_for_log if original_user_question_for_log else text_to_translate
        print(
            f"App (Dịch): Phát hiện văn bản tiếng Việt: '{log_query_display[:100]}...'. Đang dịch sang tiếng Anh...")

        prompt = f'''Translate the following Vietnamese text to English. Provide only the English translation, without any introductory phrases, explanations, or quotation marks. Vietnamese text: "{text_to_translate}"'''

        translated_text_or_error = _call_gemini_api_with_rotation(prompt)

        if translated_text_or_error.startswith("Lỗi:") or translated_text_or_error.startswith("Rất tiếc,"):
            print(
                f"App (Dịch): Lỗi khi dịch sang tiếng Anh: {translated_text_or_error}. Sử dụng văn bản gốc.")
            return text_to_translate
        else:
            translated_text = translated_text_or_error.strip()
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]
            if translated_text.startswith("'") and translated_text.endswith("'"):
                translated_text = translated_text[1:-1]
            print(
                f'''App (Dịch): Đã dịch sang tiếng Anh: "{translated_text[:100]}..."''')
            return translated_text
    return text_to_translate


def get_answer_from_gemini_custom(user_question, context_chunks, history, queried_student_name=None):
    # global loaded_resources # Không cần global ở đây nếu chỉ đọc

    if not loaded_resources.get("gemini_api_keys"):
        return "Lỗi: Không có API key nào của Gemini được cấu hình để tạo câu trả lời."

    context_str = "Không có thông tin tham khảo nào được tìm thấy từ tài liệu FPTU."
    if context_chunks:
        # Lọc bỏ các chunk không có nội dung hoặc nội dung quá ngắn (ví dụ chỉ là ID)
        valid_chunks_content = [chunk.get('content', '').strip(
        ) for chunk in context_chunks if chunk.get('content', '').strip()]
        if valid_chunks_content:
            context_str = "\\n\\n---\\n\\n".join(valid_chunks_content)
        else:
            context_str = "Không có nội dung tham khảo nào phù hợp được tìm thấy từ tài liệu FPTU."

    history_prompt_parts = []
    # Lấy vài lượt chat cuối cùng để làm context, ví dụ 5 cặp (user, model) gần nhất = 10 tin nhắn
    recent_history = history[-10:]
    for message in recent_history:
        role = "Người dùng" if message["role"] == "user" else "Chatbot FPTU"
        history_prompt_parts.append(f"{role}: {message['parts'][0]}")
    chat_history_string = "\\n".join(history_prompt_parts)

    specific_student_instruction = ""
    # Kiểm tra xem 'type' có trong metadata của chunk không
    # Giả sử chunk có dạng: {'content': '...', 'metadata': {'type': 'student_list_by_major', ...}}
    if queried_student_name and any(chunk.get('metadata', {}).get('type') == 'student_list_by_major' for chunk in context_chunks):
        specific_student_instruction = f"""
LƯU Ý ĐẶC BIỆT: Câu hỏi này có thể liên quan đến sinh viên '{queried_student_name}'.
Nếu "Thông tin tham khảo" có chứa danh sách sinh viên, hãy tìm và trích xuất đầy đủ thông tin của sinh viên này (MSSV, Họ tên, Email)."""

    prompt = f"""Bạn là Chatbot FPTU, một trợ lý AI thông minh, thân thiện và chuyên nghiệp của Đại học FPT.
Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng về chương trình học, môn học, và các thông tin liên quan đến sinh viên của Đại học FPT.

QUY TẮC TRẢ LỜI RẤT QUAN TRỌNG:
1.  **DỰA HOÀN TOÀN VÀO "THÔNG TIN THAM KHẢO" ĐƯỢC CUNG CẤP**: Đây là các trích đoạn từ tài liệu chính thức của FPTU. Không sử dụng kiến thức bên ngoài trừ khi được yêu cầu rõ ràng hoặc câu hỏi mang tính chất chung chung không thể tìm thấy trong tài liệu.
2.  **TRẢ LỜI TỰ NHIÊN, THÂN THIỆN**: Sử dụng ngôn ngữ như đang trò chuyện. Xưng "mình" và gọi người dùng là "bạn".
3.  **SỬ DỤNG LỊCH SỬ TRÒ CHUYỆN**: Để hiểu rõ ngữ cảnh và câu hỏi nối tiếp của người dùng. Xem xét "Lịch sử trò chuyện trước đó".
4.  **TRẢ LỜI BẰNG TIẾNG VIỆT.**
5.  **ĐỊNH DẠNG RÕ RÀNG**: Sử dụng markdown (ví dụ: `**in đậm**`, `*in nghiêng*`, gạch đầu dòng như `- Mục 1`, danh sách số như `1. Mục A`) để câu trả lời được rõ ràng, mạch lạc và dễ đọc.
6.  **NẾU KHÔNG CÓ THÔNG TIN**: Nếu "Thông tin tham khảo" không chứa thông tin để trả lời câu hỏi, hãy lịch sự thông báo rằng: "Xin lỗi, mình không tìm thấy thông tin bạn yêu cầu trong tài liệu hiện có của FPTU." hoặc một câu tương tự. Đừng cố bịa thông tin.
7.  **TẬP TRUNG VÀO FPTU**: Khi người dùng hỏi chung chung (ví dụ "AI học gì?"), hãy cố gắng liên hệ và trả lời dựa trên chương trình đào tạo AI tại FPTU nếu "Thông tin tham khảo" có chứa nội dung liên quan đến các môn học hoặc chương trình đào tạo của FPTU. Nếu không có trong tài liệu tham khảo, hãy nói rõ là bạn sẽ cung cấp thông tin tổng quan dựa trên kiến thức chung (nếu được phép) hoặc thông báo không có thông tin trong tài liệu FPTU.
8.  **KHI NGƯỜI DÙNG HỎI CHUNG CHUNG (VÍ DỤ "bạn là ai", "bạn biết gì", "bạn làm được gì"):** Hãy giới thiệu bạn là Chatbot FPTU, được thiết kế để cung cấp thông tin về chương trình học, môn học, và các thông tin liên quan đến sinh viên dựa trên tài liệu của Đại học FPT. Bạn có thể đề cập đến các loại thông tin chính mà bạn có thể cung cấp (ví dụ: thông tin chi tiết về môn học, điều kiện tiên quyết, chuẩn đầu ra). Tránh liệt kê các môn học hoặc thông tin quá chi tiết cụ thể từ "Thông tin tham khảo" trừ khi người dùng hỏi sâu hơn về một chủ đề cụ thể đã được đề cập trong "Thông tin tham khảo". Mục tiêu là một lời giới thiệu tổng quan và mời người dùng đặt câu hỏi cụ thể hơn.
{specific_student_instruction}

Lịch sử trò chuyện trước đó (để bạn hiểu ngữ cảnh):
{chat_history_string}

Thông tin tham khảo (Trích xuất từ tài liệu FPTU - Đây là nguồn chính để bạn trả lời):
---
{context_str}
---

Câu hỏi hiện tại của người dùng: {user_question}

Câu trả lời của Chatbot FPTU (nhớ sử dụng markdown nếu cần và chỉ dựa vào thông tin tham khảo đã cung cấp):"""

    # Cấu hình generation và safety (nếu có, hiện tại đang comment out)
    # generation_config_dict = {"temperature": 0.2}
    generation_config_dict = {}  # Mặc định không có config đặc biệt
    # safety_settings_list = [
    #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    # ]
    safety_settings_list = None  # Mặc định không có safety setting đặc biệt

    bot_response_text = _call_gemini_api_with_rotation(
        prompt_text=prompt,
        generation_config_dict=generation_config_dict if generation_config_dict else None,
        safety_settings_list=safety_settings_list if safety_settings_list else None
    )
    return bot_response_text


def initialize_resources():
    global loaded_resources, GEMINI_API_KEYS, CURRENT_GEMINI_KEY_INDEX, INITIALIZATION_ATTEMPTS
    INITIALIZATION_ATTEMPTS += 1
    print(
        f"App (Initialize Attempt {INITIALIZATION_ATTEMPTS}): Starting initialization...")

    # Nếu đã khởi tạo thành công trước đó, không cần làm lại
    if loaded_resources.get("initialized", False):
        print(
            f"App (Initialize Attempt {INITIALIZATION_ATTEMPTS}): Already initialized. Skipping.")
        return True

    initialization_successful = True
    gemini_keys_loaded = False
    load_dotenv()

    # 1. Load Gemini API Keys and attempt to initialize first model
    api_keys_loaded = []
    for i in range(1, 4):  # GEMINI_API_KEY_1, _2, _3
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            api_keys_loaded.append(key)

    if not api_keys_loaded:
        print("LỖI NGHIÊM TRỌNG: Không có GEMINI_API_KEY_n nào (n=1,2,3) được tìm thấy trong file .env. Chatbot sẽ không thể sử dụng Gemini.")
        loaded_resources["gemini_api_keys"] = []
        loaded_resources["gemini_llm_model"] = None
        initialization_successful = False  # Đây là lỗi nghiêm trọng cho chức năng chính
    else:
        loaded_resources["gemini_api_keys"] = api_keys_loaded
        loaded_resources["current_gemini_key_index"] = 0
        print(f"App: Đã tải {len(api_keys_loaded)} Gemini API key(s).")
        try:
            first_key = loaded_resources["gemini_api_keys"][0]
            genai.configure(api_key=first_key)
            # Model này có thể được sử dụng như một tham chiếu, nhưng _call_gemini_api_with_rotation sẽ tạo instance mới
            loaded_resources["gemini_llm_model"] = genai.GenerativeModel(
                GEMINI_LLM_FOR_RAG_NAME)
            print(
                f"App: Mô hình Gemini LLM chính được cấu hình và khởi tạo ban đầu với key index 0.")
        except Exception as e:
            print(
                f"LƯU Ý: Lỗi khi cấu hình/khởi tạo Gemini với key đầu tiên (index 0): {e}. Các key sẽ được thử xoay vòng khi gọi API.")
            loaded_resources["gemini_llm_model"] = None
            # Vẫn coi là thành công nếu các tài nguyên khác tải được, vì có thể key khác hoạt động

    # 2. Tải mô hình Embedding
    # Tiếp tục tải các tài nguyên khác ngay cả khi key Gemini đầu tiên lỗi, vì key rotation có thể cứu vãn
    try:
        print(
            f"App: Đang tải mô hình Sentence Transformer ({EMBEDDING_MODEL_NAME})...")
        loaded_resources["embedding_model"] = SentenceTransformer(
            EMBEDDING_MODEL_NAME)
        print("App: Tải mô hình Sentence Transformer thành công.")
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG: Không thể tải mô hình embedding: {e}.")
        initialization_successful = False  # Embedding model là thiết yếu

    # 3. Tải dữ liệu chunks đầy đủ
    if not os.path.exists(ALL_CHUNKS_DATA_FILE):
        print(
            f"LỖI NGHIÊM TRỌNG: File chunks chính '{ALL_CHUNKS_DATA_FILE}' không tìm thấy.")
        initialization_successful = False
    else:
        try:
            with open(ALL_CHUNKS_DATA_FILE, 'r', encoding='utf-8') as f:
                loaded_resources["all_chunks_data"] = json.load(f)
            print(
                f"App: Đã tải {len(loaded_resources['all_chunks_data'])} chunks từ '{ALL_CHUNKS_DATA_FILE}'.")
        except Exception as e:
            print(
                f"LỖI NGHIÊM TRỌNG: Lỗi tải file '{ALL_CHUNKS_DATA_FILE}': {e}")
            initialization_successful = False

    # 4. Tải FAISS index và mapping (Nếu có)
    if initialization_successful:
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_MAPPING_FILE):
            if loaded_resources.get("all_chunks_data") is None:
                print(f"LỖI: Không thể tải FAISS vì all_chunks_data chưa được tải.")
                loaded_resources["faiss_index"] = None
                loaded_resources["faiss_mapping"] = None
            else:
                try:
                    loaded_resources["faiss_index"] = faiss.read_index(
                        FAISS_INDEX_FILE)
                    with open(FAISS_MAPPING_FILE, 'r', encoding='utf-8') as f:
                        faiss_mapping_raw = json.load(f)
                    loaded_resources["faiss_mapping"] = {
                        int(k): v for k, v in faiss_mapping_raw.items()}
                    print(
                        f"App: Đã tải FAISS Index ({loaded_resources['faiss_index'].ntotal} vectors).")
                except Exception as e:
                    print(f"Lỗi khi tải FAISS (sẽ bỏ qua FAISS): {e}")
                    loaded_resources["faiss_index"] = None
                    loaded_resources["faiss_mapping"] = None
        else:
            print(
                f"Thông báo: Không tìm thấy file FAISS index (\'{FAISS_INDEX_FILE}\') hoặc mapping (\'{FAISS_MAPPING_FILE}\'). Sẽ không sử dụng FAISS.")
            loaded_resources["faiss_index"] = None
            loaded_resources["faiss_mapping"] = None

    # 5. Kết nối ChromaDB (Nếu có)
    if initialization_successful:
        if os.path.exists(CHROMA_PERSIST_DIRECTORY):
            try:
                client = chromadb.PersistentClient(
                    path=CHROMA_PERSIST_DIRECTORY)
                loaded_resources["chroma_collection"] = client.get_collection(
                    name=CHROMA_COLLECTION_NAME)
                print(
                    f"App: Đã kết nối ChromaDB Collection \'{CHROMA_COLLECTION_NAME}\' ({loaded_resources['chroma_collection'].count()} items).")
            except Exception as e:
                print(f"Lỗi khi kết nối ChromaDB (sẽ bỏ qua ChromaDB): {e}")
                loaded_resources["chroma_collection"] = None
        else:
            print(
                f"Thông báo: Không tìm thấy thư mục ChromaDB (\'{CHROMA_PERSIST_DIRECTORY}\'). Sẽ không sử dụng ChromaDB.")
            loaded_resources["chroma_collection"] = None

    if initialization_successful:
        if loaded_resources.get("gemini_api_keys"):
            print("App: Khởi tạo các tài nguyên cần thiết hoàn tất.")
        else:  # Trường hợp không có key Gemini nào được load
            print(
                "App: Khởi tạo tài nguyên hoàn tất nhưng KHÔNG CÓ KEY GEMINI nào, chatbot sẽ không hoạt động.")
    else:
        print("App: Khởi tạo tài nguyên thất bại. Một số chức năng có thể không hoạt động.")

    # Sau khi tất cả tài nguyên đã được tải
    loaded_resources["initialized"] = True
    print(
        f"App (Initialize Attempt {INITIALIZATION_ATTEMPTS}): Đã tải xong tất cả tài nguyên. `loaded_resources['initialized']` đã được đặt thành True.")
    print(
        f"App (Initialize Attempt {INITIALIZATION_ATTEMPTS}): Keys in loaded_resources after init: {list(loaded_resources.keys())}")
    return True

# --- Search Functions (Ported from app_streamlit.py) ---


def search_faiss(index, id_to_chunk_info, all_chunks_data_global, query_embedding, k=5):
    if not index or not id_to_chunk_info or not all_chunks_data_global:
        print("FAISS Search: FAISS chưa sẵn sàng hoặc thiếu dữ liệu (all_chunks_data).")
        return []

    # print(f"FAISS: Chuẩn bị tìm kiếm với k={k}...")
    # Đảm bảo k không lớn hơn số vector trong index
    if k > index.ntotal:
        # print(f"FAISS: Điều chỉnh k từ {k} xuống {index.ntotal} vì index chỉ có {index.ntotal} vectors.")
        k = index.ntotal

    if k == 0:
        return []  # Không có gì để tìm

    distances, indices = index.search(
        np.array([query_embedding]).astype('float32'), k)
    results = []
    if indices.size > 0:
        for i in range(len(indices[0])):
            faiss_id = indices[0][i]
            if faiss_id < 0:  # Invalid index from FAISS
                # print(f"FAISS: Bỏ qua faiss_id không hợp lệ: {faiss_id}")
                continue

            mapped_info = id_to_chunk_info.get(faiss_id)
            if mapped_info:
                original_chunk_index = mapped_info.get('original_chunk_index')
                # Kiểm tra kỹ original_chunk_index
                if original_chunk_index is not None and \
                   isinstance(original_chunk_index, int) and \
                   0 <= original_chunk_index < len(all_chunks_data_global):

                    chunk_data = all_chunks_data_global[original_chunk_index]
                    result_id = chunk_data.get(
                        'id', f"faiss_idx_{faiss_id}_orig_idx_{original_chunk_index}")

                    # Lấy metadata và type từ chunk_data, ưu tiên metadata bên trong
                    metadata = chunk_data.get('metadata', {})
                    chunk_type = metadata.get(
                        'type', chunk_data.get('type', 'N/A'))
                    course_id = metadata.get('course_id', chunk_data.get(
                        'course_id', mapped_info.get('course_id', 'N/A')))

                    results.append({
                        'id': result_id,
                        'content': chunk_data.get('content', 'N/A'),
                        # Đây là distance, score thấp hơn là tốt hơn
                        'score': float(distances[0][i]),
                        'metadata': metadata,
                        'type': chunk_type,
                        'course_id': course_id
                    })
                else:
                    print(
                        f"FAISS: original_chunk_index không hợp lệ ({original_chunk_index}) cho FAISS ID {faiss_id} hoặc all_chunks_data_global không đúng.")
            else:
                print(
                    f"FAISS: Không tìm thấy mapping cho FAISS ID {faiss_id}.")
    # print(f"FAISS: Tìm thấy {len(results)} kết quả.")
    return results


def search_chroma(collection, query_embedding, k=5):
    if not collection:
        print("ChromaDB Search: collection chưa sẵn sàng.")
        return []

    # Đảm bảo k không lớn hơn số item trong collection
    collection_count = collection.count()
    if k > collection_count:
        # print(f"ChromaDB: Điều chỉnh k từ {k} xuống {collection_count} vì collection chỉ có {collection_count} items.")
        k = collection_count

    if k == 0:
        return []  # Không có gì để tìm

    # print(f"ChromaDB: Đang tìm kiếm với k={k}...")
    try:
        query_results = collection.query(
            # Chroma cần list của list
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
    except Exception as e:
        print(f"Lỗi khi query ChromaDB: {e}")
        return []

    results = []
    if query_results and query_results['ids'] and query_results['ids'][0]:
        for i in range(len(query_results['ids'][0])):
            doc_id = query_results['ids'][0][i]
            doc_content = query_results['documents'][0][i] if query_results[
                'documents'] and query_results['documents'][0] else 'N/A'
            doc_distance = float(
                query_results['distances'][0][i]) if query_results['distances'] and query_results['distances'][0] else float('inf')
            metadata = query_results['metadatas'][0][i] if query_results['metadatas'] and query_results['metadatas'][0] else {
            }

            chunk_type = metadata.get('type', 'N/A')
            course_id = metadata.get('course_id', 'N/A')

            results.append({
                'id': doc_id,
                'content': doc_content,
                'score': doc_distance,  # Đây là distance, score thấp hơn là tốt hơn
                'metadata': metadata,
                'type': chunk_type,
                'course_id': course_id
            })
    # print(f"ChromaDB: Tìm thấy {len(results)} kết quả.")
    return results

# --- Flask Routes ---


@app.route('/')
def index_page():  # Đổi tên hàm để tránh trùng với biến index trong Python
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    global chat_history, loaded_resources

    # Debugging: Log the type and content of loaded_resources
    print(
        f"App (Debug Chat Endpoint Entry): Type of loaded_resources: {type(loaded_resources)}")
    if isinstance(loaded_resources, dict):
        print(
            f"App (Debug Chat Endpoint Entry): Keys in loaded_resources: {list(loaded_resources.keys())}")
        print(
            f"App (Debug Chat Endpoint Entry): 'initialized' in loaded_resources: {'initialized' in loaded_resources}")
        if 'initialized' in loaded_resources:  # Safe check before direct access for logging
            # Use .get for safety in log too
            print(
                f"App (Debug Chat Endpoint Entry): Value of loaded_resources['initialized']: {loaded_resources.get('initialized')}")
    else:
        print(
            f"App (Debug Chat Endpoint Entry): loaded_resources is not a dict. Value: {str(loaded_resources)[:500]}")

    # Sử dụng .get() để tránh KeyError nếu "initialized" chưa có (mặc dù nên có)
    # This line should be robust against KeyError
    if not loaded_resources.get("initialized", False):
        # Log why this condition is met
        actual_value_for_log = "N/A"
        if isinstance(loaded_resources, dict):
            actual_value_for_log = loaded_resources.get(
                "initialized", "Key Missing")  # Get actual value or "Key Missing"
        else:
            actual_value_for_log = f"Not a dict, type is {type(loaded_resources)}"
        print(
            f"App (Chat Endpoint): Resources not initialized or check failed. Condition `not loaded_resources.get('initialized', False)` is True. Actual 'initialized' value/status: {actual_value_for_log}")
        return jsonify({'response': "Xin lỗi, các tài nguyên chưa được tải xong. Vui lòng thử lại sau ít phút."}), 503

    data = request.json
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'response': 'Tin nhắn không được để trống.'}), 400

    chat_history.append({'role': 'user', 'parts': [user_message]})

    # Initialize and translate message EARLIER
    translated_user_message = None
    original_lang_is_vietnamese = is_vietnamese(user_message)
    if original_lang_is_vietnamese:
        print(
            f"App (Dịch): Phát hiện văn bản tiếng Việt: '{user_message[:100]}...'. Đang dịch sang tiếng Anh...")
        translated_user_message = translate_to_english_if_vietnamese(
            user_message, original_user_question_for_log=user_message)
        if translated_user_message:
            print(
                f"App (Dịch): Đã dịch sang tiếng Anh: '{translated_user_message[:100]}...'")
        else:
            print(
                f"App (Dịch): Dịch sang tiếng Anh thất bại hoặc không cần thiết cho: '{user_message[:100]}...'. Sử dụng tin nhắn gốc.")
    else:
        print(
            f"App (Dịch): Tin nhắn không phải tiếng Việt hoặc không phát hiện được: '{user_message[:100]}...'. Không dịch.")
        # Nếu không phải tiếng Việt, translated_user_message có thể được gán bằng user_message nếu muốn xử lý nhất quán sau này
        # Hoặc để là None và chỉ dùng user_message cho các logic không cần dịch thuật.
        # For now, if not Vietnamese, translated_user_message remains None, and downstream logic should handle it or use user_message.

    # --- Intent Detection ---
    # (Moved curriculum check and other intent detections after translation setup)

    # Detect curriculum-specific questions (e.g., "Chương trình học ngành AI")
    # This pattern is CRITICAL for routing to _build_structured_curriculum_for_major
    curriculum_major_pattern_str = r"(chương trình học|khung chương trình|lộ trình học|học những gì|học môn gì|bao gồm môn gì)\\s*(của)?\\s*(ngành|chuyên ngành)?\\s*(?P<major_name>[\\w\\s\\-\\&\\#\\+]+)\\s*(gồm những gì|như thế nào|có những môn nào|học kỳ\\s*\\d+.*)?"
    # Ensure to escape special characters for regex if any present in major_name_keywords
    # For now, it's simple words.
    # The (?P<major_name>...) part should capture the major name.
    curriculum_major_pattern = re.compile(
        curriculum_major_pattern_str, re.IGNORECASE)

    is_curriculum_query = False
    major_name_from_query = None
    # Use translated_user_message for regex matching as patterns are more English-aligned or general
    # But the original Vietnamese query might contain specific major names better.
    # For "Chương trình học ngành AI", translated_user_message is "AI curriculum"

    print(
        f"App (Curriculum Check): Attempting to match curriculum pattern on user_message: '{user_message}'")
    # Log the compiled pattern
    print(
        f"App (Curriculum Check): Using pattern: {curriculum_major_pattern.pattern}")
    match_curriculum = curriculum_major_pattern.search(
        user_message)  # CHECKING ORIGINAL MESSAGE

    if match_curriculum:
        major_name_from_query = match_curriculum.group("major_name")
        print(
            f"App (Curriculum Check): Match found on original. Raw major_name captured: '{major_name_from_query}'")
        if major_name_from_query:
            major_name_from_query = major_name_from_query.strip()
            # Further clean common stop words if necessary, e.g., "ngành", "chuyên ngành"
            major_name_from_query = re.sub(
                r"^(ngành|chuyên ngành)\\s+", "", major_name_from_query, flags=re.IGNORECASE).strip()

            if major_name_from_query:  # Check if it's not empty after cleaning
                print(
                    f"App (Curriculum Query): Detected curriculum query for major (from original, cleaned): '{major_name_from_query}' based on user_message: '{user_message}'")
                is_curriculum_query = True
            else:
                print(
                    f"App (Curriculum Query): Major name became empty after cleaning. Original capture: '{match_curriculum.group('major_name')}'")
        else:  # Pattern matched but no major_name captured, or major_name_from_query is empty/None
            print(
                f"App (Curriculum Check): Match found on original, but 'major_name' group is None or empty. Captured value: '{major_name_from_query}'")
            # This can happen if the major_name part of regex is too restrictive or query is simple like "Chương trình học"
    else:
        print(
            f"App (Curriculum Check): NO match for curriculum pattern on user_message: '{user_message}'")
        # Try matching on translated message if original failed to capture group or match
        if translated_user_message:  # Ensure translated_user_message is not None
            print(
                f"App (Curriculum Check): Attempting to match curriculum pattern on translated_user_message: '{translated_user_message}'")
            match_translated_curriculum = curriculum_major_pattern.search(
                translated_user_message)
            if match_translated_curriculum:
                major_name_from_translated = match_translated_curriculum.group(
                    "major_name")
                print(
                    f"App (Curriculum Check): Match found on translated. Raw major_name captured: '{major_name_from_translated}'")
                if major_name_from_translated:
                    major_name_from_query = major_name_from_translated.strip()
                    # Further clean common stop words if necessary
                    major_name_from_query = re.sub(
                        r"^(major|program|specialization|field of|curriculum of|course of|program of)\\s+", "", major_name_from_query, flags=re.IGNORECASE).strip()
                    # Remove trailing "curriculum", "program", "major"
                    major_name_from_query = re.sub(
                        r"\\s+(curriculum|program|major)$", "", major_name_from_query, flags=re.IGNORECASE).strip()

                    if major_name_from_query:  # Check if not empty after cleaning
                        print(
                            f"App (Curriculum Query): Detected curriculum query for major (from translated, cleaned): '{major_name_from_query}' based on translated_user_message: '{translated_user_message}'")
                        is_curriculum_query = True
                    else:
                        print(
                            f"App (Curriculum Query): Major name from translated became empty after cleaning. Original capture: '{major_name_from_translated}'")
                else:
                    print(
                        f"App (Curriculum Check): Match found on translated, but 'major_name' group is None or empty. Captured value: '{major_name_from_translated}'")
            else:
                print(
                    f"App (Curriculum Check): NO match for curriculum pattern on translated_user_message: '{translated_user_message}'")
        else:
            print(f"App (Curriculum Check): Translated user message is None or empty, skipping curriculum check on it.")

        # Fallback heuristic if regex fails for both original and translated
        if not is_curriculum_query:
            print(f"App (Curriculum Check): Regex failed for both original and translated. Attempting fallback heuristic.")
            # Heuristic: Check for keywords in original and translated messages
            original_contains_keywords = any(kw in user_message.lower() for kw in [
                                             "chương trình học", "lộ trình học", "khung chương trình", "học những gì"])
            translated_contains_keywords = translated_user_message and any(
                kw in translated_user_message.lower() for kw in ["curriculum", "program", "study plan", "what to study"])

            if original_contains_keywords or translated_contains_keywords:
                print(
                    f"App (Curriculum Check): Keyword heuristic: Original keywords: {original_contains_keywords}, Translated keywords: {translated_contains_keywords}")
                # Attempt to extract potential major keywords if specific terms like "AI", "SE" are present
                # This is a simplified heuristic and might need a list of known major acronyms/names.
                # For "AI curriculum", translated_user_message is "AI curriculum" -> "AI"
                # For "Chương trình học ngành Trí tuệ nhân tạo" -> "Trí tuệ nhân tạo"

                # Try to extract from user_message by removing keywords
                temp_major_original = user_message.lower()
                for kw in ["chương trình học", "lộ trình học", "khung chương trình", "học những gì", "của ngành", "ngành", "của chuyên ngành", "chuyên ngành", "là gì", "gồm những gì", "như thế nào", "có những môn nào"]:
                    temp_major_original = temp_major_original.replace(kw, "")
                temp_major_original = temp_major_original.strip()

                # Try to extract from translated_user_message by removing keywords
                temp_major_translated = ""
                if translated_user_message:
                    temp_major_translated = translated_user_message.lower()
                    for kw in ["curriculum", "program", "study plan", "what to study", "of the major", "major", "of the specialization", "specialization", "what is", "what are", "how is"]:
                        temp_major_translated = temp_major_translated.replace(
                            kw, "")
                    temp_major_translated = temp_major_translated.strip()

                # Prioritize a more specific extraction if possible
                # A simple check: if the remaining string is short and potentially an acronym or known major.
                # For now, let's prefer the cleaned original if it's not too long.
                # e.g., "trí tuệ nhân tạo" or "AI"
                if temp_major_original and len(temp_major_original.split()) <= 3:
                    major_name_from_query = temp_major_original
                    print(
                        f"App (Curriculum Query): Heuristic fallback for major (from original): '{major_name_from_query}'")
                    is_curriculum_query = True
                # e.g., "artificial intelligence" or "AI"
                elif temp_major_translated and len(temp_major_translated.split()) <= 3:
                    major_name_from_query = temp_major_translated
                    print(
                        f"App (Curriculum Query): Heuristic fallback for major (from translated): '{major_name_from_query}'")
                    is_curriculum_query = True
                else:
                    print(
                        f"App (Curriculum Query): Fallback heuristic failed to confidently determine major. Original cleaned: '{temp_major_original}', Translated cleaned: '{temp_major_translated}'")
            else:
                print(
                    f"App (Curriculum Check): Fallback heuristic: No relevant keywords found in original or translated messages.")

    # Final check on major_name_from_query before proceeding
    if is_curriculum_query and not major_name_from_query:
        print(f"App (Curriculum Warning): 'is_curriculum_query' is true, but 'major_name_from_query' is empty or None. Resetting to non-curriculum query.")
        is_curriculum_query = False

    if is_curriculum_query and major_name_from_query:
        # If major_name_from_query is very generic like "information technology", "công nghệ thông tin"
        # and the original query mentioned a specialization like "AI", "An Toàn Thông Tin",
        # try to use the more specific one if available in metadata of _build_structured_curriculum_for_major
        # For now, we pass the extracted major_name_from_query directly.

        print(
            f"App (Curriculum Route): Routing to structured curriculum for major: '{major_name_from_query}'")
        # If using all_chunks_data which should be already loaded
        structured_curriculum_context, courses_found_count, num_semesters = _build_structured_curriculum_for_major(
            major_name_from_query, loaded_resources["all_chunks_data"])

        if structured_curriculum_context and courses_found_count > 0:
            print(
                f"App: Đã tạo context CT học có cấu trúc cho '{major_name_from_query}' với {courses_found_count} môn.")

            # Lấy lịch sử chat để đưa vào prompt
            history_prompt_parts = []
            # Lấy 10 tin nhắn gần nhất (5 lượt)
            recent_history = chat_history[-10:]
            for message_hist in recent_history:
                role = "Người dùng" if message_hist["role"] == "user" else "Chatbot FPTU"
                history_prompt_parts.append(
                    f"{role}: {message_hist['parts'][0]}")
            chat_history_string = "\n".join(history_prompt_parts)

            final_prompt_for_gemini = f"""Bạn là Chatbot FPTU, một trợ lý AI thông minh và thân thiện.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng về chương trình học Đại học FPT.

Lịch sử trò chuyện trước đó:
{chat_history_string}

Yêu cầu hiện tại của người dùng: {user_message}

Dựa vào thông tin chương trình học được cung cấp dưới đây cho ngành '{major_name_from_query}', hãy trình bày cho người dùng chương trình học của ngành này theo từng học kỳ.
Liệt kê rõ ràng các môn học (mã môn và tên môn) trong mỗi học kỳ.
Sử dụng tiếng Việt và định dạng markdown cho dễ đọc.
Nếu một học kỳ có nhiều môn, hãy liệt kê mỗi môn trên một dòng mới bắt đầu bằng dấu gạch ngang.

Thông tin tham khảo có cấu trúc về chương trình học ngành '{major_name_from_query}':
---
{structured_curriculum_context}
---

Câu trả lời của Chatbot FPTU:"""

            bot_response_text = _call_gemini_api_with_rotation(
                final_prompt_for_gemini)

        else:  # Không xây dựng được context có cấu trúc
            print(
                f"App: Không thể tạo context CT học cấu trúc cho '{major_name_from_query}'. Phản hồi mặc định.")
            bot_response_text = f"Xin lỗi, mình chưa tìm thấy thông tin chi tiết theo từng học kỳ cho chương trình học ngành '{major_name_from_query}'. Mình có thể giúp gì khác cho bạn về ngành này không?"

        chat_history.append({'role': 'model', 'parts': [bot_response_text]})
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        return jsonify({'response': bot_response_text})

    # --- Xử lý các câu hỏi đặc biệt (Liệt kê môn học, Sinh viên) ---
    if "liệt kê tất cả các môn học" in user_message.lower() or \
       "cho tôi danh sách tất cả môn học" in user_message.lower() or \
       "kể tên các môn học bạn biết" in user_message.lower() or \
       (translated_user_message and ("list all subjects" in translated_user_message.lower() or
                                     "list all courses" in translated_user_message.lower())):
        print("App: Phát hiện truy vấn liệt kê tất cả môn học.")
        # ... (existing logic for listing all subjects, seems okay)
        # Ensure this block returns jsonify to skip further processing
        known_courses = {}
        for chunk in loaded_resources["all_chunks_data"]:
            meta = chunk.get('metadata', {})
            course_id = meta.get('course_id')
            course_name = meta.get('course_name', meta.get('syllabus_name'))
            if course_id:
                display_name = course_name if course_name else course_id
                if course_id not in known_courses or (course_name and known_courses[course_id] == course_id):
                    known_courses[course_id] = display_name
        if known_courses:
            sorted_courses = sorted(known_courses.items())
            course_list_str = "\n".join(
                [f"- {cid} ({cname})" if cname != cid and cname else f"- {cid}" for cid, cname in sorted_courses])
            context_for_list_all = f"Dưới đây là danh sách các mã môn học và tên môn học (nếu có) mà tôi được cung cấp thông tin trong tài liệu FPTU:\n{course_list_str}\n Hãy trình bày danh sách này một cách rõ ràng cho người dùng."
            virtual_chunk_for_list_all = [
                {'content': context_for_list_all, 'id': 'virtual_course_list', 'score': 0.0, 'metadata': {'type': 'course_list_summary'}}]
            bot_response_text = get_answer_from_gemini_custom(
                user_question="Hãy giúp tôi liệt kê tất cả các môn học bạn có thông tin từ FPTU.",
                context_chunks=virtual_chunk_for_list_all,
                history=chat_history[:-1]
            )
            chat_history.append(
                {'role': 'model', 'parts': [bot_response_text]})
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
            return jsonify({'response': bot_response_text})
        else:
            bot_response_text = "Xin lỗi, mình đã thử tìm nhưng hiện tại chưa thể tổng hợp được danh sách các môn học từ dữ liệu được cung cấp."
            chat_history.append(
                {'role': 'model', 'parts': [bot_response_text]})
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
            return jsonify({'response': bot_response_text})

    # --- Fallback to General RAG if no special handling matched or produced a response ---
    print("App (Vector Search Fallback): No special query matched or direct search failed. Proceeding to vector search.")

    # Sử dụng translated_user_message cho vector search nếu có, ngược lại dùng user_message
    query_for_embedding = translated_user_message if translated_user_message else user_message
    if not query_for_embedding:
        print("App Error: query_for_embedding is empty after translation logic.")
        return jsonify({'response': "Có lỗi xảy ra với truy vấn của bạn. Vui lòng thử lại."}), 500

    print(
        f"App (Vector Search): query='{query_for_embedding[:100]}...', k={k_search_results}")
    query_embedding = loaded_resources["embedding_model"].encode(
        query_for_embedding).tolist()

    # Direct student name scan
    specific_name_pattern = re.compile(
        r"\\b([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:\\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+){1,3})\\b")
    student_list_major_pattern = re.compile(
        r"(danh sách sinh viên|sinh viên ngành|list of students|students in major|student list for)\\s*([A-Z]{2,4})\\b", re.IGNORECASE)
    course_code_pattern = re.compile(
        r"\\b([A-Z]{3,4}[0-9]{3}[a-z]?)\\b", re.IGNORECASE)
    course_related_keywords = re.compile(
        r"(môn học|học phần|tiên quyết|syllabus|course code|mã môn|CLO|học gì|nội dung|đề cương|thông tin môn)", re.IGNORECASE)

    potential_name_match_in_query = specific_name_pattern.search(user_message)
    is_student_list_query_flag = False  # Reset for this request

    if potential_name_match_in_query and not student_list_major_pattern.search(user_message) and not course_related_keywords.search(user_message) and not course_code_pattern.search(user_message):
        extracted_name = potential_name_match_in_query.group(1).strip()
        print(
            f"App: Phát hiện truy vấn có thể là tên SV cụ thể: '{extracted_name}'. Thử tìm trực tiếp.")
        for idx, chunk_item in enumerate(loaded_resources["all_chunks_data"]):
            chunk_type_meta = chunk_item.get('metadata', {})
            if chunk_type_meta == 'student_list_by_major' and extracted_name.lower() in chunk_item.get('content', '').lower():
                retrieved_chunks = [
                    {**chunk_item, 'score': 0.0, 'id': chunk_item.get('id', f"direct_student_{idx}")}]
                major_name_for_msg = retrieved_chunks[0].get(
                    'metadata', {}).get('major_name', 'N/A')
                search_method_message = f"Tìm thấy TT trực tiếp cho SV '{extracted_name}' trong DS ngành {major_name_for_msg}."
                extracted_name_for_gemini_prompt = extracted_name
                print(f"App: {search_method_message}")
                break
        if not retrieved_chunks:
            print(
                f"App: Không tìm thấy '{extracted_name}' trực tiếp. Sẽ dùng vector search nếu cần.")

    # Direct student list by major scan
    if not retrieved_chunks:
        student_major_match_result = student_list_major_pattern.search(
            user_message)
        if student_major_match_result:
            is_student_list_query_flag = True
            target_major_code = student_major_match_result.group(2).upper()
            print(
                f"App: Phát hiện câu hỏi DS SV cho ngành: {target_major_code}. Thử tìm trực tiếp.")
            for idx, chunk_item in enumerate(loaded_resources["all_chunks_data"]):
                meta = chunk_item.get('metadata', {})
                chunk_type_meta = meta.get('type', chunk_item.get('type'))
                if chunk_type_meta == 'student_list_by_major' and meta.get('major_code', '').upper() == target_major_code:
                    retrieved_chunks = [
                        {**chunk_item, 'score': 0.0, 'id': chunk_item.get('id', f"direct_major_list_{idx}")}]
                    search_method_message = f"Đã tìm thấy trực tiếp chunk DS SV cho ngành {target_major_code}."
                    print(f"App: {search_method_message}")
                    break
            if not retrieved_chunks:
                print(
                    f"App: Không tìm thấy DS SV cho ngành {target_major_code} trực tiếp. Sẽ dùng vector search.")
                search_method_message = f"Chuẩn bị tìm DS SV cho ngành {target_major_code} bằng vector search."

    # --- Vector Search if no direct results or for other query types ---
    if not retrieved_chunks:  # Only proceed to vector search if no direct scan yielded results
        query_embedding = embedding_model.encode(query_for_embedding)

        k_search_results = 7  # Default k value for general vector search
        current_query_embedding_to_use = query_embedding
        refined_query_text_for_log = query_for_embedding

        # Determine k_search_results for non-curriculum, non-direct-scan queries
        general_intro_questions = [
            "bạn là ai", "ban la ai", "bạn biết gì", "ban biet gi",
            "bạn làm được gì", "ban lam duoc gi", "help", "giúp với", "trợ giúp",
            "giới thiệu bản thân", "bạn có thể làm gì"
        ]
        is_general_intro_query = any(
            phrase in user_message.lower() for phrase in general_intro_questions)

        if is_general_intro_query:
            k_search_results = 0
            search_method_message = "Câu hỏi giới thiệu chung (k=0). "
        elif is_student_list_query_flag:  # If it was flagged as student list but direct scan failed
            k_search_results = 5
            search_method_message = f"Sử dụng vector search cho DS SV (k={k_search_results}). "
        elif len(course_code_pattern.findall(query_for_embedding)) > 1:
            k_search_results = 15  # Reduced from 20
            search_method_message = f"Sử dụng vector search cho nhiều môn (k={k_search_results}). "
        elif course_code_pattern.search(query_for_embedding) or course_related_keywords.search(user_message):
            k_search_results = 10
            search_method_message = f"Sử dụng vector search cho môn học/từ khóa liên quan (k={k_search_results}). "
        else:  # Default for other general queries
            k_search_results = 7
            search_method_message = f"Sử dụng vector search chung (k={k_search_results}). "

        print(
            f"App (Vector Search): query='{refined_query_text_for_log[:100]}...', k={k_search_results}")
        retrieved_chunks_vector = []

        if k_search_results > 0:
            faiss_available = loaded_resources.get(
                "faiss_index") and loaded_resources.get("faiss_mapping")
            # chroma_available = loaded_resources.get("chroma_collection") # Assuming FAISS is primary for now

            if faiss_available:
                retrieved_chunks_vector = search_faiss(
                    loaded_resources["faiss_index"], loaded_resources["faiss_mapping"], loaded_resources["all_chunks_data"], current_query_embedding_to_use, k=k_search_results)
                search_method_message += f"FAISS: {len(retrieved_chunks_vector)} chunks."
            # Add ChromaDB logic here if needed as primary or fallback
            else:
                search_method_message += "Không có vector store (FAISS) phù hợp hoặc hoạt động."
        else:  # k_search_results == 0 (e.g. intro query)
            search_method_message += "Không tìm kiếm vector do k=0."

        # Post-process vector search results if it was a student list query
        if is_student_list_query_flag and student_major_match_result and retrieved_chunks_vector:
            target_major_code_filter = student_major_match_result.group(
                2).upper()
            filtered_student_list_chunks = []
            other_vector_chunks = []
            for chunk_v in retrieved_chunks_vector:
                meta_v = chunk_v.get('metadata', {})
                type_v = meta_v.get('type', chunk_v.get('type'))
                if type_v == 'student_list_by_major' and meta_v.get('major_code', '').upper() == target_major_code_filter:
                    filtered_student_list_chunks.append(chunk_v)
                else:
                    other_vector_chunks.append(chunk_v)
            filtered_student_list_chunks.sort(
                key=lambda x: x.get('score', float('inf')))
            other_vector_chunks.sort(key=lambda x: x.get(
                'score', float('inf')))  # Sort others by score too
            retrieved_chunks = (filtered_student_list_chunks +
                                # Combine and cap
                                other_vector_chunks)[:k_search_results]
            search_method_message += f" Lọc DS SV cho ngành {target_major_code_filter}, {len(filtered_student_list_chunks)} chunk chuyên biệt."
        else:
            # Use all vector results if not student list or no specific filter found
            retrieved_chunks = retrieved_chunks_vector
            if retrieved_chunks:
                retrieved_chunks.sort(key=lambda x: x.get(
                    'score', float('inf')))  # Sort by score

    # --- Final response generation using Gemini ---
    # This part is reached if query was not "list all" or "curriculum by semester" handled above
    print(f"App (Search Debug): {search_method_message}")
    if retrieved_chunks:
        print(f"App: {len(retrieved_chunks)} chunks đưa vào context Gemini.")
    elif k_search_results == 0 and is_general_intro_query:
        print("App: Không có chunks nào được đưa vào context Gemini (k=0 cho câu hỏi giới thiệu).")
    elif not retrieved_chunks and k_search_results > 0:
        print(
            f"App: Không tìm thấy chunks nào từ tìm kiếm vector (k={k_search_results}).")

    bot_response_text = get_answer_from_gemini_custom(
        user_question=user_message,
        # These are from vector search or earlier direct scans
        context_chunks=retrieved_chunks,
        history=chat_history[:-1],
        queried_student_name=extracted_name_for_gemini_prompt
    )

    chat_history.append({'role': 'model', 'parts': [bot_response_text]})
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    return jsonify({'response': bot_response_text})


if __name__ == '__main__':
    print("App: Bắt đầu chạy __main__ block.")
    init_result = initialize_resources()
    print(
        f"App: Kết quả của initialize_resources() trong __main__: {init_result}")

    if init_result:
        print("Khởi chạy Flask app trên cổng 5001...")
        app.run(debug=True, port=5001)
    else:
        print("Không thể khởi chạy Flask app do lỗi khởi tạo tài nguyên.")
