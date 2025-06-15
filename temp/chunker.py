import pymysql
import os
from dotenv import load_dotenv
import json

# --- Database Connection ---
load_dotenv()


def create_chunker_connection():  # Đổi tên hàm để tránh nhầm lẫn nếu có hàm create_connection khác
    try:
        conn = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            port=int(os.getenv('DB_PORT', 3306)),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("ChunkerV2: Kết nối MySQL thành công.")
        return conn
    except pymysql.Error as e:
        print(f"ChunkerV2: Kết nối thất bại. Lỗi: {e}")
        return None

# --- Syllabus Data Fetching (Giữ nguyên) ---


def fetch_reconstructed_syllabus_data(conn, syllabus_id_to_fetch):
    data = {}
    cursor = conn.cursor()
    try:
        sql_metadata = """
            SELECT s.*, c.course_name_from_curriculum, c.semester AS semester_from_curriculum, 
                   c.combo_name_from_curriculum, c.combo_short_name_from_curriculum, c.course_type_guess
            FROM Syllabuses s JOIN Courses c ON s.course_id = c.course_id
            WHERE s.syllabus_id = %s;
        """
        cursor.execute(sql_metadata, (syllabus_id_to_fetch,))
        metadata_db = cursor.fetchone()
        if not metadata_db:
            print(
                f"ChunkerV2: Không tìm thấy syllabus với ID: {syllabus_id_to_fetch}")
            return None
        if 'has_download_materials_button' in metadata_db:
            metadata_db['has_download_materials_button'] = bool(
                metadata_db['has_download_materials_button'])
        metadata_db['major_instructional_areas'] = 'N/A'  # Placeholder
        data['metadata'] = metadata_db
        sql_los = "SELECT outcome_id as id, details FROM LearningOutcomes WHERE syllabus_id = %s;"
        cursor.execute(sql_los, (syllabus_id_to_fetch,))
        data['learning_outcomes'] = cursor.fetchall()
        sql_materials = "SELECT * FROM Materials WHERE syllabus_id = %s;"
        cursor.execute(sql_materials, (syllabus_id_to_fetch,))
        materials_db = cursor.fetchall()
        for mat in materials_db:
            mat['is_main_material'] = bool(mat['is_main_material'])
            mat['is_hard_copy'] = bool(mat['is_hard_copy'])
            mat['is_online'] = bool(mat['is_online'])
        data['materials'] = materials_db
        sql_schedule = "SELECT session, topic, teaching_type, itu, materials, tasks, download_link, urls FROM Schedules WHERE syllabus_id = %s;"
        cursor.execute(sql_schedule, (syllabus_id_to_fetch,))
        data['schedule'] = cursor.fetchall()
        sql_assessments = "SELECT * FROM Assessments WHERE syllabus_id = %s;"
        cursor.execute(sql_assessments, (syllabus_id_to_fetch,))
        assessments_db = cursor.fetchall()
        for asm in assessments_db:
            asm['clos'] = [asm.get('clos_text', '')]
        data['assessments'] = assessments_db
        return data
    except pymysql.Error as e:
        print(
            f"ChunkerV2: Lỗi fetch_reconstructed_syllabus_data cho {syllabus_id_to_fetch}: {e}")
        return None
    finally:
        if cursor:
            cursor.close()

# --- Student Data Fetching (MỚI) ---


def fetch_student_data_by_major_from_db(conn):
    """Fetches student data (roll_number, fullname, email) grouped by major, including major_name."""
    student_data_map = {}
    cursor = conn.cursor()
    try:
        sql_students = """
            SELECT 
                s.roll_number, s.fullname, s.email, 
                m.major_code, m.curriculum_title AS major_name
            FROM Students s
            JOIN Majors m ON s.major_code = m.major_code
            ORDER BY m.major_code, s.roll_number;
        """
        cursor.execute(sql_students)
        all_students_rows = cursor.fetchall()

        if not all_students_rows:
            print("ChunkerV2: Không tìm thấy dữ liệu sinh viên nào trong DB.")
            return student_data_map

        for student_row_data in all_students_rows:
            major_code_val = student_row_data['major_code']
            if major_code_val not in student_data_map:
                student_data_map[major_code_val] = {
                    'major_name': student_row_data['major_name'] if student_row_data['major_name'] else major_code_val,
                    'students_list': []
                }
            student_data_map[major_code_val]['students_list'].append({
                'roll_number': student_row_data['roll_number'],
                'fullname': student_row_data['fullname'],
                # Email được fetch nhưng sẽ không đưa vào chunk content
                'email': student_row_data['email']
            })

        print(
            f"ChunkerV2: Đã tải dữ liệu sinh viên cho {len(student_data_map)} chuyên ngành.")
        return student_data_map
    except pymysql.Error as e:
        print(f"ChunkerV2: Lỗi khi tải dữ liệu sinh viên: {e}")
        return student_data_map  # Return empty or partially filled map on error
    finally:
        if cursor:
            cursor.close()

# --- Chunking Logic (Giữ nguyên resolve_clos, cập nhật chunk_syllabus_data, thêm chunk_student_info) ---


def resolve_clos(clo_ids_str, clo_map):
    resolved_details = []
    if not clo_ids_str or not isinstance(clo_ids_str, str):
        return "N/A (Invalid CLO input)"
    if "One or many of" in clo_ids_str:
        clo_ids_str = clo_ids_str.replace("One or many of ", "")
    if "." in clo_ids_str and (clo_ids_str.lower().startswith("presentation") or clo_ids_str.lower().startswith("asm") or clo_ids_str.lower().startswith("assignment")):
        parts = clo_ids_str.split(". ", 1)
        if len(parts) > 1:
            clo_ids_str = parts[1]
    if "." in clo_ids_str and clo_ids_str.lower().startswith("test"):
        parts = clo_ids_str.split(". ", 1)
        if len(parts) > 1:
            clo_ids_str = parts[1]
    cleaned_ids_str = clo_ids_str.replace(
        ";", ",").replace(" ", ",").replace("-", ",")
    cleaned_ids = [c.strip() for c in cleaned_ids_str.split(
        ',') if c.strip() and c.strip().lower() != 'clo']
    for clo_id in cleaned_ids:
        if clo_id in clo_map:
            resolved_details.append(f"- {clo_id}: {clo_map[clo_id]}")
        elif clo_id == ">0" or clo_id.lower() == "all clos" or clo_id.lower() == "all":
            resolved_details.append(f"- All CLOs defined for this course.")
        elif clo_id.lower() in ["it", "tu", "hw", "qs", "bt", "lab", "dg"]:
            resolved_details.append(f"- {clo_id} (Activity Type)")  # Sửa lại
        elif not any(char.isdigit() for char in clo_id) and len(clo_id) < 5:
            resolved_details.append(
                f"- {clo_id} (Possible Activity Code)")  # Sửa lại
        else:
            resolved_details.append(
                f"- {clo_id} (Details not found/General ref)")  # Sửa lại
    return "\n".join(resolved_details) if resolved_details else "N/A"


def chunk_syllabus_data(syllabus_db_data):
    if not syllabus_db_data:
        return []
    chunks = []
    metadata = syllabus_db_data['metadata']
    clo_map = {str(clo['id']): clo['details']
               for clo in syllabus_db_data.get('learning_outcomes', [])}

    # --- Lấy các thông tin chung từ metadata chính của syllabus để thêm vào từng chunk con ---
    base_course_id = metadata.get('course_id')
    base_course_name = metadata.get('course_name_from_curriculum')
    base_syllabus_id = metadata.get('syllabus_id')
    base_semester = metadata.get('semester_from_curriculum')  # Rất quan trọng
    base_combo_name = metadata.get('combo_name_from_curriculum')
    base_course_type = metadata.get('course_type_guess')
    # Lấy major_name từ bảng Majors (đã join trong fetch)
    base_major_name = metadata.get('major_name')
    # Nếu major_name không có, thử suy từ course_name_from_curriculum hoặc tên combo
    if not base_major_name:
        if base_course_name and ("Artificial Intelligence" in base_course_name or "Trí tuệ nhân tạo" in base_course_name):
            base_major_name = "Trí tuệ nhân tạo"
        elif base_course_name and ("Software Engineering" in base_course_name or "Kỹ thuật phần mềm" in base_course_name):
            base_major_name = "Kỹ thuật phần mềm"
        elif base_combo_name:  # Thử dùng tên combo nếu có
            base_major_name = base_combo_name
    # Hoặc có thể lấy từ curriculum_title nếu có trong metadata

    # Metadata chung cho tất cả các chunks của syllabus này
    common_chunk_metadata = {
        'course_id': base_course_id,
        'course_name': base_course_name,
        'syllabus_id': base_syllabus_id,
        'semester': base_semester,  # Thêm học kỳ
        'major_name': base_major_name,  # Thêm tên ngành (nếu có)
        'combo_name': base_combo_name,  # Thêm tên combo (nếu có)
        'course_type': base_course_type  # Thêm loại môn (nếu có)
    }

    overview_chunk_content = f"""Thông tin tổng quan Khóa học: "{base_course_name if base_course_name else 'N/A'}" (Mã môn: {base_course_id if base_course_id else 'N/A'}, Syllabus ID: {base_syllabus_id if base_syllabus_id else 'N/A'}). Tiếng Anh: {metadata.get('english_title', 'N/A')}. Mô tả: {metadata.get('description', 'N/A')}. Yêu cầu SV: {metadata.get('student_tasks', 'N/A')}. Công cụ: {metadata.get('tools') if metadata.get('tools') else 'N/A'}. Tín chỉ: {metadata.get('credits', 'N/A')}. Cấp độ: {metadata.get('degree_level', 'N/A')}. Điểm qua môn: {metadata.get('min_avg_mark_to_pass', 'N/A')}. Phê duyệt: {"Đã phê duyệt" if metadata.get('is_approved') == "True" or metadata.get('is_approved') == True else "Chưa/Không rõ"}, QĐ số: {metadata.get('decision_no', 'N/A')} ngày {metadata.get('approved_date', 'N/A')}."""
    prerequisites_text_for_overview = metadata.get(
        'prerequisites', 'Không yêu cầu môn tiên quyết hoặc thông tin không có.')
    if not prerequisites_text_for_overview or prerequisites_text_for_overview.lower() == 'n/a' or len(prerequisites_text_for_overview) < 100:  # Ngưỡng ngắn hơn
        overview_chunk_content += f" Môn tiên quyết: {prerequisites_text_for_overview}"
    chunks.append({'type': 'overview', 'content': overview_chunk_content.strip(
    ), 'metadata': {**common_chunk_metadata}})

    prerequisites_text = metadata.get('prerequisites')
    if prerequisites_text and prerequisites_text.lower() != 'n/a' and len(prerequisites_text) >= 100:
        prereq_chunk_content = f"""Môn học: {base_course_name if base_course_name else 'N/A'} ({base_course_id if base_course_id else 'N/A'}). Thông tin chi tiết về các môn học tiên quyết (Prerequisites): {prerequisites_text}"""
        chunks.append({'type': 'prerequisites', 'content': prereq_chunk_content.strip(
        ), 'metadata': {**common_chunk_metadata}})

    for clo in syllabus_db_data.get('learning_outcomes', []):
        clo_chunk_content = f"""Môn học: {base_course_name if base_course_name else 'N/A'} ({base_course_id if base_course_id else 'N/A'}). Kết quả học tập (Learning Outcome) mã số {clo.get('id', 'N/A')} là: {clo.get('details', 'N/A')}."""
        chunks.append({'type': 'learning_outcome', 'content': clo_chunk_content.strip(
        ), 'metadata': {**common_chunk_metadata, 'clo_id': clo.get('id')}})

    for material in syllabus_db_data.get('materials', []):
        material_chunk_content = f"""Môn học: {base_course_name if base_course_name else 'N/A'} ({base_course_id if base_course_id else 'N/A'}). Tài liệu: {material.get('description', 'N/A')}. Tác giả: {material.get('author', 'N/A')}, NXB: {material.get('publisher', 'N/A')}, Năm XB: {material.get('published_date', 'N/A')}, ISBN: {material.get('isbn', 'N/A')}. Chính: {"Có" if material.get('is_main_material') else "Không"}."""
        chunks.append({'type': 'material', 'content': material_chunk_content.strip(), 'metadata': {
                      **common_chunk_metadata, 'material_description': material.get('description')}})

    current_main_session_heading = ""
    for i, session_entry in enumerate(syllabus_db_data.get('schedule', [])):
        session_title = session_entry.get('session', '').strip()
        parts_by_dot = session_title.split('.', 1)
        if session_title and session_title[0].isdigit():
            if len(parts_by_dot) > 1 and not parts_by_dot[1].strip().isdigit() and not parts_by_dot[1].strip().startswith('.'):
                current_main_session_heading = session_title
            elif len(parts_by_dot) == 1 and len(session_title) < 50:
                current_main_session_heading = session_title
        teaching_type_clos = session_entry.get('teaching_type', '')
        resolved_teaching_clos_details = resolve_clos(
            teaching_type_clos, clo_map)
        schedule_chunk_content = f"""Môn học: {base_course_name if base_course_name else 'N/A'} ({base_course_id if base_course_id else 'N/A'}). Buổi học: {current_main_session_heading + " - " if current_main_session_heading and not session_title.startswith(current_main_session_heading.split('.')[0]) and current_main_session_heading != session_title else ''}{session_title}. Chủ đề: {session_entry.get('topic', 'N/A')}. Loại giảng dạy: {session_entry.get('teaching_type', 'N/A')}. CLOs liên quan (từ loại giảng dạy): {resolved_teaching_clos_details}. ITU: {session_entry.get('itu', 'N/A')}. Tài liệu: {session_entry.get('materials', 'N/A')}. Nhiệm vụ: {session_entry.get('tasks', 'N/A')}."""
        chunks.append({'type': 'schedule_entry', 'content': schedule_chunk_content.strip(), 'metadata': {
                      **common_chunk_metadata, 'session_title': session_title, 'session_index': i}})

    for i, assessment in enumerate(syllabus_db_data.get('assessments', [])):
        assessment_clos_str = assessment.get(
            'clos', [''])[0] if assessment.get('clos') else ''
        resolved_assessment_clos_details = resolve_clos(
            assessment_clos_str, clo_map)
        question_type_text = assessment.get('question_type', '')
        resolved_question_type_clos_details = ""
        potential_clos_in_qtype = False
        if isinstance(question_type_text, str):
            if "CLO" in question_type_text.upper() and any(char.isdigit() for char in question_type_text):
                potential_clos_in_qtype = True
            elif "ALL CLO" in question_type_text.upper():
                potential_clos_in_qtype = True
        if potential_clos_in_qtype:
            resolved_question_type_clos_details = f" Loại câu hỏi (có thể chứa CLO): {resolve_clos(question_type_text, clo_map)}"
        else:
            resolved_question_type_clos_details = f" Loại câu hỏi: {question_type_text if question_type_text else 'N/A'}"
        assessment_chunk_content = f"""Môn học: {base_course_name if base_course_name else 'N/A'} ({base_course_id if base_course_id else 'N/A'}). Đánh giá: {assessment.get('category', 'N/A')} - {assessment.get('type', 'N/A')}. Trọng số: {assessment.get('weight', 'N/A')}%. CLOs (từ clos_text): {resolved_assessment_clos_details}.{resolved_question_type_clos_details}. Kiến thức & Kỹ năng: {assessment.get('knowledge_and_skill', 'N/A')}. Hoàn thành: {assessment.get('completion_criteria', 'N/A')}."""
        chunks.append({'type': 'assessment', 'content': assessment_chunk_content.strip(), 'metadata': {
                      **common_chunk_metadata, 'assessment_category': assessment.get('category'), 'assessment_index': i}})
    return chunks

# --- Student Data Chunking (MỚI) ---


def chunk_student_info_data(student_data_map_input):
    """Chunks student data, creating one chunk per major, listing student names and roll numbers."""
    student_info_chunks = []
    if not student_data_map_input:
        print("ChunkerV2: Không có dữ liệu sinh viên để chunk.")
        return student_info_chunks

    for major_code_key, major_data_item in student_data_map_input.items():
        major_name_val = major_data_item.get('major_name', major_code_key)
        students_list_val = major_data_item.get('students_list', [])

        if not students_list_val:
            continue  # Bỏ qua chuyên ngành không có sinh viên

        # Giới hạn số lượng sinh viên trong content để chunk không quá lớn
        # và chỉ bao gồm Tên, MSSV. KHÔNG BAO GỒM EMAIL TRONG CONTENT.
        max_students_in_chunk_content = 50  # Có thể điều chỉnh
        student_details_for_content = []
        for std_idx, student_item in enumerate(students_list_val):
            if std_idx < max_students_in_chunk_content:
                student_details_for_content.append(
                    f"- {student_item.get('fullname', 'N/A')} (MSSV: {student_item.get('roll_number', 'N/A')}) - Email: {student_item.get('email', 'N/A')}")
            else:
                student_details_for_content.append(
                    f"- ... và {len(students_list_val) - max_students_in_chunk_content} sinh viên khác.")
                break  # Dừng thêm sau khi đạt giới hạn

        student_list_content_str = "\n".join(student_details_for_content)
        if not student_list_content_str:
            student_list_content_str = "(Không có thông tin chi tiết sinh viên để hiển thị trong chunk này)"

        chunk_content_final = f"""
Chuyên ngành: {major_name_val} (Mã: {major_code_key}).
Danh sách một phần sinh viên thuộc chuyên ngành này:
{student_list_content_str}
Để có danh sách đầy đủ, vui lòng truy vấn trực tiếp cơ sở dữ liệu quản lý sinh viên.
"""
        student_info_chunks.append({
            'type': 'student_list_by_major',  # Loại chunk mới
            'content': chunk_content_final.strip(),
            'metadata': {
                'major_code': major_code_key,
                'major_name': major_name_val,
                'number_of_students_in_content': min(len(students_list_val), max_students_in_chunk_content),
                'total_students_in_major_db': len(students_list_val)
            }
        })
    print(
        f"ChunkerV2: Đã tạo {len(student_info_chunks)} chunks thông tin sinh viên.")
    return student_info_chunks


# --- Main Execution (Cập nhật để xử lý cả syllabus và student) ---
if __name__ == "__main__":
    db_connection = create_chunker_connection()  # Sử dụng tên hàm mới
    final_system_chunks = []

    if db_connection:
        try:
            # 1. Xử lý Syllabus Data
            print("\n--- ChunkerV2: Bắt đầu xử lý dữ liệu Syllabus ---")
            syllabus_cursor = db_connection.cursor()
            syllabus_cursor.execute(
                "SELECT DISTINCT syllabus_id FROM Syllabuses;")
            all_syllabus_ids_from_db = syllabus_cursor.fetchall()
            syllabus_cursor.close()

            if not all_syllabus_ids_from_db:
                print("ChunkerV2: Không tìm thấy syllabus IDs nào trong DB.")
            else:
                for syllabus_id_row in all_syllabus_ids_from_db:
                    current_syllabus_id = syllabus_id_row['syllabus_id']
                    print(
                        f"--- ChunkerV2: Đang xử lý Syllabus ID: {current_syllabus_id} ---")
                    reconstructed_syllabus = fetch_reconstructed_syllabus_data(
                        db_connection, current_syllabus_id)
                    if reconstructed_syllabus:
                        syllabus_specific_chunks = chunk_syllabus_data(
                            reconstructed_syllabus)
                        final_system_chunks.extend(syllabus_specific_chunks)
                        print(
                            f"ChunkerV2: Đã tạo {len(syllabus_specific_chunks)} chunks cho Syllabus ID: {current_syllabus_id}")
                    else:
                        print(
                            f"ChunkerV2: Bỏ qua Syllabus ID: {current_syllabus_id} do không fetch được data.")

            # 2. Xử lý Student Data
            print("\n--- ChunkerV2: Bắt đầu xử lý dữ liệu Sinh viên ---")
            student_data_map_fetched = fetch_student_data_by_major_from_db(
                db_connection)
            if student_data_map_fetched:
                student_generated_chunks = chunk_student_info_data(
                    student_data_map_fetched)
                final_system_chunks.extend(student_generated_chunks)
            else:
                print("ChunkerV2: Không có dữ liệu sinh viên để chunk.")

        except pymysql.Error as e:
            print(f"ChunkerV2: Lỗi trong quá trình xử lý chính: {e}")
        finally:
            db_connection.close()
            print("\nChunkerV2: Đã đóng kết nối MySQL.")

        output_target_filename = "all_syllabus_and_student_chunks.json"
        print(
            f"\nChunkerV2: Tổng số chunks đã tạo (syllabus và student): {len(final_system_chunks)}")
        try:
            with open(output_target_filename, 'w', encoding='utf-8') as f_out:
                json.dump(final_system_chunks, f_out,
                          ensure_ascii=False, indent=2)
            print(
                f"ChunkerV2: Đã lưu tất cả chunks vào file: {output_target_filename}")
        except IOError as e_io:
            print(f"ChunkerV2: Lỗi khi lưu file chunks: {e_io}")
    else:
        print("ChunkerV2: Không thể kết nối tới database. Chunker không thể chạy.")
