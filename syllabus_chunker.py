import json
import re
import os
import uuid

def clean_text(text):
    """Làm sạch văn bản cơ bản: loại bỏ khoảng trắng thừa, chuẩn hóa newline."""
    if not isinstance(text, str):
        return str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def create_chunks_from_syllabus(subject_code, syllabus_data):
    """Tạo các chunk từ dữ liệu syllabus đã được trích xuất."""
    chunks = []
    base_metadata = {
        "subject_code": subject_code,
        "syllabus_id": syllabus_data.get("syllabus_id", "N/A"),
        "chunk_id": None  # Sẽ được cập nhật cho mỗi chunk
    }

    # Chunk 1: Thông tin chung của môn học
    general_info_texts = []
    if "general_details" in syllabus_data:
        gd = syllabus_data["general_details"]
        
        syllabus_name = gd.get("Syllabus Name", "N/A")
        syllabus_english = gd.get("Syllabus English", "N/A")
        general_info_texts.append(f"Tên môn học: {syllabus_name} ({syllabus_english}).")
        
        subject_code_val = gd.get("Subject Code", "N/A")
        no_credit = gd.get("NoCredit", "N/A")
        general_info_texts.append(f"Mã môn: {subject_code_val}. Số tín chỉ: {no_credit}.")
        
        description = gd.get("Description", "N/A")
        general_info_texts.append(f"Mô tả: {description}")
        
        pre_requisite = gd.get("Pre-Requisite")
        if pre_requisite:
            general_info_texts.append(f"Điều kiện tiên quyết: {pre_requisite}.")
            
        student_tasks = gd.get("StudentTasks", "N/A")
        general_info_texts.append(f"Nhiệm vụ sinh viên: {student_tasks}.")
        
        tools = gd.get("Tools")
        if tools:
            general_info_texts.append(f"Công cụ sử dụng: {tools}.")
        
        chunk_text = clean_text(" ".join(general_info_texts))
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_id"] = str(uuid.uuid4())
        chunk_metadata["source_section"] = "general_details"
        chunk_metadata["title"] = f"Thông tin chung - {subject_code}"
        
        chunks.append({
            "type": "general_info",
            "content": chunk_text,
            "metadata": chunk_metadata
        })

    # Chunk 2: Mỗi Chuẩn đầu ra (CLO)
    if "clos" in syllabus_data and syllabus_data["clos"]:
        for i, clo in enumerate(syllabus_data["clos"]):
            clo_name = clo.get("CLO Name", f"CLO {i+1}")
            clo_details_text = clo.get("CLO Details", "")
            lo_details = clo.get("LO Details", "N/A")
            
            if clo_name.strip().lower() == clo_details_text.strip().lower():
                chunk_text = clean_text(f"Chuẩn đầu ra {clo_name}: {lo_details}")
            else:
                 chunk_text = clean_text(f"Chuẩn đầu ra {clo_name} ({clo_details_text}): {lo_details}")
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = str(uuid.uuid4())
            chunk_metadata["source_section"] = "clos"
            chunk_metadata["clo_id"] = clo_name
            chunk_metadata["title"] = f"CLO {clo_name} - {subject_code}"
            
            chunks.append({
                "type": "clo",
                "content": chunk_text,
                "metadata": chunk_metadata
            })

    # Chunk 3: Mỗi Buổi học (Session)
    if "sessions" in syllabus_data and syllabus_data["sessions"]:
        for session in syllabus_data["sessions"]:
            session_texts = []
            session_number = session.get("Session", "N/A")
            topic = session.get("Topic", "N/A")
            session_texts.append(f"Buổi học số {session_number}: Chủ đề - {topic}.")
            
            learning_type = session.get("Learning-Teaching Type", "N/A")
            lo = session.get("LO", "N/A")
            session_texts.append(f"Loại hình: {learning_type}. Chuẩn đầu ra liên quan (LO): {lo}.")
            
            student_materials = session.get("Student Materials")
            if student_materials:
                 session_texts.append(f"Tài liệu sinh viên: {student_materials}.")
                 
            download_info = session.get("S-Download")
            if download_info:
                if isinstance(download_info, dict):
                    download_text = download_info.get("text", "")
                    download_link = download_info.get("link", "N/A")
                    session_texts.append(f"Link tải tài liệu buổi học: {download_text} - {download_link}.")
                elif isinstance(download_info, str) and download_info.strip():
                     session_texts.append(f"Link tải tài liệu buổi học: {download_info}.")
                     
            student_tasks_val = session.get("Student's Tasks", "N/A")
            session_texts.append(f"Nhiệm vụ sinh viên: {student_tasks_val}.")
            
            urls = session.get("URLs")
            if urls:
                session_texts.append(f"URLs liên quan: {urls}.")
            
            chunk_text = clean_text(" ".join(session_texts))
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = str(uuid.uuid4())
            chunk_metadata["source_section"] = "sessions"
            chunk_metadata["session_number"] = session_number
            chunk_metadata["title"] = f"Buổi {session_number} - {subject_code}"
            
            chunks.append({
                "type": "session",
                "content": chunk_text,
                "metadata": chunk_metadata
            })

    # Chunk 4: Mỗi Hình thức Đánh giá (Assessment)
    if "assessments" in syllabus_data and syllabus_data["assessments"]:
        for i, assessment in enumerate(syllabus_data["assessments"]):
            assessment_texts = []
            category = assessment.get("Category", f"Đánh giá {i+1}")
            assessment_type = assessment.get("Type", "N/A")
            assessment_texts.append(f"Hình thức đánh giá: {category} ({assessment_type}).")
            
            weight = assessment.get("Weight", "N/A")
            clo_assessment = assessment.get("CLO", "N/A")
            assessment_texts.append(f"Trọng số: {weight}. Chuẩn đầu ra liên quan (CLO): {clo_assessment}.")
            
            question_type = assessment.get("Question Type", "N/A")
            duration = assessment.get("Duration", "N/A")
            assessment_texts.append(f"Loại câu hỏi/Nhiệm vụ: {question_type}. Thời gian: {duration}.")
            
            completion_criteria = assessment.get("Completion Criteria")
            if completion_criteria:
                assessment_texts.append(f"Tiêu chí hoàn thành: {completion_criteria}.")
                
            knowledge_skill = assessment.get("Knowledge and Skill")
            if knowledge_skill:
                 assessment_texts.append(f"Kiến thức và kỹ năng: {knowledge_skill}.")
                 
            note_assessment = assessment.get("Note")
            if note_assessment:
                 assessment_texts.append(f"Ghi chú: {note_assessment}.")

            chunk_text = clean_text(" ".join(assessment_texts))
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = str(uuid.uuid4())
            chunk_metadata["source_section"] = "assessments"
            chunk_metadata["assessment_category"] = category
            chunk_metadata["title"] = f"Đánh giá {category} - {subject_code}"
            
            chunks.append({
                "type": "assessment",
                "content": chunk_text,
                "metadata": chunk_metadata
            })

    # Chunk 5: Thông tin Tài liệu học tập (Materials)
    if "materials_table" in syllabus_data and syllabus_data["materials_table"]:
        for i, material in enumerate(syllabus_data["materials_table"]):
            material_texts = []
            desc = material.get("MaterialDescription", f"Tài liệu {i+1}")
            material_texts.append(f"Tài liệu học tập: {desc}.")
            
            author = material.get("Author")
            if author:
                material_texts.append(f"Tác giả: {author}.")
                
            publisher = material.get("Publisher")
            if publisher:
                material_texts.append(f"Nhà xuất bản: {publisher}.")
                
            published_date = material.get("PublishedDate")
            if published_date:
                material_texts.append(f"Năm xuất bản: {published_date}.")
                
            note_material = material.get("Note")
            if note_material and ("http" in note_material or "www" in note_material):
                 material_texts.append(f"Link/Ghi chú: {note_material}.")
            elif note_material:
                 material_texts.append(f"Ghi chú: {note_material}.")

            chunk_text = clean_text(" ".join(material_texts))
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = str(uuid.uuid4())
            chunk_metadata["source_section"] = "materials_table"
            chunk_metadata["material_description"] = desc[:50]
            chunk_metadata["title"] = f"Tài liệu {i+1} - {subject_code}"
            
            chunks.append({
                "type": "material",
                "content": chunk_text,
                "metadata": chunk_metadata
            })
            
    # Thêm chunk cho liên kết CLO-PLO nếu có
    clo_plo_link = syllabus_data.get("clo_plo_mapping_link")
    if clo_plo_link:
        chunk_text = clean_text(f"Liên kết xem bản đồ Chuẩn đầu ra môn học (CLO) với Chuẩn đầu ra chương trình (PLO) của môn {subject_code} là: {clo_plo_link}")
        
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_id"] = str(uuid.uuid4())
        chunk_metadata["source_section"] = "clo_plo_mapping_link"
        chunk_metadata["title"] = f"Mapping CLO-PLO - {subject_code}"
        
        chunks.append({
            "type": "clo_plo_mapping",
            "content": chunk_text,
            "metadata": chunk_metadata
        })

    return chunks

def process_all_syllabi(input_syllabus_file, output_dir="Chunk"):
    """Xử lý tất cả các syllabus từ file đầu vào và lưu các chunks theo môn học."""
    try:
        with open(input_syllabus_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu syllabus tại: {input_syllabus_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"Lỗi: File dữ liệu syllabus không phải là JSON hợp lệ: {e}")
        return None

    if not isinstance(raw_data, dict) or not raw_data:
        print("Lỗi: Dữ liệu syllabus không có cấu trúc dictionary hoặc rỗng.")
        return None
    
    all_chunks = []
    processed_subjects = []
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Xử lý từng môn học
    for subject_code, syllabus_content in raw_data.items():
        print(f"Đang xử lý syllabus cho môn {subject_code}...")
        
        if not isinstance(syllabus_content, dict):
            print(f"Bỏ qua: Nội dung syllabus cho môn {subject_code} không phải là dictionary.")
            continue
            
        subject_chunks = create_chunks_from_syllabus(subject_code, syllabus_content)
        print(f"Đã tạo {len(subject_chunks)} chunks cho môn {subject_code}.")
        
        # Lưu chunks của môn học này vào file riêng
        subject_output_file = f"{output_dir}/{subject_code}_chunks.json"
        try:
            with open(subject_output_file, "w", encoding="utf-8") as f_out:
                json.dump(subject_chunks, f_out, ensure_ascii=False, indent=2)
            print(f"Đã lưu chunks của môn {subject_code} vào file: {subject_output_file}")
            processed_subjects.append(subject_code)
        except IOError as e:
            print(f"Lỗi khi lưu file chunks cho môn {subject_code}: {e}")
        
        # Thêm chunks vào danh sách tổng hợp
        all_chunks.extend(subject_chunks)
    
    # Lưu tất cả chunks vào một file tổng hợp
    all_chunks_file = f"{output_dir}/all_chunks.json"
    try:
        with open(all_chunks_file, "w", encoding="utf-8") as f_out:
            json.dump(all_chunks, f_out, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu tất cả {len(all_chunks)} chunks từ {len(processed_subjects)} môn học vào file: {all_chunks_file}")
        return all_chunks_file
    except IOError as e:
        print(f"Lỗi khi lưu file tổng hợp chunks: {e}")
        return None

if __name__ == "__main__":
    input_syllabus_file = "Syllabus_crawler/fpt_syllabus_data_appended_vi.json"
    output_directory = "Chunk"
    
    all_chunks_file = process_all_syllabi(input_syllabus_file, output_directory)
    
    if all_chunks_file:
        print(f"Hoàn thành xử lý tất cả syllabus. Dữ liệu chunks được lưu tại: {all_chunks_file}")
    else:
        print("Không thể hoàn thành việc xử lý syllabus.")