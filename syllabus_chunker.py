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

def extract_important_metadata(syllabus_data):
    """Trích xuất các metadata quan trọng từ syllabus để sử dụng trong nhiều chunks."""
    metadata = {}
    
    if "general_details" in syllabus_data:
        gd = syllabus_data["general_details"]
        
        metadata["syllabus_name"] = gd.get("Syllabus Name", "N/A")
        metadata["syllabus_english"] = gd.get("Syllabus English", "")
        metadata["subject_code"] = gd.get("Subject Code", "N/A")
        metadata["credits"] = gd.get("NoCredit", "N/A")
        metadata["degree_level"] = gd.get("Degree Level", "N/A")
        metadata["time_allocation"] = gd.get("Time Allocation", "N/A")
        metadata["pre_requisite"] = gd.get("Pre-Requisite", "")
        metadata["description"] = gd.get("Description", "N/A")
        metadata["student_tasks"] = gd.get("StudentTasks", "N/A")
        metadata["tools"] = gd.get("Tools", "")
        metadata["scoring_scale"] = gd.get("Scoring Scale", "N/A")
        metadata["min_avg_mark_to_pass"] = gd.get("MinAvgMarkToPass", "N/A")
        metadata["note"] = gd.get("Note", "")
        
    # Extract assessment summary if available
    if "assessments" in syllabus_data and syllabus_data["assessments"]:
        assessment_weights = []
        for assessment in syllabus_data["assessments"]:
            category = assessment.get("Category", "")
            weight = assessment.get("Weight", "")
            if category and weight:
                assessment_weights.append(f"{category}: {weight}")
        
        if assessment_weights:
            metadata["assessment_summary"] = ", ".join(assessment_weights)
    
    # Count total materials
    if "materials_table" in syllabus_data:
        metadata["total_materials"] = len(syllabus_data["materials_table"])
    
    # Count total CLOs
    if "clos" in syllabus_data:
        metadata["total_clos"] = len(syllabus_data["clos"])
    
    # Count total sessions
    if "sessions" in syllabus_data:
        metadata["total_sessions"] = len(syllabus_data["sessions"])
    
    return metadata

def create_chunks_from_syllabus(subject_code, syllabus_data):
    """Tạo các chunk từ dữ liệu syllabus đã được trích xuất với cải tiến để bảo toàn thông tin quan trọng."""
    chunks = []
    
    # Extract important metadata to use across chunks
    important_metadata = extract_important_metadata(syllabus_data)
    
    base_metadata = {
        "subject_code": subject_code,
        "syllabus_id": syllabus_data.get("syllabus_id", "N/A"),
        "chunk_id": None,  # Sẽ được cập nhật cho mỗi chunk
        "syllabus_name": important_metadata.get("syllabus_name", "N/A"),
        "credits": important_metadata.get("credits", "N/A"),
        "total_clos": important_metadata.get("total_clos", 0),
        "total_sessions": important_metadata.get("total_sessions", 0),
        "total_materials": important_metadata.get("total_materials", 0),
    }

    # CHUNK 0: Course Overview (NEW) - Contains the most important/searchable information
    overview_texts = []
    overview_texts.append(f"Môn học {subject_code} - {important_metadata.get('syllabus_name', 'N/A')}.")
    overview_texts.append(f"Tên tiếng Anh: {important_metadata.get('syllabus_english', '')}.")
    overview_texts.append(f"Số tín chỉ: {important_metadata.get('credits', 'N/A')}.")
    
    if important_metadata.get("pre_requisite"):
        overview_texts.append(f"Môn học tiên quyết: {important_metadata.get('pre_requisite')}.")
    
    overview_texts.append(f"Mô tả: {important_metadata.get('description', 'N/A')}")
    
    # Add assessment summary
    if "assessment_summary" in important_metadata:
        overview_texts.append(f"Thành phần đánh giá: {important_metadata.get('assessment_summary')}.")
    
    overview_texts.append(f"Thang điểm: {important_metadata.get('scoring_scale', '10')}. Điểm trung bình tối thiểu để đạt: {important_metadata.get('min_avg_mark_to_pass', '5')}.")
    
    chunk_text = clean_text(" ".join(overview_texts))
    chunk_metadata = base_metadata.copy()
    chunk_metadata["chunk_id"] = str(uuid.uuid4())
    chunk_metadata["source_section"] = "overview"
    chunk_metadata["title"] = f"Tổng quan - {subject_code}"
    
    chunks.append({
        "type": "overview",
        "content": chunk_text,
        "metadata": chunk_metadata
    })

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
        
        degree_level = gd.get("Degree Level", "N/A")
        general_info_texts.append(f"Bậc đào tạo: {degree_level}.")
        
        time_allocation = gd.get("Time Allocation", "N/A")
        general_info_texts.append(f"Phân bổ thời gian: {time_allocation}.")
        
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
        
        # Add more details about grading criteria
        note = gd.get("Note")
        if note:
            general_info_texts.append(f"Thông tin đánh giá: {note}.")
            
        min_mark = gd.get("MinAvgMarkToPass")
        if min_mark:
            general_info_texts.append(f"Điểm trung bình tối thiểu để đạt: {min_mark}.")
        
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

    # CHUNK 1.5: Course Structure Summary (NEW)
    structure_texts = []
    structure_texts.append(f"Cấu trúc môn học {subject_code} - {important_metadata.get('syllabus_name', 'N/A')}.")
    structure_texts.append(f"Số tín chỉ: {important_metadata.get('credits', 'N/A')}.")
    
    if "total_clos" in important_metadata:
        structure_texts.append(f"Môn học có {important_metadata.get('total_clos')} chuẩn đầu ra (CLO).")
    
    if "total_sessions" in important_metadata:
        structure_texts.append(f"Môn học gồm {important_metadata.get('total_sessions')} buổi học.")
    
    if "total_materials" in important_metadata:
        structure_texts.append(f"Môn học sử dụng {important_metadata.get('total_materials')} tài liệu học tập.")
    
    # Add assessment summary if available
    if "assessment_summary" in important_metadata:
        structure_texts.append(f"Thành phần đánh giá: {important_metadata.get('assessment_summary')}.")
    
    if "min_avg_mark_to_pass" in important_metadata:
        structure_texts.append(f"Điểm trung bình tối thiểu để đạt: {important_metadata.get('min_avg_mark_to_pass')}.")
    
    chunk_text = clean_text(" ".join(structure_texts))
    chunk_metadata = base_metadata.copy()
    chunk_metadata["chunk_id"] = str(uuid.uuid4())
    chunk_metadata["source_section"] = "structure"
    chunk_metadata["title"] = f"Cấu trúc môn học - {subject_code}"
    
    chunks.append({
        "type": "structure",
        "content": chunk_text,
        "metadata": chunk_metadata
    })

    # Chunk 2: Mỗi Chuẩn đầu ra (CLO) - Thêm thông tin môn học vào mỗi CLO
    if "clos" in syllabus_data and syllabus_data["clos"]:
        # Create a CLO overview chunk first
        clo_overview_texts = []
        clo_overview_texts.append(f"Tổng quan các chuẩn đầu ra (CLO) của môn {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ):")
        
        for i, clo in enumerate(syllabus_data["clos"]):
            clo_name = clo.get("CLO Name", f"CLO {i+1}")
            clo_details_text = clo.get("CLO Details", "")
            lo_details = clo.get("LO Details", "N/A")
            
            if clo_name.strip().lower() == clo_details_text.strip().lower():
                clo_overview_texts.append(f"CLO {clo_name}: {lo_details}")
            else:
                clo_overview_texts.append(f"CLO {clo_name} ({clo_details_text}): {lo_details}")
        
        chunk_text = clean_text(" ".join(clo_overview_texts))
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_id"] = str(uuid.uuid4())
        chunk_metadata["source_section"] = "clos_overview"
        chunk_metadata["title"] = f"Tổng quan CLO - {subject_code}"
        
        chunks.append({
            "type": "clos_overview",
            "content": chunk_text,
            "metadata": chunk_metadata
        })
        
        # Then create individual CLO chunks
        for i, clo in enumerate(syllabus_data["clos"]):
            clo_texts = []
            clo_name = clo.get("CLO Name", f"CLO {i+1}")
            clo_details_text = clo.get("CLO Details", "")
            lo_details = clo.get("LO Details", "N/A")
            
            # Add subject context to each CLO chunk
            clo_texts.append(f"Môn học {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ).")
            
            if clo_name.strip().lower() == clo_details_text.strip().lower():
                clo_texts.append(f"Chuẩn đầu ra {clo_name}: {lo_details}")
            else:
                clo_texts.append(f"Chuẩn đầu ra {clo_name} ({clo_details_text}): {lo_details}")
            
            # Find assessments related to this CLO
            if "assessments" in syllabus_data and syllabus_data["assessments"]:
                related_assessments = []
                for assessment in syllabus_data["assessments"]:
                    assessment_clos = assessment.get("CLO", "").split(", ")
                    if any(clo_ref in [clo_name, clo_details_text, f"CLO{clo_name}"] for clo_ref in assessment_clos):
                        related_assessments.append(f"{assessment.get('Category', '')} ({assessment.get('Weight', '')})")
                
                if related_assessments:
                    clo_texts.append(f"Được đánh giá thông qua: {', '.join(related_assessments)}.")
            
            chunk_text = clean_text(" ".join(clo_texts))
            
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

    # Chunk 3: Mỗi Buổi học (Session) - Thêm thông tin môn học vào mỗi session
    if "sessions" in syllabus_data and syllabus_data["sessions"]:
        # First create a sessions overview
        sessions_overview = []
        sessions_overview.append(f"Tổng quan các buổi học môn {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ):")
        
        # Group sessions by their topics for a more organized view
        topic_groups = {}
        for session in syllabus_data["sessions"]:
            session_number = session.get("Session", "N/A")
            topic = session.get("Topic", "N/A")
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(session_number)
        
        for topic, sessions in topic_groups.items():
            sessions_overview.append(f"Buổi {', '.join(sessions)}: {topic}")
        
        chunk_text = clean_text(" ".join(sessions_overview))
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_id"] = str(uuid.uuid4())
        chunk_metadata["source_section"] = "sessions_overview"
        chunk_metadata["title"] = f"Tổng quan buổi học - {subject_code}"
        
        chunks.append({
            "type": "sessions_overview",
            "content": chunk_text,
            "metadata": chunk_metadata
        })
        
        # Then create individual session chunks
        for session in syllabus_data["sessions"]:
            session_texts = []
            session_number = session.get("Session", "N/A")
            topic = session.get("Topic", "N/A")
            
            # Add subject context to each session chunk
            session_texts.append(f"Môn học {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ).")
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

    # Chunk 4: Đánh giá (Assessment) - Tạo tổng quan đánh giá và chi tiết từng đánh giá
    if "assessments" in syllabus_data and syllabus_data["assessments"]:
        # Create assessment overview first
        assessment_overview_texts = []
        assessment_overview_texts.append(f"Tổng quan đánh giá môn {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ):")
        
        # Group assessments by type
        ongoing_assessments = []
        final_assessments = []
        
        total_weight = 0
        for assessment in syllabus_data["assessments"]:
            assessment_type = assessment.get("Type", "").lower()
            category = assessment.get("Category", "")
            weight = assessment.get("Weight", "0%")
            
            # Extract numeric value from weight
            weight_value = float(weight.replace("%", "").strip()) if "%" in weight else 0
            total_weight += weight_value
            
            assessment_text = f"{category} ({weight})"
            if "final" in assessment_type:
                final_assessments.append(assessment_text)
            else:
                ongoing_assessments.append(assessment_text)
        
        if ongoing_assessments:
            assessment_overview_texts.append(f"Đánh giá quá trình: {', '.join(ongoing_assessments)}.")
        
        if final_assessments:
            assessment_overview_texts.append(f"Đánh giá cuối kỳ: {', '.join(final_assessments)}.")
        
        # Add completion criteria if available in the general details
        note = important_metadata.get("note", "")
        if "Completion Criteria" in note:
            completion_match = re.search(r"Completion Criteria:(.+?)(?=\n\d|\Z)", note, re.DOTALL)
            if completion_match:
                criteria = completion_match.group(1).strip()
                assessment_overview_texts.append(f"Tiêu chí hoàn thành: {criteria}.")
        
        chunk_text = clean_text(" ".join(assessment_overview_texts))
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_id"] = str(uuid.uuid4())
        chunk_metadata["source_section"] = "assessments_overview"
        chunk_metadata["title"] = f"Tổng quan đánh giá - {subject_code}"
        
        chunks.append({
            "type": "assessments_overview",
            "content": chunk_text,
            "metadata": chunk_metadata
        })
        
        # Then create individual assessment chunks
        for i, assessment in enumerate(syllabus_data["assessments"]):
            assessment_texts = []
            
            # Add subject context to each assessment chunk
            assessment_texts.append(f"Môn học {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ).")
            
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
            chunk_metadata["assessment_weight"] = weight
            chunk_metadata["title"] = f"Đánh giá {category} - {subject_code}"
            
            chunks.append({
                "type": "assessment",
                "content": chunk_text,
                "metadata": chunk_metadata
            })

    # Chunk 5: Tài liệu học tập (Materials) - Tạo tổng quan tài liệu và chi tiết từng tài liệu
    if "materials_table" in syllabus_data and syllabus_data["materials_table"]:
        # Create materials overview first
        materials_overview_texts = []
        materials_overview_texts.append(f"Tổng quan tài liệu học tập môn {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ):")
        
        for i, material in enumerate(syllabus_data["materials_table"]):
            desc = material.get("MaterialDescription", f"Tài liệu {i+1}")
            author = material.get("Author", "")
            publisher = material.get("Publisher", "")
            
            material_text = desc
            if author:
                material_text += f" (Tác giả: {author})"
            if publisher:
                material_text += f", {publisher}"
            
            materials_overview_texts.append(f"- {material_text}")
        
        chunk_text = clean_text(" ".join(materials_overview_texts))
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_id"] = str(uuid.uuid4())
        chunk_metadata["source_section"] = "materials_overview"
        chunk_metadata["title"] = f"Tổng quan tài liệu - {subject_code}"
        
        chunks.append({
            "type": "materials_overview",
            "content": chunk_text,
            "metadata": chunk_metadata
        })
        
        # Then create individual material chunks
        for i, material in enumerate(syllabus_data["materials_table"]):
            material_texts = []
            
            # Add subject context to each material chunk
            material_texts.append(f"Môn học {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ).")
            
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
        chunk_text = clean_text(f"Môn học {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ). Liên kết xem bản đồ Chuẩn đầu ra môn học (CLO) với Chuẩn đầu ra chương trình (PLO) của môn {subject_code} là: {clo_plo_link}")
        
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
    input_syllabus_file = "Syllabus_crawler/fpt_syllabus_data_appended_en.json"
    output_directory = "Chunk"
    
    all_chunks_file = process_all_syllabi(input_syllabus_file, output_directory)
    
    if all_chunks_file:
        print(f"Hoàn thành xử lý tất cả syllabus. Dữ liệu chunks được lưu tại: {all_chunks_file}")
    else:
        print("Không thể hoàn thành việc xử lý syllabus.")