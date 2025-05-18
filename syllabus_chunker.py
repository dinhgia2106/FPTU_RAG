"""
Enhanced Syllabus Chunker - Cải tiến hệ thống tạo chunk cho Syllabus FPT

Module này cung cấp các chức năng cải tiến để tạo chunk từ dữ liệu syllabus,
với metadata phong phú hơn và liên kết rõ ràng giữa các thực thể.
"""

import json
import re
import os
import uuid
from typing import Dict, List, Any, Optional, Union

class EntityType:
    """Định nghĩa các loại thực thể được hỗ trợ."""
    COURSE = "course"
    SESSION = "session"
    CLO = "clo"
    ASSESSMENT = "assessment"
    MATERIAL = "material"

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
    
    # Extract subject area keywords
    subject_areas = extract_subject_areas(syllabus_data)
    if subject_areas:
        metadata["subject_areas"] = subject_areas
    
    return metadata

def extract_subject_areas(syllabus_data):
    """Trích xuất các lĩnh vực chủ đề của môn học."""
    subject_areas = []
    
    # Danh sách các lĩnh vực chủ đề phổ biến
    common_areas = {
        "math": ["math", "mathematics", "toán học", "toán", "đại số", "giải tích", "xác suất", "thống kê"],
        "programming": ["programming", "lập trình", "coding", "development", "phát triển", "software", "phần mềm"],
        "ai": ["ai", "artificial intelligence", "trí tuệ nhân tạo", "machine learning", "học máy", "deep learning"],
        "database": ["database", "cơ sở dữ liệu", "data", "dữ liệu", "sql", "nosql"],
        "networking": ["network", "mạng", "internet", "web", "protocol", "giao thức"],
        "security": ["security", "bảo mật", "cryptography", "mã hóa", "privacy", "riêng tư"],
        "business": ["business", "kinh doanh", "marketing", "management", "quản lý", "finance", "tài chính"],
        "english": ["english", "tiếng anh", "language", "ngôn ngữ"],
        "chinese": ["chinese", "tiếng trung", "language", "ngôn ngữ"],
        "japanese": ["japanese", "tiếng nhật", "language", "ngôn ngữ"],
        "graphic": ["graphic", "đồ họa", "graphics", "animation", "hoạt hình"],
        "design": ["design", "thiết kế", "ui", "ux", "interface", "giao diện"],
        "mobile": ["mobile", "di động", "android", "ios", "app", "ứng dụng"],
        "vovinam": ["vovinam", "võ", "martial arts", "nghệ thuật chiến đấu"],
        "bartending": ["bartending", "pha chế", "cocktail", "đồ uống"],
        "cooking": ["cooking", "nấu ăn", "ẩm thực", "food", "đồ ăn"],
    }
    
    # Kiểm tra trong tên môn học và mô tả
    if "general_details" in syllabus_data:
        gd = syllabus_data["general_details"]
        syllabus_name = gd.get("Syllabus Name", "").lower()
        syllabus_english = gd.get("Syllabus English", "").lower()
        description = gd.get("Description", "").lower()
        
        combined_text = f"{syllabus_name} {syllabus_english} {description}"
        
        for area, keywords in common_areas.items():
            for keyword in keywords:
                if keyword in combined_text:
                    subject_areas.append(area)
                    break
    
    return list(set(subject_areas))  # Remove duplicates

def create_entity_id(entity_type, subject_code, identifier=None):
    """Tạo ID duy nhất cho thực thể."""
    if identifier:
        return f"{entity_type}_{subject_code}_{identifier}"
    else:
        return f"{entity_type}_{subject_code}"

def create_enhanced_chunks_from_syllabus(subject_code, syllabus_data):
    """Tạo các chunk từ dữ liệu syllabus với metadata phong phú và liên kết thực thể."""
    chunks = []
    
    # Extract important metadata to use across chunks
    important_metadata = extract_important_metadata(syllabus_data)
    
    # Create entity IDs
    course_entity_id = create_entity_id(EntityType.COURSE, subject_code)
    
    # Create session entity IDs
    session_entity_ids = {}
    if "sessions" in syllabus_data and syllabus_data["sessions"]:
        for session in syllabus_data["sessions"]:
            session_number = session.get("Session", "N/A")
            session_entity_ids[session_number] = create_entity_id(EntityType.SESSION, subject_code, session_number)
    
    # Create CLO entity IDs
    clo_entity_ids = {}
    if "clos" in syllabus_data and syllabus_data["clos"]:
        for i, clo in enumerate(syllabus_data["clos"]):
            clo_name = clo.get("CLO Name", f"{i+1}")
            clo_entity_ids[clo_name] = create_entity_id(EntityType.CLO, subject_code, f"CLO{clo_name}")
    
    # Create assessment entity IDs
    assessment_entity_ids = {}
    if "assessments" in syllabus_data and syllabus_data["assessments"]:
        for i, assessment in enumerate(syllabus_data["assessments"]):
            category = assessment.get("Category", "")
            assessment_entity_ids[category] = create_entity_id(EntityType.ASSESSMENT, subject_code, f"{i+1}")
    
    # Create material entity IDs
    material_entity_ids = {}
    if "materials_table" in syllabus_data and syllabus_data["materials_table"]:
        for i, material in enumerate(syllabus_data["materials_table"]):
            material_desc = material.get("MaterialDescription", "")
            material_entity_ids[material_desc] = create_entity_id(EntityType.MATERIAL, subject_code, f"{i+1}")
    
    base_metadata = {
        "subject_code": subject_code,
        "syllabus_id": syllabus_data.get("syllabus_id", "N/A"),
        "chunk_id": None,  # Will be updated for each chunk
        "entity_id": course_entity_id,  # Course entity ID
        "syllabus_name": important_metadata.get("syllabus_name", "N/A"),
        "syllabus_english": important_metadata.get("syllabus_english", ""),
        "credits": important_metadata.get("credits", "N/A"),
        "total_clos": important_metadata.get("total_clos", 0),
        "total_sessions": important_metadata.get("total_sessions", 0),
        "total_materials": important_metadata.get("total_materials", 0),
    }
    
    # Add subject areas if available
    if "subject_areas" in important_metadata:
        base_metadata["subject_areas"] = important_metadata["subject_areas"]

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
    chunk_metadata["entity_type"] = EntityType.COURSE
    
    # Add related entity IDs
    chunk_metadata["related_entities"] = {
        "sessions": list(session_entity_ids.values()),
        "clos": list(clo_entity_ids.values()),
        "assessments": list(assessment_entity_ids.values()),
        "materials": list(material_entity_ids.values())
    }
    
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
        chunk_metadata["entity_type"] = EntityType.COURSE
        
        # Add related entity IDs
        chunk_metadata["related_entities"] = {
            "sessions": list(session_entity_ids.values()),
            "clos": list(clo_entity_ids.values()),
            "assessments": list(assessment_entity_ids.values()),
            "materials": list(material_entity_ids.values())
        }
        
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
    chunk_metadata["entity_type"] = EntityType.COURSE
    
    # Add related entity IDs
    chunk_metadata["related_entities"] = {
        "sessions": list(session_entity_ids.values()),
        "clos": list(clo_entity_ids.values()),
        "assessments": list(assessment_entity_ids.values()),
        "materials": list(material_entity_ids.values())
    }
    
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
        chunk_metadata["entity_type"] = EntityType.COURSE
        
        # Add related entity IDs
        chunk_metadata["related_entities"] = {
            "clos": list(clo_entity_ids.values())
        }
        
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
            related_assessment_ids = []
            if "assessments" in syllabus_data and syllabus_data["assessments"]:
                related_assessments = []
                for assessment in syllabus_data["assessments"]:
                    assessment_clos = assessment.get("CLO", "").split(", ")
                    category = assessment.get("Category", "")
                    
                    if any(clo_ref in [clo_name, clo_details_text, f"CLO{clo_name}"] for clo_ref in assessment_clos):
                        related_assessments.append(f"{category} ({assessment.get('Weight', '')})")
                        if category in assessment_entity_ids:
                            related_assessment_ids.append(assessment_entity_ids[category])
                
                if related_assessments:
                    clo_texts.append(f"Được đánh giá thông qua: {', '.join(related_assessments)}.")
            
            # Find sessions related to this CLO
            related_session_ids = []
            if "sessions" in syllabus_data and syllabus_data["sessions"]:
                related_sessions = []
                for session in syllabus_data["sessions"]:
                    session_lo = session.get("LO", "").split(", ")
                    session_number = session.get("Session", "")
                    
                    if any(lo_ref in [clo_name, clo_details_text, f"CLO{clo_name}"] for lo_ref in session_lo):
                        related_sessions.append(session_number)
                        if session_number in session_entity_ids:
                            related_session_ids.append(session_entity_ids[session_number])
                
                if related_sessions:
                    clo_texts.append(f"Được dạy trong các buổi: {', '.join(related_sessions)}.")
            
            chunk_text = clean_text(" ".join(clo_texts))
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = str(uuid.uuid4())
            chunk_metadata["source_section"] = "clos"
            chunk_metadata["clo_id"] = clo_name
            chunk_metadata["title"] = f"CLO {clo_name} - {subject_code}"
            chunk_metadata["entity_type"] = EntityType.CLO
            chunk_metadata["entity_id"] = clo_entity_ids.get(clo_name, f"clo_{subject_code}_CLO{clo_name}")
            
            # Add related entity IDs
            chunk_metadata["related_entities"] = {
                "course": course_entity_id,
                "assessments": related_assessment_ids,
                "sessions": related_session_ids
            }
            
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
        chunk_metadata["entity_type"] = EntityType.COURSE
        
        # Add related entity IDs
        chunk_metadata["related_entities"] = {
            "sessions": list(session_entity_ids.values())
        }
        
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
            
            # Find related CLOs
            related_clo_ids = []
            lo_values = lo.split(", ")
            for clo_name in clo_entity_ids:
                if clo_name in lo_values or f"CLO{clo_name}" in lo_values:
                    related_clo_ids.append(clo_entity_ids[clo_name])
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = str(uuid.uuid4())
            chunk_metadata["source_section"] = "sessions"
            chunk_metadata["session_number"] = session_number
            chunk_metadata["title"] = f"Buổi {session_number} - {subject_code}"
            chunk_metadata["entity_type"] = EntityType.SESSION
            chunk_metadata["entity_id"] = session_entity_ids.get(session_number, f"session_{subject_code}_{session_number}")
            
            # Add related entity IDs
            chunk_metadata["related_entities"] = {
                "course": course_entity_id,
                "clos": related_clo_ids
            }
            
            # Add URLs if available
            if urls:
                chunk_metadata["urls"] = urls
            
            # Add download link if available
            if download_info:
                if isinstance(download_info, dict) and "link" in download_info:
                    chunk_metadata["download_link"] = download_info["link"]
                elif isinstance(download_info, str) and download_info.strip():
                    chunk_metadata["download_link"] = download_info
            
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
        
        for assessment in syllabus_data["assessments"]:
            category = assessment.get("Category", "N/A")
            weight = assessment.get("Weight", "N/A")
            assessment_overview_texts.append(f"{category}: {weight}")
        
        chunk_text = clean_text(" ".join(assessment_overview_texts))
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_id"] = str(uuid.uuid4())
        chunk_metadata["source_section"] = "assessments_overview"
        chunk_metadata["title"] = f"Tổng quan đánh giá - {subject_code}"
        chunk_metadata["entity_type"] = EntityType.COURSE
        
        # Add related entity IDs
        chunk_metadata["related_entities"] = {
            "assessments": list(assessment_entity_ids.values())
        }
        
        chunks.append({
            "type": "assessments_overview",
            "content": chunk_text,
            "metadata": chunk_metadata
        })
        
        # Then create individual assessment chunks
        for i, assessment in enumerate(syllabus_data["assessments"]):
            assessment_texts = []
            category = assessment.get("Category", "N/A")
            weight = assessment.get("Weight", "N/A")
            
            # Add subject context to each assessment chunk
            assessment_texts.append(f"Môn học {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ).")
            assessment_texts.append(f"Đánh giá: {category}. Trọng số: {weight}.")
            
            clo = assessment.get("CLO", "N/A")
            assessment_texts.append(f"Chuẩn đầu ra liên quan (CLO): {clo}.")
            
            part = assessment.get("Part")
            if part:
                assessment_texts.append(f"Phần: {part}.")
            
            duration = assessment.get("Duration")
            if duration:
                assessment_texts.append(f"Thời lượng: {duration}.")
            
            note = assessment.get("Note")
            if note:
                assessment_texts.append(f"Ghi chú: {note}.")
            
            chunk_text = clean_text(" ".join(assessment_texts))
            
            # Find related CLOs
            related_clo_ids = []
            clo_values = clo.split(", ")
            for clo_name in clo_entity_ids:
                if clo_name in clo_values or f"CLO{clo_name}" in clo_values:
                    related_clo_ids.append(clo_entity_ids[clo_name])
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = str(uuid.uuid4())
            chunk_metadata["source_section"] = "assessments"
            chunk_metadata["assessment_category"] = category
            chunk_metadata["assessment_weight"] = weight
            chunk_metadata["title"] = f"Đánh giá {category} - {subject_code}"
            chunk_metadata["entity_type"] = EntityType.ASSESSMENT
            chunk_metadata["entity_id"] = assessment_entity_ids.get(category, f"assessment_{subject_code}_{i+1}")
            
            # Add related entity IDs
            chunk_metadata["related_entities"] = {
                "course": course_entity_id,
                "clos": related_clo_ids
            }
            
            chunks.append({
                "type": "assessment",
                "content": chunk_text,
                "metadata": chunk_metadata
            })

    # Chunk 5: Tài liệu học tập (Material) - Tạo tổng quan tài liệu và chi tiết từng tài liệu
    if "materials_table" in syllabus_data and syllabus_data["materials_table"]:
        # Create materials overview first
        materials_overview_texts = []
        materials_overview_texts.append(f"Tổng quan tài liệu học tập môn {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ):")
        
        for material in syllabus_data["materials_table"]:
            material_desc = material.get("MaterialDescription", "N/A")
            author = material.get("Author", "")
            publisher = material.get("Publisher", "")
            
            material_text = f"{material_desc}"
            if author:
                material_text += f", tác giả: {author}"
            if publisher:
                material_text += f", nhà xuất bản: {publisher}"
            
            materials_overview_texts.append(material_text)
        
        chunk_text = clean_text(" ".join(materials_overview_texts))
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_id"] = str(uuid.uuid4())
        chunk_metadata["source_section"] = "materials_overview"
        chunk_metadata["title"] = f"Tổng quan tài liệu - {subject_code}"
        chunk_metadata["entity_type"] = EntityType.COURSE
        
        # Add related entity IDs
        chunk_metadata["related_entities"] = {
            "materials": list(material_entity_ids.values())
        }
        
        chunks.append({
            "type": "materials_overview",
            "content": chunk_text,
            "metadata": chunk_metadata
        })
        
        # Then create individual material chunks
        for i, material in enumerate(syllabus_data["materials_table"]):
            material_texts = []
            material_desc = material.get("MaterialDescription", "N/A")
            
            # Add subject context to each material chunk
            material_texts.append(f"Môn học {subject_code} - {important_metadata.get('syllabus_name', 'N/A')} ({important_metadata.get('credits', 'N/A')} tín chỉ).")
            material_texts.append(f"Tài liệu học tập: {material_desc}.")
            
            author = material.get("Author")
            if author:
                material_texts.append(f"Tác giả: {author}.")
            
            publisher = material.get("Publisher")
            if publisher:
                material_texts.append(f"Nhà xuất bản: {publisher}.")
            
            published_date = material.get("PublishedDate")
            if published_date:
                material_texts.append(f"Năm xuất bản: {published_date}.")
            
            edition = material.get("Edition")
            if edition:
                material_texts.append(f"Phiên bản: {edition}.")
            
            isbn = material.get("ISBN")
            if isbn:
                material_texts.append(f"ISBN: {isbn}.")
            
            note = material.get("Note")
            if note:
                material_texts.append(f"Ghi chú: {note}.")
            
            chunk_text = clean_text(" ".join(material_texts))
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = str(uuid.uuid4())
            chunk_metadata["source_section"] = "materials"
            chunk_metadata["material_description"] = material_desc
            chunk_metadata["title"] = f"Tài liệu {material_desc} - {subject_code}"
            chunk_metadata["entity_type"] = EntityType.MATERIAL
            chunk_metadata["entity_id"] = material_entity_ids.get(material_desc, f"material_{subject_code}_{i+1}")
            
            # Check if material is from Coursera
            is_coursera = False
            if "coursera" in material_desc.lower() or (note and "coursera" in note.lower()):
                is_coursera = True
                chunk_metadata["is_coursera"] = True
            
            # Add related entity IDs
            chunk_metadata["related_entities"] = {
                "course": course_entity_id
            }
            
            chunks.append({
                "type": "material",
                "content": chunk_text,
                "metadata": chunk_metadata
            })

    return chunks

def process_syllabus_data(syllabus_data_file, output_chunks_file):
    """Process syllabus data and create enhanced chunks."""
    try:
        with open(syllabus_data_file, 'r', encoding='utf-8') as f:
            syllabus_data = json.load(f)
    except Exception as e:
        print(f"Error loading syllabus data: {e}")
        return None
    
    all_chunks = []
    
    for subject_code, subject_data in syllabus_data.items():
        print(f"Processing subject: {subject_code}")
        subject_data["syllabus_id"] = subject_data.get("syllabus_id", "")
        chunks = create_enhanced_chunks_from_syllabus(subject_code, subject_data)
        all_chunks.extend(chunks)
        print(f"Created {len(chunks)} chunks for {subject_code}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_chunks_file), exist_ok=True)
    
    # Save chunks to file
    try:
        with open(output_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_chunks)} chunks to {output_chunks_file}")
        return output_chunks_file
    except Exception as e:
        print(f"Error saving chunks: {e}")
        return None

if __name__ == "__main__":
    input_file = "Syllabus_crawler/fpt_syllabus_data_appended_en.json"
    output_file = "Chunk/enhanced_chunks.json"
    
    process_syllabus_data(input_file, output_file)
