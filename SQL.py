import json
import os
import pymysql
from pymysql import Error
from dotenv import load_dotenv
import time

# Load biến môi trường từ file .env
load_dotenv()

# Hàm kết nối tới MySQL
def create_connection():
    try:
        conn = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            port=int(os.getenv('DB_PORT', 3306)),
            charset='utf8mb4'
        )
        print("Kết nối MySQL thành công")
        return conn
    except Error as e:
        print(f"Lỗi: {e}")
        return None

# Hàm tạo cơ sở dữ liệu và các bảng
def create_database_structure(conn):
    try:
        cursor = conn.cursor()
        
        cursor.execute("DROP TABLE IF EXISTS assessments_vi")
        cursor.execute("DROP TABLE IF EXISTS assessments_en")
        
        # Tạo bảng central_subjects - bảng trung tâm liên kết các phiên bản ngôn ngữ
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS central_subjects (
            subject_code VARCHAR(10) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng subjects_vi - lưu thông tin chung về môn học bằng tiếng Việt
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS subjects_vi (
            subject_code VARCHAR(10) PRIMARY KEY,
            subject_name NVARCHAR(255),
            credits INT,
            degree_level NVARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_code) REFERENCES central_subjects(subject_code),
            FULLTEXT INDEX ft_subject_name (subject_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng subjects_en - lưu thông tin chung về môn học bằng tiếng Anh
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS subjects_en (
            subject_code VARCHAR(10) PRIMARY KEY,
            subject_name NVARCHAR(255),
            credits INT,
            degree_level NVARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_code) REFERENCES central_subjects(subject_code),
            FULLTEXT INDEX ft_subject_name (subject_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng syllabi_vi - lưu thông tin về đề cương chi tiết bằng tiếng Việt
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS syllabi_vi (
            syllabus_id VARCHAR(10) PRIMARY KEY,
            subject_code VARCHAR(10),
            syllabus_name NVARCHAR(255),
            time_allocation TEXT,
            pre_requisite TEXT,
            description TEXT,
            student_tasks TEXT,
            tools TEXT,
            scoring_scale VARCHAR(10),
            decision_no VARCHAR(100),
            is_approved BOOLEAN,
            note TEXT,
            min_avg_mark_to_pass FLOAT,
            is_active BOOLEAN,
            approved_date DATE,
            page_url VARCHAR(255),
            materials_info VARCHAR(255),
            extraction_time DATETIME,
            FOREIGN KEY (subject_code) REFERENCES central_subjects(subject_code),
            INDEX idx_subject_code (subject_code),
            FULLTEXT INDEX ft_description (description),
            FULLTEXT INDEX ft_syllabus_name (syllabus_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng syllabi_en - lưu thông tin về đề cương chi tiết bằng tiếng Anh
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS syllabi_en (
            syllabus_id VARCHAR(10) PRIMARY KEY,
            subject_code VARCHAR(10),
            syllabus_name NVARCHAR(255),
            time_allocation TEXT,
            pre_requisite TEXT,
            description TEXT,
            student_tasks TEXT,
            tools TEXT,
            scoring_scale VARCHAR(10),
            decision_no VARCHAR(100),
            is_approved BOOLEAN,
            note TEXT,
            min_avg_mark_to_pass FLOAT,
            is_active BOOLEAN,
            approved_date DATE,
            page_url VARCHAR(255),
            materials_info VARCHAR(255),
            extraction_time DATETIME,
            FOREIGN KEY (subject_code) REFERENCES central_subjects(subject_code),
            INDEX idx_subject_code (subject_code),
            FULLTEXT INDEX ft_description (description),
            FULLTEXT INDEX ft_syllabus_name (syllabus_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng materials_vi - lưu thông tin về tài liệu học tập bằng tiếng Việt
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS materials_vi (
            id INT AUTO_INCREMENT PRIMARY KEY,
            syllabus_id VARCHAR(10),
            material_description TEXT,
            author TEXT,
            publisher TEXT,
            published_date VARCHAR(50),
            edition VARCHAR(50),
            isbn VARCHAR(50),
            is_main_material BOOLEAN,
            is_hard_copy BOOLEAN,
            is_online BOOLEAN,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (syllabus_id) REFERENCES syllabi_vi(syllabus_id),
            INDEX idx_syllabus_id (syllabus_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng materials_en - lưu thông tin về tài liệu học tập bằng tiếng Anh
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS materials_en (
            id INT AUTO_INCREMENT PRIMARY KEY,
            syllabus_id VARCHAR(10),
            material_description TEXT,
            author TEXT,
            publisher TEXT,
            published_date VARCHAR(50),
            edition VARCHAR(50),
            isbn VARCHAR(50),
            is_main_material BOOLEAN,
            is_hard_copy BOOLEAN,
            is_online BOOLEAN,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (syllabus_id) REFERENCES syllabi_en(syllabus_id),
            INDEX idx_syllabus_id (syllabus_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng learning_outcomes_vi - lưu thông tin về kết quả học tập bằng tiếng Việt
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_outcomes_vi (
            id INT AUTO_INCREMENT PRIMARY KEY,
            syllabus_id VARCHAR(10),
            clo_name VARCHAR(50),
            clo_details TEXT,
            lo_details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (syllabus_id) REFERENCES syllabi_vi(syllabus_id),
            INDEX idx_syllabus_id (syllabus_id),
            FULLTEXT INDEX ft_lo_details (lo_details)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng learning_outcomes_en - lưu thông tin về kết quả học tập bằng tiếng Anh
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_outcomes_en (
            id INT AUTO_INCREMENT PRIMARY KEY,
            syllabus_id VARCHAR(10),
            clo_name VARCHAR(50),
            clo_details TEXT,
            lo_details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (syllabus_id) REFERENCES syllabi_en(syllabus_id),
            INDEX idx_syllabus_id (syllabus_id),
            FULLTEXT INDEX ft_lo_details (lo_details)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng sessions_vi và sessions_en
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions_vi (
            id INT AUTO_INCREMENT PRIMARY KEY,
            syllabus_id VARCHAR(10),
            session_number VARCHAR(10),
            topic TEXT,
            learning_teaching_type VARCHAR(50),
            lo TEXT,
            itu VARCHAR(50),
            student_materials TEXT,
            student_tasks TEXT,
            urls TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (syllabus_id) REFERENCES syllabi_vi(syllabus_id),
            INDEX idx_syllabus_id (syllabus_id),
            FULLTEXT INDEX ft_topic (topic)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions_en (
            id INT AUTO_INCREMENT PRIMARY KEY,
            syllabus_id VARCHAR(10),
            session_number VARCHAR(10),
            topic TEXT,
            learning_teaching_type VARCHAR(50),
            lo TEXT,
            itu VARCHAR(50),
            student_materials TEXT,
            student_tasks TEXT,
            urls TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (syllabus_id) REFERENCES syllabi_en(syllabus_id),
            INDEX idx_syllabus_id (syllabus_id),
            FULLTEXT INDEX ft_topic (topic)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng assessments_vi và assessments_en
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments_vi (
            id INT AUTO_INCREMENT PRIMARY KEY,
            syllabus_id VARCHAR(10),
            category VARCHAR(100),
            type VARCHAR(100),
            part INT,
            weight FLOAT,
            completion_criteria TEXT,
            duration MEDIUMTEXT,
            clo TEXT,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (syllabus_id) REFERENCES syllabi_vi(syllabus_id),
            INDEX idx_syllabus_id (syllabus_id),
            FULLTEXT INDEX ft_note (note)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments_en (
            id INT AUTO_INCREMENT PRIMARY KEY,
            syllabus_id VARCHAR(10),
            category VARCHAR(100),
            type VARCHAR(100),
            part INT,
            weight FLOAT,
            completion_criteria TEXT,
            duration MEDIUMTEXT,
            clo TEXT,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (syllabus_id) REFERENCES syllabi_en(syllabus_id),
            INDEX idx_syllabus_id (syllabus_id),
            FULLTEXT INDEX ft_note (note)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng students (để chuẩn bị cho việc mở rộng trong tương lai)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id VARCHAR(20) PRIMARY KEY,
            full_name NVARCHAR(255),
            email VARCHAR(255),
            phone VARCHAR(20),
            date_of_birth DATE,
            address TEXT,
            enrollment_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_email (email),
            INDEX idx_enrollment_date (enrollment_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Tạo bảng student_subjects (để lưu thông tin về môn học của sinh viên)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_subjects (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(20),
            subject_code VARCHAR(10),
            syllabus_id VARCHAR(10),
            semester VARCHAR(20),
            academic_year VARCHAR(10),
            grade FLOAT,
            status VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            FOREIGN KEY (subject_code) REFERENCES central_subjects(subject_code),
            INDEX idx_student_id (student_id),
            INDEX idx_subject_code (subject_code),
            INDEX idx_semester (semester),
            INDEX idx_academic_year (academic_year)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        conn.commit()
        print("Tạo cấu trúc cơ sở dữ liệu thành công")
        
    except Error as e:
        print(f"Lỗi khi tạo cấu trúc cơ sở dữ liệu: {e}")

# Hàm để chèn dữ liệu từ file JSON vào cơ sở dữ liệu
def insert_data_from_json(conn, json_data, language):
    try:
        cursor = conn.cursor()
        
        for subject_code, subject_data in json_data.items():
            # Chèn vào bảng central_subjects nếu chưa có
            cursor.execute('''
            INSERT IGNORE INTO central_subjects (subject_code)
            VALUES (%s)
            ''', (subject_code,))
            
            # Lấy thông tin chung
            general_details = subject_data.get('general_details', {})
            
            if language == 'vi':
                # Chèn vào bảng subjects_vi
                subject_name = general_details.get('Syllabus Name', '')
                credits = general_details.get('NoCredit', 0)
                degree_level = general_details.get('Degree Level', '')
                
                cursor.execute('''
                INSERT INTO subjects_vi (subject_code, subject_name, credits, degree_level)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    subject_name = VALUES(subject_name),
                    credits = VALUES(credits),
                    degree_level = VALUES(degree_level)
                ''', (subject_code, subject_name, credits, degree_level))
                
                # Chèn vào bảng syllabi_vi
                syllabus_id = subject_data.get('syllabus_id', '')
                time_allocation = general_details.get('Time Allocation', '')
                pre_requisite = general_details.get('Pre-Requisite', '')
                description = general_details.get('Description', '')
                student_tasks = general_details.get('StudentTasks', '')
                tools = general_details.get('Tools', '')
                scoring_scale = general_details.get('Scoring Scale', '')
                decision_no = general_details.get('DecisionNo MM/dd/yyyy', '')
                is_approved = general_details.get('IsApproved', 'False') == 'True'
                note = general_details.get('Note', '')
                min_avg_mark_to_pass = general_details.get('MinAvgMarkToPass', 0)
                is_active = general_details.get('IsActive', 'False') == 'True'
                
                # Xử lý ngày tháng
                approved_date_str = general_details.get('ApprovedDate', '')
                approved_date = None
                if approved_date_str:
                    try:
                        parts = approved_date_str.split('/')
                        if len(parts) == 3:
                            month, day, year = parts
                            approved_date = f"{year}-{month}-{day}"
                    except Exception as e:
                        print(f"Lỗi xử lý ngày tháng {approved_date_str}: {e}")
                
                page_url = subject_data.get('page_url', '')
                materials_info = subject_data.get('materials_info', '')
                extraction_time = subject_data.get('extraction_time', '')
                # Thêm dòng này để làm sạch giá trị extraction_time
                if extraction_time:
                    extraction_time = clean_datetime(extraction_time)
                
                # Chèn syllabus tiếng Việt
                cursor.execute('''
                INSERT INTO syllabi_vi (
                    syllabus_id, subject_code, syllabus_name, time_allocation,
                    pre_requisite, description, student_tasks, tools, scoring_scale,
                    decision_no, is_approved, note, min_avg_mark_to_pass, is_active,
                    approved_date, page_url, materials_info, extraction_time
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    subject_code = VALUES(subject_code),
                    syllabus_name = VALUES(syllabus_name),
                    time_allocation = VALUES(time_allocation),
                    pre_requisite = VALUES(pre_requisite),
                    description = VALUES(description),
                    student_tasks = VALUES(student_tasks),
                    tools = VALUES(tools),
                    scoring_scale = VALUES(scoring_scale),
                    decision_no = VALUES(decision_no),
                    is_approved = VALUES(is_approved),
                    note = VALUES(note),
                    min_avg_mark_to_pass = VALUES(min_avg_mark_to_pass),
                    is_active = VALUES(is_active),
                    approved_date = VALUES(approved_date),
                    page_url = VALUES(page_url),
                    materials_info = VALUES(materials_info),
                    extraction_time = VALUES(extraction_time)
                ''', (
                    syllabus_id, subject_code, subject_name, time_allocation,
                    pre_requisite, description, student_tasks, tools, scoring_scale,
                    decision_no, is_approved, note, min_avg_mark_to_pass, is_active,
                    approved_date, page_url, materials_info, extraction_time
                ))
                
                # Chèn materials tiếng Việt
                materials = subject_data.get('materials_table', [])
                if materials and len(materials) > 0:
                    cursor.execute("DELETE FROM materials_vi WHERE syllabus_id = %s", (syllabus_id,))
                    for material in materials:
                        material_description = material.get('MaterialDescription', '')
                        author = material.get('Author', '')
                        publisher = material.get('Publisher', '')
                        published_date = material.get('PublishedDate', '')
                        edition = material.get('Edition', '')
                        isbn = material.get('ISBN', '')
                        is_main_material = material.get('IsMainMaterial', '') == 'True'
                        is_hard_copy = material.get('IsHardCopy', '') == 'True'
                        is_online = material.get('IsOnline', '') == 'True'
                        note = material.get('Note', '')
                        
                        cursor.execute('''
                        INSERT INTO materials_vi (
                            syllabus_id, material_description, author, publisher,
                            published_date, edition, isbn, is_main_material,
                            is_hard_copy, is_online, note
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            syllabus_id, material_description, author, publisher,
                            published_date, edition, isbn, is_main_material,
                            is_hard_copy, is_online, note
                        ))
                
                # Chèn CLOs tiếng Việt
                clos = subject_data.get('clos', [])
                if clos and len(clos) > 0:
                    cursor.execute("DELETE FROM learning_outcomes_vi WHERE syllabus_id = %s", (syllabus_id,))
                    for clo in clos:
                        clo_name = clo.get('CLO Name', '')
                        clo_details = clo.get('CLO Details', '')
                        lo_details = clo.get('LO Details', '')
                        
                        cursor.execute('''
                        INSERT INTO learning_outcomes_vi (
                            syllabus_id, clo_name, clo_details, lo_details
                        )
                        VALUES (%s, %s, %s, %s)
                        ''', (syllabus_id, clo_name, clo_details, lo_details))
                
                # Chèn sessions tiếng Việt
                sessions = subject_data.get('sessions', [])
                if sessions and len(sessions) > 0:
                    cursor.execute("DELETE FROM sessions_vi WHERE syllabus_id = %s", (syllabus_id,))
                    for session in sessions:
                        session_number = session.get('Session', '')
                        topic = session.get('Topic', '')
                        learning_teaching_type = session.get('Learning-Teaching Type', '')
                        lo = session.get('LO', '')
                        itu = session.get('ITU', '')
                        student_materials = session.get('Student Materials', '')
                        student_tasks = session.get('Student\'s Tasks', '')
                        urls = session.get('URLs', '')
                        
                        cursor.execute('''
                        INSERT INTO sessions_vi (
                            syllabus_id, session_number, topic, learning_teaching_type,
                            lo, itu, student_materials, student_tasks, urls
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            syllabus_id, session_number, topic, learning_teaching_type,
                            lo, itu, student_materials, student_tasks, urls
                        ))
                
                # Chèn assessments tiếng Việt
                assessments = subject_data.get('assessments', [])
                if assessments and len(assessments) > 0:
                    cursor.execute("DELETE FROM assessments_vi WHERE syllabus_id = %s", (syllabus_id,))
                    for assessment in assessments:
                        category = assessment.get('Category', '')
                        assess_type = assessment.get('Type', '')
                        part = assessment.get('Part', 0)
                        weight = assessment.get('Weight', 0)
                        weight = clean_weight(weight)
                        completion_criteria = assessment.get('CompletionCriteria', '')
                        duration = truncate_long_text(assessment.get('Duration', ''))
                        clo = assessment.get('CLO', '')
                        note = assessment.get('Note', '')
                        
                        cursor.execute('''
                        INSERT INTO assessments_vi (
                            syllabus_id, category, type, part, weight,
                            completion_criteria, duration, clo, note
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            syllabus_id, category, assess_type, part, weight,
                            completion_criteria, duration, clo, note
                        ))
                
            elif language == 'en':
                # Tương tự như trên nhưng cho tiếng Anh
                subject_name = general_details.get('Syllabus English', '')
                credits = general_details.get('NoCredit', 0)
                degree_level = general_details.get('Degree Level', '')
                
                cursor.execute('''
                INSERT INTO subjects_en (subject_code, subject_name, credits, degree_level)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    subject_name = VALUES(subject_name),
                    credits = VALUES(credits),
                    degree_level = VALUES(degree_level)
                ''', (subject_code, subject_name, credits, degree_level))
                
                # Chèn vào bảng syllabi_en và các bảng con khác tương tự như trên
                # (Code tương tự nhưng thay đổi tên bảng và trường phù hợp)
                # Chèn vào bảng syllabi_en
                syllabus_id = subject_data.get('syllabus_id', '')
                time_allocation = general_details.get('Time Allocation', '')
                pre_requisite = general_details.get('Pre-Requisite', '')
                description = general_details.get('Description', '')
                student_tasks = general_details.get('StudentTasks', '')
                tools = general_details.get('Tools', '')
                scoring_scale = general_details.get('Scoring Scale', '')
                decision_no = general_details.get('DecisionNo MM/dd/yyyy', '')
                is_approved = general_details.get('IsApproved', 'False') == 'True'
                note = general_details.get('Note', '')
                min_avg_mark_to_pass = general_details.get('MinAvgMarkToPass', 0)
                is_active = general_details.get('IsActive', 'False') == 'True'
                
                # Xử lý ngày tháng
                approved_date_str = general_details.get('ApprovedDate', '')
                approved_date = None
                if approved_date_str:
                    try:
                        parts = approved_date_str.split('/')
                        if len(parts) == 3:
                            month, day, year = parts
                            approved_date = f"{year}-{month}-{day}"
                    except Exception as e:
                        print(f"Lỗi xử lý ngày tháng {approved_date_str}: {e}")
                
                page_url = subject_data.get('page_url', '')
                materials_info = subject_data.get('materials_info', '')
                extraction_time = subject_data.get('extraction_time', '')
                # Thêm dòng này để làm sạch giá trị extraction_time
                if extraction_time:
                    extraction_time = clean_datetime(extraction_time)
                
                # Chèn syllabus tiếng Anh
                cursor.execute('''
                INSERT INTO syllabi_en (
                    syllabus_id, subject_code, syllabus_name, time_allocation,
                    pre_requisite, description, student_tasks, tools, scoring_scale,
                    decision_no, is_approved, note, min_avg_mark_to_pass, is_active,
                    approved_date, page_url, materials_info, extraction_time
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    subject_code = VALUES(subject_code),
                    syllabus_name = VALUES(syllabus_name),
                    time_allocation = VALUES(time_allocation),
                    pre_requisite = VALUES(pre_requisite),
                    description = VALUES(description),
                    student_tasks = VALUES(student_tasks),
                    tools = VALUES(tools),
                    scoring_scale = VALUES(scoring_scale),
                    decision_no = VALUES(decision_no),
                    is_approved = VALUES(is_approved),
                    note = VALUES(note),
                    min_avg_mark_to_pass = VALUES(min_avg_mark_to_pass),
                    is_active = VALUES(is_active),
                    approved_date = VALUES(approved_date),
                    page_url = VALUES(page_url),
                    materials_info = VALUES(materials_info),
                    extraction_time = VALUES(extraction_time)
                ''', (
                    syllabus_id, subject_code, subject_name, time_allocation,
                    pre_requisite, description, student_tasks, tools, scoring_scale,
                    decision_no, is_approved, note, min_avg_mark_to_pass, is_active,
                    approved_date, page_url, materials_info, extraction_time
                ))
                
                # Chèn materials tiếng Anh
                materials = subject_data.get('materials_table', [])
                if materials and len(materials) > 0:
                    cursor.execute("DELETE FROM materials_en WHERE syllabus_id = %s", (syllabus_id,))
                    for material in materials:
                        material_description = material.get('MaterialDescription', '')
                        author = material.get('Author', '')
                        publisher = material.get('Publisher', '')
                        published_date = material.get('PublishedDate', '')
                        edition = material.get('Edition', '')
                        isbn = material.get('ISBN', '')
                        is_main_material = material.get('IsMainMaterial', '') == 'True'
                        is_hard_copy = material.get('IsHardCopy', '') == 'True'
                        is_online = material.get('IsOnline', '') == 'True'
                        note = material.get('Note', '')
                        
                        cursor.execute('''
                        INSERT INTO materials_en (
                            syllabus_id, material_description, author, publisher,
                            published_date, edition, isbn, is_main_material,
                            is_hard_copy, is_online, note
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            syllabus_id, material_description, author, publisher,
                            published_date, edition, isbn, is_main_material,
                            is_hard_copy, is_online, note
                        ))
                
                # Chèn CLOs tiếng Anh
                clos = subject_data.get('clos', [])
                if clos and len(clos) > 0:
                    cursor.execute("DELETE FROM learning_outcomes_en WHERE syllabus_id = %s", (syllabus_id,))
                    for clo in clos:
                        clo_name = clo.get('CLO Name', '')
                        clo_details = clo.get('CLO Details', '')
                        lo_details = clo.get('LO Details', '')
                        
                        cursor.execute('''
                        INSERT INTO learning_outcomes_en (
                            syllabus_id, clo_name, clo_details, lo_details
                        )
                        VALUES (%s, %s, %s, %s)
                        ''', (syllabus_id, clo_name, clo_details, lo_details))
                
                # Chèn sessions tiếng Anh
                sessions = subject_data.get('sessions', [])
                if sessions and len(sessions) > 0:
                    cursor.execute("DELETE FROM sessions_en WHERE syllabus_id = %s", (syllabus_id,))
                    for session in sessions:
                        session_number = session.get('Session', '')
                        topic = session.get('Topic', '')
                        learning_teaching_type = session.get('Learning-Teaching Type', '')
                        lo = session.get('LO', '')
                        itu = session.get('ITU', '')
                        student_materials = session.get('Student Materials', '')
                        student_tasks = session.get('Student\'s Tasks', '')
                        urls = session.get('URLs', '')
                        
                        cursor.execute('''
                        INSERT INTO sessions_en (
                            syllabus_id, session_number, topic, learning_teaching_type,
                            lo, itu, student_materials, student_tasks, urls
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            syllabus_id, session_number, topic, learning_teaching_type,
                            lo, itu, student_materials, student_tasks, urls
                        ))
                
                # Chèn assessments tiếng Anh
                assessments = subject_data.get('assessments', [])
                if assessments and len(assessments) > 0:
                    cursor.execute("DELETE FROM assessments_en WHERE syllabus_id = %s", (syllabus_id,))
                    for assessment in assessments:
                        category = assessment.get('Category', '')
                        assess_type = assessment.get('Type', '')
                        part = assessment.get('Part', 0)
                        weight = assessment.get('Weight', 0)
                        weight = clean_weight(weight)
                        completion_criteria = assessment.get('CompletionCriteria', '')
                        duration = truncate_long_text(assessment.get('Duration', ''))
                        clo = assessment.get('CLO', '')
                        note = assessment.get('Note', '')
                        
                        cursor.execute('''
                        INSERT INTO assessments_en (
                            syllabus_id, category, type, part, weight,
                            completion_criteria, duration, clo, note
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            syllabus_id, category, assess_type, part, weight,
                            completion_criteria, duration, clo, note
                        ))
        
        conn.commit()
        print(f"Chèn dữ liệu {language} vào cơ sở dữ liệu thành công")
        
    except Error as e:
        conn.rollback()
        print(f"Lỗi khi chèn dữ liệu {language}: {e}")

def clean_datetime(datetime_str):
    # Loại bỏ khoảng trắng không cần thiết
    datetime_str = datetime_str.replace(" ", "")
    
    # Loại bỏ 'Z' hoặc 'z' ở cuối
    if (datetime_str.endswith('Z') or datetime_str.endswith('z')):
        datetime_str = datetime_str[:-1]
    
    # Thay thế 'T' bằng khoảng trắng
    datetime_str = datetime_str.replace('T', ' ')
    
    # Xử lý phần mili-giây
    if '.' in datetime_str:
        datetime_parts = datetime_str.split('.')
        datetime_str = datetime_parts[0]  # Loại bỏ mili-giây
    
    return datetime_str

# Thêm hàm để xử lý giá trị weight
def clean_weight(weight_str):
    if isinstance(weight_str, str):
        # Loại bỏ ký tự % nếu có
        weight_str = weight_str.replace('%', '')
        try:
            # Chuyển đổi thành số thực
            return float(weight_str)
        except ValueError:
            # Trả về 0 nếu không thể chuyển đổi
            return 0
    return weight_str or 0

def truncate_long_text(text, max_length=16000000):
    """Truncate text to prevent data length errors"""
    if text and isinstance(text, str) and len(text) > max_length:
        return text[:max_length]
    return text

# Xử lý chính
def main():
    # Tạo kết nối đến MySQL
    conn = create_connection()
    if not conn:
        return
    
    # Tạo cấu trúc cơ sở dữ liệu
    create_database_structure(conn)
    
    # Đọc file JSON tiếng Việt
    try:
        print("Đang xử lý file dữ liệu tiếng Việt...")
        with open('Syllabus_crawler/fpt_syllabus_data_appended_vi.json', 'r', encoding='utf-8') as file:
            json_data_vi = json.load(file)
            
            # Chèn dữ liệu vào cơ sở dữ liệu
            insert_data_from_json(conn, json_data_vi, 'vi')
        
        print("Đang xử lý file dữ liệu tiếng Anh...")
        with open('Syllabus_crawler/fpt_syllabus_data_appended_en.json', 'r', encoding='utf-8') as file:
            json_data_en = json.load(file)
            
            # Chèn dữ liệu vào cơ sở dữ liệu
            insert_data_from_json(conn, json_data_en, 'en')
    
    except Exception as e:
        print(f"Lỗi khi đọc file JSON: {e}")
    
    finally:
        if conn:  # Simply check if conn exists rather than using is_connected()
            conn.close()
            print("Kết nối MySQL đã đóng")

if __name__ == "__main__":
    main()