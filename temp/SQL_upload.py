import os
import pymysql
from pymysql import Error
from dotenv import load_dotenv
import json
import csv

load_dotenv()


def create_connection():
    try:
        conn = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            port=int(os.getenv('DB_PORT', 3306)),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.Cursor
        )
        print("Kết nối MySQL thành công")
        return conn
    except Error as e:
        print(f"Kết nối thất bại. Lỗi: {e}")
        return None


def create_tables(conn):
    """Tạo các bảng trong database nếu chưa tồn tại."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Majors (
                major_code VARCHAR(50) PRIMARY KEY,
                curriculum_title VARCHAR(255),
                curriculum_url VARCHAR(512)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng Majors (hoặc đã tồn tại).")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Courses (
                course_id VARCHAR(50) PRIMARY KEY,
                major_code VARCHAR(50),
                course_name_from_curriculum TEXT,
                semester VARCHAR(50), /* Có thể là số hoặc text */
                combo_name_from_curriculum TEXT,
                combo_short_name_from_curriculum VARCHAR(255),
                course_type_guess VARCHAR(50),
                FOREIGN KEY (major_code) REFERENCES Majors(major_code) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng Courses (hoặc đã tồn tại).")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Syllabuses (
                syllabus_id VARCHAR(50) PRIMARY KEY,
                course_id VARCHAR(50),
                syllabus_url VARCHAR(512),
                syllabus_id_from_url VARCHAR(50),
                title TEXT,
                english_title TEXT,
                subject_code_on_page VARCHAR(50),
                credits VARCHAR(20),
                degree_level VARCHAR(100),
                time_allocation TEXT,
                prerequisites TEXT,
                description LONGTEXT,
                student_tasks LONGTEXT,
                tools TEXT,
                scoring_scale VARCHAR(50),
                min_avg_mark_to_pass VARCHAR(50),
                is_approved VARCHAR(10),
                is_active VARCHAR(10),
                decision_no VARCHAR(255),
                approved_date VARCHAR(50),
                note LONGTEXT,
                has_download_materials_button BOOLEAN,
                FOREIGN KEY (course_id) REFERENCES Courses(course_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng Syllabuses (hoặc đã tồn tại).")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Materials (
                material_id INT AUTO_INCREMENT PRIMARY KEY,
                syllabus_id VARCHAR(50),
                description TEXT,
                author VARCHAR(255),
                publisher VARCHAR(255),
                published_date VARCHAR(50),
                edition VARCHAR(100),
                isbn VARCHAR(50),
                is_main_material BOOLEAN,
                is_hard_copy BOOLEAN,
                is_online BOOLEAN,
                note TEXT,
                FOREIGN KEY (syllabus_id) REFERENCES Syllabuses(syllabus_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng Materials (hoặc đã tồn tại).")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS LearningOutcomes (
                outcome_auto_id INT AUTO_INCREMENT PRIMARY KEY,
                syllabus_id VARCHAR(50),
                outcome_id VARCHAR(50), /* CLO1, CLO2 etc. */
                details TEXT,
                UNIQUE KEY unique_syllabus_outcome (syllabus_id, outcome_id),
                FOREIGN KEY (syllabus_id) REFERENCES Syllabuses(syllabus_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng LearningOutcomes (hoặc đã tồn tại).")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Schedules (
                schedule_id INT AUTO_INCREMENT PRIMARY KEY,
                syllabus_id VARCHAR(50),
                session TEXT,
                topic TEXT,
                teaching_type TEXT,
                learning_outcomes_text TEXT, /* Raw text of CLOs, e.g., "CLO1, CLO2" */
                itu TEXT,
                materials TEXT,
                tasks TEXT,
                download_link VARCHAR(1024),
                urls TEXT,
                FOREIGN KEY (syllabus_id) REFERENCES Syllabuses(syllabus_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng Schedules (hoặc đã tồn tại).")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ScheduleLearningOutcomes (
                schedule_id INT,
                outcome_auto_id INT,
                PRIMARY KEY (schedule_id, outcome_auto_id),
                FOREIGN KEY (schedule_id) REFERENCES Schedules(schedule_id) ON DELETE CASCADE,
                FOREIGN KEY (outcome_auto_id) REFERENCES LearningOutcomes(outcome_auto_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng ScheduleLearningOutcomes (hoặc đã tồn tại).")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Assessments (
                assessment_id INT AUTO_INCREMENT PRIMARY KEY,
                syllabus_id VARCHAR(50),
                category VARCHAR(255),
                type VARCHAR(255),
                part TEXT,
                weight VARCHAR(50),
                clos_text TEXT, /* Raw text of CLOs */
                duration VARCHAR(100),
                question_type TEXT,
                no_question TEXT,
                knowledge_and_skill TEXT,
                grading_guide LONGTEXT,
                note TEXT,
                completion_criteria TEXT,
                FOREIGN KEY (syllabus_id) REFERENCES Syllabuses(syllabus_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng Assessments (hoặc đã tồn tại).")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS AssessmentCLOs (
                assessment_id INT,
                outcome_auto_id INT,
                PRIMARY KEY (assessment_id, outcome_auto_id),
                FOREIGN KEY (assessment_id) REFERENCES Assessments(assessment_id) ON DELETE CASCADE,
                FOREIGN KEY (outcome_auto_id) REFERENCES LearningOutcomes(outcome_auto_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng AssessmentCLOs (hoặc đã tồn tại).")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Students (
                roll_number VARCHAR(50) PRIMARY KEY,
                last_name VARCHAR(100),
                middle_name VARCHAR(100),
                first_name VARCHAR(100),
                fullname VARCHAR(255),
                email VARCHAR(255) UNIQUE,
                major_code VARCHAR(50),
                FOREIGN KEY (major_code) REFERENCES Majors(major_code) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Đã tạo bảng Students (hoặc đã tồn tại).")

        conn.commit()
        print("Tất cả các bảng đã được tạo thành công (hoặc đã tồn tại).")
    except Error as e:
        print(f"Lỗi khi tạo bảng: {e}")
        conn.rollback()
    finally:
        if cursor:
            cursor.close()


def insert_json_data(conn, json_file_path="Data/flm_data_AI.json"):
    """Chèn dữ liệu từ file JSON vào các bảng tương ứng."""
    cursor = conn.cursor()
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Chèn Major
        major_code = data.get("major_code_input")
        if major_code:
            cursor.execute("""
                INSERT INTO Majors (major_code, curriculum_title, curriculum_url)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                curriculum_title = VALUES(curriculum_title),
                curriculum_url = VALUES(curriculum_url);
            """, (major_code, data.get("curriculum_title_on_page"), data.get("curriculum_url")))
            print(f"Đã chèn/cập nhật Major: {major_code}")

            # Cache Course_id đã được thêm để tránh lỗi khóa ngoại khi thêm syllabus
            inserted_courses = set()

            for syllabus_data in data.get("syllabuses", []):
                metadata = syllabus_data.get("metadata", {})
                course_id = metadata.get("course_id")

                if course_id and course_id not in inserted_courses:
                    cursor.execute("""
                        INSERT INTO Courses (course_id, major_code, course_name_from_curriculum, semester,
                                           combo_name_from_curriculum, combo_short_name_from_curriculum, course_type_guess)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        major_code = VALUES(major_code),
                        course_name_from_curriculum = VALUES(course_name_from_curriculum),
                        semester = VALUES(semester),
                        combo_name_from_curriculum = VALUES(combo_name_from_curriculum),
                        combo_short_name_from_curriculum = VALUES(combo_short_name_from_curriculum),
                        course_type_guess = VALUES(course_type_guess);
                    """, (
                        course_id,
                        major_code,
                        metadata.get("course_name_from_curriculum"),
                        # Đảm bảo là chuỗi
                        str(metadata.get("semester_from_curriculum")),
                        metadata.get("combo_name_from_curriculum"),
                        metadata.get("combo_short_name_from_curriculum"),
                        metadata.get("course_type_guess")
                    ))
                    inserted_courses.add(course_id)
                    print(f"  Đã chèn/cập nhật Course: {course_id}")

                syllabus_id = metadata.get(
                    "syllabus_id") or metadata.get("syllabus_id_from_url")
                if not syllabus_id:
                    print(
                        f"  Bỏ qua syllabus không có ID cho course: {course_id}")
                    continue

                cursor.execute("""
                    INSERT INTO Syllabuses (syllabus_id, course_id, syllabus_url, syllabus_id_from_url, title, english_title,
                                           subject_code_on_page, credits, degree_level, time_allocation, prerequisites,
                                           description, student_tasks, tools, scoring_scale, min_avg_mark_to_pass,
                                           is_approved, is_active, decision_no, approved_date, note, has_download_materials_button)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    course_id = VALUES(course_id), syllabus_url = VALUES(syllabus_url),
                    syllabus_id_from_url = VALUES(syllabus_id_from_url), title = VALUES(title),
                    english_title = VALUES(english_title), subject_code_on_page = VALUES(subject_code_on_page),
                    credits = VALUES(credits), degree_level = VALUES(degree_level),
                    time_allocation = VALUES(time_allocation), prerequisites = VALUES(prerequisites),
                    description = VALUES(description), student_tasks = VALUES(student_tasks),
                    tools = VALUES(tools), scoring_scale = VALUES(scoring_scale),
                    min_avg_mark_to_pass = VALUES(min_avg_mark_to_pass), is_approved = VALUES(is_approved),
                    is_active = VALUES(is_active), decision_no = VALUES(decision_no),
                    approved_date = VALUES(approved_date), note = VALUES(note),
                    has_download_materials_button = VALUES(has_download_materials_button);
                """, (
                    syllabus_id, course_id, metadata.get(
                        "syllabus_url"), metadata.get("syllabus_id_from_url"),
                    metadata.get("title"), metadata.get(
                        "english_title"), metadata.get("subject_code_on_page"),
                    str(metadata.get("credits")), metadata.get(
                        "degree_level"), metadata.get("time_allocation"),
                    metadata.get("prerequisites"), metadata.get(
                        "description"), metadata.get("student_tasks"),
                    metadata.get("tools"), str(metadata.get("scoring_scale")), str(
                        metadata.get("min_avg_mark_to_pass")),
                    metadata.get("is_approved"), metadata.get(
                        "is_active"), metadata.get("decision_no"),
                    metadata.get("approved_date"), metadata.get("note"),
                    syllabus_data.get("has_download_materials_button", False)
                ))
                print(
                    f"    Đã chèn/cập nhật Syllabus: {syllabus_id} cho Course: {course_id}")

                # Chèn Materials
                for mat in syllabus_data.get("materials", []):
                    cursor.execute("""
                        INSERT INTO Materials (syllabus_id, description, author, publisher, published_date, edition,
                                               isbn, is_main_material, is_hard_copy, is_online, note)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, (
                        syllabus_id, mat.get("description"), mat.get(
                            "author"), mat.get("publisher"),
                        mat.get("published_date"), mat.get(
                            "edition"), mat.get("isbn"),
                        mat.get("is_main_material", False), mat.get(
                            "is_hard_copy", False),
                        mat.get("is_online", False), mat.get("note")
                    ))
                # print(f"      Đã chèn {len(syllabus_data.get('materials', []))} Materials cho Syllabus: {syllabus_id}")

                # Chèn LearningOutcomes và cache outcome_auto_id
                outcome_id_to_auto_id = {}
                for lo in syllabus_data.get("learning_outcomes", []):
                    outcome_id_val = lo.get("id")
                    if outcome_id_val:  # Chỉ chèn nếu có id
                        cursor.execute("""
                            INSERT INTO LearningOutcomes (syllabus_id, outcome_id, details)
                            VALUES (%s, %s, %s)
                            ON DUPLICATE KEY UPDATE details = VALUES(details);
                        """, (syllabus_id, outcome_id_val, lo.get("details")))
                        # Lấy outcome_auto_id vừa được chèn hoặc đã tồn tại
                        # MySQL không trả về lastrowid một cách đáng tin cậy với ON DUPLICATE KEY UPDATE
                        # nên chúng ta query lại dựa trên unique key
                        cursor.execute(
                            "SELECT outcome_auto_id FROM LearningOutcomes WHERE syllabus_id = %s AND outcome_id = %s", (syllabus_id, outcome_id_val))
                        result = cursor.fetchone()
                        if result:
                            outcome_id_to_auto_id[outcome_id_val] = result[0]
                # print(f"      Đã chèn {len(syllabus_data.get('learning_outcomes', []))} LearningOutcomes cho Syllabus: {syllabus_id}")

                # Chèn Schedules và liên kết ScheduleLearningOutcomes
                for sched in syllabus_data.get("schedule", []):
                    learning_outcomes_text_sched = ", ".join(
                        sched.get("learning_outcomes", []))
                    cursor.execute("""
                        INSERT INTO Schedules (syllabus_id, session, topic, teaching_type, learning_outcomes_text,
                                               itu, materials, tasks, download_link, urls)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, (
                        syllabus_id, sched.get("session"), sched.get(
                            "topic"), sched.get("teaching_type"),
                        learning_outcomes_text_sched, sched.get(
                            "itu"), sched.get("materials"),
                        sched.get("tasks"), sched.get(
                            "download_link"), sched.get("urls")
                    ))
                    schedule_auto_id = cursor.lastrowid
                    if schedule_auto_id:
                        for clo_text in sched.get("learning_outcomes", []):
                            outcome_text_clean = clo_text.strip()
                            if outcome_text_clean in outcome_id_to_auto_id:
                                outcome_auto_id = outcome_id_to_auto_id[outcome_text_clean]
                                cursor.execute("""
                                    INSERT INTO ScheduleLearningOutcomes (schedule_id, outcome_auto_id)
                                    VALUES (%s, %s)
                                    ON DUPLICATE KEY UPDATE schedule_id = VALUES(schedule_id);
                                """, (schedule_auto_id, outcome_auto_id))
                # print(f"      Đã chèn {len(syllabus_data.get('schedule', []))} Schedules cho Syllabus: {syllabus_id}")

                # Chèn Assessments và liên kết AssessmentCLOs
                for asm in syllabus_data.get("assessments", []):
                    clos_text_asm = ", ".join(asm.get("clos", []))
                    cursor.execute("""
                        INSERT INTO Assessments (syllabus_id, category, type, part, weight, clos_text, duration,
                                                 question_type, no_question, knowledge_and_skill, grading_guide, note,
                                                 completion_criteria)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, (
                        syllabus_id, asm.get("category"), asm.get(
                            "type"), asm.get("part"),
                        str(asm.get("weight")), clos_text_asm, asm.get(
                            "duration"), asm.get("question_type"),
                        str(asm.get("no_question")), asm.get(
                            "knowledge_and_skill"), asm.get("grading_guide"),
                        asm.get("note"), asm.get("completion_criteria")
                    ))
                    assessment_auto_id = cursor.lastrowid
                    if assessment_auto_id:
                        for clo_text in asm.get("clos", []):
                            outcome_text_clean = clo_text.strip()
                            if outcome_text_clean in outcome_id_to_auto_id:
                                outcome_auto_id = outcome_id_to_auto_id[outcome_text_clean]
                                cursor.execute("""
                                    INSERT INTO AssessmentCLOs (assessment_id, outcome_auto_id)
                                    VALUES (%s, %s)
                                    ON DUPLICATE KEY UPDATE assessment_id = VALUES(assessment_id);
                                """, (assessment_auto_id, outcome_auto_id))
                # print(f"      Đã chèn {len(syllabus_data.get('assessments', []))} Assessments cho Syllabus: {syllabus_id}")

        conn.commit()
        print("Đã chèn thành công dữ liệu từ JSON.")
    except Error as e:
        print(f"Lỗi khi chèn dữ liệu JSON: {e}")
        conn.rollback()
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {json_file_path}")
    except json.JSONDecodeError:
        print(f"Lỗi: File {json_file_path} không phải là JSON hợp lệ.")
    finally:
        if cursor:
            cursor.close()


def insert_csv_data(conn, csv_file_path="Data/Students_AI17D.csv"):
    """Chèn dữ liệu từ file CSV vào bảng Students."""
    cursor = conn.cursor()
    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig để xử lý BOM nếu có
            reader = csv.DictReader(f)
            students_data = list(reader)

        if not students_data:
            print(f"Không có dữ liệu trong file CSV: {csv_file_path}")
            return

        for student in students_data:
            major_code = student.get("Major")
            # Kiểm tra xem Major có tồn tại trong bảng Majors không
            cursor.execute(
                "SELECT major_code FROM Majors WHERE major_code = %s", (major_code,))
            if not cursor.fetchone() and major_code:
                # Nếu Major chưa có, có thể bạn muốn thêm nó hoặc báo lỗi
                # Hiện tại sẽ bỏ qua sinh viên này nếu major_code không tồn tại
                print(
                    f"  Cảnh báo: Major '{major_code}' cho sinh viên '{student.get('RollNumber')}' không tồn tại trong bảng Majors. Bỏ qua sinh viên này.")
                # Hoặc tự động thêm major nếu cần:
                # cursor.execute("INSERT INTO Majors (major_code) VALUES (%s) ON DUPLICATE KEY UPDATE major_code=major_code;", (major_code,))
                # print(f"  Tự động thêm Major: {major_code}")
                continue  # Bỏ qua nếu major không tồn tại và không tự động thêm

            cursor.execute("""
                INSERT INTO Students (roll_number, last_name, middle_name, first_name, fullname, email, major_code)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                last_name = VALUES(last_name),
                middle_name = VALUES(middle_name),
                first_name = VALUES(first_name),
                fullname = VALUES(fullname),
                email = VALUES(email),
                major_code = VALUES(major_code);
            """, (
                student.get("RollNumber"), student.get(
                    "LastName"), student.get("MiddleName"),
                student.get("FirstName"), student.get(
                    "Fullname"), student.get("Email"),
                major_code
            ))
        conn.commit()
        print(f"Đã chèn thành công {len(students_data)} sinh viên từ CSV.")
    except Error as e:
        print(f"Lỗi khi chèn dữ liệu CSV: {e}")
        conn.rollback()
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_file_path}")
    except Exception as ex:  # Bắt các lỗi chung khác như lỗi đọc CSV
        print(f"Lỗi không xác định khi xử lý file CSV {csv_file_path}: {ex}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()


def main():
    conn = create_connection()
    if conn:
        try:
            create_tables(conn)
            # Bạn có thể thay đổi đường dẫn file ở đây nếu cần
            insert_json_data(conn, "Data/flm_data_AI.json")
            insert_csv_data(conn, "Data/Students_AI17D.csv")
        finally:
            conn.close()
            print("Đã đóng kết nối MySQL.")


if __name__ == "__main__":
    main()
