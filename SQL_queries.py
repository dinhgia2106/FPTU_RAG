import json
import os
import pymysql
from pymysql import Error
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate
import sys

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
        print(f"Lỗi kết nối: {e}")
        return None

# Function to display query results in a nice table
def display_results(results, headers=None):
    if not results:
        print("Không có kết quả nào được tìm thấy.")
        return
        
    if headers:
        df = pd.DataFrame(results, columns=headers)
    else:
        df = pd.DataFrame(results)
        
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
    print(f"Tìm thấy {len(results)} kết quả.")

# 1. Tìm kiếm môn học theo mã môn
def search_subject_by_code(conn, subject_code, language='vi'):
    try:
        cursor = conn.cursor()
        table_prefix = f"subjects_{language}"
        
        query = f"""
        SELECT s.subject_code, s.subject_name, s.credits, s.degree_level
        FROM {table_prefix} s
        WHERE s.subject_code = %s
        """
        
        cursor.execute(query, (subject_code,))
        results = cursor.fetchall()
        
        headers = ["Mã môn", "Tên môn học", "Số tín chỉ", "Bậc đào tạo"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# 2. Tìm kiếm môn học theo từ khóa (sử dụng FULLTEXT INDEX)
def search_subjects_by_keyword(conn, keyword, language='vi'):
    try:
        cursor = conn.cursor()
        table_prefix = f"subjects_{language}"
        
        query = f"""
        SELECT s.subject_code, s.subject_name, s.credits, s.degree_level
        FROM {table_prefix} s
        WHERE MATCH(s.subject_name) AGAINST(%s IN BOOLEAN MODE)
        ORDER BY s.subject_code
        """
        
        cursor.execute(query, (keyword + "*",))
        results = cursor.fetchall()
        
        headers = ["Mã môn", "Tên môn học", "Số tín chỉ", "Bậc đào tạo"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# 3. Tìm kiếm thông tin chi tiết về đề cương môn học
def search_syllabus_details(conn, syllabus_id, language='vi'):
    try:
        cursor = conn.cursor()
        table_prefix = f"syllabi_{language}"
        
        query = f"""
        SELECT syl.syllabus_id, syl.subject_code, syl.syllabus_name, 
               syl.description, syl.pre_requisite, syl.time_allocation,
               syl.min_avg_mark_to_pass, syl.is_approved, syl.approved_date
        FROM {table_prefix} syl
        WHERE syl.syllabus_id = %s
        """
        
        cursor.execute(query, (syllabus_id,))
        results = cursor.fetchall()
        
        headers = ["ID Đề cương", "Mã môn", "Tên đề cương", "Mô tả", "Điều kiện tiên quyết", 
                  "Phân bổ thời gian", "Điểm chuẩn đạt", "Đã duyệt", "Ngày duyệt"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# 4. Tìm tất cả đề cương của một môn học
def search_syllabi_by_subject(conn, subject_code, language='vi'):
    try:
        cursor = conn.cursor()
        table_prefix = f"syllabi_{language}"
        
        query = f"""
        SELECT syl.syllabus_id, syl.syllabus_name, syl.is_active, syl.approved_date,
               syl.decision_no, syl.page_url
        FROM {table_prefix} syl
        WHERE syl.subject_code = %s
        ORDER BY syl.approved_date DESC
        """
        
        cursor.execute(query, (subject_code,))
        results = cursor.fetchall()
        
        headers = ["ID Đề cương", "Tên đề cương", "Đang áp dụng", "Ngày duyệt", "Số quyết định", "URL"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# 5. Tìm tất cả tài liệu học tập của một đề cương
def search_materials(conn, syllabus_id, language='vi'):
    try:
        cursor = conn.cursor()
        table_prefix = f"materials_{language}"
        
        query = f"""
        SELECT m.material_description, m.author, m.publisher, m.published_date,
               m.edition, m.is_main_material, m.is_hard_copy, m.is_online
        FROM {table_prefix} m
        WHERE m.syllabus_id = %s
        ORDER BY m.is_main_material DESC
        """
        
        cursor.execute(query, (syllabus_id,))
        results = cursor.fetchall()
        
        headers = ["Mô tả tài liệu", "Tác giả", "Nhà xuất bản", "Năm xuất bản", 
                  "Phiên bản", "Tài liệu chính", "Bản cứng", "Trực tuyến"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# 6. Tìm kết quả học tập của một đề cương
def search_learning_outcomes(conn, syllabus_id, language='vi'):
    try:
        cursor = conn.cursor()
        table_prefix = f"learning_outcomes_{language}"
        
        query = f"""
        SELECT lo.clo_name, lo.clo_details, lo.lo_details
        FROM {table_prefix} lo
        WHERE lo.syllabus_id = %s
        """
        
        cursor.execute(query, (syllabus_id,))
        results = cursor.fetchall()
        
        headers = ["Mã CLO", "Chi tiết CLO", "Chi tiết LO"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# 7. Tìm phiên học và nội dung của một đề cương
def search_sessions(conn, syllabus_id, language='vi'):
    try:
        cursor = conn.cursor()
        table_prefix = f"sessions_{language}"
        
        query = f"""
        SELECT s.session_number, s.topic, s.learning_teaching_type, s.lo, s.itu, s.student_tasks
        FROM {table_prefix} s
        WHERE s.syllabus_id = %s
        ORDER BY CAST(NULLIF(REGEXP_REPLACE(s.session_number, '[^0-9]', ''), '') AS UNSIGNED)
        """
        
        cursor.execute(query, (syllabus_id,))
        results = cursor.fetchall()
        
        headers = ["Phiên", "Chủ đề", "Phương pháp dạy-học", "LO", "ITU", "Nhiệm vụ sinh viên"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# 8. Tìm đánh giá của một đề cương
def search_assessments(conn, syllabus_id, language='vi'):
    try:
        cursor = conn.cursor()
        table_prefix = f"assessments_{language}"
        
        query = f"""
        SELECT a.category, a.type, a.part, a.weight, a.completion_criteria, 
               a.duration, a.clo
        FROM {table_prefix} a
        WHERE a.syllabus_id = %s
        ORDER BY a.part
        """
        
        cursor.execute(query, (syllabus_id,))
        results = cursor.fetchall()
        
        headers = ["Hạng mục", "Loại", "Phần", "Trọng số", "Tiêu chí hoàn thành", 
                  "Thời lượng", "CLO"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# 9. Tìm kiếm fulltext trong mô tả đề cương
def search_syllabus_by_description(conn, keyword, language='vi'):
    try:
        cursor = conn.cursor()
        table_prefix = f"syllabi_{language}"
        
        query = f"""
        SELECT syl.syllabus_id, syl.subject_code, syl.syllabus_name, 
               LEFT(syl.description, 100) as short_desc
        FROM {table_prefix} syl
        WHERE MATCH(syl.description) AGAINST(%s IN BOOLEAN MODE)
        """
        
        cursor.execute(query, (keyword + "*",))
        results = cursor.fetchall()
        
        headers = ["ID Đề cương", "Mã môn", "Tên đề cương", "Mô tả tóm tắt"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# 10. Liệt kê tất cả môn học và số đề cương
def list_subjects_with_syllabus_count(conn, language='vi'):
    try:
        cursor = conn.cursor()
        subj_table = f"subjects_{language}"
        syl_table = f"syllabi_{language}"
        
        query = f"""
        SELECT s.subject_code, s.subject_name, COUNT(syl.syllabus_id) as syllabus_count
        FROM {subj_table} s
        LEFT JOIN {syl_table} syl ON s.subject_code = syl.subject_code
        GROUP BY s.subject_code, s.subject_name
        ORDER BY syllabus_count DESC
        LIMIT 20
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        headers = ["Mã môn", "Tên môn học", "Số đề cương"]
        display_results(results, headers)
        
    except Error as e:
        print(f"Lỗi khi truy vấn: {e}")

# Menu chính
def main_menu():
    print("\n===== HỆ THỐNG TRUY VẤN DỮ LIỆU SYLLABI FPT =====")
    print("1. Tìm kiếm môn học theo mã môn")
    print("2. Tìm kiếm môn học theo từ khóa")
    print("3. Xem chi tiết đề cương")
    print("4. Xem tất cả đề cương của một môn học")
    print("5. Xem tài liệu học tập của đề cương")
    print("6. Xem kết quả học tập của đề cương")
    print("7. Xem phiên học và nội dung của đề cương")
    print("8. Xem đánh giá của đề cương")
    print("9. Tìm kiếm đề cương theo từ khóa trong mô tả")
    print("10. Liệt kê môn học và số đề cương")
    print("0. Thoát")
    
    choice = input("\nNhập lựa chọn của bạn: ")
    return choice

def select_language():
    print("\nChọn ngôn ngữ: ")
    print("1. Tiếng Việt (vi)")
    print("2. Tiếng Anh (en)")
    choice = input("Nhập lựa chọn của bạn (mặc định: vi): ")
    
    if choice == '2':
        return 'en'
    return 'vi'

# Hàm chính
def main():
    conn = create_connection()
    if not conn:
        print("Không thể kết nối đến cơ sở dữ liệu. Vui lòng kiểm tra lại thông tin kết nối.")
        return
    
    try:
        while True:
            choice = main_menu()
            language = select_language()
            
            if choice == '0':
                print("Đang thoát chương trình...")
                break
            
            elif choice == '1':
                subject_code = input("Nhập mã môn học (ví dụ: PRF192): ").strip().upper()
                search_subject_by_code(conn, subject_code, language)
            
            elif choice == '2':
                keyword = input("Nhập từ khóa tìm kiếm trong tên môn học: ").strip()
                search_subjects_by_keyword(conn, keyword, language)
            
            elif choice == '3':
                syllabus_id = input("Nhập ID đề cương (ví dụ: PRF192_SE): ").strip()
                search_syllabus_details(conn, syllabus_id, language)
            
            elif choice == '4':
                subject_code = input("Nhập mã môn học (ví dụ: PRF192): ").strip().upper()
                search_syllabi_by_subject(conn, subject_code, language)
            
            elif choice == '5':
                syllabus_id = input("Nhập ID đề cương (ví dụ: PRF192_SE): ").strip()
                search_materials(conn, syllabus_id, language)
            
            elif choice == '6':
                syllabus_id = input("Nhập ID đề cương (ví dụ: PRF192_SE): ").strip()
                search_learning_outcomes(conn, syllabus_id, language)
            
            elif choice == '7':
                syllabus_id = input("Nhập ID đề cương (ví dụ: PRF192_SE): ").strip()
                search_sessions(conn, syllabus_id, language)
            
            elif choice == '8':
                syllabus_id = input("Nhập ID đề cương (ví dụ: PRF192_SE): ").strip()
                search_assessments(conn, syllabus_id, language)
            
            elif choice == '9':
                keyword = input("Nhập từ khóa tìm kiếm trong mô tả đề cương: ").strip()
                search_syllabus_by_description(conn, keyword, language)
                
            elif choice == '10':
                list_subjects_with_syllabus_count(conn, language)
                
            else:
                print("Lựa chọn không hợp lệ, vui lòng thử lại!")
            
            input("\nNhấn Enter để tiếp tục...")
    
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
    
    finally:
        if conn:
            conn.close()
            print("Đã đóng kết nối đến cơ sở dữ liệu.")

if __name__ == "__main__":
    main()