import json
import os

def filter_main_notes_only(input_file_path, output_file_path):
    """
    Chỉ lọc ra ghi chú chính (main_note) của từng môn học từ file JSON.

    Args:
        input_file_path (str): Đường dẫn đến file JSON đầu vào.
        output_file_path (str): Đường dẫn để lưu file JSON kết quả.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{input_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: File '{input_file_path}' không phải là định dạng JSON hợp lệ.")
        return

    # Dictionary để lưu trữ kết quả
    main_notes_result = {}

    # Lặp qua từng syllabus (môn học) trong danh sách
    for syllabus in data.get('syllabuses', []):
        metadata = syllabus.get('metadata', {})
        
        # Lấy các thông tin cần thiết
        course_id = metadata.get('course_id')
        course_name = metadata.get('course_name_from_curriculum')
        main_note = metadata.get('note', '').strip() # .strip() để loại bỏ khoảng trắng thừa

        # Chỉ lưu lại nếu môn học có 'note' và 'note' đó có nội dung
        if course_id and main_note:
            key = f"{course_id} - {course_name}"
            main_notes_result[key] = main_note
            
    # Ghi kết quả ra file output
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(main_notes_result, f, indent=2, ensure_ascii=False)

    print(f"Đã lọc xong! Kết quả được lưu tại: {output_file_path}")
    
    if not main_notes_result:
        print("Lưu ý: Không tìm thấy môn học nào có 'main_note' trong dữ liệu mẫu của bạn.")


if __name__ == "__main__":
    # Xác định đường dẫn file
    INPUT_FILE = os.path.join('data', 'combined_data.json')
    OUTPUT_FILE = os.path.join('output', 'main_notes_only.json')
    
    # Chạy hàm lọc
    filter_main_notes_only(INPUT_FILE, OUTPUT_FILE)