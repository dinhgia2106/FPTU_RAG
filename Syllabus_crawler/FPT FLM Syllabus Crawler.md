# FPT FLM Syllabus Crawler

Công cụ tự động crawl dữ liệu syllabus từ hệ thống FLM của FPT University, được thiết kế để người dùng tự chạy trên máy cá nhân.

## Tính năng nổi bật

- **Cấu hình qua file .env**: Đọc thông tin đăng nhập (email, password), danh sách backup codes, và danh sách mã môn học từ file .env để tăng tính bảo mật và tiện lợi.
- **Đăng nhập thủ công có hướng dẫn**: Hiển thị rõ các bước để người dùng tự đăng nhập, giúp tránh các vấn đề về xác thực và bảo mật.
- **Hiển thị quá trình**: Chạy trình duyệt ở chế độ hiển thị (không ẩn) để người dùng có thể theo dõi trực tiếp quá trình crawl.
- **Cộng dồn dữ liệu**: Khi crawl, dữ liệu mới sẽ được thêm vào file JSON kết quả (`fpt_syllabus_data_appended.json`). Nếu một môn học đã được crawl trước đó, script sẽ bỏ qua việc crawl lại môn đó.
- **Trích xuất toàn diện**: Lấy đầy đủ thông tin từ trang chi tiết syllabus bao gồm thông tin chung, tài liệu, CLOs, sessions và assessments.

## Yêu cầu hệ thống

- Python 3.7 trở lên
- Thư viện Playwright
- Thư viện `python-dotenv` (để đọc file .env)

## Cài đặt

1. **Cài đặt Python**: Nếu chưa có, tải và cài đặt Python từ [python.org](https://www.python.org/downloads/).
2. **Cài đặt thư viện cần thiết**:
   Mở terminal (Command Prompt, PowerShell, hoặc Terminal trên Linux/macOS) và chạy các lệnh sau:
   ```bash
   pip install playwright python-dotenv
   python -m playwright install
   ```
   Lệnh `python -m playwright install` sẽ tải về các trình duyệt cần thiết cho Playwright.

## Chuẩn bị file .env

1. Trong cùng thư mục với file script syllabus_crawler.py, tạo một file mới có tên là .env (lưu ý dấu chấm ở đầu).
2. Mở file .env bằng một trình soạn thảo văn bản (như Notepad, VS Code, Sublime Text, v.v.) và thêm nội dung sau, **thay thế các giá trị mẫu bằng thông tin thực tế của bạn**:

   ```env
   # Mã môn học cần crawl, cách nhau bởi dấu phẩy. Ví dụ: ADY201m,DBI202,SEG301
   SUBJECT_CODES="SEG301,ADY201m,SWP391"
   # Tên file output (tùy chọn)
   OUTPUT_FILE="fpt_syllabus_data_appended.json"
   # Chạy ẩn trình duyệt (true/false, tùy chọn)
   HEADLESS="false"
   ```

   **Lưu ý quan trọng về file .env**:

   - Không sử dụng dấu cách (space) xung quanh dấu bằng (`=`).
   - Nếu mật khẩu hoặc các giá trị khác của bạn có chứa ký tự đặc biệt, hãy đặt chúng trong dấu ngoặc kép `""` như ví dụ.

## Hướng dẫn sử dụng

1. Đảm bảo bạn đã hoàn tất các bước Cài đặt và Chuẩn bị file .env.
2. Tải file script syllabus_crawler.py về máy của bạn và đặt nó trong cùng thư mục với file .env.
3. Mở terminal/command prompt, di chuyển đến thư mục chứa file script và file .env.
4. Chạy script bằng lệnh:
   ```bash
   python syllabus_crawler.py
   ```
5. Một cửa sổ trình duyệt sẽ mở ra và hiển thị trang đăng nhập của FLM.
6. **Đăng nhập thủ công**: Script sẽ hiển thị hướng dẫn để bạn tự đăng nhập vào hệ thống FLM và sau đó nhấn Enter để tiếp tục.
7. Script sẽ tự động thực hiện các bước tìm kiếm và crawl dữ liệu cho các môn học bạn đã cấu hình.
8. Sau khi hoàn tất, dữ liệu sẽ được lưu (hoặc cộng dồn) vào file fpt_syllabus_data_appended.json trong cùng thư mục.

## Cấu trúc dữ liệu JSON đầu ra

Dữ liệu được lưu với cấu trúc JSON như sau:

```json
{
  "SUBJECT_CODE": {
    "page_title": "FPT University Learning Materials",
    "page_url": "https://flm.fpt.edu.vn/gui/role/student/SyllabusDetails?sylID=XXXXX",
    "subject_code": "SUBJECT_CODE",
    "syllabus_id": "XXXXX",
    "general_details": {
      "Syllabus ID": "XXXXX",
      "Syllabus Name": "Tên Tiếng Việt_Tên Tiếng Anh",
      "Syllabus English": "Tên Tiếng Anh",
      "Subject Code": "SUBJECT_CODE",
      "NoCredit": "3",
      "Degree Level": "Bachelor",
      "Time Allocation": "45h contact hours + 1h final exam + 104h self-study",
      "Pre-Requisite": "Các môn tiên quyết",
      "Description": "Mô tả chi tiết về môn học...",
      "StudentTasks": "Nhiệm vụ của sinh viên...",
      "Tools": "Công cụ sử dụng trong môn học...",
      "Scoring Scale": "10",
      "DecisionNo MM/dd/yyyy": "XXX/QĐ-ĐHFPT dated MM/DD/YYYY",
      "IsApproved": "True",
      "Note": "Các ghi chú khác...",
      "MinAvgMarkToPass": "5",
      "IsActive": "True",
      "ApprovedDate": "MM/DD/YYYY"
    },
    "materials_table": [
      {
        "Note": "Thông tin về tài liệu 1"
      },
      {
        "Note": "Thông tin về tài liệu 2"
      }
    ],
    "clos": [
      {
        "LO Details": "Chi tiết về chuẩn đầu ra 1"
      },
      {
        "LO Details": "Chi tiết về chuẩn đầu ra 2"
      }
    ],
    "clo_plo_mapping_link": "https://flm.fpt.edu.vn/CLOMapping/View?syllabusID=XXXXX",
    "sessions": [
      {
        "URLs": "Link tài liệu của buổi học 1"
      },
      {
        "URLs": "Link tài liệu của buổi học 2"
      }
    ],
    "assessments": [
      {
        "Note": "Thông tin về phương pháp đánh giá 1"
      },
      {
        "Note": "Thông tin về phương pháp đánh giá 2"
      }
    ],
    "extraction_errors": [],
    "extraction_time": "YYYY-MM-DDThh:mm:ss.sssZ",
    "materials_info": "X material(s)"
  }
}
```

Trong đó:

- `SUBJECT_CODE`: Mã môn học (ví dụ: PFP191, DBI202)
- Mỗi môn học chứa đầy đủ thông tin từ syllabus, được tổ chức thành các phần khác nhau
- `extraction_errors`: Ghi nhận các lỗi xảy ra trong quá trình trích xuất (nếu có)

## Lưu ý quan trọng

- **Bảo mật**: File .env chứa thông tin nhạy cảm. Hãy đảm bảo bạn bảo vệ file này cẩn thận và không chia sẻ nó.
- **Cấu trúc trang web**: Trang web FLM có thể thay đổi cấu trúc HTML theo thời gian. Nếu điều này xảy ra, script có thể không hoạt động chính xác và cần được cập nhật (chủ yếu là các selectors trong code).
- **Lỗi Timeout**: Script đã được thiết lập thời gian chờ cho các thao tác. Tuy nhiên, nếu mạng của bạn quá chậm hoặc trang web phản hồi chậm bất thường, lỗi timeout vẫn có thể xảy ra. Trong trường hợp này, bạn có thể thử chạy lại script.

## Xử lý sự cố thường gặp

- **Lỗi `ModuleNotFoundError: No module named 'dotenv'`**: Bạn chưa cài đặt thư viện `python-dotenv`. Chạy `pip install python-dotenv`.
- **Lỗi liên quan đến Playwright hoặc trình duyệt**: Đảm bảo bạn đã chạy `python -m playwright install`.
- **Script không tìm thấy file .env hoặc đọc sai thông tin**: Kiểm tra lại tên file là .env (có dấu chấm ở đầu) và nội dung file đúng định dạng như hướng dẫn.
- **Lỗi khi chọn "Education Level"**: Script đã cố gắng sử dụng nhiều selector khác nhau. Nếu vẫn lỗi, cấu trúc trang có thể đã thay đổi đáng kể ở bước này.
- **Đăng nhập thất bại dù thông tin đúng**: Google có thể áp dụng các biện pháp bảo mật bổ sung nếu phát hiện đăng nhập tự động từ một môi trường lạ. Việc chạy trình duyệt ở chế độ hiển thị và đăng nhập thủ công giúp giảm thiểu vấn đề này.
- **Không crawl được môn học cụ thể**: Kiểm tra lại mã môn trong file .env. Đảm bảo mã môn chính xác và tồn tại trên hệ thống FLM.

## Tùy chỉnh (Nâng cao)

- **Tên file output**: Bạn có thể thay đổi giá trị `OUTPUT_FILE` trong file .env nếu muốn lưu kết quả vào file khác.
- **Chế độ Headless**: Nếu muốn script chạy ẩn (không hiển thị cửa sổ trình duyệt), bạn có thể đặt `HEADLESS="true"` trong file .env. Tuy nhiên, điều này có thể gây khó khăn khi đăng nhập thủ công.
- **Thời gian chờ (Timeout)**: Các giá trị timeout có thể được điều chỉnh trong script nếu cần thiết, nhưng nên thận trọng.

## Miễn trừ trách nhiệm

Script này được cung cấp cho mục đích học tập và sử dụng cá nhân. Người dùng chịu hoàn toàn trách nhiệm về việc sử dụng script này và phải đảm bảo tuân thủ các điều khoản dịch vụ của FPT FLM và Google. Không sử dụng script cho các mục đích gây hại hoặc vi phạm quy định.
