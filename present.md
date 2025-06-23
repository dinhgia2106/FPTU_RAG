Chào mừng mọi người, hôm nay tôi sẽ trình bày về luồng hoạt động của hệ thống Hỏi-Đáp thông minh mà chúng ta đang xây dựng cho sinh viên Đại học FPT. Để dễ hình dung nhất, tôi sẽ sửdụng một mẫu dữ liệu thực tế và dẫn dắt mọi người đi qua từng bước xử lý của hệ thống, từ "nguyên liệu" thô cho đến khi có một câu trả lời hoàn chỉnh.

### Bước 1: Nguyên liệu đầu vào - Một Mẫu Dữ liệu Cụ thể

Hãy bắt đầu với "nguyên liệu". Tưởng tượng chúng ta có một khối thông tin về một môn học duy nhất, được lấy từ file `combined_data.json`. Trông nó sẽ như thế này:

```json
{
  "metadata": {
    "course_id": "CSI106",
    "course_name_from_curriculum": "Introduction to Computer Science",
    "credits": "3",
    "prerequisites": "",
    "description": "This course provides an overview of computer fundamentals..."
  },
  "learning_outcomes": [
    { "id": "CLO1", "details": "Understand the subsystems of a computer..." },
    { "id": "CLO2", "details": "Know how to convert a number from one base to another..." }
  ],
  "assessments": [
    { "component": "Final Exam", "weight": "40%" },
    { "component": "Assignments", "weight": "60%" }
  ]
}
```

Đây là một mẩu nhỏ, nhưng nó chứa đựng thông tin tổng quan (`metadata`), chuẩn đầu ra (`learning_outcomes`), và cách đánh giá (`assessments`) của môn `CSI106`.

### Bước 2: Xây dựng "Bộ não" - Chúng ta "Tiêu hóa" Mẫu Dữ liệu Này Như Thế Nào?

Khi hệ thống khởi động, nó sẽ xử lý khối dữ liệu trên.

1.  **"Xé nhỏ" kiến thức**: Thay vì giữ nguyên khối JSON lớn, hệ thống của tôi sẽ "bóc tách" nó thành các "mẩu giấy" văn bản (chunks) riêng biệt, mỗi mẩu được gắn với thông tin nhận dạng. Từ mẫu dữ liệu trên, tôi sẽ tạo ra:

    *   **Mẩu giấy 1 (Tổng quan)**:
        *   Nội dung: `Môn học CSI106 - Introduction to Computer Science, 3 tín chỉ. Mô tả: This course provides an overview of computer fundamentals...`
        *   Metadata: `{ subject_code: 'CSI106', document_type: 'overview' }`

    *   **Mẩu giấy 2 (Chuẩn đầu ra)**:
        *   Nội dung: `Chuẩn đầu ra: CLO1 - Understand the subsystems of a computer. CLO2 - Know how to convert a number from one base to another.`
        *   Metadata: `{ subject_code: 'CSI106', document_type: 'learning_outcomes' }`

    *   **Mẩu giấy 3 (Đánh giá)**:
        *   Nội dung: `Đánh giá: Final Exam chiếm 40%. Assignments chiếm 60%.`
        *   Metadata: `{ subject_code: 'CSI106', document_type: 'assessment' }`

    Như vậy, từ một khối dữ liệu phức tạp, chúng ta đã có 3 "mẩu giấy" thông tin rõ ràng, mạch lạc.

2.  **Tạo "Dấu vân tay" cho từng mẩu giấy**: Tiếp theo, tôi sẽ đưa nội dung của từng mẩu giấy này vào mô hình AI để tạo "dấu vân tay số" (vector embedding).
    *   `"Môn học CSI106..."` -> `[0.12, -0.45, 0.89, ...]`
    *   `"Chuẩn đầu ra: CLO1..."` -> `[0.34, 0.56, -0.11, ...]`
    *   `"Đánh giá: Final Exam..."` -> `[-0.78, 0.02, 0.67, ...]`

3.  **Lập "Danh bạ thông minh"**: Cuối cùng, tôi lưu các "dấu vân tay số" này cùng với thông tin nhận dạng của chúng (ví dụ: `CSI106-overview`, `CSI106-learning_outcomes`) vào thư viện tìm kiếm FAISS. "Bộ não" của chúng ta giờ đã được nạp và sắp xếp kiến thức về môn CSI106.

### Bước 3: Khi Người dùng Đặt câu hỏi - Minh họa Luồng Truy vấn

Bây giờ, hãy xem điều gì xảy ra khi một bạn sinh viên hỏi: **"Môn CSI106 thi cuối kỳ bao nhiêu phần trăm?"**

1.  **Phân tích câu hỏi**: Hệ thống nhận ra câu hỏi này đang hỏi về "tỷ lệ phần trăm", "thi cuối kỳ" của môn "CSI106".

2.  **Tạo "Dấu vân tay" cho câu hỏi**: Câu hỏi của người dùng cũng được chuyển thành một "dấu vân tay số":
    *   `"Môn CSI106 thi cuối kỳ bao nhiêu phần trăm?"` -> `[-0.75, 0.05, 0.69, ...]`

3.  **Tìm kiếm trong "Danh bạ"**: Hệ thống mang dấu vân tay này `[-0.75, 0.05, 0.69, ...]` đi so sánh trong "danh bạ" FAISS. Nó sẽ thấy rằng dấu vân tay này **gần nhất** với dấu vân tay của "Mẩu giấy 3 (Đánh giá)": `[-0.78, 0.02, 0.67, ...]`.

    Kết quả là, hệ thống đã truy xuất thành công mẩu thông tin liên quan nhất:
    *   **Nội dung truy xuất được (Context)**: `Đánh giá: Final Exam chiếm 40%. Assignments chiếm 60%.`

4.  **Tổng hợp và Tạo câu trả lời**: Tôi không chỉ đơn thuần vứt mẩu thông tin này cho người dùng. Tôi đưa nó cho trợ lý ngôn ngữ Gemini và ra lệnh: *"Dựa vào thông tin 'Đánh giá: Final Exam chiếm 40%. Assignments chiếm 60%', hãy trả lời câu hỏi 'Môn CSI106 thi cuối kỳ bao nhiêu phần trăm?' một cách tự nhiên."*

    Và trợ lý sẽ tạo ra câu trả lời cuối cùng, thân thiện và chính xác:
    > "Chào bạn, môn CSI106 có hình thức thi cuối kỳ chiếm trọng số **40%** tổng điểm."

Đây chính là hành trình của một mẩu dữ liệu, từ một khối JSON thô, được xử lý, lập chỉ mục, và cuối cùng được sử dụng để trả lời một cách chính xác một câu hỏi rất cụ thể của sinh viên. Bằng cách này, hệ thống đảm bảo câu trả lời luôn dựa trên dữ liệu thực tế của nhà trường.

Cảm ơn mọi người đã lắng nghe