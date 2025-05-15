# Hệ thống RAG Truy vấn Syllabus FPT

## Giới thiệu

Đây là một hệ thống Truy xuất Tăng cường Sinh Tạo (Retrieval Augmented Generation - RAG) được xây dựng để cho phép người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên và nhận câu trả lời dựa trên nội dung chi tiết của các syllabus môn học tại Đại học FPT. Hệ thống sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên tiên tiến, bao gồm tạo embedding cho văn bản, xây dựng cơ sở dữ liệu vector, và tích hợp với Mô hình Ngôn ngữ Lớn (LLM) như Google Gemini để sinh câu trả lời.

**Các thành phần chính của hệ thống:**

1.  **Thu thập và Chuẩn bị Dữ liệu**: Syllabus được thu thập (ví dụ, từ file JSON đã crawl) và chuẩn bị.
2.  **Chunking**: Nội dung syllabus được chia thành các "mảnh" (chunks) nhỏ hơn, có ý nghĩa.
3.  **Embedding**: Mỗi chunk được chuyển đổi thành một vector embedding (dãy số) biểu diễn ý nghĩa ngữ nghĩa của nó.
4.  **Cơ sở dữ liệu Vector (FAISS)**: Các vector embedding được lưu trữ và đánh chỉ mục trong FAISS để tìm kiếm tương đồng nhanh chóng.
5.  **Pipeline RAG**: Khi người dùng đặt câu hỏi:
    *   Câu hỏi được chuyển thành vector embedding.
    *   FAISS tìm kiếm các chunk syllabus có embedding gần nhất (liên quan nhất).
    *   Nội dung các chunk này được cung cấp làm ngữ cảnh cho LLM (Google Gemini).
    *   LLM sinh ra câu trả lời dựa trên câu hỏi và ngữ cảnh được cung cấp.
6.  **Giao diện Người dùng (CLI)**: Một ứng dụng dòng lệnh đơn giản cho phép người dùng tương tác với hệ thống.

## Điều kiện Tiên quyết

*   Python 3.9 trở lên.
*   `pip` (Python package installer).
*   Một API Key hợp lệ từ Google AI Studio cho các mô hình Gemini.

## Cài đặt

1.  **Tải mã nguồn**: Sao chép tất cả các file Python (`syllabus_chunker.py`, `generate_embeddings.py`, `create_faiss_index.py`, `rag_cli_app.py`) và các file dữ liệu mẫu (nếu có) vào một thư mục trên máy của bạn.

2.  **Cài đặt các thư viện Python cần thiết**: Mở terminal hoặc command prompt, điều hướng đến thư mục chứa mã nguồn và chạy lệnh sau:
    ```bash
    pip install sentence-transformers faiss-cpu google-generativeai numpy
    ```

## Chuẩn bị Dữ liệu Syllabus

1.  **Dữ liệu đầu vào**: Hệ thống yêu cầu dữ liệu syllabus dưới dạng file JSON. File này nên chứa một danh sách các đối tượng, mỗi đối tượng đại diện cho một môn học và bao gồm các thông tin chi tiết đã được crawl (ví dụ: thông tin chung, CLO, lịch trình buổi học, đánh giá, tài liệu).
    *   Ví dụ: Bạn cần có một file như `fpt_syllabus_data_appended_vi.json` (đã được sử dụng trong quá trình phát triển).
    *   Kịch bản `syllabus_chunker.py` hiện đang được cấu hình để đọc dữ liệu từ một môn học cụ thể (PFP191) được nhúng sẵn trong code hoặc từ file `/home/ubuntu/upload/fpt_syllabus_data_appended_vi.json`. Bạn cần điều chỉnh phần đọc dữ liệu trong `syllabus_chunker.py` nếu muốn xử lý file syllabus tổng hợp của bạn.

2.  **API Key cho Google Gemini**:
    *   Truy cập [Google AI Studio](https://aistudio.google.com/) và tạo một API Key mới.
    *   **QUAN TRỌNG**: Kịch bản `rag_cli_app.py` (và `rag_pipeline_gemini_list_models.py` trước đó) hiện đang hardcode API key. Đây là cách làm không an toàn cho môi trường sản phẩm.
        ```python
        GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # Thay YOUR_GEMINI_API_KEY bằng key của bạn
        ```
    *   Để sử dụng an toàn hơn, bạn nên lưu API key vào biến môi trường và đọc từ đó trong code. Ví dụ:
        ```python
        # Trong terminal
        # export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        # Trong Python
        # import os
        # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        ```

## Hướng dẫn Chạy Hệ thống RAG (Cho một môn học)

Các bước sau đây mô tả quy trình để xử lý dữ liệu cho một môn học (ví dụ PFP191, như trong các kịch bản hiện tại) và sau đó truy vấn nó.

**Lưu ý**: Các kịch bản hiện tại (`syllabus_chunker.py`, `generate_embeddings.py`, `create_faiss_index.py`, `rag_cli_app.py`) đang được cấu hình để xử lý dữ liệu của môn "PFP191" từ file `/home/ubuntu/PFP191_embeddings.json` hoặc các file có tên tương tự. Nếu bạn muốn xử lý một môn học khác hoặc nhiều môn, bạn cần điều chỉnh logic đọc/ghi file và xử lý dữ liệu trong các kịch bản này.

1.  **Bước 1: Chunking Dữ liệu Syllabus (`syllabus_chunker.py`)**
    *   Mở file `syllabus_chunker.py`.
    *   Đảm bảo rằng biến `input_syllabus_file` trỏ đúng đến file JSON chứa dữ liệu syllabus của bạn (ví dụ: `/home/ubuntu/upload/fpt_syllabus_data_appended_vi.json`).
    *   Đảm bảo rằng `subject_code_to_process` được đặt thành mã môn bạn muốn xử lý (ví dụ: "PFP191").
    *   Chạy kịch bản:
        ```bash
        python syllabus_chunker.py
        ```
    *   Kết quả: Sẽ tạo ra một file JSON chứa các chunk đã được xử lý, ví dụ: `PFP191_chunks.json` (tên file có thể thay đổi tùy theo cấu hình trong script).

2.  **Bước 2: Tạo Vector Embedding (`generate_embeddings.py`)**
    *   Mở file `generate_embeddings.py`.
    *   Đảm bảo `chunks_file_path` trỏ đến file chunks đã tạo ở Bước 1 (ví dụ: `PFP191_chunks.json`).
    *   Đảm bảo `output_embeddings_file_path` là nơi bạn muốn lưu file embeddings (ví dụ: `PFP191_embeddings.json`).
    *   Chạy kịch bản:
        ```bash
        python generate_embeddings.py
        ```
    *   Kết quả: Sẽ tạo ra một file JSON mới chứa các chunk cùng với vector embedding của chúng.

3.  **Bước 3: Xây dựng FAISS Index (`create_faiss_index.py`)**
    *   Mở file `create_faiss_index.py`.
    *   Đảm bảo `embeddings_file_path` trỏ đến file embeddings đã tạo ở Bước 2 (ví dụ: `PFP191_embeddings.json`).
    *   Đảm bảo `faiss_index_path` là nơi bạn muốn lưu FAISS index (ví dụ: `PFP191_faiss.index`).
    *   Chạy kịch bản:
        ```bash
        python create_faiss_index.py
        ```
    *   Kết quả: Sẽ tạo ra một file index FAISS.

4.  **Bước 4: Chạy Ứng dụng Truy vấn RAG CLI (`rag_cli_app.py`)**
    *   Mở file `rag_cli_app.py`.
    *   **Quan trọng**: Đảm bảo bạn đã cập nhật `GEMINI_API_KEY` bằng API key hợp lệ của bạn.
    *   Đảm bảo các đường dẫn `faiss_index_path` và `chunks_json_path` trỏ đúng đến các file đã tạo ở các bước trước.
    *   Chạy ứng dụng:
        ```bash
        python rag_cli_app.py
        ```
    *   Ứng dụng sẽ khởi tạo và bạn có thể bắt đầu nhập câu hỏi về syllabus PFP191.
    *   Nhập `quit`, `exit`, hoặc `bye` để thoát ứng dụng.

**Ví dụ câu hỏi cho PFP191:**
*   `Môn PFP191 có bao nhiêu tín chỉ?`
*   `Mô tả về môn học PFP191 là gì?`
*   `CLO1 của môn PFP191 là gì?`
*   `Buổi học số 1 của PFP191 nói về chủ đề gì?`
*   `Hình thức đánh giá Progress test 1 của PFP191 chiếm bao nhiêu phần trăm?`

## Xử lý Sự cố Cơ bản

*   **Lỗi `FileNotFoundError`**: Kiểm tra kỹ lại đường dẫn đến các file dữ liệu (`.json`, `.index`) trong các kịch bản Python.
*   **Lỗi `JSONDecodeError`**: File JSON đầu vào có thể bị lỗi cú pháp. Kiểm tra lại file JSON.
*   **Lỗi liên quan đến API Key Gemini**: Đảm bảo API key là chính xác, còn hiệu lực và có quyền truy cập vào model Gemini đang được sử dụng (`gemini-1.5-flash-latest`).
*   **Lỗi tải mô hình (Embedding/LLM)**: Có thể do vấn đề kết nối mạng hoặc không đủ tài nguyên (RAM/CPU) nếu chạy các mô hình lớn local (không áp dụng cho Gemini API).
*   **FAISS Index không tải được**: Đảm bảo file index không bị hỏng và được tạo bởi cùng phiên bản FAISS.

## Đề xuất Mở rộng và Phát triển Tương lai

1.  **Hỗ trợ Nhiều Môn học**: Mở rộng hệ thống để có thể xử lý và truy vấn dữ liệu từ nhiều syllabus môn học khác nhau. Điều này đòi hỏi:
    *   Cơ chế quản lý dữ liệu (chunks, embeddings, FAISS indices) cho từng môn.
    *   Khả năng chọn môn học khi truy vấn.
2.  **Giao diện Người dùng Web/API**: Xây dựng một giao diện web thân thiện hơn thay vì CLI, hoặc cung cấp một API để các ứng dụng khác có thể tích hợp.
3.  **Cải thiện Chiến lược Chunking**: Thử nghiệm các kích thước chunk khác nhau, các phương pháp chia chunk dựa trên ngữ nghĩa hoặc cấu trúc tài liệu để tối ưu hóa chất lượng truy xuất.
4.  **Thử nghiệm Mô hình Embedding/LLM Khác**: Đánh giá các mô hình embedding hoặc LLM khác (cả mã nguồn mở và API) để tìm ra sự kết hợp tối ưu nhất về chi phí và hiệu năng cho tiếng Việt và dữ liệu syllabus.
5.  **Tinh chỉnh (Fine-tuning)**: Nếu có đủ dữ liệu và tài nguyên, có thể xem xét việc tinh chỉnh mô hình embedding hoặc LLM trên dữ liệu chuyên ngành syllabus để cải thiện độ chính xác.
6.  **Quản lý và Cập nhật Dữ liệu**: Xây dựng quy trình tự động hoặc bán tự động để cập nhật dữ liệu syllabus khi có thay đổi.
7.  **Nâng cao Bảo mật API Key**: Triển khai các giải pháp quản lý secret an toàn hơn cho API key thay vì hardcode.
8.  **Đánh giá và Giám sát Liên tục**: Xây dựng bộ dữ liệu đánh giá (benchmark) và cơ chế giám sát chất lượng câu trả lời của hệ thống theo thời gian.
9.  **Xử lý Câu hỏi Phức tạp hơn**: Cải thiện khả năng hiểu và trả lời các câu hỏi phức tạp, câu hỏi yêu cầu suy luận đa bước hoặc tổng hợp thông tin từ nhiều nguồn.

Chúc bạn thành công với hệ thống RAG truy vấn syllabus!

