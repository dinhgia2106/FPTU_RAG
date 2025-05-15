import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def load_faiss_index(index_path):
    """Tải FAISS index từ file."""
    try:
        index = faiss.read_index(index_path)
        print(f"Đã tải FAISS index từ: {index_path} với {index.ntotal} vectors.")
        return index
    except Exception as e:
        print(f"Lỗi khi tải FAISS index: {e}")
        return None

def load_chunks_data(chunks_file_path):
    """Tải dữ liệu chunks gốc (bao gồm text và metadata)."""
    try:
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        print(f"Đã tải dữ liệu chunks từ: {chunks_file_path}")
        return chunks_data
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file chunks tại: {chunks_file_path}")
    except json.JSONDecodeError as e:
        print(f"Lỗi: File chunks không phải là JSON hợp lệ: {e}")
    return None

def get_answer_from_rag_gemini(query_text, faiss_index, all_chunks_data, embedding_model, gemini_model, top_k=3):
    """Thực hiện pipeline RAG với Gemini để lấy câu trả lời."""
    if not query_text:
        return "Vui lòng cung cấp câu hỏi."

    # 1. Xử lý Câu hỏi (Query Processing)
    print(f"\nĐang tạo embedding cho câu hỏi... ")
    try:
        query_embedding = embedding_model.encode([query_text])
    except Exception as e:
        print(f"Lỗi khi tạo embedding cho câu hỏi: {e}")
        return "Lỗi khi xử lý câu hỏi."

    # 2. Truy xuất Thông tin (Information Retrieval)
    print(f"Đang tìm kiếm trong FAISS index...")
    try:
        distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    except Exception as e:
        print(f"Lỗi khi tìm kiếm trong FAISS: {e}")
        return "Lỗi khi truy xuất thông tin."

    # 3. Chuẩn bị Ngữ cảnh (Context Preparation)
    retrieved_chunks_content = []
    print(f"Đã truy xuất {len(indices[0])} chunks liên quan.")
    for i in indices[0]:
        if 0 <= i < len(all_chunks_data):
            retrieved_chunks_content.append(all_chunks_data[i].get("content", ""))
        else:
            print(f"Cảnh báo: Index {i} nằm ngoài phạm vi của all_chunks_data.")
    
    if not retrieved_chunks_content:
        return "Không tìm thấy thông tin liên quan trong syllabus để trả lời câu hỏi này."

    context_for_llm = "\n\n".join(retrieved_chunks_content)
    
    prompt = f"Dựa vào các thông tin sau đây từ syllabus của một môn học:\n\n-- BẮT ĐẦU NGỮ CẢNH SYLLABUS --\n{context_for_llm}\n-- KẾT THÚC NGỮ CẢNH SYLLABUS --\n\nHãy trả lời câu hỏi sau một cách ngắn gọn và chính xác, chỉ dựa vào thông tin được cung cấp trong ngữ cảnh syllabus ở trên. Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không tìm thấy thông tin đó trong tài liệu được cung cấp.\nCâu hỏi: {query_text}"
    
    # print(f"\n--- Prompt cho Gemini ---\n{prompt}\n-----------------------") # Bỏ comment nếu muốn xem prompt

    # 4. Sinh Câu trả lời (Answer Generation) với Gemini
    print("Đang sinh câu trả lời bằng Gemini...")
    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        print(f"Lỗi khi Gemini sinh câu trả lời: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Gemini API Response Error: {e.response}")
        if hasattr(e, 'message'):
             print(f"Error message: {e.message}")
        return "Lỗi khi tạo câu trả lời bằng Gemini."
    
    return answer

def main_cli():
    """Hàm chính cho ứng dụng RAG CLI."""
    faiss_index_path = "Faiss/PFP191_faiss.index"
    chunks_json_path = "Embedded/PFP191_embeddings.json"
    
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    gemini_model_name = "gemini-1.5-flash-latest"

    # Cấu hình Gemini API Key
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Lỗi khi cấu hình Gemini API: {e}. Kiểm tra API Key.")
        return

    # Tải các thành phần
    print("Đang khởi tạo hệ thống RAG...")
    print("Đang tải FAISS index...")
    faiss_index = load_faiss_index(faiss_index_path)
    print("Đang tải dữ liệu chunks...")
    all_chunks_data = load_chunks_data(chunks_json_path)
    
    if not faiss_index or not all_chunks_data:
        print("Không thể tải index hoặc dữ liệu chunks. Thoát chương trình.")
        return

    print(f"Đang tải mô hình embedding: {embedding_model_name}...")
    try:
        embedding_model = SentenceTransformer(embedding_model_name)
        print("Tải mô hình embedding thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình embedding: {e}. Thoát.")
        return

    print(f"Đang khởi tạo mô hình Gemini: {gemini_model_name}...")
    try:
        gemini_llm = genai.GenerativeModel(gemini_model_name)
        print("Khởi tạo mô hình Gemini thành công.")
    except Exception as e:
        print(f"Lỗi khi khởi tạo mô hình Gemini: {e}. Thoát.")
        return
    
    print("\n--- Chào mừng bạn đến với RAG Syllabus Query App! ---")
    print("Nhập câu hỏi của bạn về syllabus PFP191.")
    print("Nhập 'quit', 'exit' hoặc 'bye' để thoát.")

    while True:
        try:
            user_query = input("\nCâu hỏi của bạn: ")
            if user_query.lower() in ["quit", "exit", "bye"]:
                print("Cảm ơn bạn đã sử dụng! Tạm biệt.")
                break
            if not user_query.strip():
                continue
            
            answer = get_answer_from_rag_gemini(user_query, faiss_index, all_chunks_data, embedding_model, gemini_llm)
            print(f"\nTrả lời: {answer}")
        
        except KeyboardInterrupt:
            print("\nĐã nhận tín hiệu dừng. Tạm biệt!")
            break
        except Exception as e:
            print(f"Đã có lỗi xảy ra trong vòng lặp chính: {e}")
            # Có thể thêm logic để thử lại hoặc thoát tùy theo lỗi

if __name__ == "__main__":
    main_cli()

