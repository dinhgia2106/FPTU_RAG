import json
# import os # Uncomment if you need environment variables for API keys for other models
from sentence_transformers import SentenceTransformer

# --- Model Loading ---
# THAM SỐ THỰC: Bạn có thể thay đổi model_name nếu muốn dùng model Sentence Transformer khác.
# Đảm bảo model đó tương thích với thư viện sentence_transformers.
model_name = 'all-MiniLM-L6-v2'
print(f"Embedder: Đang tải model Sentence Transformer: {model_name}...")
model = SentenceTransformer(model_name)
print("Embedder: Model đã được tải.")


def embed_content_with_sbert(text_to_embed):
    """
    Nhúng văn bản cung cấp bằng model Sentence Transformer đã được tải.
    """
    # print(f"DEBUG: Nhúng: '{text_to_embed[:70].replace('\n', ' ')}...'") # Uncomment for debugging
    try:
        embedding = model.encode(text_to_embed).tolist()
        return embedding
    except Exception as e:
        print(f"Embedder: Lỗi khi nhúng bằng Sentence Transformers: {e}")
        # THAM SỐ THỰC: Kích thước vector của model embedding_model_name (all-MiniLM-L6-v2 là 384)
        # Cập nhật số này nếu bạn thay đổi model_name sang một model có kích thước vector khác.
        dimension = model.get_sentence_embedding_dimension() if hasattr(
            model, 'get_sentence_embedding_dimension') else 384
        return [0.0] * dimension


def main():
    chunks_file_path = "all_syllabus_and_student_chunks.json"
    embedded_chunks_file_path = "embedded_all_chunks_with_students.json"
    all_embedded_chunks = []

    print(f"Embedder: Đang đọc chunks từ: {chunks_file_path}")
    try:
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
        print(f"Embedder: Đã tải thành công {len(all_chunks)} chunks.")
    except FileNotFoundError:
        print(
            f"Embedder: Lỗi - Không tìm thấy file chunks '{chunks_file_path}'.")
        print("Embedder: Hãy đảm bảo file này được tạo bởi script 'chunker.py' (đã cập nhật) trước khi chạy 'embedder.py'.")
        return
    except json.JSONDecodeError:
        print(
            f"Embedder: Lỗi - File chunks '{chunks_file_path}' không phải là JSON hợp lệ.")
        return
    except Exception as e:
        print(f"Embedder: Lỗi không xác định khi đọc file chunks: {e}")
        return

    if not all_chunks:
        print("Embedder: Không có chunks nào để xử lý.")
        return

    print(
        f"Embedder: Bắt đầu quá trình nhúng (embedding) cho {len(all_chunks)} chunks sử dụng model {model_name}...")
    for i, chunk in enumerate(all_chunks):
        if 'content' in chunk and isinstance(chunk['content'], str) and chunk['content'].strip():
            text_to_embed = chunk['content']

            embedding_vector = embed_content_with_sbert(text_to_embed)

            chunk['embedding'] = embedding_vector
            all_embedded_chunks.append(chunk)

            if (i + 1) % 100 == 0 or (i + 1) == len(all_chunks):  # Thông báo tiến trình mỗi 100 chunks
                print(
                    f"  Embedder: Đã xử lý embedding cho {i+1}/{len(all_chunks)} chunks.")
        else:
            print(
                f"Embedder: Cảnh báo - Chunk thứ {i+1} (Type: {chunk.get('type', 'N/A')}, Syllabus ID: {chunk.get('metadata', {}).get('syllabus_id', 'N/A')}) không có 'content' hợp lệ. Bỏ qua embedding.")
            # all_embedded_chunks.append(chunk) # Uncomment nếu vẫn muốn giữ chunk lỗi (không có embedding) trong output

    print(
        f"Embedder: Đã hoàn tất quá trình nhúng. Tổng số chunks được nhúng: {len(all_embedded_chunks)}.")

    if not all_embedded_chunks:
        print("Embedder: Không có chunk nào được nhúng thành công để lưu.")
        return

    print(
        f"Embedder: Đang lưu các chunks đã nhúng vào: {embedded_chunks_file_path}")
    try:
        with open(embedded_chunks_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_embedded_chunks, f, ensure_ascii=False, indent=2)
        print(
            f"Embedder: Đã lưu thành công {len(all_embedded_chunks)} chunks đã nhúng vào '{embedded_chunks_file_path}'.")
    except IOError as e:
        print(f"Embedder: Lỗi khi lưu file embedded chunks: {e}")
    except Exception as e:
        print(
            f"Embedder: Lỗi không xác định khi lưu file embedded chunks: {e}")

    return embedded_chunks_file_path


if __name__ == "__main__":
    output_file_generated_by_main = main()

    print("\n---------------------------------------------------------------------")
    print("| HOÀN TẤT SCRIPT EMBEDDER                                          |")
    print("---------------------------------------------------------------------")
    print(
        f"| Script này đã sử dụng model Sentence Transformer '{model_name}'    |")
    print("| để tạo vector embedding cho các chunks.                           |")
    if output_file_generated_by_main:
        print(
            f"| Kết quả đã được lưu vào file: '{output_file_generated_by_main}'                   |")
    else:
        print(
            f"| Kết quả (có thể) đã được lưu vào file: 'embedded_all_chunks_with_students.json' |")
    print("| Hãy kiểm tra file output và đảm bảo thư viện cần thiết đã        |")
    print("| được cài đặt (pip install sentence-transformers).                 |")
    print("---------------------------------------------------------------------")
