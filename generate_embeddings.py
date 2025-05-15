import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(input_chunk_file, output_embedding_file, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
    """Tạo vector embedding cho các chunk từ file và lưu kết quả."""
    try:
        with open(input_chunk_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file chunks tại: {input_chunk_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Lỗi: File chunks không phải là JSON hợp lệ: {e}")
        return

    if not chunks_data or not isinstance(chunks_data, list):
        print("Lỗi: Dữ liệu chunks rỗng hoặc không phải là một danh sách.")
        return

    print(f"Đang tải mô hình embedding: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("Tải mô hình thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình embedding: {e}")
        return

    chunks_with_embeddings = []
    print(f"Bắt đầu tạo embeddings cho {len(chunks_data)} chunks...")

    contents_to_encode = [chunk.get('content', '') for chunk in chunks_data]

    try:
        embeddings = model.encode(contents_to_encode, show_progress_bar=True)
        print("Tạo embeddings thành công.")
    except Exception as e:
        print(f"Lỗi trong quá trình tạo embeddings: {e}")
        return

    for i, chunk in enumerate(chunks_data):
        chunk_with_embedding = chunk.copy()
        chunk_with_embedding['embedding'] = embeddings[i].tolist()
        chunks_with_embeddings.append(chunk_with_embedding)

    # Tạo thư mục chứa output nếu chưa tồn tại
    output_dir = os.path.dirname(output_embedding_file)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_embedding_file, 'w', encoding='utf-8') as f_out:
            json.dump(chunks_with_embeddings, f_out, ensure_ascii=False, indent=2)
        print(f"Đã lưu các chunks cùng với embeddings vào file: {output_embedding_file}")
    except IOError as e:
        print(f"Lỗi khi lưu file embeddings: {e}")

if __name__ == "__main__":
    input_file = "Chunk/PFP191_chunks.json"
    output_file = "Embedded/PFP191_embeddings.json"
    
    generate_embeddings(input_file, output_file)
