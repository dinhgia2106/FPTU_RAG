import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

def generate_embeddings(input_chunk_file, output_embedding_file, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', batch_size=32):
    """Tạo vector embedding cho các chunk từ file và lưu kết quả với xử lý batch."""
    try:
        with open(input_chunk_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file chunks tại: {input_chunk_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"Lỗi: File chunks không phải là JSON hợp lệ: {e}")
        return None

    if not chunks_data or not isinstance(chunks_data, list):
        print("Lỗi: Dữ liệu chunks rỗng hoặc không phải là một danh sách.")
        return None

    print(f"Đang tải mô hình embedding: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        # Đưa model lên GPU nếu có
        import torch
        if torch.cuda.is_available():
            print("Đang sử dụng GPU để tạo embeddings.")
            model = model.to("cuda")
        else:
            print("Không tìm thấy GPU, sẽ chạy trên CPU.")
        print("Tải mô hình thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình embedding: {e}")
        return None

    print(f"Bắt đầu tạo embeddings cho {len(chunks_data)} chunks với batch_size={batch_size}...")
    chunks_with_embeddings = []
    
    # Tạo danh sách các nội dung cần encode
    contents_to_encode = [chunk.get('content', '') for chunk in chunks_data]
    
    # Xử lý theo batch để tránh tiêu thụ quá nhiều RAM
    embeddings = []
    
    # Sử dụng tqdm để hiển thị thanh tiến trình
    for i in tqdm(range(0, len(contents_to_encode), batch_size), desc="Tạo embeddings theo batch"):
        batch_content = contents_to_encode[i:i+batch_size]
        try:
            batch_embeddings = model.encode(
                batch_content, 
                show_progress_bar=False, 
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            embeddings.extend(batch_embeddings.tolist())
        except Exception as e:
            print(f"Lỗi khi tạo embeddings cho batch {i//batch_size + 1}: {e}")
            # Nếu lỗi với một batch, gán vector zero để giữ cho indices khớp với chunks
            batch_embeddings = np.zeros((len(batch_content), model.get_sentence_embedding_dimension()))
            embeddings.extend(batch_embeddings.tolist())
    
    print("Đã hoàn thành tạo embeddings.")
    
    # Gán embedding vào mỗi chunk
    for i, chunk in enumerate(chunks_data):
        chunk_with_embedding = chunk.copy()
        chunk_with_embedding['embedding'] = embeddings[i]
        chunks_with_embeddings.append(chunk_with_embedding)

    # Tạo thư mục output nếu chưa tồn tại
    output_dir = os.path.dirname(output_embedding_file)
    os.makedirs(output_dir, exist_ok=True)

    # Lưu kết quả
    try:
        with open(output_embedding_file, 'w', encoding='utf-8') as f_out:
            json.dump(chunks_with_embeddings, f_out, ensure_ascii=False)
        print(f"Đã lưu {len(chunks_with_embeddings)} chunks cùng với embeddings vào file: {output_embedding_file}")
        return output_embedding_file
    except IOError as e:
        print(f"Lỗi khi lưu file embeddings: {e}")
        return None

def get_all_subject_codes(chunks_data):
    """Lấy danh sách tất cả mã môn học từ dữ liệu chunks."""
    subject_codes = set()
    for chunk in chunks_data:
        if 'metadata' in chunk and 'subject_code' in chunk['metadata']:
            subject_codes.add(chunk['metadata']['subject_code'])
    return sorted(list(subject_codes))

if __name__ == "__main__":
    input_file = "Chunk/all_chunks.json"
    output_file = "Embedded/all_embeddings.json"
    
    # Tạo thư mục embedding nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Tạo embedding cho tất cả chunks
    embedding_file = generate_embeddings(input_file, output_file, batch_size=64)
    
    if embedding_file:
        print("Đã hoàn thành tạo embeddings.")
        
        # Hiển thị thông tin về các môn học đã xử lý
        try:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                embedded_data = json.load(f)
            
            subject_codes = get_all_subject_codes(embedded_data)
            print(f"\nĐã xử lý embeddings cho {len(subject_codes)} môn học: {', '.join(subject_codes)}")
        except Exception as e:
            print(f"Lỗi khi đọc file embeddings đã tạo: {e}")
    else:
        print("Không thể hoàn thành việc tạo embeddings.")