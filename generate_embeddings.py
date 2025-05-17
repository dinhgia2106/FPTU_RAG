import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import torch
import time
from datetime import datetime

def generate_embeddings(input_chunk_file, output_embedding_file, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', batch_size=32, use_fp16=True, normalize_vectors=True):
    """Tạo vector embedding cho các chunk từ file và lưu kết quả với xử lý batch."""
    start_time = time.time()
    
    # Kiểm tra file đầu vào
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

    # Kiểm tra xem đã có embedding cũ chưa để tránh tạo lại các chunk đã có
    existing_embeddings = {}
    if os.path.exists(output_embedding_file):
        try:
            print(f"Tìm thấy file embedding cũ: {output_embedding_file}. Kiểm tra để cập nhật thêm...")
            with open(output_embedding_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                
            for item in existing_data:
                if 'metadata' in item and 'chunk_id' in item['metadata'] and 'embedding' in item:
                    existing_embeddings[item['metadata']['chunk_id']] = {
                        'embedding': item['embedding'],
                        'content': item.get('content', '')
                    }
            
            print(f"Đã tải {len(existing_embeddings)} embeddings hiện có.")
        except Exception as e:
            print(f"Không thể đọc file embedding cũ: {e}")
            existing_embeddings = {}

    # Tải mô hình
    print(f"Đang tải mô hình embedding: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        # Đưa model lên GPU nếu có
        if torch.cuda.is_available():
            print(f"Đang sử dụng GPU {torch.cuda.get_device_name(0)} để tạo embeddings.")
            model = model.to("cuda")
            
            # Sử dụng FP16 nếu được yêu cầu và GPU hỗ trợ
            if use_fp16:
                print("Đang kích hoạt half-precision (FP16) để tăng tốc và tiết kiệm bộ nhớ.")
                model = model.half()
        else:
            print("Không tìm thấy GPU, sẽ chạy trên CPU.")
            
        print("Tải mô hình thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình embedding: {e}")
        return None

    # Xác định những chunk cần tạo embedding mới
    chunks_to_process = []
    chunks_to_skip = []
    
    for chunk in chunks_data:
        chunk_id = chunk.get('metadata', {}).get('chunk_id')
        content = chunk.get('content', '')
        
        # Nếu chunk đã có embedding và nội dung không thay đổi, bỏ qua
        if chunk_id in existing_embeddings and existing_embeddings[chunk_id]['content'] == content:
            chunks_to_skip.append(chunk)
        else:
            chunks_to_process.append(chunk)
    
    print(f"Cần tạo mới: {len(chunks_to_process)} embeddings, giữ nguyên: {len(chunks_to_skip)} embeddings.")
    
    # Nếu không có chunk nào cần xử lý, trả về file hiện tại
    if not chunks_to_process and existing_embeddings:
        print("Không có chunk nào cần cập nhật. Giữ nguyên file embedding hiện tại.")
        return output_embedding_file

    # Xử lý các chunk cần tạo embedding mới
    print(f"Bắt đầu tạo embeddings cho {len(chunks_to_process)} chunks với batch_size={batch_size}...")
    
    # Tạo danh sách các nội dung cần encode
    contents_to_encode = [chunk.get('content', '') for chunk in chunks_to_process]
    
    # Xử lý theo batch để tránh tiêu thụ quá nhiều RAM
    all_embeddings = []
    max_retries = 3
    
    # Sử dụng tqdm để hiển thị thanh tiến trình
    for i in tqdm(range(0, len(contents_to_encode), batch_size), desc="Tạo embeddings theo batch"):
        batch_content = contents_to_encode[i:i+batch_size]
        
        # Thêm retry logic
        for retry in range(max_retries):
            try:
                with torch.no_grad():  # Tăng tốc và tiết kiệm bộ nhớ
                    batch_embeddings = model.encode(
                        batch_content, 
                        show_progress_bar=False, 
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        normalize_embeddings=normalize_vectors  # Chuẩn hóa vector nếu được yêu cầu
                    )
                
                # Chuyển sang float32 để đảm bảo tương thích với JSON
                if batch_embeddings.dtype != np.float32:
                    batch_embeddings = batch_embeddings.astype(np.float32)
                
                all_embeddings.extend(batch_embeddings.tolist())
                break  # Nếu thành công, thoát khỏi retry loop
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Lỗi khi tạo embeddings cho batch {i//batch_size + 1}, thử lại ({retry+1}/{max_retries}): {e}")
                    time.sleep(1)  # Đợi một chút trước khi thử lại
                else:
                    print(f"Lỗi khi tạo embeddings cho batch {i//batch_size + 1} sau {max_retries} lần thử: {e}")
                    # Nếu vẫn lỗi sau tất cả các lần thử, gán vector zero
                    zero_embeddings = np.zeros((len(batch_content), model.get_sentence_embedding_dimension()))
                    all_embeddings.extend(zero_embeddings.tolist())
    
    print("Đã hoàn thành tạo embeddings.")
    
    # Gán embedding vào mỗi chunk cần xử lý
    for i, chunk in enumerate(chunks_to_process):
        chunk['embedding'] = all_embeddings[i]
    
    # Kết hợp các chunk đã xử lý với các chunk đã có embedding
    final_chunks = chunks_to_process.copy()
    
    # Thêm lại các chunk đã có embedding từ trước
    for chunk in chunks_to_skip:
        chunk_id = chunk.get('metadata', {}).get('chunk_id')
        if chunk_id in existing_embeddings:
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = existing_embeddings[chunk_id]['embedding']
            final_chunks.append(chunk_with_embedding)
    
    # Tạo thư mục output nếu chưa tồn tại
    output_dir = os.path.dirname(output_embedding_file)
    os.makedirs(output_dir, exist_ok=True)

    # Lưu kết quả
    try:
        backup_file = f"{output_dir}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_embeddings.json"
        
        # Tạo backup file trước khi ghi đè
        if os.path.exists(output_embedding_file):
            import shutil
            shutil.copy2(output_embedding_file, backup_file)
            print(f"Đã tạo backup tại: {backup_file}")
        
        with open(output_embedding_file, 'w', encoding='utf-8') as f_out:
            json.dump(final_chunks, f_out, ensure_ascii=False)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Đã lưu {len(final_chunks)} chunks cùng với embeddings vào file: {output_embedding_file}")
        print(f"Thời gian xử lý: {duration:.2f} giây.")
        
        return output_embedding_file
    except IOError as e:
        print(f"Lỗi khi lưu file embeddings: {e}")
        
        # Kiểm tra xem có backup không
        if os.path.exists(backup_file):
            print(f"Đã tạo backup tại: {backup_file}")
        
        return None

def get_all_subject_codes(chunks_data):
    """Lấy danh sách tất cả mã môn học từ dữ liệu chunks."""
    subject_codes = set()
    for chunk in chunks_data:
        if 'metadata' in chunk and 'subject_code' in chunk['metadata']:
            subject_codes.add(chunk['metadata']['subject_code'])
    return sorted(list(subject_codes))

if __name__ == "__main__":
    input_file = "Chunk/enhanced_chunks.json"
    output_file = "Embedded/all_embeddings.json"
    
    # Tạo thư mục embedding nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Tạo embedding cho tất cả chunks
    embedding_file = generate_embeddings(
        input_file, 
        output_file, 
        batch_size=64,
        use_fp16=True,  # Sử dụng FP16 để tăng tốc độ và tiết kiệm bộ nhớ
        normalize_vectors=True  # Chuẩn hóa vector cho tìm kiếm cosine similarity tốt hơn
    )
    
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