import json
import numpy as np
import os
import time
from datetime import datetime
import torch

# Set environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Now import faiss after setting the environment variable
import faiss

def create_faiss_index(input_embedding_file, output_index_file, index_type='auto', use_gpu=True, nprobe=10):
    """Tạo FAISS index từ file embeddings và lưu index với nhiều tùy chọn cải tiến."""
    start_time = time.time()
    
    try:
        with open(input_embedding_file, 'r', encoding='utf-8') as f:
            chunks_with_embeddings = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file embeddings tại: {input_embedding_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"Lỗi: File embeddings không phải là JSON hợp lệ: {e}")
        return None

    if not chunks_with_embeddings or not isinstance(chunks_with_embeddings, list):
        print("Lỗi: Dữ liệu embeddings rỗng hoặc không phải là một danh sách.")
        return None

    # Trích xuất các vector embedding
    embeddings = []
    for item in chunks_with_embeddings:
        if 'embedding' in item and isinstance(item['embedding'], list):
            embeddings.append(item['embedding'])
        else:
            print(f"Cảnh báo: Bỏ qua item không có embedding hợp lệ: {item.get('metadata', {}).get('chunk_id', 'Unknown ID')}")
            
    if not embeddings:
        print("Lỗi: Không có embeddings nào được trích xuất.")
        return None

    embeddings_np = np.array(embeddings).astype('float32')
    
    if embeddings_np.ndim != 2:
        print(f"Lỗi: Mảng embeddings phải là 2 chiều, nhưng có {embeddings_np.ndim} chiều.")
        return None
        
    num_vectors = embeddings_np.shape[0]
    dimension = embeddings_np.shape[1]
    print(f"Đã trích xuất {num_vectors} embeddings với chiều là {dimension}.")

    # Kiểm tra xem vector đã được chuẩn hóa chưa
    sample_norm = np.linalg.norm(embeddings_np[0])
    is_normalized = abs(sample_norm - 1.0) < 0.01
    
    if is_normalized:
        print("Phát hiện vectors đã được chuẩn hóa. Sẽ sử dụng cosine similarity.")
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        print("Vectors chưa được chuẩn hóa. Sẽ sử dụng L2 distance.")
        metric_type = faiss.METRIC_L2

    # Chọn loại index phù hợp dựa trên số lượng vectors
    if index_type == 'auto':
        if num_vectors < 10000:
            if is_normalized:
                index_type = 'flat_ip'  # Inner product for normalized vectors
            else:
                index_type = 'flat_l2'  # L2 distance for non-normalized vectors
        elif num_vectors < 100000:
            index_type = 'ivf'  # IVF for medium datasets
        else:
            index_type = 'hnsw'  # HNSW for large datasets
    
    print(f"Sử dụng index type: {index_type}")

    # Tạo FAISS index dựa trên loại được chọn
    try:
        if index_type == 'flat_l2':
            index = faiss.IndexFlatL2(dimension)
            
        elif index_type == 'flat_ip':
            index = faiss.IndexFlatIP(dimension)
            
        elif index_type == 'ivf':
            # IVF với số lượng clusters phù hợp với kích thước dữ liệu
            nlist = min(4096, int(np.sqrt(num_vectors) * 4))
            
            if is_normalized:
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
            else:
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
                
            # Cần train index trước khi thêm vectors
            print(f"Training IVF index với {nlist} clusters...")
            index.train(embeddings_np)
            
            # Thiết lập nprobe (số lượng clusters để tìm kiếm)
            index.nprobe = nprobe
            print(f"Đã thiết lập nprobe={nprobe} để cân bằng tốc độ và độ chính xác")
            
        elif index_type == 'hnsw':
            # HNSW index - hiệu quả cho tập dữ liệu lớn
            M = 16  # Số lượng kết nối tối đa cho mỗi node
            ef_construction = 200  # Số lượng neighbors mở rộng lúc xây dựng
            
            index = faiss.IndexHNSWFlat(dimension, M, metric_type)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = 128  # Số lượng neighbors mở rộng lúc tìm kiếm
            
            print(f"Đã tạo HNSW index với M={M}, efConstruction={ef_construction}")
            
        elif index_type == 'ivf_pq':
            # IVF với Product Quantization - cho tập dữ liệu cực lớn, yêu cầu ít bộ nhớ nhưng giảm độ chính xác
            nlist = min(4096, int(np.sqrt(num_vectors) * 4))
            m = min(64, dimension // 2)  # Số lượng subvectors
            bits = 8  # Bits per component
            
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits, metric_type)
            
            print(f"Training IVF-PQ index với {nlist} clusters, {m} subvectors, {bits} bits...")
            index.train(embeddings_np)
            index.nprobe = nprobe
            
        else:
            print("Không nhận dạng được loại index, sử dụng IndexFlatL2 mặc định.")
            index = faiss.IndexFlatL2(dimension)
        
        print(f"Đã tạo FAISS index loại {index_type} với chiều {dimension}.")
        
        # Sử dụng GPU nếu có
        if use_gpu and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print(f"Phát hiện {gpu_count} GPU, sử dụng tất cả để tăng tốc")
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True  # Phân chia index giữa các GPU
                gpu_resources = [faiss.StandardGpuResources() for i in range(gpu_count)]
                index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index, co)
            else:
                print("Đang sử dụng 1 GPU để tăng tốc")
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
    
    except Exception as e:
        print(f"Lỗi khi tạo FAISS index: {e}")
        return None

    # Thêm các vector vào index
    try:
        index.add(embeddings_np)
        print(f"Đã thêm {index.ntotal} vectors vào FAISS index.")
    except Exception as e:
        print(f"Lỗi khi thêm vectors vào FAISS index: {e}")
        return None

    # Đánh giá recall của index
    try:
        # Tạo một bản sao của index để đánh giá
        if use_gpu and torch.cuda.is_available():
            eval_index = faiss.index_gpu_to_cpu(index)
        else:
            eval_index = index
        
        # Đánh giá chất lượng của index với một mẫu ngẫu nhiên
        sample_size = min(100, num_vectors)
        sample_indices = np.random.choice(num_vectors, sample_size, replace=False)
        sample_queries = embeddings_np[sample_indices]
        
        # Tìm kiếm chính xác với Flat index (ground truth)
        if is_normalized:
            gt_index = faiss.IndexFlatIP(dimension)
        else:
            gt_index = faiss.IndexFlatL2(dimension)
        gt_index.add(embeddings_np)
        
        k = 10  # Top-k để đánh giá
        gt_D, gt_I = gt_index.search(sample_queries, k)
        
        # Tìm kiếm với index cần đánh giá
        D, I = eval_index.search(sample_queries, k)
        
        # Tính recall@k
        recall = 0
        for i in range(sample_size):
            recall += len(set(gt_I[i]) & set(I[i])) / k
        recall /= sample_size
        
        print(f"Đánh giá index: Recall@{k} = {recall:.4f}")
        
    except Exception as e:
        print(f"Không thể đánh giá index: {e}")

    # Nếu đang dùng GPU, chuyển về CPU trước khi lưu
    if use_gpu and torch.cuda.is_available():
        try:
            index = faiss.index_gpu_to_cpu(index)
            print("Đã chuyển index từ GPU về CPU để lưu.")
        except Exception as e:
            print(f"Lỗi khi chuyển index từ GPU về CPU: {e}")
            return None

    # Lưu index ra file
    try:
        # Tạo backup nếu file đã tồn tại
        if os.path.exists(output_index_file):
            backup_file = f"{os.path.dirname(output_index_file)}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_faiss.index"
            import shutil
            shutil.copy2(output_index_file, backup_file)
            print(f"Đã tạo backup của index hiện tại tại: {backup_file}")
        
        faiss.write_index(index, output_index_file)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Đã lưu FAISS index vào file: {output_index_file}")
        print(f"Thời gian xử lý: {duration:.2f} giây.")
        
        # Lưu thông tin chi tiết về index
        index_info_file = f"{os.path.dirname(output_index_file)}/index_info.json"
        with open(index_info_file, 'w', encoding='utf-8') as f:
            # Create the JSON data dictionary with proper boolean handling
            index_info = {
                "index_type": index_type,
                "metric_type": "inner_product" if metric_type == faiss.METRIC_INNER_PRODUCT else "l2",
                "dimension": int(dimension),  # Ensure this is an int
                "num_vectors": int(num_vectors),  # Ensure this is an int
                "creation_time": datetime.now().isoformat(),
                "normalized_vectors": bool(is_normalized),  # Convert to Python bool
                "parameters": {
                    "nprobe": int(nprobe) if 'nprobe' in dir(index) else None,
                    "M": 16 if index_type == 'hnsw' else None,
                    "ef_search": 128 if index_type == 'hnsw' else None,
                }
            }
            
            # Add evaluation data if available
            if 'recall' in locals():
                index_info["evaluation"] = {
                    "recall@10": float(recall)
                }
            else:
                index_info["evaluation"] = None
            
            # Use a custom JSON encoder class to handle non-serializable types
            class CustomJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, bool):
                        return int(obj)  # Convert booleans to 0/1
                    elif isinstance(obj, (np.int64, np.int32)):
                        return int(obj)  # Convert numpy integers to Python integers
                    elif isinstance(obj, (np.float64, np.float32)):
                        return float(obj)  # Convert numpy floats to Python floats
                    return super().default(obj)
            
            # Use the custom encoder
            json.dump(index_info, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
            
            print(f"Đã lưu thông tin chi tiết về index vào: {index_info_file}")
        
        return output_index_file
    except Exception as e:
        print(f"Lỗi khi lưu FAISS index: {e}")
        return None

def extract_subject_codes(embedded_chunks):
    """Trích xuất tất cả các mã môn học từ dữ liệu chunks đã embedding."""
    subject_codes = set()
    for chunk in embedded_chunks:
        if 'metadata' in chunk and 'subject_code' in chunk['metadata']:
            subject_codes.add(chunk['metadata']['subject_code'])
    return sorted(list(subject_codes))

if __name__ == "__main__":
    input_file = "Embedded/all_embeddings.json"
    output_index_path = "Faiss/all_syllabi_faiss.index"
    
    output_dir = os.path.dirname(output_index_path)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Cho phép lựa chọn loại index mong muốn
    import argparse
    parser = argparse.ArgumentParser(description='Tạo FAISS index từ file embeddings')
    parser.add_argument('--index-type', choices=['auto', 'flat_l2', 'flat_ip', 'ivf', 'hnsw', 'ivf_pq'], 
                        default='auto', help='Loại index muốn tạo')
    parser.add_argument('--use-gpu', action='store_true', default=True, 
                        help='Sử dụng GPU để tăng tốc độ tạo index')
    parser.add_argument('--nprobe', type=int, default=10, 
                        help='Số lượng clusters để tìm kiếm trong IVF index')
    args = parser.parse_args()
    
    created_index_file = create_faiss_index(
        input_file, 
        output_index_path, 
        index_type=args.index_type,
        use_gpu=args.use_gpu,
        nprobe=args.nprobe
    )
    
    if created_index_file:
        # Lưu thông tin về mã môn học vào file riêng để tham khảo sau này
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                embedded_data = json.load(f)
            
            subject_codes = extract_subject_codes(embedded_data)
            metadata_file = os.path.join(output_dir, "syllabus_metadata.json")
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "subject_codes": subject_codes,
                    "total_chunks": len(embedded_data),
                    "index_path": output_index_path,
                    "embeddings_file": input_file,
                    "created_at": datetime.now().isoformat(),
                    "index_type": args.index_type
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Đã lưu metadata về các môn học ({len(subject_codes)} môn) vào file: {metadata_file}")
        except Exception as e:
            print(f"Lỗi khi lưu metadata về các môn học: {e}")
            
        print(f"Hoàn thành tạo và lưu FAISS index: {created_index_file}")
    else:
        print("Không thể tạo hoặc lưu FAISS index.")