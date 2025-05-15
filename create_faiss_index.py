import json
import numpy as np
import faiss
import os

def create_faiss_index(input_embedding_file, output_index_file):
    """Tạo FAISS index từ file embeddings và lưu index."""
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
        
    dimension = embeddings_np.shape[1]
    print(f"Đã trích xuất {embeddings_np.shape[0]} embeddings với chiều là {dimension}.")

    # Tạo FAISS index
    # IndexFlatL2 là một index đơn giản, thực hiện tìm kiếm L2 brute-force.
    # Phù hợp cho số lượng vector không quá lớn hoặc để thử nghiệm ban đầu.
    try:
        index = faiss.IndexFlatL2(dimension)
        print(f"Đã tạo FAISS index (IndexFlatL2) với chiều {dimension}.")
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

    # Lưu index ra file
    try:
        faiss.write_index(index, output_index_file)
        print(f"Đã lưu FAISS index vào file: {output_index_file}")
        return output_index_file # Trả về đường dẫn file index đã lưu
    except Exception as e:
        print(f"Lỗi khi lưu FAISS index: {e}")
        return None

if __name__ == "__main__":
    input_file = "Embedded/PFP191_embeddings.json"
    output_index_path = "Faiss/PFP191_faiss.index"
    
    output_dir = os.path.dirname(output_index_path)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)
        
    created_index_file = create_faiss_index(input_file, output_index_path)
    
    if created_index_file:
        print(f"Hoàn thành tạo và lưu FAISS index: {created_index_file}")
        # Bạn có thể thêm phần kiểm thử truy vấn ở đây nếu muốn
        # Ví dụ: tải lại index và thực hiện tìm kiếm
        # index_loaded = faiss.read_index(created_index_file)
        # D, I = index_loaded.search(np.array([embeddings_np[0]]), k=5) # Tìm 5 vector gần nhất với vector đầu tiên
        # print("Kết quả tìm kiếm mẫu:")
        # print("Distances:", D)
        # print("Indices:", I)
    else:
        print("Không thể tạo hoặc lưu FAISS index.")

