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

    # Tạo FAISS index tối ưu hơn
    # IndexFlatL2 là đơn giản nhưng chính xác
    # Nếu số lượng vectors lớn (>10k), có thể cân nhắc dùng IndexIVFFlat hoặc IndexHNSW
    try:
        if embeddings_np.shape[0] > 10000:
            # IndexIVFFlat cho tập dữ liệu lớn: nhanh hơn nhưng độ chính xác thấp hơn một chút
            print("Tạo index IndexIVFFlat cho tập dữ liệu lớn...")
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(4096, embeddings_np.shape[0] // 39)  # Số lượng clusters
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            # Cần train index trước khi thêm vectors
            print(f"Training index với {embeddings_np.shape[0]} vectors...")
            index.train(embeddings_np)
        else:
            # IndexFlatL2 cho tập dữ liệu nhỏ hơn: chính xác và đủ nhanh
            print("Tạo index IndexFlatL2 cho tập dữ liệu vừa và nhỏ...")
            index = faiss.IndexFlatL2(dimension)
        
        print(f"Đã tạo FAISS index với chiều {dimension}.")
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
        os.makedirs(output_dir)
        
    created_index_file = create_faiss_index(input_file, output_index_path)
    
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
                    "embeddings_file": input_file
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Đã lưu metadata về các môn học ({len(subject_codes)} môn) vào file: {metadata_file}")
        except Exception as e:
            print(f"Lỗi khi lưu metadata về các môn học: {e}")
            
        print(f"Hoàn thành tạo và lưu FAISS index: {created_index_file}")
    else:
        print("Không thể tạo hoặc lưu FAISS index.")