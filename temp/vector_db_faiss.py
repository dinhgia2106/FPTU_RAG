import json
import numpy as np
import faiss
import os

# --- Configuration ---
# THAM SỐ THỰC: Đường dẫn đến file JSON chứa các chunks đã được nhúng (output từ embedder.py)
INPUT_JSON_FILE = "embedded_all_chunks_with_students.json"

# THAM SỐ THỰC: Tên file để lưu FAISS index
FAISS_INDEX_FILE = "all_data.faiss"

# THAM SỐ THỰC: Tên file để lưu mapping từ FAISS ID ngược lại thông tin chunk
FAISS_MAPPING_FILE = "all_data_faiss_mapping.json"

# THAM SỐ THỰC: Loại index FAISS bạn muốn sử dụng.
# "Flat" (IndexFlatL2) là lựa chọn đơn giản, tốt cho tập dữ liệu nhỏ đến trung bình.
# Cho tập dữ liệu lớn hơn, cân nhắc các index như "IVFxxx,Flat" (ví dụ: "IVF256,Flat").
# Nếu sử dụng IVF, bạn sẽ cần đặt nlist (số lượng centroids) khi tạo index.
# faiss.index_factory(dimension, "IVF256,Flat", faiss.METRIC_L2)
# Và index cần được train: index.train(embeddings_np)
FAISS_INDEX_TYPE_STRING = "Flat"
# NLIST_FOR_IVF = 256 # Chỉ cần nếu dùng index loại IVF


def load_embedded_chunks(filepath):
    """Tải các chunks đã nhúng từ file JSON."""
    print(f"FAISS_DB: Đang đọc các chunks đã nhúng từ: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"FAISS_DB: Tải thành công {len(chunks)} chunks.")
        # Xác thực cấu trúc cơ bản
        if not chunks or not isinstance(chunks, list) or not isinstance(chunks[0], dict) or 'embedding' not in chunks[0] or 'content' not in chunks[0]:
            print("FAISS_DB: Lỗi - Định dạng file chunks không hợp lệ. Cần list các dict với keys 'embedding' và 'content'.")
            return []
        return chunks
    except FileNotFoundError:
        print(f"FAISS_DB: Lỗi - Không tìm thấy file: {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"FAISS_DB: Lỗi - File không phải là JSON hợp lệ: {filepath}")
        return []
    except Exception as e:
        print(f"FAISS_DB: Lỗi không xác định khi đọc file: {e}")
        return []


def create_and_save_faiss_index(chunks, index_filepath, mapping_filepath, index_type_str):
    """Tạo, điền dữ liệu và lưu FAISS index cùng với ID mapping."""
    if not chunks:
        print("FAISS_DB: Không có chunks nào để tạo index.")
        return False

    embeddings = [chunk['embedding']
                  for chunk in chunks if 'embedding' in chunk and chunk['embedding']]
    if not embeddings:
        print("FAISS_DB: Không tìm thấy embeddings hợp lệ trong các chunks.")
        return False

    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]

    print(
        f"FAISS_DB: Tạo FAISS index loại '{index_type_str}' với dimension={dimension} cho {len(embeddings_np)} vectors.")
    try:
        if index_type_str == "Flat":
            index = faiss.IndexFlatL2(dimension)
        # Bạn có thể thêm các loại index khác ở đây nếu muốn
        # elif "IVF" in index_type_str:
        #     nlist = NLIST_FOR_IVF
        #     quantizer = faiss.IndexFlatL2(dimension)
        #     index = faiss.index_factory(dimension, index_type_str, faiss.METRIC_L2)
        #     index.train(embeddings_np) # Cần train cho IVF index
        else:
            print(
                f"FAISS_DB: Lỗi - Loại index '{index_type_str}' không được hỗ trợ bởi script này (chỉ hỗ trợ 'Flat').")
            # Hoặc thử index_factory một cách tổng quát hơn:
            # index = faiss.index_factory(dimension, index_type_str, faiss.METRIC_L2)
            # if "IVF" in index_type_str and not index.is_trained:
            #     index.train(embeddings_np)
            return False

        index.add(embeddings_np)
        print(f"FAISS_DB: Đã thêm {index.ntotal} vectors vào index.")

        print(f"FAISS_DB: Đang lưu FAISS index vào: {index_filepath}")
        faiss.write_index(index, index_filepath)
        print("FAISS_DB: Lưu FAISS index thành công.")

        # Tạo và lưu mapping
        # id_to_chunk_info mapping sẽ lưu original_chunk_index và các metadata khác
        # để khi search có thể lấy lại thông tin đầy đủ của chunk.
        id_to_chunk_info = {}
        for i, chunk_data in enumerate(chunks):
            # FAISS ID chính là thứ tự của vector khi add vào (0-indexed)
            # Chúng ta cần lưu index gốc của chunk trong file JSON ban đầu
            # và các thông tin khác không phải là embedding.
            chunk_info_to_save = {
                'original_chunk_index': i,  # Index của chunk trong file input_json_file
                # Xem trước content
                'content_preview': chunk_data.get('content', '')[:100] + "...",
                'type': chunk_data.get('type', 'N/A'),
                'metadata': chunk_data.get('metadata', {})
            }
            # Nếu metadata có syllabus_id hoặc course_id, thêm vào cho tiện
            if 'syllabus_id' in chunk_data.get('metadata', {}):
                chunk_info_to_save['syllabus_id'] = chunk_data['metadata']['syllabus_id']
            if 'course_id' in chunk_data.get('metadata', {}):
                chunk_info_to_save['course_id'] = chunk_data['metadata']['course_id']
            # FAISS ID (i) maps to this info
            id_to_chunk_info[i] = chunk_info_to_save

        print(f"FAISS_DB: Đang lưu mapping file vào: {mapping_filepath}")
        with open(mapping_filepath, 'w', encoding='utf-8') as f_map:
            json.dump(id_to_chunk_info, f_map, ensure_ascii=False, indent=2)
        print("FAISS_DB: Lưu mapping file thành công.")
        return True

    except Exception as e:
        print(
            f"FAISS_DB: Lỗi trong quá trình tạo hoặc lưu FAISS index/mapping: {e}")
        return False


if __name__ == "__main__":
    print("--- Bắt đầu Script Tạo FAISS Vector DB ---")
    # 1. Tải các chunks đã nhúng
    all_loaded_chunks = load_embedded_chunks(INPUT_JSON_FILE)

    if all_loaded_chunks:
        # 2. Tạo và lưu FAISS index + mapping file
        success = create_and_save_faiss_index(
            all_loaded_chunks, FAISS_INDEX_FILE, FAISS_MAPPING_FILE, FAISS_INDEX_TYPE_STRING)
        if success:
            print(
                f"\n[THÀNH CÔNG] FAISS index đã được tạo tại: {FAISS_INDEX_FILE}")
            print(
                f"[THÀNH CÔNG] File mapping đã được tạo tại: {FAISS_MAPPING_FILE}")
        else:
            print("\n[THẤT BẠI] Không thể tạo FAISS index hoặc mapping file.")
    else:
        print(
            f"\n[THẤT BẠI] Không thể tải chunks từ {INPUT_JSON_FILE}. Không thể tạo index.")

    print("--- Kết thúc Script Tạo FAISS Vector DB ---")
