import json
import chromadb
from chromadb.utils import embedding_functions
import os

# --- Configuration ---
# THAM SỐ THỰC: Đường dẫn đến file JSON chứa các chunks đã được nhúng (output từ embedder.py)
INPUT_JSON_FILE = "embedded_all_chunks_with_students.json"

# THAM SỐ THỰC: Đường dẫn để lưu trữ persistent ChromaDB. Nếu để trống, sẽ dùng in-memory.
# Để persistent, hãy tạo thư mục này trước nếu nó chưa tồn tại.
CHROMA_PERSIST_DIRECTORY = "all_data_chroma_db_store"

# THAM SỐ THỰC: Tên collection trong ChromaDB
CHROMA_COLLECTION_NAME = "all_syllabus_and_students_collection"

# THAM SỐ THỰC: Model embedding được sử dụng khi tạo embeddings (phải khớp với embedder.py)
# ChromaDB có thể tự tạo embedding nếu bạn cung cấp text và không cung cấp embedding vector.
# Tuy nhiên, vì chúng ta đã có sẵn embedding từ embedder.py, chúng ta sẽ không dùng tính năng này của Chroma.
# Thay vào đó, chúng ta sẽ dùng embedding_function mặc định "mock" của Chroma hoặc không chỉ định nó
# và truyền thẳng embedding vectors.
# EMBEDDING_MODEL_NAME_FOR_CHROMA = 'all-MiniLM-L6-v2' # Không cần thiết nếu đã có vector

# THAM SỐ THỰC: Xử lý theo batch (số lượng documents mỗi lần add vào collection)
BATCH_SIZE = 100


def load_embedded_chunks(filepath):
    """Tải các chunks đã nhúng từ file JSON."""
    print(f"ChromaDB_Script: Đang đọc các chunks đã nhúng từ: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"ChromaDB_Script: Tải thành công {len(chunks)} chunks.")
        if not chunks or not isinstance(chunks, list) or not isinstance(chunks[0], dict) or 'embedding' not in chunks[0] or 'content' not in chunks[0]:
            print("ChromaDB_Script: Lỗi - Định dạng file chunks không hợp lệ.")
            return []
        return chunks
    except FileNotFoundError:
        print(f"ChromaDB_Script: Lỗi - Không tìm thấy file: {filepath}")
        return []
    except json.JSONDecodeError:
        print(
            f"ChromaDB_Script: Lỗi - File không phải là JSON hợp lệ: {filepath}")
        return []
    except Exception as e:
        print(f"ChromaDB_Script: Lỗi không xác định khi đọc file: {e}")
        return []


def store_chunks_in_chroma(chunks, persist_directory, collection_name):
    """Tạo/Lấy collection trong ChromaDB và điền dữ liệu."""
    if not chunks:
        print("ChromaDB_Script: Không có chunks nào để lưu trữ.")
        return False

    print(f"ChromaDB_Script: Khởi tạo ChromaDB client...")
    try:
        if persist_directory:
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)  # Tạo thư mục nếu chưa có
                print(
                    f"ChromaDB_Script: Đã tạo thư mục persistent: {persist_directory}")
            client = chromadb.PersistentClient(path=persist_directory)
            print(
                f"ChromaDB_Script: Sử dụng PersistentClient tại: {persist_directory}")
        else:
            client = chromadb.Client()  # In-memory client
            print("ChromaDB_Script: Sử dụng In-memory Client.")

        # Tạo hoặc lấy collection.
        # Nếu bạn muốn dùng một embedding function cụ thể của Chroma, hãy chỉ định ở đây.
        # Vì chúng ta đã có embeddings, chúng ta không cần Chroma tự tạo.
        # embedding_function_chroma = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME_FOR_CHROMA)
        # collection = client.get_or_create_collection(
        # name=collection_name, embedding_function=embedding_function_chroma
        # )
        collection = client.get_or_create_collection(name=collection_name)
        print(
            f"ChromaDB_Script: Đã lấy/tạo collection: '{collection_name}' (hiện có {collection.count()} documents)")

        # Chuẩn bị dữ liệu để thêm vào collection
        all_ids = []
        all_embeddings = []
        all_documents = []
        all_metadatas = []

        for i, chunk_data in enumerate(chunks):
            if 'embedding' not in chunk_data or not chunk_data['embedding']:
                print(
                    f"ChromaDB_Script: Cảnh báo - Chunk {i} không có embedding. Bỏ qua.")
                continue

            # Tạo ID duy nhất cho mỗi chunk (có thể dựa trên index hoặc hash của content)
            # Sử dụng index gốc làm ID để có thể dễ dàng tham chiếu nếu cần
            chunk_id = f"chunk_{i}_{chunk_data.get('metadata', {}).get('syllabus_id', 'NA')}_{chunk_data.get('type', 'NA')}"
            all_ids.append(chunk_id)
            all_embeddings.append(chunk_data['embedding'])
            all_documents.append(chunk_data.get('content', ''))

            # Metadata có thể bao gồm bất cứ thông tin gì bạn muốn query sau này
            metadata_to_store = chunk_data.get('metadata', {}).copy()
            metadata_to_store['type'] = chunk_data.get('type', 'N/A')
            # Đảm bảo tất cả giá trị metadata là kiểu dữ liệu Chroma hỗ trợ (str, int, float, bool)
            for key, value in metadata_to_store.items():
                if not isinstance(value, (str, int, float, bool)):
                    # Chuyển sang string nếu không phải kiểu cơ bản
                    metadata_to_store[key] = str(value)
            all_metadatas.append(metadata_to_store)

        if not all_ids:  # Nếu không có document nào hợp lệ để thêm
            print(
                "ChromaDB_Script: Không có document hợp lệ nào để thêm vào collection sau khi xử lý.")
            return False

        print(
            f"ChromaDB_Script: Chuẩn bị thêm {len(all_ids)} documents vào collection '{collection_name}' theo batch size {BATCH_SIZE}...")

        num_batches = (len(all_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(all_ids))

            batch_ids = all_ids[start_idx:end_idx]
            batch_embeddings = all_embeddings[start_idx:end_idx]
            batch_documents = all_documents[start_idx:end_idx]
            batch_metadatas = all_metadatas[start_idx:end_idx]

            print(
                f"  ChromaDB_Script: Đang thêm batch {i+1}/{num_batches} (docs {start_idx+1}-{end_idx}) ...")
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        print(
            f"ChromaDB_Script: Đã thêm thành công tất cả documents. Collection '{collection_name}' hiện có {collection.count()} documents.")
        return True

    except Exception as e:
        print(
            f"ChromaDB_Script: Lỗi trong quá trình lưu trữ vào ChromaDB: {e}")
        # In thêm traceback nếu cần debug chi tiết
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("--- Bắt đầu Script Tạo ChromaDB Vector Store ---")
    # 1. Tải các chunks đã nhúng
    all_loaded_chunks = load_embedded_chunks(INPUT_JSON_FILE)

    if all_loaded_chunks:
        # 2. Lưu trữ chunks vào ChromaDB
        success = store_chunks_in_chroma(
            all_loaded_chunks, CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME)
        if success:
            print(
                f"\n[THÀNH CÔNG] Dữ liệu đã được lưu trữ/cập nhật trong ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
            if CHROMA_PERSIST_DIRECTORY:
                print(
                    f"   Dữ liệu được lưu trữ tại thư mục: {CHROMA_PERSIST_DIRECTORY}")
        else:
            print("\n[THẤT BẠI] Không thể lưu trữ dữ liệu vào ChromaDB.")
    else:
        print(
            f"\n[THẤT BẠI] Không thể tải chunks từ {INPUT_JSON_FILE}. Không thể cập nhật ChromaDB.")

    print("--- Kết thúc Script Tạo ChromaDB Vector Store ---")
