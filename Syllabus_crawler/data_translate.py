import json
import os
import hashlib
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import sys
import subprocess
from datetime import datetime

# Thêm code xác định thư mục script để làm việc với đường dẫn tương đối
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Function để in thông báo thay vì logging nếu cần
def safe_print(message):
    """In thông báo mà không gặp lỗi Unicode trên Windows"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Thay thế các ký tự không hiển thị được
        print(message.encode('ascii', 'replace').decode('ascii'))

# Thống kê
translation_stats = {
    "total_texts": 0,
    "translated": 0,
    "skipped": 0,
    "from_cache": 0,
    "failed": 0,
    "total_chars": 0,
    "start_time": None
}

# Danh sách các mẫu chuỗi không cần dịch
SKIP_PATTERNS = [
    r'^[A-Z]{2,}[0-9]{3}[a-z]?$',  # Mã môn học như CEA201, PFP191, JPD113
    r'^[0-9]+$',                  # Chuỗi chỉ chứa số
    r'^https?://\S+$',            # URLs
    r'^[A-Z]{2,}$',               # Mã viết tắt như CLO, LO, PE, TE
    r'^[^a-zA-Z]*$',              # Chuỗi không chứa chữ cái
    r'^[A-Za-z0-9_]+\.[A-Za-z0-9_]+$',  # Định dạng file/module như json.load
    r'^[0-9]+/[0-9]+/[0-9]+$'     # Định dạng ngày tháng
]

# Danh sách các field không cần dịch
SKIP_FIELDS = [
    "page_url",
    "subject_code",
    "syllabus_id",
    "Syllabus ID",
    "Subject Code",
    "NoCredit",
    "Pre-Requisite",
    "Tools",
    "Scoring Scale",
    "IsApproved",
    "MinAvgMarkToPass",
    "IsActive",
    "ApprovedDate",
    "MaterialDescription",
    "Author",
    "ISBN",
    "Note",
    "CLO Name",
    "CLO Details",
    "S-Download",
    "URLs",
    "link",
    "text",
    "Publisher",
    "PublishedDate",
    "Edition",
    "IsMainMaterial",
    "IsHardCopy",
    "IsOnline",
    "Session",
    "LO",
    "ITU"
]

def should_skip_translation(text):
    """Kiểm tra xem có nên bỏ qua dịch chuỗi này không"""
    if not isinstance(text, str):
        return True
    
    # Bỏ qua chuỗi quá ngắn
    if len(text.strip()) <= 2:
        return True
    
    # Kiểm tra theo các mẫu regex
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, text.strip()):
            return True
            
    return False

def setup_translation_client():
    """Thiết lập client translation"""
    try:
        from deep_translator import GoogleTranslator
        safe_print("Su dung deep_translator.GoogleTranslator")
        return GoogleTranslator(source='en', target='vi')
    except ImportError:
        safe_print("Khong tim thay deep_translator. Dang cai dat...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='en', target='vi')

def load_cache(cache_file):
    """Tải cache từ file nếu tồn tại"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                safe_print(f"Da tai {len(cache_data)} muc tu cache.")
                return cache_data
        except Exception as e:
            safe_print(f"Loi khi doc cache: {e}")
            return {}
    safe_print("Khong tim thay file cache. Tao cache moi.")
    return {}

def save_cache(cache, cache_file):
    """Lưu cache vào file"""
    safe_print(f"Dang luu cache voi {len(cache)} muc vao {cache_file}")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    safe_print("Da luu cache thanh cong")

def translate_text(text, client, cache):
    """Dịch văn bản sang tiếng Việt với cache"""
    global translation_stats
    
    translation_stats["total_texts"] += 1
    translation_stats["total_chars"] += len(text) if isinstance(text, str) else 0
    
    if not isinstance(text, str) or not text.strip():
        return text
    
    # Kiểm tra nếu chuỗi không cần dịch
    if should_skip_translation(text):
        translation_stats["skipped"] += 1
        return text
    
    # Sử dụng hash để làm key cho cache để tránh vấn đề với ký tự đặc biệt
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    if text_hash in cache:
        translation_stats["from_cache"] += 1
        return cache[text_hash]
    
    try:
        start_time = time.time()
        # Sử dụng phương thức translate của deep_translator.GoogleTranslator
        translated = client.translate(text)
        end_time = time.time()
        
        # Giới hạn độ dài văn bản (nếu quá dài có thể dịch từng phần)
        if len(text) > 5000:
            safe_print(f"Van ban qua dai ({len(text)} ky tu), dang chia nho...")
            chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
            translated = ''.join([translate_text(chunk, client, cache) for chunk in chunks])
        
        # Tránh dịch lại các thuật ngữ chuyên ngành
        cache[text_hash] = translated
        translation_stats["translated"] += 1
        return translated
    except Exception as e:
        safe_print(f"Loi dich: {str(e)}")
        safe_print(f"Van ban goc: {text[:50]}..." if len(text) > 50 else f"Van ban goc: {text}")
        
        # Thử lại với độ dài văn bản ngắn hơn nếu là lỗi do độ dài
        if len(text) > 500:
            try:
                safe_print("Thu dich voi do dai van ban ngan hon...")
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                translated_chunks = []
                for i, chunk in enumerate(chunks):
                    chunk_translated = client.translate(chunk)
                    translated_chunks.append(chunk_translated)
                    # Nghỉ giữa các lần dịch
                    time.sleep(0.5)
                translated = ''.join(translated_chunks)
                cache[text_hash] = translated
                translation_stats["translated"] += 1
                return translated
            except Exception as retry_error:
                safe_print(f"Loi khi thu lai: {str(retry_error)}")
        
        # Trả về văn bản gốc nếu lỗi
        translation_stats["failed"] += 1
        return text

def batch_translate(texts, client, cache, max_batch=10):
    """Dịch một loạt văn bản cùng lúc"""
    results = []
    
    # Giảm kích thước batch để tránh vấn đề rate limit
    for i in range(0, len(texts), max_batch):
        batch = texts[i:i+max_batch]
        batch_results = []
        
        batch_start = time.time()
        safe_print(f"Bat dau dich batch {i//max_batch + 1}/{(len(texts)-1)//max_batch + 1} ({len(batch)} van ban)")
        
        # Xử lý tuần tự để tránh quá nhiều yêu cầu cùng lúc
        for j, text in enumerate(batch):
            try:
                result = translate_text(text, client, cache)
                batch_results.append(result)
                # Thêm thời gian nghỉ ngắn giữa các yêu cầu
                time.sleep(0.5)
                # In dấu hiệu tiến độ
                if j % 5 == 0:
                    print(".", end="", flush=True)
            except Exception as e:
                safe_print(f"Loi khi dich: {e}")
                batch_results.append(text)  # Trả về văn bản gốc nếu lỗi
        
        results.extend(batch_results)
        batch_end = time.time()
        print()  # Xuống dòng sau khi hiển thị dấu tiến độ
        
        safe_print(f"Hoan thanh batch {i//max_batch + 1} trong {batch_end - batch_start:.2f}s")
        
        # Tăng thời gian nghỉ để tránh rate limit
        if i + max_batch < len(texts):
            rest_time = 3
            safe_print(f"Da dich {i + max_batch}/{len(texts)} muc. Dang nghi {rest_time}s de tranh rate limit...")
            time.sleep(rest_time)
    
    return results

def process_object(obj, client, cache, path=""):
    """Xử lý đệ quy để dịch các giá trị văn bản trong object JSON"""
    if isinstance(obj, dict):
        safe_print(f"Xu ly dictionary tai {path}")
        result = {}
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            
            # Kiểm tra nếu key nằm trong danh sách các field cần bỏ qua
            if key in SKIP_FIELDS:
                result[key] = value  # Giữ nguyên giá trị không dịch
                continue
                
            # Xử lý đệ quy cho các giá trị khác
            result[key] = process_object(value, client, cache, current_path)
        return result
    elif isinstance(obj, list):
        safe_print(f"Xu ly list tai {path} ({len(obj)} phan tu)")
        # Tối ưu: Thu thập tất cả văn bản cần dịch
        to_translate = []
        indices = []
        result = []
        
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                result.append(process_object(item, client, cache, f"{path}[{i}]"))
            elif isinstance(item, str) and item.strip() and not should_skip_translation(item):
                text_hash = hashlib.md5(item.encode()).hexdigest()
                if text_hash in cache:
                    translation_stats["from_cache"] += 1
                    result.append(cache[text_hash])
                else:
                    to_translate.append(item)
                    indices.append(i)
                    result.append(None)  # Placeholder
            else:
                if isinstance(item, str) and item.strip():
                    translation_stats["skipped"] += 1
                result.append(item)
        
        # Dịch tất cả văn bản một lúc
        if to_translate:
            safe_print(f"Dang dich batch voi {len(to_translate)} muc tai {path}")
            translations = batch_translate(to_translate, client, cache)
            
            # Cập nhật kết quả và cache
            for i, idx in enumerate(indices):
                if i < len(translations):
                    text_hash = hashlib.md5(to_translate[i].encode()).hexdigest()
                    cache[text_hash] = translations[i]
                    result[idx] = translations[i]
                else:
                    safe_print(f"Thieu ban dich cho muc {i} tai {path}")
                    result[idx] = to_translate[i]
        
        return result
    elif isinstance(obj, str) and obj.strip() and not should_skip_translation(obj):
        # Dịch văn bản đơn lẻ
        return translate_text(obj, client, cache)
    else:
        if isinstance(obj, str) and obj.strip():
            translation_stats["skipped"] += 1
        # Giữ nguyên các kiểu dữ liệu khác
        return obj

def print_translation_summary():
    """In ra bảng thống kê về quá trình dịch"""
    if translation_stats["start_time"]:
        total_time = time.time() - translation_stats["start_time"]
        
        print("=" * 50)
        print("THONG KE DICH THUAT")
        print("=" * 50)
        print(f"Tong so van ban xu ly: {translation_stats['total_texts']}")
        print(f"So van ban da dich: {translation_stats['translated']}")
        print(f"So van ban lay tu cache: {translation_stats['from_cache']}")
        print(f"So van ban bo qua: {translation_stats['skipped']}")
        print(f"So van ban loi: {translation_stats['failed']}")
        print(f"Tong so ky tu xu ly: {translation_stats['total_chars']}")
        print(f"Thoi gian chay: {total_time:.2f} giay")
        if total_time > 0:
            print(f"Toc do dich: {translation_stats['translated'] / total_time:.2f} van ban/giay")
            print(f"Toc do xu ly: {translation_stats['total_chars'] / total_time:.2f} ky tu/giay")
        print("=" * 50)

def translate_json_file(input_file, output_file, cache_file='translation_cache.json'):
    """Hàm chính để dịch file JSON"""
    global translation_stats
    translation_stats["start_time"] = time.time()
    
    # Đảm bảo đường dẫn tuyệt đối cho tất cả các file
    input_file_path = os.path.abspath(input_file)
    output_file_path = os.path.abspath(output_file)
    cache_file_path = os.path.abspath(cache_file)
    
    safe_print(f"Duong dan file dau vao: {input_file_path}")
    safe_print(f"Duong dan file dau ra: {output_file_path}")
    safe_print(f"Duong dan file cache: {cache_file_path}")
    
    if not os.path.exists(input_file_path):
        safe_print(f"Loi: Khong tim thay file dau vao tai {input_file_path}")
        return
    
    # Tải dữ liệu và cache
    safe_print(f"Dang tai du lieu tu {input_file_path}...")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        safe_print(f"Da tai du lieu JSON thanh cong ({os.path.getsize(input_file_path) / 1024:.2f} KB)")
    except json.JSONDecodeError as e:
        safe_print(f"Loi khi doc file JSON: {e}")
        return
    except Exception as e:
        safe_print(f"Loi khi mo file: {e}")
        return
    
    # Tải cache
    cache = load_cache(cache_file_path)
    
    # Thiết lập translation client
    client = setup_translation_client()
    
    # Xử lý dịch
    safe_print("Bat dau qua trinh dich...")
    start_time = time.time()
    translated_data = process_object(data, client, cache)
    
    # Lưu kết quả
    safe_print(f"Luu ket qua vao {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    # Lưu cache
    save_cache(cache, cache_file_path)
    
    elapsed_time = time.time() - start_time
    safe_print(f"Hoan thanh trong {elapsed_time:.2f} giay.")
    safe_print(f"Cache hien co {len(cache)} muc.")
    
    print_translation_summary()

if __name__ == "__main__":
    input_file = os.path.join(SCRIPT_DIR, "fpt_syllabus_data_appended_en.json")
    output_file = os.path.join(SCRIPT_DIR, "fpt_syllabus_data_appended_vi.json")
    cache_file = os.path.join(SCRIPT_DIR, "translation_cache.json")
    
    print(f"Bat dau qua trinh dich JSON vao {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    translate_json_file(input_file, output_file, cache_file)