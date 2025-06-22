import json

def reduce_json(data, max_items=3):
    if isinstance(data, dict):
        return {key: reduce_json(value, max_items) for key, value in data.items()}
    elif isinstance(data, list):
        # Lấy 3 item đầu tiên, xử lý từng cái nếu là dict hoặc list
        return [reduce_json(item, max_items) for item in data[:max_items]]
    else:
        return data  # Các kiểu khác giữ nguyên

# Ví dụ: đọc từ file json
with open('Data/combined_data.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Rút gọn
reduced_data = reduce_json(json_data)

# Ghi ra file mới
with open('reduced_data.json', 'w', encoding='utf-8') as f:
    json.dump(reduced_data, f, ensure_ascii=False, indent=4)

print("Đã rút gọn dữ liệu.")
