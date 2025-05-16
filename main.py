import os
import subprocess
import time
import sys
import io

def find_python_command():
    """Tìm lệnh Python phù hợp trên hệ thống."""
    # Các lệnh Python phổ biến trên Windows
    commands = ["python", "py", "python3"]
    
    for cmd in commands:
        try:
            # Kiểm tra xem lệnh có tồn tại không
            result = subprocess.run(f"{cmd} --version", 
                                   shell=True, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
            if result.returncode == 0:
                print(f"Đã tìm thấy Python: {result.stdout.decode().strip()}")
                return cmd
        except:
            pass
    
    # Nếu không tìm thấy lệnh Python nào
    print("CẢNH BÁO: Không tìm thấy lệnh Python trong hệ thống.")
    print("Sử dụng đường dẫn đầy đủ của Python hiện tại...")
    return sys.executable

def run_command(command, description):
    """Thực thi một lệnh và hiển thị tiến trình."""
    print(f"\n==== {description} ====")
    print(f"Lệnh: {command}")
    start_time = time.time()
    
    # Thêm PYTHONIOENCODING=utf-8 để đảm bảo output UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            env=env
        )
        
        # Hiển thị output theo thời gian thực
        while process.poll() is None:
            output = process.stdout.readline()
            if output:
                print(output.strip())
        
        # Lấy output và error còn lại
        stdout, stderr = process.communicate()
        
        if stdout:
            print(stdout.strip())
        
        # Hiển thị lỗi nếu có
        if stderr:
            print(f"Lỗi:\n{stderr.strip()}")
            
        if process.returncode != 0:
            print(f"Lệnh thất bại với mã lỗi: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"Lỗi khi thực thi lệnh: {e}")
        return False
        
    end_time = time.time()
    print(f"==== Hoàn thành {description} trong {end_time - start_time:.2f} giây ====\n")
    return True

def main():
    # Đặt UTF-8 cho sys.stdout để đảm bảo in được tiếng Việt
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    # Tạo các thư mục cần thiết
    os.makedirs("Chunk", exist_ok=True)
    os.makedirs("Embedded", exist_ok=True)
    os.makedirs("Faiss", exist_ok=True)
    
    # Tìm lệnh Python phù hợp
    python_cmd = find_python_command()
    
    # Kiểm tra tồn tại của các file script
    scripts = {
        "syllabus_chunker.py": "Xử lý và chia nhỏ syllabus",
        "generate_embeddings.py": "Tạo embeddings",
        "create_faiss_index.py": "Tạo FAISS index",
        "app.py": "Ứng dụng web Streamlit"
    }
    
    for script in scripts:
        if not os.path.exists(script):
            alternative = script.replace("", "")
            if os.path.exists(alternative):
                print(f"Không tìm thấy {script}, nhưng tìm thấy {alternative}. Sẽ sử dụng file này thay thế.")
                scripts[script] = alternative
            else:
                print(f"CẢNH BÁO: Không tìm thấy script {script} hoặc phiên bản thay thế!")
    
    # Bước 1: Xử lý và chia nhỏ tất cả các syllabus
    chunker_script = "syllabus_chunker.py"
    if os.path.exists("syllabus_chunker.py") and not os.path.exists(chunker_script):
        chunker_script = "syllabus_chunker.py"
        
    if not run_command(f"{python_cmd} {chunker_script}", "Xử lý và chia nhỏ syllabus"):
        print("Không thể tiếp tục do lỗi ở bước xử lý syllabus.")
        return
    
    # Bước 2: Tạo embeddings cho tất cả các chunks
    embedding_script = "generate_embeddings.py" 
    if os.path.exists("generate_embeddings.py") and not os.path.exists(embedding_script):
        embedding_script = "generate_embeddings.py"
        
    if not run_command(f"{python_cmd} {embedding_script}", "Tạo embeddings"):
        print("Không thể tiếp tục do lỗi ở bước tạo embeddings.")
        return
    
    # Bước 3: Tạo FAISS index
    index_script = "create_faiss_index.py"
    if os.path.exists("create_faiss_index.py") and not os.path.exists(index_script):
        index_script = "create_faiss_index.py"
        
    if not run_command(f"{python_cmd} {index_script}", "Tạo FAISS index"):
        print("Không thể tiếp tục do lỗi ở bước tạo FAISS index.")
        return
    
    # Bước 4: Chạy ứng dụng web Streamlit
    print("\n==== Khởi động Web Interface ====")
    print("Đang chạy ứng dụng Streamlit...")
    print("Vui lòng đợi trong khi ứng dụng khởi động...")
    run_command(f"streamlit run app.py", "Chạy ứng dụng web")

if __name__ == "__main__":
    main()