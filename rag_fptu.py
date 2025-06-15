\
import json
import os
import re
from dotenv import load_dotenv
import numpy as np
import faiss
import google.generativeai as genai

# --- Configuration ---
DOTENV_PATH = os.path.join(os.path.dirname(__file__), '.env')
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'Data', 'reduced_data.json')
EMBEDDING_MODEL_NAME = 'models/embedding-001'
GENERATION_MODEL_NAME = 'gemini-1.5-flash' # Or 'gemini-pro'
MAX_RETRIEVED_DOCS = 10 # Max documents to feed into LLM context

# Global variable to store all documents for filtering in retrieval
all_preprocessed_documents = []

# --- Load API Key ---
load_dotenv(DOTENV_PATH)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_1") # Using the first key as an example
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY_1 not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

# --- Helper Functions ---
def get_embedding(text, model_name=EMBEDDING_MODEL_NAME, task_type="RETRIEVAL_DOCUMENT"): # Default to DOCUMENT
    """Generates embedding for the given text using Gemini."""
    try:
        result = genai.embed_content(model=model_name, content=text, task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding for text '{text[:50]}...' (task: {task_type}): {e}")
        return None

def generate_llm_answer(prompt, model_name=GENERATION_MODEL_NAME):
    """Generates an answer using the Gemini LLM."""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        return "Xin lỗi, đã có lỗi xảy ra khi tạo câu trả lời."

# --- Step 1: Data Ingestion & Preprocessing ---
def load_and_preprocess_data(json_file_path):
    """Loads and preprocesses data from a JSON file."""
    global all_preprocessed_documents
    documents = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {json_file_path}")
        return documents
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return documents

    # 1. Create document for major info
    if isinstance(json_data, dict): # Assuming root is a dict with major info
        major_code = json_data.get("major_code_input", "N/A")
        curriculum_title = json_data.get("curriculum_title_on_page", "N/A")
        curriculum_url = json_data.get("curriculum_url", "N/A")
        major_info_doc = {
            "type": "major_info",
            "major_code": major_code,
            "curriculum_title": curriculum_title,
            "curriculum_url": curriculum_url,
            "text_content": f"Ngành {curriculum_title} có mã là {major_code}. Đường dẫn chương trình là {curriculum_url}."
        }
        documents.append(major_info_doc)

        syllabuses_data = json_data.get("syllabuses", [])
        students_data = json_data.get("students", [])
    elif isinstance(json_data, list): # Fallback if root is a list (e.g. only syllabuses or students)
        print("Warning: JSON root is a list. Assuming it contains syllabuses or students. Major info might be missing.")
        syllabuses_data = [item for item in json_data if "metadata" in item and "course_id" in item["metadata"]] # Heuristic
        students_data = [item for item in json_data if "RollNumber" in item] # Heuristic
    else:
        print("Error: Unexpected JSON structure.")
        return documents


    # 2. Process each syllabus (course)
    for syllabus in syllabuses_data:
        metadata = syllabus.get("metadata", {})
        course_id = metadata.get("course_id", "N/A")
        english_title = metadata.get("english_title", "N/A")
        vietnamese_title = metadata.get("title", "N/A")
        semester = metadata.get("semester_from_curriculum", "N/A")
        credits = metadata.get("credits", "N/A")
        prerequisites = metadata.get("prerequisites", "None")
        description = metadata.get("description", "N/A")
        student_tasks = metadata.get("student_tasks", "N/A")
        course_type_guess = metadata.get("course_type_guess", "N/A")
        approved_date = metadata.get("approved_date", "N/A")
        min_avg_mark_to_pass = metadata.get("min_avg_mark_to_pass", "N/A")

        course_doc = {
            "type": "course",
            "id": course_id,
            "english_title": english_title,
            "vietnamese_title": vietnamese_title,
            "semester": semester,
            "credits": credits,
            "prerequisites": prerequisites if prerequisites else "None",
            "description": description,
            "student_tasks": student_tasks,
            "course_type": course_type_guess,
            "approved_date": approved_date,
            "min_avg_mark_to_pass": min_avg_mark_to_pass,
            "text_content": f"Môn học {english_title} ({course_id}) là môn {course_type_guess}. Nó có {credits} tín chỉ. Mô tả: {description}. Nhiệm vụ sinh viên: {student_tasks}. Môn tiên quyết: {prerequisites if prerequisites else 'Không có'}. Được phê duyệt vào {approved_date}."
        }
        documents.append(course_doc)

        for lo in syllabus.get("learning_outcomes", []):
            lo_doc = {
                "type": "learning_outcome",
                "course_id": course_id,
                "lo_id": lo.get("id", "N/A"),
                "details": lo.get("details", "N/A"),
                "text_content": f"Môn {course_id} có Learning Outcome {lo.get('id', 'N/A')}: {lo.get('details', 'N/A')}"
            }
            documents.append(lo_doc)

        for material in syllabus.get("materials", []):
            is_hard_copy = material.get("is_hard_copy", False)
            is_online = material.get("is_online", False)
            form = 'bản cứng' if is_hard_copy else 'online' if is_online else 'không rõ'
            material_doc = {
                "type": "material",
                "course_id": course_id,
                "description": material.get("description", "N/A"),
                "author": material.get("author", "N/A"),
                "publisher": material.get("publisher", "N/A"),
                "published_date": material.get("published_date", "N/A"),
                "edition": material.get("edition", "N/A"),
                "isbn": material.get("isbn", "N/A"),
                "is_main_material": material.get("is_main_material", False),
                "is_hard_copy": is_hard_copy,
                "is_online": is_online,
                "note": material.get("note", "N/A"),
                "text_content": f"Môn {course_id} có tài liệu: {material.get('description', 'N/A')} của {material.get('author', 'N/A')}, xuất bản bởi {material.get('publisher', 'N/A')} ({material.get('published_date', 'N/A')}). Là tài liệu chính: {material.get('is_main_material', False)}. Hình thức: {form}."
            }
            documents.append(material_doc)

        for assessment in syllabus.get("assessments", []):
            assessment_doc = {
                "type": "assessment",
                "course_id": course_id,
                "category": assessment.get("category", "N/A"),
                "weight": assessment.get("weight", "N/A"),
                "duration": assessment.get("duration", "N/A"),
                "clos": assessment.get("clos", []),
                "note": assessment.get("note", "N/A"),
                "text_content": f"Môn {course_id} có hình thức đánh giá {assessment.get('category', 'N/A')} với trọng số {assessment.get('weight', 'N/A')}%. Thời lượng: {assessment.get('duration', 'N/A')}. Đánh giá các CLO: {assessment.get('clos', [])}. Ghi chú: {assessment.get('note', 'N/A')}."
            }
            documents.append(assessment_doc)

        for schedule_entry in syllabus.get("schedule", []):
            schedule_doc = {
                "type": "schedule_entry",
                "course_id": course_id,
                "session": schedule_entry.get("session", "N/A"),
                "topic": schedule_entry.get("topic", "N/A"),
                "teaching_type": schedule_entry.get("teaching_type", "N/A"),
                "learning_outcomes_ids": schedule_entry.get("learning_outcomes", []),
                "itu": schedule_entry.get("itu", "N/A"),
                "download_link": schedule_entry.get("download_link", "N/A"),
                "text_content": f"Môn {course_id} có buổi học: {schedule_entry.get('session', 'N/A')}, chủ đề: {schedule_entry.get('topic', 'N/A')}. ITU: {schedule_entry.get('itu', 'N/A')}. Link download: {schedule_entry.get('download_link', 'N/A')}."
            }
            documents.append(schedule_doc)

    # 3. Process each student
    for student in students_data:
        student_doc = {
            "type": "student",
            "roll_number": student.get("RollNumber", "N/A"),
            "full_name": student.get("Fullname", "N/A"),
            "email": student.get("Email", "N/A"),
            "major": student.get("Major", "N/A"),
            "text_content": f"Sinh viên {student.get('Fullname', 'N/A')}, mã số {student.get('RollNumber', 'N/A')}, email {student.get('Email', 'N/A')}, học ngành {student.get('Major', 'N/A')}."
        }
        documents.append(student_doc)
    
    all_preprocessed_documents = documents # Store for global access if needed by retrieval
    return documents

# --- Step 2: Indexing ---
def create_indexes(documents):
    """Creates vector and metadata indexes from documents."""
    if not documents:
        print("No documents to index.")
        return None, {}, []

    # Initialize Vector Database (FAISS)
    # Determine embedding dimension by embedding a sample text
    sample_embedding = get_embedding("sample text", task_type="RETRIEVAL_DOCUMENT")
    if sample_embedding is None:
        print("Failed to get sample embedding. Cannot initialize FAISS index.")
        return None, {}, []
    embedding_dim = len(sample_embedding)
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    
    metadata_index = {}
    indexed_documents_for_faiss = [] # To map FAISS indices back to documents

    print(f"Starting indexing for {len(documents)} documents...")
    for i, doc in enumerate(documents):
        text_for_embedding = doc.get("text_content", "")
        if not text_for_embedding:
            print(f"Warning: Document {i} has no text_content. Skipping embedding.")
            continue

        embedding = get_embedding(text_for_embedding, task_type="RETRIEVAL_DOCUMENT")
        if embedding is None:
            print(f"Warning: Failed to generate embedding for document {i}. Skipping.")
            continue

        faiss_index.add(np.array([embedding], dtype=np.float32))
        indexed_documents_for_faiss.append(doc) # Store the doc itself

        # Populate Metadata Index
        doc_type = doc.get("type")
        if doc_type == "course":
            course_id = doc.get("id", "").lower()
            eng_title = doc.get("english_title", "").lower()
            vie_title = doc.get("vietnamese_title", "").lower()
            semester = doc.get("semester") # Semester can be int or string
            prereqs = doc.get("prerequisites", "None")
            if prereqs is None: prereqs = "None"


            metadata_index.setdefault("course_id_lookup", {})[course_id] = doc
            if eng_title: metadata_index.setdefault("course_name_lookup", {})[eng_title] = doc
            if vie_title: metadata_index.setdefault("course_name_lookup", {})[vie_title] = doc
            if semester is not None: # Handle potential None semester
                 metadata_index.setdefault("semester_lookup", {}).setdefault(semester, []).append(doc)

            if prereqs != "None" and prereqs != "":
                metadata_index.setdefault("prerequisite_relations", {})[course_id] = prereqs.lower()
                # Ensure prereqs.lower() is treated as a single key or split if it represents multiple courses
                # For simplicity, assuming prereqs is a single course ID or a comma-separated string
                prereq_items = [p.strip() for p in prereqs.lower().split(',')]
                for p_item in prereq_items:
                    if p_item:
                        metadata_index.setdefault("is_prerequisite_for", {}).setdefault(p_item, []).append(course_id)


        elif doc_type == "learning_outcome":
            course_id = doc.get("course_id", "").lower()
            lo_id = doc.get("lo_id", "").lower()
            metadata_index.setdefault("clo_lookup", {})[f"{course_id}_{lo_id}"] = doc

        elif doc_type == "student":
            roll_number = doc.get("roll_number", "").lower()
            full_name = doc.get("full_name", "").lower()
            major = doc.get("major", "").lower()

            if roll_number: metadata_index.setdefault("student_id_lookup", {})[roll_number] = doc
            if full_name: metadata_index.setdefault("student_name_lookup", {})[full_name] = doc
            if major: metadata_index.setdefault("students_by_major", {}).setdefault(major, []).append(doc)
        
        if (i + 1) % 10 == 0:
            print(f"Indexed {i+1}/{len(documents)} documents.")

    print("Indexing complete.")
    return faiss_index, metadata_index, indexed_documents_for_faiss

# --- Step 3: Query Understanding ---
def understand_query(query_string, metadata_index):
    """Analyzes the user query to identify entities and intent."""
    processed_query = query_string.lower()
    original_query_for_entity_removal = processed_query # Work on a copy for entity removal

    identified_entities = {
        "course_ids": [], "student_ids": [], "student_names": [],
        "clo_ids": [], "semesters": [], "keywords": []
    }

    # Check course IDs
    for course_id_key in metadata_index.get("course_id_lookup", {}).keys():
        if course_id_key in original_query_for_entity_removal:
            identified_entities["course_ids"].append(course_id_key)
            original_query_for_entity_removal = original_query_for_entity_removal.replace(course_id_key, "").strip()

    # Check course names
    for course_name_key in metadata_index.get("course_name_lookup", {}).keys():
        if course_name_key in original_query_for_entity_removal:
            course_doc = metadata_index["course_name_lookup"][course_name_key]
            identified_entities["course_ids"].append(course_doc["id"].lower())
            original_query_for_entity_removal = original_query_for_entity_removal.replace(course_name_key, "").strip()
    identified_entities["course_ids"] = list(set(identified_entities["course_ids"])) # Unique

    # Check student IDs
    for student_id_key in metadata_index.get("student_id_lookup", {}).keys():
        if student_id_key in original_query_for_entity_removal:
            identified_entities["student_ids"].append(student_id_key)
            original_query_for_entity_removal = original_query_for_entity_removal.replace(student_id_key, "").strip()

    # Check student names
    for student_name_key in metadata_index.get("student_name_lookup", {}).keys():
        if student_name_key in original_query_for_entity_removal:
            student_doc = metadata_index["student_name_lookup"][student_name_key]
            identified_entities["student_names"].append(student_name_key)
            identified_entities["student_ids"].append(student_doc["roll_number"].lower())
            original_query_for_entity_removal = original_query_for_entity_removal.replace(student_name_key, "").strip()
    identified_entities["student_ids"] = list(set(identified_entities["student_ids"])) # Unique


    # Check CLO IDs (e.g., clo1, clo10 for a specific course if identified)
    # This needs context of a course. If a course is identified, we can be more specific.
    # For now, a general regex.
    for match in re.finditer(r"clo\s*\d+", original_query_for_entity_removal):
        identified_entities["clo_ids"].append(match.group(0))
        original_query_for_entity_removal = original_query_for_entity_removal.replace(match.group(0), "").strip()
    
    # Check semester
    # Using lookbehind and lookahead to avoid partial matches like "semester" in "semesters"
    semester_matches = re.finditer(r"(?i)(?:k[iìy]\s*|semester\s*)(\d+)", original_query_for_entity_removal)
    for match in semester_matches:
        semester_num_str = match.group(1)
        if semester_num_str:
            try:
                identified_entities["semesters"].append(int(semester_num_str))
                # Be careful with replacing, as "kỳ 1" vs "1"
                original_query_for_entity_removal = original_query_for_entity_removal.replace(match.group(0), "").strip()
            except ValueError:
                print(f"Warning: Could not parse semester number {semester_num_str}")


    identified_entities["keywords"] = [kw for kw in original_query_for_entity_removal.split() if kw]

    # Intent Classification (Simplified)
    intent = "general_info"
    q_lower = query_string.lower()
    if "tiên quyết" in q_lower or "prerequisite" in q_lower: intent = "prerequisite_info"
    elif "learning outcome" in q_lower or "clo" in q_lower: intent = "learning_outcome_info"
    elif "tài liệu" in q_lower or "material" in q_lower: intent = "material_info"
    elif "sinh viên" in q_lower or "student" in q_lower: intent = "student_info"
    elif any(k in q_lower for k in ["kì", "kỳ", "semester"]) and identified_entities["semesters"]: intent = "semester_info"
    elif "đánh giá" in q_lower or "assessment" in q_lower: intent = "assessment_info"
    elif "lịch học" in q_lower or "schedule" in q_lower: intent = "schedule_info"
    elif "bao nhiêu môn" in q_lower or "tổng cộng bao nhiêu" in q_lower: intent = "count_courses" # Needs specific handling
    elif "liên quan đến toán" in q_lower or "môn toán" in q_lower : intent = "math_related_courses"

    return {
        "original_query": query_string,
        "identified_entities": identified_entities,
        "intent": intent,
        "remaining_query": original_query_for_entity_removal # For semantic search if needed
    }

# --- Step 4: Retrieval ---
def retrieve_documents(processed_query_info, faiss_idx, indexed_docs_list, metadata_idx, top_k_semantic=5):
    """Retrieves relevant documents based on query analysis."""
    retrieved_docs = []
    seen_doc_keys = set() # To ensure uniqueness

    def add_doc_if_not_seen(doc):
        # Create a unique key for the document
        # Prioritize specific IDs, then fall back to text_content hash if no ID.
        unique_key_parts = [doc.get("type")]
        if doc.get("type") == "course": unique_key_parts.append(doc.get("id"))
        elif doc.get("type") == "student": unique_key_parts.append(doc.get("roll_number"))
        elif doc.get("type") == "learning_outcome": unique_key_parts.append(doc.get("course_id") + "_" + doc.get("lo_id"))
        # Add more specific ID parts for other types if available
        
        # If no specific ID, use a hash of text_content as a fallback (less reliable for exact matches but helps)
        if len(unique_key_parts) == 1: # Only type was added
             unique_key_parts.append(str(hash(doc.get("text_content", ""))))
        
        doc_key = tuple(filter(None,unique_key_parts))


        if doc_key not in seen_doc_keys:
            retrieved_docs.append(doc)
            seen_doc_keys.add(doc_key)

    entities = processed_query_info["identified_entities"]
    intent = processed_query_info["intent"]

    # 1. Metadata Index Retrieval
    if entities["course_ids"]:
        for course_id in entities["course_ids"]:
            course_doc = metadata_idx.get("course_id_lookup", {}).get(course_id)
            if course_doc: add_doc_if_not_seen(course_doc)

            # Add related info based on intent
            if intent == "learning_outcome_info":
                for doc in all_preprocessed_documents: # Search all docs
                    if doc.get("type") == "learning_outcome" and doc.get("course_id", "").lower() == course_id:
                        add_doc_if_not_seen(doc)
            elif intent == "material_info":
                 for doc in all_preprocessed_documents:
                    if doc.get("type") == "material" and doc.get("course_id", "").lower() == course_id:
                        add_doc_if_not_seen(doc)
            elif intent == "assessment_info":
                 for doc in all_preprocessed_documents:
                    if doc.get("type") == "assessment" and doc.get("course_id", "").lower() == course_id:
                        add_doc_if_not_seen(doc)
            elif intent == "schedule_info":
                 for doc in all_preprocessed_documents:
                    if doc.get("type") == "schedule_entry" and doc.get("course_id", "").lower() == course_id:
                        add_doc_if_not_seen(doc)


    if entities["student_ids"]:
        for student_id in entities["student_ids"]:
            student_doc = metadata_idx.get("student_id_lookup", {}).get(student_id)
            if student_doc: add_doc_if_not_seen(student_doc)

    if intent == "prerequisite_info":
        if entities["course_ids"]:
            target_course_id = entities["course_ids"][0] # Assuming first one
            prereq_ids_str = metadata_idx.get("prerequisite_relations", {}).get(target_course_id)
            if prereq_ids_str:
                prereq_ids = [p.strip() for p in prereq_ids_str.split(',')]
                for prereq_id in prereq_ids:
                    if prereq_id:
                        prereq_course_doc = metadata_idx.get("course_id_lookup", {}).get(prereq_id)
                        if prereq_course_doc: add_doc_if_not_seen(prereq_course_doc)
        # elif "có môn tiên quyết không" in processed_query_info["original_query"].lower():
        #     for doc in all_preprocessed_documents:
        #         if doc.get("type") == "course" and doc.get("prerequisites") and doc.get("prerequisites") != "None":
        #             add_doc_if_not_seen(doc)


    if intent == "semester_info" and entities["semesters"]:
        semester_num = entities["semesters"][0] # Assuming first one
        courses_in_sem = metadata_idx.get("semester_lookup", {}).get(semester_num, [])
        for course_doc in courses_in_sem:
            add_doc_if_not_seen(course_doc)
    
    if intent == "student_info" and "danh sách sinh viên ngành ai" in processed_query_info["original_query"].lower():
        # Assuming "ai" is the major key. This should be more robust, e.g. extract major from query.
        ai_students = metadata_idx.get("students_by_major", {}).get("ai", []) # 'ai' should be normalized
        for student_doc in ai_students:
            add_doc_if_not_seen(student_doc)


    # 2. Vector Database (Semantic Search)
    # Use remaining query keywords or original query if few/no metadata hits
    query_text_for_semantic_search = " ".join(entities["keywords"])
    if not query_text_for_semantic_search.strip() and not retrieved_docs: # If no keywords and no metadata hits
        query_text_for_semantic_search = processed_query_info["original_query"]

    effective_query_for_semantic_search = query_text_for_semantic_search
    if intent == "math_related_courses":
        # For math_related_courses intent, use a more specific set of keywords for semantic search
        effective_query_for_semantic_search = "môn học toán, giải tích, đại số, logic, xác suất thống kê, mathematical concepts, calculus, algebra, statistics, discrete math, algorithms"
        print(f"DEBUG: Using specialized semantic query for math: '{effective_query_for_semantic_search}'")


    if effective_query_for_semantic_search.strip() and \
       (intent == "general_info" or not retrieved_docs or intent == "math_related_courses"):
        if faiss_idx and faiss_idx.ntotal > 0:
            query_embedding = get_embedding(effective_query_for_semantic_search, task_type="RETRIEVAL_QUERY") # Use RETRIEVAL_QUERY here
            if query_embedding:
                distances, indices = faiss_idx.search(np.array([query_embedding], dtype=np.float32), k=top_k_semantic)
                print(f"DEBUG: Semantic search for '{effective_query_for_semantic_search}' returned indices: {indices}, distances: {distances}") # Debugging
                for idx_val in indices[0]: # indices is 2D array, e.g., [[idx1, idx2, ...]]
                    if idx_val != -1: # FAISS can return -1 if not enough neighbors or k > ntotal
                        doc = indexed_docs_list[idx_val]
                        # Specific filter for math_related_courses: only add if it's a course
                        if intent == "math_related_courses" and doc.get("type") != "course":
                            print(f"DEBUG: Skipping non-course doc ({doc.get('type')}) for math_related_courses: {doc.get('id', doc.get('text_content','N/A')[:30])}") # Debugging
                            continue
                        add_doc_if_not_seen(doc)
        else:
            print("FAISS index is empty or not initialized. Skipping semantic search.")


    return retrieved_docs[:MAX_RETRIEVED_DOCS]


# --- Step 5: Generation ---
def generate_formatted_answer(original_query, retrieved_documents):
    """Generates a formatted answer using LLM with retrieved documents as context."""
    context_string = ""
    if not retrieved_documents:
        return "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong dữ liệu."

    for doc in retrieved_documents:
        context_string += f"--- Thông tin loại: {doc.get('type', 'N/A').upper()} ---\n"
        if doc.get("type") == "course":
            context_string += f"Mã môn: {doc.get('id')}, Tên: {doc.get('english_title')} ({doc.get('vietnamese_title')}), Tín chỉ: {doc.get('credits')}, Kì: {doc.get('semester')}, Tiên quyết: {doc.get('prerequisites', 'None')}, Mô tả: {doc.get('description')}\n"
        elif doc.get("type") == "learning_outcome":
            context_string += f"Môn {doc.get('course_id')}, CLO {doc.get('lo_id')}: {doc.get('details')}\n"
        elif doc.get("type") == "material":
            context_string += f"Môn {doc.get('course_id')}, Tài liệu: {doc.get('description')}, Tác giả: {doc.get('author')}, Nhà xuất bản: {doc.get('publisher')}, Là chính: {doc.get('is_main_material')}, Online: {doc.get('is_online')}, Ghi chú: {doc.get('note')}\n"
        elif doc.get("type") == "assessment":
            context_string += f"Môn {doc.get('course_id')}, Đánh giá: {doc.get('category')}, Trọng số: {doc.get('weight')}%, CLO: {doc.get('clos')}, Ghi chú: {doc.get('note')}\n"
        elif doc.get("type") == "schedule_entry":
            context_string += f"Môn {doc.get('course_id')}, Buổi: {doc.get('session')}, Chủ đề: {doc.get('topic')}, ITU: {doc.get('itu')}, Link: {doc.get('download_link')}\n"
        elif doc.get("type") == "student":
            context_string += f"Sinh viên: {doc.get('full_name')}, MSV: {doc.get('roll_number')}, Email: {doc.get('email')}, Ngành: {doc.get('major')}\n"
        elif doc.get("type") == "major_info":
            context_string += f"Ngành: {doc.get('curriculum_title')}, Mã ngành: {doc.get('major_code')}, URL: {doc.get('curriculum_url')}\n"
        else: # Generic fallback for other types
            context_string += json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
        context_string += "\n"

    prompt = f"""
Bạn là một trợ lý thông minh chuyên về thông tin chương trình đào tạo của FPT University.
Dựa vào các thông tin sau đây, hãy trả lời câu hỏi của người dùng một cách chính xác, đầy đủ và thân thiện.
Nếu thông tin không có trong ngữ cảnh được cung cấp, hãy cho biết bạn không tìm thấy thông tin đó.
Không bịa đặt thông tin.

Ngữ cảnh thông tin:
{context_string}

Câu hỏi của người dùng: {original_query}

Trả lời:
"""
    return generate_llm_answer(prompt)

# --- Main Pipeline Execution ---
if __name__ == "__main__":
    print("Initializing RAG pipeline...")

    # 1. Load and Preprocess Data
    print(f"Loading data from: {DATA_FILE_PATH}")
    documents = load_and_preprocess_data(DATA_FILE_PATH)
    if not documents:
        print("No documents loaded. Exiting.")
        exit()
    print(f"Loaded and preprocessed {len(documents)} documents.")

    # 2. Create Indexes
    faiss_index, metadata_idx, indexed_docs_for_faiss_retrieval = create_indexes(documents)
    if not faiss_index:
        print("Failed to create indexes. Exiting.")
        exit()
    print("Indexes created successfully.")

    # 3. Example Query
    # query = "Môn SWP391 là gì và có bao nhiêu tín chỉ?"
    # query = "Thông tin về sinh viên Nguyễn Văn An?" # Requires student data in reduced_data.json
    # query = "Môn nào là tiên quyết của SWP391?"
    # query = "Các môn học trong kỳ 1?"
    # query = "CLO 1 của môn SWP391 là gì?"
    # query = "Danh sách sinh viên ngành AI?" # Requires student data with major "AI"
    query = "Tìm các môn học liên quan đến toán"


    print(f"\\n--- Processing Query: \"{query}\" ---")

    # 3. Understand Query
    processed_query_info = understand_query(query, metadata_idx)
    print(f"Processed Query Info: {json.dumps(processed_query_info, ensure_ascii=False, indent=2)}")

    # 4. Retrieve Documents
    retrieved = retrieve_documents(processed_query_info, faiss_index, indexed_docs_for_faiss_retrieval, metadata_idx)
    print(f"Retrieved {len(retrieved)} documents for the query.")
    # for i, doc in enumerate(retrieved):
    # print(f"  Doc {i+1} ({doc.get('type')}): {doc.get('text_content', '')[:100]}...")


    # 5. Generate Answer
    answer = generate_formatted_answer(query, retrieved)
    print("\\n--- Generated Answer ---")
    print(answer)

    # Example of how to run a different query:
    # while True:
    #     user_query = input("\\nEnter your query (or type 'exit' to quit): ")
    #     if user_query.lower() == 'exit':
    #         break
    #     if not user_query.strip():
    #         continue
        
    #     print(f"--- Processing Query: \"{user_query}\" ---")
    #     processed_query_info = understand_query(user_query, metadata_idx)
    #     # print(f"Processed Query Info: {json.dumps(processed_query_info, ensure_ascii=False, indent=2)}")
        
    #     retrieved = retrieve_documents(processed_query_info, faiss_index, indexed_docs_for_faiss_retrieval, metadata_idx)
    #     # print(f"Retrieved {len(retrieved)} documents.")
        
    #     final_answer = generate_formatted_answer(user_query, retrieved)
    #     print("\\n--- Generated Answer ---")
    #     print(final_answer)
    print("\\nPipeline finished.")
