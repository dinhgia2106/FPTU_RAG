import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    """Định nghĩa các loại truy vấn"""
    CURRICULUM_OVERVIEW = "curriculum_overview"  # Lộ trình học tổng quan
    SEMESTER_DETAILS = "semester_details"        # Chi tiết học kỳ
    COURSE_INFO = "course_info"                  # Thông tin môn học
    STUDENT_INFO = "student_info"                # Thông tin sinh viên
    PREREQUISITES = "prerequisites"              # Môn tiên quyết
    ASSESSMENT_INFO = "assessment_info"          # Thông tin đánh giá
    LEARNING_OUTCOMES = "learning_outcomes"      # Kết quả học tập
    SCHEDULE_INFO = "schedule_info"              # Lịch học
    MATERIALS_INFO = "materials_info"            # Tài liệu học tập
    UNKNOWN = "unknown"

@dataclass
class QueryIntent:
    """Cấu trúc mô tả ý định truy vấn"""
    query_type: QueryType
    major_code: Optional[str] = None
    semester: Optional[int] = None
    course_id: Optional[str] = None
    course_name: Optional[str] = None
    student_id: Optional[str] = None
    confidence: float = 0.0
    raw_query: str = ""

class CurriculumQueryProcessor:
    """Bộ xử lý truy vấn thông minh cho dữ liệu chương trình đào tạo"""
    
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path
        self.curriculum_data = self._load_curriculum_data()
        self.course_mapping = self._build_course_mapping()
        self.student_mapping = self._build_student_mapping()
        
        # Từ khóa để nhận diện intent
        self.query_patterns = {
            QueryType.CURRICULUM_OVERVIEW: [
                r"lộ\s*trình\s*học",
                r"chương\s*trình\s*đào\s*tạo",
                r"khung\s*chương\s*trình",
                r"cấu\s*trúc\s*học\s*tập",
                r"toàn\s*bộ\s*môn\s*học",
                r"tổng\s*quan\s*ngành"
            ],
            QueryType.SEMESTER_DETAILS: [
                r"học\s*kỳ\s*(\d+)",
                r"kỳ\s*(\d+)",
                r"semester\s*(\d+)",
                r"năm\s*thứ\s*(\d+)",
                r"môn\s*học\s*kỳ\s*(\d+)"
            ],
            QueryType.COURSE_INFO: [
                r"môn\s*học\s*([A-Z]{3}\d{3})",
                r"môn\s*([A-Z]{3}\d{3})",
                r"thông\s*tin\s*môn",
                r"chi\s*tiết\s*môn",
                r"mô\s*tả\s*môn"
            ],
            QueryType.STUDENT_INFO: [
                r"sinh\s*viên\s*([A-Z]{2}\d+)",
                r"học\s*sinh\s*([A-Z]{2}\d+)",
                r"thông\s*tin\s*sinh\s*viên",
                r"danh\s*sách\s*sinh\s*viên"
            ],
            QueryType.PREREQUISITES: [
                r"môn\s*tiên\s*quyết",
                r"điều\s*kiện\s*tiên\s*quyết",
                r"prerequisite",
                r"yêu\s*cầu\s*trước"
            ],
            QueryType.ASSESSMENT_INFO: [
                r"đánh\s*giá",
                r"kiểm\s*tra",
                r"thi\s*cử",
                r"điểm\s*số",
                r"assessment",
                r"hình\s*thức\s*thi"
            ],
            QueryType.LEARNING_OUTCOMES: [
                r"kết\s*quả\s*học\s*tập",
                r"learning\s*outcome",
                r"CLO",
                r"mục\s*tiêu\s*học\s*tập"
            ],
            QueryType.SCHEDULE_INFO: [
                r"lịch\s*học",
                r"thời\s*khóa\s*biểu",
                r"schedule",
                r"buổi\s*học"
            ],
            QueryType.MATERIALS_INFO: [
                r"tài\s*liệu",
                r"sách\s*giáo\s*khoa",
                r"material",
                r"giáo\s*trình"
            ]
        }
    
    def _load_curriculum_data(self) -> Dict:
        """Tải dữ liệu chương trình đào tạo từ file JSON"""
        try:
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Không tìm thấy file dữ liệu: {self.data_file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Lỗi đọc file JSON: {self.data_file_path}")
            return {}
    
    def _build_course_mapping(self) -> Dict[str, Dict]:
        """Xây dựng bản đồ ánh xạ môn học để tìm kiếm nhanh"""
        mapping = {}
        syllabuses = self.curriculum_data.get("syllabuses", [])
        
        for syllabus in syllabuses:
            metadata = syllabus.get("metadata", {})
            course_id = metadata.get("course_id", "")
            course_name = metadata.get("course_name_from_curriculum", "")
            english_title = metadata.get("english_title", "")
            
            if course_id:
                mapping[course_id.upper()] = syllabus
                
                # Thêm mapping cho tên môn học (cả tiếng Việt và tiếng Anh)
                if course_name:
                    mapping[course_name.lower()] = syllabus
                if english_title:
                    mapping[english_title.lower()] = syllabus
        
        return mapping
    
    def _build_student_mapping(self) -> Dict[str, Dict]:
        """Xây dựng bản đồ ánh xạ sinh viên"""
        mapping = {}
        students = self.curriculum_data.get("students", [])
        
        for student in students:
            roll_number = student.get("RollNumber", "")
            if roll_number:
                mapping[roll_number.upper()] = student
                
        return mapping
    
    def understand_query(self, query: str) -> QueryIntent:
        """Phân tích và hiểu ý định truy vấn của người dùng"""
        query_lower = query.lower()
        intent = QueryIntent(query_type=QueryType.UNKNOWN, raw_query=query)
        
        # Tìm ngành học (mặc định là ngành trong data)
        major_code = self.curriculum_data.get("major_code_input", "")
        if major_code:
            intent.major_code = major_code
        
        # Phân tích pattern để xác định loại truy vấn
        max_confidence = 0.0
        detected_type = QueryType.UNKNOWN
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    confidence = len(match.group(0)) / len(query_lower)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        detected_type = query_type
                        
                        # Trích xuất thông tin bổ sung từ pattern
                        if query_type == QueryType.SEMESTER_DETAILS and match.groups():
                            try:
                                intent.semester = int(match.group(1))
                            except (ValueError, IndexError):
                                pass
                        elif query_type == QueryType.COURSE_INFO and match.groups():
                            intent.course_id = match.group(1).upper()
                        elif query_type == QueryType.STUDENT_INFO and match.groups():
                            intent.student_id = match.group(1).upper()
        
        intent.query_type = detected_type
        intent.confidence = max_confidence
        
        # Tìm mã môn học trong query nếu chưa có
        if not intent.course_id:
            course_pattern = r'\b([A-Z]{3}\d{3})\b'
            course_match = re.search(course_pattern, query.upper())
            if course_match:
                intent.course_id = course_match.group(1)
        
        # Tìm mã sinh viên nếu chưa có
        if not intent.student_id:
            student_pattern = r'\b([A-Z]{2}\d{6})\b'
            student_match = re.search(student_pattern, query.upper())
            if student_match:
                intent.student_id = student_match.group(1)
        
        return intent
    
    def extract_relevant_data(self, intent: QueryIntent) -> Dict:
        """Trích xuất dữ liệu phù hợp dựa trên ý định truy vấn"""
        result = {
            "query_type": intent.query_type.value,
            "major_info": {
                "major_code": self.curriculum_data.get("major_code_input", ""),
                "curriculum_title": self.curriculum_data.get("curriculum_title_on_page", ""),
                "curriculum_url": self.curriculum_data.get("curriculum_url", "")
            },
            "data": {}
        }
        
        syllabuses = self.curriculum_data.get("syllabuses", [])
        
        if intent.query_type == QueryType.CURRICULUM_OVERVIEW:
            # Trả về tổng quan chương trình đào tạo được sắp xếp theo kỳ
            result["data"] = self._get_curriculum_overview(syllabuses)
            
        elif intent.query_type == QueryType.SEMESTER_DETAILS and intent.semester:
            # Trả về chi tiết học kỳ cụ thể
            result["data"] = self._get_semester_details(syllabuses, intent.semester)
            
        elif intent.query_type == QueryType.COURSE_INFO and intent.course_id:
            # Trả về thông tin chi tiết môn học
            result["data"] = self._get_course_details(intent.course_id)
            
        elif intent.query_type == QueryType.STUDENT_INFO:
            # Trả về thông tin sinh viên
            if intent.student_id:
                result["data"] = self._get_student_details(intent.student_id)
            else:
                result["data"] = self._get_all_students()
                
        elif intent.query_type == QueryType.PREREQUISITES:
            # Trả về thông tin môn tiên quyết
            result["data"] = self._get_prerequisites_info(syllabuses, intent.course_id)
            
        elif intent.query_type == QueryType.ASSESSMENT_INFO:
            # Trả về thông tin đánh giá
            result["data"] = self._get_assessment_info(syllabuses, intent.course_id)
            
        elif intent.query_type == QueryType.LEARNING_OUTCOMES:
            # Trả về kết quả học tập
            result["data"] = self._get_learning_outcomes(syllabuses, intent.course_id)
            
        elif intent.query_type == QueryType.SCHEDULE_INFO:
            # Trả về thông tin lịch học
            result["data"] = self._get_schedule_info(syllabuses, intent.course_id)
            
        elif intent.query_type == QueryType.MATERIALS_INFO:
            # Trả về thông tin tài liệu
            result["data"] = self._get_materials_info(syllabuses, intent.course_id)
        
        return result
    
    def _get_curriculum_overview(self, syllabuses: List[Dict]) -> Dict:
        """Tạo tổng quan chương trình đào tạo theo kỳ"""
        curriculum_by_semester = {}
        
        for syllabus in syllabuses:
            metadata = syllabus.get("metadata", {})
            semester = metadata.get("semester_from_curriculum", 0)
            
            if semester not in curriculum_by_semester:
                curriculum_by_semester[semester] = []
            
            course_info = {
                "course_id": metadata.get("course_id", ""),
                "course_name": metadata.get("course_name_from_curriculum", ""),
                "english_title": metadata.get("english_title", ""),
                "credits": metadata.get("credits", ""),
                "prerequisites": metadata.get("prerequisites", ""),
                "course_type": metadata.get("course_type_guess", "")
            }
            curriculum_by_semester[semester].append(course_info)
        
        # Sắp xếp theo kỳ
        sorted_curriculum = {}
        for semester in sorted(curriculum_by_semester.keys()):
            sorted_curriculum[f"Học kỳ {semester}"] = curriculum_by_semester[semester]
        
        return sorted_curriculum
    
    def _get_semester_details(self, syllabuses: List[Dict], target_semester: int) -> Dict:
        """Lấy chi tiết môn học của một kỳ cụ thể"""
        semester_courses = []
        
        for syllabus in syllabuses:
            metadata = syllabus.get("metadata", {})
            semester = metadata.get("semester_from_curriculum", 0)
            
            if semester == target_semester:
                course_details = {
                    "course_id": metadata.get("course_id", ""),
                    "course_name": metadata.get("course_name_from_curriculum", ""),
                    "english_title": metadata.get("english_title", ""),
                    "credits": metadata.get("credits", ""),
                    "prerequisites": metadata.get("prerequisites", ""),
                    "description": metadata.get("description", ""),
                    "course_type": metadata.get("course_type_guess", ""),
                    "assessments_count": len(syllabus.get("assessments", [])),
                    "materials_count": len(syllabus.get("materials", []))
                }
                semester_courses.append(course_details)
        
        return {
            "semester": target_semester,
            "total_courses": len(semester_courses),
            "courses": semester_courses
        }
    
    def _get_course_details(self, course_id: str) -> Dict:
        """Lấy thông tin chi tiết của môn học"""
        if course_id.upper() in self.course_mapping:
            return self.course_mapping[course_id.upper()]
        return {"error": f"Không tìm thấy môn học {course_id}"}
    
    def _get_student_details(self, student_id: str) -> Dict:
        """Lấy thông tin chi tiết của sinh viên"""
        if student_id.upper() in self.student_mapping:
            return self.student_mapping[student_id.upper()]
        return {"error": f"Không tìm thấy sinh viên {student_id}"}
    
    def _get_all_students(self) -> Dict:
        """Lấy danh sách tất cả sinh viên"""
        return {
            "total_students": len(self.curriculum_data.get("students", [])),
            "students": self.curriculum_data.get("students", [])
        }
    
    def _get_prerequisites_info(self, syllabuses: List[Dict], course_id: Optional[str] = None) -> Dict:
        """Lấy thông tin môn tiên quyết"""
        if course_id:
            course_data = self._get_course_details(course_id)
            if "error" not in course_data:
                prerequisites = course_data.get("metadata", {}).get("prerequisites", "")
                return {"course_id": course_id, "prerequisites": prerequisites}
            return course_data
        else:
            # Trả về tất cả các môn có môn tiên quyết
            prerequisites_info = []
            for syllabus in syllabuses:
                metadata = syllabus.get("metadata", {})
                prerequisites = metadata.get("prerequisites", "")
                if prerequisites and prerequisites.strip():
                    prerequisites_info.append({
                        "course_id": metadata.get("course_id", ""),
                        "course_name": metadata.get("course_name_from_curriculum", ""),
                        "prerequisites": prerequisites
                    })
            return {"courses_with_prerequisites": prerequisites_info}
    
    def _get_assessment_info(self, syllabuses: List[Dict], course_id: Optional[str] = None) -> Dict:
        """Lấy thông tin đánh giá"""
        if course_id:
            course_data = self._get_course_details(course_id)
            if "error" not in course_data:
                return {
                    "course_id": course_id,
                    "assessments": course_data.get("assessments", [])
                }
            return course_data
        else:
            # Tổng hợp thông tin đánh giá của tất cả môn
            all_assessments = {}
            for syllabus in syllabuses:
                metadata = syllabus.get("metadata", {})
                course_id = metadata.get("course_id", "")
                assessments = syllabus.get("assessments", [])
                if assessments:
                    all_assessments[course_id] = assessments
            return {"all_course_assessments": all_assessments}
    
    def _get_learning_outcomes(self, syllabuses: List[Dict], course_id: Optional[str] = None) -> Dict:
        """Lấy thông tin kết quả học tập"""
        if course_id:
            course_data = self._get_course_details(course_id)
            if "error" not in course_data:
                return {
                    "course_id": course_id,
                    "learning_outcomes": course_data.get("learning_outcomes", [])
                }
            return course_data
        else:
            # Tổng hợp kết quả học tập của tất cả môn
            all_outcomes = {}
            for syllabus in syllabuses:
                metadata = syllabus.get("metadata", {})
                course_id = metadata.get("course_id", "")
                outcomes = syllabus.get("learning_outcomes", [])
                if outcomes:
                    all_outcomes[course_id] = outcomes
            return {"all_course_outcomes": all_outcomes}
    
    def _get_schedule_info(self, syllabuses: List[Dict], course_id: Optional[str] = None) -> Dict:
        """Lấy thông tin lịch học"""
        if course_id:
            course_data = self._get_course_details(course_id)
            if "error" not in course_data:
                return {
                    "course_id": course_id,
                    "schedule": course_data.get("schedule", [])
                }
            return course_data
        else:
            # Tổng hợp lịch học của tất cả môn
            all_schedules = {}
            for syllabus in syllabuses:
                metadata = syllabus.get("metadata", {})
                course_id = metadata.get("course_id", "")
                schedule = syllabus.get("schedule", [])
                if schedule:
                    all_schedules[course_id] = schedule
            return {"all_course_schedules": all_schedules}
    
    def _get_materials_info(self, syllabuses: List[Dict], course_id: Optional[str] = None) -> Dict:
        """Lấy thông tin tài liệu"""
        if course_id:
            course_data = self._get_course_details(course_id)
            if "error" not in course_data:
                return {
                    "course_id": course_id,
                    "materials": course_data.get("materials", [])
                }
            return course_data
        else:
            # Tổng hợp tài liệu của tất cả môn
            all_materials = {}
            for syllabus in syllabuses:
                metadata = syllabus.get("metadata", {})
                course_id = metadata.get("course_id", "")
                materials = syllabus.get("materials", [])
                if materials:
                    all_materials[course_id] = materials
            return {"all_course_materials": all_materials}
    
    def process_query(self, query: str) -> Tuple[QueryIntent, Dict]:
        """Xử lý truy vấn hoàn chỉnh và trả về kết quả"""
        intent = self.understand_query(query)
        relevant_data = self.extract_relevant_data(intent)
        return intent, relevant_data
    
    def format_for_llm(self, intent: QueryIntent, relevant_data: Dict) -> str:
        """Định dạng dữ liệu phù hợp để đưa vào LLM"""
        formatted_context = f"""
THÔNG TIN TRUY VẤN:
- Truy vấn gốc: {intent.raw_query}
- Loại truy vấn: {intent.query_type.value}
- Ngành học: {intent.major_code or 'Không xác định'}
- Độ tin cậy: {intent.confidence:.2f}

THÔNG TIN CHƯƠNG TRÌNH ĐÀO TẠO:
- Mã ngành: {relevant_data['major_info']['major_code']}
- Tên chương trình: {relevant_data['major_info']['curriculum_title']}
- URL chương trình: {relevant_data['major_info']['curriculum_url']}

DỮ LIỆU LIÊN QUAN:
{json.dumps(relevant_data['data'], ensure_ascii=False, indent=2)}
"""
        return formatted_context

# Test function
def test_query_processor():
    """Hàm test cho query processor"""
    processor = CurriculumQueryProcessor("Data/reduced_data.json")
    
    test_queries = [
        "Cho tôi xem lộ trình học ngành AI",
        "Môn học kỳ 1 có gì?",
        "Thông tin về môn CSI106",
        "Danh sách sinh viên ngành AI",
        "Môn tiên quyết của MAD101 là gì?",
        "Hình thức đánh giá của môn MAE101"
    ]
    
    for query in test_queries:
        print(f"\nTEST QUERY: {query}")
        print("-" * 50)
        intent, data = processor.process_query(query)
        print(f"Intent: {intent.query_type.value}, Confidence: {intent.confidence:.2f}")
        if intent.course_id:
            print(f"Course ID: {intent.course_id}")
        if intent.semester:
            print(f"Semester: {intent.semester}")
        print(f"Data keys: {list(data['data'].keys())}")

if __name__ == "__main__":
    test_query_processor() 