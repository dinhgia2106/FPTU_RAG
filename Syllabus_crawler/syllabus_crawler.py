#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPT FLM Syllabus Crawler - Final Version
Tác giả: Trần Đình Gia

Script này tự động crawl dữ liệu syllabus từ hệ thống FLM của FPT University.
"""

import os
import json
import time
import random
import sys
from datetime import datetime
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Tải biến môi trường từ file .env
load_dotenv()

# Cấu hình từ file .env hoặc giá trị mặc định
FPT_EMAIL = os.getenv('FPT_EMAIL', '')
FPT_PASSWORD = os.getenv('FPT_PASSWORD', '')
BACKUP_CODES = os.getenv('BACKUP_CODES', '').split(',')
SUBJECT_CODES = os.getenv('SUBJECT_CODES', '').split(',')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'fpt_syllabus_data_appended.json')
HEADLESS = os.getenv('HEADLESS', 'false').lower() == 'true'

class FLMSyllabusCrawler:
    def __init__(self, headless=False):
        """Khởi tạo crawler với các tham số cấu hình."""
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.logged_in = False
        self.data = {}
        self.load_existing_data()

    def load_existing_data(self):
        """Tải dữ liệu hiện có từ file JSON nếu tồn tại."""
        try:
            if os.path.exists(OUTPUT_FILE):
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"Đã tải dữ liệu hiện có từ {OUTPUT_FILE} với {len(self.data)} môn học.")
            else:
                self.data = {}
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu hiện có: {e}")
            self.data = {}

    def save_data(self):
        """Lưu dữ liệu đã crawl vào file JSON."""
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            print(f"Đã lưu dữ liệu vào {OUTPUT_FILE} với {len(self.data)} môn học.")
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu: {e}")

    def start(self):
        """Khởi động trình duyệt và mở trang đăng nhập FLM."""
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context()
            self.page = self.context.new_page()
            self.page.goto("https://flm.fpt.edu.vn/DefaultSignin")
            print("Đã mở trang đăng nhập FLM.")
            return True
        except Exception as e:
            print(f"Lỗi khi khởi động trình duyệt: {e}")
            self.close()
            return False

    def login(self, email, password, backup_codes=None):
        """Cho phép người dùng tự đăng nhập vào hệ thống FLM."""
        try:
            # Chọn Education Level 'FPTU'
            print("Chọn Education Level 'FPTU'...")
            self.page.select_option("select", "fptu")
            
            # Hướng dẫn người dùng tự đăng nhập
            print("\n" + "="*60)
            print("YÊU CẦU ĐĂNG NHẬP THỦ CÔNG")
            print("-"*60)
            print("1. Trên cửa sổ trình duyệt đã mở, hãy click vào nút đăng nhập Google")
            print("2. Đăng nhập bằng tài khoản FPT của bạn (email và mật khẩu)")
            print("3. Hoàn thành xác thực 2FA nếu được yêu cầu")
            print("4. Đợi cho đến khi trang web FLM hiển thị đầy đủ")
            print("5. Nhấn Enter sau khi đã đăng nhập thành công")
            print("="*60)
            
            # Đợi người dùng đăng nhập thủ công
            input("\nNhấn Enter sau khi bạn đã đăng nhập thành công: ")
            
            # Đợi cho trang ổn định sau đăng nhập
            print("Đợi cho trang ổn định sau đăng nhập...")
            self.page.wait_for_load_state("networkidle", timeout=10000)
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            
            # Kiểm tra trạng thái đăng nhập với xử lý lỗi cẩn thận
            print(f"URL hiện tại sau khi đăng nhập: {self.page.url}")
            
            # Kiểm tra URL là cách an toàn nhất
            if "flm.fpt.edu.vn/gui" in self.page.url:
                print("Đăng nhập thủ công thành công (dựa trên URL)!")
                self.logged_in = True
                return True
            
            # Nếu URL check không thành công, thử các cách khác an toàn hơn
            try:
                # Đợi thêm để đảm bảo trang đã ổn định
                self.page.wait_for_timeout(2000)
                
                # Thử tải lại trang chủ FLM
                try:
                    self.page.goto("https://flm.fpt.edu.vn/gui/Home", timeout=10000)
                    print("Đã điều hướng đến trang chủ FLM để xác minh đăng nhập")
                    self.page.wait_for_load_state("networkidle", timeout=5000)
                except Exception as e:
                    print(f"Không thể điều hướng đến trang chủ FLM: {e}")
                
                # Kiểm tra URL sau khi tải trang chủ - nếu vẫn ở FLM, có lẽ đã đăng nhập thành công
                if "flm.fpt.edu.vn" in self.page.url and "DefaultSignin" not in self.page.url:
                    print("Đăng nhập thủ công thành công (dựa trên điều hướng trang chủ)!")
                    self.logged_in = True
                    return True
                    
                # Hỏi người dùng xác nhận
                print("\n" + "="*60)
                print("KIỂM TRA TRẠNG THÁI ĐĂNG NHẬP")
                print("-"*60)
                print("Không thể tự động xác định bạn đã đăng nhập thành công chưa.")
                print(f"URL hiện tại: {self.page.url}")
                print("Vui lòng xác nhận:")
                confirm = input("Bạn đã đăng nhập thành công chưa? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    print("Người dùng xác nhận đã đăng nhập thành công.")
                    self.logged_in = True
                    return True
                else:
                    print("Người dùng xác nhận chưa đăng nhập thành công.")
                    return False
                    
            except Exception as e:
                print(f"Lỗi khi kiểm tra trạng thái đăng nhập: {e}")
                
                # Nếu có lỗi kiểm tra, vẫn hỏi người dùng
                print("\n" + "="*60)
                print("LỖI KHI KIỂM TRA TRẠNG THÁI ĐĂNG NHẬP")
                print("-"*60) 
                print(f"Lỗi: {e}")
                print("Vui lòng xác nhận thủ công:")
                confirm = input("Bạn đã đăng nhập thành công chưa? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    print("Người dùng xác nhận đã đăng nhập thành công dù có lỗi kiểm tra.")
                    self.logged_in = True
                    return True
                else:
                    print("Người dùng xác nhận chưa đăng nhập thành công.")
                    return False
                    
        except Exception as e:
            print(f"Lỗi trong quá trình đăng nhập thủ công: {e}")
            return False

    def navigate_to_syllabus_management(self):
        """Điều hướng đến trang quản lý syllabus."""
        try:
            # Kiểm tra đã đăng nhập chưa
            if not self.logged_in:
                print("Chưa đăng nhập, không thể điều hướng đến trang quản lý syllabus.")
                return False
            
            # Điều hướng đến trang chủ trước
            self.page.goto("https://flm.fpt.edu.vn/gui/Home")
            
            # Click vào liên kết "View Syllabus"
            print("Điều hướng đến trang quản lý syllabus...")
            self.page.click("a:text('View Syllabus')")
            
            # Đợi trang tải xong
            self.page.wait_for_selector("input[type='text']", timeout=5000)
            print("Đã vào trang quản lý syllabus.")
            return True
        except Exception as e:
            print(f"Lỗi khi điều hướng đến trang quản lý syllabus: {e}")
            return False

    def search_subject(self, subject_code):
        """Tìm kiếm môn học theo mã môn."""
        try:
            # Chọn tìm kiếm theo mã môn
            print(f"Tìm kiếm môn học: {subject_code}")
            self.page.select_option("select:near(:text('Search by subject'))", "Code")
            
            # Nhập mã môn và tìm kiếm
            self.page.fill("input[type='text']", subject_code)
            self.page.click("input[value='Search']")
            
            # Đợi kết quả tìm kiếm
            self.page.wait_for_load_state("networkidle", timeout=5000)
            
            # Kiểm tra có kết quả không
            no_results = self.page.query_selector("text=No syllabus(es) found")
            if no_results:
                print(f"Không tìm thấy môn học: {subject_code}")
                return False
            
            # Kiểm tra có kết quả phù hợp không
            results = self.page.query_selector_all("table tbody tr")
            if len(results) == 0:
                print(f"Không tìm thấy kết quả cho môn học: {subject_code}")
                return False
            
            print(f"Tìm thấy {len(results)} kết quả cho môn học: {subject_code}")
            return True
        except Exception as e:
            print(f"Lỗi khi tìm kiếm môn học {subject_code}: {e}")
            return False

    def extract_syllabus_data(self, subject_code):
        """Trích xuất dữ liệu syllabus cho môn học đã tìm thấy."""
        try:
            # Kiểm tra nếu môn học đã có trong dữ liệu
            if subject_code in self.data:
                print(f"Môn học {subject_code} đã tồn tại trong dữ liệu. Bỏ qua.")
                return True
            
            # Click vào liên kết syllabus đầu tiên
            print(f"Truy cập chi tiết syllabus cho môn học: {subject_code}")
            syllabus_links = self.page.query_selector_all("table tbody tr td:nth-child(4) a")
            if len(syllabus_links) == 0:
                print(f"Không tìm thấy liên kết syllabus cho môn học: {subject_code}")
                return False
            
            # Click vào liên kết đầu tiên
            syllabus_links[0].click()
            
            # Đợi trang chi tiết tải xong
            self.page.wait_for_selector("table", timeout=10000)
            
            # Trích xuất dữ liệu bằng JavaScript
            print("Đang trích xuất dữ liệu syllabus...")
            result = self.page.evaluate("""() => {
                const data = {
                    page_title: document.title,
                    page_url: window.location.href,
                    subject_code: null,
                    syllabus_id: null,
                    general_details: {},
                    materials_table: [],
                    clos: [],
                    clo_plo_mapping_link: null,
                    sessions: [],
                    assessments: [],
                    extraction_errors: [],
                    extraction_time: new Date().toISOString()
                };

                const getText = (element, selector) => {
                    if (!element) return null;
                    const el = selector ? element.querySelector(selector) : element;
                    return el ? el.innerText.trim() : null;
                };

                const getCleanedTextFromCell = (cellElement) => {
                    if (!cellElement) return null;
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = cellElement.innerHTML.replace(/<br\\s*\\/?>/gi, '\\n');
                    return (tempDiv.textContent || tempDiv.innerText || "").trim();
                };

                const mainContent = document.querySelector('.content.container-fluid') || document.querySelector('form .container-fluid .row .col-md-12') || document.body;
                const allTables = Array.from(mainContent.querySelectorAll('table'));
                let generalDetailsTable, materialsTable, cloTable, sessionsTable, assessmentTable;

                // Actual headers based on analysis
                const actualHeaders = {
                    materials: ["MaterialDescription", "Author", "Publisher", "PublishedDate", "Edition", "ISBN", "IsMainMaterial", "IsHardCopy", "IsOnline", "Note"],
                    clo: ["CLO Name", "CLO Details", "LO Details"],
                    sessions: ["Session", "Topic", "Learning-Teaching Type", "LO", "ITU", "Student Materials", "S-Download", "Student's Tasks", "URLs"],
                    assessment: ["Category", "Type", "Part", "Weight", "Completion Criteria", "Duration", "CLO", "Question Type", "No Question", "Knowledge and Skill", "Grading Guide", "Note"]
                };

                const checkTableHeadersStrict = (table, expectedHeaders) => {
                    if (!table) return false;
                    const currentHeaders = Array.from(table.querySelectorAll("thead tr th, tbody tr:first-child th, tbody tr:first-child td")).map(h => getText(h));
                    if (currentHeaders.length < expectedHeaders.length) return false;
                    let matchCount = 0;
                    for (const expected of expectedHeaders) {
                        if (currentHeaders.some(current => current && current.toLowerCase().includes(expected.toLowerCase()))) {
                            matchCount++;
                        }
                    }
                    return matchCount >= Math.min(expectedHeaders.length, 3);
                };

                // Identify tables based on content and headers
                allTables.forEach(table => {
                    const firstRowFirstCellText = getText(table, 'tbody tr:first-child td:first-child');
                    if (!generalDetailsTable && firstRowFirstCellText && firstRowFirstCellText.toLowerCase().includes('syllabus id')) {
                        generalDetailsTable = table;
                        return; 
                    }
                    if (!materialsTable && checkTableHeadersStrict(table, actualHeaders.materials)) {
                        materialsTable = table;
                        return;
                    }
                    if (!cloTable && checkTableHeadersStrict(table, actualHeaders.clo)) {
                        cloTable = table;
                        return;
                    }
                    if (!sessionsTable && checkTableHeadersStrict(table, actualHeaders.sessions)) {
                        sessionsTable = table;
                        return;
                    }
                    if (!assessmentTable && checkTableHeadersStrict(table, actualHeaders.assessment)) {
                        assessmentTable = table;
                        return;
                    }
                });

                // Extract General Details
                if (generalDetailsTable) {
                    const rows = generalDetailsTable.querySelectorAll("tbody > tr");
                    rows.forEach(row => {
                        const cells = row.querySelectorAll("td");
                        if (cells.length >= 2) {
                            let key = getText(cells[0]);
                            if (key) key = key.replace(/:$/, "").trim();
                            const value = getCleanedTextFromCell(cells[1]);
                            if (key && value !== null) {
                                data.general_details[key] = value;
                                if (key.toLowerCase() === 'subject code') data.subject_code = value;
                                if (key.toLowerCase() === 'syllabus id') data.syllabus_id = value;
                            }
                        }
                    });
                } else {
                    data.extraction_errors.push("General details table (containing 'Syllabus ID') not found.");
                }
                
                // Extract Materials Table
                if (materialsTable) {
                    const rows = materialsTable.querySelectorAll("tbody > tr");
                    if (rows.length > 0) {
                        const headers = Array.from(rows[0].querySelectorAll("th, td")).map(th => getText(th));
                        for (let i = 1; i < rows.length; i++) {
                            const cells = rows[i].querySelectorAll("td");
                            if (cells.length === 0) continue;
                            const materialEntry = {};
                            headers.forEach((header, index) => {
                                if (header && cells[index]) {
                                    materialEntry[header] = getCleanedTextFromCell(cells[index]);
                                }
                            });
                            if (Object.keys(materialEntry).length > 0) data.materials_table.push(materialEntry);
                        }
                    }
                } else {
                    data.extraction_errors.push("Materials table not found based on headers.");
                }

                // Extract CLOs
                if (cloTable) {
                    const rows = cloTable.querySelectorAll("tbody > tr");
                    if (rows.length > 0) {
                        const headers = Array.from(rows[0].querySelectorAll("th, td")).map(th => getText(th));
                        for (let i = 1; i < rows.length; i++) {
                            const cells = rows[i].querySelectorAll("td");
                            if (cells.length === 0) continue;
                            const cloEntry = {};
                            headers.forEach((header, index) => {
                                if (header && cells[index]) {
                                    cloEntry[header] = getCleanedTextFromCell(cells[index]);
                                }
                            });
                            if (Object.keys(cloEntry).length > 0) data.clos.push(cloEntry);
                        }
                    }
                } else {
                    data.extraction_errors.push("CLO table not found based on headers.");
                }

                const cloPloLink = Array.from(mainContent.querySelectorAll("a")).find(a => getText(a) && getText(a).toLowerCase().includes("view mapping of clos to plos"));
                if (cloPloLink) {
                    data.clo_plo_mapping_link = cloPloLink.href;
                }

                // Extract Sessions
                if (sessionsTable) {
                    const rows = sessionsTable.querySelectorAll("tbody > tr");
                    if (rows.length > 0) {
                        const headers = Array.from(rows[0].querySelectorAll("th, td")).map(th => getText(th));
                        for (let i = 1; i < rows.length; i++) {
                            const cells = rows[i].querySelectorAll("td");
                            if (cells.length === 0) continue;
                            const sessionEntry = {};
                            headers.forEach((header, index) => {
                                if (header && cells[index]) {
                                    if (header.toLowerCase().includes("download") && cells[index].querySelector("a")) {
                                        sessionEntry[header] = {
                                            text: getText(cells[index].querySelector("a")) || "Download",
                                            link: cells[index].querySelector("a").href
                                        };
                                    } else {
                                        sessionEntry[header] = getCleanedTextFromCell(cells[index]);
                                    }
                                }
                            });
                            if (Object.keys(sessionEntry).length > 0) data.sessions.push(sessionEntry);
                        }
                    }
                } else {
                    data.extraction_errors.push("Sessions table not found based on headers.");
                }

                // Extract Assessments
                if (assessmentTable) {
                    const rows = assessmentTable.querySelectorAll("tbody > tr");
                    if (rows.length > 0) {
                        const headers = Array.from(rows[0].querySelectorAll("th, td")).map(th => getText(th));
                        for (let i = 1; i < rows.length; i++) {
                            const cells = rows[i].querySelectorAll("td");
                            if (cells.length === 0) continue;
                            const assessmentEntry = {};
                            headers.forEach((header, index) => {
                                if (header && cells[index]) {
                                    assessmentEntry[header] = getCleanedTextFromCell(cells[index]);
                                }
                            });
                            if (Object.keys(assessmentEntry).length > 0) data.assessments.push(assessmentEntry);
                        }
                    }
                } else {
                    data.extraction_errors.push("Assessment table not found based on headers.");
                }
                
                // Materials info (text like "10 material(s)")
                const allTextNodes = Array.from(mainContent.querySelectorAll("p, span, div, b, strong"));
                const materialRegex = /(\\d+\\s*material\\(s\\))/i;
                const materialElement = allTextNodes.find(el => materialRegex.test(getText(el)));
                if (materialElement) {
                    data.materials_info = getText(materialElement).match(materialRegex)[0];
                }

                return data;
            }""")
            
            # Lưu dữ liệu vào dictionary
            if result and result.get('subject_code'):
                self.data[subject_code] = result
                print(f"Đã trích xuất thành công dữ liệu cho môn học: {subject_code}")
                return True
            else:
                print(f"Không thể trích xuất dữ liệu cho môn học: {subject_code}")
                return False
        except Exception as e:
            print(f"Lỗi khi trích xuất dữ liệu syllabus cho môn học {subject_code}: {e}")
            return False

    def crawl_subjects(self, subject_codes):
        """Crawl dữ liệu cho danh sách các môn học."""
        if not self.navigate_to_syllabus_management():
            print("Không thể điều hướng đến trang quản lý syllabus. Dừng crawl.")
            return False
        
        success_count = 0
        for subject_code in subject_codes:
            subject_code = subject_code.strip()
            if not subject_code:
                continue
                
            print(f"Bắt đầu crawl môn học: {subject_code}")
            
            # Kiểm tra nếu môn học đã có trong dữ liệu
            if subject_code in self.data:
                print(f"Môn học {subject_code} đã tồn tại trong dữ liệu. Bỏ qua.")
                success_count += 1
                continue
            
            # Tìm kiếm môn học
            if self.search_subject(subject_code):
                # Trích xuất dữ liệu
                if self.extract_syllabus_data(subject_code):
                    success_count += 1
                    # Lưu dữ liệu sau mỗi môn học thành công
                    self.save_data()
                    
                    # Quay lại trang quản lý syllabus để tìm môn tiếp theo
                    self.navigate_to_syllabus_management()
                else:
                    print(f"Không thể trích xuất dữ liệu cho môn học: {subject_code}")
            else:
                print(f"Không tìm thấy môn học: {subject_code}")
            
            # Tạm dừng ngẫu nhiên để tránh bị chặn
            delay = random.uniform(1.0, 3.0)
            print(f"Tạm dừng {delay:.2f} giây trước khi tiếp tục...")
            time.sleep(delay)
        
        print(f"Đã crawl thành công {success_count}/{len(subject_codes)} môn học.")
        return success_count > 0

    def close(self):
        """Đóng trình duyệt và giải phóng tài nguyên."""
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            print("Đã đóng trình duyệt và giải phóng tài nguyên.")
        except Exception as e:
            print(f"Lỗi khi đóng trình duyệt: {e}")

def main():
    """Hàm chính để chạy crawler."""
    print("\n" + "="*50)
    print("FPT FLM SYLLABUS CRAWLER - FINAL VERSION")
    print("="*50 + "\n")
    
    if not SUBJECT_CODES or not SUBJECT_CODES[0]:
        print("Lỗi: Thiếu danh sách mã môn học trong file .env")
        return
    
    # Hiển thị thông tin cấu hình
    print(f"Email: {FPT_EMAIL}")
    print(f"Backup codes: {len(BACKUP_CODES)} mã")
    print(f"Môn học cần crawl: {', '.join(SUBJECT_CODES)}")
    print(f"File lưu dữ liệu: {OUTPUT_FILE}")
    print(f"Chế độ headless: {HEADLESS}")
    print("\n" + "-"*50 + "\n")
    
    # Khởi tạo và chạy crawler
    crawler = FLMSyllabusCrawler(headless=HEADLESS)
    try:
        if crawler.start():
            if crawler.login(FPT_EMAIL, FPT_PASSWORD, BACKUP_CODES):
                # Call crawl_subjects with the subject codes
                if crawler.crawl_subjects(SUBJECT_CODES):
                    print("Crawl hoàn tất thành công!")
                else:
                    print("Crawl không thành công hoặc không tìm thấy môn học nào.")
            else:
                print("Lỗi: Không thể đăng nhập vào hệ thống FLM.")
    except KeyboardInterrupt:
        print("\nĐã dừng crawler theo yêu cầu của người dùng.")
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")
    finally:
        crawler.save_data()
        crawler.close()
        print("\n" + "="*50)
        print("CRAWLER ĐÃ HOÀN THÀNH")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()