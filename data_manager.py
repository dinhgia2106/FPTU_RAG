import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
from datetime import datetime
import os

class JSONManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QUAN LY DU LIEU CHUONG TRINH HOC AI")
        self.root.geometry("1400x800")
        
        # Du lieu
        self.data = None
        self.syllabuses = []
        self.current_file = "Data/combined_data.json"
        
        # Tao giao dien
        self.setup_ui()
        
        # Tai du lieu mac dinh
        self.load_data()
    
    def setup_ui(self):
        """Thiet lap giao dien nguoi dung"""
        
        # Menu bar
        self.create_menu()
        
        # Toolbar
        self.create_toolbar()
        
        # Main content area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Danh sach mon hoc
        left_frame = ttk.LabelFrame(main_frame, text="DANH SACH MON HOC", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        left_frame.configure(width=400)
        
        self.create_course_list(left_frame)
        
        # Right panel - Chi tiet mon hoc
        right_frame = ttk.LabelFrame(main_frame, text="CHI TIET MON HOC", padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_detail_tabs(right_frame)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("San sang")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_menu(self):
        """Tao menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Mo file...", command=self.open_file)
        file_menu.add_command(label="Luu", command=self.save_data)
        file_menu.add_command(label="Luu thanh...", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Thoat", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Chinh sua", menu=edit_menu)
        edit_menu.add_command(label="Them mon hoc", command=self.add_course)
        edit_menu.add_command(label="Xoa mon hoc", command=self.delete_course)
        edit_menu.add_command(label="Tim kiem", command=self.focus_search)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Xem", menu=view_menu)
        view_menu.add_command(label="Thong ke", command=self.show_statistics)
        view_menu.add_command(label="Lam moi", command=self.refresh_data)
    
    def create_toolbar(self):
        """Tao thanh cong cu"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        ttk.Button(toolbar, text="Mo file", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Luu", command=self.save_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Them", command=self.add_course).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Xoa", command=self.delete_course).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Thong ke", command=self.show_statistics).pack(side=tk.LEFT, padx=2)
        
        # Tim kiem
        ttk.Label(toolbar, text="Tim kiem:").pack(side=tk.LEFT, padx=(20, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search_change)
        search_entry = ttk.Entry(toolbar, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=2)
        self.search_entry = search_entry
    
    def create_course_list(self, parent):
        """Tao danh sach mon hoc"""
        
        # Treeview cho danh sach mon hoc
        columns = ("Ma mon", "Ten mon", "Hoc ky", "Tin chi")
        self.course_tree = ttk.Treeview(parent, columns=columns, show="headings", height=25)
        
        # Cau hinh cot
        self.course_tree.heading("Ma mon", text="Ma mon")
        self.course_tree.heading("Ten mon", text="Ten mon")
        self.course_tree.heading("Hoc ky", text="Hoc ky")
        self.course_tree.heading("Tin chi", text="Tin chi")
        
        self.course_tree.column("Ma mon", width=80)
        self.course_tree.column("Ten mon", width=200)
        self.course_tree.column("Hoc ky", width=60)
        self.course_tree.column("Tin chi", width=60)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.course_tree.yview)
        self.course_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.course_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind events
        self.course_tree.bind("<<TreeviewSelect>>", self.on_course_select)
        self.course_tree.bind("<Double-1>", self.on_course_double_click)
    
    def create_detail_tabs(self, parent):
        """Tao cac tab chi tiet"""
        
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Thong tin co ban
        self.create_basic_info_tab()
        
        # Tab 2: Tai lieu
        self.create_materials_tab()
        
        # Tab 3: Danh gia
        self.create_assessments_tab()
        
        # Tab 4: Lich trinh
        self.create_schedule_tab()
        
        # Tab 5: Muc tieu hoc tap
        self.create_outcomes_tab()
    
    def create_basic_info_tab(self):
        """Tab thong tin co ban"""
        basic_frame = ttk.Frame(self.notebook)
        self.notebook.add(basic_frame, text="Thong tin co ban")
        
        # Scroll frame
        canvas = tk.Canvas(basic_frame)
        scrollbar = ttk.Scrollbar(basic_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Cac truong thong tin
        self.basic_fields = {}
        fields = [
            ("Ma mon:", "subject_code_on_page"),
            ("Ten mon:", "course_name_from_curriculum"),
            ("Ten tieng Anh:", "english_title"),
            ("Hoc ky:", "semester_from_curriculum"),
            ("Tin chi:", "credits"),
            ("Mon tien quyet:", "prerequisites"),
            ("Cap do:", "degree_level"),
            ("Diem toi thieu:", "min_avg_mark_to_pass"),
            ("Ngay phe duyet:", "approved_date"),
            ("So quyet dinh:", "decision_no")
        ]
        
        row = 0
        for label, field in fields:
            ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            entry = ttk.Entry(scrollable_frame, width=50)
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            self.basic_fields[field] = entry
            row += 1
        
        # Mo ta
        ttk.Label(scrollable_frame, text="Mo ta:").grid(row=row, column=0, sticky="nw", padx=5, pady=2)
        self.description_text = scrolledtext.ScrolledText(scrollable_frame, width=60, height=8)
        self.description_text.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        row += 1
        
        # Nhiem vu sinh vien
        ttk.Label(scrollable_frame, text="Nhiem vu SV:").grid(row=row, column=0, sticky="nw", padx=5, pady=2)
        self.tasks_text = scrolledtext.ScrolledText(scrollable_frame, width=60, height=6)
        self.tasks_text.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        
        scrollable_frame.columnconfigure(1, weight=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_materials_tab(self):
        """Tab tai lieu"""
        materials_frame = ttk.Frame(self.notebook)
        self.notebook.add(materials_frame, text="Tai lieu")
        
        # Toolbar cho materials
        mat_toolbar = ttk.Frame(materials_frame)
        mat_toolbar.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        ttk.Button(mat_toolbar, text="Them tai lieu", command=self.add_material).pack(side=tk.LEFT, padx=2)
        ttk.Button(mat_toolbar, text="Xoa tai lieu", command=self.delete_material).pack(side=tk.LEFT, padx=2)
        
        # Treeview cho tai lieu
        mat_columns = ("STT", "Ten tai lieu", "Tac gia", "NXB", "Nam", "Chinh")
        self.materials_tree = ttk.Treeview(materials_frame, columns=mat_columns, show="headings")
        
        for col in mat_columns:
            self.materials_tree.heading(col, text=col)
            self.materials_tree.column(col, width=100)
        
        self.materials_tree.pack(fill=tk.BOTH, expand=True)
    
    def create_assessments_tab(self):
        """Tab danh gia"""
        assess_frame = ttk.Frame(self.notebook)
        self.notebook.add(assess_frame, text="Danh gia")
        
        # Toolbar
        assess_toolbar = ttk.Frame(assess_frame)
        assess_toolbar.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        ttk.Button(assess_toolbar, text="Them danh gia", command=self.add_assessment).pack(side=tk.LEFT, padx=2)
        ttk.Button(assess_toolbar, text="Xoa danh gia", command=self.delete_assessment).pack(side=tk.LEFT, padx=2)
        
        # Treeview
        assess_columns = ("STT", "Loai", "Phan", "Trong so", "Thoi gian", "Ghi chu")
        self.assessments_tree = ttk.Treeview(assess_frame, columns=assess_columns, show="headings")
        
        for col in assess_columns:
            self.assessments_tree.heading(col, text=col)
            self.assessments_tree.column(col, width=100)
        
        self.assessments_tree.pack(fill=tk.BOTH, expand=True)
    
    def create_schedule_tab(self):
        """Tab lich trinh"""
        schedule_frame = ttk.Frame(self.notebook)
        self.notebook.add(schedule_frame, text="Lich trinh")
        
        # Treeview cho lich trinh
        sch_columns = ("STT", "Buoi hoc", "Chu de", "Loai", "Muc tieu", "Tai lieu")
        self.schedule_tree = ttk.Treeview(schedule_frame, columns=sch_columns, show="headings")
        
        for col in sch_columns:
            self.schedule_tree.heading(col, text=col)
            self.schedule_tree.column(col, width=120)
        
        self.schedule_tree.pack(fill=tk.BOTH, expand=True)
    
    def create_outcomes_tab(self):
        """Tab muc tieu hoc tap"""
        outcomes_frame = ttk.Frame(self.notebook)
        self.notebook.add(outcomes_frame, text="Muc tieu")
        
        # Treeview
        out_columns = ("STT", "Ma CLO", "Noi dung")
        self.outcomes_tree = ttk.Treeview(outcomes_frame, columns=out_columns, show="headings")
        
        self.outcomes_tree.heading("STT", text="STT")
        self.outcomes_tree.heading("Ma CLO", text="Ma CLO")
        self.outcomes_tree.heading("Noi dung", text="Noi dung")
        
        self.outcomes_tree.column("STT", width=50)
        self.outcomes_tree.column("Ma CLO", width=80)
        self.outcomes_tree.column("Noi dung", width=400)
        
        self.outcomes_tree.pack(fill=tk.BOTH, expand=True)
    
    def load_data(self):
        """Tai du lieu tu file"""
        try:
            if os.path.exists(self.current_file):
                with open(self.current_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                
                if isinstance(self.data, dict) and 'syllabuses' in self.data:
                    self.syllabuses = self.data['syllabuses']
                elif isinstance(self.data, list):
                    self.syllabuses = self.data
                else:
                    self.syllabuses = [self.data]
                
                self.populate_course_list()
                self.status_var.set(f"Da tai {len(self.syllabuses)} mon hoc tu {self.current_file}")
            else:
                self.status_var.set("File khong ton tai")
                
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the tai file: {str(e)}")
    
    def populate_course_list(self):
        """Dien du lieu vao danh sach mon hoc"""
        
        # Xoa du lieu cu
        for item in self.course_tree.get_children():
            self.course_tree.delete(item)
        
        # Them du lieu moi
        for i, syllabus in enumerate(self.syllabuses):
            if 'metadata' in syllabus:
                meta = syllabus['metadata']
                course_code = meta.get('subject_code_on_page', '')
                course_name = meta.get('course_name_from_curriculum', '')
                semester = meta.get('semester_from_curriculum', '')
                credits = meta.get('credits', '')
                
                # Rut gon ten mon neu qua dai
                if len(course_name) > 40:
                    course_name = course_name[:37] + "..."
                
                self.course_tree.insert('', 'end', iid=i, values=(course_code, course_name, semester, credits))
    
    def on_course_select(self, event):
        """Xu ly khi chon mon hoc"""
        selection = self.course_tree.selection()
        if selection:
            index = int(selection[0])
            self.display_course_details(index)
    
    def display_course_details(self, index):
        """Hien thi chi tiet mon hoc"""
        if 0 <= index < len(self.syllabuses):
            syllabus = self.syllabuses[index]
            
            # Hien thi thong tin co ban
            if 'metadata' in syllabus:
                meta = syllabus['metadata']
                for field, entry in self.basic_fields.items():
                    entry.delete(0, tk.END)
                    value = meta.get(field, '')
                    entry.insert(0, str(value))
                
                # Mo ta
                self.description_text.delete(1.0, tk.END)
                self.description_text.insert(1.0, meta.get('description', ''))
                
                # Nhiem vu
                self.tasks_text.delete(1.0, tk.END)
                self.tasks_text.insert(1.0, meta.get('student_tasks', ''))
            
            # Hien thi tai lieu
            self.display_materials(syllabus.get('materials', []))
            
            # Hien thi danh gia
            self.display_assessments(syllabus.get('assessments', []))
            
            # Hien thi lich trinh
            self.display_schedule(syllabus.get('schedule', []))
            
            # Hien thi muc tieu
            self.display_outcomes(syllabus.get('learning_outcomes', []))
    
    def display_materials(self, materials):
        """Hien thi danh sach tai lieu"""
        for item in self.materials_tree.get_children():
            self.materials_tree.delete(item)
        
        for i, material in enumerate(materials):
            values = (
                i + 1,
                material.get('description', ''),
                material.get('author', ''),
                material.get('publisher', ''),
                material.get('published_date', ''),
                'Co' if material.get('is_main_material', False) else 'Khong'
            )
            self.materials_tree.insert('', 'end', values=values)
    
    def display_assessments(self, assessments):
        """Hien thi danh sach danh gia"""
        for item in self.assessments_tree.get_children():
            self.assessments_tree.delete(item)
        
        for i, assessment in enumerate(assessments):
            values = (
                i + 1,
                assessment.get('category', ''),
                assessment.get('part', ''),
                assessment.get('weight', ''),
                assessment.get('duration', ''),
                assessment.get('note', '')
            )
            self.assessments_tree.insert('', 'end', values=values)
    
    def display_schedule(self, schedule):
        """Hien thi lich trinh"""
        for item in self.schedule_tree.get_children():
            self.schedule_tree.delete(item)
        
        for i, session in enumerate(schedule):
            values = (
                i + 1,
                session.get('session', ''),
                session.get('topic', ''),
                session.get('teaching_type', ''),
                str(session.get('learning_outcomes', [])),
                session.get('materials', '')
            )
            self.schedule_tree.insert('', 'end', values=values)
    
    def display_outcomes(self, outcomes):
        """Hien thi muc tieu hoc tap"""
        for item in self.outcomes_tree.get_children():
            self.outcomes_tree.delete(item)
        
        for i, outcome in enumerate(outcomes):
            values = (
                i + 1,
                outcome.get('id', ''),
                outcome.get('details', '')
            )
            self.outcomes_tree.insert('', 'end', values=values)
    
    def on_search_change(self, *args):
        """Tim kiem mon hoc"""
        search_term = self.search_var.get().lower()
        
        # Xoa du lieu cu
        for item in self.course_tree.get_children():
            self.course_tree.delete(item)
        
        # Loc va hien thi
        for i, syllabus in enumerate(self.syllabuses):
            if 'metadata' in syllabus:
                meta = syllabus['metadata']
                course_code = meta.get('subject_code_on_page', '').lower()
                course_name = meta.get('course_name_from_curriculum', '').lower()
                
                if search_term in course_code or search_term in course_name:
                    semester = meta.get('semester_from_curriculum', '')
                    credits = meta.get('credits', '')
                    
                    display_name = meta.get('course_name_from_curriculum', '')
                    if len(display_name) > 40:
                        display_name = display_name[:37] + "..."
                    
                    self.course_tree.insert('', 'end', iid=i, values=(
                        meta.get('subject_code_on_page', ''),
                        display_name,
                        semester,
                        credits
                    ))
    
    def focus_search(self):
        """Focus vao o tim kiem"""
        self.search_entry.focus()
    
    def open_file(self):
        """Mo file khac"""
        filename = filedialog.askopenfilename(
            title="Chon file JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.current_file = filename
            self.load_data()
    
    def save_data(self):
        """Luu du lieu"""
        try:
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            self.status_var.set("Da luu thanh cong")
            messagebox.showinfo("Thanh cong", "Da luu du lieu thanh cong!")
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the luu file: {str(e)}")
    
    def save_as(self):
        """Luu thanh file khac"""
        filename = filedialog.asksaveasfilename(
            title="Luu file JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.current_file = filename
            self.save_data()
    
    def add_course(self):
        """Them mon hoc moi"""
        messagebox.showinfo("Thong bao", "Chuc nang dang phat trien")
    
    def delete_course(self):
        """Xoa mon hoc"""
        selection = self.course_tree.selection()
        if selection:
            if messagebox.askyesno("Xac nhan", "Ban co chac muon xoa mon hoc nay?"):
                index = int(selection[0])
                del self.syllabuses[index]
                self.populate_course_list()
                self.status_var.set("Da xoa mon hoc")
        else:
            messagebox.showwarning("Canh bao", "Vui long chon mon hoc can xoa")
    
    def add_material(self):
        """Them tai lieu"""
        messagebox.showinfo("Thong bao", "Chuc nang dang phat trien")
    
    def delete_material(self):
        """Xoa tai lieu"""
        messagebox.showinfo("Thong bao", "Chuc nang dang phat trien")
    
    def add_assessment(self):
        """Them danh gia"""
        messagebox.showinfo("Thong bao", "Chuc nang dang phat trien")
    
    def delete_assessment(self):
        """Xoa danh gia"""
        messagebox.showinfo("Thong bao", "Chuc nang dang phat trien")
    
    def show_statistics(self):
        """Hien thi thong ke"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("THONG KE DU LIEU")
        stats_window.geometry("600x400")
        
        stats_text = scrolledtext.ScrolledText(stats_window, width=70, height=25)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tinh toan thong ke
        total_courses = len(self.syllabuses)
        semester_counts = {}
        credit_counts = {}
        
        for syllabus in self.syllabuses:
            if 'metadata' in syllabus:
                meta = syllabus['metadata']
                semester = meta.get('semester_from_curriculum', 'Unknown')
                credits = meta.get('credits', '0')
                
                semester_counts[semester] = semester_counts.get(semester, 0) + 1
                credit_counts[credits] = credit_counts.get(credits, 0) + 1
        
        # Hien thi thong ke
        stats_content = f"""THONG KE CHUONG TRINH HOC AI
{'='*50}

TONG QUAN:
- Tong so mon hoc: {total_courses}
- So hoc ky: {len(semester_counts)}

PHAN PHOI THEO HOC KY:
"""
        for semester in sorted(semester_counts.keys()):
            stats_content += f"- Hoc ky {semester}: {semester_counts[semester]} mon\n"
        
        stats_content += f"\nPHAN PHOI THEO TIN CHI:\n"
        for credits in sorted(credit_counts.keys()):
            stats_content += f"- {credits} tin chi: {credit_counts[credits]} mon\n"
        
        stats_content += f"\nTHOI GIAN CAP NHAT: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        
        stats_text.insert(1.0, stats_content)
    
    def refresh_data(self):
        """Lam moi du lieu"""
        self.load_data()
    
    def on_course_double_click(self, event):
        """Xu ly double click vao mon hoc"""
        selection = self.course_tree.selection()
        if selection:
            index = int(selection[0])
            # Co the mo dialog chinh sua chi tiet
            messagebox.showinfo("Thong bao", f"Chinh sua mon hoc thu {index + 1}")

def main():
    """Ham chinh"""
    root = tk.Tk()
    app = JSONManagerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()