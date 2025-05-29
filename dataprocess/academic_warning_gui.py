import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from contextlib import contextmanager
import matplotlib.pyplot as plt
import warnings
import matplotlib
from sqlalchemy.orm import sessionmaker
from database.db import engine
from dataprocess.data_processor import AcademicWarningSystem

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class AcademicWarningGUI:
    def __init__(self):
        self.system = AcademicWarningSystem()
        self.root = tk.Tk()
        self.root.title("学业预警系统")
        self.root.geometry("800x600")
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.setup_gui()

    @contextmanager
    def get_db_session(self):
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="学业预警系统", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        history_frame = ttk.Frame(main_frame)
        history_frame.pack(pady=5, fill=tk.X)

        ttk.Label(history_frame, text="选择历史导入数据：").pack(side=tk.LEFT, padx=5)
        self.import_combo = ttk.Combobox(history_frame, state="readonly", width=50)
        self.import_combo.pack(side=tk.LEFT, padx=5)
        self.import_combo.bind("<<ComboboxSelected>>", self.load_selected_import)
        ttk.Button(history_frame, text="加载历史数据", command=self.load_selected_import).pack(side=tk.LEFT, padx=5)

        self.refresh_import_records()

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        # 按钮和对应方法
        self.buttons = {}
        button_defs = [
            ("加载新数据", self.load_data),
            ("数据预处理", self.preprocess_data),
            ("构建模型", self.build_model),
            ("生成预警", self.generate_warnings),
            ("显示图表", self.show_visualizations),
            ("导出结果", self.export_results)
        ]
        for i, (text, cmd) in enumerate(button_defs):
            btn = ttk.Button(button_frame, text=text, command=cmd, width=15)
            btn.grid(row=i // 3, column=i % 3, padx=5, pady=5)
            self.buttons[text] = btn

        self.result_text = tk.Text(main_frame, height=25, width=90)
        self.result_text.pack(pady=10, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=scrollbar.set)

    def refresh_import_records(self):
        with self.get_db_session() as db:
            records = self.system.get_import_records(db)
            values = [
                f"{rec[1]} - {rec[2]} ({rec[3]} 行{', ' + rec[4] if rec[4] else ''})"
                for rec in records
            ]
            self.import_combo['values'] = values
            self.import_combo.import_ids = [rec[0] for rec in records]
            if values:
                self.import_combo.current(0)

    def load_selected_import(self, event=None):
        if not self.import_combo.get():
            return
        selected_index = self.import_combo.current()
        if selected_index < 0 or selected_index >= len(self.import_combo.import_ids):
            messagebox.showwarning("警告", "请选择有效的历史记录！")
            return
        import_id = self.import_combo.import_ids[selected_index]

        def task():
            with self.get_db_session() as db:
                success = self.system.load_data_from_db(import_id, db)
            if success:
                self.root.after(0, lambda: self.log_message(f"加载历史数据成功！导入ID: {import_id}"))
                self.root.after(0, lambda: self.log_message(f"数据形状: {self.system.df.shape}"))
                self.root.after(0, lambda: self.log_message(f"列名: {list(self.system.df.columns)}"))
            else:
                self.root.after(0, lambda: messagebox.showerror("错误", "加载历史数据失败！"))

        threading.Thread(target=task).start()

    def log_message(self, message):
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.root.update()

    def disable_buttons(self):
        for btn in self.buttons.values():
            btn.config(state=tk.DISABLED)

    def enable_buttons(self):
        for btn in self.buttons.values():
            btn.config(state=tk.NORMAL)

    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        def task():
            self.disable_buttons()
            success = self.system.load_data(file_path)
            if success:
                self.root.after(0, lambda: self.log_message(f"数据加载成功！文件: {file_path}"))
                self.root.after(0, lambda: self.log_message(f"数据形状: {self.system.df.shape}"))
                self.root.after(0, lambda: self.log_message(f"列名: {list(self.system.df.columns)}"))
                self.root.after(0, self.refresh_import_records)
            else:
                self.root.after(0, lambda: messagebox.showerror("错误", "数据加载失败！"))
            self.root.after(0, self.enable_buttons)

        threading.Thread(target=task).start()

    def preprocess_data(self):
        if self.system.df is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        def task():
            self.disable_buttons()
            self.root.after(0, lambda: self.log_message("开始数据预处理..."))
            success = self.system.clean_data()
            if success:
                self.system.prepare_features()
                self.root.after(0, lambda: self.log_message("数据预处理完成！"))
            else:
                self.root.after(0, lambda: messagebox.showerror("错误", "数据预处理失败！"))
            self.root.after(0, self.enable_buttons)

        threading.Thread(target=task).start()

    def build_model(self):
        if self.system.df is None or not self.system.feature_columns:
            messagebox.showwarning("警告", "请先完成数据预处理！")
            return

        def task():
            self.disable_buttons()
            self.root.after(0, lambda: self.log_message("开始构建预警模型..."))
            try:
                self.system.build_model()
                self.root.after(0, lambda: self.log_message("模型构建完成！"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"模型构建失败: {str(e)}"))
            self.root.after(0, self.enable_buttons)

        threading.Thread(target=task).start()

    def generate_warnings(self):
        if self.system.model is None:
            messagebox.showwarning("警告", "请先构建模型！")
            return

        def task():
            self.disable_buttons()
            self.root.after(0, lambda: self.log_message("正在生成学业预警..."))
            try:
                results = self.system.predict_risk()
                if results is not None:
                    self.root.after(0, lambda: self.log_message("预警生成完成！"))
                    self.root.after(0, lambda: self.log_message("\n前10名学生预警结果:"))
                    self.root.after(0, lambda: self.log_message(str(results.head(10))))
                    self.system.show_warning_statistics()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"预警生成失败: {str(e)}"))
            self.root.after(0, self.enable_buttons)

        threading.Thread(target=task).start()

    def show_visualizations(self):
        if self.system.df is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        def task():
            self.disable_buttons()
            try:
                self.system.generate_visualizations()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"图表生成失败: {str(e)}"))
            self.root.after(0, self.enable_buttons)

        threading.Thread(target=task).start()

    def export_results(self):
        if self.system.df is None or 'Risk_Probability' not in self.system.df.columns:
            messagebox.showwarning("警告", "请先生成预警结果！")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存预警结果",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        def task():
            self.disable_buttons()
            try:
                export_df = self.system.df[['Student_ID', 'First_Name', 'Last_Name',
                                           'Total_Score', 'Risk_Probability', 'Risk_Level']]
                export_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                self.root.after(0, lambda: self.log_message(f"结果已导出到: {file_path}"))
                self.root.after(0, lambda: messagebox.showinfo("成功", "预警结果导出成功！"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"导出失败: {str(e)}"))
            self.root.after(0, self.enable_buttons)

        threading.Thread(target=task).start()

    def run(self):
        self.root.mainloop()

