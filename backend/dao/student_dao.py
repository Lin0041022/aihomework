from typing import List

from backend.models.student_models import StudentData
from sqlalchemy.orm import Session


def save_student_batch(df, import_id: str, db: Session):
    student_entries = []
    for _, row in df.iterrows():
        student_entries.append(StudentData(
            import_id=import_id,
            Student_ID=row.get('Student_ID', ''),
            First_Name=row.get('First_Name', ''),
            Last_Name=row.get('Last_Name', ''),
            Total_Score=row.get('Total_Score', 0),
            Age=int(row.get('Age', 0)),
            Attendance=row.get('Attendance (%)', 0),
            Midterm_Score=row.get('Midterm_Score', 0),
            Final_Score=row.get('Final_Score', 0),
            Assignments_Avg=row.get('Assignments_Avg', 0),
            Quizzes_Avg=row.get('Quizzes_Avg', 0),
            Projects_Score=row.get('Projects_Score', 0),
            Study_Hours_per_Week=row.get('Study_Hours_per_Week', 0),
            Sleep_Hours_per_Night=row.get('Sleep_Hours_per_Night', 0),
            Stress_Level=int(row.get('Stress_Level (1-10)', 0)),
            Gender=row.get('Gender', ''),
            Department=row.get('Department', ''),
            Extracurricular_Activities=row.get('Extracurricular_Activities', ''),
            Internet_Access_at_Home=row.get('Internet_Access_at_Home', ''),
            Participation_Score=row.get('Participation_Score', 0),
            Academic_Risk=int(row.get('Academic_Risk', 0)),
            Risk_Probability=None,
            Risk_Level=None
        ))
    db.bulk_save_objects(student_entries)
    db.commit()


# academic_dao.py
from contextlib import contextmanager
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database

from backend.database.db import engine, SessionLocal
from backend.models.student_models import Base, ImportRecord, StudentData


class AcademicDAO:
    """学业预警系统数据访问对象"""

    def __init__(self):
        """初始化DAO"""
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.init_database()

    @contextmanager
    def get_db_session(self):
        """获取数据库会话上下文管理器"""
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def init_database(self):
        """初始化数据库"""
        try:
            # 创建数据库（如果不存在）
            if not database_exists(engine.url):
                create_database(engine.url)
                print("数据库 student_mis 创建成功！")
            else:
                print("数据库已存在，无需创建。")

            # 创建所有表（如果不存在）
            Base.metadata.create_all(bind=engine)
            print("表结构初始化成功！")

        except Exception as e:
            print(f"数据库初始化失败：{e}")
            raise

    def save_import_record(self, new_record: ImportRecord):
        """保存导入记录"""
        with self.get_db_session() as db:
            db.add(new_record)

    def save_student_batch(self, df, import_id: str):
        student_entries = []
        for _, row in df.iterrows():
            student_entries.append(StudentData(
                import_id=import_id,
                Student_ID=row.get('Student_ID', ''),
                First_Name=row.get('First_Name', ''),
                Last_Name=row.get('Last_Name', ''),
                Total_Score=row.get('Total_Score', 0),
                Age=int(row.get('Age', 0)),
                Attendance=row.get('Attendance (%)', 0),
                Midterm_Score=row.get('Midterm_Score', 0),
                Final_Score=row.get('Final_Score', 0),
                Assignments_Avg=row.get('Assignments_Avg', 0),
                Quizzes_Avg=row.get('Quizzes_Avg', 0),
                Projects_Score=row.get('Projects_Score', 0),
                Study_Hours_per_Week=row.get('Study_Hours_per_Week', 0),
                Sleep_Hours_per_Night=row.get('Sleep_Hours_per_Night', 0),
                Stress_Level=int(row.get('Stress_Level (1-10)', 0)),
                Gender=row.get('Gender', ''),
                Department=row.get('Department', ''),
                Extracurricular_Activities=row.get('Extracurricular_Activities', ''),
                Internet_Access_at_Home=row.get('Internet_Access_at_Home', ''),
                Participation_Score=row.get('Participation_Score', 0),
                Academic_Risk=int(row.get('Academic_Risk', 0)),
                Risk_Probability=None,
                Risk_Level=None
            ))
        with self.get_db_session() as db:
            db.bulk_save_objects(student_entries)

    def get_students_by_import_id(self, import_id: str) -> list[dict]:
        """
        根据 import_id 获取学生记录，并转换为字典列表（用于 DataFrame 构造）
        """
        with self.get_db_session() as db:
            students = db.query(StudentData).filter(StudentData.import_id == import_id).all()

            return [{
                'Student_ID': s.Student_ID,
                'First_Name': s.First_Name,
                'Last_Name': s.Last_Name,
                'Total_Score': s.Total_Score,
                'Age': s.Age,
                'Attendance (%)': s.Attendance,
                'Midterm_Score': s.Midterm_Score,
                'Final_Score': s.Final_Score,
                'Assignments_Avg': s.Assignments_Avg,
                'Quizzes_Avg': s.Quizzes_Avg,
                'Projects_Score': s.Projects_Score,
                'Study_Hours_per_Week': s.Study_Hours_per_Week,
                'Sleep_Hours_per_Night': s.Sleep_Hours_per_Night,
                'Stress_Level (1-10)': s.Stress_Level,
                'Gender': s.Gender,
                'Department': s.Department,
                'Extracurricular_Activities': s.Extracurricular_Activities,
                'Internet_Access_at_Home': s.Internet_Access_at_Home,
                'Participation_Score': s.Participation_Score,
                'Academic_Risk': s.Academic_Risk,
                'Risk_Probability': s.Risk_Probability,
                'Risk_Level': s.Risk_Level
            } for s in students]

    def get_import_records(self):
        try:
            with self.get_db_session() as db:
                records = db.query(ImportRecord).order_by(ImportRecord.import_time.desc()).all()
                return [(r.import_id, r.import_time, r.file_path, r.row_count, r.description) for r in records]
        except Exception as e:
            print(f"获取导入记录失败: {e}")
            return []

    def update_academic_risk(self, df, import_id: str):
        # ORM 更新数据库中 Academic_Risk 字段
        from backend.models.student_models import StudentData  # 你的 ORM 模型
        try:
            with self.get_db_session() as db:
                for _, row in df.iterrows():
                    db.query(StudentData).filter(
                        StudentData.import_id == import_id,
                        StudentData.Student_ID == str(row['Student_ID'])
                    ).update({"Academic_Risk": int(row['Academic_Risk'])})
            return True
        except Exception as e:
            print(f"更新Academic_Risk失败: {e}")
            return False

    def update_predictions(self, df, import_id: str):
        try:
            for _, row in df.iterrows():
                with  self.get_db_session() as db:
                    db.query(StudentData).filter(
                        StudentData.import_id == import_id,
                        StudentData.Student_ID == str(row['Student_ID'])
                    ).update({
                        StudentData.Risk_Probability: row.get('Risk_Probability', 0),
                        StudentData.Risk_Level: row.get('Risk_Level', '')
                    })
            print("预测结果已更新到数据库！")
        except Exception as e:
            print(f"更新预测结果失败: {e}")
            db.rollback()
            raise
