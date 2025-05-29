from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, BigInteger
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ImportRecord(Base):
    __tablename__ = "import_records"

    import_id = Column(String(50), primary_key=True)
    import_time = Column(DateTime)
    file_path = Column(String(255))
    row_count = Column(Integer)
    description = Column(Text)

class StudentData(Base):
    __tablename__ = "student_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    import_id = Column(String(50), ForeignKey("import_records.import_id"))

    Student_ID = Column(String(50))
    First_Name = Column(String(100))
    Last_Name = Column(String(100))
    Total_Score = Column(Float)
    Age = Column(Integer)
    Attendance = Column(Float)
    Midterm_Score = Column(Float)
    Final_Score = Column(Float)
    Assignments_Avg = Column(Float)
    Quizzes_Avg = Column(Float)
    Projects_Score = Column(Float)
    Study_Hours_per_Week = Column(Float)
    Sleep_Hours_per_Night = Column(Float)
    Stress_Level = Column(Integer)
    Gender = Column(String(50))
    Department = Column(String(100))
    Extracurricular_Activities = Column(String(50))
    Internet_Access_at_Home = Column(String(50))
    Participation_Score = Column(Float)
    Academic_Risk = Column(Integer)
    Risk_Probability = Column(Float)
    Risk_Level = Column(String(50))
