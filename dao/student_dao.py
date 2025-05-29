from models.student_models import StudentData
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
