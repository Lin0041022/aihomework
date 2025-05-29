from contextlib import contextmanager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import matplotlib
import time
from datetime import datetime

from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database

from dao.student_dao import save_student_batch
from database.db import engine, SessionLocal
from models.student_models import Base, ImportRecord, StudentData

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class AcademicWarningSystem:
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.optimal_threshold = 0.5
        self.db_conn = None
        self.import_id = None
        self.init_database()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    @contextmanager
    def get_db_session(self):
        db = SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def init_database(self):
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

    from sqlalchemy.orm import Session
    def load_data(self, file_path: str, db: Session) -> bool:
        try:
            self.df = pd.read_csv(file_path)
            print(f"数据加载成功！共有 {len(self.df)} 条记录")
            print("加载的列名:", self.df.columns.tolist())
            print("每列非空值数量:\n", self.df.notnull().sum())

            self.import_id = str(int(time.time()))
            import_time = datetime.now()

            # 保存 import_record 和 student_data
            new_record = ImportRecord(
                import_id=self.import_id,
                import_time=import_time,
                file_path=file_path,
                row_count=len(self.df),
                description=""
            )
            db.add(new_record)
            db.commit()

            save_student_batch(self.df, self.import_id, db)
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            db.rollback()
            return False

    def update_predictions_in_db(self, db: Session):
        try:
            for _, row in self.df.iterrows():
                db.query(StudentData).filter(
                    StudentData.import_id == self.import_id,
                    StudentData.Student_ID == str(row['Student_ID'])
                ).update({
                    StudentData.Risk_Probability: row.get('Risk_Probability', 0),
                    StudentData.Risk_Level: row.get('Risk_Level', '')
                })
            db.commit()
            print("预测结果已更新到数据库！")
        except Exception as e:
            print(f"更新预测结果失败: {e}")
            db.rollback()
            raise

    def load_data_from_db(self, import_id: str, db: Session) -> bool:
        try:
            students = db.query(StudentData).filter(StudentData.import_id == import_id).all()
            if not students:
                print(f"未找到导入ID为 {import_id} 的记录")
                return False

            self.df = pd.DataFrame([{
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
            } for s in students])
            self.import_id = import_id

            print(f"从数据库加载数据成功！导入ID: {import_id}，共有 {len(self.df)} 条记录")
            print("加载的列名:", self.df.columns.tolist())
            print("每列非空值数量:\n", self.df.notnull().sum())
            return True
        except Exception as e:
            print(f"从数据库加载数据失败: {e}")
            return False

    def get_import_records(self, db: Session):
        try:
            records = db.query(ImportRecord).order_by(ImportRecord.import_time.desc()).all()
            return [(r.import_id, r.import_time, r.file_path, r.row_count, r.description) for r in records]
        except Exception as e:
            print(f"获取导入记录失败: {e}")
            return []

    def clean_data(self):
        """数据清洗和预处理"""
        if self.df is None:
            print("请先加载数据")
            return False

        print("开始数据清洗...")

        print("\n=== 数据基本信息 ===")
        print(f"数据维度: {self.df.shape}")
        print(f"缺失值统计:\n{self.df.isnull().sum()}")
        print(f"每列非空值数量:\n{self.df.notnull().sum()}")

        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)

        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].notnull().sum() == 0:
                    print(f"警告: 列 '{col}' 全为空，使用默认值 'Unknown' 填充")
                    self.df[col].fillna('Unknown', inplace=True)
                else:
                    try:
                        mode_value = self.df[col].mode()
                        if not mode_value.empty:
                            self.df[col].fillna(mode_value[0], inplace=True)
                        else:
                            print(f"警告: 列 '{col}' 没有有效众数，使用 'Unknown' 填充")
                            self.df[col].fillna('Unknown', inplace=True)
                    except Exception as e:
                        print(f"处理列 '{col}' 时出错: {e}")
                        self.df[col].fillna('Unknown', inplace=True)

        self.detect_anomalies()

        self.df['Academic_Risk'] = (self.df['Total_Score'] < 60).astype(int)

        # ORM 更新数据库中 Academic_Risk 字段
        from models.student_models import StudentData  # 你的 ORM 模型
        from contextlib import contextmanager
        try:
            with self.get_db_session() as db:
                for _, row in self.df.iterrows():
                    db.query(StudentData).filter(
                        StudentData.import_id == self.import_id,
                        StudentData.Student_ID == str(row['Student_ID'])
                    ).update({"Academic_Risk": int(row['Academic_Risk'])})
            print("数据清洗完成！")
            return True
        except Exception as e:
            print(f"更新Academic_Risk失败: {e}")
            return False

    def detect_anomalies(self):
        """异常值检测"""
        print("\n=== 异常值检测 ===")

        numeric_cols = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score',
                        'Assignments_Avg', 'Quizzes_Avg', 'Total_Score',
                        'Study_Hours_per_Week', 'Sleep_Hours_per_Night']

        anomaly_count = 0
        for col in numeric_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                if len(outliers) > 0:
                    print(f"{col}: 发现 {len(outliers)} 个异常值")
                    anomaly_count += len(outliers)
                    self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)

        print(f"总共处理了 {anomaly_count} 个异常值")

    def prepare_features(self):
        """特征工程"""
        print("\n=== 特征工程 ===")

        print("检查Total_Score分布:")
        print(f"最小值: {self.df['Total_Score'].min()}")
        print(f"最大值: {self.df['Total_Score'].max()}")
        print(f"平均值: {self.df['Total_Score'].mean():.2f}")
        print(
            f"低于60分的学生: {(self.df['Total_Score'] < 60).sum()} ({(self.df['Total_Score'] < 60).mean() * 100:.1f}%)")

        self.df['Exam_Average'] = (self.df['Midterm_Score'] + self.df['Final_Score']) / 2
        self.df['Coursework_Average'] = (self.df['Assignments_Avg'] + self.df['Quizzes_Avg'] +
                                         self.df['Projects_Score']) / 3
        self.df['Score_Volatility'] = abs(self.df['Midterm_Score'] - self.df['Final_Score'])
        self.df['Performance_Gap'] = self.df['Exam_Average'] - self.df['Coursework_Average']
        self.df['Academic_Engagement'] = (self.df['Attendance (%)'] * 0.3 +
                                          self.df['Participation_Score'] * 0.7)
        self.df['Study_Efficiency'] = self.df['Total_Score'] / (self.df['Study_Hours_per_Week'] + 1)
        self.df['Rest_Study_Balance'] = self.df['Sleep_Hours_per_Night'] / (self.df['Study_Hours_per_Week'] + 1)
        self.df['High_Stress_Flag'] = (self.df['Stress_Level (1-10)'] >= 7).astype(int)
        self.df['Low_Attendance_Flag'] = (self.df['Attendance (%)'] < 80).astype(int)
        self.df['Poor_Sleep_Flag'] = (self.df['Sleep_Hours_per_Night'] < 6).astype(int)

        core_features = [
            'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg',
            'Quizzes_Avg', 'Participation_Score', 'Projects_Score',
            'Study_Hours_per_Week', 'Sleep_Hours_per_Night',
            'Age', 'Stress_Level (1-10)',
            'Exam_Average', 'Coursework_Average', 'Score_Volatility',
            'Performance_Gap', 'Academic_Engagement', 'Study_Efficiency',
            'Rest_Study_Balance', 'High_Stress_Flag', 'Low_Attendance_Flag',
            'Poor_Sleep_Flag'
        ]

        important_categorical = ['Gender', 'Department']
        for col in important_categorical:
            if col in self.df.columns:
                unique_vals = self.df[col].unique()
                if len(unique_vals) <= 5:
                    le = LabelEncoder()
                    self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                    self.label_encoders[col] = le
                    core_features.append(col + '_encoded')

        binary_cols = ['Extracurricular_Activities', 'Internet_Access_at_Home']
        for col in binary_cols:
            if col in self.df.columns:
                self.df[col + '_binary'] = (self.df[col] == 'Yes').astype(int)
                core_features.append(col + '_binary')

        self.feature_columns = [col for col in core_features if col in self.df.columns]

        print(f"选择了 {len(self.feature_columns)} 个特征:")
        for i, feature in enumerate(self.feature_columns):
            print(f"{i + 1:2d}. {feature}")

        if 'Academic_Risk' in self.df.columns:
            correlations = []
            for feature in self.feature_columns:
                if self.df[feature].dtype in ['int64', 'float64']:
                    corr = self.df[feature].corr(self.df['Academic_Risk'])
                    correlations.append((feature, abs(corr)))

            correlations.sort(key=lambda x: x[1], reverse=True)
            print(f"\n前10个与学业风险相关性最高的特征:")
            for i, (feature, corr) in enumerate(correlations[:10]):
                print(f"{i + 1:2d}. {feature}: {corr:.3f}")

    def build_model(self):
        """构建预警模型"""
        print("\n=== 构建预警模型 ===")

        X = self.df[self.feature_columns]
        y = self.df['Academic_Risk']

        print(f"特征矩阵形状: {X.shape}")
        print(f"类别分布:\n{y.value_counts()}")
        print(f"风险学生比例: {y.mean():.2%}")

        if y.sum() < 10:
            print("⚠️  警告: 风险学生样本太少，可能影响模型性能")
            risk_threshold = self.df['Total_Score'].quantile(0.25)
            print(f"新的风险阈值: {risk_threshold:.1f}")
            self.df['Academic_Risk'] = (self.df['Total_Score'] < risk_threshold).astype(int)
            y = self.df['Academic_Risk']
            print(f"调整后类别分布:\n{y.value_counts()}")

        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        print(f"清理后数据量: {len(X)}")

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        from sklearn.tree import DecisionTreeClassifier
        models = {
            'DecisionTree': DecisionTreeClassifier(
                max_depth=5, min_samples_split=20, min_samples_leaf=10,
                random_state=42, class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=50, max_depth=5, min_samples_split=20,
                min_samples_leaf=10, random_state=42, class_weight='balanced'
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, class_weight='balanced', C=1.0, max_iter=1000
            )
        }

        best_f1 = 0
        best_model = None
        best_model_name = ""
        results = {}

        print("\n=== 模型比较 ===")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            results[name] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc_score,
                'model': model
            }

            print(f"\n{name}:")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  AUC: {auc_score:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

        self.model = best_model
        self.optimal_threshold = 0.5

        print(f"\n=== 最佳模型: {best_model_name} (F1: {best_f1:.4f}) ===")

        y_pred_final = self.model.predict(X_test)
        print("\n=== 最终模型性能 ===")
        print(classification_report(y_test, y_pred_final, target_names=['正常', '风险']))

        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\n=== 特征重要性 TOP 10 ===")
            for i, row in feature_importance.head(10).iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")

        self.plot_confusion_matrix(y_test, y_pred_final)

        return X_test, y_test, y_pred_final, self.model.predict_proba(X_test)[:, 1]

    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '风险'], yticklabels=['正常', '风险'])
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.show()

    def predict_risk(self, student_data=None):
        """预测学业风险"""
        if self.model is None:
            print("请先构建模型")
            return None

        if student_data is None:
            X = self.df[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict_proba(X_scaled)[:, 1]
            risk_predictions = (predictions >= self.optimal_threshold).astype(int)

            self.df['Risk_Probability'] = predictions
            self.df['Risk_Prediction'] = risk_predictions

            def categorize_risk(prob):
                if prob < 0.3:
                    return '低风险'
                elif prob < 0.6:
                    return '中风险'
                elif prob < 0.8:
                    return '高风险'
                else:
                    return '极高风险'

            self.df['Risk_Level'] = self.df['Risk_Probability'].apply(categorize_risk)
            with self.get_db_session() as db:
                self.update_predictions_in_db(db)

            return self.df[['Student_ID', 'First_Name', 'Last_Name',
                            'Total_Score', 'Risk_Probability', 'Risk_Level', 'Risk_Prediction']]

    def generate_visualizations(self):
        """生成可视化图表"""
        if self.df is None:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('学业预警系统 - 数据分析报告', fontsize=16)

        axes[0, 0].hist(self.df['Total_Score'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(x=60, color='red', linestyle='--', label='及格线')
        axes[0, 0].set_title('总成绩分布')
        axes[0, 0].set_xlabel('总成绩')
        axes[0, 0].set_ylabel('学生人数')
        axes[0, 0].legend()

        if 'Risk_Level' in self.df.columns:
            risk_counts = self.df['Risk_Level'].value_counts()
            axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('学业风险等级分布')

        axes[0, 2].scatter(self.df['Attendance (%)'], self.df['Total_Score'], alpha=0.6)
        axes[0, 2].set_title('出勤率与总成绩关系')
        axes[0, 2].set_xlabel('出勤率 (%)')
        axes[0, 2].set_ylabel('总成绩')

        if 'Department' in self.df.columns:
            dept_scores = []
            dept_names = []
            for dept in self.df['Department'].unique():
                dept_data = self.df[self.df['Department'] == dept]['Total_Score']
                dept_scores.append(dept_data)
                dept_names.append(dept)
            axes[1, 0].boxplot(dept_scores, labels=dept_names)
            axes[1, 0].set_title('各系别成绩分布')
            axes[1, 0].tick_params(axis='x', rotation=45)

        axes[1, 1].scatter(self.df['Study_Hours_per_Week'], self.df['Total_Score'], alpha=0.6)
        axes[1, 1].set_title('每周学习时间与成绩关系')
        axes[1, 1].set_xlabel('每周学习时间(小时)')
        axes[1, 1].set_ylabel('总成绩')

        stress_avg = self.df.groupby('Stress_Level (1-10)')['Total_Score'].mean()
        axes[1, 2].bar(stress_avg.index, stress_avg.values)
        axes[1, 2].set_title('压力水平与平均成绩关系')
        axes[1, 2].set_xlabel('压力水平')
        axes[1, 2].set_ylabel('平均成绩')

        plt.tight_layout()
        plt.show()

        self.show_warning_statistics()

    def show_warning_statistics(self):
        """显示预警统计信息"""
        if 'Academic_Risk' not in self.df.columns:
            return

        print("\n=== 学业预警统计 ===")
        total_students = len(self.df)
        at_risk_students = self.df['Academic_Risk'].sum()

        print(f"总学生数: {total_students}")
        print(f"预警学生数: {at_risk_students}")
        print(f"预警比例: {at_risk_students / total_students * 100:.2f}%")

        if 'Risk_Level' in self.df.columns:
            print("\n风险等级分布:")
            print(self.df['Risk_Level'].value_counts())

            high_risk = self.df[self.df['Risk_Level'] == '高风险']
            if len(high_risk) > 0:
                print(f"\n高风险学生列表 (共{len(high_risk)}人):")
                print(high_risk[['Student_ID', 'First_Name', 'Last_Name',
                                 'Total_Score', 'Risk_Probability']].head(10))
