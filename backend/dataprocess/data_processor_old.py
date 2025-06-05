from contextlib import contextmanager

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import matplotlib
import time

from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database

from backend.dao.student_dao import save_student_batch
from backend.database.db import engine, SessionLocal
from backend.models.student_models import Base, ImportRecord, StudentData

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
        from backend.models.student_models import StudentData  # 你的 ORM 模型
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

    # def generate_visualizations(self):
    #     """生成可视化图表"""
    #     if self.df is None:
    #         return
    #
    #     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    #     fig.suptitle('学业预警系统 - 数据分析报告', fontsize=16)
    #
    #     axes[0, 0].hist(self.df['Total_Score'], bins=20, alpha=0.7, color='skyblue')
    #     axes[0, 0].axvline(x=60, color='red', linestyle='--', label='及格线')
    #     axes[0, 0].set_title('总成绩分布')
    #     axes[0, 0].set_xlabel('总成绩')
    #     axes[0, 0].set_ylabel('学生人数')
    #     axes[0, 0].legend()
    #
    #     if 'Risk_Level' in self.df.columns:
    #         risk_counts = self.df['Risk_Level'].value_counts()
    #         axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
    #         axes[0, 1].set_title('学业风险等级分布')
    #
    #     axes[0, 2].scatter(self.df['Attendance (%)'], self.df['Total_Score'], alpha=0.6)
    #     axes[0, 2].set_title('出勤率与总成绩关系')
    #     axes[0, 2].set_xlabel('出勤率 (%)')
    #     axes[0, 2].set_ylabel('总成绩')
    #
    #     if 'Department' in self.df.columns:
    #         dept_scores = []
    #         dept_names = []
    #         for dept in self.df['Department'].unique():
    #             dept_data = self.df[self.df['Department'] == dept]['Total_Score']
    #             dept_scores.append(dept_data)
    #             dept_names.append(dept)
    #         axes[1, 0].boxplot(dept_scores, labels=dept_names)
    #         axes[1, 0].set_title('各系别成绩分布')
    #         axes[1, 0].tick_params(axis='x', rotation=45)
    #
    #     axes[1, 1].scatter(self.df['Study_Hours_per_Week'], self.df['Total_Score'], alpha=0.6)
    #     axes[1, 1].set_title('每周学习时间与成绩关系')
    #     axes[1, 1].set_xlabel('每周学习时间(小时)')
    #     axes[1, 1].set_ylabel('总成绩')
    #
    #     stress_avg = self.df.groupby('Stress_Level (1-10)')['Total_Score'].mean()
    #     axes[1, 2].bar(stress_avg.index, stress_avg.values)
    #     axes[1, 2].set_title('压力水平与平均成绩关系')
    #     axes[1, 2].set_xlabel('压力水平')
    #     axes[1, 2].set_ylabel('平均成绩')
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #     self.show_warning_statistics()

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


# 学业预警系统 - 缺失函数实现
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sqlalchemy.orm import Session


class AcademicWarningSystemExtended:
    """学业预警系统扩展类 - 实现缺失的方法"""

    def __init__(self, academic_system:AcademicWarningSystem):
        """
        初始化扩展类

        Args:
            academic_system: 主要的AcademicWarningSystem实例
        """
        self.system = academic_system

    def preprocess_data(self):
        """
        数据预处理主函数
        整合数据清洗、特征工程等步骤

        Returns:
            bool: 预处理是否成功
        """
        try:
            print("开始数据预处理...")

            # 检查数据是否已加载
            if self.system.df is None:
                print("错误：请先加载数据")
                return False

            # 步骤1：数据清洗
            if not self.system.clean_data():
                print("数据清洗失败")
                return False

            # 步骤2：特征工程
            self.system.prepare_features()

            # 步骤3：数据验证
            self._validate_preprocessed_data()

            print("数据预处理完成！")
            return True

        except Exception as e:
            print(f"数据预处理失败：{e}")
            return False

    def _validate_preprocessed_data(self):
        """验证预处理后的数据质量"""
        df = self.system.df

        print("\n=== 数据预处理验证 ===")
        print(f"最终数据维度: {df.shape}")
        print(f"缺失值数量: {df.isnull().sum().sum()}")
        print(f"特征列数量: {len(self.system.feature_columns)}")

        # 检查目标变量分布
        if 'Academic_Risk' in df.columns:
            risk_dist = df['Academic_Risk'].value_counts()
            print(f"目标变量分布: {risk_dist.to_dict()}")

    def load_data_by_import_id(self, import_id: str, db: Session) -> bool:
        """
        根据导入ID从数据库加载数据

        Args:
            import_id (str): 导入记录ID
            db (Session): 数据库会话

        Returns:
            bool: 加载是否成功
        """
        try:
            print(f"正在从数据库加载导入ID为 {import_id} 的数据...")

            # 调用系统的加载方法
            success = self.system.load_data_from_db(import_id, db)

            if success:
                print(f"数据加载成功！共 {len(self.system.df)} 条记录")
                # 显示数据概览
                self._show_data_overview()
            else:
                print("数据加载失败")

            return success

        except Exception as e:
            print(f"加载数据时出错：{e}")
            return False

    def _show_data_overview(self):
        """显示数据概览"""
        df = self.system.df
        if df is not None:
            print("\n=== 数据概览 ===")
            print(f"学生总数: {len(df)}")
            print(f"数据列数: {len(df.columns)}")

            if 'Total_Score' in df.columns:
                print(f"平均成绩: {df['Total_Score'].mean():.2f}")
                print(f"及格学生数: {(df['Total_Score'] >= 60).sum()}")
                print(f"不及格学生数: {(df['Total_Score'] < 60).sum()}")

    def generate_warnings(self) -> dict:
        """
        生成预警信息

        Returns:
            dict: 包含各类预警信息的字典
        """
        try:
            print("正在生成预警信息...")

            if self.system.df is None:
                return {"error": "未加载数据"}

            df = self.system.df
            warnings = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_students": len(df),
                "warnings": []
            }

            # 成绩预警
            score_warnings = self._generate_score_warnings(df)
            warnings["warnings"].extend(score_warnings)

            # 出勤预警
            attendance_warnings = self._generate_attendance_warnings(df)
            warnings["warnings"].extend(attendance_warnings)

            # 学习时间预警
            study_time_warnings = self._generate_study_time_warnings(df)
            warnings["warnings"].extend(study_time_warnings)

            # 压力水平预警
            stress_warnings = self._generate_stress_warnings(df)
            warnings["warnings"].extend(stress_warnings)

            # 如果已经进行了风险预测
            if 'Risk_Level' in df.columns:
                risk_warnings = self._generate_risk_warnings(df)
                warnings["warnings"].extend(risk_warnings)

            # 统计预警数量
            warnings["warning_count"] = len(warnings["warnings"])
            warnings["high_priority_count"] = len([w for w in warnings["warnings"]
                                                   if w.get("priority") == "高"])

            print(f"预警生成完成！共生成 {warnings['warning_count']} 条预警")
            return warnings

        except Exception as e:
            print(f"生成预警信息失败：{e}")
            return {"error": str(e)}

    def _generate_score_warnings(self, df) -> list:
        """生成成绩相关预警"""
        warnings = []

        if 'Total_Score' in df.columns:
            # 不及格预警
            failing_students = df[df['Total_Score'] < 60]
            for _, student in failing_students.iterrows():
                warnings.append({
                    "type": "成绩预警",
                    "student_id": student.get('Student_ID', 'Unknown'),
                    "student_name": f"{student.get('First_Name', '')} {student.get('Last_Name', '')}".strip(),
                    "message": f"总成绩不及格 ({student['Total_Score']:.1f}分)",
                    "priority": "高",
                    "score": student['Total_Score']
                })

            # 成绩下滑预警（如果有期中和期末成绩）
            if 'Midterm_Score' in df.columns and 'Final_Score' in df.columns:
                declining_students = df[df['Final_Score'] < df['Midterm_Score'] - 10]
                for _, student in declining_students.iterrows():
                    score_drop = student['Midterm_Score'] - student['Final_Score']
                    warnings.append({
                        "type": "成绩下滑预警",
                        "student_id": student.get('Student_ID', 'Unknown'),
                        "student_name": f"{student.get('First_Name', '')} {student.get('Last_Name', '')}".strip(),
                        "message": f"成绩显著下滑 (下降{score_drop:.1f}分)",
                        "priority": "中",
                        "score_drop": score_drop
                    })

        return warnings

    def _generate_attendance_warnings(self, df) -> list:
        """生成出勤相关预警"""
        warnings = []

        if 'Attendance (%)' in df.columns:
            low_attendance = df[df['Attendance (%)'] < 80]
            for _, student in low_attendance.iterrows():
                priority = "高" if student['Attendance (%)'] < 60 else "中"
                warnings.append({
                    "type": "出勤预警",
                    "student_id": student.get('Student_ID', 'Unknown'),
                    "student_name": f"{student.get('First_Name', '')} {student.get('Last_Name', '')}".strip(),
                    "message": f"出勤率偏低 ({student['Attendance (%)']:.1f}%)",
                    "priority": priority,
                    "attendance": student['Attendance (%)']
                })

        return warnings

    def _generate_study_time_warnings(self, df) -> list:
        """生成学习时间相关预警"""
        warnings = []

        if 'Study_Hours_per_Week' in df.columns:
            low_study_time = df[df['Study_Hours_per_Week'] < 10]
            for _, student in low_study_time.iterrows():
                warnings.append({
                    "type": "学习时间预警",
                    "student_id": student.get('Student_ID', 'Unknown'),
                    "student_name": f"{student.get('First_Name', '')} {student.get('Last_Name', '')}".strip(),
                    "message": f"每周学习时间不足 ({student['Study_Hours_per_Week']:.1f}小时)",
                    "priority": "中",
                    "study_hours": student['Study_Hours_per_Week']
                })

        return warnings

    def _generate_stress_warnings(self, df) -> list:
        """生成压力水平相关预警"""
        warnings = []

        if 'Stress_Level (1-10)' in df.columns:
            high_stress = df[df['Stress_Level (1-10)'] >= 8]
            for _, student in high_stress.iterrows():
                warnings.append({
                    "type": "压力水平预警",
                    "student_id": student.get('Student_ID', 'Unknown'),
                    "student_name": f"{student.get('First_Name', '')} {student.get('Last_Name', '')}".strip(),
                    "message": f"压力水平过高 ({student['Stress_Level (1-10)']}/10)",
                    "priority": "中",
                    "stress_level": student['Stress_Level (1-10)']
                })

        return warnings

    def _generate_risk_warnings(self, df) -> list:
        """生成风险预测相关预警"""
        warnings = []

        high_risk_students = df[df['Risk_Level'].isin(['高风险', '极高风险'])]
        for _, student in high_risk_students.iterrows():
            priority = "高" if student['Risk_Level'] == '极高风险' else "中"
            warnings.append({
                "type": "学业风险预警",
                "student_id": student.get('Student_ID', 'Unknown'),
                "student_name": f"{student.get('First_Name', '')} {student.get('Last_Name', '')}".strip(),
                "message": f"学业风险等级：{student['Risk_Level']} (概率：{student.get('Risk_Probability', 0):.2%})",
                "priority": priority,
                "risk_level": student['Risk_Level'],
                "risk_probability": student.get('Risk_Probability', 0)
            })

        return warnings

    def export_results(self, export_format='csv', filename=None) -> str:
        """
        导出分析结果

        Args:
            export_format (str): 导出格式 ('csv', 'excel', 'json')
            filename (str): 文件名（可选）

        Returns:
            str: 导出文件的路径
        """
        try:
            if self.system.df is None:
                raise ValueError("没有可导出的数据")

            # 生成默认文件名
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"academic_warning_results_{timestamp}"

            df = self.system.df.copy()

            # 选择要导出的核心列
            export_columns = [
                'Student_ID', 'First_Name', 'Last_Name', 'Total_Score',
                'Attendance (%)', 'Department'
            ]

            # 添加预测结果列（如果存在）
            if 'Risk_Probability' in df.columns:
                export_columns.extend(['Risk_Probability', 'Risk_Level', 'Risk_Prediction'])

            if 'Academic_Risk' in df.columns:
                export_columns.append('Academic_Risk')

            # 过滤存在的列
            available_columns = [col for col in export_columns if col in df.columns]
            export_df = df[available_columns]

            # 根据格式导出
            if export_format.lower() == 'csv':
                filepath = f"{filename}.csv"
                export_df.to_csv(filepath, index=False, encoding='utf-8-sig')

            elif export_format.lower() == 'excel':
                filepath = f"{filename}.xlsx"
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='学生预警结果', index=False)

                    # 如果有预警信息，也导出
                    warnings = self.generate_warnings()
                    if warnings.get('warnings'):
                        warning_df = pd.DataFrame(warnings['warnings'])
                        warning_df.to_excel(writer, sheet_name='预警详情', index=False)

            elif export_format.lower() == 'json':
                filepath = f"{filename}.json"
                export_data = {
                    'export_time': datetime.now().isoformat(),
                    'total_students': len(export_df),
                    'data': export_df.to_dict('records')
                }

                # 添加预警信息
                warnings = self.generate_warnings()
                export_data['warnings'] = warnings

                import json
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

            else:
                raise ValueError(f"不支持的导出格式: {export_format}")

            print(f"结果已导出到: {filepath}")
            print(f"导出记录数: {len(export_df)}")

            return filepath

        except Exception as e:
            print(f"导出结果失败：{e}")
            return None

    # def show_visualizations(self):
    #     """
    #     显示数据可视化图表
    #     调用系统的可视化方法并添加额外的图表
    #     """
    #     try:
    #         print("正在生成可视化图表...")
    #
    #         if self.system.df is None:
    #             print("错误：没有可视化的数据")
    #             return
    #
    #         # 调用系统原有的可视化方法
    #         self.system.generate_visualizations()
    #
    #         # 添加额外的可视化
    #         self._show_additional_visualizations()
    #
    #     except Exception as e:
    #         print(f"显示可视化图表失败：{e}")

    def _show_additional_visualizations(self):
        """显示额外的可视化图表"""
        df = self.system.df

        # 如果有风险预测结果，显示风险分析图
        if 'Risk_Probability' in df.columns and 'Risk_Level' in df.columns:
            self._plot_risk_analysis()

        # 显示相关性热力图
        self._plot_correlation_heatmap()

    def _plot_risk_analysis(self):
        """绘制风险分析图表"""
        df = self.system.df

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('学业风险分析', fontsize=14)

        # 风险概率分布
        axes[0].hist(df['Risk_Probability'], bins=20, alpha=0.7, color='orange')
        axes[0].axvline(x=0.5, color='red', linestyle='--', label='风险阈值')
        axes[0].set_title('风险概率分布')
        axes[0].set_xlabel('风险概率')
        axes[0].set_ylabel('学生人数')
        axes[0].legend()

        # 各系别风险分布
        if 'Department' in df.columns:
            risk_by_dept = df.groupby('Department')['Risk_Probability'].mean().sort_values(ascending=False)
            axes[1].bar(range(len(risk_by_dept)), risk_by_dept.values)
            axes[1].set_title('各系别平均风险概率')
            axes[1].set_xlabel('系别')
            axes[1].set_ylabel('平均风险概率')
            axes[1].set_xticks(range(len(risk_by_dept)))
            axes[1].set_xticklabels(risk_by_dept.index, rotation=45)

        plt.tight_layout()
        plt.show()

    def _plot_correlation_heatmap(self):
        """绘制相关性热力图"""
        df = self.system.df

        # 选择数值型列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numeric_cols].corr()

            # 只显示重要的相关性
            mask = np.abs(correlation_matrix) < 0.1

            sns.heatmap(correlation_matrix,
                        annot=True,
                        cmap='coolwarm',
                        center=0,
                        fmt='.2f',
                        mask=mask,
                        square=True)
            plt.title('特征相关性热力图')
            plt.tight_layout()
            plt.show()


    def update_predictions_in_db(self, db):
        return self.system.update_predictions_in_db(db)

    # def build_model(self):
    #     return self.system.build_model()
    #
    # def load_data(self, temp_path, db):
    #     self.system.load_data(temp_path, db)


