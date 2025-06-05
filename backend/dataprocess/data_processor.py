# academic_warning_system.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from datetime import datetime
import time
import warnings
from backend.dao.student_dao import AcademicDAO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib

from backend.models.student_models import ImportRecord

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class AcademicWarningSystem:
    """学业预警系统主类"""

    def __init__(self):
        """
        初始化学业预警系统

        Args:
            dao: 数据访问对象，用于数据库操作
        """
        self.dao = AcademicDAO()
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.optimal_threshold = 0.5
        self.import_id = None

    def load_data(self, file_path: str) -> bool:
        """
        加载数据文件

        Args:
            file_path (str): 数据文件路径

        Returns:
            bool: 是否成功加载
        """
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

            # 通过DAO保存数据
            self.dao.save_import_record(new_record)
            self.dao.save_student_batch(self.df, self.import_id)

            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def load_data_from_db(self, import_id: str) -> bool:
        """
        从数据库加载数据

        Args:
            import_id (str): 导入ID

        Returns:
            bool: 是否成功加载
        """
        try:
            students_data = self.dao.get_students_by_import_id(import_id)
            if not students_data:
                print(f"未找到导入ID为 {import_id} 的记录")
                return False

            # 转换为DataFrame
            self.df = pd.DataFrame(students_data)
            self.import_id = import_id

            print(f"从数据库加载数据成功！导入ID: {import_id}，共有 {len(self.df)} 条记录")
            return True
        except Exception as e:
            print(f"从数据库加载数据失败: {e}")
            return False

    def get_import_records(self):
        """获取所有导入记录"""
        return self.dao.get_import_records()

    def preprocess_data(self) -> bool:
        """
        数据预处理主函数

        Returns:
            bool: 预处理是否成功
        """
        try:
            print("开始数据预处理...")

            if self.df is None:
                print("错误：请先加载数据")
                return False

            # 数据清洗
            if not self._clean_data():
                print("数据清洗失败")
                return False

            # 特征工程
            self._prepare_features()

            # 数据验证
            self._validate_data()

            print("数据预处理完成！")
            return True

        except Exception as e:
            print(f"数据预处理失败：{e}")
            return False

    def _clean_data(self) -> bool:
        """数据清洗"""
        print("开始数据清洗...")

        # 处理缺失值
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)

        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()
                fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                self.df[col].fillna(fill_value, inplace=True)

        # 异常值检测和处理
        self._detect_and_handle_anomalies()

        # 生成目标变量
        self.df['Academic_Risk'] = (self.df['Total_Score'] < 60).astype(int)

        # 更新数据库
        if self.dao.update_academic_risk(self.df, self.import_id):
            print("数据清洗完成！")

        return True

    def _detect_and_handle_anomalies(self):
        """异常值检测和处理"""
        print("检测和处理异常值...")

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
                    anomaly_count += len(outliers)
                    self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)

        print(f"总共处理了 {anomaly_count} 个异常值")

    def _prepare_features(self):
        """特征工程"""
        print("开始特征工程...")

        # 创建组合特征
        self.df['Exam_Average'] = (self.df['Midterm_Score'] + self.df['Final_Score']) / 2
        self.df['Coursework_Average'] = (self.df['Assignments_Avg'] + self.df['Quizzes_Avg'] +
                                         self.df['Projects_Score']) / 3
        self.df['Score_Volatility'] = abs(self.df['Midterm_Score'] - self.df['Final_Score'])
        self.df['Performance_Gap'] = self.df['Exam_Average'] - self.df['Coursework_Average']
        self.df['Academic_Engagement'] = (self.df['Attendance (%)'] * 0.3 +
                                          self.df['Participation_Score'] * 0.7)
        self.df['Study_Efficiency'] = self.df['Total_Score'] / (self.df['Study_Hours_per_Week'] + 1)
        self.df['Rest_Study_Balance'] = self.df['Sleep_Hours_per_Night'] / (self.df['Study_Hours_per_Week'] + 1)

        # 创建标志特征
        self.df['High_Stress_Flag'] = (self.df['Stress_Level (1-10)'] >= 7).astype(int)
        self.df['Low_Attendance_Flag'] = (self.df['Attendance (%)'] < 80).astype(int)
        self.df['Poor_Sleep_Flag'] = (self.df['Sleep_Hours_per_Night'] < 6).astype(int)

        # 选择核心特征
        core_features = [
            'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg',
            'Quizzes_Avg', 'Participation_Score', 'Projects_Score',
            'Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Age', 'Stress_Level (1-10)',
            'Exam_Average', 'Coursework_Average', 'Score_Volatility', 'Performance_Gap',
            'Academic_Engagement', 'Study_Efficiency', 'Rest_Study_Balance',
            'High_Stress_Flag', 'Low_Attendance_Flag', 'Poor_Sleep_Flag'
        ]

        # 处理分类特征
        categorical_features = ['Gender', 'Department']
        for col in categorical_features:
            if col in self.df.columns:
                unique_vals = self.df[col].unique()
                if len(unique_vals) <= 5:
                    le = LabelEncoder()
                    self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                    self.label_encoders[col] = le
                    core_features.append(col + '_encoded')

        # 处理二元特征
        binary_cols = ['Extracurricular_Activities', 'Internet_Access_at_Home']
        for col in binary_cols:
            if col in self.df.columns:
                self.df[col + '_binary'] = (self.df[col] == 'Yes').astype(int)
                core_features.append(col + '_binary')

        # 确定最终特征列
        self.feature_columns = [col for col in core_features if col in self.df.columns]

        print(f"选择了 {len(self.feature_columns)} 个特征")

    def _validate_data(self):
        """验证预处理后的数据质量"""
        print("\n=== 数据预处理验证 ===")
        print(f"最终数据维度: {self.df.shape}")
        print(f"缺失值数量: {self.df.isnull().sum().sum()}")
        print(f"特征列数量: {len(self.feature_columns)}")

        if 'Academic_Risk' in self.df.columns:
            risk_dist = self.df['Academic_Risk'].value_counts()
            print(f"目标变量分布: {risk_dist.to_dict()}")

    def build_model(self):
        """构建预警模型"""
        print("\n=== 构建预警模型 ===")

        X = self.df[self.feature_columns]
        y = self.df['Academic_Risk']

        print(f"特征矩阵形状: {X.shape}")
        print(f"类别分布:\n{y.value_counts()}")

        # 处理类别不平衡
        if y.sum() < 10:
            print("调整风险阈值以增加正样本数量...")
            risk_threshold = self.df['Total_Score'].quantile(0.25)
            self.df['Academic_Risk'] = (self.df['Total_Score'] < risk_threshold).astype(int)
            y = self.df['Academic_Risk']

        # 数据清理
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 模型训练和比较
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

        print("\n=== 模型比较 ===")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            auc_score = roc_auc_score(y_test, y_pred_proba)

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
        print(f"\n=== 最佳模型: {best_model_name} (F1: {best_f1:.4f}) ===")

        # 最终评估
        y_pred_final = self.model.predict(X_test)
        print("\n=== 最终模型性能 ===")
        print(classification_report(y_test, y_pred_final, target_names=['正常', '风险']))

        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\n=== 特征重要性 TOP 10 ===")
            for i, row in feature_importance.head(10).iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")

        return X_test, y_test, y_pred_final, self.model.predict_proba(X_test)[:, 1]

    def predict_risk(self):
        """预测学业风险"""
        if self.model is None:
            print("请先构建模型")
            return None

        X = self.df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        risk_predictions = (predictions >= self.optimal_threshold).astype(int)

        self.df['Risk_Probability'] = predictions
        self.df['Risk_Prediction'] = risk_predictions

        # 风险等级分类
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

        # 更新数据库
        self.dao.update_predictions(self.df, self.import_id)

        return self.df[['Student_ID', 'First_Name', 'Last_Name',
                        'Total_Score', 'Risk_Probability', 'Risk_Level', 'Risk_Prediction']]

    def generate_warnings(self) -> dict:
        """
        生成预警信息

        Returns:
            dict: 包含各类预警信息的字典
        """
        try:
            print("正在生成预警信息...")

            if self.df is None:
                return {"error": "未加载数据"}

            df = self.df
            warnings = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_students": len(df),
                "warnings": []
            }

            # 成绩预警
            score_warnings = self._generate_score_warnings()
            warnings["warnings"].extend(score_warnings)

            # 出勤预警
            attendance_warnings = self._generate_attendance_warnings()
            warnings["warnings"].extend(attendance_warnings)

            # 学习时间预警
            study_time_warnings = self._generate_study_time_warnings()
            warnings["warnings"].extend(study_time_warnings)

            # 压力水平预警
            stress_warnings = self._generate_stress_warnings()
            warnings["warnings"].extend(stress_warnings)

            # 如果已经进行了风险预测
            if 'Risk_Level' in df.columns:
                risk_warnings = self._generate_risk_warnings()
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

    def _generate_score_warnings(self) -> list:
        """生成成绩相关预警"""
        warnings = []

        if 'Total_Score' in self.df.columns:
            # 不及格预警
            failing_students = self.df[self.df['Total_Score'] < 60]
            for _, student in failing_students.iterrows():
                warnings.append({
                    "type": "成绩预警",
                    "student_id": student.get('Student_ID', 'Unknown'),
                    "student_name": f"{student.get('First_Name', '')} {student.get('Last_Name', '')}".strip(),
                    "message": f"总成绩不及格 ({student['Total_Score']:.1f}分)",
                    "priority": "高",
                    "score": student['Total_Score']
                })

            # 成绩下滑预警
            if 'Midterm_Score' in self.df.columns and 'Final_Score' in self.df.columns:
                declining_students = self.df[self.df['Final_Score'] < self.df['Midterm_Score'] - 10]
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

    def _generate_attendance_warnings(self) -> list:
        """生成出勤相关预警"""
        warnings = []

        if 'Attendance (%)' in self.df.columns:
            low_attendance = self.df[self.df['Attendance (%)'] < 80]
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

    def _generate_study_time_warnings(self) -> list:
        """生成学习时间相关预警"""
        warnings = []

        if 'Study_Hours_per_Week' in self.df.columns:
            low_study_time = self.df[self.df['Study_Hours_per_Week'] < 10]
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

    def _generate_stress_warnings(self) -> list:
        """生成压力水平相关预警"""
        warnings = []

        if 'Stress_Level (1-10)' in self.df.columns:
            high_stress = self.df[self.df['Stress_Level (1-10)'] >= 8]
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

    def _generate_risk_warnings(self) -> list:
        """生成风险预测相关预警"""
        warnings = []

        high_risk_students = self.df[self.df['Risk_Level'].isin(['高风险', '极高风险'])]
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
        """导出分析结果"""
        try:
            if self.df is None:
                raise ValueError("没有可导出的数据")

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"academic_warning_results_{timestamp}"

            # 选择导出列
            export_columns = [
                'Student_ID', 'First_Name', 'Last_Name', 'Total_Score',
                'Attendance (%)', 'Department'
            ]

            if 'Risk_Probability' in self.df.columns:
                export_columns.extend(['Risk_Probability', 'Risk_Level', 'Risk_Prediction'])

            if 'Academic_Risk' in self.df.columns:
                export_columns.append('Academic_Risk')

            available_columns = [col for col in export_columns if col in self.df.columns]
            export_df = self.df[available_columns]

            # 根据格式导出
            if export_format.lower() == 'csv':
                filepath = f"{filename}.csv"
                export_df.to_csv(filepath, index=False, encoding='utf-8-sig')

            elif export_format.lower() == 'excel':
                filepath = f"{filename}.xlsx"
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='学生预警结果', index=False)

                    warnings = self.generate_warnings()
                    if warnings.get('warnings'):
                        warning_df = pd.DataFrame(warnings['warnings'])
                        warning_df.to_excel(writer, sheet_name='预警详情', index=False)

            elif export_format.lower() == 'json':
                filepath = f"{filename}.json"
                export_data = {
                    'export_time': datetime.now().isoformat(),
                    'total_students': len(export_df),
                    'data': export_df.to_dict('records'),
                    'warnings': self.generate_warnings()
                }

                import json
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

            print(f"结果已导出到: {filepath}")
            return filepath

        except Exception as e:
            print(f"导出结果失败：{e}")
            return None

    def show_statistics(self):
        """显示统计信息"""
        if self.df is None:
            print("没有可显示的数据")
            return

        print("\n=== 学业预警统计 ===")
        total_students = len(self.df)

        if 'Academic_Risk' in self.df.columns:
            at_risk_students = self.df['Academic_Risk'].sum()
            print(f"总学生数: {total_students}")
            print(f"预警学生数: {at_risk_students}")
            print(f"预警比例: {at_risk_students / total_students * 100:.2f}%")

        if 'Risk_Level' in self.df.columns:
            print("\n风险等级分布:")
            print(self.df['Risk_Level'].value_counts())

        if 'Total_Score' in self.df.columns:
            print(f"\n成绩统计:")
            print(f"平均成绩: {self.df['Total_Score'].mean():.2f}")
            print(f"及格率: {(self.df['Total_Score'] >= 60).mean() * 100:.1f}%")