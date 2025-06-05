"""
学业预警系统 - 业务层
Business Layer for Academic Warning System

将业务逻辑从数据处理层分离出来，为接口层提供服务
"""

from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy.orm import Session
import os

# from backend.dataprocess.data_processor import AcademicWarningSystem, AcademicWarningSystemExtended
from backend.database.db import SessionLocal


class AcademicWarningBusinessService:
    """学业预警系统业务服务层"""

    def __init__(self):
        self.core_system = AcademicWarningSystem()
        self.extended_system = AcademicWarningSystemExtended(self.core_system)
        self.current_import_id = None

    def get_db_session(self):
        """获取数据库会话"""
        return SessionLocal()

    # ==================== 数据管理业务 ====================

    def load_new_data_from_file(self, file_path: str, db: Session) -> Dict[str, Any]:
        """
        从文件加载新数据的业务逻辑

        Args:
            file_path: 文件路径
            db: 数据库会话

        Returns:
            Dict: 加载结果信息
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "message": "文件不存在",
                    "error_code": "FILE_NOT_FOUND"
                }

            # 加载数据
            success = self.core_system.load_data(file_path, db)

            if success:
                self.current_import_id = self.core_system.import_id
                return {
                    "success": True,
                    "message": "数据加载成功",
                    "import_id": self.current_import_id,
                    "row_count": len(self.core_system.df) if self.core_system.df is not None else 0,
                    "columns": self.core_system.df.columns.tolist() if self.core_system.df is not None else []
                }
            else:
                return {
                    "success": False,
                    "message": "数据加载失败",
                    "error_code": "LOAD_FAILED"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"加载数据时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    def load_historical_data(self, import_id: str, db: Session) -> Dict[str, Any]:
        """
        加载历史数据的业务逻辑

        Args:
            import_id: 导入记录ID
            db: 数据库会话

        Returns:
            Dict: 加载结果信息
        """
        try:
            success = self.extended_system.load_data_by_import_id(import_id, db)

            if success:
                self.current_import_id = import_id
                return {
                    "success": True,
                    "message": "历史数据加载成功",
                    "import_id": import_id,
                    "row_count": len(self.core_system.df) if self.core_system.df is not None else 0,
                    "data_overview": self._get_data_overview()
                }
            else:
                return {
                    "success": False,
                    "message": "历史数据加载失败",
                    "error_code": "LOAD_FAILED"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"加载历史数据时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    def get_import_records_list(self, db: Session) -> Dict[str, Any]:
        """
        获取导入记录列表的业务逻辑

        Args:
            db: 数据库会话

        Returns:
            Dict: 导入记录列表
        """
        try:
            records = self.core_system.get_import_records(db)

            # 格式化记录
            formatted_records = []
            for record in records:
                formatted_records.append({
                    "import_id": record[0],
                    "import_time": record[1].strftime("%Y-%m-%d %H:%M:%S") if record[1] else "",
                    "file_path": record[2],
                    "row_count": record[3],
                    "description": record[4] or ""
                })

            return {
                "success": True,
                "records": formatted_records,
                "total_count": len(formatted_records)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"获取导入记录失败: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    # ==================== 数据处理业务 ====================

    def execute_data_preprocessing(self) -> Dict[str, Any]:
        """
        执行数据预处理的业务逻辑

        Returns:
            Dict: 预处理结果信息
        """
        try:
            if self.core_system.df is None:
                return {
                    "success": False,
                    "message": "没有可处理的数据，请先加载数据",
                    "error_code": "NO_DATA"
                }

            # 记录预处理前的数据状态
            before_shape = self.core_system.df.shape
            before_null_count = self.core_system.df.isnull().sum().sum()

            # 执行预处理
            success = self.extended_system.preprocess_data()

            if success:
                # 记录预处理后的数据状态
                after_shape = self.core_system.df.shape
                after_null_count = self.core_system.df.isnull().sum().sum()

                return {
                    "success": True,
                    "message": "数据预处理完成",
                    "preprocessing_summary": {
                        "before_shape": before_shape,
                        "after_shape": after_shape,
                        "before_null_count": int(before_null_count),
                        "after_null_count": int(after_null_count),
                        "feature_count": len(self.core_system.feature_columns),
                        "features": self.core_system.feature_columns
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "数据预处理失败",
                    "error_code": "PREPROCESSING_FAILED"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"数据预处理时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    def build_prediction_model(self) -> Dict[str, Any]:
        """
        构建预测模型的业务逻辑

        Returns:
            Dict: 模型构建结果信息
        """
        try:
            if self.core_system.df is None:
                return {
                    "success": False,
                    "message": "没有可用的数据，请先加载并预处理数据",
                    "error_code": "NO_DATA"
                }

            if not self.core_system.feature_columns:
                return {
                    "success": False,
                    "message": "特征列未准备好，请先执行数据预处理",
                    "error_code": "NO_FEATURES"
                }

            # 构建模型
            model_results = self.core_system.build_model()

            if self.core_system.model is not None:
                # 准备模型性能信息
                model_info = {
                    "model_type": type(self.core_system.model).__name__,
                    "feature_count": len(self.core_system.feature_columns),
                    "training_data_size": len(self.core_system.df),
                    "optimal_threshold": self.core_system.optimal_threshold
                }

                # 如果有特征重要性
                if hasattr(self.core_system.model, 'feature_importances_'):
                    feature_importance = list(zip(
                        self.core_system.feature_columns,
                        self.core_system.model.feature_importances_
                    ))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    model_info["top_features"] = feature_importance[:10]

                return {
                    "success": True,
                    "message": "模型构建完成",
                    "model_info": model_info
                }
            else:
                return {
                    "success": False,
                    "message": "模型构建失败",
                    "error_code": "MODEL_BUILD_FAILED"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"模型构建时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    # ==================== 预警业务 ====================

    def generate_academic_warnings(self, db: Session) -> Dict[str, Any]:
        """
        生成学业预警的业务逻辑

        Args:
            db: 数据库会话

        Returns:
            Dict: 预警生成结果
        """
        try:
            if self.core_system.df is None:
                return {
                    "success": False,
                    "message": "没有可用的数据，请先加载数据",
                    "error_code": "NO_DATA"
                }

            # 先进行风险预测（如果模型已构建）
            if self.core_system.model is not None:
                predictions = self.core_system.predict_risk()
                if predictions is not None:
                    # 更新数据库中的预测结果
                    self.extended_system.update_predictions_in_db(db)

            # 生成预警信息
            warnings_data = self.extended_system.generate_warnings()

            if "error" in warnings_data:
                return {
                    "success": False,
                    "message": f"生成预警失败: {warnings_data['error']}",
                    "error_code": "WARNING_GENERATION_FAILED"
                }

            # 分析预警统计
            warning_stats = self._analyze_warnings(warnings_data["warnings"])

            return {
                "success": True,
                "message": "预警生成成功",
                "warning_summary": {
                    "total_students": warnings_data["total_students"],
                    "total_warnings": warnings_data["warning_count"],
                    "high_priority_warnings": warnings_data["high_priority_count"],
                    "warning_types": warning_stats["warning_types"],
                    "department_risks": warning_stats.get("department_risks", {}),
                    "generated_at": warnings_data["timestamp"]
                },
                "warnings": warnings_data["warnings"]
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"生成预警时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    def get_student_risk_details(self, student_id: str) -> Dict[str, Any]:
        """
        获取特定学生的风险详情

        Args:
            student_id: 学生ID

        Returns:
            Dict: 学生风险详情
        """
        try:
            if self.core_system.df is None:
                return {
                    "success": False,
                    "message": "没有可用的数据",
                    "error_code": "NO_DATA"
                }

            # 查找学生
            student_data = self.core_system.df[
                self.core_system.df['Student_ID'].astype(str) == str(student_id)
                ]

            if student_data.empty:
                return {
                    "success": False,
                    "message": "未找到指定学生",
                    "error_code": "STUDENT_NOT_FOUND"
                }

            student = student_data.iloc[0]

            # 准备学生详情
            student_details = {
                "student_id": str(student['Student_ID']),
                "name": f"{student.get('First_Name', '')} {student.get('Last_Name', '')}".strip(),
                "department": student.get('Department', ''),
                "basic_info": {
                    "total_score": float(student.get('Total_Score', 0)),
                    "attendance": float(student.get('Attendance (%)', 0)),
                    "age": int(student.get('Age', 0)),
                    "gender": student.get('Gender', '')
                },
                "academic_performance": {
                    "midterm_score": float(student.get('Midterm_Score', 0)),
                    "final_score": float(student.get('Final_Score', 0)),
                    "assignments_avg": float(student.get('Assignments_Avg', 0)),
                    "quizzes_avg": float(student.get('Quizzes_Avg', 0)),
                    "projects_score": float(student.get('Projects_Score', 0)),
                    "participation_score": float(student.get('Participation_Score', 0))
                },
                "lifestyle_factors": {
                    "study_hours_per_week": float(student.get('Study_Hours_per_Week', 0)),
                    "sleep_hours_per_night": float(student.get('Sleep_Hours_per_Night', 0)),
                    "stress_level": int(student.get('Stress_Level (1-10)', 0)),
                    "extracurricular_activities": student.get('Extracurricular_Activities', ''),
                    "internet_access": student.get('Internet_Access_at_Home', '')
                }
            }

            # 添加风险评估信息（如果有）
            if 'Risk_Probability' in student:
                student_details["risk_assessment"] = {
                    "risk_probability": float(student.get('Risk_Probability', 0)),
                    "risk_level": student.get('Risk_Level', ''),
                    "academic_risk": int(student.get('Academic_Risk', 0))
                }

            return {
                "success": True,
                "student_details": student_details
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"获取学生详情时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    # ==================== 导出业务 ====================

    def export_analysis_results(self, export_format: str = 'excel',
                                include_warnings: bool = True) -> Dict[str, Any]:
        """
        导出分析结果的业务逻辑

        Args:
            export_format: 导出格式 ('csv', 'excel', 'json')
            include_warnings: 是否包含预警信息

        Returns:
            Dict: 导出结果信息
        """
        try:
            if self.core_system.df is None:
                return {
                    "success": False,
                    "message": "没有可导出的数据",
                    "error_code": "NO_DATA"
                }

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"academic_warning_export_{timestamp}"

            # 导出数据
            export_path = self.extended_system.export_results(
                export_format=export_format,
                filename=filename
            )

            if export_path:
                # 获取文件信息
                file_size = os.path.getsize(export_path) if os.path.exists(export_path) else 0

                return {
                    "success": True,
                    "message": "数据导出成功",
                    "export_info": {
                        "file_path": export_path,
                        "file_name": os.path.basename(export_path),
                        "file_size": file_size,
                        "export_format": export_format,
                        "export_time": datetime.now().isoformat(),
                        "record_count": len(self.core_system.df)
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "数据导出失败",
                    "error_code": "EXPORT_FAILED"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"导出数据时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    # ==================== 可视化业务 ====================

    # def generate_visualization_data(self) -> Dict[str, Any]:
    #     """
    #     生成可视化数据的业务逻辑
    #
    #     Returns:
    #         Dict: 可视化数据
    #     """
    #     try:
    #         if self.core_system.df is None:
    #             return {
    #                 "success": False,
    #                 "message": "没有可用的数据",
    #                 "error_code": "NO_DATA"
    #             }
    #
    #         df = self.core_system.df
    #
    #         # 准备可视化数据
    #         visualization_data = {
    #             "basic_statistics": {
    #                 "total_students": len(df),
    #                 "average_score": float(df['Total_Score'].mean()) if 'Total_Score' in df.columns else 0,
    #                 "passing_rate": float((df['Total_Score'] >= 60).mean() * 100) if 'Total_Score' in df.columns else 0,
    #                 "average_attendance": float(df['Attendance (%)'].mean()) if 'Attendance (%)' in df.columns else 0
    #             },
    #             "score_distribution": self._get_score_distribution_data(df),
    #             "department_analysis": self._get_department_analysis_data(df),
    #             "risk_analysis": self._get_risk_analysis_data(df) if 'Risk_Level' in df.columns else None,
    #             "correlation_data": self._get_correlation_data(df)
    #         }
    #
    #         return {
    #             "success": True,
    #             "visualization_data": visualization_data
    #         }
    #
    #     except Exception as e:
    #         return {
    #             "success": False,
    #             "message": f"生成可视化数据时发生错误: {str(e)}",
    #             "error_code": "EXCEPTION",
    #             "error_details": str(e)
    #         }

    import matplotlib
    matplotlib.use('TkAgg')
    def generate_visualization_data(self, save_image: bool = False) -> Dict[str, Any]:
        """
        生成可视化数据的业务逻辑

        Args:
            save_image: 是否保存为图像文件

        Returns:
            Dict: 可视化数据或图像文件路径
        """
        try:
            if self.core_system.df is None:
                return {
                    "success": False,
                    "message": "没有可用的数据",
                    "error_code": "NO_DATA"
                }

            df = self.core_system.df

            # 准备可视化数据（JSON 格式）
            visualization_data = {
                "basic_statistics": {
                    "total_students": len(df),
                    "average_score": float(df['Total_Score'].mean()) if 'Total_Score' in df.columns else 0,
                    "passing_rate": float((df['Total_Score'] >= 60).mean() * 100) if 'Total_Score' in df.columns else 0,
                    "average_attendance": float(df['Attendance (%)'].mean()) if 'Attendance (%)' in df.columns else 0
                },
                "score_distribution": self._get_score_distribution_data(df),
                "department_analysis": self._get_department_analysis_data(df),
                "risk_analysis": self._get_risk_analysis_data(df) if 'Risk_Level' in df.columns else None,
                "correlation_data": self._get_correlation_data(df)
            }

            # 如果需要生成图像
            image_path = None
            if save_image:
                # 创建 2x3 子图布局
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle('学业预警系统 - 数据分析报告', fontsize=16)

                # 子图 1: 总成绩分布（直方图）
                axes[0, 0].hist(df['Total_Score'], bins=20, alpha=0.7, color='skyblue')
                axes[0, 0].axvline(x=60, color='red', linestyle='--', label='及格线')
                axes[0, 0].set_title('总成绩分布')
                axes[0, 0].set_xlabel('总成绩')
                axes[0, 0].set_ylabel('学生人数')
                axes[0, 0].legend()

                # 子图 2: 学业风险等级分布（饼图）
                if 'Risk_Level' in df.columns:
                    risk_counts = df['Risk_Level'].value_counts()
                    axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
                    axes[0, 1].set_title('学业风险等级分布')

                # 子图 3: 出勤率与总成绩关系（散点图）
                axes[0, 2].scatter(df['Attendance (%)'], df['Total_Score'], alpha=0.6)
                axes[0, 2].set_title('出勤率与总成绩关系')
                axes[0, 2].set_xlabel('出勤率 (%)')
                axes[0, 2].set_ylabel('总成绩')

                # 子图 4: 各系别成绩分布（箱线图）
                if 'Department' in df.columns:
                    dept_scores = []
                    dept_names = []
                    for dept in df['Department'].unique():
                        dept_data = df[df['Department'] == dept]['Total_Score']
                        dept_scores.append(dept_data)
                        dept_names.append(dept)
                    axes[1, 0].boxplot(dept_scores, labels=dept_names)
                    axes[1, 0].set_title('各系别成绩分布')
                    axes[1, 0].tick_params(axis='x', rotation=45)

                # 子图 5: 每周学习时间与成绩关系（散点图）
                axes[1, 1].scatter(df['Study_Hours_per_Week'], df['Total_Score'], alpha=0.6)
                axes[1, 1].set_title('每周学习时间与成绩关系')
                axes[1, 1].set_xlabel('每周学习时间(小时)')
                axes[1, 1].set_ylabel('总成绩')

                # 子图 6: 压力水平与平均成绩关系（柱状图）
                stress_avg = df.groupby('Stress_Level (1-10)')['Total_Score'].mean()
                axes[1, 2].bar(stress_avg.index, stress_avg.values)
                axes[1, 2].set_title('压力水平与平均成绩关系')
                axes[1, 2].set_xlabel('压力水平')
                axes[1, 2].set_ylabel('平均成绩')

                # 调整布局
                plt.tight_layout()

                # 保存图像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"visualizations/academic_report_{timestamp}.png"
                os.makedirs("visualizations", exist_ok=True)
                plt.savefig(image_path)
                plt.close(fig)  # 关闭图形以释放内存

                visualization_data["image_path"] = image_path

            return {
                "success": True,
                "visualization_data": visualization_data
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"生成可视化数据时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }
    # ==================== 系统状态业务 ====================

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息

        Returns:
            Dict: 系统状态信息
        """
        try:
            status = {
                "data_loaded": self.core_system.df is not None,
                "current_import_id": self.current_import_id,
                "model_built": self.core_system.model is not None,
                "features_prepared": len(self.core_system.feature_columns) > 0,
                "system_ready": False
            }

            if status["data_loaded"]:
                status["data_info"] = {
                    "row_count": len(self.core_system.df),
                    "column_count": len(self.core_system.df.columns),
                    "has_predictions": 'Risk_Probability' in self.core_system.df.columns
                }

            # 判断系统是否就绪
            status["system_ready"] = all([
                status["data_loaded"],
                status["features_prepared"],
                status["model_built"]
            ])

            return {
                "success": True,
                "status": status
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"获取系统状态时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    # ==================== 辅助方法 ====================

    def _get_data_overview(self) -> Dict[str, Any]:
        """获取数据概览"""
        if self.core_system.df is None:
            return {}

        df = self.core_system.df
        return {
            "total_records": len(df),
            "columns_count": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "average_score": float(df['Total_Score'].mean()) if 'Total_Score' in df.columns else 0,
            "departments": df['Department'].unique().tolist() if 'Department' in df.columns else []
        }

    def _analyze_warnings(self, warnings: List[Dict]) -> Dict[str, Any]:
        """分析预警统计"""
        warning_types = {}
        department_risks = {}

        for warning in warnings:
            # 统计预警类型
            warning_type = warning.get("type", "未知")
            warning_types[warning_type] = warning_types.get(warning_type, 0) + 1

            # 统计系别风险（如果有系别信息）
            if "department" in warning:
                dept = warning["department"]
                department_risks[dept] = department_risks.get(dept, 0) + 1

        return {
            "warning_types": warning_types,
            "department_risks": department_risks
        }

    def _get_score_distribution_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取分数分布数据"""
        if 'Total_Score' not in df.columns:
            return {}

        scores = df['Total_Score']
        return {
            "bins": [0, 40, 50, 60, 70, 80, 90, 100],
            "counts": [
                int((scores < 40).sum()),
                int(((scores >= 40) & (scores < 50)).sum()),
                int(((scores >= 50) & (scores < 60)).sum()),
                int(((scores >= 60) & (scores < 70)).sum()),
                int(((scores >= 70) & (scores < 80)).sum()),
                int(((scores >= 80) & (scores < 90)).sum()),
                int((scores >= 90).sum())
            ]
        }

    def _get_department_analysis_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取系别分析数据"""
        if 'Department' not in df.columns or 'Total_Score' not in df.columns:
            return {}

        dept_stats = df.groupby('Department')['Total_Score'].agg(['mean', 'count']).reset_index()

        return {
            "departments": dept_stats['Department'].tolist(),
            "average_scores": dept_stats['mean'].round(2).tolist(),
            "student_counts": dept_stats['count'].tolist()
        }

    def _get_risk_analysis_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取风险分析数据"""
        if 'Risk_Level' not in df.columns:
            return {}

        risk_counts = df['Risk_Level'].value_counts()

        return {
            "risk_levels": risk_counts.index.tolist(),
            "counts": risk_counts.values.tolist(),
            "percentages": (risk_counts / len(df) * 100).round(2).tolist()
        }

    def _get_correlation_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取相关性数据"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {}

        # 选择主要的数值列
        main_cols = ['Total_Score', 'Attendance (%)', 'Study_Hours_per_Week',
                     'Sleep_Hours_per_Night', 'Stress_Level (1-10)']
        available_cols = [col for col in main_cols if col in numeric_cols]

        if len(available_cols) < 2:
            return {}

        corr_matrix = df[available_cols].corr()

        return {
            "features": available_cols,
            "correlation_matrix": corr_matrix.round(3).to_dict()
        }