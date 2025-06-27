"""
房价数据分析系统 - 业务层
Business Layer for House Price Analysis System

将业务逻辑从数据处理层分离出来，为接口层提供服务
"""

from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.orm import Session
import os
import re
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve, 
                             precision_recall_curve, confusion_matrix)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# from backend.dataprocess.data_processor import HousePriceAnalysisSystem
from backend.dataprocess.data_processor_new import HousePriceAnalysisSystem
from backend.database.db import SessionLocal


class HousePriceAnalysisBusinessService:
    """房价数据分析系统业务服务层"""

    def __init__(self):
        self.system = HousePriceAnalysisSystem()
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
            success = self.system.load_data(file_path)

            if success:
                self.current_import_id = self.system.import_id
                return {
                    "success": True,
                    "message": "数据加载成功",
                    "import_id": self.current_import_id,
                    "row_count": int(len(self.system.df)) if self.system.df is not None else 0,
                    "columns": self.system.df.columns.tolist() if self.system.df is not None else [],
                    "data_overview": self._get_data_overview()
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
            success = self.system.load_data_from_db(import_id)

            if success:
                self.current_import_id = import_id
                return {
                    "success": True,
                    "message": "历史数据加载成功",
                    "import_id": import_id,
                    "row_count": int(len(self.system.df)) if self.system.df is not None else 0,
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
            records = self.system.get_import_records()
            formatted_records = []
            for record in records:
                formatted_records.append({
                    "import_id": record[0],
                    "import_time": record[1].strftime("%Y-%m-%d %H:%M:%S") if record[1] else "",
                    "file_path": record[2],
                    "row_count": int(record[3]) if record[3] is not None else 0,
                    "description": record[4] or ""
                })
            return {
                "success": True,
                "records": formatted_records,
                "total_count": int(len(formatted_records))
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
            if self.system.df is None:
                return {
                    "success": False,
                    "message": "没有可处理的数据，请先加载数据",
                    "error_code": "NO_DATA"
                }
            before_shape = self.system.df.shape
            before_null_count = int(self.system.df.isnull().sum().sum())
            success = self.system.preprocess_data()
            if success:
                after_shape = self.system.df.shape
                after_null_count = int(self.system.df.isnull().sum().sum())
                return {
                    "success": True,
                    "message": "数据预处理完成",
                    "before_shape": tuple(map(int, before_shape)),
                    "after_shape": tuple(map(int, after_shape)),
                    "before_null_count": before_null_count,
                    "after_null_count": after_null_count,
                    "cleaned_rows": int(before_shape[0]) - int(after_shape[0]),
                    "data_overview": self._get_data_overview()
                }
            else:
                return {
                    "success": False,
                    "message": "数据预处理失败",
                    "error_code": "PREPROCESS_FAILED"
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
            if self.system.df is None:
                return {
                    "success": False,
                    "message": "没有可处理的数据，请先加载并预处理数据",
                    "error_code": "NO_DATA"
                }

            # 构建模型
            success = self.system.build_model()

            if success:
                model_info = self.system.get_model_info()
                return {
                    "success": True,
                    "message": "模型构建完成",
                    "model_type": model_info.get("model_type", "未知"),
                    "features": model_info.get("features", []),
                    "target": model_info.get("target", "未知"),
                    "model_performance": model_info.get("performance", {})
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
                "message": f"构建模型时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    def generate_house_analysis(self, db: Session) -> Dict[str, Any]:
        """
        生成房价分析的业务逻辑

        Args:
            db: 数据库会话

        Returns:
            Dict: 分析结果信息
        """
        try:
            if self.system.df is None:
                return {
                    "success": False,
                    "message": "没有可处理的数据，请先加载数据",
                    "error_code": "NO_DATA"
                }

            # 执行分析
            success = self.system.analyze_house_data()

            if success:
                analysis_results = self.system.get_analysis_results()
                
                # 保存分析结果到数据库
                if self.current_import_id:
                    self.system.save_analysis_results_to_db(self.current_import_id, db)

                return {
                    "success": True,
                    "message": "房价分析完成",
                    "total_houses": int(len(self.system.df)),
                    "analysis_summary": {
                        "price_analysis": self._get_price_analysis_data(),
                        "location_analysis": self._get_location_analysis_data(),
                        "house_type_analysis": self._get_house_type_analysis_data(),
                        "investment_recommendations": self._get_investment_recommendations()
                    },
                    "detailed_results": analysis_results
                }
            else:
                return {
                    "success": False,
                    "message": "房价分析失败",
                    "error_code": "ANALYSIS_FAILED"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"生成房价分析时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    def get_house_analysis_details(self, house_id: str) -> Dict[str, Any]:
        """
        获取特定房屋分析详情的业务逻辑

        Args:
            house_id: 房屋ID

        Returns:
            Dict: 房屋分析详情
        """
        try:
            if self.system.df is None:
                return {
                    "success": False,
                    "message": "没有可处理的数据",
                    "error_code": "NO_DATA"
                }

            # 查找房屋数据
            house_data = self.system.get_house_details(house_id)

            if house_data is not None:
                return {
                    "success": True,
                    "house_data": house_data,
                    "analysis": {
                        "price_analysis": self._analyze_house_price(house_data),
                        "location_analysis": self._analyze_house_location(house_data),
                        "investment_analysis": self._analyze_investment_potential(house_data)
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "未找到指定的房屋数据",
                    "error_code": "HOUSE_NOT_FOUND"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"获取房屋分析详情时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    # ==================== 数据导出业务 ====================

    def export_analysis_results(self, export_format: str = 'excel',
                                include_analysis: bool = True) -> Dict[str, Any]:
        """
        导出分析结果的业务逻辑

        Args:
            export_format: 导出格式 ('excel', 'csv', 'json')
            include_analysis: 是否包含分析结果

        Returns:
            Dict: 导出结果信息
        """
        try:
            if self.system.df is None:
                return {
                    "success": False,
                    "message": "没有可导出的数据",
                    "error_code": "NO_DATA"
                }

            # 生成导出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"house_price_analysis_{timestamp}"

            # 执行导出
            export_path = self.system.export_results(
                filename=filename,
                format=export_format,
                include_analysis=include_analysis
            )

            if export_path and os.path.exists(export_path):
                return {
                    "success": True,
                    "message": f"分析结果已导出到: {export_path}",
                    "file_path": export_path,
                    "file_size": os.path.getsize(export_path),
                    "format": export_format
                }
            else:
                return {
                    "success": False,
                    "message": "导出失败",
                    "error_code": "EXPORT_FAILED"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"导出分析结果时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    # ==================== 可视化业务 ====================

    def generate_visualization_data(self, save_image: bool = False) -> Dict[str, Any]:
        """
        生成可视化数据的业务逻辑

        Args:
            save_image: 是否保存图片

        Returns:
            Dict: 可视化数据
        """
        try:
            if self.system.df is None:
                return {
                    "success": False,
                    "message": "没有可可视化的数据",
                    "error_code": "NO_DATA"
                }

            # 生成可视化
            viz_data = self.system.generate_visualizations(save_image=save_image)

            if viz_data:
                return {
                    "success": True,
                    "message": "可视化数据生成完成",
                    "charts": {
                        "price_distribution": self._get_price_distribution_data(),
                        "location_analysis": self._get_location_analysis_data(),
                        "house_type_analysis": self._get_house_type_analysis_data(),
                        "price_trends": self._get_price_trends_data(),
                        "correlation_analysis": self._get_correlation_data()
                    },
                    "image_paths": viz_data.get("image_paths", []) if save_image else []
                }
            else:
                return {
                    "success": False,
                    "message": "可视化数据生成失败",
                    "error_code": "VISUALIZATION_FAILED"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"生成可视化数据时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    def generate_model_evaluation_charts(self, save_image: bool = False) -> Dict[str, Any]:
        """
        生成模型评估图表的业务逻辑

        Args:
            save_image: 是否保存图片

        Returns:
            Dict: 模型评估数据
        """
        try:
            if not self.system.model:
                return {
                    "success": False,
                    "message": "没有可评估的模型，请先构建模型",
                    "error_code": "NO_MODEL"
                }

            # 生成模型评估图表
            eval_data = self.system.generate_model_evaluation_charts(save_image=save_image)

            if eval_data:
                return {
                    "success": True,
                    "message": "模型评估图表生成完成",
                    "evaluation_metrics": self._get_model_performance_data(),
                    "charts": {
                        "prediction_vs_actual": eval_data.get("prediction_vs_actual", {}),
                        "residual_plot": eval_data.get("residual_plot", {}),
                        "feature_importance": eval_data.get("feature_importance", {})
                    },
                    "image_paths": eval_data.get("image_paths", []) if save_image else []
                }
            else:
                return {
                    "success": False,
                    "message": "模型评估图表生成失败",
                    "error_code": "EVALUATION_FAILED"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"生成模型评估图表时发生错误: {str(e)}",
                "error_code": "EXCEPTION",
                "error_details": str(e)
            }

    # ==================== 系统状态业务 ====================

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态的业务逻辑

        Returns:
            Dict: 系统状态信息
        """
        try:
            status = {
                "data_loaded": self.system.df is not None,
                "data_shape": self.system.df.shape if self.system.df is not None else None,
                "model_built": self.system.model is not None,
                "current_import_id": self.current_import_id,
                "system_version": "1.0.0",
                "last_analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            if self.system.df is not None:
                status.update({
                    "data_overview": self._get_data_overview(),
                    "missing_values": self.system.df.isnull().sum().to_dict(),
                    "data_types": self.system.df.dtypes.to_dict()
                })

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

    # ==================== 私有辅助方法 ====================

    def _get_data_overview(self) -> Dict[str, Any]:
        """获取数据概览"""
        if self.system.df is None:
            return {}

        df = self.system.df
        return {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "numeric_columns": int(len(df.select_dtypes(include=[np.number]).columns)),
            "categorical_columns": int(len(df.select_dtypes(include=['object']).columns)),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "price_range": {
                "min": float(df['price'].min()) if 'price' in df.columns else None,
                "max": float(df['price'].max()) if 'price' in df.columns else None,
                "mean": float(df['price'].mean()) if 'price' in df.columns else None
            },
            "area_range": {
                "min": float(df['area_sqm'].min()) if 'area_sqm' in df.columns else None,
                "max": float(df['area_sqm'].max()) if 'area_sqm' in df.columns else None,
                "mean": float(df['area_sqm'].mean()) if 'area_sqm' in df.columns else None
            }
        }

    def _get_price_analysis_data(self) -> Dict[str, Any]:
        """获取价格分析数据"""
        if self.system.df is None:
            return {}

        df = self.system.df
        if 'price' not in df.columns:
            return {}
        price_levels = {}
        if 'price_level' in df.columns:
            price_levels = {str(k): int(v) for k, v in df['price_level'].value_counts().to_dict().items()}
        return {
            "price_statistics": {
                "mean": float(df['price'].mean()),
                "median": float(df['price'].median()),
                "std": float(df['price'].std()),
                "min": float(df['price'].min()),
                "max": float(df['price'].max()),
                "q25": float(df['price'].quantile(0.25)),
                "q75": float(df['price'].quantile(0.75))
            },
            "price_levels": price_levels,
            "price_per_sqm_stats": {
                "mean": float(df['price_per_sqm'].mean()) if 'price_per_sqm' in df.columns else None,
                "median": float(df['price_per_sqm'].median()) if 'price_per_sqm' in df.columns else None
            }
        }

    def _get_location_analysis_data(self) -> Dict[str, Any]:
        """获取位置分析数据"""
        if self.system.df is None:
            return {}

        df = self.system.df
        if 'district' not in df.columns:
            return {}
        district_distribution = {str(k): int(v) for k, v in df['district'].value_counts().to_dict().items()}
        location_scores_distribution = {}
        if 'location_score' in df.columns:
            location_scores_distribution = {str(k): int(v) for k, v in df['location_score'].value_counts(bins=5).to_dict().items()}
        return {
            "district_distribution": district_distribution,
            "location_scores": {
                "mean": float(df['location_score'].mean()) if 'location_score' in df.columns else None,
                "distribution": location_scores_distribution
            },
            "subway_analysis": {
                "near_subway": int(len(df[df['subway_distance'] <= 1000])) if 'subway_distance' in df.columns else 0,
                "far_from_subway": int(len(df[df['subway_distance'] > 1000])) if 'subway_distance' in df.columns else 0
            }
        }

    def _get_house_type_analysis_data(self) -> Dict[str, Any]:
        """获取房屋类型分析数据"""
        if self.system.df is None:
            return {}

        df = self.system.df
        house_type_distribution = {}
        room_count_distribution = {}
        decoration_distribution = {}
        orientation_distribution = {}
        if 'house_type' in df.columns:
            house_type_distribution = {str(k): int(v) for k, v in df['house_type'].value_counts().to_dict().items()}
        if 'room_count' in df.columns:
            room_count_distribution = {str(k): int(v) for k, v in df['room_count'].value_counts().to_dict().items()}
        if 'decoration' in df.columns:
            decoration_distribution = {str(k): int(v) for k, v in df['decoration'].value_counts().to_dict().items()}
        if 'orientation' in df.columns:
            orientation_distribution = {str(k): int(v) for k, v in df['orientation'].value_counts().to_dict().items()}
        return {
            "house_type_distribution": house_type_distribution,
            "room_count_distribution": room_count_distribution,
            "decoration_distribution": decoration_distribution,
            "orientation_distribution": orientation_distribution
        }

    def _get_price_trends_data(self) -> Dict[str, Any]:
        """获取价格趋势数据"""
        if self.system.df is None:
            return {}

        df = self.system.df
        if 'price' not in df.columns or 'district' not in df.columns:
            return {}
        district_price_comparison = {k: {stat: float(vv) for stat, vv in v.items()} for k, v in df.groupby('district')['price'].agg(['mean', 'median', 'count']).to_dict().items()}
        house_type_price_comparison = {}
        if 'house_type' in df.columns:
            house_type_price_comparison = {k: {stat: float(vv) for stat, vv in v.items()} for k, v in df.groupby('house_type')['price'].agg(['mean', 'median', 'count']).to_dict().items()}
        return {
            "district_price_comparison": district_price_comparison,
            "house_type_price_comparison": house_type_price_comparison
        }

    def _get_correlation_data(self) -> Dict[str, Any]:
        """获取相关性分析数据"""
        if self.system.df is None:
            return {}

        df = self.system.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}

        correlation_matrix = df[numeric_cols].corr()
        return {
            "correlation_matrix": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in correlation_matrix.to_dict().items()},
            "price_correlations": {str(k): float(v) for k, v in correlation_matrix['price'].to_dict().items()} if 'price' in correlation_matrix.columns else {}
        }

    def _get_investment_recommendations(self) -> Dict[str, Any]:
        """获取投资建议数据"""
        if self.system.df is None:
            return {}

        df = self.system.df
        if 'investment_potential' not in df.columns:
            return {}

        top_recommendations_df = df.nlargest(10, 'investment_potential')[['description', 'price', 'location', 'investment_potential']]
        top_recommendations = [
            {
                "description": str(row["description"]),
                "price": float(row["price"]),
                "location": str(row["location"]),
                "investment_potential": float(row["investment_potential"])
            } for _, row in top_recommendations_df.iterrows()
        ]
        return {
            "high_potential_houses": int(len(df[df['investment_potential'] >= 0.8])),
            "medium_potential_houses": int(len(df[(df['investment_potential'] >= 0.6) & (df['investment_potential'] < 0.8)])),
            "low_potential_houses": int(len(df[df['investment_potential'] < 0.6])),
            "top_recommendations": top_recommendations
        }

    def _analyze_house_price(self, house_data: Dict) -> Dict[str, Any]:
        """分析单个房屋价格"""
        if not house_data or 'price' not in house_data:
            return {}

        price = house_data['price']
        df = self.system.df

        if df is None or 'price' not in df.columns:
            return {}

        return {
            "price_percentile": float((df['price'] <= price).mean() * 100),
            "price_comparison": {
                "vs_mean": price - float(df['price'].mean()),
                "vs_median": price - float(df['price'].median()),
                "price_level": house_data.get('price_level', '未知')
            }
        }

    def _analyze_house_location(self, house_data: Dict) -> Dict[str, Any]:
        """分析单个房屋位置"""
        if not house_data:
            return {}

        return {
            "location_score": house_data.get('location_score', None),
            "subway_accessibility": "便利" if house_data.get('subway_distance', 9999) <= 1000 else "一般",
            "district_ranking": self._get_district_ranking(house_data.get('district', ''))
        }

    def _analyze_investment_potential(self, house_data: Dict) -> Dict[str, Any]:
        """分析单个房屋投资潜力"""
        if not house_data:
            return {}

        potential = house_data.get('investment_potential', 0)
        return {
            "investment_potential": potential,
            "potential_level": "高" if potential >= 0.8 else "中" if potential >= 0.6 else "低",
            "recommendation": self._get_investment_recommendation(potential)
        }

    def _get_district_ranking(self, district: str) -> str:
        """获取区域排名"""
        if not district or self.system.df is None:
            return "未知"

        df = self.system.df
        if 'district' not in df.columns or 'price' not in df.columns:
            return "未知"

        district_prices = df.groupby('district')['price'].mean().sort_values(ascending=False)
        try:
            district_index_list = [str(idx) for idx in district_prices.index]
            rank = int(district_index_list.index(str(district))) + 1
            total = int(len(district_prices))
            return f"{rank}/{total}"
        except:
            return "未知"

    def _get_investment_recommendation(self, potential: float) -> str:
        """获取投资建议"""
        if potential >= 0.8:
            return "强烈推荐"
        elif potential >= 0.6:
            return "推荐"
        elif potential >= 0.4:
            return "一般"
        else:
            return "不推荐"

    def _get_model_performance_data(self) -> Dict[str, Any]:
        """获取模型性能数据"""
        if not self.system.model:
            return {}

        try:
            return self.system.get_model_performance()
        except:
            return {}

    def _get_price_distribution_data(self) -> Dict[str, Any]:
        """获取价格分布数据"""
        if self.system.df is None:
            return {}

        df = self.system.df
        if 'price' not in df.columns:
            return {}

        return {
            "price_ranges": {
                "0-200万": int(len(df[df['price'] <= 200])),
                "200-500万": int(len(df[(df['price'] > 200) & (df['price'] <= 500)])),
                "500-1000万": int(len(df[(df['price'] > 500) & (df['price'] <= 1000)])),
                "1000万以上": int(len(df[df['price'] > 1000]))
            },
            "price_statistics": {
                "mean": float(df['price'].mean()),
                "median": float(df['price'].median()),
                "std": float(df['price'].std()),
                "min": float(df['price'].min()),
                "max": float(df['price'].max())
            }
        }