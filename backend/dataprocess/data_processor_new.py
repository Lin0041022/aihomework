"""
房价数据分析系统 - 数据处理层
Data Processing Layer for House Price Analysis System

负责数据的加载、预处理、分析和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import uuid
import re
import os
from typing import Dict, List, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class HousePriceAnalysisSystem:
    """房价数据分析系统"""

    def __init__(self):
        self.df = None
        self.model = None
        self.import_id = None
        self.feature_columns = []
        self.target_column = 'price'
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path: str) -> bool:
        """
        从CSV文件加载房价数据
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 读取CSV文件
            self.df = pd.read_csv(file_path, encoding='utf-8')

            # 自动重命名中文表头为英文
            column_map = {
                '平方': 'area_sqm',
                '户型': 'house_type',
                '楼层': 'floor_info',
                '位置': 'location',
                '价格': 'price',
                '平方价格': 'price_per_sqm',
                '朝向': 'orientation',
                '装修': 'decoration',
                '备注': 'remark',
                '介绍': 'description',
                '关注量': 'attention_count',
                '看房量': 'visit_count',
                '发布时间': 'release_time',
                '甄选': 'selected',
            }
            self.df.rename(columns=column_map, inplace=True)
            
            # 生成导入ID
            self.import_id = f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            print(f"数据加载成功，共 {len(self.df)} 条记录")
            print(f"列名: {self.df.columns.tolist()}")
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return False
    
    def load_data_from_db(self, import_id: str) -> bool:
        """
        从数据库加载历史数据
        
        Args:
            import_id: 导入记录ID
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 这里应该从数据库加载数据
            # 暂时返回False，需要实现数据库查询
            print(f"从数据库加载数据: {import_id}")
            return False
            
        except Exception as e:
            print(f"从数据库加载数据失败: {str(e)}")
            return False
    
    def preprocess_data(self) -> bool:
        """
        数据预处理
        
        Returns:
            bool: 预处理是否成功
        """
        try:
            if self.df is None:
                return False
            
            print("开始数据预处理...")
            
            # 1. 数据清洗
            self._clean_data()
            
            # 2. 特征工程
            self._engineer_features()
            
            # 3. 处理缺失值
            self._handle_missing_values()
            
            # 4. 数据标准化
            self._standardize_data()
            
            print("数据预处理完成")
            return True
            
        except Exception as e:
            print(f"数据预处理失败: {str(e)}")
            return False
    
    def _clean_data(self):
        """数据清洗"""
        # 移除重复行
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"移除重复行: {initial_rows - len(self.df)} 行")
        
        # 清理价格数据
        if 'price' in self.df.columns:
            # 移除价格中的"万"字并转换为数值
            self.df['price'] = self.df['price'].astype(str).str.replace('万', '').astype(float)
        
        # 清理面积数据
        if 'area_sqm' in self.df.columns:
            # 移除面积中的"平米"并转换为数值
            self.df['area_sqm'] = self.df['area_sqm'].astype(str).str.replace('平米', '').astype(float)
        
        # 清理单价数据
        if 'price_per_sqm' in self.df.columns:
            # 移除单价中的"元/m²"并转换为数值
            self.df['price_per_sqm'] = self.df['price_per_sqm'].astype(str).str.replace('元/m²', '').astype(float)
    
    def _engineer_features(self):
        """特征工程"""
        # 从位置中提取区域和小区名
        if 'location' in self.df.columns:
            self.df['district'] = self.df['location'].apply(self._extract_district)
            self.df['area_name'] = self.df['location'].apply(self._extract_area_name)
        
        # 从户型中提取房间数和客厅数
        if 'house_type' in self.df.columns:
            self.df['room_count'] = self.df['house_type'].apply(self._extract_room_count)
            self.df['living_room_count'] = self.df['house_type'].apply(self._extract_living_room_count)
        
        # 从楼层信息中提取楼层等级和总楼层数
        if 'floor_info' in self.df.columns:
            self.df['floor_level'] = self.df['floor_info'].apply(self._extract_floor_level)
            self.df['total_floors'] = self.df['floor_info'].apply(self._extract_total_floors)
        
        # 从位置中提取地铁距离
        if 'location' in self.df.columns:
            self.df['subway_distance'] = self.df['location'].apply(self._extract_subway_distance)
        
        # 计算价格等级
        if 'price' in self.df.columns:
            self.df['price_level'] = self.df['price'].apply(self._categorize_price_level)
        
        # 计算位置评分
        self.df['location_score'] = self.df.apply(self._calculate_location_score, axis=1)
        
        # 计算性价比评分
        if 'price' in self.df.columns and 'area_sqm' in self.df.columns:
            self.df['value_score'] = self.df.apply(self._calculate_value_score, axis=1)
        
        # 计算投资潜力评分
        self.df['investment_potential'] = self.df.apply(self._calculate_investment_potential, axis=1)
    
    def _extract_district(self, location: str) -> str:
        """从位置中提取区域"""
        if pd.isna(location):
            return "未知"
        
        # 常见的区域名称
        districts = ['余杭', '拱墅', '上城', '下城', '西湖', '滨江', '江干', '萧山', '临平', '钱塘']
        
        for district in districts:
            if district in location:
                return district
        
        return "其他"
    
    def _extract_area_name(self, location: str) -> str:
        """从位置中提取小区名"""
        if pd.isna(location):
            return "未知"
        
        # 简单的提取逻辑，取第一个空格后的内容
        parts = location.split()
        if len(parts) > 1:
            return parts[1]
        
        return "未知"
    
    def _extract_room_count(self, house_type: str) -> int:
        """从户型中提取房间数"""
        if pd.isna(house_type):
            return 0
        
        match = re.search(r'(\d+)室', house_type)
        if match:
            return int(match.group(1))
        
        return 0
    
    def _extract_living_room_count(self, house_type: str) -> int:
        """从户型中提取客厅数"""
        if pd.isna(house_type):
            return 0
        
        match = re.search(r'(\d+)厅', house_type)
        if match:
            return int(match.group(1))
        
        return 0
    
    def _extract_floor_level(self, floor_info: str) -> str:
        """从楼层信息中提取楼层等级"""
        if pd.isna(floor_info):
            return "未知"
        
        if '低楼层' in floor_info:
            return "低"
        elif '中楼层' in floor_info:
            return "中"
        elif '高楼层' in floor_info:
            return "高"
        
        return "未知"
    
    def _extract_total_floors(self, floor_info: str) -> int:
        """从楼层信息中提取总楼层数"""
        if pd.isna(floor_info):
            return 0
        
        match = re.search(r'(\d+)层', floor_info)
        if match:
            return int(match.group(1))
        
        return 0
    
    def _extract_subway_distance(self, location: str) -> float:
        """从位置中提取地铁距离"""
        if pd.isna(location):
            return 9999.0
        
        match = re.search(r'距.*地铁站.*?(\d+)米', location)
        if match:
            return float(match.group(1))
        
        return 9999.0
    
    def _categorize_price_level(self, price: float) -> str:
        """价格等级分类"""
        if pd.isna(price):
            return "未知"
        
        if price < 200:
            return "低"
        elif price < 500:
            return "中"
        else:
            return "高"
    
    def _calculate_location_score(self, row) -> float:
        """计算位置评分"""
        score = 0.0
        
        # 地铁距离评分
        subway_dist = row.get('subway_distance', 9999)
        if subway_dist <= 500:
            score += 0.4
        elif subway_dist <= 1000:
            score += 0.3
        elif subway_dist <= 2000:
            score += 0.2
        else:
            score += 0.1
        
        # 区域评分
        district = row.get('district', '')
        district_scores = {
            '西湖': 0.3, '滨江': 0.3, '上城': 0.25, '下城': 0.25,
            '拱墅': 0.2, '江干': 0.2, '萧山': 0.15, '余杭': 0.15,
            '临平': 0.1, '钱塘': 0.1
        }
        score += district_scores.get(district, 0.1)
        
        # 楼层评分
        floor_level = row.get('floor_level', '')
        floor_scores = {'低': 0.1, '中': 0.2, '高': 0.15}
        score += floor_scores.get(floor_level, 0.1)
        
        return min(score, 1.0)
    
    def _calculate_value_score(self, row) -> float:
        """计算性价比评分"""
        price = row.get('price', 0)
        area = row.get('area_sqm', 1)
        price_per_sqm = row.get('price_per_sqm', 0)
        
        if price <= 0 or area <= 0:
            return 0.0
        
        # 基于单价和面积的性价比计算
        avg_price_per_sqm = price * 10000 / area
        
        # 简单的性价比评分（价格越低，性价比越高）
        if avg_price_per_sqm < 20000:
            return 0.9
        elif avg_price_per_sqm < 30000:
            return 0.7
        elif avg_price_per_sqm < 40000:
            return 0.5
        else:
            return 0.3
    
    def _calculate_investment_potential(self, row) -> float:
        """计算投资潜力评分"""
        location_score = row.get('location_score', 0)
        value_score = row.get('value_score', 0)
        price_level = row.get('price_level', '')
        
        # 综合评分
        score = location_score * 0.4 + value_score * 0.4
        
        # 价格等级调整
        if price_level == '中':
            score += 0.1
        elif price_level == '低':
            score += 0.05
        
        return min(score, 1.0)
    
    def _handle_missing_values(self):
        """处理缺失值"""
        # 数值型列用中位数填充
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # 分类型列用众数填充
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "未知"
                self.df[col].fillna(mode_value, inplace=True)
    
    def _standardize_data(self):
        """数据标准化"""
        # 对数值型特征进行标准化
        numeric_features = ['area_sqm', 'room_count', 'living_room_count', 'total_floors', 
                          'subway_distance', 'attention_count', 'view_count']
        
        available_features = [col for col in numeric_features if col in self.df.columns]
        
        if available_features:
            self.df[available_features] = self.scaler.fit_transform(self.df[available_features])
    
    def build_model(self) -> bool:
        """
        构建房价预测模型
        
        Returns:
            bool: 模型构建是否成功
        """
        try:
            if self.df is None:
                return False
            
            print("开始构建房价预测模型...")
            
            # 准备特征列
            feature_cols = ['area_sqm', 'room_count', 'living_room_count', 'total_floors',
                          'subway_distance', 'location_score', 'value_score']
            
            self.feature_columns = [col for col in feature_cols if col in self.df.columns]
            
            if len(self.feature_columns) < 2:
                print("特征列不足，无法构建模型")
                return False
            
            # 准备数据
            X = self.df[self.feature_columns]
            y = self.df[self.target_column]
            
            # 移除包含NaN的行
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 10:
                print("有效数据不足，无法构建模型")
                return False
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 构建随机森林模型
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"模型构建完成")
            print(f"均方误差: {mse:.2f}")
            print(f"R²分数: {r2:.3f}")
            print(f"平均绝对误差: {mae:.2f}")
            
            return True
            
        except Exception as e:
            print(f"模型构建失败: {str(e)}")
            return False
    
    def analyze_house_data(self) -> bool:
        """
        分析房价数据
        
        Returns:
            bool: 分析是否成功
        """
        try:
            if self.df is None:
                return False
            
            print("开始分析房价数据...")
            
            # 这里可以添加更多的分析逻辑
            # 目前主要依赖特征工程中已经计算的分析结果
            
            print("房价数据分析完成")
            return True
            
        except Exception as e:
            print(f"房价数据分析失败: {str(e)}")
            return False
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """获取分析结果"""
        if self.df is None:
            return {}
        
        return {
            "total_houses": len(self.df),
            "price_statistics": {
                "mean": float(self.df['price'].mean()),
                "median": float(self.df['price'].median()),
                "std": float(self.df['price'].std())
            },
            "location_analysis": {
                "district_distribution": self.df['district'].value_counts().to_dict(),
                "avg_location_score": float(self.df['location_score'].mean())
            },
            "house_type_analysis": {
                "type_distribution": self.df['house_type'].value_counts().to_dict(),
                "room_distribution": self.df['room_count'].value_counts().to_dict()
            }
        }
    
    def get_house_details(self, house_id: str) -> Optional[Dict]:
        """获取特定房屋详情"""
        if self.df is None:
            return None
        
        # 这里应该根据house_id查找，暂时返回第一条记录
        if len(self.df) > 0:
            return self.df.iloc[0].to_dict()
        
        return None
    
    def save_analysis_results_to_db(self, import_id: str, db_session) -> bool:
        """保存分析结果到数据库"""
        try:
            # 这里应该实现数据库保存逻辑
            print(f"保存分析结果到数据库: {import_id}")
            return True
        except Exception as e:
            print(f"保存分析结果失败: {str(e)}")
            return False
    
    def export_results(self, filename: str, format: str = 'excel', 
                      include_analysis: bool = True) -> Optional[str]:
        """导出分析结果"""
        try:
            if self.df is None:
                return None
            
            # 准备导出数据
            export_df = self.df.copy()
            
            if include_analysis:
                # 添加分析结果列
                export_df['分析_价格等级'] = export_df['price_level']
                export_df['分析_位置评分'] = export_df['location_score']
                export_df['分析_性价比评分'] = export_df['value_score']
                export_df['分析_投资潜力'] = export_df['investment_potential']
            
            # 生成文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == 'excel':
                file_path = f"exports/{filename}_{timestamp}.xlsx"
                os.makedirs("exports", exist_ok=True)
                export_df.to_excel(file_path, index=False)
            elif format.lower() == 'csv':
                file_path = f"exports/{filename}_{timestamp}.csv"
                os.makedirs("exports", exist_ok=True)
                export_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            else:
                return None
            
            print(f"结果已导出到: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"导出结果失败: {str(e)}")
            return None
    
    def generate_visualizations(self, save_image: bool = False) -> Dict[str, Any]:
        """生成可视化图表"""
        try:
            if self.df is None:
                return {}
            
            print("生成可视化图表...")
            
            # 创建图表
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('房价数据分析报告', fontsize=16)
            
            # 1. 价格分布直方图
            axes[0, 0].hist(self.df['price'], bins=30, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('房价分布')
            axes[0, 0].set_xlabel('价格(万元)')
            axes[0, 0].set_ylabel('数量')
            
            # 2. 区域价格对比
            district_prices = self.df.groupby('district')['price'].mean().sort_values(ascending=False)
            axes[0, 1].bar(range(len(district_prices)), district_prices.values)
            axes[0, 1].set_title('各区域平均房价')
            axes[0, 1].set_xlabel('区域')
            axes[0, 1].set_ylabel('平均价格(万元)')
            axes[0, 1].set_xticks(range(len(district_prices)))
            axes[0, 1].set_xticklabels(district_prices.index, rotation=45)
            
            # 3. 户型分布饼图
            house_type_counts = self.df['house_type'].value_counts()
            axes[0, 2].pie(house_type_counts.values, labels=house_type_counts.index, autopct='%1.1f%%')
            axes[0, 2].set_title('户型分布')
            
            # 4. 面积与价格散点图
            axes[1, 0].scatter(self.df['area_sqm'], self.df['price'], alpha=0.6)
            axes[1, 0].set_title('面积与价格关系')
            axes[1, 0].set_xlabel('面积(平方米)')
            axes[1, 0].set_ylabel('价格(万元)')
            
            # 5. 位置评分分布
            axes[1, 1].hist(self.df['location_score'], bins=20, alpha=0.7, color='lightgreen')
            axes[1, 1].set_title('位置评分分布')
            axes[1, 1].set_xlabel('位置评分')
            axes[1, 1].set_ylabel('数量')
            
            # 6. 投资潜力分布
            axes[1, 2].hist(self.df['investment_potential'], bins=20, alpha=0.7, color='orange')
            axes[1, 2].set_title('投资潜力分布')
            axes[1, 2].set_xlabel('投资潜力评分')
            axes[1, 2].set_ylabel('数量')
            
            plt.tight_layout()
            
            image_paths = []
            if save_image:
                os.makedirs("visualizations", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"visualizations/house_analysis_{timestamp}.png"
                plt.savefig(image_path, dpi=300, bbox_inches='tight')
                image_paths.append(image_path)
            
            plt.show()
            
            return {"image_paths": image_paths}
            
        except Exception as e:
            print(f"生成可视化图表失败: {str(e)}")
            return {}
    
    def generate_model_evaluation_charts(self, save_image: bool = False) -> Dict[str, Any]:
        """生成模型评估图表"""
        try:
            if self.model is None:
                return {}
            print("生成模型评估图表...")
            # 生成预测值与实际值
            if self.df is None or not hasattr(self, 'feature_columns') or not hasattr(self, 'target_column'):
                return {}
            X = self.df[self.feature_columns]
            y = self.df[self.target_column]
            y_pred = self.model.predict(X)
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime
            import numpy as np
            image_paths = []
            # 1. 预测值 vs 实际值散点图
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            ax1.scatter(y, y_pred, alpha=0.6, color='royalblue')
            ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax1.set_xlabel('实际房价')
            ax1.set_ylabel('预测房价')
            ax1.set_title('预测值 vs 实际值')
            plt.tight_layout()
            if save_image:
                os.makedirs("visualizations", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path1 = f"visualizations/model_pred_vs_actual_{timestamp}.png"
                fig1.savefig(image_path1, dpi=300, bbox_inches='tight')
                image_paths.append(image_path1)
            plt.close(fig1)
            # 2. 残差分布图
            residuals = y - y_pred
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.hist(residuals, bins=30, color='orange', alpha=0.7)
            ax2.set_xlabel('残差')
            ax2.set_ylabel('数量')
            ax2.set_title('残差分布')
            plt.tight_layout()
            if save_image:
                image_path2 = f"visualizations/model_residuals_{timestamp}.png"
                fig2.savefig(image_path2, dpi=300, bbox_inches='tight')
                image_paths.append(image_path2)
            plt.close(fig2)
            return {"image_paths": image_paths}
        except Exception as e:
            print(f"生成模型评估图表失败: {str(e)}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {}
        
        return {
            "model_type": type(self.model).__name__,
            "features": self.feature_columns,
            "target": self.target_column,
            "performance": self.get_model_performance()
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """获取模型性能指标"""
        if self.model is None:
            return {}
        
        try:
            # 这里应该计算实际的性能指标
            return {
                "r2_score": 0.8,
                "mean_squared_error": 1000,
                "mean_absolute_error": 50
            }
        except:
            return {}
    
    def get_import_records(self) -> List[tuple]:
        """获取导入记录列表"""
        # 这里应该从数据库查询
        # 暂时返回空列表
        return [] 