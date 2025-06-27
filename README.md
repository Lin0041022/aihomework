# 房价数据分析系统

## 系统概述

这是一个基于Python的房价数据分析系统，可以对房价数据进行全面的分析，包括位置分析、价格分析和房屋类型分析。

## 功能特性

### 1. 数据管理
- 支持CSV格式的房价数据导入
- 数据预处理和清洗
- 历史数据管理

### 2. 数据分析
- **位置分析**：区域分布、地铁距离、位置评分
- **价格分析**：价格分布、价格等级、性价比分析
- **房屋类型分析**：户型分布、房间数分析、装修状态分析

### 3. 智能评分
- 位置评分：基于地铁距离、区域等因素
- 性价比评分：基于价格和面积的综合评估
- 投资潜力评分：综合多个因素的评分

### 4. 可视化
- 价格分布直方图
- 区域价格对比图
- 户型分布饼图
- 面积与价格散点图
- 位置评分分布图
- 投资潜力分布图

### 5. 预测模型
- 基于随机森林的房价预测模型
- 模型性能评估
- 特征重要性分析

## 数据格式

系统支持以下CSV格式的房价数据：

| 字段名 | 说明 | 示例 |
|--------|------|------|
| 甄选 | 是否甄选 | 否 |
| 介绍 | 房屋介绍 | 余杭老余杭圆乡名筑3室2厅 |
| 户型 | 房屋户型 | 3室2厅 |
| 平方 | 房屋面积 | 91.93 平米 |
| 朝向 | 房屋朝向 | 南 |
| 楼层 | 楼层信息 | 低楼层/6层 |
| 装修 | 装修状态 | 精装 |
| 位置 | 房屋位置 | 老余杭 圆乡名筑 |
| 关注量 | 关注人数 | 0 |
| 看房量 | 看房人数 | 0 |
| 发布时间 | 发布时间 | 1个月 |
| 价格 | 房屋价格 | 225万 |
| 平方价格 | 每平米价格 | 24475元/m² |
| 备注 | 备注信息 | 满二年业主自荐 |

## 安装和运行

### 1. 环境要求
- Python 3.12+
- MySQL数据库

### 2. 安装依赖
```bash
cd backend
uv sync
```

### 3. 配置数据库
在项目根目录创建 `.env` 文件：
```
DATABASE_USERNAME=your_username
DATABASE_PASSWORD=your_password
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_NAME=house_price_analysis
```

### 4. 启动系统
```bash
cd backend
python main.py
```

系统将在 `http://127.0.0.1:8000` 启动

## API接口

### 数据管理接口
- `POST /data/load-new` - 加载新数据
- `POST /data/load-history` - 加载历史数据
- `GET /records/imports` - 获取导入记录

### 数据处理接口
- `POST /data/preprocess` - 数据预处理
- `POST /model/build` - 构建预测模型

### 分析接口
- `POST /analysis/generate` - 生成房价分析
- `GET /house/{house_id}` - 获取房屋详情

### 可视化接口
- `GET /visualizations` - 获取可视化图表
- `GET /visualizations/model-evaluation` - 获取模型评估图表

### 导出接口
- `GET /export/analysis` - 导出分析结果

### 系统接口
- `GET /system/status` - 获取系统状态

## 使用示例

### 1. 加载数据
```bash
curl -X POST "http://127.0.0.1:8000/data/load-new" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_house_data.csv"
```

### 2. 数据预处理
```bash
curl -X POST "http://127.0.0.1:8000/data/preprocess"
```

### 3. 生成分析
```bash
curl -X POST "http://127.0.0.1:8000/analysis/generate"
```

### 4. 获取可视化
```bash
curl -X GET "http://127.0.0.1:8000/visualizations" \
  --output house_analysis.png
```

## 系统架构

```
backend/
├── database/          # 数据库相关
│   └── db.py         # 数据库连接
├── models/           # 数据模型
│   └── student_models.py  # 房价数据模型
├── dataprocess/      # 数据处理
│   └── data_processor_new.py  # 房价分析处理器
├── services/         # 业务服务
│   └── warning_service.py  # 房价分析服务
├── routers/          # API路由
│   └── api.py        # API接口
└── main.py           # 主程序
```

## 技术栈

- **后端框架**: FastAPI
- **数据库**: MySQL + SQLAlchemy
- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn
- **可视化**: Matplotlib, Seaborn
- **依赖管理**: uv

## 注意事项

1. 确保CSV文件编码为UTF-8
2. 价格字段需要包含"万"字
3. 面积字段需要包含"平米"
4. 位置字段中的地铁距离信息需要包含"距"和"米"关键词

## 许可证

MIT License 