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

class HouseData(Base):
    __tablename__ = "house_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    import_id = Column(String(50), ForeignKey("import_records.import_id"))

    # CSV原始字段
    selected = Column(String(10))  # 甄选
    description = Column(Text)  # 介绍
    house_type = Column(String(100))  # 户型
    area_sqm = Column(Float)  # 平方
    orientation = Column(String(50))  # 朝向
    floor_info = Column(String(100))  # 楼层
    decoration = Column(String(100))  # 装修
    location = Column(String(255))  # 位置
    attention_count = Column(Integer)  # 关注量
    view_count = Column(Integer)  # 看房量
    publish_time = Column(String(100))  # 发布时间
    price = Column(Float)  # 价格(万元)
    price_per_sqm = Column(Float)  # 平方价格(元/m²)
    remarks = Column(Text)  # 备注
    
    # 分析结果字段
    district = Column(String(100))  # 区域（从位置中提取）
    area_name = Column(String(100))  # 小区名（从位置中提取）
    room_count = Column(Integer)  # 房间数（从户型中提取）
    living_room_count = Column(Integer)  # 客厅数（从户型中提取）
    floor_level = Column(String(50))  # 楼层等级（低/中/高）
    total_floors = Column(Integer)  # 总楼层数
    subway_distance = Column(Float)  # 距离地铁站距离(米)
    price_level = Column(String(50))  # 价格等级：低、中、高
    location_score = Column(Float)  # 位置评分
    value_score = Column(Float)  # 性价比评分
    investment_potential = Column(Float)  # 投资潜力评分
