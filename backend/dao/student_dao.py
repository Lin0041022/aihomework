from typing import List

from backend.models.student_models import HouseData, ImportRecord, Base
from sqlalchemy.orm import Session, sessionmaker
from backend.database.db import engine
from contextlib import contextmanager
from datetime import datetime
import pandas as pd
from sqlalchemy_utils import database_exists, create_database

class HouseDAO:
    """房价数据分析系统数据访问对象"""

    def __init__(self):
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.init_database()

    @contextmanager
    def get_db_session(self):
        db = self.SessionLocal()
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
            if not database_exists(engine.url):
                create_database(engine.url)
                print("数据库 house_price_analysis 创建成功！")
            else:
                print("数据库已存在，无需创建。")
            Base.metadata.create_all(bind=engine)
            print("表结构初始化成功！")
        except Exception as e:
            print(f"数据库初始化失败：{e}")
            raise

    def save_import_record(self, new_record: ImportRecord):
        with self.get_db_session() as db:
            db.add(new_record)

    def save_house_batch(self, df, import_id: str):
        house_entries = []
        for _, row in df.iterrows():
            house_entries.append(HouseData(
                import_id=import_id,
                selected=row.get('甄选', ''),
                description=row.get('介绍', ''),
                house_type=row.get('户型', ''),
                area_sqm=float(str(row.get('平方', '0')).replace('平米', '').strip()),
                orientation=row.get('朝向', ''),
                floor_info=row.get('楼层', ''),
                decoration=row.get('装修', ''),
                location=row.get('位置', ''),
                attention_count=int(row.get('关注量', 0)),
                view_count=int(row.get('看房量', 0)),
                publish_time=row.get('发布时间', ''),
                price=float(str(row.get('价格', '0')).replace('万', '').strip()),
                price_per_sqm=float(str(row.get('平方价格', '0')).replace('元/m²', '').strip()),
                remarks=row.get('备注', ''),
                # 分析结果字段
                district=row.get('district', ''),
                area_name=row.get('area_name', ''),
                room_count=row.get('room_count', 0),
                living_room_count=row.get('living_room_count', 0),
                floor_level=row.get('floor_level', ''),
                total_floors=row.get('total_floors', 0),
                subway_distance=row.get('subway_distance', 9999.0),
                price_level=row.get('price_level', ''),
                location_score=row.get('location_score', 0),
                value_score=row.get('value_score', 0),
                investment_potential=row.get('investment_potential', 0)
            ))
        with self.get_db_session() as db:
            db.bulk_save_objects(house_entries)

    def get_houses_by_import_id(self, import_id: str) -> list[dict]:
        with self.get_db_session() as db:
            houses = db.query(HouseData).filter(HouseData.import_id == import_id).all()
            return [{
                'selected': h.selected,
                'description': h.description,
                'house_type': h.house_type,
                'area_sqm': h.area_sqm,
                'orientation': h.orientation,
                'floor_info': h.floor_info,
                'decoration': h.decoration,
                'location': h.location,
                'attention_count': h.attention_count,
                'view_count': h.view_count,
                'publish_time': h.publish_time,
                'price': h.price,
                'price_per_sqm': h.price_per_sqm,
                'remarks': h.remarks,
                'district': h.district,
                'area_name': h.area_name,
                'room_count': h.room_count,
                'living_room_count': h.living_room_count,
                'floor_level': h.floor_level,
                'total_floors': h.total_floors,
                'subway_distance': h.subway_distance,
                'price_level': h.price_level,
                'location_score': h.location_score,
                'value_score': h.value_score,
                'investment_potential': h.investment_potential
            } for h in houses]

    def get_import_records(self):
        try:
            with self.get_db_session() as db:
                records = db.query(ImportRecord).order_by(ImportRecord.import_time.desc()).all()
                return [(r.import_id, r.import_time, r.file_path, r.row_count, r.description) for r in records]
        except Exception as e:
            print(f"获取导入记录失败: {e}")
            return []

    def update_analysis_results(self, df, import_id: str):
        try:
            with self.get_db_session() as db:
                for _, row in df.iterrows():
                    db.query(HouseData).filter(
                        HouseData.import_id == import_id,
                        HouseData.location == row.get('位置', '')
                    ).update({
                        HouseData.price_level: row.get('price_level', ''),
                        HouseData.location_score: row.get('location_score', 0),
                        HouseData.value_score: row.get('value_score', 0),
                        HouseData.investment_potential: row.get('investment_potential', 0)
                    })
            return True
        except Exception as e:
            print(f"更新分析结果失败: {e}")
            return False
