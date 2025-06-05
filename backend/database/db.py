from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import os
from dotenv import load_dotenv

import os

load_dotenv(override=True)

USERNAME = os.getenv('DATABASE_USERNAME')
PASSWORD = os.getenv('DATABASE_PASSWORD')
HOST = os.getenv('DATABASE_HOST')
PORT = os.getenv('DATABASE_PORT')
DATABASE_NAME = os.getenv('DATABASE_NAME')

# MySQL数据库连接URL格式：mysql+pymysql://用户名:密码@主机:端口/数据库名
DATABASE_URL = (f"mysql+pymysql://{USERNAME}:{PASSWORD}"
                f"@{HOST}:"
                f"{PORT}/"
                f"{DATABASE_NAME}?charset=utf8mb4")
print(f"+++++++++++++++++++++++{DATABASE_URL}")
# DATABASE_URL = f"mysql+pymysql://root:@localhost:3306/student_mis?charset=utf8mb4"

# 创建引擎
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# 创建会话工厂（线程安全）
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
