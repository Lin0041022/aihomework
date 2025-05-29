from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

# MySQL数据库连接URL格式：mysql+pymysql://用户名:密码@主机:端口/数据库名
DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/student_mis?charset=utf8mb4"

# 创建引擎
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# 创建会话工厂（线程安全）
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
