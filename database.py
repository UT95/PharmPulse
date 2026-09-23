import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 讀取 Render 設定的 DATABASE_URL，若無則預設使用本地 SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pharmpulse.db")

# 修正 Render PostgreSQL 網址開頭 (SQLAlchemy 1.4+ 需要 postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLite 需要 connect_args，PostgreSQL 不需要
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
