from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 建立 SQLite 資料庫檔案 (PharmPulse.db)
SQLALCHEMY_DATABASE_URL = "sqlite:///./pharmpulse_v2.db"

# connect_args={"check_same_thread": False} 是 SQLite 在 FastAPI 中非同步呼叫時需要的設定
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 取得 DB Session 的 Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()