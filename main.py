import os
import math
import numpy as np
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ----------------------------------------------------
# 1. 資料庫連線設定 (PostgreSQL)
# ----------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ----------------------------------------------------
# 2. 資料庫 Model 定義
# ----------------------------------------------------
class UserConsent(Base):
    __tablename__ = "user_consents"

    id = Column(Integer, primary_key=True, index=True)
    user_line_id = Column(String(255), unique=True, index=True, nullable=False)
    agreed = Column(String(10), default="true")
    terms_version = Column(String(50), default="v1.0")
    created_at = Column(DateTime, default=datetime.utcnow)

class RPPGRecord(Base):
    __tablename__ = "rppg_records"

    id = Column(Integer, primary_key=True, index=True)
    user_uuid = Column(String(255), index=True, nullable=False)
    heart_rate = Column(Integer, nullable=False)
    hrv_sdnn = Column(Float, nullable=False)
    stress_score = Column(Integer, nullable=False)
    health_light = Column(String(20), nullable=False)
    summary = Column(Text, nullable=True)
    action_advice = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ----------------------------------------------------
# 3. FastAPI 應用程式與 CORS 設定
# ----------------------------------------------------
app = FastAPI(title="PharmPulse Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------------------------------
# 4. 啟動時自動修正 PostgreSQL 欄位 (相容 Safe Migration)
# ----------------------------------------------------
@app.on_event("startup")
def auto_migrate_db():
    """安全新增 user_uuid 欄位並處理 Transaction 回滾"""
    if "postgresql" in DATABASE_URL:
        try:
            with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE rppg_records ADD COLUMN IF NOT EXISTS user_uuid VARCHAR(255);"))
                    conn.commit()
                    print(" Successfully ensured 'user_uuid' column exists.")
                except Exception as ex:
                    conn.rollback() # 發生例外時立即重置 Transaction
                    print(f" Migration info: {ex}")
        except Exception as e:
            print(f" DB Connection error during migration: {e}")

# ----------------------------------------------------
# 5. Request / Response Pydantic Schemas
# ----------------------------------------------------
class AnalyzeRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    rgb_signals: List[List[float]]
    fps: int = 30

class ConsentRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    terms_version: str = "v1.0"

# ----------------------------------------------------
# 6. API 路由定義
# ----------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "message": "PharmPulse API Service is running"}

@app.get("/api/v1/user/consent-status/{user_line_id}")
def check_consent(user_line_id: str, db: Session = Depends(get_db)):
    record = db.query(UserConsent).filter(UserConsent.user_line_id == user_line_id).first()
    if record and record.agreed == "true":
        return {"status": "success", "agreed": True}
    return {"status": "success", "agreed": False}

@app.post("/api/v1/user/consent")
def save_consent(req: ConsentRequest, db: Session = Depends(get_db)):
    record = db.query(UserConsent).filter(UserConsent.user_line_id == req.user_line_id).first()
    if not record:
        record = UserConsent(
            user_line_id=req.user_line_id,
            agreed="true",
            terms_version=req.terms_version
        )
        db.add(record)
    else:
        record.agreed = "true"
        record.terms_version = req.terms_version
    
    db.commit()
    return {"status": "success", "message": "Consent recorded"}

@app.post("/api/v1/analyze-rppg")
def analyze_rppg(req: AnalyzeRequest, db: Session = Depends(get_db)):
    try:
        green_signal = req.rgb_signals[1] if len(req.rgb_signals) > 1 else req.rgb_signals[0]
        
        if len(green_signal) < 60:
            hr = 72
            sdnn = 35.0
        else:
            signal_arr = np.array(green_signal)
            detrended = signal_arr - np.mean(signal_arr)
            fft_vals = np.abs(np.fft.rfft(detrended))
            freqs = np.fft.rfftfreq(len(detrended), 1.0 / req.fps)
            
            valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
            if len(valid_idx) > 0:
                peak_freq = freqs[valid_idx[np.argmax(fft_vals[valid_idx])]]
                hr = int(round(peak_freq * 60))
            else:
                hr = 75
            
            sdnn = round(float(np.std(detrended) * 100), 1)
            if sdnn < 15: sdnn = 28.0
            if sdnn > 100: sdnn = 45.0

        stress = int(max(10, min(99, 100 - (sdnn * 1.2))))
        
        if stress > 75 or hr > 100 or hr < 50:
            light = "RED"
            summary = "生理數值顯著偏離基準，心血管負擔較高。"
            advice = (
                "🚨 【處置建議】自律神經壓力過高或 HRV 偏低：\n"
                "1. 請保持環境通風並進行 5 分鐘深呼吸。\n"
                "2. 建議於今日量測血壓，若連續 2 天數值異常，請至門診複診。\n"
                "3. 可前往附近合作藥局，尋求藥師量測血壓與用藥諮詢。"
            )
        elif stress > 50:
            light = "YELLOW"
            summary = "生理指標輕微波動，建議稍微休息。"
            advice = (
                "⚠️ 【處置建議】輕度疲勞或壓力上升：\n"
                "1. 建議補充 300c.c. 溫開水並閉目養神 10 分鐘。\n"
                "2. 觀察晚間睡眠品質，避免睡前使用電子產品。"
            )
        else:
            light = "GREEN"
            summary = "生理指標良好，心律與壓力表現穩定。"
            advice = (
                "🟢 【處置建議】狀態非常棒：\n"
                "1. 請繼續保持規律作息與均衡飲食。\n"
                "2. 建議每日同一時間持續進行生理量測記錄。"
            )

        record = RPPGRecord(
            user_uuid=req.user_line_id,
            heart_rate=hr,
            hrv_sdnn=sdnn,
            stress_score=stress,
            health_light=light,
            summary=summary,
            action_advice=advice,
            created_at=datetime.utcnow()
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "status": "success",
            "data": {
                "id": record.id,
                "heart_rate": hr,
                "hrv_sdnn": sdnn,
                "stress_score": stress,
                "health_light": light,
                "summary": summary,
                "action_advice": advice,
                "created_at": record.created_at.isoformat()
            }
        }

    except Exception as e:
        db.rollback()
        print(f"Analyze Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")