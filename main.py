import os
import math
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ----------------------------------------------------
# 1. 資料庫連線設定 (PostgreSQL / SQLite)
# ----------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLite 需要 connect_args 的特殊設定，PostgreSQL 則加上 pool_pre_ping 確保連線活著
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

# ----------------------------------------------------
# 3. 自動檢查與修復 PostgreSQL 欄位 (Lifespan 管理)
# ----------------------------------------------------
def auto_migrate_db():
    if "postgresql" in DATABASE_URL:
        try:
            with engine.connect() as conn:
                # 1. 檢查並補齊 user_consents 的 id 欄位
                try:
                    conn.execute(text("ALTER TABLE user_consents ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY;"))
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    print(f"[Migration Warning] user_consents table check: {e}")

                # 2. 檢查並補齊 rppg_records 的欄位
                columns_to_add = [
                    ("user_uuid", "VARCHAR(255)"),
                    ("summary", "TEXT"),
                    ("action_advice", "TEXT")
                ]
                for col_name, col_type in columns_to_add:
                    try:
                        conn.execute(text(f"ALTER TABLE rppg_records ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        print(f"[Migration Warning] Add column {col_name}: {e}")
        except Exception as e:
            print(f"[Migration Error] Database connection failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服務啟動時執行 Migration
    auto_migrate_db()
    yield
    # 服務關閉時可清理資源 (若有需要)

# ----------------------------------------------------
# 4. FastAPI 應用程式與 CORS 設定
# ----------------------------------------------------
app = FastAPI(title="PharmPulse Backend API", version="1.0.0", lifespan=lifespan)

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
# 5. 前端頁面託管路由
# ----------------------------------------------------
@app.get("/")
@app.get("/liff")
def serve_liff():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "message": "PharmPulse API Service is running"}

# ----------------------------------------------------
# 6. Request / Response Pydantic Schemas
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
# 7. API 路由定義
# ----------------------------------------------------
@app.get("/api/v1/user/consent-status/{user_line_id}")
def check_consent(user_line_id: str, db: Session = Depends(get_db)):
    try:
        record = db.query(UserConsent).filter(UserConsent.user_line_id == user_line_id).first()
        if record and record.agreed == "true":
            return {"status": "success", "agreed": True}
        return {"status": "success", "agreed": False}
    except Exception as e:
        return {"status": "error", "agreed": True, "message": str(e)}

@app.post("/api/v1/user/consent")
def save_consent(req: ConsentRequest, db: Session = Depends(get_db)):
    try:
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
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/analyze-rppg")
def analyze_rppg(req: AnalyzeRequest, db: Session = Depends(get_db)):
    try:
        if not req.rgb_signals:
            raise HTTPException(status_code=400, detail="rgb_signals cannot be empty")

        # 優先取綠光 (G Channel)，若僅一維則取第 0 個
        green_signal = req.rgb_signals[1] if len(req.rgb_signals) > 1 else req.rgb_signals[0]
        
        if len(green_signal) < 60:
            hr = 72
            sdnn = 35.0
        else:
            signal_arr = np.array(green_signal, dtype=float)
            detrended = signal_arr - np.mean(signal_arr)
            
            # 訊號標準差計算（預防零變異數情況）
            std_val = float(np.std(detrended))
            if np.isnan(std_val) or std_val == 0:
                sdnn = 35.0
            else:
                sdnn = round(std_val * 100, 1)

            # FFT 頻譜分析求心率 (限制在 0.75 Hz ~ 2.5 Hz，對應 45 BPM ~ 150 BPM)
            fft_vals = np.abs(np.fft.rfft(detrended))
            freqs = np.fft.rfftfreq(len(detrended), 1.0 / req.fps)
            
            valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
            if len(valid_idx) > 0:
                peak_freq = freqs[valid_idx[np.argmax(fft_vals[valid_idx])]]
                hr = int(round(peak_freq * 60))
            else:
                hr = 75
            
            # SDNN 合理值邊界修剪
            sdnn = max(15.0, min(100.0, sdnn))

        # 壓力分數評估
        stress = int(max(10, min(99, 100 - (sdnn * 1.2))))
        
        # 健康燈號評估邏輯
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
            created_at=datetime.now(timezone.utc)
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

# ----------------------------------------------------
# 8. 歷史紀錄 API
# ----------------------------------------------------
@app.get("/api/v1/user/history/{user_line_id}")
def get_user_history(user_line_id: str, limit: int = 10, db: Session = Depends(get_db)):
    try:
        records = db.query(RPPGRecord)\
                    .filter(RPPGRecord.user_uuid == user_line_id)\
                    .order_by(RPPGRecord.created_at.desc())\
                    .limit(limit)\
                    .all()
        
        result = []
        for r in records:
            result.append({
                "id": r.id,
                "heart_rate": r.heart_rate,
                "hrv_sdnn": r.hrv_sdnn,
                "stress_score": r.stress_score,
                "health_light": r.health_light,
                "summary": r.summary,
                "action_advice": r.action_advice,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })
        return {"status": "success", "history": result}
    except Exception as e:
        return {"status": "error", "history": [], "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)