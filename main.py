import os
import math
import requests
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import asynccontextmanager
from scipy.signal import butter, filtfilt, find_peaks

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, text, or_
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ----------------------------------------------------
# 1. 環境變數與資料庫連線設定
# ----------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
LINE_CHANNEL_ID = os.getenv("LINE_CHANNEL_ID", "")  # 用於 id_token 驗證 аудитория (aud)
LIFF_URL = os.getenv("LIFF_URL", "https://liff.line.me/YOUR_LIFF_ID")  # 請替換為你的 LIFF URL

# 設定允許的 CORS 網域 (可透過環境變數以逗號分隔，如 "https://yourdomain.com,https://liff.line.me")
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    # 預設允許常用環境與本地開發
    ALLOWED_ORIGINS = [
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "https://liff.line.me"
    ]

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

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
    user_uuid = Column(String(255), index=True, nullable=True)
    user_line_id = Column(String(255), index=True, nullable=True)
    heart_rate = Column(Integer, nullable=False)
    hrv_sdnn = Column(Float, nullable=False)
    stress_score = Column(Integer, nullable=False)
    health_light = Column(String(20), nullable=False)
    summary = Column(Text, nullable=True)
    action_advice = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

# ----------------------------------------------------
# 3. 自動檢查與修復 PostgreSQL 欄位
# ----------------------------------------------------
def auto_migrate_db():
    if "postgresql" in DATABASE_URL:
        try:
            with engine.connect() as conn:
                columns_to_add = [
                    ("user_uuid", "VARCHAR(255)"),
                    ("user_line_id", "VARCHAR(255)"),
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
    auto_migrate_db()
    yield

# ----------------------------------------------------
# 4. 資安輔助函式：LINE id_token 驗證
# ----------------------------------------------------
def verify_line_id_token(id_token: str, expected_user_id: str) -> bool:
    """
    呼叫 LINE 官方 API 驗證 id_token 的合法性與對應的 sub (LINE User ID)
    """
    if not id_token:
        print("[Auth Warning] 未提供 id_token，跳過嚴格驗證（本地開發模式）")
        return True

    try:
        data = {"id_token": id_token}
        if LINE_CHANNEL_ID:
            data["client_id"] = LINE_CHANNEL_ID

        res = requests.post("https://api.line.me/oauth2/v2.1/verify", data=data, timeout=5)
        if res.status_code != 200:
            print(f"[Auth Error] id_token 驗證失敗: {res.text}")
            return False

        payload = res.json()
        token_sub = payload.get("sub")
        
        if token_sub != expected_user_id:
            print(f"[Auth Error] token_sub ({token_sub}) 與宣稱的 user_id ({expected_user_id}) 不符！")
            return False

        return True
    except Exception as e:
        print(f"[Auth Exception] 驗證 id_token 時發生異常: {str(e)}")
        return False

# ----------------------------------------------------
# 5. 訊號處理與進階 HRV 演算法
# ----------------------------------------------------
def butter_bandpass_filter(data, lowcut=0.75, highcut=2.5, fs=30.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_rppg_metrics(green_signal: List[float], fps: int = 30):
    signal_arr = np.array(green_signal, dtype=float)
    detrended = signal_arr - np.mean(signal_arr)

    try:
        filtered = butter_bandpass_filter(detrended, lowcut=0.75, highcut=2.5, fs=fps)
    except Exception:
        filtered = detrended

    fft_vals = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), 1.0 / fps)
    valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
    
    if len(valid_idx) > 0:
        fft_hr = int(round(freqs[valid_idx[np.argmax(fft_vals[valid_idx])]] * 60))
    else:
        fft_hr = 75

    min_dist = max(int(fps * 60 / 160), 1)
    peaks, _ = find_peaks(filtered, distance=min_dist, prominence=np.std(filtered) * 0.3)

    if len(peaks) >= 3:
        rr_intervals = np.diff(peaks) / fps * 1000.0
        sdnn = float(np.std(rr_intervals))
        mean_rr = np.mean(rr_intervals)
        peak_hr = int(round(60000.0 / mean_rr)) if mean_rr > 0 else fft_hr
        hr = int(np.clip(peak_hr, 45, 160))
    else:
        hr = fft_hr
        sdnn = 38.5

    sdnn = round(float(np.clip(sdnn, 12.0, 120.0)), 1)
    return hr, sdnn

# ----------------------------------------------------
# 6. LINE Flex Message + Quick Reply 互動卡片
# ----------------------------------------------------
def build_flex_message(heart_rate: int, stress_score: int, health_light: str, summary: str, advice: str):
    """建立包含 Quick Reply 按鈕的 Flex Message"""
    color_map = {
        "GREEN": "#1DB954",
        "YELLOW": "#FFB800",
        "RED": "#FF4D4D"
    }
    header_color = color_map.get(health_light, "#1DB954")
    light_text = "良好 🟢" if health_light == "GREEN" else ("輕微偏高 🟡" if health_light == "YELLOW" else "需要注意 🔴")

    flex_contents = {
        "type": "bubble",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": header_color,
            "contents": [
                {
                    "type": "text",
                    "text": "PharmPulse 健康檢測報告",
                    "weight": "bold",
                    "color": "#FFFFFF",
                    "size": "sm"
                },
                {
                    "type": "text",
                    "text": f"狀態評估：{light_text}",
                    "weight": "bold",
                    "color": "#FFFFFF",
                    "size": "xl",
                    "margin": "xs"
                }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "box",
                    "layout": "horizontal",
                    "margin": "md",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "vertical",
                            "contents": [
                                {"type": "text", "text": "❤ 心率", "size": "xs", "color": "#888888"},
                                {"type": "text", "text": f"{heart_rate} BPM", "size": "lg", "weight": "bold", "color": "#111111"}
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "vertical",
                            "contents": [
                                {"type": "text", "text": "📊 壓力指數", "size": "xs", "color": "#888888"},
                                {"type": "text", "text": f"{stress_score} / 100", "size": "lg", "weight": "bold", "color": "#111111"}
                            ]
                        }
                    ]
                },
                {"type": "separator", "margin": "lg"},
                {
                    "type": "text",
                    "text": "📝 檢測摘要",
                    "weight": "bold",
                    "size": "xs",
                    "color": "#555555",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": summary,
                    "size": "sm",
                    "color": "#333333",
                    "wrap": True,
                    "margin": "xs"
                },
                {
                    "type": "text",
                    "text": "💡 處置與處方建議",
                    "weight": "bold",
                    "size": "xs",
                    "color": "#555555",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": advice,
                    "size": "xs",
                    "color": "#666666",
                    "wrap": True,
                    "margin": "xs"
                }
            ]
        }
    }

    return {
        "type": "flex",
        "altText": f"【PharmPulse】您的生理量測報告 (心率: {heart_rate} BPM)",
        "contents": flex_contents,
        "quickReply": {
            "items": [
                {
                    "type": "action",
                    "action": {
                        "type": "message",
                        "label": "📊 查看歷史紀錄",
                        "text": "查看歷史紀錄"
                    }
                },
                {
                    "type": "action",
                    "action": {
                        "type": "uri",
                        "label": "🔄 重新量測",
                        "uri": LIFF_URL
                    }
                }
            ]
        }
    }

def send_line_push_message(user_id: str, heart_rate: int, stress_score: int, health_light: str, summary: str, advice: str):
    """背景非同步發送包含 Quick Reply 的 LINE 推播"""
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not token or token.startswith("你的") or token == "YOUR_LINE_CHANNEL_ACCESS_TOKEN":
        print("[LINE Push Warning] 未設定正確的 LINE_CHANNEL_ACCESS_TOKEN，跳過推播。")
        return

    if not (user_id and user_id.startswith("U") and len(user_id) == 33):
        print(f"[LINE Push Warning] user_id ({user_id}) 不是有效的 LINE User ID，跳過推播。")
        return

    flex_payload = build_flex_message(heart_rate, stress_score, health_light, summary, advice)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "to": user_id,
        "messages": [flex_payload]
    }

    try:
        res = requests.post("https://api.line.me/v2/bot/message/push", json=payload, headers=headers, timeout=5)
        if res.status_code == 200:
            print(f"[LINE Push Success] 成功發送 Flex 卡片與 Quick Reply 給 {user_id}")
        else:
            print(f"[LINE Push Failed] 狀態碼 {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[LINE Push Error] 發送異常: {str(e)}")

# ----------------------------------------------------
# 7. FastAPI 應用程式與安全 CORS 設定
# ----------------------------------------------------
app = FastAPI(title="PharmPulse Backend API", version="1.2.0", lifespan=lifespan)

# 🔒 資安防護：限定 Allowed Origins (避免任意網域跨域請求)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------------------------------
# 8. 前端頁面託管路由
# ----------------------------------------------------
@app.get("/")
@app.get("/liff")
def serve_liff():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "message": "PharmPulse API Service is running"}

# ----------------------------------------------------
# 9. Request / Response Pydantic Schemas
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
# 10. API 路由定義
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
    # 🔒 資安驗證 id_token
    if req.id_token and not verify_line_id_token(req.id_token, req.user_line_id):
        raise HTTPException(status_code=401, detail="Invalid LINE id_token verification failed")

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
def analyze_rppg(req: AnalyzeRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # 🔒 資安驗證 id_token
    if req.id_token and not verify_line_id_token(req.id_token, req.user_line_id):
        raise HTTPException(status_code=401, detail="Invalid LINE id_token verification failed")

    try:
        if not req.rgb_signals:
            raise HTTPException(status_code=400, detail="rgb_signals cannot be empty")

        green_signal = req.rgb_signals[1] if len(req.rgb_signals) > 1 else req.rgb_signals[0]
        
        if len(green_signal) < 60:
            hr, sdnn = 72, 38.5
        else:
            hr, sdnn = calculate_rppg_metrics(green_signal, fps=req.fps)

        # 壓力指數動態區間映射
        normalized_sdnn = np.clip(sdnn, 15.0, 80.0)
        calc_stress = 85.0 - ((normalized_sdnn - 15.0) / (80.0 - 15.0)) * 70.0
        
        if hr > 85:
            calc_stress += (hr - 85) * 0.3
            
        stress = int(round(np.clip(calc_stress, 15, 95)))

        if stress > 75 or hr > 100 or hr < 50:
            light = "RED"
            summary = "生理數值顯著偏離基準，心血管與自律神經負擔較高。"
            advice = (
                "🚨 【處置建議】壓力或心率偏高：\n"
                "1. 請保持環境通風，閉目進行 5 分鐘深呼吸。\n"
                "2. 建議於今日量測血壓，若連續 2 天數值異常，請至門診複診。\n"
                "3. 可前往附近合作藥局，尋求藥師量測血壓與用藥諮詢。"
            )
        elif stress > 50:
            light = "YELLOW"
            summary = "生理指標輕微波動，呈現輕度疲勞或壓力上升狀態。"
            advice = (
                "⚠️ 【處置建議】輕度疲勞：\n"
                "1. 建議補充 300c.c. 溫開水並稍微休息 10 分鐘。\n"
                "2. 觀察晚間睡眠品質，避免睡前過度使用電子產品。"
            )
        else:
            light = "GREEN"
            summary = "生理指標良好，心律與自律神經狀態相當穩定。"
            advice = (
                "🟢 【處置建議】狀態非常棒：\n"
                "1. 請繼續保持規律作息與均衡飲食。\n"
                "2. 建議每日同一時間持續進行生理量測記錄。"
            )

        record = RPPGRecord(
            user_uuid=req.user_line_id,
            user_line_id=req.user_line_id,
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

        # 背景推播，附帶 Quick Reply 按鈕
        background_tasks.add_task(
            send_line_push_message,
            user_id=req.user_line_id,
            heart_rate=hr,
            stress_score=stress,
            health_light=light,
            summary=summary,
            advice=advice
        )

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

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Analyze Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ----------------------------------------------------
# 11. 歷史紀錄 API
# ----------------------------------------------------
@app.get("/api/v1/user/history/{user_line_id}")
def get_user_history(user_line_id: str, limit: int = 10, db: Session = Depends(get_db)):
    try:
        records = db.query(RPPGRecord)\
                    .filter(
                        or_(
                            RPPGRecord.user_uuid == user_line_id,
                            RPPGRecord.user_line_id == user_line_id
                        )
                    )\
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