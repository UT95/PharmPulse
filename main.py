from datetime import datetime
import random
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import Session

from database import Base, engine, get_db

import os

LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

app = FastAPI(title="PharmPulse C2C API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- SQLite ORM Table Schema ---
class RppgRecord(Base):
    __tablename__ = "rppg_records"

    id = Column(Integer, primary_key=True, index=True)
    user_line_id = Column(String, index=True)
    heart_rate = Column(Integer)
    hrv_sdnn = Column(Float)
    stress_score = Column(Integer)
    health_light = Column(String)
    summary = Column(String)
    created_at = Column(DateTime, default=datetime.now)


Base.metadata.create_all(bind=engine)


# --- Pydantic Request Schema ---
class RppgAnalyzeRequest(BaseModel):
    user_line_id: str
    measurement_duration_sec: int = 30
    is_fallback: bool = True


# --- 發送 LINE Flex Message 函式 ---
def push_line_flex_message(
    user_id: str, hr: int, hrv: float, stress: int, light: str, summary: str
):
    # 如果是測試 ID 或未設定 Token 則跳過
    if user_id == "U_TEST_USER_001" or LINE_ACCESS_TOKEN.startswith("YOUR_LINE"):
        print(f"[LINE Push Skip] 使用測試帳號或未設定 Token: {user_id}")
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}",
    }

    # 設定燈號顏色與文字
    light_color = "#28a745" if light == "GREEN" else "#ffc107"
    if light == "RED":
        light_color = "#dc3545"

    # Flex Message JSON 結構
    flex_payload = {
        "to": user_id,
        "messages": [
            {
                "type": "flex",
                "altText": "PharmPulse 檢測報告已完成！",
                "contents": {
                    "type": "bubble",
                    "header": {
                        "type": "box",
                        "layout": "vertical",
                        "backgroundColor": "#1DB954",
                        "contents": [
                            {
                                "type": "text",
                                "text": "PharmPulse 檢測報告",
                                "color": "#FFFFFF",
                                "weight": "bold",
                                "size": "lg",
                            }
                        ],
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "box",
                                "layout": "horizontal",
                                "contents": [
                                    {
                                        "type": "text",
                                        "text": "健康狀態",
                                        "color": "#666666",
                                    },
                                    {
                                        "type": "text",
                                        "text": light,
                                        "color": light_color,
                                        "weight": "bold",
                                        "align": "end",
                                    },
                                ],
                            },
                            {"type": "separator", "margin": "md"},
                            {
                                "type": "box",
                                "layout": "vertical",
                                "margin": "md",
                                "spacing": "sm",
                                "contents": [
                                    {
                                        "type": "text",
                                        "text": f"平均心率：{hr} bpm",
                                        "size": "sm",
                                    },
                                    {
                                        "type": "text",
                                        "text": f"HRV (SDNN)：{hrv} ms",
                                        "size": "sm",
                                    },
                                    {
                                        "type": "text",
                                        "text": f"壓力指數：{stress} / 100",
                                        "size": "sm",
                                    },
                                ],
                            },
                            {"type": "separator", "margin": "md"},
                            {
                                "type": "text",
                                "text": f"建議：{summary}",
                                "wrap": True,
                                "size": "xs",
                                "color": "#888888",
                                "margin": "md",
                            },
                        ],
                    },
                },
            }
        ],
    }

    try:
        res = requests.post(url, json=flex_payload, headers=headers, timeout=5)
        print(f"[LINE Push] Status: {res.status_code}, Response: {res.text}")
    except Exception as e:
        print(f"[LINE Push Error] {e}")


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "online", "system": "PharmPulse C2C Backend Server"}


@app.get("/liff", response_class=HTMLResponse)
def serve_liff_page():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="index.html 檔案未找到，請確認檔案位置。"
        )


@app.post("/api/v1/rppg/analyze")
def analyze_rppg(request: RppgAnalyzeRequest, db: Session = Depends(get_db)):
    heart_rate = random.randint(62, 88)
    hrv_sdnn = round(random.uniform(35.0, 65.0), 1)
    stress_score = random.randint(20, 50)

    health_light = "GREEN"
    summary = "自律神經狀態良好，交感與副交感神經運作平衡。"
    if stress_score > 40:
        health_light = "YELLOW"
        summary = "輕微壓力偏高，建議適度休息與深呼吸。"

    db_record = RppgRecord(
        user_line_id=request.user_line_id,
        heart_rate=heart_rate,
        hrv_sdnn=hrv_sdnn,
        stress_score=stress_score,
        health_light=health_light,
        summary=summary,
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    # 🚀 在分析完成後，主動發送 LINE 訊息卡片給使用者
    push_line_flex_message(
        user_id=request.user_line_id,
        hr=heart_rate,
        hrv=hrv_sdnn,
        stress=stress_score,
        light=health_light,
        summary=summary,
    )

    return {
        "status": "success",
        "data": {
            "record_id": db_record.id,
            "user_line_id": db_record.user_line_id,
            "heart_rate": db_record.heart_rate,
            "hrv_sdnn": db_record.hrv_sdnn,
            "stress_score": db_record.stress_score,
            "health_light": db_record.health_light,
            "summary": db_record.summary,
            "created_at": db_record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


@app.get("/api/v1/rppg/history/{user_line_id}")
def get_user_history(user_line_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(RppgRecord)
        .filter(RppgRecord.user_line_id == user_line_id)
        .all()
    )
    return {"status": "success", "count": len(records), "records": records}
