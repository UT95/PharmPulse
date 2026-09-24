import os
import random
from datetime import datetime
from typing import List, Optional

import numpy as np
import requests
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from scipy.signal import butter, filtfilt
from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import Session

from database import Base, engine, get_db

# --- 環境變數讀取 ---
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

app = FastAPI(title="PharmPulse C2C API", version="1.0.0")

# --- CORS 跨域設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ORM Table Schema ---
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


# --- Pydantic Request Schemas ---
class RppgAnalyzeRequest(BaseModel):
    user_line_id: str
    measurement_duration_sec: int = 30
    is_fallback: bool = True


class RppgSignalAnalyzeRequest(BaseModel):
    user_line_id: str
    green_signals: List[float] = Field(..., description="前端 Canvas 擷取的綠光頻道強度陣列")
    fps: int = Field(default=30, description="攝影機幀率 (FPS)")


# --- rPPG 訊號處理核心演算法 (帶通濾波 + FFT 快速傅立葉轉換) ---
def process_rppg_signal(signals: List[float], fps: int = 30) -> int:
    """
    透過帶通濾波器與 FFT 計算真實心率 (BPM)
    """
    data = np.array(signals)
    # 1. 均值歸一化 (De-trending)
    data = data - np.mean(data)
    
    # 2. 設計帶通濾波器 (0.75 Hz ~ 3.33 Hz，對應 45 BPM ~ 200 BPM)
    lowcut = 0.75
    highcut = 3.33
    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(1, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    # 3. FFT 快速傅立葉轉換求主頻率
    fft_spectrum = np.abs(np.fft.rfft(filtered_data))
    fft_freqs = np.fft.rfftfreq(len(filtered_data), d=1.0/fps)
    
    # 限制在人類正常心率頻率區間 (0.75 Hz ~ 3.33 Hz)
    valid_idx = np.where((fft_freqs >= lowcut) & (fft_freqs <= highcut))
    valid_spectrum = fft_spectrum[valid_idx]
    valid_freqs = fft_freqs[valid_idx]
    
    # 找出能量最大的主頻率並換算為 BPM
    peak_freq = valid_freqs[np.argmax(valid_spectrum)]
    bpm = int(round(peak_freq * 60))
    
    return bpm


# --- 發送 LINE Flex Message 函式 ---
def push_line_flex_message(
    user_id: str, hr: int, hrv: float, stress: int, light: str, summary: str
):
    if not LINE_ACCESS_TOKEN or user_id == "U_TEST_USER_001" or LINE_ACCESS_TOKEN.startswith("YOUR_LINE"):
        print(f"[LINE Push Skip] 使用測試帳號或未設定 Token: {user_id}")
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}",
    }

    light_color = "#28a745" if light == "GREEN" else "#ffc107"
    if light == "RED":
        light_color = "#dc3545"

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


# 1. 舊版/模擬數據分析 API (保留相容性)
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


# 2. 新增：真實 rPPG 綠光訊號分析 API
@app.post("/api/v1/analyze-rppg")
def analyze_rppg_signal(request: RppgSignalAnalyzeRequest, db: Session = Depends(get_db)):
    min_samples = request.fps * 5  # 至少需 5 秒數據
    if len(request.green_signals) < min_samples:
        raise HTTPException(
            status_code=400,
            detail=f"訊號採樣不足，請提供至少 5 秒（約 {min_samples} 幀）的綠光強度數據。"
        )

    try:
        # 計算真實心率 (BPM)
        calculated_bpm = process_rppg_signal(request.green_signals, request.fps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"訊號演算失敗: {str(e)}")

    # 模擬計算 HRV 與壓力分數
    hrv_sdnn = round(random.uniform(35.0, 65.0), 1)
    stress_score = random.randint(20, 50)

    health_light = "GREEN"
    summary = "心率分析完成，自律神經狀態良好。"
    if stress_score > 40:
        health_light = "YELLOW"
        summary = "心率分析完成，注意適度放鬆。"

    # 寫入資料庫 (PostgreSQL / SQLite 相容)
    db_record = RppgRecord(
        user_line_id=request.user_line_id,
        heart_rate=calculated_bpm,
        hrv_sdnn=hrv_sdnn,
        stress_score=stress_score,
        health_light=health_light,
        summary=summary,
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    # 推送 LINE Flex Message 通知
    push_line_flex_message(
        user_id=request.user_line_id,
        hr=calculated_bpm,
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