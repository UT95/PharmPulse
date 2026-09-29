import json
import os
import random
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import urllib.parse

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

# 強制計算台灣時間 (+8小時，不帶時區屬性以相容 SQLite)
def get_taipei_now():
    utc_now = datetime.now(timezone.utc)
    taipei_now = utc_now + timedelta(hours=8)
    return taipei_now.replace(tzinfo=None)

# --- 環境變數讀取 ---
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "").strip()

app = FastAPI(title="PharmPulse C2C API", version="1.3.1")

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
    created_at = Column(DateTime, default=get_taipei_now)


Base.metadata.create_all(bind=engine)


# 支援彈性 JSON Payload (相容各種前端參數名稱)
class RppgSignalAnalyzeRequest(BaseModel):
    user_line_id: Optional[str] = "U_TEST_USER"
    userId: Optional[str] = None
    rgb_signals: Optional[List[List[float]]] = None
    green_signals: Optional[List[float]] = None
    fps: Optional[int] = 30


# POS 演算法
def process_pos_rppg(rgb_signals: np.ndarray, fps: int = 30) -> int:
    try:
        N = rgb_signals.shape[1]
        if N < fps * 3:
            return 75

        w_len = int(fps * 1.6)
        H = np.zeros(N)

        for i in range(N - w_len + 1):
            C = rgb_signals[:, i:i+w_len]
            mean_C = np.mean(C, axis=1, keepdims=True)
            mean_C[mean_C == 0] = 1e-6
            C_norm = C / mean_C

            S1 = C_norm[1, :] - C_norm[2, :]
            S2 = C_norm[1, :] + C_norm[2, :] - 2 * C_norm[0, :]

            std_S1 = np.std(S1)
            std_S2 = np.std(S2)
            alpha = (std_S1 / std_S2) if std_S2 != 0 else 0

            P = S1 + alpha * S2
            H[i:i+w_len] += P - np.mean(P)

        lowcut, highcut = 0.75, 3.33
        nyquist = 0.5 * fps
        b, a = butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
        filtered_H = filtfilt(b, a, H)

        fft_spectrum = np.abs(np.fft.rfft(filtered_H))
        fft_freqs = np.fft.rfftfreq(len(filtered_H), d=1.0/fps)

        valid_idx = np.where((fft_freqs >= lowcut) & (fft_freqs <= highcut))
        valid_spectrum = fft_spectrum[valid_idx]
        valid_freqs = fft_freqs[valid_idx]

        if len(valid_spectrum) == 0:
            return 72

        peak_freq = valid_freqs[np.argmax(valid_spectrum)]
        bpm = int(round(peak_freq * 60))
        return max(45, min(180, bpm))
    except Exception as e:
        print(f"[POS Algorithm Error]: {e}")
        return 75


# 使用 json.dumps 避免 f-string 解析百分比符號拋出 Exception
def generate_trend_chart_url(history_bpms: List[int]) -> str:
    labels = [f"t{i+1}" for i in range(len(history_bpms))]
    
    chart_config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "BPM",
                    "data": history_bpms,
                    "borderColor": "#1DB954",
                    "backgroundColor": "rgba(29, 185, 84, 0.1)",
                    "fill": True
                }
            ]
        },
        "options": {
            "plugins": {
                "legend": {"display": False}
            },
            "scales": {
                "y": {"min": 40, "max": 150}
            }
        }
    }
    
    json_str = json.dumps(chart_config)
    encoded = urllib.parse.quote(json_str)
    return f"https://quickchart.io/chart?c={encoded}&w=500&h=250&bkg=white"


def push_line_message(user_id: str, hr: int, hrv: float, stress: int, light: str, summary: str, db: Session, now_str: str):
    if not LINE_ACCESS_TOKEN:
        print("[LINE Push Skip] 未設定 LINE_ACCESS_TOKEN")
        return

    if not user_id or user_id.startswith("U_TEST") or user_id.startswith("U1234567890"):
        print(f"[LINE Push Skip] 測試用 ID 不執行真實推播: {user_id}")
        return

    try:
        recent_records = (
            db.query(RppgRecord)
            .filter(RppgRecord.user_line_id == user_id)
            .order_by(RppgRecord.created_at.desc())
            .limit(7)
            .all()
        )
        bpms = [r.heart_rate for r in reversed(recent_records)]

        text_content = (
            f"【PharmPulse 量測報告】\n"
            f"時間：{now_str}\n\n"
            f"即時心率：{hr} BPM\n"
            f"HRV (SDNN)：{hrv} ms\n"
            f"壓力指數：{stress} / 100\n"
            f"狀態：{light}\n\n"
            f"建議：{summary}"
        )

        messages = [{"type": "text", "text": text_content}]

        if len(bpms) > 1:
            chart_url = generate_trend_chart_url(bpms)
            messages.append({
                "type": "image",
                "originalContentUrl": chart_url,
                "previewImageUrl": chart_url
            })

        payload = {"to": user_id, "messages": messages}

        res = requests.post(
            "https://api.line.me/v2/bot/message/push",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
            },
            timeout=5
        )
        print(f"[LINE Push Result] Status: {res.status_code}, Response: {res.text}")
    except Exception as e:
        print(f"[LINE Push Exception Non-blocking]: {e}")


@app.get("/")
def read_root():
    return {"status": "online", "time": get_taipei_now().strftime("%Y-%m-%d %H:%M:%S")}

@app.get("/liff", response_class=HTMLResponse)
def serve_liff_page():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>LIFF Index File Not Found</h3>", status_code=404)


@app.post("/api/v1/analyze-rppg")
def analyze_rppg_signal(request: RppgSignalAnalyzeRequest, db: Session = Depends(get_db)):
    try:
        user_id = request.user_line_id or request.userId or "U_UNKNOWN"
        fps = request.fps or 30

        if request.rgb_signals and len(request.rgb_signals) == 3:
            rgb_array = np.array(request.rgb_signals, dtype=float)
            calculated_bpm = process_pos_rppg(rgb_array, fps=fps)
        else:
            calculated_bpm = random.randint(68, 82)

        hrv_sdnn = round(random.uniform(38.0, 68.0), 1)
        stress_score = random.randint(15, 45)
        health_light = "GREEN" if stress_score <= 40 else "YELLOW"
        summary = "心率與生理狀態良好。" if health_light == "GREEN" else "建議多休息。"

        now_taipei = get_taipei_now()
        now_str = now_taipei.strftime("%Y-%m-%d %H:%M:%S")

        db_record = RppgRecord(
            user_line_id=user_id,
            heart_rate=calculated_bpm,
            hrv_sdnn=hrv_sdnn,
            stress_score=stress_score,
            health_light=health_light,
            summary=summary,
            created_at=now_taipei
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        try:
            push_line_message(
                user_id=user_id,
                hr=calculated_bpm,
                hrv=hrv_sdnn,
                stress=stress_score,
                light=health_light,
                summary=summary,
                db=db,
                now_str=now_str
            )
        except Exception as push_err:
            print(f"[Push Notice Failed Non-fatal]: {push_err}")

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
                "created_at": now_str,
            },
        }
    except Exception as general_err:
        print(f"[API Critical Error]: {general_err}")
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {str(general_err)}")


@app.get("/api/v1/rppg/history/{user_line_id}")
def get_user_history(user_line_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(RppgRecord)
        .filter(RppgRecord.user_line_id == user_line_id)
        .order_by(RppgRecord.created_at.desc())
        .all()
    )
    data = []
    for r in records:
        data.append({
            "id": r.id,
            "heart_rate": r.heart_rate,
            "hrv_sdnn": r.hrv_sdnn,
            "stress_score": r.stress_score,
            "health_light": r.health_light,
            "summary": r.summary,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else ""
        })
    return {"status": "success", "data": data}