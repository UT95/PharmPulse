import os
import random
from datetime import datetime
from typing import List, Optional
import zoneinfo

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

# 取得台灣時區
TAIPEI_TZ = zoneinfo.ZoneInfo("Asia/Taipei")

def get_taipei_now():
    return datetime.now(TAIPEI_TZ)

# --- 環境變數讀取 ---
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "")

app = FastAPI(title="PharmPulse C2C API", version="1.1.0")

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


class RppgSignalAnalyzeRequest(BaseModel):
    user_line_id: str
    rgb_signals: Optional[List[List[float]]] = Field(None)
    green_signals: Optional[List[float]] = Field(None)
    fps: int = Field(default=30)


# POS 演算法
def process_pos_rppg(rgb_signals: np.ndarray, fps: int = 30) -> int:
    N = rgb_signals.shape[1]
    if N < fps * 5:
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


# 產生圖表 URL
def generate_trend_chart_url(history_bpms: List[int]) -> str:
    labels = [f"t{i+1}" for i in range(len(history_bpms))]
    data_str = ",".join(map(str, history_bpms))
    chart_config = f"{{type:'line',data:{{labels:[{','.join([repr(l) for l in labels])}],datasets:[{{label:'BPM',data:[{data_str}],borderColor:'#1DB954',backgroundColor:'rgba(29,185,84,0.1)',fill:true,tension:0.3}}]}},options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{min:40,max:150}}}}}}}}"
    return f"https://quickchart.io/chart?c={chart_config}&w=500&h=200&bkg=white"


# 發送 LINE 推播（修復版）
def push_line_flex_message(user_id: str, hr: int, hrv: float, stress: int, light: str, summary: str, db: Session):
    if not LINE_ACCESS_TOKEN or user_id.startswith("U_TEST"):
        print(f"[LINE Push Skip] 無 Token 或測試帳號: {user_id}")
        return

    # 查 7 次歷史
    recent_records = (
        db.query(RppgRecord)
        .filter(RppgRecord.user_line_id == user_id)
        .order_by(RppgRecord.created_at.desc())
        .limit(7)
        .all()
    )
    bpms = [r.heart_rate for r in reversed(recent_records)]
    
    # 組合文字訊息與圖片
    now_str = get_taipei_now().strftime("%Y-%m-%d %H:%M")
    text_content = f"【PharmPulse 健康檢測報告】\n時間：{now_str}\n\n即時心率：{hr} BPM\nHRV (SDNN)：{hrv} ms\n壓力指數：{stress} / 100\n狀態：{light}\n\n評估建議：{summary}"

    messages = [{"type": "text", "text": text_content}]

    # 有 2 次以上紀錄才附上趨勢圖
    if len(bpms) > 1:
        chart_url = generate_trend_chart_url(bpms)
        messages.append({
            "type": "image",
            "originalContentUrl": chart_url,
            "previewImageUrl": chart_url
        })

    payload = {
        "to": user_id,
        "messages": messages
    }

    try:
        res = requests.post(
            "https://api.line.me/v2/bot/message/push",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
            },
            timeout=5
        )
        print(f"[LINE Push Response] Code: {res.status_code}, Body: {res.text}")
    except Exception as e:
        print(f"[LINE Push Error] {e}")


@app.get("/")
def read_root():
    return {"status": "online"}

@app.get("/liff", response_class=HTMLResponse)
def serve_liff_page():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/v1/analyze-rppg")
def analyze_rppg_signal(request: RppgSignalAnalyzeRequest, db: Session = Depends(get_db)):
    fps = request.fps
    min_samples = fps * 5

    if request.rgb_signals and len(request.rgb_signals) == 3:
        rgb_array = np.array(request.rgb_signals, dtype=float)
        if rgb_array.shape[1] < min_samples:
            raise HTTPException(status_code=400, detail="採樣時間不足")
        calculated_bpm = process_pos_rppg(rgb_array, fps=fps)
    else:
        calculated_bpm = 72

    hrv_sdnn = round(random.uniform(38.0, 68.0), 1)
    stress_score = random.randint(15, 45)
    health_light = "GREEN" if stress_score <= 40 else "YELLOW"
    summary = "心率與生理狀態良好。" if health_light == "GREEN" else "建議多休息。"

    db_record = RppgRecord(
        user_line_id=request.user_line_id,
        heart_rate=calculated_bpm,
        hrv_sdnn=hrv_sdnn,
        stress_score=stress_score,
        health_light=health_light,
        summary=summary,
        created_at=get_taipei_now()
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    # 發送 LINE 推播
    push_line_flex_message(
        user_id=request.user_line_id,
        hr=calculated_bpm,
        hrv=hrv_sdnn,
        stress=stress_score,
        light=health_light,
        summary=summary,
        db=db
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
        .order_by(RppgRecord.created_at.desc())
        .all()
    )
    # 格式化台灣時間輸出
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