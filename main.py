import os
import math
import json
import re
import time
import random
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager
from scipy.signal import butter, filtfilt, find_peaks, hilbert

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, text, or_
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

# LINE Bot SDK Imports
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
except ImportError:
    LineBotApi = WebhookHandler = InvalidSignatureError = MessageEvent = TextMessage = TextSendMessage = None

# ✅ 新版 Google Gemini SDK & OpenAI SDK
from google import genai
from openai import OpenAI

# ----------------------------------------------------
# 1. 環境變數與 AI 模型 / 資料庫設定
# ----------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
LINE_CHANNEL_ID = os.getenv("LINE_CHANNEL_ID", "").strip()
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
LIFF_URL = os.getenv("LIFF_URL", "").strip()

# AI 模型 Key 設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()

handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if (LineBotApi and LINE_CHANNEL_ACCESS_TOKEN) else None

def utc_iso(dt):
    if not dt:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc).isoformat()

# ✅ 1. 初始化主要模型：OpenAI Client
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"[AI System] ✅ 主要模型 OpenAI 已初始化 (Model: {OPENAI_MODEL_NAME})。")
else:
    print("[AI System Warning] 未設定 OPENAI_API_KEY，將無法以 OpenAI 作為主要模型。")

# ✅ 2. 初始化備援模型：Google Gemini Client
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[AI System] 🔄 備援模型 Google Gemini 已初始化 (Model: {GEMINI_MODEL_NAME})。")
else:
    print("[AI System Warning] 未設定 GEMINI_API_KEY，將無 Gemini 備援模型支援。")

# 設定允許的 CORS 網域
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "https://liff.line.me"
    ]

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL or not DATABASE_URL else {}
engine = create_engine(
    DATABASE_URL if DATABASE_URL else "sqlite:///./pharmpulse.db",
    pool_pre_ping=True,
    connect_args=connect_args
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

# ----------------------------------------------------
# 2. 資料庫 Model 定義
# ----------------------------------------------------
class UserConsent(Base):
    __tablename__ = "user_consents"

    id = Column(Integer, primary_key=True, index=True)
    user_line_id = Column(String(255), unique=True, index=True, nullable=False)
    agreed = Column(String(10), default="true")
    terms_version = Column(String(50), default="v1.0")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class RPPGRecord(Base):
    __tablename__ = "rppg_records"

    id = Column(Integer, primary_key=True, index=True)
    user_uuid = Column(String(255), index=True, nullable=True)
    user_line_id = Column(String(255), index=True, nullable=True)
    heart_rate = Column(Integer, nullable=False)
    hrv_sdnn = Column(Float, nullable=False)
    hrv_rmssd = Column(Float, nullable=True)
    respiratory_rate = Column(Float, nullable=True)
    irregular_pulse_score = Column(Integer, nullable=True)
    signal_quality = Column(Integer, nullable=True)
    signal_quality_label = Column(String(20), nullable=True)
    spo2_experimental = Column(Float, nullable=True)
    monitoring_mode = Column(String(30), nullable=False, default="GENERAL")
    monitoring_mode_label = Column(String(80), nullable=True)
    mode_alert_reasons = Column(Text, nullable=True)
    stress_score = Column(Integer, nullable=False)
    health_light = Column(String(20), nullable=False)
    is_resolved = Column(Boolean, default=False)
    summary = Column(Text, nullable=True)
    action_advice = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class PharmacyIntervention(Base):
    __tablename__ = "pharmacy_interventions"

    id = Column(Integer, primary_key=True, index=True)
    user_line_id = Column(String(255), index=True, nullable=False)
    rppg_record_id = Column(Integer, nullable=True)

    # LINE / PHONE / IN_STORE / MEDICAL
    care_method = Column(String(50), nullable=False)
    symptoms = Column(Text, nullable=True)

    # 到藥局複測時才填
    systolic_bp = Column(Integer, nullable=True)
    diastolic_bp = Column(Integer, nullable=True)
    pulse = Column(Integer, nullable=True)

    pharmacist_note = Column(Text, nullable=True)

    # OBSERVE / RETEST / FOLLOW_UP / REFER
    action_result = Column(String(50), nullable=True)
    followup_date = Column(DateTime(timezone=True), nullable=True)

    # NONE / REFERRED
    referral_status = Column(String(50), default="NONE")

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


class HospitalReferral(Base):
    __tablename__ = "hospital_referrals"

    id = Column(Integer, primary_key=True, index=True)
    user_line_id = Column(String(255), index=True, nullable=False)

    # 來源：藥師關懷與對應 rPPG
    intervention_id = Column(Integer, index=True, nullable=True)
    rppg_record_id = Column(Integer, index=True, nullable=True)

    # GENERAL / CARDIO / RESPIRATORY
    monitoring_mode = Column(String(30), nullable=False, default="GENERAL")

    # 藥師轉介時留下的資訊
    symptoms = Column(Text, nullable=True)
    pharmacist_note = Column(Text, nullable=True)
    systolic_bp = Column(Integer, nullable=True)
    diastolic_bp = Column(Integer, nullable=True)
    pulse = Column(Integer, nullable=True)

    # WAITING / COMPLETED（醫師處置功能下一階段再接）
    status = Column(String(30), nullable=False, default="WAITING")

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )



class HospitalDecision(Base):
    __tablename__ = "hospital_decisions"

    id = Column(Integer, primary_key=True, index=True)
    referral_id = Column(Integer, unique=True, index=True, nullable=False)
    user_line_id = Column(String(255), index=True, nullable=False)

    # HOME_OBSERVE / ROUTINE_VISIT / PRIORITY_VISIT / URGENT_CARE
    decision = Column(String(40), nullable=False)
    decision_label = Column(String(100), nullable=False)
    doctor_note = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


class PatientChronicProfile(Base):
    __tablename__ = "patient_chronic_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_line_id = Column(String(255), unique=True, index=True, nullable=False)
    display_name = Column(String(120), nullable=True)

    # GENERAL / CARDIO / RESPIRATORY
    monitoring_mode = Column(String(30), nullable=False, default="GENERAL")

    # 使用者自述，不作疾病診斷
    chronic_conditions = Column(Text, nullable=True)
    medication_note = Column(Text, nullable=True)
    care_note = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )


Base.metadata.create_all(bind=engine)

# ----------------------------------------------------
# 3. 自動檢查與修復 資料庫欄位
# ----------------------------------------------------
def auto_migrate_db():
    """相容 PostgreSQL 與 SQLite 的自動 Column 補充 Migration"""
    if "postgresql" in DATABASE_URL:
        try:
            with engine.begin() as conn:
                columns_to_add = [
                    ("user_uuid", "VARCHAR(255)"),
                    ("user_line_id", "VARCHAR(255)"),
                    ("is_resolved", "BOOLEAN DEFAULT FALSE"),
                    ("summary", "TEXT"),
                    ("action_advice", "TEXT"),
                    ("hrv_rmssd", "DOUBLE PRECISION"),
                    ("respiratory_rate", "DOUBLE PRECISION"),
                    ("irregular_pulse_score", "INTEGER"),
                    ("signal_quality", "INTEGER"),
                    ("signal_quality_label", "VARCHAR(20)"),
                    ("spo2_experimental", "DOUBLE PRECISION"),
                    ("monitoring_mode", "VARCHAR(30) DEFAULT 'GENERAL'"),
                    ("monitoring_mode_label", "VARCHAR(80)"),
                    ("mode_alert_reasons", "TEXT")
                ]
                for col_name, col_type in columns_to_add:
                    try:
                        conn.execute(text(f"ALTER TABLE rppg_records ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                    except Exception as e:
                        print(f"[Migration Warning] Add column {col_name}: {e}")
        except Exception as e:
            print(f"[Migration Error] Database connection failed: {e}")
    elif "sqlite" in DATABASE_URL:
        try:
            with engine.begin() as conn:
                existing = {row[1] for row in conn.execute(text("PRAGMA table_info(rppg_records)"))}
                sqlite_columns = {
                    "hrv_rmssd": "FLOAT",
                    "respiratory_rate": "FLOAT",
                    "irregular_pulse_score": "INTEGER",
                    "signal_quality": "INTEGER",
                    "signal_quality_label": "VARCHAR(20)",
                    "spo2_experimental": "FLOAT",
                    "monitoring_mode": "VARCHAR(30) DEFAULT 'GENERAL'",
                    "monitoring_mode_label": "VARCHAR(80)",
                    "mode_alert_reasons": "TEXT",
                }
                for col_name, col_type in sqlite_columns.items():
                    if col_name not in existing:
                        conn.execute(text(f"ALTER TABLE rppg_records ADD COLUMN {col_name} {col_type}"))
        except Exception as e:
            print(f"[Migration Warning] SQLite advanced metrics migration: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    auto_migrate_db()
    yield

# ----------------------------------------------------
# 4. 資安輔助函式：LINE id_token 驗證
# ----------------------------------------------------
def verify_line_id_token(id_token: str, expected_user_id: str) -> bool:
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
        token_sub = payload.get("sub", "")
        
        if token_sub.strip().lower() != expected_user_id.strip().lower():
            print(f"[Auth Error] token_sub ({token_sub}) 與宣稱的 user_id ({expected_user_id}) 不符！")
            return False

        return True
    except Exception as e:
        print(f"[Auth Exception] 驗證 id_token 時發生異常: {str(e)}")
        return False

# ----------------------------------------------------
# 5. 跨廠商 AI 核心調度模組 (OpenAI 主模型 + Gemini 自動備援)
# ----------------------------------------------------
def extract_json_from_text(text_content: str) -> Optional[dict]:
    """強健的 JSON 解析器，能自動從 AI 回應文字中提取 JSON 內容"""
    if not text_content:
        return None
    try:
        return json.loads(text_content.strip())
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", text_content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return None

def call_gemini_with_retry(contents: str, config: Optional[dict] = None, max_retries: int = 3):
    """呼叫 Gemini API 並包含指數退避重試機制，應對 503 過載"""
    if not gemini_client:
        return None

    base_delays = [0.5, 1.5, 3.0]

    for attempt in range(max_retries + 1):
        try:
            kwargs = {"model": GEMINI_MODEL_NAME, "contents": contents}
            if config:
                kwargs["config"] = config

            response = gemini_client.models.generate_content(**kwargs)
            return response
        except Exception as e:
            err_msg = str(e)
            is_503 = "503" in err_msg or "UNAVAILABLE" in err_msg or "high demand" in err_msg

            if is_503 and attempt < max_retries:
                jitter = random.uniform(0.1, 0.4)
                sleep_time = base_delays[attempt] + jitter
                print(f"[Gemini 503 重試] 第 {attempt + 1} 次遇過載，等待 {sleep_time:.2f} 秒後進行重試...")
                time.sleep(sleep_time)
            else:
                raise e

def call_llm_with_fallback(prompt: str, is_json: bool = False) -> Optional[str]:
    """
    跨廠商 AI 呼叫整合：
    1. 主要模型：OpenAI (gpt-4o-mini)
    2. 備援模型：Google Gemini (帶 503 重試)
    """
    if openai_client:
        try:
            messages = [
                {"role": "system", "content": "你是一位 PharmPulse 社區智慧藥局系統的「AI 臨床衛教藥師」。"},
                {"role": "user", "content": prompt}
            ]
            kwargs = {
                "model": OPENAI_MODEL_NAME,
                "messages": messages,
                "temperature": 0.4,
            }
            if is_json:
                kwargs["response_format"] = {"type": "json_object"}

            response = openai_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content:
                print(f"[AI Call] ✅ 主要模型 OpenAI ({OPENAI_MODEL_NAME}) 呼叫成功！")
                return content.strip()
        except Exception as e:
            print(f"[AI Warning] ⚠️ 主要模型 OpenAI 呼叫失敗: {e}，準備自動切換至 Gemini 備援...")
            time.sleep(0.3)

    if gemini_client:
        try:
            config = {"temperature": 0.4}
            if is_json:
                config["response_mime_type"] = "application/json"

            response = call_gemini_with_retry(contents=prompt, config=config)
            if response and response.text:
                print(f"[AI Call] 🔄 備援模型 Google Gemini ({GEMINI_MODEL_NAME}) 呼叫成功！")
                return response.text.strip()
        except Exception as e:
            print(f"[AI Error] ❌ 備援模型 Gemini 呼叫失敗: {e}")

    print("[AI Error] ❌ 所有 AI 模型 (OpenAI 與 Gemini) 均無法連線。")
    return None

MONITORING_MODES = {
    "GENERAL": {
        "label": "一般健康監測",
        "note": "日常追蹤心率、HRV、呼吸與壓力趨勢。"
    },
    "CARDIO": {
        "label": "高血壓 / 心血管監測",
        "note": "著重心率、脈搏規律與 HRV 趨勢；高血壓仍需搭配合格血壓計確認。"
    },
    "RESPIRATORY": {
        "label": "COPD / 呼吸慢病監測",
        "note": "著重呼吸頻率與心肺負荷；SpO₂ 為實驗性估值，不作醫療判斷。"
    }
}

def normalize_monitoring_mode(value: Optional[str]) -> str:
    mode = (value or "GENERAL").strip().upper()
    return mode if mode in MONITORING_MODES else "GENERAL"

def evaluate_monitoring_mode(mode: str, heart_rate: int, stress_score: int, metrics: dict):
    """依監測模式提供保守的風險提示，不作疾病診斷。"""
    mode = normalize_monitoring_mode(mode)
    quality = metrics.get("signal_quality")
    rr = metrics.get("respiratory_rate")
    irregular = metrics.get("irregular_pulse_score")

    light = "GREEN"
    reasons = []

    # 所有模式共用的基礎安全條件
    if heart_rate > 110 or heart_rate < 45 or stress_score > 80:
        light = "RED"
        if heart_rate > 110: reasons.append("心率明顯偏高")
        if heart_rate < 45: reasons.append("心率明顯偏低")
        if stress_score > 80: reasons.append("生理壓力指數偏高")
    elif heart_rate >= 90 or heart_rate < 55 or stress_score > 55:
        light = "YELLOW"
        if heart_rate >= 90: reasons.append("心率偏高")
        if heart_rate < 55: reasons.append("心率偏低")
        if stress_score > 55: reasons.append("生理壓力上升")

    # 心血管模式：只有訊號品質足夠時才參考脈搏規律性
    if mode == "CARDIO" and quality is not None and quality >= 60 and irregular is not None:
        if irregular >= 70:
            if light == "GREEN": light = "YELLOW"
            reasons.append("脈搏規律性需重新確認")
        elif irregular >= 45 and light == "GREEN":
            light = "YELLOW"
            reasons.append("脈搏規律性略有波動")

    # 呼吸模式：只有訊號品質足夠時才參考呼吸頻率
    if mode == "RESPIRATORY" and quality is not None and quality >= 60 and rr is not None:
        if rr >= 25 or rr < 8:
            light = "RED"
            reasons.append("呼吸頻率明顯偏離常見靜息範圍")
        elif rr >= 21 or rr < 10:
            if light == "GREEN": light = "YELLOW"
            reasons.append("呼吸頻率需持續追蹤")

    if not reasons:
        reasons.append("目前未出現模式特定警示")

    return {
        "mode": mode,
        "label": MONITORING_MODES[mode]["label"],
        "note": MONITORING_MODES[mode]["note"],
        "light": light,
        "reasons": reasons
    }

def generate_gemini_health_advice(heart_rate: int, sdnn: float, stress_score: int, health_light: str, monitoring_mode: str = "GENERAL", respiratory_rate=None, irregular_pulse_score=None) -> Optional[dict]:
    """根據生理數據產生專業且溫暖的藥局衛教摘要與建議"""
    prompt = f"""
你是一位 PharmPulse 社區智慧藥局系統的「AI 臨床衛教藥師」。
請根據以下民眾透過手機鏡頭 rPPG 測得的即時生理數據，撰寫一份簡明、具專業衛教價值且充滿關懷的分析報告。

【生理數據資料】
- 心率 (Heart Rate): {heart_rate} BPM
- 心率變異度 (HRV SDNN): {sdnn} ms
- 壓力指數 (Stress Score): {stress_score} / 100
- 健康狀態燈號 (Health Light): {health_light}（僅作風險提示）
- 慢性病監測模式: {MONITORING_MODES.get(normalize_monitoring_mode(monitoring_mode), MONITORING_MODES["GENERAL"])["label"]}
- 呼吸頻率: {respiratory_rate if respiratory_rate is not None else "未取得"} 次/分
- 脈搏不規則風險分數: {irregular_pulse_score if irregular_pulse_score is not None else "未取得"} / 100

【輸出要求】
請務必以繁體中文 (台灣醫療衛教用語) 並且回傳「標準 JSON 格式」，包含以下兩個欄位：
1. "summary": 25 ~ 45 字的簡短摘要，評估其心血管與自律神經狀態。
2. "action_advice": 100 ~ 150 字的具體處置建議（需包含生活作息調整、水分補充、社區藥局血壓量測諮詢或就醫警示，請適度加入 emoji 並以編號條列）。

輸出 JSON 範例：
{{
    "summary": "您的心率趨於穩定，自律神經調節功能良好，目前生理壓力處於理想狀態。",
    "action_advice": "🟢 【衛教藥師建議】\\n1. 請繼續保持規律作息與均衡飲食。\\n2. 建議每日補充足量水分 (1500-2000c.c.)。\\n3. 歡迎隨時至合作藥局免費測量血壓與諮詢專業藥師。"
}}
"""
    raw_res = call_llm_with_fallback(prompt, is_json=True)
    if raw_res:
        parsed_data = extract_json_from_text(raw_res)
        if parsed_data and "summary" in parsed_data and "action_advice" in parsed_data:
            return parsed_data
    return None

def generate_ai_chat_response(db: Session, user_line_id: str, question: str, record_id: Optional[int] = None) -> str:
    """共通 AI 對話邏輯：供 LINE Webhook 與 API 共同呼叫"""
    record = None
    if record_id:
        record = db.query(RPPGRecord).filter(RPPGRecord.id == record_id).first()
    else:
        record = db.query(RPPGRecord)\
                   .filter(or_(RPPGRecord.user_line_id == user_line_id, RPPGRecord.user_uuid == user_line_id))\
                   .order_by(RPPGRecord.created_at.desc()).first()

    context_str = ""
    if record:
        created_str = record.created_at.strftime('%Y-%m-%d %H:%M') if record.created_at else '近期'
        context_str = f"""
【民眾最新生理紀錄】
- 心率: {record.heart_rate} BPM
- HRV (SDNN): {record.hrv_sdnn} ms
- HRV (RMSSD): {record.hrv_rmssd if record.hrv_rmssd is not None else '未取得'} ms
- 呼吸頻率: {record.respiratory_rate if record.respiratory_rate is not None else '未取得'} 次/分
- 脈搏不規則風險分數: {record.irregular_pulse_score if record.irregular_pulse_score is not None else '未取得'} / 100（僅篩檢提示）
- 訊號品質: {record.signal_quality if record.signal_quality is not None else '未取得'} / 100
- SpO₂ 實驗性估值: {record.spo2_experimental if record.spo2_experimental is not None else '未取得'} %（未校正、不可作診斷）
- 壓力指數: {record.stress_score} / 100
- 健康燈號: {record.health_light}
- 檢測時間: {created_str}
"""

    prompt = f"""
你是一位 PharmPulse 社區藥局的「AI 臨床衛教藥師」。請秉持專業、親切且嚴謹的態度回答民眾的問題。

{context_str}

【民眾的問題】
"{question}"

【回答規則】
1. 使用繁體中文，態度溫暖且專業。
2. 若涉及急重症症狀（如胸痛、呼吸困難、嚴重頭眩），請明確提醒立即就醫。
3. 強調 rPPG 與本軟體非醫療診斷設備，成果僅供個人健康管理與藥局衛教參考。
4. 控管在 200 字以內，清晰條列或分段。
"""
    reply = call_llm_with_fallback(prompt, is_json=False)
    if reply:
        return reply
    return "感謝您的諮詢！目前 AI 衛教諮詢服務忙碌中。若您有緊急身體不適，請務必先就醫或尋求社區藥師協助。"

# ----------------------------------------------------
# 6. 訊號處理與進階 HRV 演算法
# ----------------------------------------------------
def butter_bandpass_filter(data, lowcut=0.75, highcut=2.5, fs=30.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_green_signal(rgb_signals: Any) -> List[float]:
    """
    將前端傳入的 RGB 訊號統一轉成一維 Green channel。

    支援以下常見格式：
      [R, G]             -> [2, N]，取第 2 列 G
      [R, G, B]          -> [3, N]，取第 2 列 G
      [[R,G], ...]       -> [N, 2]，取第 2 欄 G
      [[R,G,B], ...]     -> [N, 3]，取第 2 欄 G
      [G, G, G, ...]     -> [N]，直接使用
    """
    try:
        raw = np.asarray(rgb_signals, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"rgb_signals 無法轉換成數值陣列：{exc}")

    raw = np.squeeze(raw)

    if raw.size == 0:
        raise ValueError("rgb_signals 為空")

    if raw.ndim == 1:
        green = raw
    elif raw.ndim == 2:
        rows, cols = raw.shape

        # 前端目前使用 [redArray, greenArray] / [red, green, blue]
        # 這類格式是 [channel, frame]，必須優先判斷，否則 [2,N]
        # 會被誤判成 [N,2]，最後只剩 2 個數值。
        if rows in (2, 3) and cols >= 5:
            green = raw[1, :]
        # [frame, channel]，例如 [[R,G], [R,G], ...]
        elif cols in (2, 3) and rows >= 5:
            green = raw[:, 1]
        else:
            raise ValueError(
                f"不支援的 rgb_signals 二維格式 shape={raw.shape}；"
                "請使用 [R,G]、[R,G,B] 或逐幀 [[R,G], ...] 格式"
            )
    else:
        raise ValueError(f"不支援的 rgb_signals 維度：ndim={raw.ndim}, shape={raw.shape}")

    green = np.asarray(green, dtype=np.float64)
    green = green[np.isfinite(green)]

    if green.size < 5:
        raise ValueError(f"有效 Green 訊號只有 {green.size} 點")

    return green.tolist()

def calculate_rppg_metrics(green_signal: List[float], fps: float = 30.0):
    """計算 rPPG 心率與估計 SDNN。此結果僅供健康管理原型使用。"""
    fps = float(np.clip(fps, 10.0, 60.0))
    signal_arr = np.asarray(green_signal, dtype=float)
    signal_arr = signal_arr[np.isfinite(signal_arr)]

    if len(signal_arr) < 30:
        raise ValueError("有效 rPPG 訊號不足，至少需要 30 個取樣點")

    detrended = signal_arr - np.mean(signal_arr)

    try:
        filtered = butter_bandpass_filter(detrended, lowcut=0.75, highcut=2.5, fs=fps)
    except Exception as exc:
        print(f"[rPPG Filter Warning] band-pass filter 失敗，改用去平均訊號：{exc}")
        filtered = detrended

    # Zero-padding 只增加 FFT 頻率網格密度，不改變實際訊號時間尺度。
    n_fft = max(1024, 2 ** int(math.ceil(math.log2(max(len(filtered), 1)))))
    fft_vals = np.abs(np.fft.rfft(filtered, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fps)
    valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]

    if len(valid_idx) > 0:
        fft_hr = int(round(freqs[valid_idx[np.argmax(fft_vals[valid_idx])]] * 60))
        fft_hr = int(np.clip(fft_hr, 45, 160))
    else:
        fft_hr = 75

    signal_std = float(np.std(filtered))
    prominence = max(signal_std * 0.3, 1e-8)
    min_dist = max(int(fps * 60 / 160), 1)
    peaks, _ = find_peaks(filtered, distance=min_dist, prominence=prominence)

    if len(peaks) >= 3:
        rr_intervals = np.diff(peaks) / fps * 1000.0
        # 排除明顯不合理的 RR 間隔，避免單一誤偵測嚴重扭曲 SDNN。
        rr_intervals = rr_intervals[(rr_intervals >= 375.0) & (rr_intervals <= 1333.0)]

        if len(rr_intervals) >= 2:
            sdnn = float(np.std(rr_intervals))
            mean_rr = float(np.mean(rr_intervals))
            peak_hr = int(round(60000.0 / mean_rr)) if mean_rr > 0 else fft_hr
            hr = int(np.clip(peak_hr, 45, 160))
        else:
            hr = fft_hr
            sdnn = 38.5
    else:
        hr = fft_hr
        sdnn = 38.5

    sdnn = round(float(np.clip(sdnn, 12.0, 120.0)), 1)
    return hr, sdnn

def extract_rgb_signals(rgb_signals: Any):
    """解析前端 [R,G,B] / [R,G] / frame-major 格式，回傳等長 red、green、blue。"""
    raw = np.asarray(rgb_signals, dtype=np.float64)
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise ValueError("進階 rPPG 指標需要提供多通道 RGB 訊號")
    rows, cols = raw.shape
    if rows in (2, 3) and cols >= 5:
        red, green = raw[0, :], raw[1, :]
        blue = raw[2, :] if rows == 3 else green.copy()
    elif cols in (2, 3) and rows >= 5:
        red, green = raw[:, 0], raw[:, 1]
        blue = raw[:, 2] if cols == 3 else green.copy()
    else:
        raise ValueError(f"無法解析 RGB 訊號 shape={raw.shape}")

    valid = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    red, green, blue = red[valid], green[valid], blue[valid]
    if len(green) < 30:
        raise ValueError("有效 RGB 訊號不足")
    return red, green, blue


def calculate_pos_signal(red_signal, green_signal, blue_signal, fps: float):
    """
    Plane-Orthogonal-to-Skin (POS) rPPG approximation.
    使用滑動視窗正規化 RGB，再投影到兩個與皮膚色調正交的方向，
    比單獨 Green channel 更能抑制共同光照變化與部分動作雜訊。
    """
    fps = float(np.clip(fps, 10.0, 60.0))
    r = np.asarray(red_signal, dtype=float)
    g = np.asarray(green_signal, dtype=float)
    b = np.asarray(blue_signal, dtype=float)
    n = min(len(r), len(g), len(b))
    r, g, b = r[:n], g[:n], b[:n]

    if n < max(60, int(fps * 3)):
        raise ValueError("POS rPPG 有效 RGB 訊號不足")

    rgb = np.vstack([r, g, b])
    window = max(int(round(1.6 * fps)), 16)
    h = np.zeros(n, dtype=float)
    weights = np.zeros(n, dtype=float)

    for end_idx in range(window, n + 1):
        start_idx = end_idx - window
        seg = rgb[:, start_idx:end_idx]
        means = np.mean(seg, axis=1, keepdims=True)
        if np.any(np.abs(means) < 1e-8):
            continue
        cn = seg / means - 1.0
        x = cn[1] - cn[2]
        y = cn[1] + cn[2] - 2.0 * cn[0]
        sy = float(np.std(y))
        alpha = float(np.std(x) / sy) if sy > 1e-8 else 0.0
        pulse = x + alpha * y
        pulse = pulse - np.mean(pulse)
        h[start_idx:end_idx] += pulse
        weights[start_idx:end_idx] += 1.0

    valid = weights > 0
    if not np.any(valid):
        raise ValueError("POS rPPG 無法建立有效訊號")
    h[valid] /= weights[valid]
    if np.any(~valid):
        h[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), h[valid])

    return h


def extract_red_green_signals(rgb_signals: Any):
    """向後相容舊程式：回傳 red、green。"""
    red, green, _ = extract_rgb_signals(rgb_signals)
    return red, green

def calculate_advanced_rppg_metrics(red_signal, green_signal, fps: float, hr: int, pulse_signal=None):
    """
    研究/健康管理用途的進階 rPPG 指標。
    SpO2 為一般 RGB 相機 Red/Green ratio-of-ratios 的未校正實驗值，
    不參與紅黃綠分流，也不可替代血氧機。
    """
    fps = float(np.clip(fps, 10.0, 60.0))
    red = np.asarray(red_signal, dtype=float)
    green = np.asarray(green_signal, dtype=float)
    n = min(len(red), len(green))
    red, green = red[:n], green[:n]
    duration = n / fps

    if pulse_signal is not None:
        gd = np.asarray(pulse_signal, dtype=float)[:n]
        gd = gd - np.mean(gd)
    else:
        gd = green - np.mean(green)
    try:
        pulse = butter_bandpass_filter(gd, 0.75, 2.5, fps)
    except Exception:
        pulse = gd

    # 訊號品質：主心率頻帶功率 / 心率頻帶總功率 + RR 穩定度。
    n_fft = max(2048, 2 ** int(math.ceil(math.log2(max(n, 1)))))
    spec = np.abs(np.fft.rfft(pulse, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fps)
    band = (freqs >= 0.75) & (freqs <= 2.5)
    target = hr / 60.0
    peak_band = band & (np.abs(freqs - target) <= 0.12)
    band_power = float(spec[band].sum()) if np.any(band) else 0.0
    peak_power = float(spec[peak_band].sum()) if np.any(peak_band) else 0.0
    spectral_ratio = peak_power / band_power if band_power > 0 else 0.0

    prominence = max(float(np.std(pulse)) * 0.3, 1e-8)
    peaks, _ = find_peaks(pulse, distance=max(int(fps * 60 / 160), 1), prominence=prominence)
    rr = np.diff(peaks) / fps * 1000.0 if len(peaks) >= 3 else np.array([])
    rr = rr[(rr >= 375.0) & (rr <= 1333.0)]

    rmssd = None
    irregular = None
    rr_consistency = 0.5
    if len(rr) >= 3:
        diffs = np.diff(rr)
        rmssd = round(float(np.sqrt(np.mean(diffs ** 2))), 1)
        median_rr = float(np.median(rr))
        mad = float(np.median(np.abs(rr - median_rr)))
        robust_cv = mad / median_rr if median_rr > 0 else 0.0
        large_diff_ratio = float(np.mean(np.abs(diffs) > 80.0)) if len(diffs) else 0.0
        irregular = int(round(np.clip((robust_cv / 0.12) * 55 + large_diff_ratio * 45, 0, 100)))
        rr_consistency = float(np.clip(1.0 - robust_cv / 0.18, 0, 1))

    quality = int(round(np.clip((spectral_ratio / 0.55) * 70 + rr_consistency * 30, 0, 100)))
    quality_label = "GOOD" if quality >= 70 else ("FAIR" if quality >= 45 else "POOR")

    # 呼吸頻率：至少約 20 秒，從 Green 低頻調變估測。品質差時不硬輸出。
    respiratory_rate = None
    if duration >= 18.0 and quality >= 45:
        try:
            # 呼吸會調變脈搏振幅；使用 Hilbert envelope 比直接看 RGB 慢漂移更穩定。
            envelope = np.abs(hilbert(pulse))
            envelope = envelope - np.mean(envelope)
            resp = butter_bandpass_filter(envelope, 0.10, 0.50, fps, order=2)
            rspec = np.abs(np.fft.rfft(resp, n=n_fft)) ** 2
            rband = (freqs >= 0.10) & (freqs <= 0.50)
            if np.any(rband):
                rf = float(freqs[np.where(rband)[0][np.argmax(rspec[rband])]])
                respiratory_rate = round(float(np.clip(rf * 60.0, 8.0, 30.0)), 1)
        except Exception:
            respiratory_rate = None

    # 實驗性 SpO2：RGB 相機沒有 IR channel，僅供趨勢研究，不作醫療判讀。
    spo2 = None
    if duration >= 18.0 and quality >= 70:
        r_dc, g_dc = float(np.mean(red)), float(np.mean(green))
        r_ac, g_ac = float(np.std(red - r_dc)), float(np.std(green - g_dc))
        if r_dc > 1e-6 and g_dc > 1e-6 and g_ac > 1e-6:
            ratio = (r_ac / r_dc) / (g_ac / g_dc)
            candidate = 110.0 - 25.0 * ratio
            if np.isfinite(candidate):
                spo2 = round(float(np.clip(candidate, 90.0, 100.0)), 1)

    return {
        "hrv_rmssd": rmssd,
        "respiratory_rate": respiratory_rate,
        "irregular_pulse_score": irregular,
        "signal_quality": quality,
        "signal_quality_label": quality_label,
        "spo2_experimental": spo2,
        "measurement_seconds": round(duration, 1),
    }

# ----------------------------------------------------
# 7. LINE Flex Message + Quick Reply 互動卡片
# ----------------------------------------------------
def build_flex_message(heart_rate: int, stress_score: int, health_light: str, summary: str, advice: str,
                       hrv_rmssd=None, respiratory_rate=None, irregular_pulse_score=None,
                       signal_quality=None, spo2_experimental=None):
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
                {
                    "type": "text",
                    "text": (
                        f"HRV RMSSD: {hrv_rmssd if hrv_rmssd is not None else '--'} ms  ·  "
                        f"呼吸: {respiratory_rate if respiratory_rate is not None else '--'} 次/分\n"
                        f"脈搏不規則風險: {irregular_pulse_score if irregular_pulse_score is not None else '--'}/100  ·  "
                        f"訊號品質: {signal_quality if signal_quality is not None else '--'}/100\n"
                        f"SpO₂(實驗性): {spo2_experimental if spo2_experimental is not None else '--'}%"
                    ),
                    "size": "xs",
                    "color": "#555555",
                    "wrap": True,
                    "margin": "md"
                },
                {"type": "text", "text": "※ SpO₂ 與脈搏不規則分數僅供研究/趨勢參考，不作診斷。", "size": "xxs", "color": "#999999", "wrap": True, "margin": "xs"},
                {"type": "separator", "margin": "lg"},
                {
                    "type": "text",
                    "text": "📝 AI 檢測摘要",
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
                    "text": "💡 藥師處置與建議",
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
                        "type": "uri",
                        "label": "📊 查看歷史紀錄",
                        "uri": f"{LIFF_URL}?page=history" if LIFF_URL else "https://liff.line.me"
                    }
                },
                {
                    "type": "action",
                    "action": {
                        "type": "uri",
                        "label": "🔄 重新量測",
                        "uri": LIFF_URL if LIFF_URL else "https://liff.line.me"
                    }
                }
            ]
        }
    }

def send_line_push_message(user_id: str, heart_rate: int, stress_score: int, health_light: str, summary: str, advice: str,
                           hrv_rmssd=None, respiratory_rate=None, irregular_pulse_score=None, signal_quality=None, spo2_experimental=None):
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
    if not token or token.startswith("你的") or token == "YOUR_LINE_CHANNEL_ACCESS_TOKEN":
        print("[LINE Push Warning] 未設定正確的 LINE_CHANNEL_ACCESS_TOKEN，跳過推播。")
        return

    if not (user_id and user_id.startswith("U") and len(user_id) == 33):
        print(f"[LINE Push Warning] user_id ({user_id}) 不是有效的 LINE User ID，跳過推播。")
        return

    flex_payload = build_flex_message(
        heart_rate, stress_score, health_light, summary, advice,
        hrv_rmssd, respiratory_rate, irregular_pulse_score, signal_quality, spo2_experimental
    )

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


HOSPITAL_DECISION_LABELS = {
    "HOME_OBSERVE": "持續居家觀察",
    "ROUTINE_VISIT": "安排一般門診",
    "PRIORITY_VISIT": "建議優先門診",
    "URGENT_CARE": "建議儘速就醫"
}

def send_line_doctor_decision_message(
    user_id: str,
    decision_label: str,
    doctor_note: Optional[str] = None
) -> dict:
    """
    將醫師端處置摘要推播至 LINE。
    回傳 sent/skipped/failed，避免 LINE 發送失敗影響醫院端資料庫處置。
    """
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()

    if not token or token.startswith("你的") or token == "YOUR_LINE_CHANNEL_ACCESS_TOKEN":
        print("[LINE Doctor Decision] 未設定正確 LINE_CHANNEL_ACCESS_TOKEN，跳過推播。")
        return {"status": "skipped", "reason": "missing_token"}

    if not (user_id and user_id.startswith("U") and len(user_id) == 33):
        print(f"[LINE Doctor Decision] user_id ({user_id}) 格式無效，跳過推播。")
        return {"status": "skipped", "reason": "invalid_user_id"}

    lines = [
        "【PharmPulse 醫療端處置更新】",
        f"醫師處置：{decision_label}"
    ]

    if doctor_note:
        clean_note = doctor_note.strip()
        if clean_note:
            lines.append(f"醫師備註：{clean_note}")

    lines.extend([
        "",
        "請依醫療人員建議安排後續照護，並持續使用 PharmPulse 追蹤生理趨勢。",
        "若出現明顯或快速惡化的不適，請依當地緊急醫療指示尋求協助。"
    ])

    payload = {
        "to": user_id,
        "messages": [{
            "type": "text",
            "text": "\n".join(lines)
        }]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    try:
        res = requests.post(
            "https://api.line.me/v2/bot/message/push",
            json=payload,
            headers=headers,
            timeout=5
        )

        if res.status_code == 200:
            print(f"[LINE Doctor Decision] 已通知 {user_id}")
            return {"status": "sent"}

        print(f"[LINE Doctor Decision] 發送失敗 {res.status_code}: {res.text}")
        return {
            "status": "failed",
            "http_status": res.status_code,
            "detail": res.text[:300]
        }

    except Exception as e:
        print(f"[LINE Doctor Decision] 發送異常: {e}")
        return {"status": "failed", "detail": str(e)}


# ----------------------------------------------------
# 8. FastAPI 應用程式與安全 CORS 設定
# ----------------------------------------------------
app = FastAPI(title="PharmPulse Backend API", version="1.4.0", lifespan=lifespan)

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
# 9. 靜態頁面與 LINE Webhook 入口
# ----------------------------------------------------
@app.get("/")
@app.get("/liff")
def serve_liff():
    """提供 LIFF 前端靜態頁面"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "message": "PharmPulse API Service is running"}

@app.get("/pharmacy")
def serve_pharmacy():
    """提供藥局端管理後台靜態頁面"""
    if os.path.exists("pharmacy.html"):
        return FileResponse("pharmacy.html")
    return {"status": "error", "message": "pharmacy.html not found"}

@app.get("/hospital")
def serve_hospital():
    """提供醫院端轉介個案管理頁面"""
    if os.path.exists("hospital.html"):
        return FileResponse("hospital.html")
    return {"status": "error", "message": "hospital.html not found"}

if handler and MessageEvent:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_message(event):
        """當使用者向 LINE 官方帳號傳送文字訊息時的自動 AI 衛教對話處理"""
        if not line_bot_api:
            print("[LINE Webhook Warning] line_bot_api 未能正常初始化，跳過訊息回覆。")
            return

        user_text = event.message.text
        user_id = event.source.user_id

        db = SessionLocal()
        try:
            reply_text = generate_ai_chat_response(db, user_line_id=user_id, question=user_text)
            
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_text)
            )
            print(f"[LINE Webhook] 已成功回覆 AI 衛教訊息給使用者 ({user_id})")
        except Exception as e:
            print(f"[LINE Webhook Error] 處理訊息時發生異常: {str(e)}")
        finally:
            db.close()

# ----------------------------------------------------
# 10. Pydantic Request Models
# ----------------------------------------------------
class AnalyzeRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    rgb_signals: Any  # 使用 Any 提升對前端陣列格式變化的相容度
    fps: float = Field(default=30.0, ge=10.0, le=60.0)
    monitoring_mode: str = "GENERAL"

class ConsentRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    terms_version: str = "v1.0"

class AIChatRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    question: str
    record_id: Optional[int] = None

class PatientChronicProfileRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    display_name: Optional[str] = None
    monitoring_mode: str = "GENERAL"
    chronic_conditions: Optional[str] = None
    medication_note: Optional[str] = None
    care_note: Optional[str] = None

class HospitalDecisionRequest(BaseModel):
    decision: str
    doctor_note: Optional[str] = None

class PharmacyInterventionRequest(BaseModel):
    user_line_id: str
    rppg_record_id: Optional[int] = None
    care_method: str
    symptoms: Optional[str] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    pulse: Optional[int] = None
    pharmacist_note: Optional[str] = None
    action_result: Optional[str] = None
    followup_date: Optional[str] = None

# ----------------------------------------------------
# 11. 使用者同意與生理分析 API
# ----------------------------------------------------

@app.get("/api/v1/user/chronic-profile/{user_line_id}")
def get_chronic_profile(user_line_id: str, db: Session = Depends(get_db)):
    """取得使用者自述的慢性病監測檔案。"""
    try:
        profile = db.query(PatientChronicProfile).filter(
            PatientChronicProfile.user_line_id == user_line_id
        ).first()

        if not profile:
            return {
                "status": "success",
                "exists": False,
                "profile": {
                    "user_line_id": user_line_id,
                    "display_name": None,
                    "monitoring_mode": "GENERAL",
                    "monitoring_mode_label": MONITORING_MODES["GENERAL"]["label"],
                    "chronic_conditions": None,
                    "medication_note": None,
                    "care_note": None
                }
            }

        mode = normalize_monitoring_mode(profile.monitoring_mode)
        return {
            "status": "success",
            "exists": True,
            "profile": {
                "user_line_id": profile.user_line_id,
                "display_name": profile.display_name,
                "monitoring_mode": mode,
                "monitoring_mode_label": MONITORING_MODES[mode]["label"],
                "chronic_conditions": profile.chronic_conditions,
                "medication_note": profile.medication_note,
                "care_note": profile.care_note,
                "updated_at": utc_iso(profile.updated_at)
            }
        }
    except Exception as e:
        return {"status": "error", "exists": False, "profile": None, "message": str(e)}


@app.post("/api/v1/user/chronic-profile")
def save_chronic_profile(req: PatientChronicProfileRequest, db: Session = Depends(get_db)):
    """新增或更新使用者自述慢性病監測檔案。"""
    if req.id_token and not verify_line_id_token(req.id_token, req.user_line_id):
        raise HTTPException(status_code=401, detail="Invalid LINE id_token verification failed")

    try:
        mode = normalize_monitoring_mode(req.monitoring_mode)
        profile = db.query(PatientChronicProfile).filter(
            PatientChronicProfile.user_line_id == req.user_line_id
        ).first()

        if not profile:
            profile = PatientChronicProfile(
                user_line_id=req.user_line_id,
                display_name=req.display_name,
                monitoring_mode=mode,
                chronic_conditions=req.chronic_conditions,
                medication_note=req.medication_note,
                care_note=req.care_note
            )
            db.add(profile)
        else:
            profile.display_name = req.display_name
            profile.monitoring_mode = mode
            profile.chronic_conditions = req.chronic_conditions
            profile.medication_note = req.medication_note
            profile.care_note = req.care_note
            profile.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(profile)

        return {
            "status": "success",
            "message": "個人慢性病監測檔案已儲存",
            "profile": {
                "monitoring_mode": mode,
                "monitoring_mode_label": MONITORING_MODES[mode]["label"],
                "chronic_conditions": profile.chronic_conditions,
                "medication_note": profile.medication_note,
                "care_note": profile.care_note,
                "updated_at": utc_iso(profile.updated_at)
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/user/consent-status/{user_line_id}")
def check_consent(user_line_id: str, db: Session = Depends(get_db)):
    """查詢使用者個人隱私與條款同意狀態"""
    try:
        record = db.query(UserConsent).filter(UserConsent.user_line_id == user_line_id).first()
        if record and record.agreed == "true":
            return {"status": "success", "agreed": True}
        return {"status": "success", "agreed": False}
    except Exception as e:
        return {"status": "error", "agreed": False, "message": str(e)}

@app.post("/api/v1/user/consent")
def save_consent(req: ConsentRequest, db: Session = Depends(get_db)):
    """更新或記錄使用者條款同意狀態"""
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

@app.post("/callback")
async def callback(
    request: Request, 
    x_line_signature: str = Header(None, alias="X-Line-Signature")
):
    """LINE Messaging API Webhook 入口"""
    if not handler:
        raise HTTPException(status_code=500, detail="LINE Webhook handler not configured")

    if not x_line_signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    body = await request.body()
    body_str = body.decode("utf-8")

    try:
        handler.handle(body_str, x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature. Check your LINE_CHANNEL_SECRET.")

    return "OK"

@app.post("/api/v1/analyze-rppg")
def analyze_rppg(req: AnalyzeRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """核心 rPPG 影像訊號真實動態分析 API"""
    if req.id_token and not verify_line_id_token(req.id_token, req.user_line_id):
        raise HTTPException(status_code=401, detail="Invalid LINE id_token verification failed")

    try:
        user_id_upper = req.user_line_id.upper()

        # 僅當明確帶有 DEMO/TEST 前綴且完全沒有傳入 rgb_signals 時才觸發 Mock 模擬
        is_mock = False
        if req.rgb_signals is None or (isinstance(req.rgb_signals, list) and len(req.rgb_signals) == 0):
            if "DEMO_RED" in user_id_upper or "TEST_HIGH" in user_id_upper:
                hr, sdnn, stress, is_mock = random.randint(105, 125), round(random.uniform(15.0, 22.0), 1), random.randint(82, 95), True
                advanced_metrics = {"hrv_rmssd": 18.0, "respiratory_rate": 23.0, "irregular_pulse_score": 62, "signal_quality": 88, "signal_quality_label": "GOOD", "spo2_experimental": 94.0}
            elif "DEMO_YELLOW" in user_id_upper or "TEST_WARN" in user_id_upper:
                hr, sdnn, stress, is_mock = random.randint(88, 98), round(random.uniform(28.0, 36.0), 1), random.randint(62, 74), True
                advanced_metrics = {"hrv_rmssd": 30.0, "respiratory_rate": 19.0, "irregular_pulse_score": 32, "signal_quality": 90, "signal_quality_label": "GOOD", "spo2_experimental": 96.0}
            elif "DEMO_GREEN" in user_id_upper:
                hr, sdnn, stress, is_mock = random.randint(65, 76), round(random.uniform(48.0, 68.0), 1), random.randint(20, 42), True
                advanced_metrics = {"hrv_rmssd": 46.0, "respiratory_rate": 15.0, "irregular_pulse_score": 10, "signal_quality": 93, "signal_quality_label": "GOOD", "spo2_experimental": 98.0}
            else:
                raise HTTPException(status_code=400, detail="rgb_signals 不能為空，請保持鏡頭對準臉部並重新開始測量")

        if not is_mock:
            # 1. 解析與提取 Green Channel 訊號
            try:
                red_signal, green_for_advanced, blue_signal = extract_rgb_signals(req.rgb_signals)
                pos_signal = calculate_pos_signal(
                    red_signal, green_for_advanced, blue_signal, float(np.clip(req.fps, 10.0, 60.0))
                )
                green_signal = green_for_advanced.tolist()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            print(
                f"[rPPG Debug] 收到來自 {req.user_line_id} 的 Green 訊號點數: "
                f"{len(green_signal)}, frontend FPS={req.fps:.2f}"
            )

            if len(green_signal) < 90:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"鏡頭有效採樣數據不足（僅收到 {len(green_signal)} 點），"
                        "請讓相機保持對準臉部至少 5 秒並保持靜止後重新量測。"
                    )
                )

            # 2. 使用前端依據實際取樣時間估算的 FPS。
            # 不再將 5~90 點人工插值到 150 點，避免扭曲時間尺度。
            target_fps = float(np.clip(req.fps, 10.0, 60.0))

            # 3. 呼叫 rPPG 演算法計算動態心率 (BPM) 與估計 HRV (SDNN)
            # POS 訊號作為主要脈搏波來源；Green channel 僅保留相容/除錯用途。
            hr, sdnn = calculate_rppg_metrics(pos_signal.tolist(), fps=target_fps)
            advanced_metrics = calculate_advanced_rppg_metrics(
                red_signal, green_for_advanced, target_fps, hr, pulse_signal=pos_signal
            )

            # 品質門檻：低品質時保留 HR 供參考，但不硬輸出較敏感的 HRV/RR/SpO2 指標。
            quality = advanced_metrics.get("signal_quality")
            if quality is not None and quality < 45:
                advanced_metrics["hrv_rmssd"] = None
                advanced_metrics["respiratory_rate"] = None
                advanced_metrics["irregular_pulse_score"] = None
                advanced_metrics["spo2_experimental"] = None

            # 4. 根據真實 SDNN 與 HR 動態換算壓力指數
            normalized_sdnn = np.clip(sdnn, 15.0, 80.0)
            calc_stress = 85.0 - ((normalized_sdnn - 15.0) / (80.0 - 15.0)) * 70.0
            if hr > 85:
                calc_stress += (hr - 85) * 0.35
            stress = int(round(np.clip(calc_stress, 15, 95)))

        if 'advanced_metrics' not in locals():
            advanced_metrics = {"hrv_rmssd": None, "respiratory_rate": None, "irregular_pulse_score": None, "signal_quality": None, "signal_quality_label": None, "spo2_experimental": None}

        # 依使用者選擇的監測模式做風險提示；不是疾病診斷。
        # 實驗性 SpO₂ 不參與紅黃綠燈判斷。
        # 若使用者已有個人慢性病檔案，優先使用檔案中的監測模式；
        # 沒有檔案時才使用前端本次選擇。
        saved_profile = db.query(PatientChronicProfile).filter(
            PatientChronicProfile.user_line_id == req.user_line_id
        ).first()
        selected_mode = (
            saved_profile.monitoring_mode
            if saved_profile and saved_profile.monitoring_mode
            else req.monitoring_mode
        )

        mode_eval = evaluate_monitoring_mode(
            selected_mode,
            heart_rate=hr,
            stress_score=stress,
            metrics=advanced_metrics
        )
        monitoring_mode = mode_eval["mode"]
        monitoring_mode_label = mode_eval["label"]
        monitoring_mode_note = mode_eval["note"]
        mode_alert_reasons = mode_eval["reasons"]
        light = mode_eval["light"]

        # AI 生成衛教分析 (優先使用 LLM 模組，失敗自動轉為範本保底)
        ai_res = generate_gemini_health_advice(
            heart_rate=hr, sdnn=sdnn, stress_score=stress, health_light=light,
            monitoring_mode=monitoring_mode,
            respiratory_rate=advanced_metrics.get("respiratory_rate"),
            irregular_pulse_score=advanced_metrics.get("irregular_pulse_score")
        )

        if ai_res:
            summary = ai_res.get("summary", "")
            advice = ai_res.get("action_advice", "")
        else:
            # 保底預設文字
            if light == "RED":
                summary = "生理數值偏離基準，心血管與自律神經負擔較高。"
                advice = (
                    "🚨 【處置建議】壓力或心率偏高：\n"
                    "1. 請保持環境通風，閉目進行 5 分鐘深呼吸。\n"
                    "2. 建議今日量測血壓，若連續 2 天數值異常，請至門診複診。\n"
                    "3. 可前往附近合作藥局，尋求藥師量測血壓與用藥諮詢。"
                )
            elif light == "YELLOW":
                summary = "生理指標輕微波動，呈現輕度疲勞或壓力上升狀態。"
                advice = (
                    "⚠️ 【處置建議】輕度疲勞：\n"
                    "1. 建議補充 300c.c. 溫開水並稍微休息 10 分鐘。\n"
                    "2. 觀察晚間睡眠品質，避免睡前過度使用電子產品。"
                )
            else:
                summary = "生理指標良好，心律與自律神經狀態相當穩定。"
                advice = (
                    "🟢 【處置建議】狀態非常棒：\n"
                    "1. 請繼續保持規律作息與均衡飲食。\n"
                    "2. 建議每日同一時間持續進行生理量測記錄。"
                )

        # 寫入資料庫
        record = RPPGRecord(
            user_uuid=req.user_line_id,
            user_line_id=req.user_line_id,
            heart_rate=hr,
            hrv_sdnn=sdnn,
            hrv_rmssd=advanced_metrics.get("hrv_rmssd"),
            respiratory_rate=advanced_metrics.get("respiratory_rate"),
            irregular_pulse_score=advanced_metrics.get("irregular_pulse_score"),
            signal_quality=advanced_metrics.get("signal_quality"),
            signal_quality_label=advanced_metrics.get("signal_quality_label"),
            spo2_experimental=advanced_metrics.get("spo2_experimental"),
            monitoring_mode=monitoring_mode,
            monitoring_mode_label=monitoring_mode_label,
            mode_alert_reasons="；".join(mode_alert_reasons),
            stress_score=stress,
            health_light=light,
            is_resolved=False,
            summary=summary,
            action_advice=advice,
            created_at=datetime.now(timezone.utc)
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        # 背景非同步推播 LINE Flex Message
        background_tasks.add_task(
            send_line_push_message,
            user_id=req.user_line_id,
            heart_rate=hr,
            stress_score=stress,
            health_light=light,
            summary=summary,
            advice=advice,
            hrv_rmssd=advanced_metrics.get("hrv_rmssd"),
            respiratory_rate=advanced_metrics.get("respiratory_rate"),
            irregular_pulse_score=advanced_metrics.get("irregular_pulse_score"),
            signal_quality=advanced_metrics.get("signal_quality"),
            spo2_experimental=advanced_metrics.get("spo2_experimental")
        )

        return {
            "status": "success",
            "data": {
                "id": record.id,
                "heart_rate": hr,
                "hrv_sdnn": sdnn,
                "hrv_rmssd": advanced_metrics.get("hrv_rmssd"),
                "respiratory_rate": advanced_metrics.get("respiratory_rate"),
                "irregular_pulse_score": advanced_metrics.get("irregular_pulse_score"),
                "signal_quality": advanced_metrics.get("signal_quality"),
                "signal_quality_label": advanced_metrics.get("signal_quality_label"),
                "spo2_experimental": advanced_metrics.get("spo2_experimental"),
                "monitoring_mode": monitoring_mode,
                "monitoring_mode_label": monitoring_mode_label,
                "monitoring_mode_note": monitoring_mode_note,
                "mode_alert_reasons": mode_alert_reasons,
                "stress_score": stress,
                "health_light": light,
                "summary": summary,
                "action_advice": advice,
                "created_at": utc_iso(record.created_at)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Analyze Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ----------------------------------------------------
# 12. AI 互動式健康諮詢 API
# ----------------------------------------------------
@app.post("/api/v1/ai/chat")
def ai_health_consultation(req: AIChatRequest, db: Session = Depends(get_db)):
    """前端 LIFF AI 諮詢對話視窗 API"""
    if req.id_token and not verify_line_id_token(req.id_token, req.user_line_id):
        raise HTTPException(status_code=401, detail="Invalid LINE id_token verification failed")

    reply_text = generate_ai_chat_response(
        db=db,
        user_line_id=req.user_line_id,
        question=req.question,
        record_id=req.record_id
    )

    return {
        "status": "success",
        "reply": reply_text
    }

# ----------------------------------------------------
# 13. 民眾歷史紀錄 API
# ----------------------------------------------------
@app.get("/api/v1/user/history/{user_line_id}")
def get_user_history(user_line_id: str, limit: int = 10, db: Session = Depends(get_db)):
    """取得特定使用者的生理檢測歷史數據"""
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
                "hrv_rmssd": r.hrv_rmssd,
                "respiratory_rate": r.respiratory_rate,
                "irregular_pulse_score": r.irregular_pulse_score,
                "signal_quality": r.signal_quality,
                "signal_quality_label": r.signal_quality_label,
                "spo2_experimental": r.spo2_experimental,
                "monitoring_mode": r.monitoring_mode or "GENERAL",
                "monitoring_mode_label": r.monitoring_mode_label or MONITORING_MODES["GENERAL"]["label"],
                "mode_alert_reasons": r.mode_alert_reasons,
                "stress_score": r.stress_score,
                "health_light": r.health_light,
                "summary": r.summary,
                "action_advice": r.action_advice,
                "created_at": utc_iso(r.created_at)
            })
        return {"status": "success", "history": result}
    except Exception as e:
        return {"status": "error", "history": [], "message": str(e)}

# ----------------------------------------------------
# 14. 藥局端專用管理 API
# ----------------------------------------------------
@app.get("/api/v1/pharmacy/alerts")
def get_pharmacy_alerts(db: Session = Depends(get_db)):
    """取得社區藥局端即時異常警示清單（黃燈/紅燈且未結案）"""
    try:
        records = db.query(RPPGRecord)\
                    .filter(RPPGRecord.health_light.in_(["RED", "YELLOW"]))\
                    .filter(or_(RPPGRecord.is_resolved == False, RPPGRecord.is_resolved.is_(None)))\
                    .order_by(RPPGRecord.created_at.desc())\
                    .limit(30)\
                    .all()
        
        user_ids = [r.user_line_id or r.user_uuid for r in records if (r.user_line_id or r.user_uuid)]
        profiles = db.query(PatientChronicProfile).filter(
            PatientChronicProfile.user_line_id.in_(user_ids)
        ).all() if user_ids else []
        profile_map = {p.user_line_id: p for p in profiles}

        result = []
        for r in records:
            profile = profile_map.get(r.user_line_id or r.user_uuid)
            result.append({
                "id": r.id,
                "user_line_id": r.user_line_id or r.user_uuid,
                "heart_rate": r.heart_rate,
                "hrv_sdnn": r.hrv_sdnn,
                "hrv_rmssd": r.hrv_rmssd,
                "respiratory_rate": r.respiratory_rate,
                "irregular_pulse_score": r.irregular_pulse_score,
                "signal_quality": r.signal_quality,
                "signal_quality_label": r.signal_quality_label,
                "spo2_experimental": r.spo2_experimental,
                "monitoring_mode": r.monitoring_mode or "GENERAL",
                "monitoring_mode_label": r.monitoring_mode_label or MONITORING_MODES["GENERAL"]["label"],
                "profile_monitoring_mode": normalize_monitoring_mode(profile.monitoring_mode) if profile else (r.monitoring_mode or "GENERAL"),
                "profile_monitoring_mode_label": MONITORING_MODES[normalize_monitoring_mode(profile.monitoring_mode)]["label"] if profile else (r.monitoring_mode_label or MONITORING_MODES["GENERAL"]["label"]),
                "chronic_conditions": profile.chronic_conditions if profile else None,
                "mode_alert_reasons": r.mode_alert_reasons,
                "stress_score": r.stress_score,
                "health_light": r.health_light,
                "summary": r.summary,
                "action_advice": r.action_advice,
                "created_at": utc_iso(r.created_at)
            })
        return {"status": "success", "alerts": result}
    except Exception as e:
        return {"status": "error", "alerts": [], "message": str(e)}

@app.get("/api/v1/pharmacy/dashboard")
def get_pharmacy_dashboard(db: Session = Depends(get_db)):
    """藥師慢性病個案管理 KPI 與已到期追蹤清單。"""
    try:
        # 1) 個人慢性病檔案：若存在，優先作為病人的分群來源。
        profiles = db.query(PatientChronicProfile).all()
        profile_map = {p.user_line_id: p for p in profiles if p.user_line_id}

        # 2) 建立每位曾量測民眾的最新監測模式，讓尚未建立個人檔案者也能納入總個案。
        measurement_rows = (
            db.query(RPPGRecord)
            .order_by(RPPGRecord.created_at.desc())
            .all()
        )
        patient_mode = {}
        for r in measurement_rows:
            user_id = r.user_line_id or r.user_uuid
            if not user_id or user_id in patient_mode:
                continue
            profile = profile_map.get(user_id)
            mode = normalize_monitoring_mode(
                profile.monitoring_mode if profile and profile.monitoring_mode else r.monitoring_mode
            )
            patient_mode[user_id] = mode

        # 有個人檔案但還沒有量測的人也計入個案管理總數。
        for user_id, profile in profile_map.items():
            patient_mode[user_id] = normalize_monitoring_mode(profile.monitoring_mode)

        mode_keys = ["GENERAL", "CARDIO", "RESPIRATORY"]
        stats = {
            mode: {
                "mode": mode,
                "label": MONITORING_MODES[mode]["label"],
                "total_patients": 0,
                "red_patients": 0,
                "yellow_patients": 0,
                "due_followups": 0,
            }
            for mode in mode_keys
        }

        for user_id, mode in patient_mode.items():
            stats[mode]["total_patients"] += 1

        # 3) 未結案紅黃燈以「民眾」計數，不因同一人多筆異常而重複灌高 KPI。
        unresolved = (
            db.query(RPPGRecord)
            .filter(RPPGRecord.health_light.in_(["RED", "YELLOW"]))
            .filter(or_(RPPGRecord.is_resolved == False, RPPGRecord.is_resolved.is_(None)))
            .order_by(RPPGRecord.created_at.desc())
            .all()
        )
        red_users = {mode: set() for mode in mode_keys}
        yellow_users = {mode: set() for mode in mode_keys}
        for r in unresolved:
            user_id = r.user_line_id or r.user_uuid
            if not user_id:
                continue
            profile = profile_map.get(user_id)
            mode = normalize_monitoring_mode(
                profile.monitoring_mode if profile and profile.monitoring_mode else r.monitoring_mode
            )
            if r.health_light == "RED":
                red_users[mode].add(user_id)
            elif r.health_light == "YELLOW":
                yellow_users[mode].add(user_id)

        for mode in mode_keys:
            stats[mode]["red_patients"] = len(red_users[mode])
            stats[mode]["yellow_patients"] = len(yellow_users[mode])

        # 4) 每位民眾只看最新一筆藥師關懷；如果其追蹤日期已到且尚未轉介醫院，就列為待追蹤。
        interventions = (
            db.query(PharmacyIntervention)
            .order_by(PharmacyIntervention.created_at.desc())
            .all()
        )
        latest_intervention = {}
        for item in interventions:
            if item.user_line_id and item.user_line_id not in latest_intervention:
                latest_intervention[item.user_line_id] = item

        today = datetime.now(timezone.utc).date()
        followups = []
        for user_id, item in latest_intervention.items():
            if not item.followup_date or item.action_result == "REFER":
                continue
            due_date = item.followup_date.date()
            if due_date > today:
                continue

            profile = profile_map.get(user_id)
            mode = normalize_monitoring_mode(
                profile.monitoring_mode if profile and profile.monitoring_mode
                else patient_mode.get(user_id, "GENERAL")
            )
            stats[mode]["due_followups"] += 1
            followups.append({
                "intervention_id": item.id,
                "user_line_id": user_id,
                "monitoring_mode": mode,
                "monitoring_mode_label": MONITORING_MODES[mode]["label"],
                "chronic_conditions": profile.chronic_conditions if profile else None,
                "care_method": item.care_method,
                "action_result": item.action_result,
                "pharmacist_note": item.pharmacist_note,
                "followup_date": due_date.isoformat(),
                "days_overdue": max(0, (today - due_date).days),
                "is_overdue": due_date < today,
            })

        followups.sort(key=lambda x: (x["followup_date"], x["user_line_id"]))

        all_stats = {
            "mode": "ALL",
            "label": "全部慢性病個案",
            "total_patients": len(patient_mode),
            "red_patients": len(set().union(*red_users.values())),
            "yellow_patients": len(set().union(*yellow_users.values())),
            "due_followups": len(followups),
        }

        return {
            "status": "success",
            "stats": {"ALL": all_stats, **stats},
            "followups": followups,
            "generated_at": utc_iso(datetime.now(timezone.utc)),
        }
    except Exception as e:
        return {
            "status": "error",
            "stats": {},
            "followups": [],
            "message": str(e),
        }


@app.post("/api/v1/pharmacy/resolve/{record_id}")
def resolve_pharmacy_alert(record_id: int, db: Session = Depends(get_db)):
    """藥師端標註警示個案為已處置/已結案"""
    try:
        rec = db.query(RPPGRecord).filter(RPPGRecord.id == record_id).first()
        if not rec:
            raise HTTPException(status_code=404, detail="Record not found")
        rec.is_resolved = True
        db.commit()
        return {"status": "success", "message": "Marked as resolved"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pharmacy/patient/{user_line_id}/history")
def get_patient_7day_history(user_line_id: str, db: Session = Depends(get_db)):
    """供藥師端調閱民眾近 7 天生理趨勢歷史記錄"""
    try:
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        records = db.query(RPPGRecord)\
                    .filter(
                        or_(
                            RPPGRecord.user_uuid == user_line_id,
                            RPPGRecord.user_line_id == user_line_id
                        )
                    )\
                    .filter(RPPGRecord.created_at >= seven_days_ago)\
                    .order_by(RPPGRecord.created_at.desc())\
                    .all()
        
        result = []
        for r in records:
            result.append({
                "id": r.id,
                "heart_rate": r.heart_rate,
                "hrv_sdnn": r.hrv_sdnn,
                "hrv_rmssd": r.hrv_rmssd,
                "respiratory_rate": r.respiratory_rate,
                "irregular_pulse_score": r.irregular_pulse_score,
                "signal_quality": r.signal_quality,
                "signal_quality_label": r.signal_quality_label,
                "spo2_experimental": r.spo2_experimental,
                "monitoring_mode": r.monitoring_mode or "GENERAL",
                "monitoring_mode_label": r.monitoring_mode_label or MONITORING_MODES["GENERAL"]["label"],
                "mode_alert_reasons": r.mode_alert_reasons,
                "stress_score": r.stress_score,
                "health_light": r.health_light,
                "summary": r.summary,
                "action_advice": r.action_advice,
                "created_at": utc_iso(r.created_at)
            })
        return {
            "status": "success",
            "user_line_id": user_line_id,
            "records": result
        }
    except Exception as e:
        return {"status": "error", "records": [], "message": str(e)}

@app.post("/api/v1/pharmacy/interventions")
def create_pharmacy_intervention(
    req: PharmacyInterventionRequest,
    db: Session = Depends(get_db)
):
    """儲存藥師關懷；若選擇 REFER，會同步建立醫院端 WAITING 轉介個案。"""
    allowed_methods = {"LINE", "PHONE", "IN_STORE", "MEDICAL"}
    allowed_results = {"OBSERVE", "RETEST", "FOLLOW_UP", "REFER"}

    if req.care_method not in allowed_methods:
        raise HTTPException(status_code=400, detail="Invalid care_method")

    if req.action_result and req.action_result not in allowed_results:
        raise HTTPException(status_code=400, detail="Invalid action_result")

    if req.care_method == "IN_STORE" and (req.systolic_bp is None or req.diastolic_bp is None):
        raise HTTPException(status_code=400, detail="到藥局複測時請填寫收縮壓與舒張壓")

    followup_dt = None
    if req.followup_date:
        try:
            # 前端 date input 是台灣日期；此欄目前只當追蹤「日期」使用。
            followup_dt = datetime.strptime(req.followup_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="followup_date 格式必須為 YYYY-MM-DD")

    referral_status = "REFERRED" if req.action_result == "REFER" else "NONE"

    try:
        intervention = PharmacyIntervention(
            user_line_id=req.user_line_id,
            rppg_record_id=req.rppg_record_id,
            care_method=req.care_method,
            symptoms=req.symptoms,
            systolic_bp=req.systolic_bp,
            diastolic_bp=req.diastolic_bp,
            pulse=req.pulse,
            pharmacist_note=req.pharmacist_note,
            action_result=req.action_result,
            followup_date=followup_dt,
            referral_status=referral_status
        )

        # flush 先取得 intervention.id，但整筆交易尚未 commit。
        db.add(intervention)
        db.flush()

        hospital_referral = None

        if req.action_result == "REFER":
            # 優先使用藥師目前正在處理的 rPPG；沒有的話找該使用者最新一筆。
            source_rppg = None
            if req.rppg_record_id:
                source_rppg = (
                    db.query(RPPGRecord)
                    .filter(RPPGRecord.id == req.rppg_record_id)
                    .first()
                )

            if not source_rppg:
                source_rppg = (
                    db.query(RPPGRecord)
                    .filter(
                        or_(
                            RPPGRecord.user_line_id == req.user_line_id,
                            RPPGRecord.user_uuid == req.user_line_id
                        )
                    )
                    .order_by(RPPGRecord.created_at.desc())
                    .first()
                )

            profile = (
                db.query(PatientChronicProfile)
                .filter(PatientChronicProfile.user_line_id == req.user_line_id)
                .first()
            )

            monitoring_mode = normalize_monitoring_mode(
                profile.monitoring_mode
                if profile and profile.monitoring_mode
                else (source_rppg.monitoring_mode if source_rppg else "GENERAL")
            )

            # 同一筆藥師關懷只允許建立一個醫院轉介。
            hospital_referral = (
                db.query(HospitalReferral)
                .filter(HospitalReferral.intervention_id == intervention.id)
                .first()
            )

            if not hospital_referral:
                hospital_referral = HospitalReferral(
                    user_line_id=req.user_line_id,
                    intervention_id=intervention.id,
                    rppg_record_id=source_rppg.id if source_rppg else req.rppg_record_id,
                    monitoring_mode=monitoring_mode,
                    symptoms=req.symptoms,
                    pharmacist_note=req.pharmacist_note,
                    systolic_bp=req.systolic_bp,
                    diastolic_bp=req.diastolic_bp,
                    pulse=req.pulse,
                    status="WAITING"
                )
                db.add(hospital_referral)
                db.flush()

        db.commit()
        db.refresh(intervention)

        return {
            "status": "success",
            "message": (
                "藥師關懷紀錄已儲存，醫院轉介已建立"
                if hospital_referral
                else "藥師關懷紀錄已儲存"
            ),
            "id": intervention.id,
            "referral_status": intervention.referral_status,
            "hospital_referral_id": hospital_referral.id if hospital_referral else None
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"儲存藥師關懷紀錄失敗: {str(e)}")


@app.get("/api/v1/pharmacy/patient/{user_line_id}/interventions")
def get_pharmacy_interventions(user_line_id: str, db: Session = Depends(get_db)):
    """取得指定民眾的藥師關懷歷史，最新一筆排在最前面。"""
    try:
        records = (
            db.query(PharmacyIntervention)
            .filter(PharmacyIntervention.user_line_id == user_line_id)
            .order_by(PharmacyIntervention.created_at.desc())
            .all()
        )

        result = []
        for r in records:
            result.append({
                "id": r.id,
                "rppg_record_id": r.rppg_record_id,
                "care_method": r.care_method,
                "symptoms": r.symptoms,
                "systolic_bp": r.systolic_bp,
                "diastolic_bp": r.diastolic_bp,
                "pulse": r.pulse,
                "pharmacist_note": r.pharmacist_note,
                "action_result": r.action_result,
                "followup_date": r.followup_date.strftime("%Y-%m-%d") if r.followup_date else None,
                "referral_status": r.referral_status,
                "created_at": utc_iso(r.created_at)
            })

        return {
            "status": "success",
            "user_line_id": user_line_id,
            "records": result
        }
    except Exception as e:
        return {"status": "error", "user_line_id": user_line_id, "records": [], "message": str(e)}


# ----------------------------------------------------
# 15. 醫院端轉介 API
# ----------------------------------------------------
@app.get("/api/v1/hospital/referrals")
def get_hospital_referrals(db: Session = Depends(get_db)):
    """醫院端待處理轉介清單。第一版只顯示 WAITING。"""
    try:
        referrals = (
            db.query(HospitalReferral)
            .filter(HospitalReferral.status == "WAITING")
            .order_by(HospitalReferral.created_at.desc())
            .all()
        )

        result = []
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

        for referral in referrals:
            rppg = None
            if referral.rppg_record_id:
                rppg = (
                    db.query(RPPGRecord)
                    .filter(RPPGRecord.id == referral.rppg_record_id)
                    .first()
                )

            # 若來源紀錄不存在，仍以該使用者最新 rPPG 作為醫院參考。
            if not rppg:
                rppg = (
                    db.query(RPPGRecord)
                    .filter(
                        or_(
                            RPPGRecord.user_line_id == referral.user_line_id,
                            RPPGRecord.user_uuid == referral.user_line_id
                        )
                    )
                    .order_by(RPPGRecord.created_at.desc())
                    .first()
                )

            profile = (
                db.query(PatientChronicProfile)
                .filter(PatientChronicProfile.user_line_id == referral.user_line_id)
                .first()
            )

            recent_records = (
                db.query(RPPGRecord)
                .filter(
                    or_(
                        RPPGRecord.user_line_id == referral.user_line_id,
                        RPPGRecord.user_uuid == referral.user_line_id
                    ),
                    RPPGRecord.created_at >= seven_days_ago
                )
                .all()
            )

            red_count = sum(1 for x in recent_records if x.health_light == "RED")
            yellow_count = sum(1 for x in recent_records if x.health_light == "YELLOW")

            mode = normalize_monitoring_mode(
                profile.monitoring_mode
                if profile and profile.monitoring_mode
                else referral.monitoring_mode
            )

            result.append({
                "referral_id": referral.id,
                "user_line_id": referral.user_line_id,
                "status": referral.status,

                "monitoring_mode": mode,
                "monitoring_mode_label": MONITORING_MODES[mode]["label"],
                "chronic_conditions": profile.chronic_conditions if profile else None,
                "medication_note": profile.medication_note if profile else None,

                "symptoms": referral.symptoms,
                "pharmacist_note": referral.pharmacist_note,
                "systolic_bp": referral.systolic_bp,
                "diastolic_bp": referral.diastolic_bp,
                "pulse": referral.pulse,

                "seven_day_red_count": red_count,
                "seven_day_yellow_count": yellow_count,

                "rppg": {
                    "record_id": rppg.id,
                    "heart_rate": rppg.heart_rate,
                    "hrv_sdnn": rppg.hrv_sdnn,
                    "hrv_rmssd": rppg.hrv_rmssd,
                    "respiratory_rate": rppg.respiratory_rate,
                    "irregular_pulse_score": rppg.irregular_pulse_score,
                    "signal_quality": rppg.signal_quality,
                    "signal_quality_label": rppg.signal_quality_label,
                    "stress_score": rppg.stress_score,
                    "health_light": rppg.health_light,
                    "mode_alert_reasons": rppg.mode_alert_reasons,
                    "summary": rppg.summary,
                    "action_advice": rppg.action_advice,
                    "created_at": utc_iso(rppg.created_at)
                } if rppg else None,

                "created_at": utc_iso(referral.created_at)
            })

        return {
            "status": "success",
            "count": len(result),
            "records": result
        }

    except Exception as e:
        return {
            "status": "error",
            "count": 0,
            "records": [],
            "message": str(e)
        }



@app.post("/api/v1/hospital/referrals/{referral_id}/decision")
def complete_hospital_referral(
    referral_id: int,
    req: HospitalDecisionRequest,
    db: Session = Depends(get_db)
):
    """
    醫師完成轉介處置：
    1. 建立 HospitalDecision
    2. HospitalReferral WAITING -> COMPLETED
    3. 對應 PharmacyIntervention 標記 HOSPITAL_COMPLETED
    4. commit 後嘗試 LINE 通知
    """
    decision_key = (req.decision or "").strip().upper()

    if decision_key not in HOSPITAL_DECISION_LABELS:
        raise HTTPException(status_code=400, detail="無效的醫師處置選項")

    referral = (
        db.query(HospitalReferral)
        .filter(HospitalReferral.id == referral_id)
        .first()
    )

    if not referral:
        raise HTTPException(status_code=404, detail="找不到此轉介個案")

    if referral.status != "WAITING":
        existing = (
            db.query(HospitalDecision)
            .filter(HospitalDecision.referral_id == referral_id)
            .first()
        )
        if existing:
            raise HTTPException(status_code=409, detail="此轉介已完成醫師處置")
        raise HTTPException(status_code=409, detail="此轉介目前不可處置")

    existing = (
        db.query(HospitalDecision)
        .filter(HospitalDecision.referral_id == referral_id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="此轉介已存在醫師處置紀錄")

    try:
        decision_label = HOSPITAL_DECISION_LABELS[decision_key]

        decision = HospitalDecision(
            referral_id=referral.id,
            user_line_id=referral.user_line_id,
            decision=decision_key,
            decision_label=decision_label,
            doctor_note=(req.doctor_note or "").strip() or None
        )
        db.add(decision)

        referral.status = "COMPLETED"

        if referral.intervention_id:
            intervention = (
                db.query(PharmacyIntervention)
                .filter(PharmacyIntervention.id == referral.intervention_id)
                .first()
            )
            if intervention:
                intervention.referral_status = "HOSPITAL_COMPLETED"

        db.commit()
        db.refresh(decision)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"醫師處置儲存失敗: {str(e)}")

    # LINE 失敗不回滾醫師處置。
    line_result = send_line_doctor_decision_message(
        referral.user_line_id,
        decision_label,
        decision.doctor_note
    )

    return {
        "status": "success",
        "message": "醫師處置已完成",
        "decision": {
            "id": decision.id,
            "referral_id": decision.referral_id,
            "user_line_id": decision.user_line_id,
            "decision": decision.decision,
            "decision_label": decision.decision_label,
            "doctor_note": decision.doctor_note,
            "created_at": utc_iso(decision.created_at)
        },
        "line_notification": line_result
    }


@app.get("/api/v1/pharmacy/patient/{user_line_id}/hospital-decisions")
def get_patient_hospital_decisions(
    user_line_id: str,
    db: Session = Depends(get_db)
):
    """藥局端查看指定民眾的醫院處置歷史，最新一筆在前。"""
    try:
        decisions = (
            db.query(HospitalDecision)
            .filter(HospitalDecision.user_line_id == user_line_id)
            .order_by(HospitalDecision.created_at.desc())
            .all()
        )

        records = []
        for item in decisions:
            referral = (
                db.query(HospitalReferral)
                .filter(HospitalReferral.id == item.referral_id)
                .first()
            )

            records.append({
                "id": item.id,
                "referral_id": item.referral_id,
                "decision": item.decision,
                "decision_label": item.decision_label,
                "doctor_note": item.doctor_note,
                "created_at": utc_iso(item.created_at),
                "referral_created_at": utc_iso(referral.created_at) if referral else None,
                "monitoring_mode": referral.monitoring_mode if referral else None
            })

        return {
            "status": "success",
            "user_line_id": user_line_id,
            "records": records
        }

    except Exception as e:
        return {
            "status": "error",
            "user_line_id": user_line_id,
            "records": [],
            "message": str(e)
        }



# ----------------------------------------------------
# 15. 本地直接執行進入點
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)