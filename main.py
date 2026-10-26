import os
import math
import json
import re
import time
import random
import hashlib
import hmac
import secrets
import base64
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager
from scipy.signal import butter, filtfilt, find_peaks, hilbert

from fastapi import FastAPI, HTTPException, Request, Response, Depends, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
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

# Staff authentication settings (藥師 / 醫師)
PHARMACIST_USERNAME = os.getenv("PHARMACIST_USERNAME", "pharmacist01").strip()
PHARMACIST_PASSWORD = os.getenv("PHARMACIST_PASSWORD", "").strip()
PHARMACIST_DISPLAY_NAME = os.getenv("PHARMACIST_DISPLAY_NAME", "藥師").strip() or "藥師"
DOCTOR_USERNAME = os.getenv("DOCTOR_USERNAME", "doctor01").strip()
DOCTOR_PASSWORD = os.getenv("DOCTOR_PASSWORD", "").strip()
DOCTOR_DISPLAY_NAME = os.getenv("DOCTOR_DISPLAY_NAME", "醫師").strip() or "醫師"
STAFF_SESSION_SECRET = os.getenv("STAFF_SESSION_SECRET", "").strip()
STAFF_SESSION_HOURS = int(os.getenv("STAFF_SESSION_HOURS", "8"))
STAFF_COOKIE_NAME = "pharmpulse_staff_session"

if not STAFF_SESSION_SECRET:
    # Local fallback only. Render should set STAFF_SESSION_SECRET so sessions survive restarts.
    STAFF_SESSION_SECRET = secrets.token_urlsafe(32)
    print("[Staff Auth Warning] 未設定 STAFF_SESSION_SECRET，本次啟動使用暫時 Session Secret；重新部署後需重新登入。")

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

    # 個人化健康分析（以過去高品質量測建立 14 天 robust baseline）
    personal_baseline_count = Column(Integer, nullable=True)
    personal_baseline_window_days = Column(Integer, nullable=True)
    personal_confidence = Column(Integer, nullable=True)
    personal_risk_score = Column(Integer, nullable=True)
    personal_status = Column(String(30), nullable=True)
    personal_analysis = Column(Text, nullable=True)

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


class StaffUser(Base):
    __tablename__ = "staff_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    display_name = Column(String(120), nullable=False)
    # PHARMACIST / DOCTOR
    role = Column(String(30), index=True, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_login_at = Column(DateTime(timezone=True), nullable=True)


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
                    ("mode_alert_reasons", "TEXT"),
                    ("personal_baseline_count", "INTEGER"),
                    ("personal_baseline_window_days", "INTEGER"),
                    ("personal_confidence", "INTEGER"),
                    ("personal_risk_score", "INTEGER"),
                    ("personal_status", "VARCHAR(30)"),
                    ("personal_analysis", "TEXT")
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
                    "personal_baseline_count": "INTEGER",
                    "personal_baseline_window_days": "INTEGER",
                    "personal_confidence": "INTEGER",
                    "personal_risk_score": "INTEGER",
                    "personal_status": "VARCHAR(30)",
                    "personal_analysis": "TEXT",
                }
                for col_name, col_type in sqlite_columns.items():
                    if col_name not in existing:
                        conn.execute(text(f"ALTER TABLE rppg_records ADD COLUMN {col_name} {col_type}"))
        except Exception as e:
            print(f"[Migration Warning] SQLite advanced metrics migration: {e}")


PBKDF2_ITERATIONS = 310_000


def hash_staff_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 password hash; no plaintext password is stored."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii")
    )


def verify_staff_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _base64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def create_staff_session_token(staff: StaffUser) -> str:
    payload = {
        "uid": staff.id,
        "role": staff.role,
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=STAFF_SESSION_HOURS)).timestamp())
    }
    payload_b64 = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        STAFF_SESSION_SECRET.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256
    ).digest()
    return f"{payload_b64}.{_base64url_encode(signature)}"


def decode_staff_session_token(token: str) -> Optional[dict]:
    if not token or "." not in token:
        return None
    try:
        payload_b64, signature_b64 = token.split(".", 1)
        expected_sig = hmac.new(
            STAFF_SESSION_SECRET.encode("utf-8"),
            payload_b64.encode("ascii"),
            hashlib.sha256
        ).digest()
        supplied_sig = _base64url_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, supplied_sig):
            return None

        payload = json.loads(_base64url_decode(payload_b64).decode("utf-8"))
        if int(payload.get("exp", 0)) <= int(datetime.now(timezone.utc).timestamp()):
            return None
        if payload.get("role") not in {"PHARMACIST", "DOCTOR"}:
            return None
        return payload
    except Exception:
        return None


def get_staff_user_from_request(request: Request, db: Session) -> Optional[StaffUser]:
    payload = decode_staff_session_token(request.cookies.get(STAFF_COOKIE_NAME, ""))
    if not payload:
        return None
    staff = db.query(StaffUser).filter(StaffUser.id == payload.get("uid")).first()
    if not staff or not staff.is_active or staff.role != payload.get("role"):
        return None
    return staff


def bootstrap_staff_accounts():
    """Create/update staff accounts from Render/.env environment variables."""
    configs = [
        (PHARMACIST_USERNAME, PHARMACIST_PASSWORD, PHARMACIST_DISPLAY_NAME, "PHARMACIST"),
        (DOCTOR_USERNAME, DOCTOR_PASSWORD, DOCTOR_DISPLAY_NAME, "DOCTOR")
    ]

    db = SessionLocal()
    try:
        for username, password, display_name, role in configs:
            if not password:
                print(f"[Staff Auth Warning] 尚未設定 {role} 密碼環境變數，{role} 登入帳號不會自動建立。")
                continue

            staff = db.query(StaffUser).filter(StaffUser.username == username).first()
            if not staff:
                staff = StaffUser(
                    username=username,
                    password_hash=hash_staff_password(password),
                    display_name=display_name,
                    role=role,
                    is_active=True
                )
                db.add(staff)
                print(f"[Staff Auth] ✅ 已建立 {role} 帳號：{username}")
            else:
                staff.display_name = display_name
                staff.role = role
                staff.is_active = True
                # Render env password is the bootstrap source of truth.
                if not verify_staff_password(password, staff.password_hash):
                    staff.password_hash = hash_staff_password(password)
                    print(f"[Staff Auth] 🔄 已更新 {role} 帳號密碼：{username}")
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[Staff Auth Error] 建立工作人員帳號失敗: {exc}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    auto_migrate_db()
    bootstrap_staff_accounts()
    yield

# ----------------------------------------------------
# 4. 資安輔助函式：LINE id_token 驗證
# ----------------------------------------------------
def verify_line_id_token(id_token: str, expected_user_id: str) -> bool:
    if not id_token:
        print("[Auth Warning] 未提供 id_token，跳過嚴格驗證（本地開發模式）")
        return True

    # LINE 的 id_token verify 需要 client_id（LINE Login Channel ID）。
    # 若 Render 尚未設定 LINE_CHANNEL_ID，不因前端開始傳 token 就把量測全部擋成 401。
    if not LINE_CHANNEL_ID:
        print("[Auth Warning] 未設定 LINE_CHANNEL_ID，暫時跳過 id_token 嚴格驗證")
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


PERSONAL_BASELINE_WINDOW_DAYS = 14
PERSONAL_BASELINE_MIN_RECORDS = 3
PERSONAL_BASELINE_TARGET_RECORDS = 7
PERSONAL_SIGNAL_QUALITY_MIN = 60

PERSONAL_STATUS_LABELS = {
    "LOW_QUALITY": "量測品質不足",
    "BUILDING": "個人基準建立中",
    "STABLE": "接近個人基準",
    "WATCH": "出現輕度個人偏差",
    "DEVIATION": "出現明顯個人偏差",
}

PERSONAL_METRIC_LABELS = {
    "heart_rate": "心率",
    "hrv_sdnn": "HRV SDNN",
    "hrv_rmssd": "HRV RMSSD",
    "respiratory_rate": "呼吸頻率",
    "stress_score": "壓力指數",
    "irregular_pulse_score": "脈搏不規則風險",
}


def _safe_number(value):
    try:
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _median_and_mad(values):
    clean = np.asarray([float(v) for v in values if v is not None], dtype=float)
    if clean.size == 0:
        return None, None
    med = float(np.median(clean))
    mad = float(np.median(np.abs(clean - med)))
    return med, mad


def _trend_info(values):
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 3:
        return {"direction": "INSUFFICIENT", "pct_per_measurement": None}
    arr = np.asarray(clean[-6:], dtype=float)
    x = np.arange(arr.size, dtype=float)
    slope = float(np.polyfit(x, arr, 1)[0])
    base = max(abs(float(np.median(arr))), 1e-6)
    pct = float((slope / base) * 100.0)
    if pct >= 2.0:
        direction = "UP"
    elif pct <= -2.0:
        direction = "DOWN"
    else:
        direction = "STABLE"
    return {"direction": direction, "pct_per_measurement": round(pct, 1)}


def _metric_comparison(metric_key, current_value, history_values):
    current = _safe_number(current_value)
    history = [_safe_number(v) for v in history_values]
    history = [v for v in history if v is not None]
    if current is None or len(history) < PERSONAL_BASELINE_MIN_RECORDS:
        return None

    med, mad = _median_and_mad(history)
    if med is None:
        return None

    deviation_pct = None
    if abs(med) > 1e-6:
        deviation_pct = ((current - med) / abs(med)) * 100.0

    robust_z = None
    if mad is not None and mad > 1e-6:
        robust_z = 0.6745 * (current - med) / mad

    recent = history[-5:] + [current]
    trend = _trend_info(recent)

    return {
        "metric": metric_key,
        "label": PERSONAL_METRIC_LABELS.get(metric_key, metric_key),
        "current": round(current, 1),
        "baseline_median": round(med, 1),
        "baseline_mad": round(mad, 2) if mad is not None else None,
        "deviation_pct": round(deviation_pct, 1) if deviation_pct is not None else None,
        "robust_z": round(float(robust_z), 2) if robust_z is not None else None,
        "trend": trend,
        "history_count": len(history),
    }


def _comparison_is_concerning(metric_key, comp):
    if not comp:
        return False, 0
    dev = comp.get("deviation_pct")
    z = comp.get("robust_z")
    if dev is None:
        return False, 0

    # 這些門檻只用來做「相較個人基準需重新確認」提示，不直接作疾病診斷。
    if metric_key == "heart_rate":
        concerning = abs(dev) >= 18 and (z is None or abs(z) >= 2.0 or abs(dev) >= 25)
        return concerning, 24 if concerning else 0
    if metric_key in {"hrv_sdnn", "hrv_rmssd"}:
        concerning = dev <= -25 and (z is None or z <= -2.0 or dev <= -35)
        return concerning, 18 if concerning else 0
    if metric_key == "respiratory_rate":
        concerning = abs(dev) >= 22 and (z is None or abs(z) >= 2.0 or abs(dev) >= 30)
        return concerning, 28 if concerning else 0
    if metric_key == "stress_score":
        concerning = dev >= 30 and (z is None or z >= 2.0 or dev >= 40)
        return concerning, 16 if concerning else 0
    if metric_key == "irregular_pulse_score":
        concerning = dev >= 35 and (z is None or z >= 2.0 or dev >= 50)
        return concerning, 22 if concerning else 0
    return False, 0


def _personal_reason(metric_key, comp):
    if not comp or comp.get("deviation_pct") is None:
        return None
    dev = comp["deviation_pct"]
    label = comp.get("label") or PERSONAL_METRIC_LABELS.get(metric_key, metric_key)
    direction = "高" if dev > 0 else "低"
    if metric_key in {"hrv_sdnn", "hrv_rmssd"} and dev < 0:
        return f"{label} 較個人 14 天基準低 {abs(dev):.0f}%"
    return f"{label} 較個人 14 天基準{direction} {abs(dev):.0f}%"


def build_personalized_health_analysis(
    db: Session,
    user_line_id: str,
    monitoring_mode: str,
    heart_rate: int,
    hrv_sdnn: float,
    stress_score: int,
    metrics: dict,
):
    """
    個人化健康分析引擎：
    - 14 天高品質量測 robust baseline（Median + MAD）
    - 個人偏差百分比 + robust z-score
    - 近期方向趨勢
    - 分析可信度（不是醫療準確率）
    - 個人偏差只可將 GREEN 提升為 YELLOW「需重新確認」，不單獨產生 RED
    """
    mode = normalize_monitoring_mode(monitoring_mode)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=PERSONAL_BASELINE_WINDOW_DAYS)
    current_quality = _safe_number(metrics.get("signal_quality"))

    history = (
        db.query(RPPGRecord)
        .filter(
            or_(
                RPPGRecord.user_line_id == user_line_id,
                RPPGRecord.user_uuid == user_line_id,
            )
        )
        .filter(RPPGRecord.created_at >= cutoff)
        .filter(RPPGRecord.signal_quality.isnot(None))
        .filter(RPPGRecord.signal_quality >= PERSONAL_SIGNAL_QUALITY_MIN)
        .order_by(RPPGRecord.created_at.asc(), RPPGRecord.id.asc())
        .limit(60)
        .all()
    )

    baseline_count = len(history)
    unique_days = len({(r.created_at.date() if r.created_at else None) for r in history if r.created_at})

    current_values = {
        "heart_rate": heart_rate,
        "hrv_sdnn": hrv_sdnn,
        "hrv_rmssd": metrics.get("hrv_rmssd"),
        "respiratory_rate": metrics.get("respiratory_rate"),
        "stress_score": stress_score,
        "irregular_pulse_score": metrics.get("irregular_pulse_score"),
    }

    mode_metrics = {
        "GENERAL": ["heart_rate", "hrv_sdnn", "stress_score"],
        "CARDIO": ["heart_rate", "hrv_sdnn", "hrv_rmssd", "irregular_pulse_score", "stress_score"],
        "RESPIRATORY": ["respiratory_rate", "heart_rate", "stress_score", "hrv_rmssd"],
    }[mode]

    comparisons = {}
    available_focus_metrics = 0
    for metric_key, current in current_values.items():
        hist_values = [getattr(r, metric_key, None) for r in history]
        comp = _metric_comparison(metric_key, current, hist_values)
        if comp:
            comparisons[metric_key] = comp
            if metric_key in mode_metrics:
                available_focus_metrics += 1

    quality_score = int(round(current_quality)) if current_quality is not None else 0
    baseline_coverage = min(100, round((baseline_count / PERSONAL_BASELINE_TARGET_RECORDS) * 100))
    day_coverage = min(100, round((unique_days / 5) * 100))
    metric_coverage = round((available_focus_metrics / max(len(mode_metrics), 1)) * 100)
    confidence = int(round(
        0.50 * quality_score +
        0.25 * baseline_coverage +
        0.15 * day_coverage +
        0.10 * metric_coverage
    ))

    reasons = []
    risk_score = 0
    concerning_metrics = []

    if current_quality is None or current_quality < PERSONAL_SIGNAL_QUALITY_MIN:
        status = "LOW_QUALITY"
        confidence = min(confidence, 45)
        reasons.append("本次訊號品質不足，暫不使用個人基準做風險升級")
    elif baseline_count < PERSONAL_BASELINE_MIN_RECORDS:
        status = "BUILDING"
        confidence = min(confidence, 55)
        reasons.append(f"目前只有 {baseline_count} 筆高品質歷史量測，至少累積 {PERSONAL_BASELINE_MIN_RECORDS} 筆後啟用個人偏差判讀")
    else:
        for metric_key in mode_metrics:
            comp = comparisons.get(metric_key)
            concerning, points = _comparison_is_concerning(metric_key, comp)
            if not concerning:
                continue
            risk_score += points
            concerning_metrics.append(metric_key)
            reason = _personal_reason(metric_key, comp)
            if reason:
                reasons.append(reason)

            trend = (comp or {}).get("trend") or {}
            trend_dir = trend.get("direction")
            # 趨勢若與風險方向一致，僅小幅加權。
            if metric_key in {"heart_rate", "respiratory_rate", "stress_score", "irregular_pulse_score"} and trend_dir == "UP":
                risk_score += 6
            elif metric_key in {"hrv_sdnn", "hrv_rmssd"} and trend_dir == "DOWN":
                risk_score += 6

        risk_score = int(min(100, risk_score))
        if risk_score >= 40:
            status = "DEVIATION"
        elif risk_score >= 20:
            status = "WATCH"
        else:
            status = "STABLE"

        if not reasons:
            reasons.append("本次主要指標接近個人 14 天基準")

    risk_score = int(min(100, risk_score))
    eligible = (
        current_quality is not None and current_quality >= PERSONAL_SIGNAL_QUALITY_MIN
        and baseline_count >= PERSONAL_BASELINE_MIN_RECORDS
    )
    should_escalate = bool(eligible and confidence >= 65 and risk_score >= 40)

    comparison_lines = []
    for metric_key in mode_metrics:
        comp = comparisons.get(metric_key)
        if not comp:
            continue
        dev = comp.get("deviation_pct")
        trend = comp.get("trend") or {}
        trend_map = {"UP": "近期上升", "DOWN": "近期下降", "STABLE": "近期穩定", "INSUFFICIENT": "趨勢資料不足"}
        dev_text = f"{dev:+.1f}%" if dev is not None else "--"
        comparison_lines.append(
            f"{comp['label']}：本次 {comp['current']} / 個人基準 {comp['baseline_median']}（{dev_text}，{trend_map.get(trend.get('direction'), '趨勢資料不足')}）"
        )

    if status == "LOW_QUALITY":
        summary = "本次量測品質不足，系統不以此筆資料做個人化風險判讀，建議保持穩定後重新量測。"
    elif status == "BUILDING":
        summary = f"正在建立您的個人健康基準；目前累積 {baseline_count} 筆高品質紀錄，持續固定情境量測可提升個人化分析可靠度。"
    elif status == "DEVIATION":
        summary = "本次部分指標相較您自己的 14 天基準出現明顯偏差，建議休息後重新量測並觀察是否持續。"
    elif status == "WATCH":
        summary = "本次部分指標較個人基準有輕度變化，建議持續觀察近期趨勢。"
    else:
        summary = "本次主要指標大致接近您的 14 天個人基準。"

    return {
        "version": "personal-v1",
        "status": status,
        "status_label": PERSONAL_STATUS_LABELS[status],
        "baseline_window_days": PERSONAL_BASELINE_WINDOW_DAYS,
        "baseline_count": baseline_count,
        "baseline_unique_days": unique_days,
        "confidence": max(0, min(100, confidence)),
        "confidence_note": "此分數代表本次個人化分析資料充分度與訊號品質，不代表醫療診斷準確率。",
        "risk_score": risk_score,
        "eligible": eligible,
        "should_escalate": should_escalate,
        "reasons": reasons[:4],
        "concerning_metrics": concerning_metrics,
        "comparisons": comparisons,
        "comparison_lines": comparison_lines[:5],
        "summary": summary,
    }


def personalized_prompt_context(personalized_analysis: Optional[dict]) -> str:
    if not personalized_analysis:
        return "尚無個人化基準資料。"
    reasons = "；".join(personalized_analysis.get("reasons") or [])
    comparisons = "；".join(personalized_analysis.get("comparison_lines") or [])
    return (
        f"個人化狀態: {personalized_analysis.get('status_label')}\n"
        f"14天高品質基準筆數: {personalized_analysis.get('baseline_count')}\n"
        f"分析可信度: {personalized_analysis.get('confidence')}/100（不是醫療準確率）\n"
        f"個人偏差風險分數: {personalized_analysis.get('risk_score')}/100\n"
        f"個人化原因: {reasons or '無'}\n"
        f"基準比較: {comparisons or '資料建立中'}"
    )


def personal_record_fields(record: RPPGRecord) -> dict:
    parsed = None
    if record.personal_analysis:
        try:
            parsed = json.loads(record.personal_analysis)
        except Exception:
            parsed = None
    return {
        "personal_baseline_count": record.personal_baseline_count,
        "personal_baseline_window_days": record.personal_baseline_window_days,
        "personal_confidence": record.personal_confidence,
        "personal_risk_score": record.personal_risk_score,
        "personal_status": record.personal_status,
        "personalized_analysis": parsed,
    }

def generate_gemini_health_advice(heart_rate: int, sdnn: float, stress_score: int, health_light: str, monitoring_mode: str = "GENERAL", respiratory_rate=None, irregular_pulse_score=None, personalized_analysis: Optional[dict] = None) -> Optional[dict]:
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

【個人化健康分析】
{personalized_prompt_context(personalized_analysis)}

請把「是否偏離這位使用者自己的基準」和「一般風險門檻」分開說明。個人偏差只能作為重新量測/追蹤提示，不可直接下疾病診斷。

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
- 個人化狀態: {personal_record_fields(record).get("personalized_analysis", {}).get("status_label", "基準建立中") if personal_record_fields(record).get("personalized_analysis") else "基準建立中"}
- 個人化分析可信度: {record.personal_confidence if record.personal_confidence is not None else "未建立"} / 100（代表資料充分度與訊號品質，不是診斷準確率）
- 個人14天基準筆數: {record.personal_baseline_count if record.personal_baseline_count is not None else 0}
- 個人化摘要: {personal_record_fields(record).get("personalized_analysis", {}).get("summary", "尚未建立完整個人基準") if personal_record_fields(record).get("personalized_analysis") else "尚未建立完整個人基準"}
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
                       signal_quality=None, spo2_experimental=None, personalized_analysis=None):
    color_map = {
        "GREEN": "#1DB954",
        "YELLOW": "#FFB800",
        "RED": "#FF4D4D"
    }
    header_color = color_map.get(health_light, "#1DB954")
    light_text = "良好 🟢" if health_light == "GREEN" else ("輕微偏高 🟡" if health_light == "YELLOW" else "需要注意 🔴")

    p = personalized_analysis or {}
    p_reasons = "、".join((p.get("reasons") or [])[:2])
    personal_text = (
        f"👤 個人化分析：{p.get('status_label', '基準建立中')}\n"
        f"14天基準：{p.get('baseline_count', 0)} 筆  ·  分析可信度：{p.get('confidence', '--')}/100\n"
        f"{p_reasons or '持續量測可建立更完整的個人基準'}"
    )

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
                {"type": "text", "text": personal_text, "size": "xs", "color": "#2563EB", "wrap": True, "margin": "md"},
                {"type": "text", "text": "※ 個人化可信度代表訊號品質與歷史資料充分度，不是醫療診斷準確率。", "size": "xxs", "color": "#999999", "wrap": True, "margin": "xs"},
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
                           hrv_rmssd=None, respiratory_rate=None, irregular_pulse_score=None, signal_quality=None, spo2_experimental=None,
                           personalized_analysis=None):
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
    if not token or token.startswith("你的") or token == "YOUR_LINE_CHANNEL_ACCESS_TOKEN":
        print("[LINE Push Warning] 未設定正確的 LINE_CHANNEL_ACCESS_TOKEN，跳過推播。")
        return

    if not (user_id and user_id.startswith("U") and len(user_id) == 33):
        print(f"[LINE Push Warning] user_id ({user_id}) 不是有效的 LINE User ID，跳過推播。")
        return

    flex_payload = build_flex_message(
        heart_rate, stress_score, health_light, summary, advice,
        hrv_rmssd, respiratory_rate, irregular_pulse_score, signal_quality, spo2_experimental, personalized_analysis
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


class StaffLoginRequest(BaseModel):
    username: str
    password: str


# ----------------------------------------------------
# 8. FastAPI 應用程式與安全 CORS 設定
# ----------------------------------------------------
app = FastAPI(title="PharmPulse Backend API", version="1.5.0", lifespan=lifespan)

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


@app.middleware("http")
async def staff_access_guard(request: Request, call_next):
    """Protect pharmacy/hospital pages and APIs with role-based staff authentication."""
    path = request.url.path
    normalized = path.rstrip("/") or "/"

    required_role = None
    login_path = "/staff/login"

    if normalized == "/pharmacy" or path.startswith("/api/v1/pharmacy/") or normalized == "/api/v1/pharmacy":
        required_role = "PHARMACIST"
    elif normalized == "/hospital" or path.startswith("/api/v1/hospital/") or normalized == "/api/v1/hospital":
        required_role = "DOCTOR"

    if not required_role:
        return await call_next(request)

    db = SessionLocal()
    try:
        staff = get_staff_user_from_request(request, db)
    finally:
        db.close()

    if not staff:
        if path.startswith("/api/"):
            return JSONResponse(
                status_code=401,
                content={"status": "error", "detail": "Staff login required"}
            )
        return RedirectResponse(url=login_path, status_code=303)

    if staff.role != required_role:
        if path.startswith("/api/"):
            return JSONResponse(
                status_code=403,
                content={"status": "error", "detail": "Insufficient staff role"}
            )
        return RedirectResponse(url=f"{login_path}?role_error=1", status_code=303)

    request.state.staff_user = staff
    return await call_next(request)

# ----------------------------------------------------
# 9. Staff login / logout (藥師與醫院端)
# ----------------------------------------------------
@app.get("/staff/login")
def serve_staff_login():
    if os.path.exists("staff_login.html"):
        return FileResponse(
            "staff_login.html",
            headers=NO_CACHE_HEADERS if "NO_CACHE_HEADERS" in globals() else None
        )
    raise HTTPException(status_code=404, detail="staff_login.html not found")


@app.get("/pharmacy/login")
def legacy_pharmacy_login():
    return RedirectResponse(url="/staff/login", status_code=303)


@app.get("/hospital/login")
def legacy_hospital_login():
    return RedirectResponse(url="/staff/login", status_code=303)


@app.post("/api/v1/staff/login")
def staff_login(req: StaffLoginRequest, request: Request, db: Session = Depends(get_db)):
    username = (req.username or "").strip()
    staff = db.query(StaffUser).filter(StaffUser.username == username).first()

    # Same response for unknown username/wrong password to avoid account enumeration.
    if (
        not staff
        or not staff.is_active
        or staff.role not in {"PHARMACIST", "DOCTOR"}
        or not verify_staff_password(req.password or "", staff.password_hash)
    ):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")

    destination = "/pharmacy" if staff.role == "PHARMACIST" else "/hospital"

    staff.last_login_at = datetime.now(timezone.utc)
    db.commit()

    token = create_staff_session_token(staff)
    response = JSONResponse({
        "status": "success",
        "destination": destination,
        "staff": {
            "id": staff.id,
            "username": staff.username,
            "display_name": staff.display_name,
            "role": staff.role
        }
    })
    response.set_cookie(
        key=STAFF_COOKIE_NAME,
        value=token,
        max_age=STAFF_SESSION_HOURS * 3600,
        httponly=True,
        secure=(request.url.scheme == "https"),
        samesite="lax",
        path="/"
    )
    return response


@app.get("/api/v1/staff/me")
def staff_me(request: Request, db: Session = Depends(get_db)):
    staff = get_staff_user_from_request(request, db)
    if not staff:
        raise HTTPException(status_code=401, detail="Staff login required")
    return {
        "status": "success",
        "staff": {
            "id": staff.id,
            "username": staff.username,
            "display_name": staff.display_name,
            "role": staff.role,
            "last_login_at": utc_iso(staff.last_login_at)
        }
    }


@app.post("/api/v1/staff/logout")
def staff_logout():
    response = JSONResponse({"status": "success", "message": "已登出"})
    response.delete_cookie(
        key=STAFF_COOKIE_NAME,
        path="/",
        samesite="lax"
    )
    return response


# ----------------------------------------------------
# 10. 靜態頁面與 LINE Webhook 入口
# ----------------------------------------------------
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/")
@app.get("/liff")
def serve_liff():
    """提供 LIFF 前端靜態頁面"""
    if os.path.exists("index.html"):
        return FileResponse(
            "index.html",
            headers=NO_CACHE_HEADERS
        )
    return {"status": "online", "message": "PharmPulse API Service is running"}

@app.get("/my-health")
def serve_my_health():
    """民眾本人健康紀錄頁，LINE LIFF 使用。"""
    if os.path.exists("my_health.html"):
        return FileResponse(
            "my_health.html",
            headers=NO_CACHE_HEADERS
        )
    raise HTTPException(status_code=404, detail="my_health.html not found")


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

class MyHistoryRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    days: int = 7

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

@app.post("/api/v1/user/my-history")
def get_my_health_history(req: MyHistoryRequest, db: Session = Depends(get_db)):
    """民眾本人近 N 天健康紀錄。LIFF 有 id_token 時會驗證 LINE 身分。"""
    try:
        if req.id_token and not verify_line_id_token(req.id_token, req.user_line_id):
            raise HTTPException(status_code=401, detail="LINE 身分驗證失敗")

        days = max(1, min(int(req.days or 7), 30))
        since = datetime.now(timezone.utc) - timedelta(days=days)

        records = (
            db.query(RPPGRecord)
            .filter(
                or_(
                    RPPGRecord.user_line_id == req.user_line_id,
                    RPPGRecord.user_uuid == req.user_line_id
                ),
                RPPGRecord.created_at >= since
            )
            .order_by(RPPGRecord.created_at.desc(), RPPGRecord.id.desc())
            .all()
        )

        result = []
        for r in records:
            personal = personal_record_fields(r)
            result.append({
                "id": r.id,
                "heart_rate": r.heart_rate,
                "hrv_sdnn": r.hrv_sdnn,
                "hrv_rmssd": r.hrv_rmssd,
                "respiratory_rate": r.respiratory_rate,
                "stress_score": r.stress_score,
                "signal_quality": r.signal_quality,
                "signal_quality_label": r.signal_quality_label,
                "health_light": r.health_light,
                "monitoring_mode": r.monitoring_mode,
                "monitoring_mode_label": r.monitoring_mode_label,
                "mode_alert_reasons": r.mode_alert_reasons,
                "personal_baseline_count": personal.get("personal_baseline_count"),
                "personal_confidence": personal.get("personal_confidence"),
                "personal_risk_score": personal.get("personal_risk_score"),
                "personal_status": personal.get("personal_status"),
                "personalized_analysis": personal.get("personalized_analysis"),
                "created_at": utc_iso(r.created_at)
            })

        return {
            "status": "success",
            "days": days,
            "count": len(result),
            "records": result
        }

    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "count": 0,
            "records": [],
            "message": str(e)
        }


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

        # 個人化健康分析：使用本次量測前的 14 天高品質紀錄建立 robust baseline。
        personalized_analysis = build_personalized_health_analysis(
            db=db,
            user_line_id=req.user_line_id,
            monitoring_mode=monitoring_mode,
            heart_rate=hr,
            hrv_sdnn=sdnn,
            stress_score=stress,
            metrics=advanced_metrics,
        )

        # 個人偏差只可將 GREEN 升級成 YELLOW「建議重新確認」；不單獨產生 RED。
        if personalized_analysis.get("should_escalate") and light == "GREEN":
            light = "YELLOW"
            mode_alert_reasons = [
                "相較個人 14 天基準出現明顯變化，建議休息後重新量測"
            ] + list(mode_alert_reasons)

        # AI 只負責衛教文字，不負責決定燈號。
        ai_res = generate_gemini_health_advice(
            heart_rate=hr, sdnn=sdnn, stress_score=stress, health_light=light,
            monitoring_mode=monitoring_mode,
            respiratory_rate=advanced_metrics.get("respiratory_rate"),
            irregular_pulse_score=advanced_metrics.get("irregular_pulse_score"),
            personalized_analysis=personalized_analysis,
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

        # 若沒有 LLM，仍用 deterministic 個人化引擎補上個人基準脈絡。
        if not ai_res:
            p_status = personalized_analysis.get("status")
            if p_status in {"DEVIATION", "WATCH"} and light != "RED":
                summary = personalized_analysis.get("summary") or summary
                advice = (
                    "👤 【個人化追蹤建議】\n"
                    "1. 請先休息 10 分鐘並在相同姿勢、光線下重新量測。\n"
                    "2. 若相較個人基準的偏差連續出現，建議由藥師協助確認血壓、用藥與近期生活狀況。\n"
                    "3. 若伴隨明顯不適或數值快速惡化，請依醫療專業建議就醫。"
                )
            elif p_status == "BUILDING" and light == "GREEN":
                summary = personalized_analysis.get("summary") or summary

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
            personal_baseline_count=personalized_analysis.get("baseline_count"),
            personal_baseline_window_days=personalized_analysis.get("baseline_window_days"),
            personal_confidence=personalized_analysis.get("confidence"),
            personal_risk_score=personalized_analysis.get("risk_score"),
            personal_status=personalized_analysis.get("status"),
            personal_analysis=json.dumps(personalized_analysis, ensure_ascii=False),
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
            spo2_experimental=advanced_metrics.get("spo2_experimental"),
            personalized_analysis=personalized_analysis
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
                "personalized_analysis": personalized_analysis,
                "personal_baseline_count": personalized_analysis.get("baseline_count"),
                "personal_confidence": personalized_analysis.get("confidence"),
                "personal_risk_score": personalized_analysis.get("risk_score"),
                "personal_status": personalized_analysis.get("status"),
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
                **personal_record_fields(r),
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

def _latest_measurements_by_user(db: Session):
    """
    每位民眾只保留最新一筆量測，不限制燈號。
    用於藥師端查詢目前最新狀態為 GREEN 的民眾。
    """
    rows = (
        db.query(RPPGRecord)
        .order_by(RPPGRecord.created_at.desc(), RPPGRecord.id.desc())
        .all()
    )

    latest_by_user = {}
    for row in rows:
        user_id = row.user_line_id or row.user_uuid
        if not user_id or user_id in latest_by_user:
            continue
        latest_by_user[user_id] = row

    return list(latest_by_user.values())


def _current_pending_alerts(db: Session):
    """
    每位民眾只保留最新一筆未結案 RED / YELLOW 警示。
    這讓警示卡、待處理人數與 Dashboard KPI 使用同一個統計口徑。
    """
    rows = (
        db.query(RPPGRecord)
        .filter(RPPGRecord.health_light.in_(["RED", "YELLOW"]))
        .filter(or_(RPPGRecord.is_resolved == False, RPPGRecord.is_resolved.is_(None)))
        .order_by(RPPGRecord.created_at.desc(), RPPGRecord.id.desc())
        .all()
    )

    latest_by_user = {}
    for row in rows:
        user_id = row.user_line_id or row.user_uuid
        if not user_id or user_id in latest_by_user:
            continue
        latest_by_user[user_id] = row

    return list(latest_by_user.values())

@app.get("/api/v1/pharmacy/alerts")
def get_pharmacy_alerts(light: Optional[str] = None, db: Session = Depends(get_db)):
    """取得藥局端清單。預設為未結案 RED/YELLOW；light=GREEN 時回傳目前最新狀態為 GREEN 的民眾。"""
    try:
        requested_light = (light or "").strip().upper()

        if requested_light == "GREEN":
            # GREEN 是歷史正常量測查詢，不是待處置警示。
            # 直接回傳近 7 天全部 GREEN 紀錄，確保剛完成的正常量測可立即被藥師查到。
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            records = (
                db.query(RPPGRecord)
                .filter(RPPGRecord.health_light == "GREEN")
                .filter(RPPGRecord.created_at >= seven_days_ago)
                .order_by(RPPGRecord.created_at.desc(), RPPGRecord.id.desc())
                .limit(300)
                .all()
            )
        else:
            records = _current_pending_alerts(db)
        
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
                **personal_record_fields(r),
                "stress_score": r.stress_score,
                "health_light": r.health_light,
                "summary": r.summary,
                "action_advice": r.action_advice,
                "created_at": utc_iso(r.created_at)
            })
        return {
            "status": "success",
            "list_type": "GREEN" if requested_light == "GREEN" else "ALERT",
            "alerts": result
        }
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

        # 3) KPI 與警示清單採相同口徑：每位民眾只看最新一筆未結案 RED / YELLOW。
        current_alerts = _current_pending_alerts(db)
        red_users = {mode: set() for mode in mode_keys}
        yellow_users = {mode: set() for mode in mode_keys}

        for r in current_alerts:
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
            stats[mode]["pending_patients"] = len(red_users[mode] | yellow_users[mode])

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

        # 追蹤日期是藥師以台灣日期輸入，KPI 也應以 Asia/Taipei（UTC+8）的「今天」判斷。
        taipei_tz = timezone(timedelta(hours=8))
        today = datetime.now(taipei_tz).date()
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
            "pending_patients": len(current_alerts),
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
    """
    藥師完成此個案關懷時，將同一民眾截至該筆量測時間的舊未處理紅黃燈一併結案。
    新於該筆的警示不會被誤關閉。
    """
    try:
        rec = db.query(RPPGRecord).filter(RPPGRecord.id == record_id).first()
        if not rec:
            raise HTTPException(status_code=404, detail="Record not found")

        user_id = rec.user_line_id or rec.user_uuid
        if not user_id:
            rec.is_resolved = True
            resolved_count = 1
        else:
            q = (
                db.query(RPPGRecord)
                .filter(
                    or_(
                        RPPGRecord.user_line_id == user_id,
                        RPPGRecord.user_uuid == user_id
                    )
                )
                .filter(RPPGRecord.health_light.in_(["RED", "YELLOW"]))
                .filter(or_(RPPGRecord.is_resolved == False, RPPGRecord.is_resolved.is_(None)))
            )
            if rec.created_at is not None:
                q = q.filter(RPPGRecord.created_at <= rec.created_at)

            resolved_count = q.update(
                {RPPGRecord.is_resolved: True},
                synchronize_session=False
            )

        db.commit()
        return {
            "status": "success",
            "message": "Patient alerts marked as resolved",
            "resolved_count": resolved_count
        }
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
                **personal_record_fields(r),
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


@app.get("/api/v1/pharmacy/interventions")
def get_all_pharmacy_interventions(db: Session = Depends(get_db)):
    """藥師主頁使用：取得所有已儲存的關懷紀錄，最新一筆排在最前面。"""
    try:
        records = (
            db.query(PharmacyIntervention)
            .order_by(PharmacyIntervention.created_at.desc(), PharmacyIntervention.id.desc())
            .limit(500)
            .all()
        )

        result = []
        for r in records:
            source_rppg = None
            if r.rppg_record_id:
                source_rppg = (
                    db.query(RPPGRecord)
                    .filter(RPPGRecord.id == r.rppg_record_id)
                    .first()
                )

            if not source_rppg:
                source_rppg = (
                    db.query(RPPGRecord)
                    .filter(
                        or_(
                            RPPGRecord.user_line_id == r.user_line_id,
                            RPPGRecord.user_uuid == r.user_line_id
                        )
                    )
                    .order_by(RPPGRecord.created_at.desc(), RPPGRecord.id.desc())
                    .first()
                )

            profile = (
                db.query(PatientChronicProfile)
                .filter(PatientChronicProfile.user_line_id == r.user_line_id)
                .first()
            )

            mode = normalize_monitoring_mode(
                profile.monitoring_mode if profile and profile.monitoring_mode
                else (source_rppg.monitoring_mode if source_rppg else "GENERAL")
            )

            result.append({
                "id": r.id,
                "user_line_id": r.user_line_id,
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
                "created_at": utc_iso(r.created_at),
                "monitoring_mode": mode,
                "monitoring_mode_label": MONITORING_MODES[mode]["label"],
                "source_health_light": source_rppg.health_light if source_rppg else None,
                "source_created_at": utc_iso(source_rppg.created_at) if source_rppg else None,
                "chronic_conditions": profile.chronic_conditions if profile else None
            })

        return {
            "status": "success",
            "records": result,
            "count": len(result)
        }

    except Exception as e:
        return {
            "status": "error",
            "records": [],
            "count": 0,
            "message": str(e)
        }


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



@app.get("/api/v1/hospital/patient/{user_line_id}/care-history")
def get_hospital_patient_care_history(
    user_line_id: str,
    db: Session = Depends(get_db)
):
    """
    醫師端使用：查看指定轉介民眾的完整藥師關懷歷史。
    此路徑位於 /api/v1/hospital/，因此會由 staff_access_guard 限制為 DOCTOR。
    """
    try:
        records = (
            db.query(PharmacyIntervention)
            .filter(PharmacyIntervention.user_line_id == user_line_id)
            .order_by(
                PharmacyIntervention.created_at.desc(),
                PharmacyIntervention.id.desc()
            )
            .all()
        )

        result = []

        for r in records:
            source_rppg = None

            if r.rppg_record_id:
                source_rppg = (
                    db.query(RPPGRecord)
                    .filter(RPPGRecord.id == r.rppg_record_id)
                    .first()
                )

            if not source_rppg:
                source_rppg = (
                    db.query(RPPGRecord)
                    .filter(
                        or_(
                            RPPGRecord.user_line_id == user_line_id,
                            RPPGRecord.user_uuid == user_line_id
                        )
                    )
                    .order_by(
                        RPPGRecord.created_at.desc(),
                        RPPGRecord.id.desc()
                    )
                    .first()
                )

            profile = (
                db.query(PatientChronicProfile)
                .filter(PatientChronicProfile.user_line_id == user_line_id)
                .first()
            )

            mode = normalize_monitoring_mode(
                profile.monitoring_mode
                if profile and profile.monitoring_mode
                else (
                    source_rppg.monitoring_mode
                    if source_rppg
                    else "GENERAL"
                )
            )

            result.append({
                "id": r.id,
                "user_line_id": r.user_line_id,
                "rppg_record_id": r.rppg_record_id,

                "care_method": r.care_method,
                "symptoms": r.symptoms,

                "systolic_bp": r.systolic_bp,
                "diastolic_bp": r.diastolic_bp,
                "pulse": r.pulse,

                "pharmacist_note": r.pharmacist_note,
                "action_result": r.action_result,

                "followup_date": (
                    r.followup_date.strftime("%Y-%m-%d")
                    if r.followup_date
                    else None
                ),

                "referral_status": r.referral_status,
                "created_at": utc_iso(r.created_at),

                "monitoring_mode": mode,
                "monitoring_mode_label": MONITORING_MODES[mode]["label"],
                "chronic_conditions": (
                    profile.chronic_conditions if profile else None
                ),

                "source_health_light": (
                    source_rppg.health_light if source_rppg else None
                ),
                "source_created_at": (
                    utc_iso(source_rppg.created_at)
                    if source_rppg
                    else None
                )
            })

        return {
            "status": "success",
            "user_line_id": user_line_id,
            "count": len(result),
            "records": result
        }

    except Exception as e:
        return {
            "status": "error",
            "user_line_id": user_line_id,
            "count": 0,
            "records": [],
            "message": str(e)
        }


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
                    **personal_record_fields(rppg),
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