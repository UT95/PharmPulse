import os
import math
import json
import re
import random
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager
from scipy.signal import butter, filtfilt, find_peaks

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
    # 備援：若使用 linebot v3 SDK 或環境未完整安裝
    LineBotApi = WebhookHandler = InvalidSignatureError = MessageEvent = TextMessage = TextSendMessage = None

# ✅ 新版 Google Gemini SDK
from google import genai

# ----------------------------------------------------
# 1. 環境變數與 Google Gemini / 資料庫設定
# ----------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
LINE_CHANNEL_ID = os.getenv("LINE_CHANNEL_ID", "")  # 用於 id_token 驗證 audience (aud)
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
LIFF_URL = os.getenv("LIFF_URL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash").strip()

handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if (LineBotApi and LINE_CHANNEL_ACCESS_TOKEN) else None

# ✅ 初始化 Google Gemini SDK (新版 google-genai Client)
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[Gemini AI] Google Gemini API 已成功初始化 (Model: {GEMINI_MODEL_NAME})。")
else:
    print("[Gemini AI Warning] 未設定 GEMINI_API_KEY，將使用預設規則生成衛教建議。")

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
    stress_score = Column(Integer, nullable=False)
    health_light = Column(String(20), nullable=False)
    is_resolved = Column(Boolean, default=False)  # 標記藥師是否已完成電話/臨櫃關懷
    summary = Column(Text, nullable=True)
    action_advice = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

# ----------------------------------------------------
# 3. 自動檢查與修復 PostgreSQL 欄位
# ----------------------------------------------------
def auto_migrate_db():
    if "postgresql" in DATABASE_URL:
        try:
            with engine.begin() as conn:
                columns_to_add = [
                    ("user_uuid", "VARCHAR(255)"),
                    ("user_line_id", "VARCHAR(255)"),
                    ("is_resolved", "BOOLEAN DEFAULT FALSE"),
                    ("summary", "TEXT"),
                    ("action_advice", "TEXT")
                ]
                for col_name, col_type in columns_to_add:
                    try:
                        conn.execute(text(f"ALTER TABLE rppg_records ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                    except Exception as e:
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
# 5. Gemini AI 衛教生成與強健 JSON 解析模組
# ----------------------------------------------------
def extract_json_from_text(text_content: str) -> Optional[dict]:
    """強健的 JSON 解析器，能自動從 Gemini 的 Markdown 回送中提取 JSON"""
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

def generate_gemini_health_advice(heart_rate: int, sdnn: float, stress_score: int, health_light: str) -> Optional[dict]:
    """使用 Google Gemini 根據生理數據產生專業且溫暖的藥局衛教摘要與建議"""
    if not gemini_client:
        print("[Gemini AI Warning] gemini_client 未初始化，切換至預設規則產出衛教內容。")
        return None

    try:
        prompt = f"""
你是一位 PharmPulse 社區智慧藥局系統的「AI 臨床衛教藥師」。
請根據以下民眾透過手機鏡頭 rPPG 測得的即時生理數據，撰寫一份簡明、具專業衛教價值且充滿關懷的分析報告。

【生理數據資料】
- 心率 (Heart Rate): {heart_rate} BPM
- 心率變異度 (HRV SDNN): {sdnn} ms
- 壓力指數 (Stress Score): {stress_score} / 100
- 健康狀態燈號 (Health Light): {health_light} (GREEN: 良好穩定, YELLOW: 輕度疲勞/壓力上升, RED: 顯著異常/負擔過重)

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

        # ✅ 改用新版 client.models.generate_content 呼叫
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.4
            }
        )

        if response and response.text:
            parsed_data = extract_json_from_text(response.text)
            if parsed_data and "summary" in parsed_data and "action_advice" in parsed_data:
                return parsed_data
        return None

    except Exception as e:
        print(f"[Gemini AI Error] 呼叫 Gemini 產生衛教建議時發生異常: {str(e)}")
        return None

def generate_ai_chat_response(db: Session, user_line_id: str, question: str, record_id: Optional[int] = None) -> str:
    """共通 Gemini 對話邏輯：供 LINE Webhook 與 API 共同呼叫"""
    if not gemini_client:
        return "感謝您的諮詢！目前 AI 衛教諮詢服務暫時維護中。若您有緊急身體不適，請務必先就醫或尋求社區藥師協助。"

    # 取得關聯的歷史檢測紀錄
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

    try:
        # ✅ 改用新版 client.models.generate_content 呼叫
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt
        )
        if response and response.text:
            return response.text.strip()
        return "目前暫時無法回應，請稍後再試。"
    except Exception as e:
        print(f"[Gemini Chat Error] {str(e)}")
        return "抱歉，AI 衛教系統處理時發生異常，請稍後再試。"

# ----------------------------------------------------
# 6. 訊號處理與進階 HRV 演算法
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
# 7. LINE Flex Message + Quick Reply 互動卡片
# ----------------------------------------------------
def build_flex_message(heart_rate: int, stress_score: int, health_light: str, summary: str, advice: str):
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
# ----------------------------------------------------
# 7.5 LINE 主動推播訊息處理
# ----------------------------------------------------
def send_line_push_message(user_id: str, heart_rate: int, stress_score: int, health_light: str, summary: str, advice: str):
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
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
# 9. 前端頁面託管與 LINE Webhook 處理
# ----------------------------------------------------
@app.get("/")
@app.get("/liff")
def serve_liff():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "message": "PharmPulse API Service is running"}

@app.get("/pharmacy")
def serve_pharmacy():
    if os.path.exists("pharmacy.html"):
        return FileResponse("pharmacy.html")
    return {"status": "error", "message": "pharmacy.html not found"}

# 註冊 LINE Webhook 對話處理程序
if handler and MessageEvent:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_message(event):
        """當使用者向 LINE 官方帳號傳送文字訊息時的自動 AI 衛教對話處理"""
        if not line_bot_api:
            print("[LINE Webhook Warning] line_bot_api 未能正常初始化，跳過訊息回覆。")
            return

        user_text = event.message.text
        user_id = event.source.user_id

        # 開啟獨立 Session 查詢檢測資料並呼叫 Gemini 生成回應
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
# 10. Request / Response Pydantic Schemas
# ----------------------------------------------------
class AnalyzeRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    rgb_signals: Union[List[List[float]], List[float]]
    fps: int = 30

class ConsentRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    terms_version: str = "v1.0"

class AIChatRequest(BaseModel):
    user_line_id: str
    id_token: Optional[str] = None
    question: str
    record_id: Optional[int] = None

# ----------------------------------------------------
# 11. 使用者同意與生理分析 API (含 Gemini 整合)
# ----------------------------------------------------
@app.get("/api/v1/user/consent-status/{user_line_id}")
def check_consent(user_line_id: str, db: Session = Depends(get_db)):
    try:
        record = db.query(UserConsent).filter(UserConsent.user_line_id == user_line_id).first()
        if record and record.agreed == "true":
            return {"status": "success", "agreed": True}
        return {"status": "success", "agreed": False}
    except Exception as e:
        return {"status": "error", "agreed": False, "message": str(e)}

@app.post("/api/v1/user/consent")
def save_consent(req: ConsentRequest, db: Session = Depends(get_db)):
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
    if req.id_token and not verify_line_id_token(req.id_token, req.user_line_id):
        raise HTTPException(status_code=401, detail="Invalid LINE id_token verification failed")

    try:
        user_id_upper = req.user_line_id.upper()

        # A. 測試模擬模式
        if "RED" in user_id_upper or "HIGH" in user_id_upper:
            hr = random.randint(105, 125)
            sdnn = round(random.uniform(15.0, 22.0), 1)
            stress = random.randint(82, 95)
        elif "YELLOW" in user_id_upper or "WARN" in user_id_upper:
            hr = random.randint(88, 98)
            sdnn = round(random.uniform(28.0, 36.0), 1)
            stress = random.randint(62, 74)
        elif "GREEN" in user_id_upper or "NORMAL" in user_id_upper:
            hr = random.randint(65, 76)
            sdnn = round(random.uniform(48.0, 68.0), 1)
            stress = random.randint(20, 42)
        else:
            # B. 正式分析邏輯
            if not req.rgb_signals or len(req.rgb_signals) == 0:
                raise HTTPException(status_code=400, detail="rgb_signals cannot be empty")

            if isinstance(req.rgb_signals[0], list):
                if len(req.rgb_signals) == 3 and len(req.rgb_signals[0]) > 3:
                    green_signal = req.rgb_signals[1]
                elif len(req.rgb_signals[0]) >= 2:
                    green_signal = [frame[1] for frame in req.rgb_signals]
                else:
                    green_signal = [frame[0] for frame in req.rgb_signals]
            else:
                green_signal = req.rgb_signals

            if len(green_signal) < 60:
                hr, sdnn = 72, 38.5
            else:
                hr, sdnn = calculate_rppg_metrics(green_signal, fps=req.fps)

            normalized_sdnn = np.clip(sdnn, 15.0, 80.0)
            calc_stress = 85.0 - ((normalized_sdnn - 15.0) / (80.0 - 15.0)) * 70.0

            if hr > 85:
                calc_stress += (hr - 85) * 0.3

            stress = int(round(np.clip(calc_stress, 15, 95)))

        # C. 判定健康燈號
        if stress > 75 or hr > 100 or hr < 50:
            light = "RED"
        elif stress > 50 or hr >= 85:
            light = "YELLOW"
        else:
            light = "GREEN"

        # D. Gemini AI 衛教與 Fallback 規則
        ai_res = generate_gemini_health_advice(heart_rate=hr, sdnn=sdnn, stress_score=stress, health_light=light)

        if ai_res:
            summary = ai_res.get("summary", "")
            advice = ai_res.get("action_advice", "")
        else:
            if light == "RED":
                summary = "生理數值顯著偏離基準，心血管與自律神經負擔較高。"
                advice = (
                    "🚨 【處置建議】壓力或心率偏高：\n"
                    "1. 請保持環境通風，閉目進行 5 分鐘深呼吸。\n"
                    "2. 建議於今日量測血壓，若連續 2 天數值異常，請至門診複診。\n"
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

        # E. 資料庫寫入與背景推播
        record = RPPGRecord(
            user_uuid=req.user_line_id,
            user_line_id=req.user_line_id,
            heart_rate=hr,
            hrv_sdnn=sdnn,
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
# 12. Gemini 互動式 AI 健康諮詢 API (提供 LIFF 與 Web 端調用)
# ----------------------------------------------------
@app.post("/api/v1/ai/chat")
def ai_health_consultation(req: AIChatRequest, db: Session = Depends(get_db)):
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

# ----------------------------------------------------
# 14. 藥局端專用 API (Pharmacy Alerts & QR Code History)
# ----------------------------------------------------

# (1) 取得未處理的紅燈/黃燈緊急卡片流
@app.get("/api/v1/pharmacy/alerts")
def get_pharmacy_alerts(db: Session = Depends(get_db)):
    try:
        records = db.query(RPPGRecord)\
                    .filter(RPPGRecord.health_light.in_(["RED", "YELLOW"]))\
                    .filter(or_(RPPGRecord.is_resolved == False, RPPGRecord.is_resolved.is_(None)))\
                    .order_by(RPPGRecord.created_at.desc())\
                    .limit(30)\
                    .all()
        
        result = []
        for r in records:
            result.append({
                "id": r.id,
                "user_line_id": r.user_line_id or r.user_uuid,
                "heart_rate": r.heart_rate,
                "hrv_sdnn": r.hrv_sdnn,
                "stress_score": r.stress_score,
                "health_light": r.health_light,
                "summary": r.summary,
                "action_advice": r.action_advice,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })
        return {"status": "success", "alerts": result}
    except Exception as e:
        return {"status": "error", "alerts": [], "message": str(e)}

# (2) 藥師點擊「標示已關懷」
@app.post("/api/v1/pharmacy/resolve/{record_id}")
def resolve_pharmacy_alert(record_id: int, db: Session = Depends(get_db)):
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

# (3) 掃描民眾 QR Code 後，調閱近 7 天歷史紀錄
@app.get("/api/v1/pharmacy/patient/{user_line_id}/history")
def get_patient_7day_history(user_line_id: str, db: Session = Depends(get_db)):
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
                    .order_by(RPPGRecord.created_at.asc())\
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
        return {
            "status": "success",
            "user_line_id": user_line_id,
            "records": result
        }
    except Exception as e:
        return {"status": "error", "records": [], "message": str(e)}

# ----------------------------------------------------
# 15. 本地直接執行進入點
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)