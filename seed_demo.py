"""
PharmPulse 假資料產生器（藥師端紅黃綠量測紀錄）

用法（在專案根目錄、已啟用 .venv）：
    python seed_demo.py            # 新增假資料
    python seed_demo.py --clear    # 只刪除之前產生的假資料
    python seed_demo.py --reset    # 先刪除再重新產生
    python seed_demo.py --yes      # 連線 Render 資料庫時略過確認

所有假資料的 user_line_id 都以 DEMO_ 開頭，不會動到真實資料。
"""
import sys
from datetime import datetime, timezone, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from main import SessionLocal, RPPGRecord, PatientChronicProfile, engine

PREFIX = "DEMO_"

# (user_id, 姓名, 模式, 慢性病, 燈號, 心率, SDNN, RMSSD, 呼吸, 不規律分數, 品質, SpO2, 壓力, 幾小時前, 摘要, 建議)
DEMO = [
    # ---------- 紅燈 ----------
    ("DEMO_R01", "王大明", "CARDIO", "高血壓", "RED", 118, 18.0, 14.0, 22.0, 62, 85, 95.0, 88, 1,
     "靜息心率明顯偏高且脈搏不規律分數偏高。", "建議盡快至藥局以血壓計複測，必要時就醫。"),
    ("DEMO_R02", "林淑芬", "RESPIRATORY", "COPD", "RED", 108, 20.0, 15.0, 28.0, 30, 80, 90.0, 82, 2,
     "呼吸頻率偏高，SpO2 實驗估值偏低。", "建議盡快聯絡確認症狀，必要時轉介醫師。"),
    ("DEMO_R03", "陳志豪", "GENERAL", "無", "RED", 125, 15.0, 11.0, 24.0, 55, 78, 96.0, 92, 5,
     "心率與壓力指數皆明顯偏高。", "建議休息後重測，若持續偏高請就醫評估。"),
    # ---------- 黃燈 ----------
    ("DEMO_Y01", "張美玲", "CARDIO", "高血壓、糖尿病", "YELLOW", 98, 28.0, 22.0, 18.0, 25, 82, 96.0, 65, 3,
     "心率略偏高，HRV 偏低。", "建議近日安排藥局血壓複測並追蹤。"),
    ("DEMO_Y02", "李國華", "RESPIRATORY", "氣喘", "YELLOW", 94, 30.0, 24.0, 21.0, 10, 75, 93.0, 60, 6,
     "呼吸頻率稍高。", "建議觀察症狀，三天內再量測。"),
    ("DEMO_Y03", "黃雅婷", "GENERAL", "無", "YELLOW", 96, 26.0, 20.0, 17.0, 15, 88, 97.0, 68, 12,
     "壓力指數偏高。", "建議充足休息、減少咖啡因，明日再測。"),
    ("DEMO_Y04", "吳建宏", "CARDIO", "高血壓", "YELLOW", 101, 24.0, 18.0, 19.0, 35, 79, 96.0, 70, 20,
     "脈搏規律性略有變化。", "建議到藥局以血壓計確認。"),
    # ---------- 綠燈（藥師端 GREEN 只看近 7 天） ----------
    ("DEMO_G01", "蔡佳穎", "GENERAL", "無", "GREEN", 72, 55.0, 48.0, 15.0, 3, 92, 98.0, 28, 2,
     "各項指標落在正常範圍。", "維持規律作息與運動。"),
    ("DEMO_G02", "許文傑", "CARDIO", "高血壓", "GREEN", 76, 48.0, 40.0, 16.0, 5, 90, 98.0, 32, 8,
     "心率與 HRV 穩定。", "請持續規律服藥與量測。"),
    ("DEMO_G03", "鄭惠敏", "RESPIRATORY", "COPD", "GREEN", 80, 45.0, 38.0, 17.0, 4, 86, 97.0, 35, 30,
     "呼吸頻率與心肺負荷穩定。", "維持目前照護方式。"),
    ("DEMO_G04", "劉俊宏", "GENERAL", "無", "GREEN", 68, 60.0, 52.0, 14.0, 2, 94, 99.0, 22, 50,
     "狀態良好。", "繼續保持。"),
]

MODE_LABEL = {
    "GENERAL": "一般健康監測",
    "CARDIO": "高血壓 / 心血管監測",
    "RESPIRATORY": "COPD / 呼吸慢病監測",
}


def clear(db):
    n1 = db.query(RPPGRecord).filter(RPPGRecord.user_line_id.like(PREFIX + "%")).delete(synchronize_session=False)
    n2 = db.query(PatientChronicProfile).filter(PatientChronicProfile.user_line_id.like(PREFIX + "%")).delete(synchronize_session=False)
    db.commit()
    print(f"已刪除假量測 {n1} 筆、假個人檔案 {n2} 筆")


def seed(db):
    now = datetime.now(timezone.utc)
    for (uid, name, mode, cond, light, hr, sdnn, rmssd, rr, irr, q, spo2,
         stress, hours_ago, summary, advice) in DEMO:
        db.add(PatientChronicProfile(
            user_line_id=uid, display_name=name, monitoring_mode=mode,
            chronic_conditions=None if cond == "無" else cond,
        ))
        db.add(RPPGRecord(
            user_uuid=uid, user_line_id=uid,
            heart_rate=hr, hrv_sdnn=sdnn, hrv_rmssd=rmssd,
            respiratory_rate=rr, irregular_pulse_score=irr,
            signal_quality=q, signal_quality_label="GOOD" if q >= 80 else "FAIR",
            spo2_experimental=spo2,
            monitoring_mode=mode, monitoring_mode_label=MODE_LABEL[mode],
            stress_score=stress, health_light=light,
            is_resolved=False, summary=summary, action_advice=advice,
            created_at=now - timedelta(hours=hours_ago),
        ))
    db.commit()
    r = sum(1 for d in DEMO if d[4] == "RED")
    y = sum(1 for d in DEMO if d[4] == "YELLOW")
    g = sum(1 for d in DEMO if d[4] == "GREEN")
    print(f"已新增：紅 {r} 筆、黃 {y} 筆、綠 {g} 筆")


if __name__ == "__main__":
    args = set(sys.argv[1:])
    url = str(engine.url)
    print("目前連線資料庫：", engine.url.render_as_string(hide_password=True))

    if "postgresql" in url and "--yes" not in args:
        if input("這是線上 PostgreSQL，確定要寫入？(y/N) ").strip().lower() != "y":
            sys.exit("已取消")

    db = SessionLocal()
    try:
        if "--clear" in args:
            clear(db)
        else:
            if "--reset" in args:
                clear(db)
            seed(db)
    finally:
        db.close()