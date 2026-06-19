import os
import ssl
import requests
import time
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

ssl._create_default_https_context = ssl._create_unverified_context

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")

# ✨ الأزواج التي سنفحصها (يجب أن تكون مرتبطة إحصائياً)
# الصيغة لـ yfinance هي إضافة -USD بدلاً من /USDT
PAIRS = [
    ("BTC-USD", "ETH-USD", "BTCUSDT", "ETHUSDT"),
    ("SOL-USD", "BNB-USD", "SOLUSDT", "BNBUSDT"),
    ("AVAX-USD", "LINK-USD", "AVAXUSDT", "LINKUSDT"),
    ("DOT-USD", "NEAR-USD", "DOTUSDT", "NEARUSDT")
]

LEVERAGE = "10"
TIMEFRAME = "1D" # Daily

def get_decimals(price):
    if price > 100: return 2
    elif price > 1: return 3
    elif price > 0.01: return 5
    else: return 8

def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHANNEL_ID, "text": text, "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload)
        time.sleep(1) # Small pause between the two messages
    except Exception as e:
        print(f"Error sending message: {e}")

def analyze_pairs():
    print("Fetching daily data and calculating Z-Scores...")
    
    for yf_ticker1, yf_ticker2, name1, name2 in PAIRS:
        try:
            # 1. سحب البيانات لآخر 6 أشهر
            data = yf.download([yf_ticker1, yf_ticker2], period="6mo")['Close']
            data = data.dropna()
            p1 = data[yf_ticker1]
            p2 = data[yf_ticker2]
            
            # 2. فحص التكامل المشترك (Cointegration)
            score, p_value, _ = sm.tsa.stattools.coint(p1, p2)
            
            # إذا كانت القيمة أكبر من 0.05 فلا توجد علاقة حقيقية، تجاوز الزوج
            if p_value > 0.05:
                print(f"{name1}/{name2} are not cointegrated. Skipping.")
                continue
                
            # 3. حساب الفجوة (Spread) ونسبة التحوط (Hedge Ratio)
            X = sm.add_constant(p2)
            model = sm.OLS(p1, X).fit()
            hedge_ratio = model.params.iloc[1]
            spread = p1 - (hedge_ratio * p2)
            
            # 4. حساب Z-Score
            spread_mean = spread.rolling(window=20).mean()
            spread_std = spread.rolling(window=20).std()
            z_score = (spread - spread_mean) / spread_std
            current_z = z_score.iloc[-1]
            
            # تجنب القسمة على صفر
            if pd.isna(current_z) or spread_std.iloc[-1] < 1e-8:
                continue
                
            print(f"{name1}/{name2} Z-Score: {current_z:.2f}")
            
            # 5. اتخاذ القرار (إذا تجاوز 2.0 أو انخفض عن -2.0)
            if abs(current_z) > 2.0:
                current_p1 = p1.iloc[-1]
                current_p2 = p2.iloc[-1]
                dec1 = get_decimals(current_p1)
                dec2 = get_decimals(current_p2)
                
                # حساب التغير المطلوب في الفجوة للعودة للمتوسط
                delta_z_tp1 = abs(current_z - 1.0)
                delta_z_tp2 = abs(current_z - 0.0)
                delta_z_sl = abs(current_z - 3.0) # إذا وصل لـ 3 يعني الاستراتيجية خسرت
                
                std_val = spread_std.iloc[-1]
                
                # ✨ إعداد رسالة العملة الأولى
                zone1_low = round(current_p1 * 0.999, dec1)
                zone1_high = round(current_p1 * 1.001, dec1)
                
                # ✨ إعداد رسالة العملة الثانية
                zone2_low = round(current_p2 * 0.999, dec2)
                zone2_high = round(current_p2 * 1.001, dec2)

                if current_z > 2.0:
                    # الفجوة واسعة جداً: العملة الأولى مبالغ فيها (نبيعها)، الثانية أقل (نشتريها)
                    dir1, emoji1 = "Short", "📉"
                    dir2, emoji2 = "Long", "📈"
                    
                    tp1_p1 = round(current_p1 + (delta_z_tp1 * std_val), dec1)
                    tp2_p1 = round(current_p1 + (delta_z_tp2 * std_val), dec1)
                    sl_p1 = round(current_p1 - (delta_z_sl * std_val), dec1)
                    
                    tp1_p2 = round(current_p2 - (delta_z_tp1 * std_val * hedge_ratio), dec2)
                    tp2_p2 = round(current_p2 - (delta_z_tp2 * std_val * hedge_ratio), dec2)
                    sl_p2 = round(current_p2 + (delta_z_sl * std_val * hedge_ratio), dec2)
                    
                else: # current_z < -2.0
                    # الفجوة ضيقة جداً: العملة الأولى أقل (نشتريها)، الثانية مبالغ فيها (نبيعها)
                    dir1, emoji1 = "Long", "📈"
                    dir2, emoji2 = "Short", "📉"
                    
                    tp1_p1 = round(current_p1 - (delta_z_tp1 * std_val), dec1)
                    tp2_p1 = round(current_p1 - (delta_z_tp2 * std_val), dec1)
                    sl_p1 = round(current_p1 + (delta_z_sl * std_val), dec1)
                    
                    tp1_p2 = round(current_p2 + (delta_z_tp1 * std_val * hedge_ratio), dec2)
                    tp2_p2 = round(current_p2 + (delta_z_tp2 * std_val * hedge_ratio), dec2)
                    sl_p2 = round(current_p2 - (delta_z_sl * std_val * hedge_ratio), dec2)

                # ✨ صياغة الرسالتين لتتوافق مع كورنيكس
                text1 = f"📩 #{name1} {TIMEFRAME} | Pairs Trade\n{emoji1} {dir1} Entry Zone: {zone1_low}-{zone1_high}\n⚡ Leverage: {LEVERAGE}x\n\n🎯 Strategy: Statistical Arbitrage (Z: {current_z:.2f})\n\n⏳ Signal Details:\nTarget 1: {tp1_p1}\nTarget 2: {tp2_p1}\n\n🔺 Stop-Loss: {sl_p1}\n💡 Paired with {name2}. Close both when mean reverts."
                
                text2 = f"📩 #{name2} {TIMEFRAME} | Pairs Trade\n{emoji2} {dir2} Entry Zone: {zone2_low}-{zone2_high}\n⚡ Leverage: {LEVERAGE}x\n\n🎯 Strategy: Statistical Arbitrage (Z: {current_z:.2f})\n\n⏳ Signal Details:\nTarget 1: {tp1_p2}\nTarget 2: {tp2_p2}\n\n🔺 Stop-Loss: {sl_p2}\n💡 Paired with {name1}. Close both when mean reverts."

                print(f"Sending Pairs Signal: {name1}({dir1}) / {name2}({dir2})")
                send_message(text1)
                send_message(text2)
                
        except Exception as e:
            print(f"Error analyzing {yf_ticker1}/{yf_ticker2}: {e}")

if __name__ == "__main__":
    print("Pairs Trading Bot started...")
    analyze_pairs()
    print("Daily scan finished.")