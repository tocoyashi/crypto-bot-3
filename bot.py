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

# Updated to reliable Yahoo Finance tickers
PAIRS = [
    ("ETH-USD", "LTC-USD", "ETHUSDT", "LTCUSDT"),
    ("BNB-USD", "SOL-USD", "BNBUSDT", "SOLUSDT"),
    ("XRP-USD", "ADA-USD", "XRPUSDT", "ADAUSDT"),
    ("BTC-USD", "ETH-USD", "BTCUSDT", "ETHUSDT")
]

LEVERAGE = "10"
TIMEFRAME = "1D"

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
        time.sleep(1)
    except Exception as e:
        print(f"Error sending message: {e}")

def analyze_pairs():
    print("Fetching 1-year daily data and calculating Z-Scores...")
    
    for yf_ticker1, yf_ticker2, name1, name2 in PAIRS:
        try:
            data = yf.download([yf_ticker1, yf_ticker2], period="1y")['Close']
            data = data.dropna()
            
            # Safety check to prevent crash if data is empty
            if len(data) < 30:
                print(f"{name1}/{name2} not enough data. Skipping.")
                continue
                
            p1 = data[yf_ticker1]
            p2 = data[yf_ticker2]
            
            score, p_value, _ = sm.tsa.stattools.coint(p1, p2)
            
            # Loosened to 0.15 because crypto needs more flexibility than traditional stocks
            if p_value > 0.15:
                print(f"{name1}/{name2} p-value: {p_value:.2f}. Skipping.")
                continue
                
            X = sm.add_constant(p2)
            model = sm.OLS(p1, X).fit()
            hedge_ratio = model.params.iloc[1]
            spread = p1 - (hedge_ratio * p2)
            
            spread_mean = spread.rolling(window=20).mean()
            spread_std = spread.rolling(window=20).std()
            z_score = (spread - spread_mean) / spread_std
            current_z = z_score.iloc[-1]
            
            if pd.isna(current_z) or spread_std.iloc[-1] < 1e-8:
                continue
                
            print(f"{name1}/{name2} Z-Score: {current_z:.2f}")
            
            if abs(current_z) > 2.0:
                current_p1 = p1.iloc[-1]
                current_p2 = p2.iloc[-1]
                dec1 = get_decimals(current_p1)
                dec2 = get_decimals(current_p2)
                
                zone1_low = round(current_p1 * 0.999, dec1)
                zone1_high = round(current_p1 * 1.001, dec1)
                zone2_low = round(current_p2 * 0.999, dec2)
                zone2_high = round(current_p2 * 1.001, dec2)

                delta_z_tp1 = abs(current_z - 1.0)
                delta_z_tp2 = abs(current_z - 0.0)
                delta_z_sl = abs(current_z - 3.0)
                
                std_val = spread_std.iloc[-1]

                if current_z > 2.0:
                    dir1, emoji1 = "Short", "📉"
                    dir2, emoji2 = "Long", "📈"
                    
                    tp1_p1 = round(current_p1 + (delta_z_tp1 * std_val), dec1)
                    tp2_p1 = round(current_p1 + (delta_z_tp2 * std_val), dec1)
                    sl_p1 = round(current_p1 - (delta_z_sl * std_val), dec1)
                    
                    tp1_p2 = round(current_p2 - (delta_z_tp1 * std_val * hedge_ratio), dec2)
                    tp2_p2 = round(current_p2 - (delta_z_tp2 * std_val * hedge_ratio), dec2)
                    sl_p2 = round(current_p2 + (delta_z_sl * std_val * hedge_ratio), dec2)
                    
                else:
                    dir1, emoji1 = "Long", "📈"
                    dir2, emoji2 = "Short", "📉"
                    
                    tp1_p1 = round(current_p1 - (delta_z_tp1 * std_val), dec1)
                    tp2_p1 = round(current_p1 - (delta_z_tp2 * std_val), dec1)
                    sl_p1 = round(current_p1 + (delta_z_sl * std_val), dec1)
                    
                    tp1_p2 = round(current_p2 + (delta_z_tp1 * std_val * hedge_ratio), dec2)
                    tp2_p2 = round(current_p2 + (delta_z_tp2 * std_val * hedge_ratio), dec2)
                    sl_p2 = round(current_p2 - (delta_z_sl * std_val * hedge_ratio), dec2)

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