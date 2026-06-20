import os
import ssl
import requests
import time
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime

ssl._create_default_https_context = ssl._create_unverified_context

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")

PAIRS = [
    ("ETH-USD", "LTC-USD", "ETHUSDT", "LTCUSDT"),
    ("BNB-USD", "SOL-USD", "BNBUSDT", "SOLUSDT"),
    ("XRP-USD", "ADA-USD", "XRPUSDT", "ADAUSDT"),
    ("BTC-USD", "ETH-USD", "BTCUSDT", "ETHUSDT")
]

TIMEFRAME = "1D"
DAYS_BACK = 730 # 2 years of daily data

def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHANNEL_ID, "text": text, "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message: {e}")

def run_pairs_backtest():
    print("=== Starting Pairs Trading Backtest (2 Years) ===")
    if not BOT_TOKEN or not CHANNEL_ID:
        print("Missing BOT_TOKEN or CHANNEL_ID.")
        return

    total_signals = 0
    total_wins = 0    # Reached Z = 0
    total_losses = 0  # Reached Z = 4

    for yf_ticker1, yf_ticker2, name1, name2 in PAIRS:
        print(f"Backtesting pair: {name1}/{name2}...")
        try:
            # Fetch 2 years of daily data
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=DAYS_BACK)
            data = yf.download([yf_ticker1, yf_ticker2], start=start_date, end=end_date)['Close']
            data = data.dropna()
            
            if len(data) < 250: # Need at least 1 year of data to calculate 1y rolling stats
                print(f"{name1}/{name2} not enough historical data. Skipping.")
                continue
                
            p1 = data[yf_ticker1]
            p2 = data[yf_ticker2]
            
            in_trade = False
            trade_hedge_ratio = 0
            trade_spread_mean = 0
            trade_spread_std = 0
            trade_direction = 0 # 1 for Z > 1.0, -1 for Z < -1.0

            # We start checking from day 250 (1 year) to have enough history for the 1y lookback
            for i in range(250, len(data)):
                current_p1 = p1.iloc[i]
                current_p2 = p2.iloc[i]

                if not in_trade:
                    # --- SIGNAL GENERATION LOGIC ---
                    hist_p1 = p1.iloc[i-250:i]
                    hist_p2 = p2.iloc[i-250:i]
                    
                    score, p_value, _ = sm.tsa.stattools.coint(hist_p1, hist_p2)
                    
                    if p_value <= 0.15:
                        X = sm.add_constant(hist_p2)
                        model = sm.OLS(hist_p1, X).fit()
                        hedge_ratio = model.params.iloc[1]
                        spread = hist_p1 - (hedge_ratio * hist_p2)
                        
                        # Rolling window is kept at 20 days as requested
                        spread_mean = spread.rolling(window=20).mean().iloc[-1]
                        spread_std = spread.rolling(window=20).std().iloc[-1]
                        current_spread = current_p1 - (hedge_ratio * current_p2)
                        current_z = (current_spread - spread_mean) / spread_std
                        
                        # ENTRY CONDITION CHANGED TO Z = 1.0
                        if not pd.isna(current_z) and spread_std > 1e-8 and abs(current_z) > 1.0:
                            # OPEN TRADE
                            in_trade = True
                            trade_hedge_ratio = hedge_ratio
                            trade_spread_mean = spread_mean
                            trade_spread_std = spread_std
                            trade_direction = 1 if current_z > 1.0 else -1
                            total_signals += 1
                            print(f"  -> Opened trade on {data.index[i].date()} (Z: {current_z:.2f})")
                
                else:
                    # --- TRADE MANAGEMENT LOGIC ---
                    current_spread = current_p1 - (trade_hedge_ratio * current_p2)
                    current_z = (current_spread - trade_spread_mean) / trade_spread_std
                    
                    # Check Win Condition (Mean Reversion / Z = 0)
                    if (trade_direction == 1 and current_z <= 0.0) or (trade_direction == -1 and current_z >= 0.0):
                        total_wins += 1
                        in_trade = False
                        print(f"  -> WIN on {data.index[i].date()} (Z reverted to 0)")
                    
                    # STOP LOSS CHANGED TO Z = 4.0
                    elif (trade_direction == 1 and current_z >= 4.0) or (trade_direction == -1 and current_z <= -4.0):
                        total_losses += 1
                        in_trade = False
                        print(f"  -> LOSS on {data.index[i].date()} (Z hit 4)")

        except Exception as e:
            print(f"Error backtesting {yf_ticker1}/{yf_ticker2}: {e}")

    # --- GENERATE REPORT ---
    win_rate = (total_wins / total_signals * 100) if total_signals > 0 else 0
    loss_rate = (total_losses / total_signals * 100) if total_signals > 0 else 0

    report = f"""📊 <b>Statistical Arbitrage Backtest Report</b> ⏱️ {DAYS_BACK} Days
                         
🔍 <b>Tested Pairs:</b> {len(PAIRS)} Crypto Pairs
📈 <b>Timeframe:</b> {TIMEFRAME}
⚙️ <b>Logic:</b> Z-Score Entry (>1.0) | TP (Z=0) | SL (Z=4)

━━━━━━━━━━━━━━━━━━━━
📊 <b>Total Trades Triggered:</b> {total_signals}

🟢 <b>Winners (Reverted to Mean):</b> {total_wins}
🔴 <b>Losers (Diverged to SL):</b> {total_losses}

━━━━━━━━━━━━━━━━━━━━
🏆 <b>Win Rate:</b> <code>{win_rate:.1f}%</code>
💀 <b>Loss Rate:</b> <code>{loss_rate:.1f}%</code>
━━━━━━━━━━━━━━━━━━━━

⚠️ <i>Note: This calculates pure statistical success. Slippage, funding rates, and execution fees are not included.</i>"""

    print("Sending Pairs Backtest Report to Telegram...")
    send_message(report)
    print("Done!")

if __name__ == "__main__":
    run_pairs_backtest()