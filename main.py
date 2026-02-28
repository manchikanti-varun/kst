import os
import sys
import time
import logging
from collections import Counter
import pandas as pd
import yfinance as yf
import requests
import schedule
from flask import Flask, jsonify
from datetime import datetime, timedelta
import pytz
import threading
from dotenv import load_dotenv

load_dotenv()

# ==============================
# Config
# ==============================
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN_TIME = (9, 15)   # 09:15 IST
MARKET_CLOSE_TIME = (15, 30) # 15:30 IST
CHECK_INTERVAL_MINUTES = 45
FETCH_DELAY_SECONDS = 0.5    # delay between symbol fetches to avoid rate limits
PORT = int(os.getenv("PORT", "5000"))

# Telegram (from .env or Render ENV)
# CHAT_ID: single ID or comma-separated (e.g. 123456,789012) for multiple recipients
BOT_TOKEN = os.getenv("BOT_TOKEN")
_chat_id_raw = os.getenv("CHAT_ID", "")
CHAT_IDS = [x.strip() for x in _chat_id_raw.split(",") if x.strip()]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,  # stdout so Railway shows as info, not error
)
log = logging.getLogger(__name__)

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

@app.route("/")
def home():
    now_ist = datetime.now(pytz.UTC).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    return jsonify({
        "status": "running",
        "time": now_ist,
        "symbols_monitored": len(symbols),
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.now(pytz.UTC).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
    })

# ==============================
# Load Symbols from File
# ==============================
def load_symbols(file_path="symbols.txt"):
    if not os.path.exists(file_path):
        log.warning("symbols.txt not found at %s", file_path)
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in raw:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


symbols = load_symbols()
log.info("Loaded %d symbols from symbols.txt", len(symbols))

# ==============================
# Multi-timeframe: strength from how many timeframes agree (3=Strong, 2=Medium, 1=Mild)
# Hourly = 1h bars; any crossover on 1h chart triggers an alert (once per new 1h candle).
# ==============================
TIMEFRAMES = {
    "weekly": {"interval": "1wk", "period": "3y", "label": "Weekly"},
    "daily": {"interval": "1d", "period": "1y", "label": "Daily"},
    "hourly": {"interval": "1h", "period": "90d", "label": "Hourly"},  # 1-hour bars
}
# Strength by count of agreeing timeframes
STRENGTH_BY_COUNT = {3: "Strong", 2: "Medium", 1: "Mild"}

# ==============================
# Data Cache (key: (symbol, interval))
# ==============================
data_cache = {}


def _normalize_df(df):
    """Flatten MultiIndex columns from yfinance so we always have 'Close'."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return pd.DataFrame()
    return df


def get_data(symbol, interval, period):
    """Fetch stock data with caching for the given interval."""
    global data_cache
    key = (symbol, interval)
    try:
        if key not in data_cache:
            df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
            data_cache[key] = _normalize_df(df)
        else:
            # Refresh with latest (small period for incremental)
            refresh_period = "5d" if interval == "1h" else "1mo" if interval == "1d" else "3mo"
            latest = yf.download(symbol, period=refresh_period, interval=interval, progress=False, threads=False)
            if not latest.empty:
                latest = _normalize_df(latest)
                if not latest.empty:
                    combined = pd.concat([data_cache[key], latest])
                    data_cache[key] = combined[~combined.index.duplicated(keep="last")].iloc[-500:]
        out = data_cache.get(key)
        return out if out is not None and not out.empty else pd.DataFrame()
    except Exception as e:
        log.exception("Error fetching %s %s: %s", symbol, interval, e)
        return pd.DataFrame()

# ==============================
# KST Calculation
# ==============================
def calculate_kst(df):
    """KST (Know Sure Thing) with 10/15/20/30 ROC and 9-period signal."""
    if df is None or df.empty or "Close" not in df.columns or len(df) < 130:
        return None
    close = df["Close"].astype(float)
    roc1 = close.pct_change(10) * 100
    roc2 = close.pct_change(15) * 100
    roc3 = close.pct_change(20) * 100
    roc4 = close.pct_change(30) * 100
    kst = (
        roc1.rolling(10).mean() * 1
        + roc2.rolling(10).mean() * 2
        + roc3.rolling(10).mean() * 3
        + roc4.rolling(15).mean() * 4
    )
    signal = kst.rolling(9).mean()
    return kst, signal

# ==============================
# Telegram Alert
# ==============================
def send_telegram(message):
    if not BOT_TOKEN or not CHAT_IDS:
        log.warning("BOT_TOKEN or CHAT_ID(s) not set; skipping Telegram send")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        try:
            response = requests.post(
                url, json={"chat_id": chat_id, "text": message}, timeout=10
            )
            if not response.ok:
                log.warning(
                    "Telegram API for chat %s: %s %s",
                    chat_id,
                    response.status_code,
                    response.text,
                )
        except Exception as e:
            log.exception("Telegram send to %s failed: %s", chat_id, e)

# ==============================
# Track crossovers per symbol (for hourly summary)
# ==============================
hourly_crossovers = {s: [] for s in symbols}

# Only alert once per new candle: (symbol, tf_name) -> last bar timestamp we alerted on
_last_alerted_bar = {}

# ==============================
# Main Crossover Check (weekly / daily / hourly → Strong / Medium / Mild)
# ==============================
def _market_hours(now):
    open_ = now.replace(hour=MARKET_OPEN_TIME[0], minute=MARKET_OPEN_TIME[1], second=0, microsecond=0)
    close = now.replace(hour=MARKET_CLOSE_TIME[0], minute=MARKET_CLOSE_TIME[1], second=0, microsecond=0)
    return open_, close


def _crossover_direction(kst, signal):
    """Returns 'bullish', 'bearish', or None if no crossover on last bar.
    KST above signal = bullish, KST below signal = bearish. Ignores NaN."""
    if kst is None or signal is None or len(kst) < 2:
        return None
    p_k, p_s = kst.iloc[-2], signal.iloc[-2]
    c_k, c_s = kst.iloc[-1], signal.iloc[-1]
    if pd.isna(p_k) or pd.isna(p_s) or pd.isna(c_k) or pd.isna(c_s):
        return None
    prev_below = p_k < p_s
    curr_above = c_k > c_s
    prev_above = p_k > p_s
    curr_below = c_k < c_s
    if prev_below and curr_above:
        return "bullish"
    if prev_above and curr_below:
        return "bearish"
    return None


def _strength_from_count(n):
    """3 timeframes → Strong, 2 → Medium, 1 → Mild."""
    return STRENGTH_BY_COUNT.get(n, "Mild")


def _strength_emoji(strength):
    """Visual indicator for signal strength."""
    return {"Strong": "🔥", "Medium": "◀️", "Mild": "▪️"}.get(strength, "▪️")


def check_crossovers():
    """Run 24/7. Any crossover → one alert per new candle (no 9:15–15:30 restriction, no spam)."""
    global _last_alerted_bar
    now = datetime.now(pytz.UTC).astimezone(IST)
    for symbol in symbols:
        try:
            bullish_tfs = []
            bearish_tfs = []
            latest_close = None
            for tf_name, tf_config in TIMEFRAMES.items():
                df = get_data(symbol, tf_config["interval"], tf_config["period"])
                if df is None or df.empty or len(df) < 2:
                    continue
                kst_signal = calculate_kst(df)
                if not kst_signal:
                    continue
                kst, signal = kst_signal
                direction = _crossover_direction(kst, signal)
                if not direction:
                    continue
                current_bar = df.index[-1]
                bar_id = getattr(current_bar, "value", current_bar)
                if isinstance(bar_id, pd.Timestamp):
                    bar_id = bar_id.value
                key = (symbol, tf_name)
                if _last_alerted_bar.get(key) == bar_id:
                    continue
                _last_alerted_bar[key] = bar_id
                try:
                    latest_close = float(df["Close"].iloc[-1])
                except (TypeError, ValueError, IndexError):
                    pass
                label = tf_config["label"]
                try:
                    bar_ts = pd.Timestamp(current_bar)
                    bar_str = bar_ts.strftime("%d %b %Y %H:%M")
                except Exception:
                    bar_str = str(current_bar)
                if direction == "bullish":
                    bullish_tfs.append((label, bar_str))
                elif direction == "bearish":
                    bearish_tfs.append((label, bar_str))
            parts = []
            labels_for_summary = []
            if bullish_tfs:
                strength = _strength_from_count(len(bullish_tfs))
                emoji = _strength_emoji(strength)
                tf_str = ", ".join(lbl for lbl, _ in bullish_tfs)
                bar_times = " | ".join(f"{lbl}: {bt}" for lbl, bt in bullish_tfs)
                parts.append(f"{emoji} 📈 {strength} Bullish · {tf_str}")
                parts.append(f"   Candle: {bar_times}")
                labels_for_summary.append(f"{strength} Bullish ({tf_str})")
            if bearish_tfs:
                strength = _strength_from_count(len(bearish_tfs))
                emoji = _strength_emoji(strength)
                tf_str = ", ".join(lbl for lbl, _ in bearish_tfs)
                bar_times = " | ".join(f"{lbl}: {bt}" for lbl, bt in bearish_tfs)
                parts.append(f"{emoji} 📉 {strength} Bearish · {tf_str}")
                parts.append(f"   Candle: {bar_times}")
                labels_for_summary.append(f"{strength} Bearish ({tf_str})")
            if parts:
                msg_lines = [
                    f"🔔 KST Crossover · {symbol}",
                    f"Checked: {now.strftime('%d %b %Y · %H:%M IST')}",
                ]
                if latest_close is not None:
                    msg_lines.append(f"Last: {latest_close:,.2f}")
                msg_lines.append("")
                msg_lines.extend(parts)
                msg = "\n".join(msg_lines)
                send_telegram(msg)
                for lbl in labels_for_summary:
                    if symbol not in hourly_crossovers:
                        hourly_crossovers[symbol] = []
                    hourly_crossovers[symbol].append(lbl)
        except Exception as e:
            log.exception("Error processing %s: %s", symbol, e)
        if FETCH_DELAY_SECONDS > 0:
            time.sleep(FETCH_DELAY_SECONDS)

# ==============================
# Hourly Summary (every hour, NSE/BSE/global; only send if different from last)
# ==============================
_last_hourly_summary = None


def send_hourly_summary():
    global _last_hourly_summary
    now = datetime.now(pytz.UTC).astimezone(IST)
    lines = []
    for symbol in symbols:
        events = hourly_crossovers.get(symbol, [])
        if not events:
            continue
        counts = Counter(events)
        parts = [f"{cnt} {label}" for label, cnt in counts.most_common()]
        lines.append(f"• {symbol}: " + ", ".join(parts))
        hourly_crossovers[symbol] = []
    if not lines:
        return
    message = f"⏰ Hourly update {now.strftime('%H:%M IST')}\n" + "\n".join(lines)
    if message == _last_hourly_summary:
        return
    _last_hourly_summary = message
    send_telegram(message)

# ==============================
# Scheduler Setup
# ==============================
def run_scheduler():
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(check_crossovers)
    schedule.every().hour.at(":00").do(send_hourly_summary)
    while True:
        schedule.run_pending()
        time.sleep(5)


def _start_background_scheduler():
    """Start scheduler thread (runs on import so gunicorn + scheduler work)."""
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    log.info("Scheduler started at %s", datetime.now(pytz.UTC).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST"))
    log.info("Monitoring %d symbols (every %s min)", len(symbols), CHECK_INTERVAL_MINUTES)
    send_telegram("🚀 Trading Bot Started and Running 24/7!")


# Start scheduler when module loads (works with both gunicorn and python main.py)
_start_background_scheduler()

# ==============================
# Main Entry
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
