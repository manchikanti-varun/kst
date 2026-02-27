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
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    return jsonify({
        "status": "running",
        "time": now_ist,
        "symbols_monitored": len(symbols),
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
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


def _is_market_open(now=None):
    """True if we should run crossover check.
    Mon–Fri: only during NSE hours (09:15–15:30 IST).
    Sat–Sun: always run (commodities like Gold/Silver trade on global markets)."""
    now = now or datetime.now(IST)
    market_open, market_close = _market_hours(now)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return True   # Weekend: run for global commodities (TVC:GOLD, etc.)
    return market_open <= now <= market_close


def _crossover_direction(kst, signal):
    """Returns 'bullish', 'bearish', or None if no crossover on last bar."""
    if kst is None or signal is None or len(kst) < 2:
        return None
    prev_below = kst.iloc[-2] < signal.iloc[-2]
    curr_above = kst.iloc[-1] > signal.iloc[-1]
    prev_above = kst.iloc[-2] > signal.iloc[-2]
    curr_below = kst.iloc[-1] < signal.iloc[-1]
    if prev_below and curr_above:
        return "bullish"
    if prev_above and curr_below:
        return "bearish"
    return None


def _strength_from_count(n):
    """3 timeframes → Strong, 2 → Medium, 1 → Mild."""
    return STRENGTH_BY_COUNT.get(n, "Mild")


def check_crossovers():
    global _last_alerted_bar
    now = datetime.now(IST)
    if not _is_market_open(now):
        log.info("Market closed (outside 09:15–15:30 IST Mon–Fri), skipping crossover check")
        return
    for symbol in symbols:
        try:
            bullish_tfs = []
            bearish_tfs = []
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
                # Only alert on a NEW candle: avoid same crossover every min when market closed
                current_bar = df.index[-1]
                key = (symbol, tf_name)
                if _last_alerted_bar.get(key) == current_bar:
                    continue
                _last_alerted_bar[key] = current_bar
                label = tf_config["label"]
                if direction == "bullish":
                    bullish_tfs.append(label)
                elif direction == "bearish":
                    bearish_tfs.append(label)
            parts = []
            labels_for_summary = []
            if bullish_tfs:
                strength = _strength_from_count(len(bullish_tfs))
                tf_str = ", ".join(bullish_tfs)
                parts.append(f"📈 {strength} Bullish ({tf_str})")
                labels_for_summary.append(f"{strength} Bullish ({tf_str})")
            if bearish_tfs:
                strength = _strength_from_count(len(bearish_tfs))
                tf_str = ", ".join(bearish_tfs)
                parts.append(f"📉 {strength} Bearish ({tf_str})")
                labels_for_summary.append(f"{strength} Bearish ({tf_str})")
            if parts:
                msg = f"KST · {symbol} · {now.strftime('%H:%M IST')}\n" + "\n".join(parts)
                send_telegram(msg)
                for lbl in labels_for_summary:
                    hourly_crossovers[symbol].append(lbl)
        except Exception as e:
            log.exception("Error processing %s: %s", symbol, e)
        if FETCH_DELAY_SECONDS > 0:
            time.sleep(FETCH_DELAY_SECONDS)

# ==============================
# Hourly Summary
# ==============================
def send_hourly_summary():
    now = datetime.now(IST)
    market_open, market_close = _market_hours(now)
    if market_open <= now <= market_close:
        lines = []
        for symbol in symbols:
            events = hourly_crossovers[symbol]
            if not events:
                continue
            # Group by label e.g. "Strong Bullish", "Mild Bearish"
            counts = Counter(events)
            parts = [f"{cnt} {label}" for label, cnt in counts.most_common()]
            lines.append(f"• {symbol}: " + ", ".join(parts))
            hourly_crossovers[symbol] = []
        if lines:
            message = f"⏰ Hourly update {now.strftime('%H:%M IST')}\n" + "\n".join(lines)
            send_telegram(message)

# ==============================
# Market Status Reminder
# ==============================
def _next_market_open(now):
    """Next NSE open (09:15 IST) on a weekday. Skips Sat/Sun."""
    market_open, market_close = _market_hours(now)
    # Before 09:15 today and today is Mon–Fri → open today
    if now.weekday() < 5 and now < market_open:
        return market_open
    # After 15:30 or weekend → next trading day (Mon–Fri only)
    # Python: Mon=0, Fri=4, Sat=5, Sun=6
    days_ahead = 1
    while True:
        next_date = now.date() + timedelta(days=days_ahead)
        if next_date.weekday() <= 4:  # Monday=0 .. Friday=4
            break
        days_ahead += 1
    return now.replace(
        year=next_date.year, month=next_date.month, day=next_date.day,
        hour=MARKET_OPEN_TIME[0], minute=MARKET_OPEN_TIME[1], second=0, microsecond=0,
    )


def market_status_reminder():
    now = datetime.now(IST)
    market_open, market_close = _market_hours(now)
    if now.weekday() >= 5:
        next_open = _next_market_open(now)
        delta = next_open - now
        hours_left = round(delta.total_seconds() / 3600, 1)
        next_str = next_open.strftime("%A %d %b at %I:%M %p IST")
        send_telegram(f"⏰ NSE closed (weekend). Opens in {hours_left} hrs → {next_str}")
    elif now < market_open:
        next_open = _next_market_open(now)
        delta = next_open - now
        hours_left = round(delta.total_seconds() / 3600, 1)
        next_str = next_open.strftime("%A %d %b at %I:%M %p IST")
        send_telegram(f"⏰ Market closed. Opens in {hours_left} hrs → {next_str}")
    elif now > market_close:
        next_open = _next_market_open(now)
        delta = next_open - now
        hours_left = round(delta.total_seconds() / 3600, 1)
        next_str = next_open.strftime("%A %d %b at %I:%M %p IST")
        # Friday evening → next open is Monday (not Saturday)
        send_telegram(f"⏰ Market closed. Next trading day in {hours_left} hrs → {next_str}")

# ==============================
# Scheduler Setup
# ==============================
def run_scheduler():
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(check_crossovers)
    schedule.every().hour.at(":00").do(send_hourly_summary)
    schedule.every().hour.at(":00").do(market_status_reminder)
    while True:
        schedule.run_pending()
        time.sleep(5)


def _start_background_scheduler():
    """Start scheduler thread (runs on import so gunicorn + scheduler work)."""
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    log.info("Scheduler started at %s", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"))
    log.info("Monitoring %d symbols (every %s min)", len(symbols), CHECK_INTERVAL_MINUTES)
    send_telegram("🚀 Trading Bot Started and Running 24/7!")


# Start scheduler when module loads (works with both gunicorn and python main.py)
_start_background_scheduler()

# ==============================
# Main Entry
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
