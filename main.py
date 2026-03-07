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
# Suppress yfinance "No data found, symbol may be delisted" spam (invalid/unavailable symbols)
logging.getLogger("yfinance").setLevel(logging.WARNING)

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
# Hourly = trigger (crossover); Daily/Weekly = trend only (KST vs Signal on last bar).
# Strength: Strong = D+W align with trigger; Medium = one aligns; Mild = both oppose.

# ==============================
# Data Cache (key: (symbol, interval))
# ==============================
data_cache = {}
# Log "no data" only once per (symbol, interval) per process to avoid log spam
_data_warned = set()


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
    global data_cache, _data_warned
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
        if out is None or out.empty:
            if key not in _data_warned:
                _data_warned.add(key)
                log.warning(
                    "No price data for %s (interval=%s). Check symbol on Yahoo Finance or remove from symbols.txt",
                    symbol, interval,
                )
            return pd.DataFrame()
        return out
    except Exception as e:
        err_msg = str(e).lower()
        if "no data" in err_msg or "delisted" in err_msg:
            if key not in _data_warned:
                _data_warned.add(key)
                log.warning("No data for %s (%s): %s", symbol, interval, e)
            return pd.DataFrame()
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
# KST Logic: Hourly = trigger (crossover); Daily/Weekly = trend only
# ==============================
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


def _trend(kst, signal):
    """Trend on latest bar: KST > Signal → bullish, KST < Signal → bearish. For Daily/Weekly only."""
    if kst is None or signal is None or len(kst) < 1:
        return None
    k, s = kst.iloc[-1], signal.iloc[-1]
    if pd.isna(k) or pd.isna(s):
        return None
    if k > s:
        return "bullish"
    if k < s:
        return "bearish"
    return None


def _signal_strength(trigger_direction, daily_trend, weekly_trend):
    """
    Classify strength from hourly trigger + daily/weekly trends.
    Returns e.g. 'Strong Bullish', 'Medium Bearish', 'Mild Bullish'.
    """
    if trigger_direction not in ("bullish", "bearish"):
        return None
    d_bull = daily_trend == "bullish"
    d_bear = daily_trend == "bearish"
    w_bull = weekly_trend == "bullish"
    w_bear = weekly_trend == "bearish"
    if trigger_direction == "bullish":
        if d_bull and w_bull:
            return "Strong Bullish"
        if (d_bull and w_bear) or (d_bear and w_bull):
            return "Medium Bullish"
        if d_bear and w_bear:
            return "Mild Bullish"
    else:  # bearish
        if d_bear and w_bear:
            return "Strong Bearish"
        if (d_bear and w_bull) or (d_bull and w_bear):
            return "Medium Bearish"
        if d_bull and w_bull:
            return "Mild Bearish"
    return "Mild " + ("Bullish" if trigger_direction == "bullish" else "Bearish")


def _strength_emoji(strength_label):
    """Visual indicator for signal strength (Strong/Medium/Mild)."""
    if "Strong" in (strength_label or ""):
        return "🔥"
    if "Medium" in (strength_label or ""):
        return "◀️"
    return "▪️"


def check_crossovers():
    """
    Run 24/7. Alert only when Hourly KST crossover occurs.
    Daily and Weekly used only for trend (KST vs Signal); strength from alignment.
    One alert per new hourly candle.
    """
    global _last_alerted_bar
    now = datetime.now(pytz.UTC).astimezone(IST)
    for symbol in symbols:
        try:
            # Fetch all three timeframes
            data = {}
            for tf_name, tf_config in TIMEFRAMES.items():
                df = get_data(symbol, tf_config["interval"], tf_config["period"])
                if df is None or df.empty or len(df) < 2:
                    data[tf_name] = None
                    continue
                kst_signal = calculate_kst(df)
                if not kst_signal:
                    data[tf_name] = None
                    continue
                data[tf_name] = {"df": df, "kst": kst_signal[0], "signal": kst_signal[1], "config": tf_config}

            hourly_data = data.get("hourly")
            if not hourly_data:
                continue

            # Trigger only on hourly crossover
            h_dir = _crossover_direction(hourly_data["kst"], hourly_data["signal"])
            if not h_dir:
                continue

            # Dedupe: one alert per new hourly candle
            h_df = hourly_data["df"]
            current_bar = h_df.index[-1]
            bar_id = getattr(current_bar, "value", current_bar)
            if isinstance(bar_id, pd.Timestamp):
                bar_id = bar_id.value
            key = (symbol, "hourly")
            if _last_alerted_bar.get(key) == bar_id:
                continue
            _last_alerted_bar[key] = bar_id

            # Daily and Weekly: trend only (no crossover)
            daily_data = data.get("daily")
            weekly_data = data.get("weekly")
            daily_trend = _trend(daily_data["kst"], daily_data["signal"]) if daily_data else None
            weekly_trend = _trend(weekly_data["kst"], weekly_data["signal"]) if weekly_data else None

            # Classify strength from trigger + trends
            strength_label = _signal_strength(h_dir, daily_trend, weekly_trend)
            if not strength_label:
                strength_label = "Mild " + ("Bullish" if h_dir == "bullish" else "Bearish")

            # Latest price from hourly
            latest_close = None
            try:
                latest_close = float(h_df["Close"].iloc[-1])
            except (TypeError, ValueError, IndexError):
                pass

            # Format trend for display (capitalize, or "—" if missing)
            def _trend_str(t):
                if t == "bullish":
                    return "Bullish"
                if t == "bearish":
                    return "Bearish"
                return "—"

            # Currency: ₹ for NSE/BSE (.NS, .BO), $ for others
            currency = "₹" if symbol.endswith(".NS") or symbol.endswith(".BO") else "$"
            price_str = f"Price: {currency}{latest_close:,.2f}" if latest_close is not None else "Price: —"

            trigger_text = "Hourly Bullish Crossover" if h_dir == "bullish" else "Hourly Bearish Crossover"
            emoji = _strength_emoji(strength_label)
            time_str = now.strftime("%d %b %Y · %H:%M IST")

            msg_lines = [
                "🔔 KST SIGNAL",
                "",
                f"Symbol: {symbol}",
                price_str,
                "",
                f"Signal: {emoji} {strength_label}",
                "",
                f"Trigger: {trigger_text}",
                f"Daily Trend: {_trend_str(daily_trend)}",
                f"Weekly Trend: {_trend_str(weekly_trend)}",
                "",
                f"Time: {time_str}",
            ]
            msg = "\n".join(msg_lines)
            send_telegram(msg)

            # For hourly summary
            if symbol not in hourly_crossovers:
                hourly_crossovers[symbol] = []
            hourly_crossovers[symbol].append(strength_label)
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
