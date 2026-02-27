# KST Trading Bot

Multi-timeframe KST (Know Sure Thing) crossover alerts via Telegram. Monitors weekly, daily, and hourly charts; sends Strong/Medium/Mild signals based on how many timeframes agree.

## Setup

1. **Python 3.9+**
2. `pip install -r requirements.txt`
3. Copy `.env.example` (or create `.env`) with:
   - `BOT_TOKEN` — Telegram bot token (from [@BotFather](https://t.me/BotFather))
   - `CHAT_ID` — Your Telegram chat ID (comma-separated for multiple)
4. Add symbols to `symbols.txt` (one per line, e.g. `TVC:GOLD`, `RELIANCE.NS`).

## Run locally

```bash
python main.py
```

Runs at http://localhost:5000. Health: http://localhost:5000/health

## Production (Render / Heroku / Railway)

- **Build:** `pip install -r requirements.txt`
- **Start:** Uses `Procfile` → `gunicorn -w 1 -b 0.0.0.0:$PORT main:app`
- Set env vars in the dashboard: `BOT_TOKEN`, `CHAT_ID`, and optionally `PORT` (platforms usually set `PORT` for you).

The app uses **one Gunicorn worker** so the background scheduler runs once. Do not increase workers.
