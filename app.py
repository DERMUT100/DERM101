import os, json, requests, time, re, random, traceback, uuid, html
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
try:
    import yfinance as yf
except Exception:
    yf = None

# ── Page Config & UI Styling ───────────────────────────────────────────────
st.set_page_config(page_title="Der-AI | News Analysis", page_icon="📰", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stButton>button { background: linear-gradient(90deg, #059669 0%, #10b981 100%); color: white; border: none; padding: 10px 24px; border-radius: 8px; font-weight: bold; font-size: 16px; width: 100%; }
    .stButton>button:hover { background: linear-gradient(90deg, #047857 0%, #059669 100%); }
    .news-card { background: #f0fdf4; color: #123026; padding: 20px; border-radius: 12px; border-left: 6px solid #10b981; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .news-card h4, .news-card p, .news-card b { color: #123026 !important; }
    .signal-card { background: #f8fafc; padding: 20px; border-radius: 12px; border-left: 6px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .buy-signal { border-left-color: #10b981; }
    .sell-signal { border-left-color: #ef4444; }
    .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .debug-box { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# ── Core Helpers & Session State ───────────────────────────────────────────
def get_secret(name, default=""):
    try:
        value = st.secrets.get(name, None)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)

TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID", "")
SYMBOLS = ['XAUUSD', 'EURUSD', 'BTCUSD', 'US30']
NEWS_ANALYSIS_SYMBOLS = ['XAUUSD']
YFINANCE_MAP = {'XAUUSD': 'GC=F', 'EURUSD': 'EURUSD=X', 'BTCUSD': 'BTC-USD', 'US30': '^DJI', 'DXY': 'DX-Y.NYB'}
MINIMUM_CONFLUENCE_SCORE = 72

GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MIN_REQUEST_INTERVAL = 3
GROQ_TOKEN_LIMIT_PER_MINUTE = 1000000
GROQ_MAX_OUTPUT_TOKENS = 4000
GROQ_ESTIMATED_RESPONSE_TOKENS = GROQ_MAX_OUTPUT_TOKENS
GROQ_MODELS = [
    'openai/gpt-oss-120b',  # Primary: Groq's recommended high-reasoning replacement
    'openai/gpt-oss-20b',   # Secondary: Lighter/faster GPT OSS fallback
    'qwen/qwen3-32b'        # Tertiary: Qwen fallback (matches your MARKETANALYSIS.PY slugs)
]
PYTHON_FALLBACK_MODEL = 'Python fallback (rule-based MTF confluence)'
NEWS_PRE_WINDOW_HOURS = 2

if 'signal_history' not in st.session_state: st.session_state.signal_history = []
if 'notifications' not in st.session_state: st.session_state.notifications = []
if 'cached_market_data' not in st.session_state: st.session_state.cached_market_data = {}
if 'last_market_fetch_time' not in st.session_state: st.session_state.last_market_fetch_time = None
if 'active_signals' not in st.session_state: st.session_state.active_signals = {}
if 'fetched_news' not in st.session_state: st.session_state.fetched_news = []
if 'news_results' not in st.session_state: st.session_state.news_results = {}
if 'news_event_statuses' not in st.session_state: st.session_state.news_event_statuses = {}
if 'news_signal_sent' not in st.session_state: st.session_state.news_signal_sent = {}
if 'directional_bias' not in st.session_state: st.session_state.directional_bias = {}
if 'signal_ledger' not in st.session_state: st.session_state.signal_ledger = []
if 'learning_stats' not in st.session_state: st.session_state.learning_stats = {}
if 'market_state' not in st.session_state: st.session_state.market_state = 'coiling'
if 'state_history' not in st.session_state: st.session_state.state_history = []

if 'groq_tokens_used' not in st.session_state: st.session_state.groq_tokens_used = 0
if 'groq_token_window_start' not in st.session_state: st.session_state.groq_token_window_start = datetime.now()
if 'last_groq_request_time' not in st.session_state: st.session_state.last_groq_request_time = None
if 'groq_rate_limit_until' not in st.session_state: st.session_state.groq_rate_limit_until = None
if 'groq_rate_limit_reason' not in st.session_state: st.session_state.groq_rate_limit_reason = ''

def add_notification(note_type, message, symbol=None, signal=None, score=None):
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    signal = normalize_ai_signal(signal) if signal is not None else signal
    notification = {
        'id': str(uuid.uuid4()),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': note_type,
        'message': message,
        'symbol': symbol,
        'signal': signal,
        'score': score,
        'read': False
    }
    st.session_state.notifications.append(notification)
    if len(st.session_state.notifications) > 200:
        st.session_state.notifications = st.session_state.notifications[-200:]
    return notification

def get_notifications():
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    return st.session_state.notifications

def clear_notifications():
    st.session_state.notifications = []

def _escape_telegram_html(text):
    if text is None:
        return ''
    return html.escape(str(text), quote=False)

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        try:
            data = resp.json()
        except Exception:
            data = None
        if resp.status_code == 200 and (data is None or data.get('ok', True)):
            return True
        else:
            print(f"Telegram send failed: status={resp.status_code} body={resp.text}")
            return False
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def build_news_event_telegram(event, results, now):
    et = event.get('event_time_utc')
    lead = format_lead_time(et - now) if et else 'N/A'
    event_time_str = event.get('time', 'N/A')
    lines = [
        "📰 <b>DER-AI NEWS IMPACT SIGNAL</b>",
        f"📌 Event: {_escape_telegram_html(event.get('event'))}",
        f"🕒 Event time: {_escape_telegram_html(event_time_str)}",
        f"⏳ Sent {_escape_telegram_html(lead)} before the release (once per event)",
        "",
    ]
    for symbol, a in results.items():
        sig = a.get('signal', 'SKIPPED')
        symbol_escaped = _escape_telegram_html(symbol)
        if sig in ('BUY', 'SELL'):
            sig_emoji = "🟢" if sig == 'BUY' else "🔴"
            model_used = _escape_telegram_html(a.get('model_used', PYTHON_FALLBACK_MODEL))
            lines.append(f"{sig_emoji} <b>{symbol_escaped}</b>: {_escape_telegram_html(sig)} | Model: {model_used}")
            hist_pattern = a.get('historical_pattern', '')
            reason = (a.get('reasoning') or '').strip()
            reason_escaped = _escape_telegram_html(reason)
            if hist_pattern and hist_pattern not in reason:
                lines.append(f"📚 Historical: {_escape_telegram_html(hist_pattern)}")
            if reason_escaped:
                lines.append(f"🧠 {reason_escaped}")
            lines.append("")
        elif sig == 'WAIT':
            model_used = _escape_telegram_html(a.get('model_used', PYTHON_FALLBACK_MODEL))
            lines.append(f"⚪ <b>{symbol_escaped}</b>: WAIT | Model: {model_used}")
            why = (a.get('rejection_reason') or a.get('reasoning') or 'No actionable edge.').strip()
            if why:
                lines.append(f"🧠 {_escape_telegram_html(why)}")
            lines.append("")
        else:
            lines.append(f"⚪ <b>{symbol_escaped}</b>: skipped ({_escape_telegram_html(a.get('reason', 'rate limit'))})")
            lines.append("")
    return "\n".join(lines)

def format_lead_time(delta):
    total = int(delta.total_seconds() // 60)
    return f"{total // 60}h {total % 60}m"

# ── News Parsing Helpers ─────────────────────────────────────────────────
def _coerce_impact(impact: Any) -> str:
    if impact is None:
        return "UNKNOWN"
    if isinstance(impact, (int, float)):
        return "HIGH" if int(impact) >= 3 else "MEDIUM" if int(impact) >= 2 else "LOW"
    text = str(impact).strip().lower()
    if text in {"3", "high", "high impact", "red", "important"}:
        return "HIGH"
    if text in {"2", "medium", "orange", "moderate"}:
        return "MEDIUM"
    return text.upper() if text else "UNKNOWN"

def normalize_event_time(date_str, time_str, timezone_name, reference_dt=None):
    if not date_str:
        return None
    try:
        if isinstance(date_str, datetime):
            event_dt = date_str
        else:
            text = str(date_str).strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            event_dt = datetime.fromisoformat(text)
            if event_dt.tzinfo is None:
                if timezone_name:
                    try:
                        event_dt = event_dt.replace(tzinfo=ZoneInfo(str(timezone_name)))
                    except Exception:
                        event_dt = event_dt.replace(tzinfo=timezone.utc)
                else:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
        return event_dt.astimezone(timezone.utc)
    except Exception:
        pass
    if not time_str:
        return None
    try:
        event_date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
    except ValueError:
        try:
            event_date = datetime.strptime(str(date_str), "%Y-%m-%d %H:%M:%S").date()
        except ValueError:
            return None
    raw_time = str(time_str).strip()
    try:
        hour, minute = map(int, raw_time.split(":")[:2])
    except ValueError:
        return None
    tzinfo = None
    if timezone_name:
        try:
            tzinfo = ZoneInfo(str(timezone_name))
        except Exception:
            tzinfo = None
    event_dt = datetime(event_date.year, event_date.month, event_date.day, hour, minute, tzinfo=tzinfo)
    if tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)
    if reference_dt is not None and reference_dt.tzinfo is None:
        reference_dt = reference_dt.replace(tzinfo=timezone.utc)
    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)
    return event_dt.astimezone(timezone.utc)

def parse_news_payload(payload, reference_dt=None, lookahead_hours=72, only_high=True):
    if reference_dt is None:
        reference_dt = datetime.now(timezone.utc)
    if reference_dt.tzinfo is None:
        reference_dt = reference_dt.replace(tzinfo=timezone.utc)
    events = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        impact = _coerce_impact(item.get("impact"))
        if only_high and impact != "HIGH":
            continue
        if not only_high and impact not in {"HIGH", "MEDIUM"}:
            continue
        event_dt = normalize_event_time(item.get("date"), item.get("time"), item.get("timezone") or item.get("tz") or item.get("timeZone"), reference_dt=reference_dt)
        if event_dt is None:
            continue
        if event_dt < reference_dt - timedelta(hours=6):
            continue
        if event_dt > reference_dt + timedelta(hours=lookahead_hours):
            continue
        minutes_until = int((event_dt - reference_dt).total_seconds() // 60)
        events.append({
            "event": str(item.get("event") or item.get("title") or "Economic Event").strip(),
            "currency": str(item.get("country") or item.get("currency") or item.get("pair") or "USD").strip(),
            "impact": "HIGH" if impact == "HIGH" else "MEDIUM",
            "time": event_dt.strftime("%Y-%m-%d %H:%M UTC"),
            "event_time_utc": event_dt,
            "minutes_until": minutes_until,
            "within_2h": minutes_until <= 120 and minutes_until >= 0,
            "timezone": item.get("timezone") or item.get("tz") or item.get("timeZone") or "UTC"
        })
    events.sort(key=lambda e: e["event_time_utc"])
    return events

def build_news_context(events, reference_dt=None):
    if reference_dt is None:
        reference_dt = datetime.now(timezone.utc)
    if reference_dt.tzinfo is None:
        reference_dt = reference_dt.replace(tzinfo=timezone.utc)
    upcoming = [e for e in events if e.get("event_time_utc") and e["event_time_utc"] >= reference_dt]
    within_2h = [e for e in upcoming if e.get("within_2h")]
    next_event = upcoming[0] if upcoming else None
    if next_event:
        minutes_until = int((next_event["event_time_utc"] - reference_dt).total_seconds() // 60)
        if minutes_until <= 120:
            bias = "opposite"
            pre_news_bias = f"High-impact event arriving in {minutes_until} minutes; expect the market to express a short-term reactive move before stabilizing."
        else:
            bias = "neutral"
            pre_news_bias = f"Upcoming high-impact event in {minutes_until} minutes; monitor for volatility expansion and a likely liquidity sweep."
    else:
        bias = "neutral"
        pre_news_bias = "No imminent high-impact event in the next 2 hours."
    return {
        "within_2h": bool(within_2h),
        "bias": bias,
        "upcoming_count": len(upcoming),
        "next_event": next_event,
        "pre_news_bias": pre_news_bias,
        "summary": "\n".join([f"- {e['time']} | {e['currency']} | {e['event']}" for e in upcoming[:5]])
    }

def format_news_summary(events, limit=5):
    if not events:
        return "No high-impact news in the upcoming window."
    items = events[:limit]
    return "\n".join([f"- {e['time']} {e['currency']}: {e['event']}" for e in items])

def format_east_africa_time(dt):
    try:
        east_africa = dt.astimezone(ZoneInfo("Africa/Nairobi"))
        return east_africa.strftime("%Y-%m-%d %H:%M EAT")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M UTC")

def is_same_day_event(event_dt, reference_dt=None):
    reference = reference_dt or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    try:
        east_ref = reference.astimezone(ZoneInfo("Africa/Nairobi"))
        east_event = event_dt.astimezone(ZoneInfo("Africa/Nairobi"))
        return east_ref.date() == east_event.date()
    except Exception:
        return event_dt.date() == reference.date()

def is_usd_sensitive_news(event):
    currency = str(event.get('currency') or '').upper()
    event_name = str(event.get('event') or '').upper()
    if currency == 'USD':
        return True
    usd_keywords = ['FED', 'FOMC', 'CPI', 'PPI', 'NFP', 'PAYROLL', 'UNEMPLOYMENT', 'JOBLESS', 'RETAIL SALES', 'GDP', 'ISM', 'PMI', 'CONSUMER CONFIDENCE', 'TREASURY', 'INFLATION', 'PCE', 'JOLTS', 'CONSTRUCTION', 'HOME SALES', 'TRADE BALANCE', 'DURABLE GOODS', 'MICHIGAN', 'FEDERAL RESERVE', 'DOLLAR', 'USD']
    return any(keyword in event_name for keyword in usd_keywords)

def filter_relevant_news(events, selected_symbols=None, reference_dt=None):
    reference = reference_dt or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    filtered = []
    for event in events:
        impact = str(event.get('impact') or '').upper()
        currency = str(event.get('currency') or '').upper()
        if impact not in ('HIGH', 'MEDIUM'):
            continue
        if currency != 'USD':
            continue
        event_dt = event.get('event_time_utc')
        if event_dt is None:
            continue
        if event_dt < reference - timedelta(hours=6):
            continue
        if event_dt > reference + timedelta(hours=168):
            continue
        filtered.append(event)
    filtered.sort(key=lambda e: e['event_time_utc'])
    return filtered

# ── Data Fetching & SMC Engines ──────────────────────────────────────────
def _build_dataframe_from_records(records):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    elif 'datetime' in df.columns:
        df['Date'] = pd.to_datetime(df['datetime'], utc=True)
    else:
        return pd.DataFrame()
    df = df.set_index('Date').sort_index()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

def fetch_candles_from_bitfinex(symbol, interval, limit=200):
    pair_map = {'BTCUSD': 'tBTCUSD', 'XAUUSD': 'tXAUUSD', 'EURUSD': 'tEURUSD', 'DXY': None}
    bitfinex_symbol = pair_map.get(symbol)
    if not bitfinex_symbol:
        return pd.DataFrame()
    interval_map = {'15m': '15m', '30m': '30m', '60m': '1h', '1h': '1h', '4h': '4h'}
    interval_code = interval_map.get(interval)
    if not interval_code:
        return pd.DataFrame()
    try:
        url = f'https://api-pub.bitfinex.com/v2/candles/trade:{interval_code}:{bitfinex_symbol}/hist?limit={limit}'
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return pd.DataFrame()
        records = []
        for item in payload:
            if not isinstance(item, list) or len(item) < 6:
                continue
            ts, open_price, close_price, high_price, low_price, volume = item[:6]
            records.append({
                'timestamp': ts,
                'Open': float(open_price),
                'High': float(high_price),
                'Low': float(low_price),
                'Close': float(close_price),
                'Volume': float(volume)
            })
        return _build_dataframe_from_records(records)
    except Exception as exc:
        print(f"⚠️ Bitfinex fetch failed for {symbol} [{interval}]: {exc}")
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_ohlcv(yf_symbol, interval, period):
    if yf is None:
        return pd.DataFrame()
    for symbol in ['BTCUSD', 'XAUUSD', 'EURUSD', 'DXY']:
        if yf_symbol in {symbol, YFINANCE_MAP.get(symbol, symbol)}:
            direct_df = fetch_candles_from_bitfinex(symbol, interval, limit=250)
            if not direct_df.empty:
                return direct_df
            break
    try:
        df = yf.download(yf_symbol, period=period, interval=interval, progress=False, auto_adjust=False, threads=False, timeout=30)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        if df.empty:
            return pd.DataFrame()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        if 'Date' in df.columns:
            df = df.set_index('Date')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        return df if len(df) >= 5 else pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Yahoo fetch failed for {yf_symbol} [{interval}/{period}]: {e}")
        return pd.DataFrame()

def fetch_symbol_data(symbol, yf_symbol):
    df_m10 = pd.DataFrame()
    for interval, period in [('10m', '5d'), ('15m', '5d')]:
        df_candidate = fetch_ohlcv(yf_symbol, interval, period)
        if not df_candidate.empty:
            df_m10 = df_candidate
            break
    if df_m10.empty:
        for interval, period in [('15m', '5d'), ('30m', '5d'), ('60m', '5d')]:
            df_candidate = fetch_ohlcv(yf_symbol, interval, period)
            if not df_candidate.empty:
                df_m10 = df_candidate
                break
    df_m15 = pd.DataFrame()
    for interval, period in [('15m', '5d'), ('30m', '5d')]:
        df_candidate = fetch_ohlcv(yf_symbol, interval, period)
        if not df_candidate.empty:
            df_m15 = df_candidate
            break
    df_m30 = pd.DataFrame()
    for interval, period in [('30m', '5d'), ('60m', '5d')]:
        df_candidate = fetch_ohlcv(yf_symbol, interval, period)
        if not df_candidate.empty:
            df_m30 = df_candidate
            break
    df_h1 = fetch_ohlcv(yf_symbol, '1h', '1mo')
    if df_h1.empty:
        df_h1 = fetch_ohlcv(yf_symbol, '60m', '1mo')
    df_h4 = fetch_ohlcv(yf_symbol, '4h', '3mo')
    if df_h4.empty:
        df_h4 = fetch_ohlcv(yf_symbol, '1d', '6mo')
    if df_m10.empty and not df_m15.empty:
        df_m10 = df_m15
    if df_m30.empty and not df_h1.empty:
        df_m30 = df_h1
    return {'M10': df_m10, 'M15': df_m15, 'M30': df_m30, 'H1': df_h1, 'H4': df_h4}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_all_data():
    data = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=min(5, len(YFINANCE_MAP))) as executor:
        for symbol, yf_symbol in YFINANCE_MAP.items():
            futures[executor.submit(fetch_symbol_data, symbol, yf_symbol)] = symbol
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data[symbol] = future.result()
            except Exception as e:
                print(f"❌ Exception fetching {symbol}: {str(e)}")
                data[symbol] = {'M10': pd.DataFrame(), 'H1': pd.DataFrame(), 'H4': pd.DataFrame()}
    return data

def calculate_microstructure(df):
    if df is None or len(df) < 2:
        return {}
    try:
        typical_price = (pd.to_numeric(df['High'], errors='coerce') + pd.to_numeric(df['Low'], errors='coerce') + pd.to_numeric(df['Close'], errors='coerce')) / 3
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
        current_vwap = float(vwap.iloc[-1])
        current_price = float(pd.to_numeric(df['Close'], errors='coerce').iloc[-1])
        avg_volume = volume.rolling(window=min(20, len(volume))).mean().iloc[-1]
        current_volume = float(volume.iloc[-1])
        rvol = current_volume / avg_volume if avg_volume > 0 else 1.0
        anchor_index = -min(5, len(df))
        price_change = current_price - float(pd.to_numeric(df['Close'], errors='coerce').iloc[anchor_index])
        return {
            "vwap": round(current_vwap, 2),
            "price_vs_vwap": "ABOVE" if current_price > current_vwap else "BELOW",
            "rvol": round(rvol, 2),
            "volume_anomaly": "HIGH_INSTITUTIONAL" if rvol > 2.0 else "NORMAL",
            "momentum": "BULLISH" if price_change > 0 else "BEARISH"
        }
    except Exception:
        return {}

def detect_bos_choch(df):
    if df is None or len(df) < 2:
        return None, None
    try:
        highs = pd.to_numeric(df['High'], errors='coerce').dropna()
        lows = pd.to_numeric(df['Low'], errors='coerce').dropna()
        if highs.empty or lows.empty or len(df) < 4:
            return None, None
        recent_high = float(highs.iloc[-1])
        prev_high = float(highs.iloc[-2]) if len(highs) >= 2 else recent_high
        recent_low = float(lows.iloc[-1])
        prev_low = float(lows.iloc[-2]) if len(lows) >= 2 else recent_low
        bos, choch = None, None
        if recent_high > prev_high * 1.001:
            bos = "BULLISH_BOS"
        elif recent_low < prev_low * 0.999:
            bos = "BEARISH_BOS"
        if bos == "BULLISH_BOS" and recent_low > prev_low:
            choch = "BULLISH_CHOCH"
        elif bos == "BEARISH_BOS" and recent_high < prev_high:
            choch = "BEARISH_CHOCH"
        return bos, choch
    except Exception:
        return None, None

def find_swings(df, window=5):
    highs = df['High'].rolling(window * 2 + 1, center=True).max()
    lows = df['Low'].rolling(window * 2 + 1, center=True).min()
    return {
        "recent_swing_highs": [round(p, 5) for p in df['High'][df['High'] == highs].tail(4).tolist()],
        "recent_swing_lows": [round(p, 5) for p in df['Low'][df['Low'] == lows].tail(4).tolist()]
    }

def detect_order_blocks(df):
    if len(df) < 5:
        return []
    order_blocks = []
    for i in range(len(df)-3, len(df)):
        if i < 2:
            continue
        candle, prev_candle = df.iloc[i], df.iloc[i-1]
        if (candle['Close'] > candle['Open'] and 
            (candle['Close'] - candle['Open']) > (candle['High'] - candle['Low']) * 0.6 and 
            prev_candle['Close'] < prev_candle['Open']):
            order_blocks.append({
                'type': 'BULLISH_OB',
                'price': candle['Low'],
                'strength': 'STRONG' if (candle['Close'] - candle['Open']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'
            })
        if (candle['Close'] < candle['Open'] and 
            (candle['Open'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.6 and 
            prev_candle['Close'] > prev_candle['Open']):
            order_blocks.append({
                'type': 'BEARISH_OB',
                'price': candle['High'],
                'strength': 'STRONG' if (candle['Open'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'
            })
    return order_blocks[-3:]

def detect_fvg(df):
    if len(df) < 3:
        return []
    fvgs = []
    for i in range(len(df)-2, len(df)):
        if i < 2:
            continue
        curr, prev, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        if prev['Low'] > prev2['High'] and curr['Low'] > prev['High']:
            fvgs.append({'type': 'BULLISH_FVG', 'top': prev['Low'], 'bottom': prev2['High']})
        if prev['High'] < prev2['Low'] and curr['High'] < prev['Low']:
            fvgs.append({'type': 'BEARISH_FVG', 'top': prev2['Low'], 'bottom': prev['High']})
    return fvgs[-2:]

def detect_liquidity_sweeps(df):
    if len(df) < 10:
        return []
    sweeps = []
    recent = df.tail(10)
    for i in range(1, len(recent)):
        candle, prev = recent.iloc[i], recent.iloc[i-1]
        if (candle['Low'] < prev['Low'] * 0.999 and 
            candle['Close'] > candle['Open'] and 
            (candle['Close'] - candle['Low']) > (candle['High'] - candle['Low']) * 0.6):
            sweeps.append({
                'type': 'BULLISH_SWEEP',
                'price': candle['Low'],
                'strength': 'STRONG' if (candle['Close'] - candle['Low']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'
            })
        if (candle['High'] > prev['High'] * 1.001 and 
            candle['Close'] < candle['Open'] and 
            (candle['High'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.6):
            sweeps.append({
                'type': 'BEARISH_SWEEP',
                'price': candle['High'],
                'strength': 'STRONG' if (candle['High'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'
            })
    return sweeps[-2:]

def calculate_atr(df, period=14):
    if len(df) < period + 2:
        return None
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None

def calculate_rsi(series, period=14):
    if series is None:
        return pd.Series(dtype=float)
    original = pd.Series(series)
    if len(original) < 2:
        return pd.Series([np.nan] * len(original), index=original.index, dtype=float)
    series = pd.to_numeric(original, errors='coerce').dropna()
    if series.empty:
        return pd.Series([np.nan] * len(original), index=original.index, dtype=float)
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    window = min(period, len(series))
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.reindex(original.index, fill_value=50).astype(float)

def detect_rsi_divergence(df, period=14):
    if df is None or df.empty:
        return None
    closes = pd.to_numeric(df['Close'], errors='coerce').dropna()
    if closes.empty or len(closes) < 4:
        return None
    rsi = calculate_rsi(closes, period=period)
    if rsi.empty:
        return None
    pivot_highs = []
    pivot_lows = []
    for i in range(2, len(closes) - 2):
        prev2 = closes.iloc[i - 2]
        prev1 = closes.iloc[i - 1]
        curr = closes.iloc[i]
        next1 = closes.iloc[i + 1]
        next2 = closes.iloc[i + 2]
        if curr >= prev1 and curr >= next1 and curr >= prev2 and curr >= next2:
            pivot_highs.append((i, float(curr), float(rsi.iloc[i])))
        if curr <= prev1 and curr <= next1 and curr <= prev2 and curr <= next2:
            pivot_lows.append((i, float(curr), float(rsi.iloc[i])))
    if len(pivot_highs) >= 2:
        prev_high = pivot_highs[-2]
        curr_high = pivot_highs[-1]
        if curr_high[1] < prev_high[1] and curr_high[2] > prev_high[2]:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is printing a lower high while RSI is holding a higher high, suggesting bullish divergence.'}
        if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is printing a higher high while RSI is failing, suggesting bearish divergence.'}
    if len(pivot_lows) >= 2:
        prev_low = pivot_lows[-2]
        curr_low = pivot_lows[-1]
        if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is printing a lower low while RSI is holding a higher low, suggesting bullish divergence.'}
        if curr_low[1] > prev_low[1] and curr_low[2] < prev_low[2]:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is printing a higher low while RSI is weakening, suggesting bearish divergence.'}
    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])
    last_rsi = float(rsi.iloc[-1])
    prev_rsi = float(rsi.iloc[-2])
    if last_close < prev_close and last_rsi > prev_rsi:
        return {'type': 'BULLISH_DIV', 'reason': 'The latest candles show a bullish RSI divergence against the recent price decline.'}
    if last_close > prev_close and last_rsi < prev_rsi:
        return {'type': 'BEARISH_DIV', 'reason': 'The latest candles show a bearish RSI divergence against the recent price advance.'}
    if len(closes) >= 6:
        lookback = min(5, len(closes) - 1)
        price_change = float(closes.iloc[-1] - closes.iloc[-lookback - 1])
        rsi_change = float(rsi.iloc[-1] - rsi.iloc[-lookback - 1])
        if price_change <= 0 and rsi_change > 0:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is failing to continue lower while RSI is rising, which is bullish divergence.'}
        if price_change >= 0 and rsi_change < 0:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is pushing higher while RSI is weakening, which is bearish divergence.'}
    return None

def detect_reversal(df):
    if len(df) < 8:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    body = abs(last['Close'] - last['Open'])
    total_range = last['High'] - last['Low']
    if total_range == 0:
        return None
    wick_ratio = max(last['High'] - max(last['Open'], last['Close']), min(last['Open'], last['Close']) - last['Low']) / total_range
    body_ratio = body / total_range
    bullish_reversal = last['Close'] > last['Open'] and last['Close'] >= prev['Close'] and wick_ratio > 0.45 and body_ratio < 0.45 and last['Low'] <= min(prev['Low'], prev2['Low'])
    bearish_reversal = last['Close'] < last['Open'] and last['Close'] <= prev['Close'] and wick_ratio > 0.45 and body_ratio < 0.45 and last['High'] >= max(prev['High'], prev2['High'])
    if bullish_reversal or bearish_reversal:
        return {
            'type': 'BULLISH_REVERSAL' if bullish_reversal else 'BEARISH_REVERSAL',
            'direction': 'BUY' if bullish_reversal else 'SELL',
            'strength': 'STRONG' if wick_ratio > 0.6 else 'MODERATE',
            'reason': 'The market is rejecting a previous extreme and showing a clean reversal candle.'
        }
    return None

def detect_continuation(df):
    if len(df) < 8:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    recent_high = max(df['High'].tail(4).iloc[:-1])
    recent_low = min(df['Low'].tail(4).iloc[:-1])
    bullish_continuation = last['Close'] > prev['Close'] and last['Close'] > recent_high and last['Low'] > recent_low
    bearish_continuation = last['Close'] < prev['Close'] and last['Close'] < recent_low and last['High'] < recent_high
    if bullish_continuation or bearish_continuation:
        return {
            'type': 'BULLISH_CONTINUATION' if bullish_continuation else 'BEARISH_CONTINUATION',
            'direction': 'BUY' if bullish_continuation else 'SELL',
            'strength': 'STRONG' if abs(last['Close'] - prev['Close']) > (last['High'] - last['Low']) * 0.5 else 'MODERATE',
            'reason': 'Price is extending beyond the recent range with follow-through, supporting continuation.'
        }
    return None

def detect_exhaustion(df):
    if len(df) < 8:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last['Close'] - last['Open'])
    total_range = last['High'] - last['Low']
    if total_range == 0:
        return None
    wick = max(last['High'] - max(last['Open'], last['Close']), min(last['Open'], last['Close']) - last['Low'])
    wick_ratio = wick / total_range
    body_ratio = body / total_range
    vol = float(last['Volume']) if 'Volume' in df.columns else 0.0
    avg_vol = float(df['Volume'].tail(10).mean()) if 'Volume' in df.columns else 0.0
    volume_spike = vol / avg_vol if avg_vol > 0 else 1.0
    bullish_exhaustion = last['Close'] > prev['Close'] and wick_ratio > 0.55 and body_ratio < 0.35 and volume_spike > 1.1
    bearish_exhaustion = last['Close'] < prev['Close'] and wick_ratio > 0.55 and body_ratio < 0.35 and volume_spike > 1.1
    if bullish_exhaustion or bearish_exhaustion:
        return {
            'type': 'BULLISH_EXHAUSTION' if bullish_exhaustion else 'BEARISH_EXHAUSTION',
            'direction': None,
            'strength': 'STRONG' if volume_spike > 1.3 else 'MODERATE',
            'reason': 'Price made a stretched wick into the current bar and closed with a weak body, signaling exhaustion.'
        }
    return None

def assess_entry_quality(df, swings, current_price, signal):
    if not swings:
        return {'entry_quality': 'unknown', 'entry_zone': None, 'distance_pct': None}
    if signal == 'BUY':
        swing_lows = [level for level in swings.get('recent_swing_lows', []) if level < current_price]
        zone = swing_lows[0] if swing_lows else None
        if zone is None:
            return {'entry_quality': 'unknown', 'entry_zone': None, 'distance_pct': None}
        distance_pct = abs(current_price - zone) / current_price
        quality = 'early' if distance_pct < 0.003 else 'acceptable' if distance_pct < 0.008 else 'late'
        return {'entry_quality': quality, 'entry_zone': zone, 'distance_pct': round(distance_pct, 6)}
    swing_highs = [level for level in swings.get('recent_swing_highs', []) if level > current_price]
    zone = swing_highs[0] if swing_highs else None
    if zone is None:
        return {'entry_quality': 'unknown', 'entry_zone': None, 'distance_pct': None}
    distance_pct = abs(zone - current_price) / current_price
    quality = 'early' if distance_pct < 0.003 else 'acceptable' if distance_pct < 0.008 else 'late'
    return {'entry_quality': quality, 'entry_zone': zone, 'distance_pct': round(distance_pct, 6)}

def detect_market_phase(df, swings=None):
    micro = calculate_microstructure(df)
    continuation = detect_continuation(df)
    reversal = detect_reversal(df)
    exhaustion = detect_exhaustion(df)
    divergence = detect_rsi_divergence(df)
    entry_quality = assess_entry_quality(df, swings, df['Close'].iloc[-1], 'BUY' if micro.get('momentum') == 'BULLISH' else 'SELL') if swings is not None else {'entry_quality': 'unknown'}
    direction = None
    if continuation:
        phase = 'continuation'
        reason = continuation['reason']
        direction = continuation.get('direction')
    elif reversal:
        phase = 'reversal'
        reason = reversal['reason']
        direction = reversal.get('direction')
    elif exhaustion:
        phase = 'exhaustion'
        reason = exhaustion['reason']
    elif micro.get('momentum') == 'BULLISH' and micro.get('price_vs_vwap') == 'ABOVE':
        phase = 'trend'
        reason = 'Price is holding above VWAP with positive momentum and is still in a directional trend.'
        direction = 'BUY'
    elif micro.get('momentum') == 'BEARISH' and micro.get('price_vs_vwap') == 'BELOW':
        phase = 'trend'
        reason = 'Price is holding below VWAP with negative momentum and is still in a directional trend.'
        direction = 'SELL'
    else:
        phase = 'coiling'
        reason = 'The market is coiling and lacks a clear continuation or reversal edge yet.'
    if divergence:
        direction = 'BUY' if divergence['type'] == 'BULLISH_DIV' else 'SELL'
        reason += f" RSI divergence hint: {divergence['reason']}"
    return {
        'phase': phase,
        'reason': reason,
        'direction': direction,
        'entry_quality': entry_quality.get('entry_quality', 'unknown'),
        'entry_zone': entry_quality.get('entry_zone'),
        'distance_pct': entry_quality.get('distance_pct')
    }

def range_position(df, current_price):
    if df is None or df.empty:
        return None
    look = df.tail(60)
    hi = float(look['High'].max())
    lo = float(look['Low'].min())
    if hi <= lo:
        return None
    return (float(current_price) - lo) / (hi - lo), lo, hi

def build_premium_discount_context(m10, current_price):
    pos_info = range_position(m10, current_price)
    if not pos_info:
        return 'Range position unavailable.'
    pos, lo, hi = pos_info
    zone = 'PREMIUM (upper half of range - SMC favors sells/shorts here)' if pos >= 0.5 else 'DISCOUNT (lower half of range - SMC favors buys/longs here)'
    return f"Recent 60-bar range {lo:.2f}-{hi:.2f}; price at {pos * 100:.0f}% of the range => {zone}."

def build_volatility_context(m10):
    atr = calculate_atr(m10)
    if atr is None or m10 is None or m10.empty:
        return 'ATR unavailable.'
    last = float(m10['Close'].iloc[-1])
    atr_pct = atr / last * 100 if last else 0.0
    return f"M10 ATR {atr:.2f} ({atr_pct:.2f}% of price). Judge whether volatility is expanded or compressed versus recent sessions."

def build_historical_context(m10):
    if m10 is None or m10.empty:
        return 'Historical context unavailable.'
    closes = [round(float(v), 4) for v in m10['Close'].tail(20).tolist()]
    if len(closes) < 2:
        return 'Historical context unavailable.'
    latest = float(m10['Close'].iloc[-1])
    prev = float(m10['Close'].iloc[-2])
    change_pct = round(((latest - prev) / prev) * 100, 2) if prev else 0.0
    high = float(m10['High'].tail(20).max())
    low = float(m10['Low'].tail(20).min())
    range_pct = round(((high - low) / latest) * 100, 2) if latest else 0.0
    return f"Last 20 closes: {closes}; latest 1-bar change: {change_pct}%; recent 20-bar range: {range_pct}%."

def compute_rsi_last(series, period=14):
    rsi = calculate_rsi(series, period)
    if rsi is None or rsi.empty:
        return None
    val = rsi.dropna()
    if val.empty:
        return None
    return round(float(val.iloc[-1]), 1)

def build_rsi_values_context(all_data, symbol):
    data = all_data.get(symbol, {}) or {}
    parts = []
    for label, key in [('M10', 'M10'), ('M15', 'M15'), ('M30', 'M30'), ('H1', 'H1'), ('H4', 'H4')]:
        df = data.get(key)
        if df is None or getattr(df, 'empty', True):
            continue
        v = compute_rsi_last(pd.to_numeric(df['Close'], errors='coerce'))
        if v is not None:
            state = 'OVERBOUGHT' if v >= 70 else 'OVERSOLD' if v <= 30 else 'neutral'
            parts.append(f"{label} RSI {v} ({state})")
    return ' | '.join(parts) if parts else 'RSI values unavailable.'

def build_market_structure_summary(df, current_price=None, swings=None, order_blocks=None, fvgs=None, sweeps=None, bos=None, choch=None, symbol=None):
    if df is None or df.empty:
        return 'Market structure unavailable.'
    current_price = float(current_price) if current_price is not None else float(df['Close'].iloc[-1])
    swings = swings or find_swings(df)
    recent_swing_highs = [float(v) for v in swings.get('recent_swing_highs', []) if v is not None]
    recent_swing_lows = [float(v) for v in swings.get('recent_swing_lows', []) if v is not None]
    zone_buffer = max(abs(current_price) * 0.0015, 0.5 if current_price >= 100 else 0.01)
    support_zones = [f"{max(level - zone_buffer, 0):.2f}-{level + zone_buffer:.2f}" for level in recent_swing_lows[:3]]
    resistance_zones = [f"{level - zone_buffer:.2f}-{level + zone_buffer:.2f}" for level in recent_swing_highs[:3]]
    demand_zones = []
    supply_zones = []
    for ob in (order_blocks or [])[:3]:
        if ob['type'].startswith('BULLISH'):
            demand_zones.append(f"{ob['price'] - zone_buffer:.2f}-{ob['price'] + zone_buffer:.2f}")
        else:
            supply_zones.append(f"{ob['price'] - zone_buffer:.2f}-{ob['price'] + zone_buffer:.2f}")
    for fvg in (fvgs or [])[:2]:
        if fvg['type'].startswith('BULLISH'):
            demand_zones.append(f"{fvg['bottom']:.2f}-{fvg['top']:.2f}")
        else:
            supply_zones.append(f"{fvg['bottom']:.2f}-{fvg['top']:.2f}")
    for sweep in (sweeps or [])[:2]:
        if sweep['type'].startswith('BULLISH'):
            demand_zones.append(f"{sweep['price'] - zone_buffer:.2f}-{sweep['price'] + zone_buffer:.2f}")
        else:
            supply_zones.append(f"{sweep['price'] - zone_buffer:.2f}-{sweep['price'] + zone_buffer:.2f}")
    structure_markers = []
    if bos:
        structure_markers.append(bos)
    if choch:
        structure_markers.append(choch)
    structure_markers = structure_markers[:2]
    support_text = ', '.join(support_zones) if support_zones else 'none'
    resistance_text = ', '.join(resistance_zones) if resistance_zones else 'none'
    demand_text = ', '.join(demand_zones) if demand_zones else 'none'
    supply_text = ', '.join(supply_zones) if supply_zones else 'none'
    marker_text = ', '.join(structure_markers) if structure_markers else 'balanced'
    return (
        f"Market structure zones: support={support_text}; resistance={resistance_text}; "
        f"demand={demand_text}; supply={supply_text}; structure markers={marker_text}; "
        f"current price={current_price:.2f}. The AI must identify whether price is approaching a demand/supply or support/resistance zone, "
        f"and whether the next move is a continuation or a reversal into the next clear invalidation zone."
    )

def build_multitimeframe_context(all_data, symbol):
    if not all_data or not symbol:
        return 'Multi-timeframe context unavailable.'
    data = all_data.get(symbol, {})
    if not isinstance(data, dict):
        return 'Multi-timeframe context unavailable.'
    frames = [('10M', data.get('M10')), ('15M', data.get('M15')), ('30M', data.get('M30')), ('1H', data.get('H1')), ('4H', data.get('H4'))]
    parts = []
    for label, frame in frames:
        if frame is None or getattr(frame, 'empty', True):
            continue
        micro = calculate_microstructure(frame)
        if not micro:
            micro = {'price_vs_vwap': 'NEUTRAL', 'momentum': 'NEUTRAL', 'rvol': 1.0}
        bos, choch = detect_bos_choch(frame)
        order_blocks = detect_order_blocks(frame)
        fvgs = detect_fvg(frame)
        sweeps = detect_liquidity_sweeps(frame)
        divergence = detect_rsi_divergence(frame)
        structure_bits = []
        if bos:
            structure_bits.append(bos)
        if choch:
            structure_bits.append(choch)
        if order_blocks:
            structure_bits.append(f"OB:{order_blocks[-1]['type']}")
        if fvgs:
            structure_bits.append(f"FVG:{fvgs[-1]['type']}")
        if sweeps:
            structure_bits.append(f"SWP:{sweeps[-1]['type']}")
        if divergence:
            structure_bits.append(divergence['type'])
        structure_text = ', '.join(structure_bits) if structure_bits else 'balanced structure'
        parts.append(f"{label}: price is {micro['price_vs_vwap']} VWAP, momentum is {micro['momentum']}, RVOL is {micro['rvol']:.2f}, and the current read is {structure_text}.")
    if not parts:
        return 'No usable higher-timeframe or lower-timeframe context is available.'
    return ' '.join(parts)

def detect_directional_confluence(df, swings=None, htf_context=None, dxy_context=None, symbol=None):
    if df is None or df.empty:
        return {'direction': None, 'bullish_evidence': [], 'bearish_evidence': [], 'bull_count': 0, 'bear_count': 0}
    bullish_evidence = []
    bearish_evidence = []
    micro = calculate_microstructure(df)
    reversal = detect_reversal(df)
    continuation = detect_continuation(df)
    sweeps = detect_liquidity_sweeps(df)
    divergence = detect_rsi_divergence(df)
    if reversal:
        if reversal.get('direction') == 'BUY':
            bullish_evidence.append(f"reversal candle rejecting lows ({reversal['type']})")
        else:
            bearish_evidence.append(f"reversal candle rejecting highs ({reversal['type']})")
    if continuation:
        if continuation.get('direction') == 'BUY':
            bullish_evidence.append(f"continuation break to the upside ({continuation['type']})")
        else:
            bearish_evidence.append(f"continuation break to the downside ({continuation['type']})")
    for sweep in sweeps[-2:]:
        if sweep['type'] == 'BULLISH_SWEEP':
            bullish_evidence.append('liquidity sweep of lows reclaimed')
        else:
            bearish_evidence.append('liquidity sweep of highs rejected')
    if divergence:
        if divergence['type'] == 'BULLISH_DIV':
            bullish_evidence.append('bullish RSI divergence')
        else:
            bearish_evidence.append('bearish RSI divergence')
    if micro.get('momentum') == 'BULLISH' and micro.get('price_vs_vwap') == 'ABOVE':
        bullish_evidence.append('price holding above VWAP with bullish momentum')
    elif micro.get('momentum') == 'BEARISH' and micro.get('price_vs_vwap') == 'BELOW':
        bearish_evidence.append('price holding below VWAP with bearish momentum')
    pos_info = range_position(df, df['Close'].iloc[-1])
    if pos_info:
        pos, lo, hi = pos_info
        if pos <= 0.35:
            bullish_evidence.append('price in discount zone of the recent range (favors longs)')
        elif pos >= 0.65:
            bearish_evidence.append('price in premium zone of the recent range (favors shorts)')
    if isinstance(htf_context, dict):
        htf_trend = str(htf_context.get('trend') or htf_context.get('bias') or '').upper()
        if htf_trend == 'BULLISH':
            bullish_evidence.append('higher-timeframe trend bullish')
        elif htf_trend == 'BEARISH':
            bearish_evidence.append('higher-timeframe trend bearish')
    if dxy_context and symbol in ['XAUUSD', 'EURUSD', 'BTCUSD']:
        trend = dxy_context.get('trend')
        pos = dxy_context.get('price_vs_vwap')
        if trend == 'BEARISH' and pos == 'BELOW':
            bullish_evidence.append('DXY weakness supporting longs')
        elif trend == 'BULLISH' and pos == 'ABOVE':
            bearish_evidence.append('DXY strength supporting shorts')
    bull = len(bullish_evidence)
    bear = len(bearish_evidence)
    direction = None
    if bull >= 2 and bull > bear:
        direction = 'BUY'
    elif bear >= 2 and bear > bull:
        direction = 'SELL'
    return {'direction': direction, 'bullish_evidence': bullish_evidence, 'bearish_evidence': bearish_evidence, 'bull_count': bull, 'bear_count': bear}

def build_mtf_picture(all_data, symbol):
    data = all_data.get(symbol, {}) or {}
    specs = [('M10', 1.0), ('M15', 1.0), ('M30', 1.5), ('H1', 2.0), ('H4', 3.0)]
    snaps = []
    for label, weight in specs:
        df = data.get(label)
        if df is None or getattr(df, 'empty', True):
            continue
        micro = calculate_microstructure(df)
        bos, choch = detect_bos_choch(df)
        sweeps = detect_liquidity_sweeps(df)
        divergence = detect_rsi_divergence(df)
        reversal = detect_reversal(df)
        continuation = detect_continuation(df)
        snaps.append({
            'label': label,
            'weight': weight,
            'micro': micro,
            'bos': bos,
            'choch': choch,
            'sweeps': sweeps,
            'divergence': divergence,
            'reversal': reversal,
            'continuation': continuation
        })
    score = 0.0
    bull_items, bear_items = [], []
    for s in snaps:
        w = s['weight']
        m = s['micro']
        if m.get('momentum') == 'BULLISH' and m.get('price_vs_vwap') == 'ABOVE':
            score += w
            bull_items.append(f"{s['label']} trend bullish (above VWAP)")
        elif m.get('momentum') == 'BEARISH' and m.get('price_vs_vwap') == 'BELOW':
            score -= w
            bear_items.append(f"{s['label']} trend bearish (below VWAP)")
        if s['bos'] == 'BULLISH_BOS' or s['choch'] == 'BULLISH_CHOCH':
            score += w
            bull_items.append(f"{s['label']} {s['bos'] or s['choch']}")
        elif s['bos'] == 'BEARISH_BOS' or s['choch'] == 'BEARISH_CHOCH':
            score -= w
            bear_items.append(f"{s['label']} {s['bos'] or s['choch']}")
        for sw in s['sweeps'][-1:]:
            if sw['type'] == 'BULLISH_SWEEP':
                score += 0.5 * w
                bull_items.append(f"{s['label']} swept lows reclaimed")
            else:
                score -= 0.5 * w
                bear_items.append(f"{s['label']} swept highs rejected")
        if s['divergence']:
            if s['divergence']['type'] == 'BULLISH_DIV':
                score += 0.5 * w
                bull_items.append(f"{s['label']} bullish RSI divergence")
            else:
                score -= 0.5 * w
                bear_items.append(f"{s['label']} bearish RSI divergence")
        if s['reversal']:
            if s['reversal']['direction'] == 'BUY':
                score += w
                bull_items.append(f"{s['label']} reversal candle rejecting lows")
            else:
                score -= w
                bear_items.append(f"{s['label']} reversal candle rejecting highs")
        if s['continuation']:
            if s['continuation']['direction'] == 'BUY':
                score += w
                bull_items.append(f"{s['label']} continuation break upside")
            else:
                score -= w
                bear_items.append(f"{s['label']} continuation break downside")
    htf_reads = []
    for s in snaps:
        if s['label'] in ('H1', 'H4'):
            m = s['micro']
            if m.get('momentum') == 'BULLISH' and m.get('price_vs_vwap') == 'ABOVE':
                htf_reads.append('BULLISH')
            elif m.get('momentum') == 'BEARISH' and m.get('price_vs_vwap') == 'BELOW':
                htf_reads.append('BEARISH')
            else:
                htf_reads.append('NEUTRAL')
    htf_bias = 'BULLISH' if htf_reads and all(r == 'BULLISH' for r in htf_reads) else ('BEARISH' if htf_reads and all(r == 'BEARISH' for r in htf_reads) else 'NEUTRAL')
    return {
        'snaps': snaps,
        'score': score,
        'bull_items': bull_items,
        'bear_items': bear_items,
        'htf_bias': htf_bias
    }

BIAS_MIN_HOLD_MINUTES = 45

def resolve_firm_direction(symbol, picture):
    score = picture.get('score', 0)
    htf = picture.get('htf_bias', 'NEUTRAL')
    notes = []
    base = 'BUY' if score >= 2.0 else ('SELL' if score <= -2.0 else None)
    if base and htf != 'NEUTRAL' and base != htf:
        if abs(score) >= 6.0:
            notes.append(f"lower-timeframe evidence is overwhelming ({score:+.1f}), overriding the {htf} HTF bias")
        else:
            base = htf
            notes.append(f"HTF bias {htf} overrides conflicting lower-timeframe noise")
    firm = base if base else (htf if htf != 'NEUTRAL' and abs(score) >= 2.0 else None)
    now = datetime.now()
    stored = st.session_state.directional_bias.get(symbol)
    if stored and (now - stored['since']).total_seconds() < BIAS_MIN_HOLD_MINUTES * 60:
        standing = stored['direction']
        if firm and firm != standing:
            counter = picture['bull_items'] if firm == 'BUY' else picture['bear_items']
            has_htf_break = any(('H1' in it or 'H4' in it) and ('BOS' in it or 'CHOCH' in it) for it in counter)
            if has_htf_break and abs(score) >= 6.0:
                notes.append(f"standing {standing} bias overridden by an H1/H4 structural break with strong evidence ({score:+.1f})")
            elif abs(score) >= 4.0:
                firm = None
                notes.append(f"tape contradicts the standing {standing} bias but lacks an H1/H4 structural break - standing aside (WAIT) instead of flipping")
            else:
                firm = standing
                notes.append(f"maintaining the standing {standing} bias set at {stored['since'].strftime('%H:%M')} - insufficient proof to flip")
        elif firm is None and abs(score) < 4.0:
            firm = standing
            notes.append(f"tape is quiet - maintaining the standing {standing} bias set at {stored['since'].strftime('%H:%M')}")
        elif firm is None:
            notes.append(f"tape strongly contradicts the standing {standing} bias - standing aside (WAIT) until structure resolves")
    if firm:
        if not stored or stored['direction'] != firm:
            st.session_state.directional_bias[symbol] = {'direction': firm, 'since': now}
    elif stored and (now - stored['since']).total_seconds() >= BIAS_MIN_HOLD_MINUTES * 60:
        st.session_state.directional_bias.pop(symbol, None)
    return firm, notes

def calculate_structural_score(df, symbol, dxy_context=None, news_context=None, phase_context=None):
    if df.empty or len(df) < 10:
        return {'structural_score': 0, 'score_reason': 'Insufficient data', 'candidate_direction': None, 'market_phase': 'coiling', 'phase_reason': 'Not enough data to assess structure.'}
    micro = calculate_microstructure(df)
    bos, choch = detect_bos_choch(df)
    order_blocks = detect_order_blocks(df)
    fvgs = detect_fvg(df)
    sweeps = detect_liquidity_sweeps(df)
    phase_context = phase_context or detect_market_phase(df)
    score = 42
    reasons = []
    if micro.get('price_vs_vwap') == 'ABOVE':
        score += 8
        reasons.append('price holding above VWAP')
    else:
        score += 4
        reasons.append('price trading near VWAP')
    if micro.get('momentum') == 'BULLISH':
        score += 6
        reasons.append('short-term momentum bullish')
    else:
        score += 4
        reasons.append('short-term momentum bearish')
    if micro.get('rvol', 0) > 2.0:
        score += 10
        reasons.append('strong institutional volume')
    elif micro.get('rvol', 0) < 0.5:
        score -= 8
        reasons.append('low volume / exhaustion risk')
    if bos == 'BULLISH_BOS' or choch == 'BULLISH_CHOCH':
        score += 10
        reasons.append('bullish BOS/CHOCH')
    elif bos == 'BEARISH_BOS' or choch == 'BEARISH_CHOCH':
        score += 10
        reasons.append('bearish BOS/CHOCH')
    if order_blocks:
        score += 6
        reasons.append('order block present')
    if fvgs:
        score += 6
        reasons.append('fair value gap present')
    if sweeps:
        score += 6
        reasons.append('liquidity sweep detected')
    phase = phase_context.get('phase')
    if phase == 'continuation':
        score += 8
        reasons.append('continuation structure is present')
    elif phase == 'reversal':
        score += 6
        reasons.append('reversal structure is forming')
    elif phase == 'exhaustion':
        score -= 5
        reasons.append('exhaustion is present and needs caution')
    entry_quality = phase_context.get('entry_quality')
    if entry_quality == 'early':
        score += 5
        reasons.append('entry zone is still early and actionable')
    elif entry_quality == 'late':
        score -= 4
        reasons.append('entry zone is late and may be chasing price')
    if dxy_context and symbol in ['XAUUSD', 'EURUSD', 'BTCUSD']:
        if dxy_context['trend'] == 'BULLISH' and dxy_context['price_vs_vwap'] == 'ABOVE':
            score -= 6
            reasons.append('DXY is suppressing the setup')
        elif dxy_context['trend'] == 'BEARISH' and dxy_context['price_vs_vwap'] == 'BELOW':
            score += 6
            reasons.append('DXY is supporting the setup')
    if news_context and news_context.get('within_2h'):
        if news_context.get('bias') == 'opposite':
            score -= 4
            reasons.append('news risk reduces conviction')
        else:
            score += 2
            reasons.append('news context remains supportive')
    score = max(0, min(100, int(score)))
    phase_dir = phase_context.get('direction')
    candidate_direction = None
    if phase_dir and score >= 70:
        candidate_direction = phase_dir
    elif score >= 75 and micro.get('momentum') == 'BULLISH':
        candidate_direction = 'BUY'
    elif score >= 75 and micro.get('momentum') == 'BEARISH':
        candidate_direction = 'SELL'
    return {
        'structural_score': score,
        'score_reason': '; '.join(reasons[-4:]),
        'candidate_direction': candidate_direction,
        'market_phase': phase,
        'phase_reason': phase_context.get('reason', 'Structure is being assessed.'),
        'entry_quality': entry_quality,
        'entry_zone': phase_context.get('entry_zone')
    }

def build_setup_context(df, swings, current_price, symbol, dxy_context=None, news_context=None):
    micro = calculate_microstructure(df)
    phase_context = detect_market_phase(df, swings=swings)
    continuation = detect_continuation(df)
    reversal = detect_reversal(df)
    exhaustion = detect_exhaustion(df)
    atr = calculate_atr(df)
    if continuation:
        setup_type = 'continuation'
        setup_bias = continuation.get('direction') or ('BUY' if micro.get('momentum') == 'BULLISH' else 'SELL')
    elif reversal:
        setup_type = 'reversal'
        setup_bias = reversal.get('direction') or ('BUY' if micro.get('momentum') == 'BULLISH' else 'SELL')
    elif exhaustion:
        setup_type = 'exhaustion'
        setup_bias = 'WAIT'
    else:
        setup_type = 'coiling'
        setup_bias = 'WAIT'
    entry_quality = phase_context.get('entry_quality', 'unknown')
    timing_state = 'ready' if entry_quality == 'early' else 'watch' if entry_quality == 'acceptable' else 'late'
    if setup_type == 'exhaustion' or entry_quality == 'late':
        timing_state = 'late'
    return {
        'setup_type': setup_type,
        'setup_bias': setup_bias,
        'phase': phase_context.get('phase', 'coiling'),
        'phase_reason': phase_context.get('reason', 'Structure is forming.'),
        'entry_quality': entry_quality,
        'entry_timing': timing_state,
        'atr': atr,
        'micro': micro
    }

def get_pair_config(symbol):
    base = {
        'digits': 2,
        'tick_size': 0.01,
        'min_dist_pct': 0.0015,
        'max_risk_pct': 0.008,
        'min_rr': 1.3,
        'target_rr': 2.0,
        'max_rr': 3.0,
        'score_floor': MINIMUM_CONFLUENCE_SCORE,
        'candidate_score': MINIMUM_CONFLUENCE_SCORE,
        'cooldown_minutes': 30,
        'max_entry_gap_pct': 0.0025,
        'max_entry_points': 10,
        'min_stop_atr': 1.0,
        'max_stop_atr': 3.0,
        'stop_buffer_atr': 0.25,
        'tp_buffer_atr': 0.12,
        'market_zone_atr': 0.20,
        'limit_zone_atr': 0.90,
        'stop_zone_atr': 0.90,
        'spread_multiplier': 1.5,
    }
    overrides = {
        'XAUUSD': {
            'digits': 2,
            'tick_size': 0.01,
            'min_dist_pct': 0.0028,
            'max_risk_pct': 0.010,
            'max_entry_gap_pct': 0.003,
            'max_entry_points': 10,
            'min_stop_atr': 1.2,
            'target_rr': 2.0,
            'max_rr': 3.0,
        },
        'EURUSD': {
            'digits': 5,
            'tick_size': 0.00001,
            'min_dist_pct': 0.0012,
            'max_risk_pct': 0.004,
            'max_entry_gap_pct': 0.002,
            'max_entry_points': 0.0025,
            'min_stop_atr': 1.0,
            'target_rr': 2.0,
            'max_rr': 3.0,
            'max_stop_atr': 2.5,
        },
        'BTCUSD': {
            'digits': 2,
            'tick_size': 0.01,
            'min_dist_pct': 0.006,
            'max_risk_pct': 0.015,
            'max_entry_gap_pct': 0.004,
            'max_entry_points': 150,
            'min_stop_atr': 1.2,
            'target_rr': 2.0,
            'max_rr': 3.0,
            'max_stop_atr': 3.5,
        },
        'US30': {
            'digits': 1,
            'tick_size': 0.1,
            'min_dist_pct': 0.004,
            'max_risk_pct': 0.010,
            'max_entry_gap_pct': 0.003,
            'max_entry_points': 120,
            'min_stop_atr': 1.0,
            'target_rr': 2.0,
            'max_rr': 3.0,
        },
    }
    return {**base, **overrides.get(symbol, {})}

# ── FRED API & News Knowledge ────────────────────────────────────────────
NEWS_EVENT_KNOWLEDGE = {
    "NONFARM PAYROLLS": ("stronger USD when actual beats consensus", "NFP causes a violent initial spike. Upside surprises strengthen USD and push XAUUSD/EURUSD down; downside surprises weaken USD and lift gold. Whipsaw first, then trend in the surprise direction."),
    "CPI": ("stronger USD when inflation is hotter than consensus", "Hotter CPI raises yields and USD, pressuring XAUUSD/EURUSD; cooler CPI weakens USD and lifts gold. Reaction front-loaded in the first 15 minutes."),
    "FOMC": ("hawkish = stronger USD, dovish = weaker USD", "FOMC reprices the rate path. Hawkish surprises lift USD and hit gold/EURUSD; dovish does the opposite. Watch dot plot and Powell tone."),
    "GDP": ("stronger USD on upside surprise", "Strong GDP supports USD; weak GDP weighs. Milder reaction than CPI/NFP unless far from consensus."),
    "UNEMPLOYMENT": ("lower unemployment = stronger USD", "Falling unemployment strengthens USD; rising weighs."),
    "JOBLESS CLAIMS": ("fewer claims = stronger USD", "Weekly claims: lower prints support USD, higher prints weaken it. USD-sensitive MEDIUM impact."),
    "RETAIL SALES": ("stronger USD on upside surprise", "Strong retail sales support USD; weak sales weigh on USD."),
    "PMI": ("above-consensus PMI = stronger USD", "ISM/Flash PMI above expectations supports USD; below weighs. Watch the 50 line."),
    "PPI": ("hotter PPI = stronger USD", "Producer inflation feeds CPI expectations; hotter prints support USD."),
    "PCE": ("hotter core PCE = stronger USD", "Fed's preferred gauge. Hotter core PCE lifts USD and hits gold; cooler does the opposite."),
    "FED": ("hawkish Fed = stronger USD", "Fed communication reprices rate expectations and moves USD across pairs."),
    "INTEREST RATE": ("higher/hawkish = stronger USD", "Rate decisions move USD via yield differentials."),
}

FRED_SERIES_MAP = {
    "NONFARM PAYROLLS": "PAYEMS",
    "NON-FARM PAYROLLS": "PAYEMS",
    "PAYROLL": "PAYEMS",
    "UNEMPLOYMENT RATE": "UNRATE",
    "UNEMPLOYMENT": "UNRATE",
    "CORE CPI": "CPILFESL",
    "CPI": "CPIAUCSL",
    "CORE PCE": "PCEPILFE",
    "PCE": "PCEPILFE",
    "GDP": "GDP",
    "FOMC": "FEDFUNDS",
    "FED FUNDS": "FEDFUNDS",
    "INTEREST RATE": "FEDFUNDS",
    "RETAIL SALES": "RSAFS",
    "JOBLESS CLAIMS": "ICSA",
    "INITIAL CLAIMS": "ICSA",
    "MICHIGAN": "UMCSENT",
    "CONSUMER SENTIMENT": "UMCSENT",
    "ISM": "INDPRO",
    "PMI": "INDPRO",
    "INDUSTRIAL PRODUCTION": "INDPRO",
    "TREASURY": "DGS10",
}

def _match_fred_series(event_name):
    try:
        name = str(event_name).upper()
        for key, series_id in FRED_SERIES_MAP.items():
            if key in name:
                return series_id
    except Exception:
        pass
    return None

def fetch_fred_observations(series_id, limit=5):
    try:
        api_key = get_secret("FRED_API_KEY", "")
        if not api_key:
            return None
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()
        obs = data.get("observations", [])
        obs = [o for o in obs if o.get("value") not in (".", None, "")]
        if not obs:
            return None
        out = []
        for o in obs:
            try:
                out.append({"date": o.get("date"), "value": float(o.get("value"))})
            except Exception:
                continue
        return out or None
    except Exception:
        return None

def fetch_fred_historical_context(event, ttl_seconds=3600):
    try:
        name = str(event.get("event", "")).upper()
        series_id = _match_fred_series(name)
        if not series_id:
            return None
        cache_key = "_fred_cache_" + series_id
        now = datetime.now()
        cached = st.session_state.get(cache_key)
        if cached and (now - cached.get("fetched_at", now)).total_seconds() < ttl_seconds:
            return cached.get("text")
        observations = fetch_fred_observations(series_id, limit=5)
        if not observations:
            return None
        parts = []
        for o in observations:
            parts.append("{} = {}".format(o["date"], o["value"]))
        text = "FRED {} last {} releases (most recent first): {}".format(
            series_id, len(observations), "; ".join(parts))
        st.session_state[cache_key] = {"text": text, "fetched_at": now}
        return text
    except Exception:
        return None

def fetch_news_historical_context(event):
    try:
        name = str(event.get("event", "")).upper()
        curated = None
        for key, (usd_impact, history) in NEWS_EVENT_KNOWLEDGE.items():
            if key in name:
                curated = "Typical USD impact: {} | Historical pattern: {}".format(usd_impact, history)
                break
        fred_data = fetch_fred_historical_context(event)
        if fred_data and curated:
            return fred_data + " || " + curated
        if fred_data:
            return fred_data
        if curated:
            return curated
    except Exception:
        pass
    return None

# ── News Fetching ────────────────────────────────────────────────────────
def get_high_impact_news(selected_symbols=None, reference_dt=None):
    now = datetime.now(timezone.utc)
    endpoints = [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    reference_dt = reference_dt or now
    final = []
    for url in endpoints:
        try:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code == 429:
                retry_after = res.headers.get("Retry-After", "1")
                try:
                    retry_delay = min(max(float(retry_after), 0.5), 3.0)
                except (TypeError, ValueError):
                    retry_delay = 1.0
                time.sleep(retry_delay)
                res = requests.get(url, headers=headers, timeout=20)
            res.raise_for_status()
            text = res.text
            if text.startswith("Title:") or "Markdown Content:" in text:
                text = text.split("Markdown Content:", 1)[-1].strip()
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload = payload.get('events') or payload.get('items') or payload.get('data') or []
            if not isinstance(payload, list):
                continue
            events = parse_news_payload(payload, reference_dt=reference_dt, lookahead_hours=168, only_high=False)
            if events:
                filtered_events = filter_relevant_news([{'event': e['event'], 'currency': e['currency'], 'impact': e['impact'], 'time': e['time'], 'event_time_utc': e['event_time_utc'], 'minutes_until': e['minutes_until'], 'within_2h': e['within_2h'], 'timezone': e['timezone']} for e in events], selected_symbols=selected_symbols, reference_dt=reference_dt)
                final = [{'time': format_east_africa_time(e['event_time_utc']), 'currency': e['currency'], 'event': e['event'], 'impact': e['impact'], 'minutes_until': e['minutes_until'], 'within_2h': e['within_2h'], 'timezone': e['timezone'], 'event_time_utc': e['event_time_utc'], 'event_id': f"{e['event']}|{e['currency']}|{e['event_time_utc'].strftime('%Y-%m-%d %H:%M:%S')}"} for e in filtered_events]
                if final:
                    break
        except Exception as exc:
            print(f"⚠️ News fetch failed for {url}: {exc}")
    return final

def sync_news_event_statuses(news_events, selected_symbols=None):
    statuses = st.session_state.news_event_statuses
    for event in news_events:
        event_id = event.get('event_id') or f"{event.get('event')}|{event.get('currency')}|{event.get('time')}"
        if not statuses.get(event_id):
            statuses[event_id] = {'event': event.get('event'), 'currency': event.get('currency'), 'time': event.get('time'), 'status': 'waiting', 'detail': 'Waiting for AI pre-news analysis (sent once, >=2h before release).'}
    stale_ids = [k for k in statuses if not any(e.get('event_id') == k for e in news_events)]
    for stale_id in stale_ids:
        del statuses[stale_id]
        st.session_state.news_signal_sent.pop(stale_id, None)
        st.session_state.news_results.pop(stale_id, None)
    st.session_state.news_event_statuses = statuses

def update_news_event_status(event, status, detail=None):
    if not event:
        return
    event_id = event.get('event_id') or f"{event.get('event')}|{event.get('currency')}|{event.get('time')}"
    st.session_state.news_event_statuses[event_id] = {'event': event.get('event'), 'currency': event.get('currency'), 'time': event.get('time'), 'status': status, 'detail': detail or ''}

# ── Rich News Analysis Prompt ────────────────────────────────────────────
def build_news_analysis_prompt():
    return """You are an elite news-driven macro analyst operating with the discipline of a professional trading desk. You analyze high-impact USD-sensitive news BEFORE it is released using a SYSTEMATIC MULTI-LAYER ANALYSIS to project how the event will affect trading pairs AT THE TIME of the news reading.
═══════════════════════════════════════════════════════════════════════════════
SYSTEMATIC ANALYSIS FRAMEWORK (FOLLOW IN ORDER):
═══════════════════════════════════════════════════════════════════════════════
LAYER 1 - HISTORICAL RELEASE PATTERN ANALYSIS:
- Recall the LAST 3-4 RELEASES of this specific event type
- For each release: What was the previous value? What was the consensus? What was the actual? What was the surprise?
- How did the market react to each surprise? (direction, magnitude, duration)
- What is the typical consensus expectation for THIS release?
- What would constitute a surprise vs consensus for THIS release?
LAYER 2 - CURRENT MARKET POSITIONING ANALYSIS:
- Analyze the CURRENT MARKET STRUCTURE provided (structure context, HTF context)
- Where is price positioned relative to key levels? (premium/discount, key support/resistance)
- What is the current momentum and trend across timeframes?
- What is the current RSI positioning across timeframes?
- What is the current DXY trend and positioning?
- Based on current positioning, is the market positioned FOR or AGAINST the expected news outcome?
LAYER 3 - NEWS IMPACT MECHANICS:
- How does THIS specific event type typically affect the US Dollar?
- How does THIS specific event type typically affect EACH SYMBOL (XAUUSD, EURUSD, BTCUSD, US30)?
- What is the typical reaction pattern? (immediate spike, delayed reaction, fade, continuation)
- What time of day is the release? (affects liquidity and reaction magnitude)
- What is the current market session? (affects liquidity and reaction magnitude)
LAYER 4 - CROSS-ASSET CORRELATION ANALYSIS:
- How do different symbols typically react to THIS event type?
- Are there any cross-asset correlations that confirm or contradict the expected move?
- Are there any divergences between assets that suggest a specific outcome?
LAYER 5 - SYNTHESIS AND DIRECTIONAL EDGE:
- Combine all layers to determine the EXPECTED NEWS OUTCOME (stronger/weaker USD)
- Determine the EXPECTED SYMBOL REACTION for each symbol
- Determine if current positioning is FOR or AGAINST the expected outcome
- Determine the directional edge: Should the trader be positioned LONG or SHORT when the news drops?
DATA PROVIDED:
{data_summary}
MICROSTRUCTURE (M10):
{microstructure_data}
STRUCTURE CONTEXT:
{structure_context}
RSI VALUES (MULTI-TIMEFRAME):
{rsi_values}
PREMIUM/DISCOUNT POSITION:
{premium_discount}
VOLATILITY (ATR):
{volatility_context}
HTF CONTEXT (H1/H4):
{htf_context}
DXY (US Dollar Index) TREND:
{dxy_data}
HISTORICAL CONTEXT:
{historical_context}
NEWS EVENT DETAILS:
{news_summary}
STRUCTURAL SCORE (PYTHON):
{structural_score_context}
PYTHON DIRECTIONAL LEDGER (REFERENCE EVIDENCE):
{directional_ledger}
═══════════════════════════════════════════════════════════════════════════════
MANDATORY ANALYSIS RULES:
═══════════════════════════════════════════════════════════════════════════════
1. COMPLETE ALL 5 LAYERS OF ANALYSIS before determining the final signal
2. For EACH SYMBOL, determine:
- Expected USD impact (stronger/weaker/neutral)
- Expected symbol reaction (up/down/neutral)
- Current positioning (for/against the expected move)
- Final directional edge (long/short/neutral)
3. Be SPECIFIC about historical patterns - cite specific previous releases and reactions
4. ALWAYS USE THE PROVIDED HISTORICAL CONTEXT: explicitly cite the last 3-4 releases (previous, consensus, actual, surprise) from the `HISTORICAL CONTEXT` field included in the DATA PROVIDED. If precise release numbers are not available in the field, state that explicitly and infer the pattern from the summary.
5. Be SPECIFIC about current positioning - cite specific levels and indicators
6. Be SPECIFIC about the expected reaction - cite the mechanism and timing
7. Output BUY or SELL when there is a clear directional edge
8. NEVER output WAIT for a news signal. Always pick BUY or SELL with detailed reasoning.
9. DO NOT output Entry, SL, or TP levels - focus ONLY on direction and reasoning
10. Write DETAILED reasoning (minimum 200 words) that shows your complete analysis process
OUTPUT STRICT JSON ONLY (NO MARKDOWN, NO CODE FENCES):
{{
"market_state": "continuation|reversal|exhaustion|trend|coiling",
"bias": "BULLISH|BEARISH|RANGING",
"signal": "BUY|SELL",
"confluence_score": 0,
"confidence": "HIGH|MEDIUM|LOW",
"dxy_correlation": "CONFIRMING|CONTRADICTING|NEUTRAL",
"microstructure_read": "Brief summary of VWAP/RVOL status",
"pre_news_bias": "Detailed explanation of expected USD impact and symbol reaction",
"directional_evidence": {{"bullish": ["..."], "bearish": ["..."]}},
"historical_pattern": "Detailed analysis of last 3-4 releases: previous values, consensus, actual, surprises, and market reactions",
"current_positioning": "Detailed analysis of current market positioning relative to expected news outcome",
"news_impact_mechanics": "Detailed explanation of how this event type affects USD and each symbol, including typical reaction patterns and timing",
"reasoning": "Complete synthesis of all 5 layers showing your complete analysis process (minimum 200 words)",
"rejection_reason": "If WAIT, detailed explanation of why there is no clear directional edge"
}}"""

# ── Groq API Integration (Robust Pattern) ────────────────────────────────
def estimate_tokens_for_text(text):
    return max(1, int(len(text) / 4))

def estimate_analysis_tokens(system_prompt, user_content):
    prompt_text = system_prompt + ' ' + ' '.join([item.get('text', '') for item in user_content if isinstance(item, dict)])
    return estimate_tokens_for_text(prompt_text) + GROQ_ESTIMATED_RESPONSE_TOKENS

def reserve_groq_tokens(estimated_tokens):
    now = datetime.now()
    window_start = st.session_state.groq_token_window_start
    if (now - window_start).total_seconds() >= 60:
        st.session_state.groq_token_window_start = now
        st.session_state.groq_tokens_used = 0
    if estimated_tokens is None:
        estimated_tokens = 0
    if st.session_state.groq_tokens_used + estimated_tokens > GROQ_TOKEN_LIMIT_PER_MINUTE:
        next_reset = st.session_state.groq_token_window_start + timedelta(minutes=1)
        st.session_state.groq_rate_limit_until = next_reset
        st.session_state.groq_rate_limit_reason = f"Token budget exceeded: {st.session_state.groq_tokens_used}/{GROQ_TOKEN_LIMIT_PER_MINUTE} used. Needs {estimated_tokens} more tokens and resets at {next_reset.strftime('%H:%M:%S')}."
        return False
    st.session_state.groq_rate_limit_reason = ''
    return True

def is_groq_rate_limited():
    retry_until = st.session_state.get('groq_rate_limit_until')
    return retry_until is not None and datetime.now() < retry_until

def get_groq_models(api_key):
    try:
        response = requests.get(
            f"{GROQ_API_URL.rsplit('/chat/completions', 1)[0]}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20
        )
        if response.status_code != 200:
            return GROQ_MODELS
        payload = response.json()
        available = [item.get('id') for item in payload.get('data', []) if item.get('id')]
        preferred = [model for model in GROQ_MODELS if model in available]
        additional = [
            model for model in available
            if model not in preferred
            and not any(blocked in model.lower() for blocked in ('whisper', 'guard', 'safety', 'tts', 'distil'))
        ]
        models = preferred + additional
        return models or GROQ_MODELS
    except Exception:
        return GROQ_MODELS

def call_groq(system_prompt, user_content, max_tokens=GROQ_MAX_OUTPUT_TOKENS, retry_count=0, estimated_tokens=None):
    api_key = get_secret("GROQ_API_KEY", "").strip()
    if not api_key:
        return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                "rejection_reason": "Missing Groq API Key.",
                "model_used": "Groq unavailable", "estimated_tokens": 0,
                "api_status": "MISSING_KEY"}
    if estimated_tokens is None:
        estimated_tokens = estimate_analysis_tokens(system_prompt, user_content)
    
    messages = [{"role": "system", "content": system_prompt}]
    for item in user_content:
        if isinstance(item, dict) and item.get("type") == "text":
            messages.append({"role": "user", "content": item.get("text", "")})
        elif isinstance(item, str):
            messages.append({"role": "user", "content": item})
            
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    request_started = False
    model_errors = []
    models = get_groq_models(api_key)
    
    for model in models:
        try:
            if is_groq_rate_limited():
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                        "rejection_reason": "RATE_LIMIT", "model_used": model,
                        "estimated_tokens": estimated_tokens, "api_status": "RATE_LIMIT"}
            time_since_last = (datetime.now() - st.session_state.last_groq_request_time).total_seconds() if st.session_state.last_groq_request_time else None
            if not request_started and time_since_last is not None and time_since_last < GROQ_MIN_REQUEST_INTERVAL:
                wait_time = int(GROQ_MIN_REQUEST_INTERVAL - time_since_last)
                st.session_state.groq_rate_limit_until = datetime.now() + timedelta(seconds=wait_time)
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                        "rejection_reason": "RATE_LIMIT", "model_used": model,
                        "estimated_tokens": estimated_tokens, "api_status": "SPACING_LIMIT"}
            if not reserve_groq_tokens(estimated_tokens):
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                        "rejection_reason": "RATE_LIMIT", "model_used": model,
                        "estimated_tokens": estimated_tokens, "api_status": "TOKEN_BUDGET"}
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": min(int(max_tokens), GROQ_MAX_OUTPUT_TOKENS),
                "response_format": {"type": "json_object"}
            }
            res = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=120)
            st.session_state.last_groq_request_time = datetime.now()
            request_started = True
            
            if res.status_code == 429:
                retry_after = int(res.headers.get('Retry-After', '60')) if res.headers.get('Retry-After') else 60
                error_text = res.text[:500]
                model_errors.append(f"{model}: HTTP 429 {error_text}")
                st.session_state.groq_rate_limit_until = datetime.now() + timedelta(seconds=retry_after)
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                    "rejection_reason": f"RATE_LIMIT: {error_text}", "model_used": model,
                    "estimated_tokens": estimated_tokens, "api_status": "RATE_LIMIT_429",
                    "raw_output": error_text}
            if res.status_code != 200:
                error_text = res.text[:500]
                model_errors.append(f"{model}: HTTP {res.status_code} {error_text}")
                continue
                
            res_data = res.json()
            usage = res_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            choices = res_data.get("choices", [])
            if not choices:
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                        "rejection_reason": "No choices returned",
                        "model_used": model, "api_status": "NO_CANDIDATES",
                        "total_tokens": total_tokens, "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens}
            
            content = choices[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
            content = str(content).strip()
            
            content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.IGNORECASE).strip()
            content = re.sub(r'\s*```$', '', content).strip()
            
            try:
                cleaned = re.sub(r',\s*([}\]])', r'\1', content)
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                first = content.find('{')
                last = content.rfind('}')
                if first != -1 and last != -1 and last > first:
                    substring = content[first:last+1]
                    try:
                        substring_clean = re.sub(r',\s*([}\]])', r'\1', substring)
                        result = json.loads(substring_clean)
                    except Exception:
                        return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                                "rejection_reason": "PARSE_ERROR",
                                "raw_output": content[:2000], "model_used": model,
                                "api_status": "PARSE_ERROR",
                                "total_tokens": total_tokens, "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens}
                else:
                    return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                            "rejection_reason": "PARSE_ERROR",
                            "raw_output": content[:2000], "model_used": model,
                            "api_status": "PARSE_ERROR",
                            "total_tokens": total_tokens}
                            
            result['model_used'] = model
            result['estimated_tokens'] = estimated_tokens
            result['total_tokens'] = total_tokens
            result['prompt_tokens'] = prompt_tokens
            result['completion_tokens'] = completion_tokens
            result['api_status'] = 'SUCCESS'
            st.session_state.groq_tokens_used += total_tokens
            st.session_state.groq_rate_limit_until = None
            st.session_state.groq_rate_limit_reason = ''
            return result
        except requests.exceptions.Timeout:
            model_errors.append(f"{model}: request timed out")
            continue
        except Exception as e:
            model_errors.append(f"{model}: {str(e)}")
            if model == models[-1]:
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                        "rejection_reason": f"Error: {str(e)}", "model_used": "None",
                        "api_status": "EXCEPTION", "estimated_tokens": estimated_tokens}
            continue
            
    return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
            "rejection_reason": "Error: all Groq models failed. " + " | ".join(model_errors[-3:]),
            "model_used": "None", "api_status": "ALL_MODELS_FAILED",
            "estimated_tokens": estimated_tokens,
            "raw_output": "\n".join(model_errors[-3:])}

# ── News Analysis Engine ─────────────────────────────────────────────────
def normalize_ai_signal(signal):
    if not isinstance(signal, str):
        return signal
    normalized = signal.strip().upper()
    if normalized in {'BULLISH', 'LONG', 'BUY'}:
        return 'BUY'
    if normalized in {'BEARISH', 'SHORT', 'SELL'}:
        return 'SELL'
    if normalized in {'WAIT', 'NO_TRADE', 'NONE'}:
        return 'WAIT'
    return signal

def normalize_analysis_signals(analysis):
    if not isinstance(analysis, dict):
        return analysis
    if 'signal' in analysis:
        analysis['signal'] = normalize_ai_signal(analysis['signal'])
    if 'candidate_direction' in analysis:
        analysis['candidate_direction'] = normalize_ai_signal(analysis['candidate_direction'])
    return analysis

def sanitize_news_event_analysis(analysis):
    if not isinstance(analysis, dict):
        return analysis
    for key in [
        'entry', 'stop_loss', 'take_profit', 'order_type', 'order_description',
        'order_blocks', 'fvgs', 'sweeps', 'candidate_levels', 'levels_source',
        'market_state', 'setup_context', 'validation_detail', 'confluence_breakdown',
        'order_description', 'order_type'
    ]:
        analysis.pop(key, None)
    analysis['is_news_signal'] = True
    return analysis

def analyze_news_for_symbol(symbol, all_data, event):
    try:
        data = all_data.get(symbol, {})
        m10 = data.get('M10', pd.DataFrame())
        h1 = data.get('H1', pd.DataFrame())
        h4 = data.get('H4', pd.DataFrame())
        if m10.empty:
            return {"error": f"Failed to fetch market data for {symbol}."}
        
        micro = calculate_microstructure(m10)
        current_price = float(m10['Close'].iloc[-1])
        swings = find_swings(m10)
        pair_config = get_pair_config(symbol)
        
        dxy_data = all_data.get('DXY', {}).get('H1', pd.DataFrame())
        dxy_summary = "DXY Data Unavailable"
        dxy_context = None
        if not dxy_data.empty:
            dxy_price = dxy_data['Close'].iloc[-1]
            dxy_micro = calculate_microstructure(dxy_data)
            dxy_context = {'trend': dxy_micro['momentum'], 'price_vs_vwap': dxy_micro['price_vs_vwap']}
            dxy_summary = f"Current: {dxy_price} | VWAP Position: {dxy_micro['price_vs_vwap']} | Momentum: {dxy_micro['momentum']} | RVOL: {dxy_micro['rvol']}"
        
        htf_context = None
        if not h4.empty:
            h4_micro = calculate_microstructure(h4)
            htf_context = {'trend': h4_micro['momentum'], 'bias': h4_micro['momentum'], 'price_vs_vwap': h4_micro['price_vs_vwap']}
        
        picture = build_mtf_picture(all_data, symbol)
        firm = None
        firm_notes = []
        if picture:
            firm, firm_notes = resolve_firm_direction(symbol, picture)
        
        news_payload = [{'event_time_utc': event.get('event_time_utc', datetime.now(timezone.utc) + timedelta(minutes=max(0, event.get('minutes_until', 0)))), 'within_2h': event.get('within_2h', False), 'time': event.get('time', ''), 'currency': event.get('currency', ''), 'event': event.get('event', '')}]
        nc = build_news_context(news_payload, reference_dt=datetime.now(timezone.utc))
        news_text = format_news_summary(news_payload, limit=len(news_payload))
        news_context = {'within_2h': nc.get('within_2h', False), 'bias': nc.get('bias', 'neutral'), 'upcoming_count': nc.get('upcoming_count', 0), 'next_event': nc.get('next_event'), 'pre_news_bias': nc.get('pre_news_bias', 'No imminent high-impact event.')}
        
        phase_context = detect_market_phase(m10, swings=swings)
        setup_context = build_setup_context(m10, swings, current_price, symbol, dxy_context=dxy_context, news_context=news_context)
        structural_context = calculate_structural_score(m10, symbol, dxy_context=dxy_context, news_context=news_context, phase_context=phase_context)
        
        bos, choch = detect_bos_choch(m10)
        order_blocks = detect_order_blocks(m10)
        fvgs = detect_fvg(m10)
        sweeps = detect_liquidity_sweeps(m10)
        
        structure_parts = []
        if bos or choch:
            structure_parts.append(f"BOS/CHOCH: {bos or choch}")
        if order_blocks:
            structure_parts.append("Order blocks: " + ", ".join([f"{ob['type']}@{ob['price']:.2f}" for ob in order_blocks]))
        if fvgs:
            structure_parts.append("FVGs: " + ", ".join([f"{fvg['type']}({fvg['top']:.2f}->{fvg['bottom']:.2f})" for fvg in fvgs]))
        if sweeps:
            structure_parts.append("Sweeps: " + ", ".join([f"{s['type']}@{s['price']:.2f}" for s in sweeps]))
        structure_context = " | ".join(structure_parts) if structure_parts else "No strong structural clues detected."
        
        market_structure_summary = build_market_structure_summary(m10, current_price=current_price, swings=swings, order_blocks=order_blocks, fvgs=fvgs, sweeps=sweeps, bos=bos, choch=choch, symbol=symbol)
        multitimeframe_context = build_multitimeframe_context(all_data, symbol)
        rsi_values = build_rsi_values_context(all_data, symbol)
        premium_discount = build_premium_discount_context(m10, current_price)
        volatility_context = build_volatility_context(m10)
        
        ledger = detect_directional_confluence(m10, swings=swings, htf_context=htf_context, dxy_context=dxy_context, symbol=symbol)
        directional_ledger = f"Bullish ({ledger['bull_count']}): {'; '.join(ledger['bullish_evidence']) or 'none'} | Bearish ({ledger['bear_count']}): {'; '.join(ledger['bearish_evidence']) or 'none'} | Ledger direction: {ledger['direction'] or 'none'}"
        
        h1_summary = f"Latest H1 close: {h1['Close'].iloc[-1]:.2f}" if not h1.empty else "H1 data unavailable"
        h4_summary = f"Latest H4 close: {h4['Close'].iloc[-1]:.2f}" if not h4.empty else "H4 data unavailable"
        htf_summary = f"H1: {h1_summary} | H4: {h4_summary}"
        
        prompt_data = f"Symbol: {symbol} | Live Price: {current_price} | Swing Highs: {swings['recent_swing_highs']} | Swing Lows: {swings['recent_swing_lows']} | Market Phase: {phase_context['phase']} | Phase Reason: {phase_context['reason']} | Setup Type: {setup_context['setup_type']} | Entry Timing: {setup_context['entry_timing']} | Entry Quality: {phase_context['entry_quality']}"
        prompt_micro = f"VWAP: {micro.get('vwap', 'N/A')} | Price vs VWAP: {micro.get('price_vs_vwap', 'N/A')} | RVOL: {micro.get('rvol', 'N/A')} ({micro.get('volume_anomaly', 'N/A')})"
        structural_score_context = f"Python structural score: {structural_context['structural_score']}/100 | Basis: {structural_context['score_reason']}"
        
        historical_context = fetch_news_historical_context(event)
        if not historical_context:
            historical_context = build_historical_context(m10)
        
        firm_bias_text = f"{firm} (standing desk bias; weighted MTF evidence {picture.get('score', 0):+.1f})" if firm else "NONE - evidence tied; stand aside unless a clear edge emerges."
        
        news_summary = f"Event: {event['event']} | Currency: {event['currency']} | Time: {event['time']} | Impact: {event['impact']} | Minutes Until: {event['minutes_until']}"
        
        all_format_kwargs = {
            'data_summary': prompt_data,
            'microstructure_data': prompt_micro,
            'structure_context': structure_context,
            'rsi_values': rsi_values,
            'premium_discount': premium_discount,
            'volatility_context': volatility_context,
            'htf_context': htf_summary,
            'dxy_data': dxy_summary,
            'historical_context': historical_context,
            'news_summary': news_summary,
            'structural_score_context': structural_score_context,
            'directional_ledger': directional_ledger,
        }
        
        prompt_template = build_news_analysis_prompt()
        try:
            prompt_text = prompt_template.format(**all_format_kwargs)
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            prompt_text = prompt_template.format(**{**all_format_kwargs, missing_key: f"[missing:{missing_key}]"})
        
        user_content = [{"type": "text", "text": prompt_text}]
        estimated_tokens = estimate_analysis_tokens(prompt_template, user_content)
        
        analysis = call_groq(prompt_template, user_content, max_tokens=GROQ_MAX_OUTPUT_TOKENS, estimated_tokens=estimated_tokens)
        
        analysis = normalize_analysis_signals(analysis)
        analysis.setdefault('model_used', PYTHON_FALLBACK_MODEL)
        analysis.setdefault('microstructure_read', prompt_micro)
        analysis.setdefault('live_price', current_price)
        analysis['setup_context'] = setup_context
        analysis['structural_score'] = structural_context['structural_score']
        analysis['atr'] = setup_context.get('atr')
        analysis['score_reason'] = structural_context['score_reason']
        analysis['candidate_direction'] = structural_context['candidate_direction']
        analysis['is_news_signal'] = True
        
        if analysis.get('api_status') not in ['SUCCESS', 'SUCCESS_EXTRACTED']:
            groq_failure = analysis.get('rejection_reason', 'Unknown API Error')
            groq_status = analysis.get('api_status', 'UNKNOWN')
            groq_model = analysis.get('model_used', 'None')
            groq_raw_output = analysis.get('raw_output', '')
            groq_tokens = {
                key: analysis.get(key, 0)
                for key in ('total_tokens', 'prompt_tokens', 'completion_tokens')
            }
            analysis = build_pre_news_fallback_analysis(symbol, m10, swings, pair_config, dxy_context, news_context, phase_context=phase_context, live_price=current_price, htf_context=htf_context, picture=picture, firm=firm, firm_notes=firm_notes, historical_context=historical_context)
            analysis = normalize_analysis_signals(analysis)
            analysis['groq_failure'] = groq_failure
            analysis['groq_api_status'] = groq_status
            analysis['groq_model'] = groq_model
            analysis.update(groq_tokens)
            if groq_raw_output:
                analysis['groq_raw_output'] = groq_raw_output
            analysis['estimated_tokens'] = analysis.get('estimated_tokens', estimated_tokens)
        
        analysis['symbol'] = symbol
        analysis['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        analysis['news_event'] = event.get('event', '')
        analysis['news_time'] = event.get('time', '')
        analysis['news_event_id'] = event.get('event_id')
        analysis = normalize_analysis_signals(analysis)
        
        return analysis
    except Exception as e:
        return {"error": str(e)}

def build_pre_news_fallback_analysis(symbol, m10, swings, pair_config, dxy_context, news_context, phase_context=None, live_price=None, htf_context=None, picture=None, firm=None, firm_notes=None, historical_context=None):
    micro = calculate_microstructure(m10) or {}
    all_data_ref = st.session_state.get("cached_market_data", {}) or {}
    htf_dir = None
    if htf_context:
        htf_dir = htf_context.get('trend') or htf_context.get('bias')
    
    nxt = (news_context or {}).get("next_event") or {}
    news_hist = fetch_news_historical_context(nxt) if isinstance(nxt, dict) else None
    usd_thesis = None
    if news_hist:
        low = news_hist.lower()
        if "weaker usd" in low:
            usd_thesis = "WEAKER_USD"
        elif "stronger usd" in low:
            usd_thesis = "STRONGER_USD"
    
    direction = None
    if usd_thesis == "STRONGER_USD":
        direction = "SELL" if symbol in ("XAUUSD", "EURUSD", "BTCUSD") else "BUY"
    elif usd_thesis == "WEAKER_USD":
        direction = "BUY" if symbol in ("XAUUSD", "EURUSD", "BTCUSD") else "SELL"
    else:
        direction = htf_dir
    
    if direction not in ("BUY", "SELL"):
        direction = htf_dir or ("BUY" if micro.get("momentum") == "BULLISH" else "SELL")
    
    confidence = "MEDIUM" if (htf_dir == direction or usd_thesis) else "LOW"
    bias = "BULLISH" if direction == "BUY" else "BEARISH"
    
    reasoning = "Pre-news directional read for {} (fallback, AI unavailable). ".format(symbol)
    reasoning += "Curated/FRED event impact: {}. ".format(news_hist or "no structured historical impact available")
    reasoning += "HTF trend: {}. ".format(htf_dir or "neutral")
    reasoning += "Direction set to {} from the USD thesis and higher-timeframe alignment. ".format(direction)
    reasoning += "Direction-only guidance; execution levels withheld until live AI analysis."
    
    return {
        "bias": bias,
        "signal": normalize_ai_signal(direction),
        "confluence_score": 72 if confidence == "MEDIUM" else 62,
        "confidence": confidence,
        "dxy_correlation": "CONFIRMING" if dxy_context else "NEUTRAL",
        "microstructure_read": "VWAP {} | RVOL {}".format(micro.get("price_vs_vwap", "NEUTRAL"), micro.get("rvol", 0)),
        "pre_news_bias": (news_context or {}).get("pre_news_bias", "News-driven reaction expected"),
        "reasoning": reasoning,
        "rejection_reason": None,
        "structural_score": 70,
        "score_reason": "Upgraded news fallback (USD thesis + HTF + FRED).",
        "candidate_direction": direction,
        "levels_source": "PYTHON",
        "historical_pattern": news_hist or (historical_context or ""),
        "api_status": "FALLBACK",
        "model_used": PYTHON_FALLBACK_MODEL
    }

def run_news_analysis_cycle(news_events, all_data, symbols):
    results_by_event = {}
    for event in news_events:
        eid = event.get('event_id') or f"{event.get('event')}|{event.get('currency')}|{event.get('time')}"
        if st.session_state.news_results.get(eid) or st.session_state.news_signal_sent.get(eid):
            update_news_event_status(event, 'sent', 'Already analyzed; skipping repeat AI run to save tokens.')
            continue
        
        update_news_event_status(event, 'analyzing', 'AI pre-news impact analysis in progress...')
        results = {}
        for symbol in NEWS_ANALYSIS_SYMBOLS:
            if st.session_state.last_groq_request_time:
                delta = (datetime.now() - st.session_state.last_groq_request_time).total_seconds()
                if delta < GROQ_MIN_REQUEST_INTERVAL:
                    time.sleep(GROQ_MIN_REQUEST_INTERVAL - delta)
            
            analysis = analyze_news_for_symbol(symbol, all_data, event)
            if not isinstance(analysis, dict):
                results[symbol] = {'signal': 'SKIPPED', 'reason': 'invalid result'}
                continue
            if 'error' in analysis:
                results[symbol] = {'signal': 'SKIPPED', 'reason': analysis.get('error')}
                continue
            if analysis.get('rejection_reason') == 'RATE_LIMIT':
                results[symbol] = {'signal': 'SKIPPED', 'reason': 'Groq rate limit'}
                continue
            results[symbol] = sanitize_news_event_analysis(analysis)
        
        st.session_state.news_results[eid] = {
            'event': event,
            'results': results,
            'analyzed_at': datetime.now()
        }
        
        valid = {s: a for s, a in results.items() if a.get('signal') in ('BUY', 'SELL', 'WAIT')}
        if not valid:
            update_news_event_status(event, 'waiting', 'Pre-news analysis failed (rate limit); retry next manual run.')
            continue
        
        st.session_state.news_signal_sent[eid] = {
            'summary_sent': False,
            'symbols': list(valid),
            'analysis_completed': True,
            'completed_at': datetime.now()
        }
        update_news_event_status(event, 'sent', 'Pre-news analysis completed; will not re-analyze.')
        
        message = build_news_event_telegram(event, results, datetime.now(timezone.utc))
        telegram_ok = send_telegram_message(message)
        if telegram_ok:
            st.session_state.news_signal_sent[eid]['summary_sent'] = True
            update_news_event_status(event, 'sent', 'Pre-news impact signal sent to Telegram (once).')
        else:
            update_news_event_status(event, 'sent', 'Pre-news analysis completed; Telegram delivery failed/not configured, but AI will not repeat.')
        
        results_by_event[eid] = results
    
    return results_by_event

# ── UI Layout ──────────────────────────────────────────────────────────────
st.title("📰 Der-AI | High-Impact News Analysis")
st.markdown("**5-Layer Systematic Analysis | FRED Historical Data | Groq AI Engine | Telegram Bridge**")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📰 Fetch News", "🧠 Analyse News", "📜 Signal History", "🔔 Notifications", "⚙️ Settings"])

with tab1:
    st.header("📅 Fetch High-Impact News")
    st.markdown("Click the button below to fetch the latest high-impact USD-sensitive economic events for the week.")
    
    if st.button("🔄 Fetch High-Impact News", type="primary"):
        with st.spinner("Fetching economic calendar..."):
            news = get_high_impact_news(selected_symbols=SYMBOLS, reference_dt=datetime.now(timezone.utc))
            if news:
                st.session_state.fetched_news = news
                sync_news_event_statuses(news, selected_symbols=SYMBOLS)
                st.success(f"✅ Successfully fetched {len(news)} high-impact events.")
            else:
                st.error("❌ Failed to fetch news. Please try again later.")
    
    if st.session_state.fetched_news:
        st.subheader("📋 Upcoming High-Impact Events")
        st.caption(f"Total events: {len(st.session_state.fetched_news)}")
        for n in st.session_state.fetched_news:
            event_id = n.get('event_id') or f"{n.get('event')}|{n.get('currency')}|{n.get('time')}"
            status_meta = st.session_state.news_event_statuses.get(event_id, {})
            status = status_meta.get('status', 'waiting')
            detail = status_meta.get('detail', 'Waiting for AI pre-news analysis.')
            urgency = "🟠 Within 2 hours" if n.get('within_2h') else "🟡 Upcoming"
            
            st.markdown(f"""
            <div class="news-card">
                <h4>{urgency} | {n['currency']}: {n['event']}</h4>
                <p><b>Time:</b> {n['time']} | <b>Impact:</b> {n['impact']} | <b>Minutes Until:</b> {n['minutes_until']}</p>
                <p><b>Status:</b> {status} - {detail}</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("🧠 Run AI News Analysis")
    if not st.session_state.fetched_news:
        st.warning("⚠️ No news fetched yet. Please go to the 'Fetch News' tab and fetch the calendar first.")
    else:
        st.info(f"Ready to analyse {len(st.session_state.fetched_news)} events for XAUUSD using Groq AI.")
        st.markdown("**Analysis Process:**")
        st.markdown("- Fetches market context for XAUUSD only")
        st.markdown("- Applies 5-layer systematic analysis framework")
        st.markdown("- Integrates FRED historical data + curated news knowledge")
        st.markdown("- Outputs direction-only signals (no Entry/SL/TP)")
        st.markdown("- Sends verified signals to Telegram")

        selected_event = None
        for event_index, event in enumerate(st.session_state.fetched_news):
            event_id = event.get('event_id') or f"{event.get('event')}|{event.get('currency')}|{event.get('time')}"
            already_analyzed = bool(st.session_state.news_results.get(event_id) or st.session_state.news_signal_sent.get(event_id))
            event_col, action_col = st.columns([4, 1])
            with event_col:
                st.markdown(f"**📌 {event['event']}**  \n`{event['currency']}` | `{event['time']}` | `{event['impact']}`")
            with action_col:
                button_label = "✅ Analysed" if already_analyzed else "🚀 Analyse News with AI"
                if st.button(button_label, key=f"analyse_news_{event_index}", type="primary", disabled=already_analyzed, use_container_width=True):
                    selected_event = event
            st.markdown("---")

        if selected_event:
            if not get_secret("GROQ_API_KEY"):
                st.error("⚠️ Please set your GROQ_API_KEY in Streamlit Secrets.")
            else:
                with st.spinner(f"Fetching market data and analysing {selected_event['event']}..."):
                    all_data = fetch_all_data()
                    st.session_state.cached_market_data = all_data
                    
                    results = run_news_analysis_cycle([selected_event], all_data, NEWS_ANALYSIS_SYMBOLS)
                    
                    for eid, event_results in results.items():
                        event = st.session_state.news_results[eid]['event']
                        st.subheader(f"📌 {event['event']} ({event['time']})")
                        
                        for symbol, analysis in event_results.items():
                            api_status = analysis.get('api_status', 'UNKNOWN')
                            model_used = analysis.get('model_used', 'Unknown')
                            total_tokens = analysis.get('total_tokens', 0)
                            prompt_tokens = analysis.get('prompt_tokens', 0)
                            completion_tokens = analysis.get('completion_tokens', 0)
                            status_color = "green" if api_status in ['SUCCESS', 'SUCCESS_EXTRACTED', 'FALLBACK'] else "red"
                            
                            st.markdown(f"**🤖 AI Model:** `{model_used}` | **🔋 Tokens Used:** `{total_tokens}` (Prompt: {prompt_tokens}, Completion: {completion_tokens}) | **📡 Status:** <span style='color:{status_color}; font-weight:bold;'>{api_status}</span>", unsafe_allow_html=True)
                            
                            if api_status == 'FALLBACK' and analysis.get('groq_failure'):
                                st.warning(f"Groq unavailable ({analysis.get('groq_api_status', 'UNKNOWN')}): {analysis['groq_failure']}")
                                
                            if api_status not in ['SUCCESS', 'SUCCESS_EXTRACTED', 'FALLBACK']:
                                with st.expander("🐛 Debug AI Response (Why it failed)"):
                                    st.code(analysis.get('raw_output', 'No raw output captured.'), language='json')
                                    st.error(f"Rejection Reason: {analysis.get('rejection_reason', 'Unknown')}")
                            
                            sig = analysis.get('signal', 'SKIPPED')
                            if sig in ('BUY', 'SELL'):
                                sig_emoji = "🟢" if sig == 'BUY' else "🔴"
                                css_class = "buy-signal" if sig == 'BUY' else "sell-signal"
                                st.markdown(f"""
                                <div class="signal-card {css_class}">
                                    <h3>{sig_emoji} {symbol} - {sig} | Score: {analysis.get('confluence_score', 0)}/100</h3>
                                    <p><b>Reasoning:</b> {analysis.get('reasoning', '')}</p>
                                    <p><b>Historical Pattern:</b> {analysis.get('historical_pattern', '')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.session_state.signal_history.append({
                                    'symbol': symbol,
                                    'signal': sig,
                                    'event': event['event'],
                                    'event_time': event['time'],
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'reasoning': analysis.get('reasoning', ''),
                                    'historical_pattern': analysis.get('historical_pattern', ''),
                                    'confluence_score': analysis.get('confluence_score', 0),
                                    'confidence': analysis.get('confidence', 'N/A'),
                                    'model_used': analysis.get('model_used', PYTHON_FALLBACK_MODEL),
                                    'total_tokens': total_tokens,
                                    'is_news_signal': True
                                })
                                
                                add_notification('success', f"✅ {symbol}: {sig} signal for {event['event']}. Score: {analysis.get('confluence_score', 0)}/100. Model: {model_used}. Tokens: {total_tokens}. Reasoning: {analysis.get('reasoning', 'Not provided.')}", symbol=symbol, signal=sig, score=analysis.get('confluence_score', 0))
                            elif sig == 'WAIT':
                                reasoning = analysis.get('reasoning', analysis.get('rejection_reason', 'No edge found.'))
                                st.info(f"⚪ {symbol}: WAIT - {reasoning}")
                                add_notification('info', f"⚪ {symbol}: WAIT for {event['event']}. Model: {model_used}. Tokens: {total_tokens}. Reasoning: {reasoning}", symbol=symbol, signal='WAIT')
                            else:
                                st.warning(f"⚪ {symbol}: SKIPPED - {analysis.get('reason', 'rate limit')}")
                                add_notification('warning', f"⚪ {symbol}: SKIPPED for {event['event']}.", symbol=symbol)
                        
                        st.markdown("---")
                    
                    st.success("✅ News analysis complete. Signals sent to Telegram.")

with tab3:
    st.header("📜 News Signal History")
    if not st.session_state.signal_history:
        st.info("📭 No news signals generated yet. Run an analysis in the 'Analyse News' tab.")
    else:
        news_signals = [s for s in st.session_state.signal_history if s.get('is_news_signal')]
        st.metric("Total News Signals Logged", len(news_signals))
        for signal in reversed(news_signals):
            with st.expander(f"{'🟢' if signal.get('signal') == 'BUY' else '🔴'} {signal.get('symbol')} - {signal.get('signal')} | {signal.get('event')} | {signal.get('timestamp')}"):
                st.write(f"**Event:** {signal.get('event')}")
                st.write(f"**Event Time:** {signal.get('event_time')}")
                st.write(f"**Model:** {signal.get('model_used', PYTHON_FALLBACK_MODEL)}")
                st.write(f"**Tokens Used:** {signal.get('total_tokens', 'N/A')}")
                st.write(f"**Confidence:** {signal.get('confidence')} | **Score:** {signal.get('confluence_score')}/100")
                st.write(f"**Reasoning:** {signal.get('reasoning')}")
                if signal.get('historical_pattern'):
                    st.write(f"**Historical Pattern:** {signal.get('historical_pattern')}")
                st.markdown("---")

with tab4:
    st.header("🔔 Notifications")
    if not st.session_state.notifications:
        st.info("📭 No notifications yet.")
    else:
        ctrl_col1, ctrl_col2 = st.columns([2, 1])
        with ctrl_col1:
            filter_type = st.selectbox("Filter by type", ["All", "Signals Only", "Warnings", "Info", "Success"], key="notif_filter")
        with ctrl_col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                clear_notifications()
                st.rerun()
        
        notifications = get_notifications()
        filtered_notifications = list(reversed(notifications))
        if filter_type == "Signals Only":
            filtered_notifications = [n for n in filtered_notifications if n.get('signal') in ('BUY', 'SELL')]
        elif filter_type == "Warnings":
            filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'warning']
        elif filter_type == "Info":
            filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'info']
        elif filter_type == "Success":
            filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'success']
        
        if not filtered_notifications:
            st.info("📭 No notifications match the current filter.")
        else:
            st.caption(f"Showing {len(filtered_notifications)} notification(s)")
            for note in filtered_notifications:
                note_type = note.get('type', 'info')
                badges = []
                if note.get('symbol'):
                    badges.append(f"`{note['symbol']}`")
                if note.get('signal'):
                    sig_emoji = "🟢" if note['signal'] == "BUY" else "🔴" if note['signal'] == "SELL" else "⚪"
                    badges.append(f"{sig_emoji} {note['signal']}")
                if note.get('score') is not None:
                    badges.append(f"📈 {note['score']}/100")
                header = f"**[{note.get('time', '')}]** {' '.join(badges)}" if badges else f"**[{note.get('time', '')}]**"
                if note_type == 'success':
                    st.success(f"{header}\n{note.get('message', '')}")
                elif note_type == 'warning':
                    st.warning(f"{header}\n{note.get('message', '')}")
                else:
                    st.info(f"{header}\n{note.get('message', '')}")

with tab5:
    st.header("⚙️ System Settings")
    st.info("Ensure `GROQ_API_KEY`, `FRED_API_KEY` (optional), `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID` are set in your Streamlit Secrets.")
    st.markdown("- **AI Model:** Llama 3.3 70B Versatile (via Groq)")
    st.markdown("- **Execution:** Manual trigger only (No auto-loop)")
    st.markdown("- **Features:** 5-Layer News Analysis, FRED API integration, Curated News Knowledge, Multi-Timeframe Context, DXY Correlation")
    st.markdown(f"- **Minimum Confluence Score:** {MINIMUM_CONFLUENCE_SCORE}/100")
    st.markdown("- **News Signals:** Direction-only (no Entry/SL/TP)")
    st.markdown("- **Telegram Bridge:** Signals sent once per event, ≥2h before release")
    st.markdown("- **Fallback:** Python rule-based model using USD thesis + HTF alignment when Groq unavailable")
