# -----------------------------------------------------------------------------
# --- 1. IMPORTS AND CONFIGURATION ---
# -----------------------------------------------------------------------------

import math
import time
import os
import ccxt
import requests
import pandas as pd
import ta
from requests.exceptions import RequestException

# Import configuration from config.py
try:
    import config
except ImportError:
    print("Error: config.py not found.")
    print("Please create a config.py file with your settings.")
    exit()

# --- Main Configuration ---
BASE_SYMBOL = config.BASE_SYMBOL
QUOTE_SYMBOL = config.QUOTE_SYMBOL
TICKER_SYMBOL_1 = config.TICKER_SYMBOL_1
TICKER_SYMBOL_2 = config.TICKER_SYMBOL_2
TARGET_RATIO = config.TARGET_RATIO
ALERT_CONDITION = config.ALERT_CONDITION
INTERVAL_NOTIFICATION = getattr(config, "INTERVAL_NOTIFICATION", 0.1)
BB_LENGTH = config.BB_LENGTH
BB_STDDEV = config.BB_STDDEV
RSI_LENGTH = config.RSI_LENGTH
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', config.TELEGRAM_BOT_TOKEN)
TELEGRAM_CHAT_ID = config.TELEGRAM_CHAT_ID
CHECK_INTERVAL_SECONDS = config.CHECK_INTERVAL_SECONDS
ANALYSIS_TIMEFRAME = config.ANALYSIS_TIMEFRAME
NOTIFICATION_TIMEFRAME = config.NOTIFICATION_TIMEFRAME
HISTORY_LIMIT = config.HISTORY_LIMIT
REQUEST_TIMEOUT_SECONDS = getattr(config, "REQUEST_TIMEOUT_SECONDS", 10)

try:
    EXCHANGE = ccxt.binance({"enableRateLimit": True})
except Exception as exc:
    print(f"Failed to initialize exchange client: {exc}")
    exit(1)


def _safe_fetch_ohlcv(symbol, timeframe, limit):
    """Fetches OHLCV data while handling common issues gracefully."""
    try:
        data = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except ccxt.BaseError as exc:
        print(f"Exchange error for {symbol} ({timeframe}): {exc}")
        return None
    except Exception as exc:  # Network or unexpected error
        print(f"Unexpected error fetching {symbol} ({timeframe}): {exc}")
        return None

    if not data:
        print(f"No OHLCV data returned for {symbol} ({timeframe}).")
        return None

    return data


# -----------------------------------------------------------------------------
# --- 2. NOTIFICATION MODULE (Unchanged) ---
# -----------------------------------------------------------------------------

def send_telegram_notification(message):
    """Sends a message to your configured Telegram chat."""
    print(f"Sending notification...")
    if not TELEGRAM_BOT_TOKEN:
        print("Telegram bot token missing. Skipping notification.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = { 'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown' }
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        print("Notification sent successfully.")
        return True
    except RequestException as exc:
        print(f"Error sending notification: {exc}")
        return False
    except Exception as exc:
        print(f"An unexpected error occurred while sending notification: {exc}")
        return False

# -----------------------------------------------------------------------------
# --- 3. DATA & LOGIC MODULE (Major Upgrade) ---
# -----------------------------------------------------------------------------

def interpret_z_score(z_score):
    """
    Interprets the Z-Score value and returns a human-readable description.
    Z-Score measures how many standard deviations the ratio is from its average.
    """
    abs_z = abs(z_score)
    if abs_z < 1:
        return "NORMAL (within 1 std dev)"
    elif abs_z < 2:
        return "MODERATE (1-2 std dev)"
    elif abs_z < 3:
        return "EXTREME (2-3 std dev)"
    else:
        return "VERY EXTREME (>3 std dev)"

def get_trading_recommendation(z_score, rsi, base_symbol, quote_symbol):
    """
    Provides actionable trading recommendations based on technical indicators.
    Returns a recommendation string with clear actions.
    """
    recommendation = ""

    # Analyze Z-Score (Ratio Trend)
    if z_score > 2:
        # Ratio is extremely high - OM is expensive relative to PIXEL
        recommendation += f"⚠️ *{base_symbol} is EXPENSIVE vs {quote_symbol}*\n"
        recommendation += f"💡 *Consider:* Swap {base_symbol} → {quote_symbol}\n"
        recommendation += f"_The ratio is {abs(z_score):.1f}x higher than normal. {base_symbol} might be overvalued._\n\n"
    elif z_score < -2:
        # Ratio is extremely low - OM is cheap relative to PIXEL
        recommendation += f"✅ *{base_symbol} is CHEAP vs {quote_symbol}*\n"
        recommendation += f"💡 *Consider:* Swap {quote_symbol} → {base_symbol}\n"
        recommendation += f"_The ratio is {abs(z_score):.1f}x lower than normal. {base_symbol} might be undervalued._\n\n"
    else:
        recommendation += f"ℹ️ *Ratio is in NORMAL range*\n"
        recommendation += f"💡 *Action:* HOLD or wait for better opportunity\n"
        recommendation += f"_No extreme price difference detected._\n\n"

    # Add RSI confirmation
    if rsi > 70 and z_score > 0:
        recommendation += f"🔴 *RSI Confirms:* Ratio is OVERBOUGHT ({rsi:.0f})\n"
        recommendation += f"_Strong signal to swap {base_symbol} → {quote_symbol}_\n"
    elif rsi < 30 and z_score < 0:
        recommendation += f"🟢 *RSI Confirms:* Ratio is OVERSOLD ({rsi:.0f})\n"
        recommendation += f"_Strong signal to swap {quote_symbol} → {base_symbol}_\n"
    elif rsi > 70:
        recommendation += f"⚠️ *RSI Warning:* Ratio momentum is high ({rsi:.0f})\n"
    elif rsi < 30:
        recommendation += f"⚠️ *RSI Warning:* Ratio momentum is low ({rsi:.0f})\n"

    return recommendation

def get_market_data_and_metrics():
    """
    Fetches historical data, calculates the ratio, and computes TA metrics.
    Uses dual timeframe: 5m for current prices, 1h/4h for analysis.
    Returns a dictionary with all the relevant data.
    """
    try:
        # Fetch current prices using short timeframe for quick notifications
        current_ohlcv1 = _safe_fetch_ohlcv(TICKER_SYMBOL_1, NOTIFICATION_TIMEFRAME, 1)
        current_ohlcv2 = _safe_fetch_ohlcv(TICKER_SYMBOL_2, NOTIFICATION_TIMEFRAME, 1)

        # Fetch historical OHLCV data using longer timeframe for stable analysis
        ohlcv1 = _safe_fetch_ohlcv(TICKER_SYMBOL_1, ANALYSIS_TIMEFRAME, HISTORY_LIMIT)
        ohlcv2 = _safe_fetch_ohlcv(TICKER_SYMBOL_2, ANALYSIS_TIMEFRAME, HISTORY_LIMIT)

        if not all([current_ohlcv1, current_ohlcv2, ohlcv1, ohlcv2]):
            return None

        # Convert to pandas DataFrame for easier manipulation
        df1 = pd.DataFrame(ohlcv1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df2 = pd.DataFrame(ohlcv2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        if df1.empty or df2.empty:
            print("Received empty OHLCV dataframes.")
            return None

        # Get current prices from the short timeframe
        current_price1 = current_ohlcv1[0][4]  # Close price
        current_price2 = current_ohlcv2[0][4]  # Close price

        if current_price1 is None or current_price2 in (None, 0):
            print("Invalid current prices received.")
            return None

        # Calculate the ratio series from the closing prices (using analysis timeframe)
        ratio_series = df1['close'] / df2['close']
        ratio_series = ratio_series.replace([pd.NA, math.inf, -math.inf], pd.NA).dropna()

        if ratio_series.empty:
            print("Ratio series is empty after cleaning.")
            return None

        min_required = max(BB_LENGTH, RSI_LENGTH) + 5  # buffer for warm-up periods
        if len(ratio_series) < min_required:
            print(f"Not enough data points for indicators (have {len(ratio_series)}, need {min_required}).")
            return None

        # --- Calculate Technical Indicators using ta ---
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(close=ratio_series, window=BB_LENGTH, window_dev=BB_STDDEV)

        # RSI
        indicator_rsi = ta.momentum.RSIIndicator(close=ratio_series, window=RSI_LENGTH)

        rsi_series = indicator_rsi.rsi().dropna()
        if rsi_series.empty:
            print("RSI series is empty; insufficient data after indicator warm-up.")
            return None

        # Use current prices from 5m timeframe for real-time monitoring
        latest_price1 = current_price1
        latest_price2 = current_price2
        # Calculate current ratio
        latest_ratio = latest_price1 / latest_price2

        # Get historical ratio for comparison
        historical_ratio = ratio_series.iloc[-1]
        latest_rsi = rsi_series.iloc[-1]

        # Calculate Z-Score manually from Bollinger Bands (using analysis timeframe data)
        # Z-Score = (Price - Moving Average) / Standard Deviation
        sma_series = indicator_bb.bollinger_mavg().dropna()
        hband_series = indicator_bb.bollinger_hband().dropna()

        if sma_series.empty or hband_series.empty:
            print("Bollinger Band series returned insufficient data.")
            return None

        sma = sma_series.iloc[-1]
        stdev = (hband_series.iloc[-1] - sma) / BB_STDDEV if BB_STDDEV > 0 else 0
        # Use current ratio for Z-Score calculation
        z_score = (latest_ratio - sma) / stdev if stdev > 0 else 0

        # Return all data in a structured dictionary
        return {
            "price1": latest_price1,
            "price2": latest_price2,
            "ratio": latest_ratio,
            "z_score": z_score,
            "rsi": latest_rsi
        }

    except ZeroDivisionError:
        print("Encountered division by zero while computing ratio.")
        return None
    except Exception as e:
        print(f"An error occurred while fetching data or calculating metrics: {e}")
        return None

# -----------------------------------------------------------------------------
# --- 4. MAIN APPLICATION LOOP (Upgraded Logic) ---
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Advanced Ratio Alerter Initialized ---")
    print(f"Monitoring Ratio: {BASE_SYMBOL}/{QUOTE_SYMBOL}")
    print(f"• Notification Speed: {NOTIFICATION_TIMEFRAME} (checked every {CHECK_INTERVAL_SECONDS}s)")
    print(f"• Analysis Timeframe: {ANALYSIS_TIMEFRAME} (for stable recommendations)")
    print(f"Alerting when ratio is {ALERT_CONDITION} {TARGET_RATIO}")
    if INTERVAL_NOTIFICATION > 0:
        print(f"• Interval Notifications: Every {INTERVAL_NOTIFICATION} ratio change")
    print("------------------------------------------")

    alert_sent = False
    last_notified_interval = None  # Track the last notified interval

    try:
        while True:
            data = get_market_data_and_metrics()

            if data:
                z_interpretation = interpret_z_score(data['z_score'])
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"{BASE_SYMBOL} Price: ${data['price1']:.4f}, "
                      f"{QUOTE_SYMBOL} Price: ${data['price2']:.4f}, "
                      f"Ratio: {data['ratio']:.6f}, "
                      f"Ratio Trend: {data['z_score']:.2f} ({z_interpretation}), "
                      f"RSI: {data['rsi']:.2f}")

                # --- Interval Notification Logic ---
                if INTERVAL_NOTIFICATION > 0:
                    # Calculate which interval we're currently in
                    current_interval = math.floor(data['ratio'] / INTERVAL_NOTIFICATION)
                    
                    # If we've crossed into a new interval, send notification
                    if last_notified_interval is None or current_interval != last_notified_interval:
                        interval_lower = current_interval * INTERVAL_NOTIFICATION
                        interval_upper = interval_lower + INTERVAL_NOTIFICATION
                        
                        # Create a concise interval notification with clear messaging
                        interval_message = (
                            f"📊 *Ratio Interval Alert*\n\n"
                            f"{BASE_SYMBOL}/{QUOTE_SYMBOL} has entered a new interval\n\n"
                            f"• *Current Ratio:* `{data['ratio']:.6f}`\n"
                            f"• *Interval Range:* `{interval_lower:.2f}` - `{interval_upper:.2f}`\n"
                            f"• *{BASE_SYMBOL} Price:* `${data['price1']:.4f}`\n"
                            f"• *{QUOTE_SYMBOL} Price:* `${data['price2']:.4f}`\n"
                            f"• *RSI:* `{data['rsi']:.2f}`\n"
                            f"• *Ratio Trend:* `{data['z_score']:.2f}` ({z_interpretation})"
                        )
                        
                        if send_telegram_notification(interval_message):
                            last_notified_interval = current_interval
                            print(f"✓ Interval notification sent - entered range {interval_lower:.2f} to {interval_upper:.2f}")

                # --- Target Ratio Alert Logic (Original) ---
                trigger = False
                if ALERT_CONDITION == 'above' and data['ratio'] > TARGET_RATIO:
                    trigger = True
                elif ALERT_CONDITION == 'below' and data['ratio'] < TARGET_RATIO:
                    trigger = True

                # If the condition is met AND we haven't sent an alert yet
                if trigger and not alert_sent:
                    # --- Build the new, detailed message ---
                    z_interpretation = interpret_z_score(data['z_score'])

                    # Additional context based on Z-Score sign
                    z_context = "ratio is ABOVE average" if data['z_score'] > 0 else "ratio is BELOW average"

                    # Get trading recommendation
                    trading_rec = get_trading_recommendation(data['z_score'], data['rsi'], BASE_SYMBOL, QUOTE_SYMBOL)

                    message = (
                        f"🔔 *Ratio Alert: {BASE_SYMBOL}/{QUOTE_SYMBOL}* 🔔\n\n"
                        f"The ratio `{data['ratio']:.6f}` has crossed *{ALERT_CONDITION}* your target of `{TARGET_RATIO}`.\n"
                        "========================================\n\n"
                        f"*📊 Current Prices*\n"
                        f"• *{BASE_SYMBOL}:* `${data['price1']:.4f}`\n"
                        f"• *{QUOTE_SYMBOL}:* `${data['price2']:.4f}`\n\n"
                        "========================================\n\n"
                        f"*📈 Technical Analysis*\n\n"
                        f"*Ratio Trend:* `{data['z_score']:.2f}` _{z_interpretation}_\n"
                        f"↳ The {z_context}\n"
                        f"_Measures how far the ratio is from normal levels_\n\n"
                        f"*RSI ({RSI_LENGTH}):* `{data['rsi']:.2f}`\n"
                        f"_Momentum indicator (>70 overbought, <30 oversold)_\n\n"
                        "========================================\n\n"
                        f"*🎯 Trading Recommendation*\n\n"
                        f"{trading_rec}\n"
                        "========================================\n\n"
                        f"*🧠 How This Works*\n\n"
                        f"*Dual-Timeframe System:*\n"
                        f"• Quick Alerts: {NOTIFICATION_TIMEFRAME} prices (checked every {CHECK_INTERVAL_SECONDS}s)\n"
                        f"• Smart Analysis: {ANALYSIS_TIMEFRAME} data (last {HISTORY_LIMIT} candles)\n\n"
                        f"*Step 1: Historical Analysis*\n"
                        f"• Analyzed {HISTORY_LIMIT} {ANALYSIS_TIMEFRAME} candles for stable patterns\n"
                        f"• {ANALYSIS_TIMEFRAME} timeframe filters out short-term noise\n"
                        f"• Calculated historical {BASE_SYMBOL}/{QUOTE_SYMBOL} ratio trends\n"
                        f"• Computed the average ratio and price volatility\n\n"
                        f"*Step 2: Statistical Comparison*\n"
                        f"• Current ratio: `{data['ratio']:.4f}`\n"
                        f"• Ratio Trend score: `{data['z_score']:.2f}` standard deviations from average\n"
                        f"• This means: Current ratio is _{z_interpretation.lower()}_\n\n"
                        f"*Step 3: Mean Reversion Strategy*\n"
                        f"• When ratio is EXTREME (±2+), it tends to revert to average\n"
                        f"• High ratio → {BASE_SYMBOL} likely to drop or {QUOTE_SYMBOL} to rise\n"
                        f"• Low ratio → {BASE_SYMBOL} likely to rise or {QUOTE_SYMBOL} to drop\n\n"
                        f"*Step 4: Momentum Confirmation (RSI)*\n"
                        f"• RSI {data['rsi']:.0f} shows if the trend is exhausted\n"
                        f"• Overbought (>70) + high ratio = strong sell signal\n"
                        f"• Oversold (<30) + low ratio = strong buy signal\n\n"
                        "========================================\n\n"
                        f"*💭 Simple Explanation*\n"
                        f"Think of it like a rubber band:\n"
                        f"• Ratio Trend = How stretched the rubber band is\n"
                        f"• +2 or higher = Stretched too far up (will snap back down)\n"
                        f"• -2 or lower = Stretched too far down (will snap back up)\n"
                        f"• Near 0 = Normal position (no strong force)\n\n"
                        f"The system predicts the rubber band will return to normal, creating profit opportunities!"
                    )
                    if send_telegram_notification(message):
                        alert_sent = True

                # Reset the alert flag if the ratio moves back to a "safe" zone
                elif not trigger and alert_sent:
                    print("Ratio has moved back to a safe zone. Resetting alert flag.")
                    alert_sent = False

            time.sleep(CHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("Received exit signal. Shutting down.")