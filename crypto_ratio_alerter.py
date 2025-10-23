# -----------------------------------------------------------------------------
# --- 1. IMPORTS AND CONFIGURATION ---
# -----------------------------------------------------------------------------

import math
import time
import os
import ccxt
import requests
import pandas as pd
import numpy as np
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
BENCHMARK_SYMBOL = getattr(config, "BENCHMARK_SYMBOL", "BTC/USDT")
TARGET_RATIO = config.TARGET_RATIO
ALERT_CONDITION = config.ALERT_CONDITION
INTERVAL_NOTIFICATION = getattr(config, "INTERVAL_NOTIFICATION", 0.1)

# Technical Analysis Parameters
BAND_CALC_METHOD = getattr(config, "BAND_CALC_METHOD", "stddev")
BB_LENGTH = config.BB_LENGTH
BB_STDDEV = config.BB_STDDEV
ATR_LENGTH = getattr(config, "ATR_LENGTH", 14)
ATR_MULTIPLIER = getattr(config, "ATR_MULTIPLIER", 2.0)
RSI_LENGTH = config.RSI_LENGTH

# Dynamic RSI Percentile Settings
RSI_PERCENTILE_LOOKBACK = getattr(config, "RSI_PERCENTILE_LOOKBACK", 250)
RSI_OB_PERCENTILE = getattr(config, "RSI_OB_PERCENTILE", 90)
RSI_OS_PERCENTILE = getattr(config, "RSI_OS_PERCENTILE", 10)
RSI_SMOOTH_LENGTH = getattr(config, "RSI_SMOOTH_LENGTH", 5)

# Weighted Confidence Scoring
WEIGHT_BAND = getattr(config, "WEIGHT_BAND", 40)
WEIGHT_RSI = getattr(config, "WEIGHT_RSI", 35)
WEIGHT_CORR = getattr(config, "WEIGHT_CORR", 15)
WEIGHT_VOL = getattr(config, "WEIGHT_VOL", 10)
CONFIDENCE_THRESHOLD = getattr(config, "CONFIDENCE_THRESHOLD", 80)
MAX_CORRELATION = getattr(config, "MAX_CORRELATION", 0.3)

# Multi-Timeframe Settings
MTF_ENABLED = getattr(config, "MTF_ENABLED", True)
MTF_TIMEFRAME = getattr(config, "MTF_TIMEFRAME", "4h")
MTF_FILTER_STRENGTH = getattr(config, "MTF_FILTER_STRENGTH", "strict")

# Telegram and Timing
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', config.TELEGRAM_BOT_TOKEN)
TELEGRAM_CHAT_ID = config.TELEGRAM_CHAT_ID
CHECK_INTERVAL_SECONDS = config.CHECK_INTERVAL_SECONDS
ANALYSIS_TIMEFRAME = config.ANALYSIS_TIMEFRAME
NOTIFICATION_TIMEFRAME = config.NOTIFICATION_TIMEFRAME
HISTORY_LIMIT = config.HISTORY_LIMIT
REQUEST_TIMEOUT_SECONDS = getattr(config, "REQUEST_TIMEOUT_SECONDS", 10)

try:
    EXCHANGE = ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "defaultSubType": "linear",
            },
        }
    )
    EXCHANGE.load_markets()
    EXCHANGE._markets_loaded = True
except ccxt.BaseError as exc:
    print(f"Failed to initialize exchange client: {exc}")
    exit(1)
except Exception as exc:
    print(f"Unexpected error during exchange setup: {exc}")
    exit(1)


def _ensure_markets_loaded():
    """Ensures market metadata is cached to avoid repeated exchangeInfo requests."""
    if getattr(EXCHANGE, "_markets_loaded", False):
        return True
    try:
        EXCHANGE.load_markets()
        EXCHANGE._markets_loaded = True
        return True
    except ccxt.BaseError as exc:
        print(f"Exchange error while loading markets: {exc}")
    except Exception as exc:
        print(f"Unexpected error loading markets: {exc}")
    return False


def _safe_fetch_ohlcv(symbol, timeframe, limit):
    """Fetches OHLCV data while handling common issues gracefully."""
    if not _ensure_markets_loaded():
        return None

    data = None
    for attempt in range(2):
        try:
            data = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            break
        except ccxt.RateLimitExceeded as exc:
            wait_seconds = 1.5 * (attempt + 1)
            print(f"Rate limit hit fetching {symbol} ({timeframe}): {exc}. Sleeping {wait_seconds:.1f}s.")
            time.sleep(wait_seconds)
        except ccxt.BaseError as exc:
            print(f"Exchange error for {symbol} ({timeframe}): {exc}")
            return None
        except Exception as exc:  # Network or unexpected error
            print(f"Unexpected error fetching {symbol} ({timeframe}): {exc}")
            return None

    if data is None:
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
# --- 3. CORE CALCULATION FUNCTIONS ---
# -----------------------------------------------------------------------------

def compute_dynamic_rsi_percentile(rsi_series, percentile, smooth_length):
    """
    Computes dynamic RSI threshold using percentile ranking and EMA smoothing.
    
    Args:
        rsi_series: pandas Series of RSI values
        percentile: Target percentile (0-100)
        smooth_length: EMA smoothing window
    
    Returns:
        float: Smoothed dynamic RSI threshold, or None if insufficient data
    """
    try:
        if len(rsi_series) < RSI_PERCENTILE_LOOKBACK:
            return None
        
        # Calculate rolling percentile
        raw_percentile = rsi_series.rolling(window=RSI_PERCENTILE_LOOKBACK, min_periods=RSI_PERCENTILE_LOOKBACK).quantile(percentile / 100.0)
        
        # Apply EMA smoothing
        smoothed = raw_percentile.ewm(span=smooth_length, adjust=False).mean()
        
        return smoothed.iloc[-1] if not smoothed.empty else None
    except Exception as e:
        print(f"Error computing dynamic RSI percentile: {e}")
        return None


def compute_atr_bands(ratio_series, ma, atr_length, atr_multiplier):
    """
    Computes ATR-based dynamic bands for the ratio.
    
    Args:
        ratio_series: pandas Series of ratio values
        ma: Moving average of the ratio
        atr_length: ATR calculation period
        atr_multiplier: Multiplier for ATR
    
    Returns:
        tuple: (upper_band, lower_band) or (None, None) if insufficient data
    """
    try:
        if len(ratio_series) < atr_length + 1:
            return None, None
        
        # Calculate ATR from ratio changes
        ratio_changes = ratio_series.diff().abs()
        atr = ratio_changes.rolling(window=atr_length, min_periods=atr_length).mean().iloc[-1]
        
        if pd.isna(atr) or pd.isna(ma):
            return None, None
        
        upper_band = ma + (atr * atr_multiplier)
        lower_band = ma - (atr * atr_multiplier)
        
        return upper_band, lower_band
    except Exception as e:
        print(f"Error computing ATR bands: {e}")
        return None, None


def compute_weighted_scores(ratio, band_upper, band_lower, rsi, dynamic_rsi_ob, dynamic_rsi_os, 
                            correlation, rel_vol, rel_vol_ma):
    """
    Computes weighted confidence scores and condition counts for buy/sell signals.
    
    Returns:
        dict: {
            'sell_confidence': float (0-100),
            'buy_confidence': float (0-100),
            'sell_score': int (0-4),
            'buy_score': int (0-4),
            'conditions': dict of individual condition states
        }
    """
    try:
        # Initialize scores
        sell_conf = 0.0
        buy_conf = 0.0
        
        # Check individual conditions
        conditions = {
            'band_sell': not pd.isna(band_upper) and ratio > band_upper,
            'rsi_sell': not pd.isna(rsi) and not pd.isna(dynamic_rsi_ob) and rsi > dynamic_rsi_ob,
            'corr_sell': not pd.isna(correlation) and correlation < MAX_CORRELATION,
            'vol_sell': not pd.isna(rel_vol) and not pd.isna(rel_vol_ma) and rel_vol > rel_vol_ma,
            
            'band_buy': not pd.isna(band_lower) and ratio < band_lower,
            'rsi_buy': not pd.isna(rsi) and not pd.isna(dynamic_rsi_os) and rsi < dynamic_rsi_os,
            'corr_buy': not pd.isna(correlation) and correlation < MAX_CORRELATION,
            'vol_buy': not pd.isna(rel_vol) and not pd.isna(rel_vol_ma) and rel_vol > rel_vol_ma,
        }
        
        # Calculate weighted confidence for SELL
        if conditions['band_sell']:
            sell_conf += WEIGHT_BAND
        if conditions['rsi_sell']:
            sell_conf += WEIGHT_RSI
        if conditions['corr_sell']:
            sell_conf += WEIGHT_CORR
        if conditions['vol_sell']:
            sell_conf += WEIGHT_VOL
        
        # Calculate weighted confidence for BUY
        if conditions['band_buy']:
            buy_conf += WEIGHT_BAND
        if conditions['rsi_buy']:
            buy_conf += WEIGHT_RSI
        if conditions['corr_buy']:
            buy_conf += WEIGHT_CORR
        if conditions['vol_buy']:
            buy_conf += WEIGHT_VOL
        
        # Count conditions (confluence score)
        sell_score = sum([conditions['band_sell'], conditions['rsi_sell'], 
                         conditions['corr_sell'], conditions['vol_sell']])
        buy_score = sum([conditions['band_buy'], conditions['rsi_buy'], 
                        conditions['corr_buy'], conditions['vol_buy']])
        
        return {
            'sell_confidence': sell_conf,
            'buy_confidence': buy_conf,
            'sell_score': sell_score,
            'buy_score': buy_score,
            'conditions': conditions
        }
    except Exception as e:
        print(f"Error computing weighted scores: {e}")
        return {
            'sell_confidence': 0.0,
            'buy_confidence': 0.0,
            'sell_score': 0,
            'buy_score': 0,
            'conditions': {}
        }


def fetch_mtf_context(symbol1, symbol2, mtf_timeframe, ma_length):
    """
    Fetches higher timeframe data and computes MTF z-score and trend.
    Uses closed candles only (non-repainting).
    
    Returns:
        dict: {
            'htf_zscore': float or None,
            'htf_is_bull': bool,
            'htf_ratio': float or None
        }
    """
    try:
        # Fetch HTF data (limit to what we need + buffer)
        htf_limit = max(ma_length + 20, 50)
        htf_ohlcv1 = _safe_fetch_ohlcv(symbol1, mtf_timeframe, htf_limit)
        htf_ohlcv2 = _safe_fetch_ohlcv(symbol2, mtf_timeframe, htf_limit)
        
        if not htf_ohlcv1 or not htf_ohlcv2:
            print(f"MTF: Could not fetch HTF data for {mtf_timeframe}")
            return {'htf_zscore': None, 'htf_is_bull': False, 'htf_ratio': None}
        
        # Convert to DataFrames
        df1_htf = pd.DataFrame(htf_ohlcv1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df2_htf = pd.DataFrame(htf_ohlcv2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        if df1_htf.empty or df2_htf.empty:
            return {'htf_zscore': None, 'htf_is_bull': False, 'htf_ratio': None}
        
        # Calculate HTF ratio from CLOSED candles (use -1 to avoid current incomplete candle)
        htf_ratio_series = df1_htf['close'] / df2_htf['close']
        htf_ratio_series = htf_ratio_series.replace([pd.NA, np.inf, -np.inf], pd.NA).dropna()
        
        if len(htf_ratio_series) < ma_length:
            return {'htf_zscore': None, 'htf_is_bull': False, 'htf_ratio': None}
        
        # Use completed candle (second to last, since last might be incomplete)
        current_htf_ratio = htf_ratio_series.iloc[-2] if len(htf_ratio_series) > 1 else htf_ratio_series.iloc[-1]
        
        # Calculate HTF MA and StdDev
        htf_ma = htf_ratio_series.rolling(window=ma_length, min_periods=ma_length).mean().iloc[-2]
        htf_stdev = htf_ratio_series.rolling(window=BB_LENGTH, min_periods=BB_LENGTH).std().iloc[-2]
        
        # Calculate HTF Z-Score
        htf_zscore = None
        if not pd.isna(htf_ma) and not pd.isna(htf_stdev) and htf_stdev > 0:
            htf_zscore = (current_htf_ratio - htf_ma) / htf_stdev
        
        # Determine HTF trend
        htf_is_bull = current_htf_ratio > htf_ma if not pd.isna(htf_ma) else False
        
        return {
            'htf_zscore': htf_zscore,
            'htf_is_bull': htf_is_bull,
            'htf_ratio': current_htf_ratio
        }
    except Exception as e:
        print(f"Error fetching MTF context: {e}")
        return {'htf_zscore': None, 'htf_is_bull': False, 'htf_ratio': None}


# -----------------------------------------------------------------------------
# --- 4. DATA & LOGIC MODULE (Enhanced with New Features) ---
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

def get_trading_recommendation(z_score, rsi, dynamic_rsi_ob, dynamic_rsi_os, 
                              scores, base_symbol, quote_symbol, mtf_context):
    """
    Provides actionable trading recommendations based on weighted confidence scores.
    Returns a recommendation string with clear actions.
    """
    recommendation = ""
    
    sell_conf = scores['sell_confidence']
    buy_conf = scores['buy_confidence']
    sell_score = scores['sell_score']
    buy_score = scores['buy_score']

    # Primary Signal based on Confidence
    if sell_conf >= CONFIDENCE_THRESHOLD:
        recommendation += f"🔴 *HIGH CONFIDENCE SELL SIGNAL*\n"
        recommendation += f"💡 *Action:* Swap {base_symbol} → {quote_symbol}\n"
        recommendation += f"_Confidence: {sell_conf:.0f}% | Confluence: {sell_score}/4 conditions met_\n"
        recommendation += f"_{base_symbol} is extremely expensive vs {quote_symbol}_\n\n"
    elif buy_conf >= CONFIDENCE_THRESHOLD:
        recommendation += f"🟢 *HIGH CONFIDENCE BUY SIGNAL*\n"
        recommendation += f"💡 *Action:* Swap {quote_symbol} → {base_symbol}\n"
        recommendation += f"_Confidence: {buy_conf:.0f}% | Confluence: {buy_score}/4 conditions met_\n"
        recommendation += f"_{base_symbol} is extremely cheap vs {quote_symbol}_\n\n"
    else:
        # Moderate signals
        if sell_conf > buy_conf and sell_conf >= 50:
            recommendation += f"⚠️ *Moderate Sell Setup*\n"
            recommendation += f"_Confidence: {sell_conf:.0f}% ({sell_score}/4 conditions)_\n"
            recommendation += f"_Consider waiting for stronger confirmation_\n\n"
        elif buy_conf > sell_conf and buy_conf >= 50:
            recommendation += f"⚠️ *Moderate Buy Setup*\n"
            recommendation += f"_Confidence: {buy_conf:.0f}% ({buy_score}/4 conditions)_\n"
            recommendation += f"_Consider waiting for stronger confirmation_\n\n"
        else:
            recommendation += f"ℹ️ *No Clear Signal*\n"
            recommendation += f"💡 *Action:* HOLD or wait for better opportunity\n"
            recommendation += f"_Sell: {sell_conf:.0f}% | Buy: {buy_conf:.0f}%_\n\n"

    # Z-Score context
    if not pd.isna(z_score):
        recommendation += f"� *Z-Score:* {z_score:.2f} ({interpret_z_score(z_score)})\n"
    
    # Dynamic RSI with thresholds
    if not pd.isna(rsi) and not pd.isna(dynamic_rsi_ob) and not pd.isna(dynamic_rsi_os):
        if rsi > dynamic_rsi_ob:
            recommendation += f"� *RSI:* {rsi:.0f} (Overbought >{dynamic_rsi_ob:.0f})\n"
        elif rsi < dynamic_rsi_os:
            recommendation += f"🟢 *RSI:* {rsi:.0f} (Oversold <{dynamic_rsi_os:.0f})\n"
        else:
            recommendation += f"⚪ *RSI:* {rsi:.0f} (Neutral {dynamic_rsi_os:.0f}-{dynamic_rsi_ob:.0f})\n"
    
    # MTF Confirmation
    if MTF_ENABLED and mtf_context:
        htf_status = "Bullish ✓" if mtf_context['htf_is_bull'] else "Bearish ✓"
        recommendation += f"🕐 *HTF ({MTF_TIMEFRAME}):* {htf_status}"
        if not pd.isna(mtf_context.get('htf_zscore')):
            recommendation += f" (Z: {mtf_context['htf_zscore']:.2f})"
        recommendation += "\n"

    return recommendation

def get_market_data_and_metrics():
    """
    Fetches historical data, calculates the ratio, and computes enhanced TA metrics.
    Now includes: dynamic RSI percentiles, weighted confidence scoring, MTF confirmation,
    ATR bands option, and correlation analysis.
    
    Returns a dictionary with all the relevant data.
    """
    try:
        min_required = max(BB_LENGTH, RSI_LENGTH, RSI_PERCENTILE_LOOKBACK) + 10
        history_limit = max(HISTORY_LIMIT, min_required)
        if HISTORY_LIMIT < min_required and not getattr(get_market_data_and_metrics, "_history_limit_warned", False):
            print(f"Configured HISTORY_LIMIT ({HISTORY_LIMIT}) fetches {HISTORY_LIMIT} candles, "
                  f"but the indicators need at least {min_required}. Requesting {history_limit} candles this run. "
                  f"Update HISTORY_LIMIT in config.py to {min_required} or higher to skip this warning.")
            get_market_data_and_metrics._history_limit_warned = True

        # Fetch current prices using short timeframe for quick notifications
        current_ohlcv1 = _safe_fetch_ohlcv(TICKER_SYMBOL_1, NOTIFICATION_TIMEFRAME, 1)
        current_ohlcv2 = _safe_fetch_ohlcv(TICKER_SYMBOL_2, NOTIFICATION_TIMEFRAME, 1)

        # Fetch historical OHLCV data using longer timeframe for stable analysis
        ohlcv1 = _safe_fetch_ohlcv(TICKER_SYMBOL_1, ANALYSIS_TIMEFRAME, history_limit)
        ohlcv2 = _safe_fetch_ohlcv(TICKER_SYMBOL_2, ANALYSIS_TIMEFRAME, history_limit)
        
        # Fetch benchmark for correlation
        benchmark_ohlcv = _safe_fetch_ohlcv(BENCHMARK_SYMBOL, ANALYSIS_TIMEFRAME, history_limit)

        if not all([current_ohlcv1, current_ohlcv2, ohlcv1, ohlcv2]):
            return None

        # Convert to pandas DataFrame
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

        # Calculate the ratio series from the closing prices
        ratio_series = df1['close'] / df2['close']
        ratio_series = ratio_series.replace([pd.NA, np.inf, -np.inf], pd.NA).dropna()

        if ratio_series.empty:
            print("Ratio series is empty after cleaning.")
            return None

        if len(ratio_series) < min_required:
            print(f"Not enough data points for indicators (have {len(ratio_series)}, need {min_required}).")
            return None

        # --- Calculate Moving Average and StdDev for Z-Score ---
        sma = ratio_series.rolling(window=BB_LENGTH, min_periods=BB_LENGTH).mean().iloc[-1]
        stdev = ratio_series.rolling(window=BB_LENGTH, min_periods=BB_LENGTH).std().iloc[-1]

        # --- Calculate Bands (StdDev or ATR based) ---
        band_upper = None
        band_lower = None
        
        if BAND_CALC_METHOD == 'atr':
            band_upper, band_lower = compute_atr_bands(ratio_series, sma, ATR_LENGTH, ATR_MULTIPLIER)
        else:  # stddev (Bollinger Bands)
            if not pd.isna(sma) and not pd.isna(stdev):
                band_upper = sma + (stdev * BB_STDDEV)
                band_lower = sma - (stdev * BB_STDDEV)

        # --- Calculate RSI ---
        indicator_rsi = ta.momentum.RSIIndicator(close=ratio_series, window=RSI_LENGTH)
        rsi_series = indicator_rsi.rsi().dropna()
        
        if rsi_series.empty:
            print("RSI series is empty; insufficient data after indicator warm-up.")
            return None

        latest_rsi = rsi_series.iloc[-1]

        # --- Calculate Dynamic RSI Percentiles ---
        dynamic_rsi_ob = compute_dynamic_rsi_percentile(rsi_series, RSI_OB_PERCENTILE, RSI_SMOOTH_LENGTH)
        dynamic_rsi_os = compute_dynamic_rsi_percentile(rsi_series, RSI_OS_PERCENTILE, RSI_SMOOTH_LENGTH)

        # --- Calculate Correlation with Benchmark ---
        correlation = None
        if benchmark_ohlcv:
            df_bench = pd.DataFrame(benchmark_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if not df_bench.empty and len(df_bench) == len(ratio_series):
                # Align indices
                benchmark_series = df_bench['close'].reset_index(drop=True)
                ratio_aligned = ratio_series.reset_index(drop=True)
                # Calculate correlation (last 50 periods or HISTORY_LIMIT)
                corr_window = min(50, len(ratio_aligned))
                if corr_window >= 10:
                    correlation = ratio_aligned.tail(corr_window).corr(benchmark_series.tail(corr_window))

        # --- Calculate Relative Volume ---
        rel_vol = None
        rel_vol_ma = None
        if not df1.empty and not df2.empty:
            vol1 = df1['volume'].iloc[-1]
            vol2 = df2['volume'].iloc[-1]
            if vol2 > 0:
                rel_vol_series = df1['volume'] / df2['volume']
                rel_vol = vol1 / vol2
                rel_vol_ma = rel_vol_series.rolling(window=20, min_periods=20).mean().iloc[-1]

        # --- Use current prices for real-time ratio ---
        latest_ratio = current_price1 / current_price2

        # --- Calculate Z-Score ---
        z_score = None
        if not pd.isna(sma) and not pd.isna(stdev) and stdev > 0:
            z_score = (latest_ratio - sma) / stdev

        # --- Compute Weighted Confidence Scores ---
        scores = compute_weighted_scores(
            latest_ratio, band_upper, band_lower, 
            latest_rsi, dynamic_rsi_ob, dynamic_rsi_os,
            correlation, rel_vol, rel_vol_ma
        )

        # --- Fetch MTF Context ---
        mtf_context = None
        if MTF_ENABLED:
            mtf_context = fetch_mtf_context(TICKER_SYMBOL_1, TICKER_SYMBOL_2, MTF_TIMEFRAME, BB_LENGTH)

        # Return all data in a structured dictionary
        return {
            "price1": current_price1,
            "price2": current_price2,
            "ratio": latest_ratio,
            "z_score": z_score,
            "rsi": latest_rsi,
            "dynamic_rsi_ob": dynamic_rsi_ob,
            "dynamic_rsi_os": dynamic_rsi_os,
            "band_upper": band_upper,
            "band_lower": band_lower,
            "correlation": correlation,
            "rel_vol": rel_vol,
            "rel_vol_ma": rel_vol_ma,
            "scores": scores,
            "mtf_context": mtf_context
        }

    except ZeroDivisionError:
        print("Encountered division by zero while computing ratio.")
        return None
    except Exception as e:
        print(f"An error occurred while fetching data or calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# --- 5. MAIN APPLICATION LOOP (Enhanced with Confidence-Based Alerts) ---
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Advanced Ratio Alerter v2.0 (Pine Script Enhanced) ---")
    print(f"Monitoring Ratio: {BASE_SYMBOL}/{QUOTE_SYMBOL}")
    print(f"• Notification Speed: {NOTIFICATION_TIMEFRAME} (checked every {CHECK_INTERVAL_SECONDS}s)")
    print(f"• Analysis Timeframe: {ANALYSIS_TIMEFRAME} (for stable recommendations)")
    print(f"• Band Method: {BAND_CALC_METHOD.upper()}")
    print(f"• Confidence Threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"• MTF Confirmation: {MTF_TIMEFRAME if MTF_ENABLED else 'Disabled'}")
    print(f"Alerting when ratio is {ALERT_CONDITION} {TARGET_RATIO}")
    if INTERVAL_NOTIFICATION > 0:
        print(f"• Interval Notifications: Every {INTERVAL_NOTIFICATION} ratio change")
    print("------------------------------------------")

    alert_sent = False
    last_notified_interval = None
    interval_notifications_enabled = False
    last_sell_confidence = 0
    last_buy_confidence = 0

    try:
        while True:
            data = get_market_data_and_metrics()

            if data:
                scores = data.get('scores', {})
                sell_conf = scores.get('sell_confidence', 0)
                buy_conf = scores.get('buy_confidence', 0)
                sell_score = scores.get('sell_score', 0)
                buy_score = scores.get('buy_score', 0)
                mtf_context = data.get('mtf_context', {})
                ratio_above_threshold = data['ratio'] >= TARGET_RATIO
                
                z_interpretation = interpret_z_score(data['z_score']) if not pd.isna(data.get('z_score')) else "N/A"
                
                # Enhanced console output
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"{BASE_SYMBOL}: ${data['price1']:.4f} | "
                      f"{QUOTE_SYMBOL}: ${data['price2']:.4f} | "
                      f"Ratio: {data['ratio']:.6f} | "
                      f"Distance From Avg (Z-Score): {data['z_score']:.2f}σ ({z_interpretation}) | "
                      f"RSI: {data['rsi']:.1f} | "
                      f"Sell Confidence: {sell_conf:.0f}% ({sell_score}/4 cond) | "
                      f"Buy Confidence: {buy_conf:.0f}% ({buy_score}/4 cond)")

                # --- MTF Filter Logic ---
                mtf_sell_ok = True
                mtf_buy_ok = True
                
                if MTF_ENABLED and mtf_context:
                    htf_zscore = mtf_context.get('htf_zscore')
                    htf_is_bull = mtf_context.get('htf_is_bull', False)
                    
                    if MTF_FILTER_STRENGTH == 'strict':
                        # Strict: require HTF z-score confirmation
                        mtf_sell_ok = not pd.isna(htf_zscore) and htf_zscore < 0
                        mtf_buy_ok = not pd.isna(htf_zscore) and htf_zscore > 0
                    else:  # loose
                        # Loose: just check HTF trend direction
                        mtf_sell_ok = not htf_is_bull
                        mtf_buy_ok = htf_is_bull

                # --- High-Confidence Signal Detection (Edge-triggered) ---
                sell_trigger = (
                    sell_conf >= CONFIDENCE_THRESHOLD and 
                    last_sell_confidence < CONFIDENCE_THRESHOLD and
                    mtf_sell_ok
                )
                
                buy_trigger = (
                    buy_conf >= CONFIDENCE_THRESHOLD and 
                    last_buy_confidence < CONFIDENCE_THRESHOLD and
                    mtf_buy_ok
                )

                # Send high-confidence alerts
                if sell_trigger or buy_trigger:
                    signal_type = "SELL" if sell_trigger else "BUY"
                    confidence = sell_conf if sell_trigger else buy_conf
                    score = sell_score if sell_trigger else buy_score
                    
                    # Get trading recommendation
                    trading_rec = get_trading_recommendation(
                        data['z_score'], data['rsi'], 
                        data.get('dynamic_rsi_ob'), data.get('dynamic_rsi_os'),
                        scores, BASE_SYMBOL, QUOTE_SYMBOL, mtf_context
                    )
                    
                    message = (
                        f"� *HIGH CONFIDENCE {signal_type} ALERT* 🚨\n\n"
                        f"*{BASE_SYMBOL}/{QUOTE_SYMBOL}*\n"
                        f"• *Confidence:* {confidence:.0f}%\n"
                        f"• *Confluence:* {score}/4 conditions met\n"
                        f"• *MTF ({MTF_TIMEFRAME}):* {'Confirmed ✓' if MTF_ENABLED else 'Disabled'}\n\n"
                        "========================================\n\n"
                        f"*📊 Current Prices*\n"
                        f"• *{BASE_SYMBOL}:* `${data['price1']:.4f}`\n"
                        f"• *{QUOTE_SYMBOL}:* `${data['price2']:.4f}`\n"
                        f"• *Ratio:* `{data['ratio']:.6f}`\n\n"
                        "========================================\n\n"
                        f"*🎯 Recommendation*\n\n"
                        f"{trading_rec}\n"
                        "========================================\n\n"
                        f"*📈 Technical Details*\n\n"
                    )
                    
                    # Add condition details
                    conditions = scores.get('conditions', {})
                    if sell_trigger:
                        message += f"✓ Band Extreme: {'Yes' if conditions.get('band_sell') else 'No'}\n"
                        message += f"✓ RSI Overbought: {'Yes' if conditions.get('rsi_sell') else 'No'}\n"
                        message += f"✓ Low Correlation: {'Yes' if conditions.get('corr_sell') else 'No'}\n"
                        message += f"✓ High Volume: {'Yes' if conditions.get('vol_sell') else 'No'}\n"
                    else:
                        message += f"✓ Band Extreme: {'Yes' if conditions.get('band_buy') else 'No'}\n"
                        message += f"✓ RSI Oversold: {'Yes' if conditions.get('rsi_buy') else 'No'}\n"
                        message += f"✓ Low Correlation: {'Yes' if conditions.get('corr_buy') else 'No'}\n"
                        message += f"✓ High Volume: {'Yes' if conditions.get('vol_buy') else 'No'}\n"
                    
                    message += (
                        f"\n"
                        f"_Weighted scoring: Band {WEIGHT_BAND}%, RSI {WEIGHT_RSI}%, "
                        f"Corr {WEIGHT_CORR}%, Vol {WEIGHT_VOL}%_"
                    )
                    
                    send_telegram_notification(message)
                    print(f"✓ High-confidence {signal_type} alert sent!")

                # Update last confidence values
                last_sell_confidence = sell_conf
                last_buy_confidence = buy_conf

                # --- Interval Notification Logic (threshold-aware) ---
                if INTERVAL_NOTIFICATION > 0:
                    if ratio_above_threshold:
                        if not interval_notifications_enabled:
                            interval_notifications_enabled = True
                            last_notified_interval = None

                        current_interval = math.floor(data['ratio'] / INTERVAL_NOTIFICATION)

                        if last_notified_interval is None or current_interval != last_notified_interval:
                            interval_lower = current_interval * INTERVAL_NOTIFICATION
                            interval_upper = interval_lower + INTERVAL_NOTIFICATION

                            interval_message = (
                                f"📊 *Ratio Interval Alert*\n\n"
                                f"{BASE_SYMBOL}/{QUOTE_SYMBOL} entered new interval\n\n"
                                f"• *Current Ratio:* `{data['ratio']:.6f}`\n"
                                f"• *Interval:* `{interval_lower:.2f}` - `{interval_upper:.2f}`\n"
                                f"• *Z-Score:* `{data['z_score']:.2f}` ({z_interpretation})\n"
                                f"• *Confidence:* Sell {sell_conf:.0f}% | Buy {buy_conf:.0f}%"
                            )

                            if send_telegram_notification(interval_message):
                                last_notified_interval = current_interval
                                print(f"✓ Interval notification sent - range {interval_lower:.2f} to {interval_upper:.2f}")
                    else:
                        if interval_notifications_enabled:
                            interval_notifications_enabled = False
                            last_notified_interval = None

                # --- Legacy Target Ratio Alert Logic (Optional - can be disabled) ---
                trigger = False
                if ALERT_CONDITION == 'above' and data['ratio'] > TARGET_RATIO:
                    trigger = True
                elif ALERT_CONDITION == 'below' and data['ratio'] < TARGET_RATIO:
                    trigger = True

                if trigger and not alert_sent:
                    trading_rec = get_trading_recommendation(
                        data['z_score'], data['rsi'],
                        data.get('dynamic_rsi_ob'), data.get('dynamic_rsi_os'),
                        scores, BASE_SYMBOL, QUOTE_SYMBOL, mtf_context
                    )

                    message = (
                        f"🔔 *Ratio Alert: {BASE_SYMBOL}/{QUOTE_SYMBOL}* 🔔\n\n"
                        f"Ratio `{data['ratio']:.6f}` crossed *{ALERT_CONDITION}* `{TARGET_RATIO}`\n\n"
                        f"*📊 Current Prices*\n"
                        f"• *{BASE_SYMBOL}:* `${data['price1']:.4f}`\n"
                        f"• *{QUOTE_SYMBOL}:* `${data['price2']:.4f}`\n\n"
                        f"*🎯 Analysis*\n\n"
                        f"{trading_rec}"
                    )
                    
                    if send_telegram_notification(message):
                        alert_sent = True

                elif not trigger and alert_sent:
                    print("Ratio moved back to safe zone. Resetting alert flag.")
                    alert_sent = False

            time.sleep(CHECK_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("Received exit signal. Shutting down.")
