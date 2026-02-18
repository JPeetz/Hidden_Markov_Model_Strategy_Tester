"""
Data Loader Module
Fetches BTC-USD hourly data using yfinance

Copyright: Joerg Peetz
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


def fetch_btc_data(
    ticker: str = "BTC-USD",
    days: int = 730,
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Fetch historical BTC-USD data from Yahoo Finance.

    Args:
        ticker: The ticker symbol (default: BTC-USD)
        days: Number of days of historical data (default: 730)
        interval: Data interval (default: 1h for hourly)

    Returns:
        DataFrame with OHLCV data and calculated features
    """
    import time

    end_date = datetime.now()

    # yfinance hourly data has strict limits - use period parameter instead
    # which is more reliable than start/end dates
    print(f"Fetching {ticker} hourly data...")

    # Try using period first (most reliable)
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="max", interval=interval)

        if df.empty:
            raise ValueError("Empty dataframe from period=max")

        print(f"  Fetched {len(df)} candles using period=max")

    except Exception as e:
        print(f"  Period method failed: {e}")
        print("  Trying chunked download...")

        # Fallback: fetch in smaller chunks
        chunk_days = 7  # Very small chunks
        all_data = []
        current_end = end_date
        max_days = min(days, 59)  # yfinance hourly limit is about 60 days

        for i in range(max_days // chunk_days + 1):
            current_start = current_end - timedelta(days=chunk_days)

            try:
                chunk = yf.download(
                    ticker,
                    start=current_start,
                    end=current_end,
                    interval=interval,
                    progress=False
                )

                if not chunk.empty:
                    all_data.append(chunk)
                    print(f"    Chunk {i+1}: {len(chunk)} candles")

            except Exception as chunk_err:
                print(f"    Chunk {i+1} failed: {chunk_err}")

            current_end = current_start
            time.sleep(0.5)  # Rate limiting

        if not all_data:
            raise ValueError(f"No data fetched for {ticker}")

        df = pd.concat(all_data)

    # Clean up dataframe
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Clean and prepare data
    df = df.dropna()
    df.index = pd.to_datetime(df.index)

    # Calculate base features for HMM
    df = calculate_features(df)

    print(f"Fetched {len(df)} hourly candles")
    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features for HMM training and technical indicators.

    Features for HMM:
    - Returns: Log returns
    - Range: (High - Low) / Close (normalized range)
    - Volume Volatility: Rolling std of volume changes

    Technical Indicators:
    - RSI, MACD, EMA50, EMA200, ADX, Momentum
    """
    df = df.copy()

    # Ensure we only have OHLCV columns to start
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ===== HMM Features =====
    # 1. Log Returns
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. Normalized Range (High-Low relative to Close)
    df['Range'] = (df['High'] - df['Low']) / df['Close']

    # 3. Volume Volatility (rolling std of volume pct change)
    df['Volume_Pct'] = df['Volume'].pct_change()
    df['Volume_Volatility'] = df['Volume_Pct'].rolling(window=24, min_periods=12).std()

    # ===== Technical Indicators =====
    # RSI (14 period)
    df['RSI'] = calculate_rsi(df['Close'], period=14)

    # Momentum (percent change over 24 periods)
    df['Momentum'] = df['Close'].pct_change(periods=24) * 100

    # Volatility (rolling std of returns, annualized for hourly)
    df['Volatility'] = df['Returns'].rolling(window=24, min_periods=12).std() * np.sqrt(24) * 100

    # Volume SMA (20 period)
    df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=10).mean()

    # EMA 50 and EMA 200
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ADX (Average Directional Index)
    df['ADX'] = calculate_adx(df, period=14)

    # Forward fill any remaining NaN in indicators (for stability)
    indicator_cols = ['RSI', 'ADX', 'Volume_Volatility', 'Volatility', 'Volume_SMA']
    for col in indicator_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Drop only the initial rows where we can't calculate anything
    # (first 200 rows needed for EMA200 warmup)
    df = df.iloc[200:]

    # Final cleanup - drop any remaining NaN rows
    df = df.dropna()

    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx


def get_cached_data(cache_file: str = "btc_data_cache.pkl") -> Optional[pd.DataFrame]:
    """Load cached data if available and fresh."""
    try:
        df = pd.read_pickle(cache_file)
        # Check if data is less than 1 hour old
        if df.index[-1] > datetime.now() - timedelta(hours=2):
            return df
    except:
        pass
    return None


def save_cache(df: pd.DataFrame, cache_file: str = "btc_data_cache.pkl"):
    """Save data to cache."""
    df.to_pickle(cache_file)


if __name__ == "__main__":
    # Test the data loader
    df = fetch_btc_data()
    print("\nData shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSample data:")
    print(df.tail())
    print("\nFeature statistics:")
    print(df[['Returns', 'Range', 'Volume_Volatility', 'RSI', 'ADX']].describe())
