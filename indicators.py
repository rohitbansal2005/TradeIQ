import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def add_indicators(df):
    """Add EMA, RSI, MACD, ATR, ADX, and Advanced Statistical ML features."""
    if isinstance(df.columns, pd.MultiIndex):
        close_col = df['Close'].iloc[:, 0]
        high_col = df['High'].iloc[:, 0]
        low_col = df['Low'].iloc[:, 0]
        volume_col = df['Volume'].iloc[:, 0]
    else:
        close_col = df['Close']
        high_col = df['High']
        low_col = df['Low']
        volume_col = df['Volume']
        
    # EMAs
    df['EMA20'] = close_col.ewm(span=20, adjust=False).mean()
    df['EMA50'] = close_col.ewm(span=50, adjust=False).mean()
    
    # RSI
    df['RSI'] = RSIIndicator(close_col, window=14).rsi()
    
    # MACD
    macd = MACD(close=close_col, window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # ATR (Average True Range)
    atr = AverageTrueRange(high=high_col, low=low_col, close=close_col, window=14)
    df['ATR'] = atr.average_true_range()
    
    # ADX (Average Directional Index) for Regime Detection
    adx = ADXIndicator(high=high_col, low=low_col, close=close_col, window=14)
    df['ADX'] = adx.adx()
    
    # --- Advanced Quant Features ---
    # Daily Return
    df['Daily_Return'] = close_col.pct_change()
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=14).std()
    
    # Z-Score of Returns (Mean Reversion proxy)
    rolling_mean = df['Daily_Return'].rolling(window=20).mean()
    rolling_std = df['Daily_Return'].rolling(window=20).std()
    df['Return_Z_Score'] = (df['Daily_Return'] - rolling_mean) / rolling_std
    
    # Price Velocity (Rate of change of price)
    df['Velocity'] = close_col.diff(periods=5) / 5
    
    # Price Acceleration (Rate of change of velocity)
    df['Acceleration'] = df['Velocity'].diff(periods=5) / 5
    
    # On-Balance Volume (OBV)
    obv = OnBalanceVolumeIndicator(close=close_col, volume=volume_col)
    df['OBV'] = obv.on_balance_volume()
    # Normalize OBV for ML (percentage change)
    df['OBV_Pct'] = df['OBV'].pct_change()
    
    return df


