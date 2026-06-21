import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from fast_indicators import get_fast_ema

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
        
    # EMAs via C++
    df['EMA20'] = get_fast_ema(close_col, window=20)
    df['EMA50'] = get_fast_ema(close_col, window=50)
    
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
    
    # --- New Ravi Mama Indicators ---
    # MFI (Money Flow Index)
    mfi = MFIIndicator(high=high_col, low=low_col, close=close_col, volume=volume_col, window=14)
    df['MFI'] = mfi.money_flow_index()
    
    # SMI (Stochastic Momentum Index)
    k_length = 10
    d_length = 3
    ll = low_col.rolling(window=k_length).min()
    hh = high_col.rolling(window=k_length).max()
    center = (hh + ll) / 2
    diff = close_col - center
    
    ema1 = diff.ewm(span=d_length, adjust=False).mean()
    ema2 = ema1.ewm(span=d_length, adjust=False).mean()
    
    hl_diff = hh - ll
    hl_ema1 = hl_diff.ewm(span=d_length, adjust=False).mean()
    hl_ema2 = hl_ema1.ewm(span=d_length, adjust=False).mean()
    
    df['SMI'] = 200 * (ema2 / hl_ema2)
    df['SMI_Signal'] = df['SMI'].ewm(span=d_length, adjust=False).mean()
    
    # Coil Pattern (Bollinger Band Width Squeeze)
    bb = BollingerBands(close=close_col, window=20, window_dev=2)
    df['BBW'] = bb.bollinger_wband()
    # Coil Squeeze is 1 if current BBW is the lowest in the last 20 days
    df['Coil_Squeeze'] = (df['BBW'] <= df['BBW'].rolling(window=20).min()).astype(int)
    
    return df


