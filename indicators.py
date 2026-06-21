import pandas as pd
from ta.momentum import RSIIndicator

def add_indicators(df):
    """Add EMA 20, EMA 50, and RSI to the dataframe."""
    # Ensure Close is flat and not MultiIndex if we only downloaded one ticker
    if isinstance(df.columns, pd.MultiIndex):
        close_col = df['Close'].iloc[:, 0]
    else:
        close_col = df['Close']
        
    df['EMA20'] = close_col.ewm(span=20).mean()
    df['EMA50'] = close_col.ewm(span=50).mean()
    
    # RSI (simple logic use library)
    df['RSI'] = RSIIndicator(close_col, window=14).rsi()
    return df
