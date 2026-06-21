import yfinance as yf
import pandas as pd

TOP_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "HCLTECH.NS", "TITAN.NS", "ULTRACEMCO.NS", "ZOMATO.NS",
    "NTPC.NS", "POWERGRID.NS", "M&M.NS", "NESTLEIND.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "BAJAJFINSV.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "HDFCLIFE.NS", "ONGC.NS", "HINDALCO.NS", "GRASIM.NS", "TATACONSUM.NS",
    "DRREDDY.NS", "WIPRO.NS", "APOLLOHOSP.NS", "CIPLA.NS", "TECHM.NS",
    "BRITANNIA.NS", "EICHERMOT.NS", "INDUSINDBK.NS", "BAJAJ-AUTO.NS", "SBILIFE.NS",
    "BPCL.NS", "MARICO.NS", "SHREECEM.NS", "HEROMOTOCO.NS", "DIVISLAB.NS"
]

def fetch_data(stock="TCS.NS", period="1y", interval="1d"):
    """Fetch historical stock data using yfinance."""
    df = yf.download(stock, period=period, interval=interval, progress=False)
    return df

def get_market_context(period="6mo", interval="1d"):
    """
    Fetches Nifty 50 (^NSEI) to determine the broader market trend.
    Returns: 'Bullish' if Close > EMA50, else 'Bearish'
    """
    try:
        nifty = yf.download("^NSEI", period=period, interval=interval, progress=False)
        if nifty.empty:
            return "Neutral"
            
        close_col = nifty['Close'].iloc[:, 0] if isinstance(nifty.columns, pd.MultiIndex) else nifty['Close']
        ema50 = close_col.ewm(span=50, adjust=False).mean()
        
        last_close = close_col.iloc[-1]
        last_ema = ema50.iloc[-1]
        
        return "Bullish" if last_close > last_ema else "Bearish"
    except Exception:
        return "Neutral"


def fetch_multiple_stocks(tickers, period="6mo", interval="1d"):
    """
    Fetch data for multiple stocks efficiently. 
    Returns a dictionary of ticker -> DataFrame.
    """
    # Grouping into a space-separated string for yfinance
    tickers_str = " ".join(tickers)
    # yfinance download with group_by="ticker" returns a MultiIndex DataFrame
    df = yf.download(tickers_str, period=period, interval=interval, group_by="ticker")
    
    stock_data = {}
    for ticker in tickers:
        # If multiple tickers are downloaded, df will have tickers as the top level of columns
        if isinstance(df.columns, pd.MultiIndex):
            # Extract data for this specific ticker
            # yfinance sometimes drops the top level if only one valid ticker was found, but here we expect multiple
            if ticker in df.columns.levels[0]:
                ticker_df = df[ticker].dropna()
                if not ticker_df.empty:
                    stock_data[ticker] = ticker_df
        else:
            # Fallback if somehow it's not a MultiIndex (e.g., only 1 ticker was passed)
            stock_data[ticker] = df.dropna()
            
    return stock_data

if __name__ == "__main__":
    df = fetch_data()
    print(df.head())
    
    print("Testing multiple fetch...")
    multi_df = fetch_multiple_stocks(TOP_STOCKS[:3], period="1mo")
    for t, d in multi_df.items():
        print(f"{t}: {len(d)} rows")
