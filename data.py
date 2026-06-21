import yfinance as yf
import pandas as pd
import streamlit as st

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

NIFTY_100_EXTRA = [
    "HAL.NS", "BEL.NS", "TRENT.NS", "TVSMOTOR.NS", "INDIGO.NS", "CHOLAFIN.NS",
    "PNB.NS", "BANKBARODA.NS", "JINDALSTEL.NS", "CUMMINSIND.NS", "SIEMENS.NS",
    "ABB.NS", "ZFCVINDIA.NS", "BOSCHLTD.NS", "PIDILITIND.NS", "HAVELLS.NS",
    "POLYCAB.NS", "CGPOWER.NS", "LODHA.NS", "DLF.NS", "GODREJCP.NS", "DABUR.NS",
    "COLPAL.NS", "PGHH.NS", "UBL.NS", "MCDOWELL-N.NS", "GAIL.NS", "IOC.NS",
    "AMBUJACEM.NS", "SRF.NS", "PIIND.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS",
    "LUPIN.NS", "AUROPHARMA.NS", "MUTHOOTFIN.NS", "SHRIRAMFIN.NS", "HDFCAMC.NS",
    "ICICIPRULI.NS", "ICICIGI.NS", "SBICARD.NS", "RECLTD.NS", "PFC.NS",
    "TATACOMM.NS", "INFOEDGE.NS", "FSNCREVICES.NS", "PAYTM.NS", "DMART.NS",
    "MAXHEALTH.NS", "AUBANK.NS"
]

NIFTY_200_EXTRA = [
    "ASHOKLEY.NS", "SAIL.NS", "BANDHANBNK.NS", "L&TFH.NS", "NMDC.NS", "BATAINDIA.NS",
    "GMRINFRA.NS", "IDFCFIRSTB.NS", "M&MFIN.NS", "AARTIIND.NS", "ABCAPITAL.NS",
    "AMARAJABAT.NS", "APOLLOTYRE.NS", "BALKRISIND.NS", "BANKINDIA.NS", "CANBK.NS",
    "CHAMBLFERT.NS", "COROMANDEL.NS", "CUB.NS", "DEEPAKNTR.NS", "DIXON.NS",
    "ESCORTS.NS", "EXIDEIND.NS", "GLENMARK.NS", "GUJGASLTD.NS", "IGL.NS",
    "INDIACEM.NS", "INDHOTEL.NS", "IPCALAB.NS", "JKCEMENT.NS", "LALPATHLAB.NS",
    "LICHSGFIN.NS", "MGL.NS", "MFSL.NS", "NATIONALUM.NS", "NAVINFLUOR.NS",
    "OBEROIRLTY.NS", "PERSISTENT.NS", "PETRONET.NS", "RAMCOCEM.NS", "TATACHEM.NS",
    "TATAPOWER.NS", "TORNTPOWER.NS", "TVSMOTOR.NS", "UPL.NS", "VOLTAS.NS"
]

MARKET_UNIVERSES = {
    "Nifty 50": TOP_STOCKS,
    "Nifty 100": TOP_STOCKS + NIFTY_100_EXTRA,
    "Nifty 200": TOP_STOCKS + NIFTY_100_EXTRA + NIFTY_200_EXTRA,
    "Nifty 500 (Top Liquid)": TOP_STOCKS + NIFTY_100_EXTRA + NIFTY_200_EXTRA + ["IDEA.NS", "YESBANK.NS", "BHEL.NS", "RVNL.NS", "IRFC.NS", "IREDA.NS", "SUZLON.NS", "JIOFIN.NS"]
}

@st.cache_data(ttl=86400) # Cache for 1 day
def get_nifty_500_live():
    """Dynamically fetches the official Nifty 500 list from NSE."""
    try:
        df = pd.read_csv('https://archives.nseindia.com/content/indices/ind_nifty500list.csv')
        tickers = [str(sym) + ".NS" for sym in df['Symbol'].tolist()]
        return tickers
    except Exception as e:
        # Fallback to the heavy static list if offline
        return MARKET_UNIVERSES["Nifty 500 (Top Liquid)"]

@st.cache_data(ttl=900)
def fetch_data(stock="TCS.NS", period="1y", interval="1d"):
    """Fetch historical stock data using yfinance."""
    df = yf.download(stock, period=period, interval=interval, progress=False)
    return df

@st.cache_data(ttl=900)
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


@st.cache_data(ttl=900)
def fetch_multiple_stocks(tickers, period="6mo", interval="1d"):
    """
    Fetch data for multiple stocks efficiently. 
    Returns a dictionary of ticker -> DataFrame.
    """
    # Grouping into a space-separated string for yfinance
    tickers_str = " ".join(tickers)
    # yfinance download with group_by="ticker" returns a MultiIndex DataFrame
    df = yf.download(tickers_str, period=period, interval=interval, group_by="ticker", threads=True, progress=False)
    
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
