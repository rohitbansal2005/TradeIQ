import streamlit as st
import pandas as pd
from data import fetch_data, fetch_multiple_stocks, TOP_STOCKS
from indicators import add_indicators
from strategy import get_signal, get_risk_params
import yfinance as yf

st.set_page_config(page_title="TradeIQ System", layout="wide")

st.title("TradeIQ System")

with st.expander("View Algorithm Logic"):
    st.markdown("""
    **Trend-Following and Momentum Algorithm**
    
    Our system evaluates stocks using a strict mathematical algorithm:
    
    **1. Trend Identification (Dual EMA)**
    - **EMA20 > EMA50:** Confirms a bullish uptrend.
    - **EMA20 < EMA50:** Confirms a bearish downtrend.
    
    **2. Momentum Confirmation (RSI)**
    - **RSI < 40:** Indicates the stock is mathematically 'oversold' and has room to rise before becoming too expensive.
    
    **3. Final Signal Rules**
    - **BUY:** Triggered ONLY if (EMA20 > EMA50) AND (RSI < 40).
    - **SELL:** Triggered if (EMA20 < EMA50), meaning the trend is broken.
    - **HOLD:** If neither condition is met, the system strictly advises waiting.
    """)

with st.sidebar:
    st.header("TradeIQ Philosophy")
    st.markdown("""
    **Our Core Edge:**
    - **Emotionless Trading:** Eliminates FOMO and panic. Trades are generated purely on mathematical signals (EMA + RSI).
    - **Time Efficiency:** Scans 50 top market stocks instantly, replacing manual chart checking.
    - **Risk Management:** Auto-calculates exact mathematical Stoploss (-2%) and Target (+4%).

    **Zerodha Daily Trading Workflow:**
    
    **Step 1: The 3:15 PM Scan**
    At 3:15 PM (near market close), open the 'Multi-Stock Live Scanner' tab and click 'Scan Market'.
    
    **Step 2: Pick the Best Setup**
    Look at the results table. Focus on stocks that have a 'BUY' signal and a 'High' confidence rating.
    
    **Step 3: Execute the Trade**
    Open your Zerodha Kite app. Search for the selected stock and place a BUY order (CNC for positional) at the current market price.
    
    **Step 4: The GTT Safety Net**
    Once the buy order is complete, go to your Zerodha portfolio and select 'Create GTT'. Choose the 'OCO' (One Cancels Other) option.
    
    **Step 5: Set Strict Risk Parameters**
    Use the exact percentages provided by TradeIQ for that stock. Set your Stoploss trigger at -2% and your Target trigger at +4%. You are now protected automatically.
    """)

tab1, tab2 = st.tabs(["Single Stock Analysis", "Multi-Stock Live Scanner"])

with tab1:
    st.header("Single Stock Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        stock_dropdown = st.selectbox("Select a Top Stock", TOP_STOCKS, index=1)
    with col2:
        custom_stock = st.text_input("Or Enter Custom Ticker", placeholder="e.g. ZOMATO.NS")
        
    stock = custom_stock.strip().upper() if custom_stock.strip() else stock_dropdown
    
    if st.button("Analyze Single"):
        with st.spinner("Fetching data and analyzing..."):
            try:
                df = fetch_data(stock, period="6mo", interval="1d")
                
                if df.empty:
                    st.error("Could not fetch data for this ticker. Please check the ticker symbol.")
                else:
                    df = add_indicators(df)
                    
                    st.subheader("Recent Data")
                    st.dataframe(df.tail())
                    
                    signal = get_signal(df)
                    risk = get_risk_params()
                    
                    current_rsi = df['RSI'].iloc[-1]
                    trend = "Bullish 📈" if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else "Bearish 📉"
                    confidence = "High" if signal == "BUY" and current_rsi < 30 else "Medium" if signal != "HOLD" else "Low"
                    
                    st.markdown("---")
                    st.subheader("🏆 FINAL OUTPUT")
                    
                    st.markdown(f"**Stock:** {stock.replace('.NS', '')}")
                    st.markdown(f"**Trend:** {trend}")
                    st.markdown(f"**RSI:** {current_rsi:.2f}")
                    st.markdown("---")
                    
                    if signal == "BUY":
                        st.success(f"**Signal:** {signal} ✅")
                    elif signal == "SELL":
                        st.error(f"**Signal:** {signal} ❌")
                    else:
                        st.warning(f"**Signal:** {signal} ⏸️")
                        
                    st.markdown(f"**Stoploss:** {risk['Stoploss']*100:.0f}%")
                    st.markdown(f"**Target:** +{risk['Target']*100:.0f}%")
                    st.markdown(f"**Confidence:** {confidence}")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

with tab2:
    st.header("Multi-Stock Live Scanner (Nifty 50)")
    st.write("Scan the top Indian stocks instantly to find live trading opportunities.")
    
    if st.button("Scan Market 🚀"):
        with st.spinner("Fetching live data for all top stocks... this may take a moment."):
            try:
                # Fetch multi-stocks
                stock_data = fetch_multiple_stocks(TOP_STOCKS, period="6mo", interval="1d")
                
                results = []
                risk = get_risk_params()
                
                for ticker, df in stock_data.items():
                    if len(df) > 50: # Need enough data for EMA50
                        df = add_indicators(df)
                        signal = get_signal(df)
                        
                        current_rsi = df['RSI'].iloc[-1]
                        trend = "Bullish" if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else "Bearish"
                        confidence = "High" if signal == "BUY" and current_rsi < 30 else "Medium" if signal != "HOLD" else "Low"
                        
                        results.append({
                            "Stock": ticker.replace('.NS', ''),
                            "Trend": trend,
                            "RSI": round(current_rsi, 2),
                            "Signal": signal,
                            "Confidence": confidence,
                            "Stoploss": f"{risk['Stoploss']*100:.0f}%",
                            "Target": f"+{risk['Target']*100:.0f}%"
                        })
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Function to color code signals
                    def highlight_signals(val):
                        if val == 'BUY':
                            return 'background-color: #d4edda; color: #155724; font-weight: bold'
                        elif val == 'SELL':
                            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                        elif val == 'HOLD':
                            return 'background-color: #fff3cd; color: #856404;'
                        return ''
                    
                    # Apply styling
                    styled_df = results_df.style.map(highlight_signals, subset=['Signal'])
                    
                    st.dataframe(styled_df, width='stretch', height=600)
                    
                    st.success("Scan complete! Look for the green BUY signals or red SELL signals.")
                else:
                    st.warning("Could not fetch enough data for the stocks.")
                    
            except Exception as e:
                st.error(f"An error occurred during scanning: {e}")
