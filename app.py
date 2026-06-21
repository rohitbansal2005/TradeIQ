import streamlit as st
import pandas as pd
from data import fetch_data, fetch_multiple_stocks, TOP_STOCKS, get_market_context
from indicators import add_indicators
from strategy import get_signal, get_risk_params, get_ema_rsi_signal, get_macd_signal, calculate_position_size, calculate_dynamic_risk, get_regime_signal
from backtest import run_backtest
from ml_model import train_and_predict_ml
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

tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "Multi-Stock Live Scanner", "Backtest & Strategy Compare"])

with tab1:
    st.header("Single Stock Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        stock_dropdown = st.selectbox("Select a Top Stock", TOP_STOCKS, index=1)
    with col2:
        custom_stock = st.text_input("Or Enter Custom Ticker", placeholder="e.g. ZOMATO.NS")
        
    market_context = get_market_context()
    st.info(f"📈 **Broader Market Context (Nifty 50):** {market_context}")
    
    capital_input = st.number_input("Enter Capital for Position Sizing (₹)", value=100000, step=10000)
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
                    
                    signal = get_regime_signal(df, market_context).iloc[-1]
                    risk = get_risk_params()
                    
                    # Fetch XGBoost Prediction
                    _, ml_latest_signal, ml_prob = train_and_predict_ml(df)
                    
                    if isinstance(df.columns, pd.MultiIndex):
                        current_price = df['Close'].iloc[-1, 0]
                    else:
                        current_price = df['Close'].iloc[-1]
                        
                    current_atr = df['ATR'].iloc[-1]
                    current_adx = df['ADX'].iloc[-1]
                    
                    stoploss_price, target_price, sl_pct, target_pct = calculate_dynamic_risk(current_price, current_atr)
                    shares_to_buy, adjusted_risk = calculate_position_size(capital_input, current_price, current_atr, ml_probability=ml_prob)
                    
                    regime = "Trending 📈" if current_adx > 25 else "Sideways ↕️"
                    
                    st.markdown("---")
                    st.subheader("🏆 INSTITUTIONAL OUTPUT")
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("Stock Regime", regime)
                    col_r2.metric("ATR Volatility", f"₹{current_atr:.2f}")
                    col_r3.metric("XGBoost Win Prob", f"{ml_prob*100:.1f}%")
                    
                    st.markdown("---")
                    
                    if signal == "BUY":
                        st.success(f"**Regime Signal:** {signal} ✅ | **XGBoost Signal:** {ml_latest_signal}")
                    elif signal == "SELL":
                        st.error(f"**Regime Signal:** {signal} ❌ | **XGBoost Signal:** {ml_latest_signal}")
                    else:
                        st.warning(f"**Regime Signal:** {signal} ⏸️ | **XGBoost Signal:** {ml_latest_signal}")
                        
                    st.markdown(f"**Stoploss (1.5x ATR):** ₹{stoploss_price:.2f} (-{sl_pct:.2f}%)")
                    st.markdown(f"**Target (3.0x ATR):** ₹{target_price:.2f} (+{target_pct:.2f}%)")
                    st.markdown(f"**Suggested Quantity (Kelly Sizing):** {shares_to_buy} shares (Risking {adjusted_risk*100:.2f}% of Capital)")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

with tab2:
    st.header("Multi-Stock Live Scanner (Nifty 50)")
    st.write("Scan the top Indian stocks instantly to find live trading opportunities.")
    
    if st.button("Scan Market 🚀"):
        with st.spinner("Fetching live data for all top stocks... this may take a moment."):
            try:
                # Fetch multi-stocks
                stock_data = fetch_multiple_stocks(TOP_STOCKS, period="2y", interval="1d")
                
                market_context = get_market_context()
                results = []
                
                for ticker, df in stock_data.items():
                    if len(df) > 50: # Need enough data for EMA50
                        try:
                            df = add_indicators(df)
                            signal = get_regime_signal(df, market_context).iloc[-1]
                            _, ml_latest_signal, ml_prob = train_and_predict_ml(df)
                            
                            current_atr = df['ATR'].iloc[-1]
                            current_adx = df['ADX'].iloc[-1]
                            
                            if isinstance(df.columns, pd.MultiIndex):
                                current_price = df['Close'].iloc[-1, 0]
                            else:
                                current_price = df['Close'].iloc[-1]
                                
                            _, _, sl_pct, target_pct = calculate_dynamic_risk(current_price, current_atr)
                            
                            regime = "Trending" if current_adx > 25 else "Sideways"
                            
                            results.append({
                                "Stock": ticker.replace('.NS', ''),
                                "Regime": regime,
                                "Regime Signal": signal,
                                "XGBoost Signal": ml_latest_signal,
                                "XGB Prob": f"{ml_prob*100:.1f}%",
                                "Stoploss": f"-{sl_pct:.2f}%",
                                "Target": f"+{target_pct:.2f}%"
                            })
                        except Exception:
                            continue
                
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
                    styled_df = results_df.style.map(highlight_signals, subset=['Regime Signal', 'XGBoost Signal'])
                    
                    st.dataframe(styled_df, width='stretch', height=600)
                    
                    st.success("Scan complete! Look for the green BUY signals or red SELL signals.")
                else:
                    st.warning("Could not fetch enough data for the stocks.")
                    
            except Exception as e:
                st.error(f"An error occurred during scanning: {e}")

with tab3:
    st.header("Backtest & Strategy Compare")
    st.write("Test multiple strategies over historical data to see their mathematical edge.")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        bt_stock = st.selectbox("Select Stock to Backtest", TOP_STOCKS, index=1, key="bt_stock")
    with col_b2:
        bt_period = st.selectbox("Historical Period", ["1y", "2y", "5y", "max"], index=1)
        
    if st.button("Run Backtest 🧪"):
        with st.spinner(f"Fetching {bt_period} data and running backtests..."):
            try:
                df_bt = fetch_data(bt_stock, period=bt_period, interval="1d")
                
                if df_bt.empty:
                    st.error("No data found.")
                else:
                    df_bt = add_indicators(df_bt)
                    
                    # Get Signals
                    market_ctx_bt = get_market_context()
                    regime_signals = get_regime_signal(df_bt, market_ctx_bt)
                    ema_signals = get_ema_rsi_signal(df_bt)
                    macd_signals = get_macd_signal(df_bt)
                    ml_signals, _, _ = train_and_predict_ml(df_bt)
                    
                    # Run Backtests
                    regime_metrics, _, regime_equity = run_backtest(df_bt, regime_signals)
                    ema_metrics, _, ema_equity = run_backtest(df_bt, ema_signals)
                    macd_metrics, _, macd_equity = run_backtest(df_bt, macd_signals)
                    ml_metrics, _, ml_equity = run_backtest(df_bt, ml_signals)
                    
                    st.subheader("Equity Curve (Portfolio Balance)")
                    if not regime_equity.empty:
                        st.line_chart(regime_equity.set_index("Date")["Equity"])
                    
                    # Display Comparison
                    st.subheader("Strategy Comparison Metrics")
                    
                    comp_data = {
                        "Strategy": ["Regime+Context (Pro)", "EMA + RSI (Trend)", "MACD (Momentum)", "Machine Learning"],
                        "Total Trades": [regime_metrics["Total Trades"], ema_metrics["Total Trades"], macd_metrics["Total Trades"], ml_metrics["Total Trades"]],
                        "Win Rate (%)": [f"{regime_metrics['Win Rate (%)']:.2f}%", f"{ema_metrics['Win Rate (%)']:.2f}%", f"{macd_metrics['Win Rate (%)']:.2f}%", f"{ml_metrics['Win Rate (%)']:.2f}%"],
                        "Profit Factor": [f"{regime_metrics['Profit Factor']:.2f}", f"{ema_metrics['Profit Factor']:.2f}", f"{macd_metrics['Profit Factor']:.2f}", f"{ml_metrics['Profit Factor']:.2f}"],
                        "Sharpe Ratio": [f"{regime_metrics['Sharpe Ratio']:.2f}", f"{ema_metrics['Sharpe Ratio']:.2f}", f"{macd_metrics['Sharpe Ratio']:.2f}", f"{ml_metrics['Sharpe Ratio']:.2f}"],
                        "Total Return (%)": [f"{regime_metrics['Total Return (%)']:.2f}%", f"{ema_metrics['Total Return (%)']:.2f}%", f"{macd_metrics['Total Return (%)']:.2f}%", f"{ml_metrics['Total Return (%)']:.2f}%"],
                        "Max Drawdown (%)": [f"{regime_metrics['Max Drawdown (%)']:.2f}%", f"{ema_metrics['Max Drawdown (%)']:.2f}%", f"{macd_metrics['Max Drawdown (%)']:.2f}%", f"{ml_metrics['Max Drawdown (%)']:.2f}%"]
                    }
                    
                    comp_df = pd.DataFrame(comp_data)
                    st.dataframe(comp_df, use_container_width=True)
                    
                    st.info("Note: The Machine Learning strategy includes look-ahead bias in this simple demonstration because it trains on the whole dataset minus the last day. Real trading systems use walk-forward validation.")
            except Exception as e:
                st.error(f"An error occurred during backtesting: {e}")
