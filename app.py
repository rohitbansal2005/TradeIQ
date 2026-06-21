import streamlit as st
import pandas as pd
from data import fetch_data, fetch_multiple_stocks, TOP_STOCKS, get_market_context, MARKET_UNIVERSES, get_nifty_500_live
from indicators import add_indicators
from strategy import get_signal, get_risk_params, get_ema_rsi_signal, get_macd_signal, calculate_position_size, calculate_dynamic_risk, get_regime_signal, get_coil_strategy_signal, get_option_suggestion
from backtest import run_backtest
from ml_model import train_and_predict_ml
import yfinance as yf

st.set_page_config(page_title="TradeIQ System", layout="wide")

st.title("TradeIQ System")

with st.expander("View Algorithm Logic"):
    st.markdown("""
    **True Alpha Institutional Algorithm**
    
    Our system evaluates stocks using a multi-layered quantitative approach:
    
    **1. Broader Market Context (Nifty 50)**
    - Verifies the overall trend of the market using EMA. If the market is Bearish, the system becomes defensive and holds back aggressive buys.
    
    **2. Regime Detection (ADX)**
    - Differentiates between 'Trending' and 'Sideways' markets. Applies Trend-Following (EMA) in trending markets and Mean-Reversion (RSI) in sideways markets.
    
    **3. Machine Learning (XGBoost)**
    - Generates a Probabilistic 'Win Rate' based on historical z-scores, velocity, acceleration, and OBV using Walk-Forward Validation. Focus on trades with >60% Win Probability.
    
    **4. Dynamic Risk & Kelly Sizing**
    - Calculates custom Target (3.0x) and Stoploss (1.5x) based on the stock's actual Volatility (ATR).
    - Allocates optimal capital per trade using the mathematical Kelly Criterion.
    """)

with st.sidebar:
    st.header("TradeIQ Philosophy")
    st.markdown("""
    **Our Core Edge:**
    - **Machine Learning Edge:** Trades are filtered by XGBoost Probability scores, eliminating standard indicator noise.
    - **Dynamic Risk Management:** Auto-calculates precise Stoploss and Target using the stock's daily ATR (Average True Range).
    - **Kelly Position Sizing:** Tells you exactly how many shares to buy to maximize compounding without ruining your portfolio.

    **Daily Trading Workflow:**
    
    **Step 1: The 3:15 PM Scan**
    At 3:15 PM (near market close), open the 'Multi-Stock Live Scanner' tab and click 'Scan Market'.
    
    **Step 2: Pick the Best Setup**
    Look at the results table. Focus on stocks that have an 'XGBoost Signal' of BUY and a high 'XGB Prob' (e.g. > 60%).
    
    **Step 3: Check Sizing & Execute**
    Go to 'Single Stock Analysis' for your chosen stock to see exactly how many shares to buy based on Kelly sizing. Place a CNC/Delivery order at the current market price.
    
    **Step 4: The GTT Safety Net**
    Once the buy order is complete, go to your broker portfolio and create an 'OCO' (One Cancels Other) GTT order.
    
    **Step 5: Set Strict Risk Parameters**
    Use the exact Stoploss % and Target % provided by the dashboard for that specific stock. Do not use generic 2% stoplosses. You are now protected mathematically.
    """)

tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "Multi-Stock Live Scanner", "Backtest & Strategy Compare"])

with tab1:
    st.header("Single Stock Analysis")
    
    universe_options = list(MARKET_UNIVERSES.keys()) + ["Nifty 500 (Official Full NSE)"]
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        selected_universe_t1 = st.selectbox("Select Market Universe", universe_options, index=0, key="univ_t1")
    
    current_list_t1 = get_nifty_500_live() if selected_universe_t1 == "Nifty 500 (Official Full NSE)" else MARKET_UNIVERSES[selected_universe_t1]
    
    with col_u2:
        stock_dropdown = st.selectbox("Select a Top Stock", current_list_t1, index=0)
        
    custom_stock = st.text_input("Or Enter Custom Ticker", placeholder="e.g. ZOMATO.NS")
        
    market_context = get_market_context()
    st.info(f"📈 **Broader Market Context (Nifty 50):** {market_context}")
    
    capital_input = st.number_input("Enter Capital for Position Sizing (₹)", value=100000, step=10000)
    stock = custom_stock.strip().upper() if custom_stock.strip() else stock_dropdown
    
    if st.button("Analyze Single"):
        with st.spinner("Fetching data and analyzing..."):
            try:
                df = fetch_data(stock, period="2y", interval="1d")
                
                if df.empty:
                    st.error("Could not fetch data for this ticker. Please check the ticker symbol.")
                else:
                    df = add_indicators(df)
                    
                    st.subheader("Recent Data")
                    st.dataframe(df.tail())
                    
                    signal = get_regime_signal(df, market_context).iloc[-1]
                    risk = get_risk_params()
                    
                    # Fetch XGBoost Prediction
                    _, ml_latest_signal, ml_prob = train_and_predict_ml(df, live_only=True)
                    
                    if isinstance(df.columns, pd.MultiIndex):
                        current_price = df['Close'].iloc[-1, 0]
                    else:
                        current_price = df['Close'].iloc[-1]
                        
                    current_atr = df['ATR'].iloc[-1]
                    current_adx = df['ADX'].iloc[-1]
                    
                    stoploss_price, target_price, sl_pct, target_pct = calculate_dynamic_risk(current_price, current_atr)
                    shares_to_buy, adjusted_risk = calculate_position_size(capital_input, current_price, current_atr, ml_probability=ml_prob)
                    
                    regime = "Trending 📈" if current_adx > 25 else "Sideways ↕️"
                    option_strategy = get_option_suggestion(current_price, ml_latest_signal)
                    
                    st.markdown("---")
                    st.subheader("🏆 INSTITUTIONAL OUTPUT")
                    
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    col_r1.metric("Stock Regime", regime)
                    col_r2.metric("ATR Volatility", f"₹{current_atr:.2f}")
                    col_r3.metric("XGBoost Win Prob", f"{ml_prob*100:.1f}%")
                    col_r4.metric("Option Suggestion", option_strategy, help="💡 System calculates the nearest Strike Price automatically. Note: Check your broker for the live option premium price.")
                    
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
    universe_options_t2 = list(MARKET_UNIVERSES.keys()) + ["Nifty 500 (Official Full NSE)"]
    selected_universe_t2 = st.selectbox("Select Market Universe to Scan", universe_options_t2, index=0, key="univ_t2")
    
    if st.button("Scan Market 🚀"):
        with st.spinner("Fetching live data for all selected stocks... this may take a moment."):
            try:
                current_list_t2 = get_nifty_500_live() if selected_universe_t2 == "Nifty 500 (Official Full NSE)" else MARKET_UNIVERSES[selected_universe_t2]
                # Fetch multi-stocks
                stock_data = fetch_multiple_stocks(current_list_t2, period="2y", interval="1d")
                
                market_context = get_market_context()
                results = []
                
                import concurrent.futures
                
                def process_stock(ticker, df, market_context):
                    if len(df) <= 50:
                        return None
                    try:
                        df = add_indicators(df)
                        signal = get_regime_signal(df, market_context).iloc[-1]
                        _, ml_latest_signal, ml_prob = train_and_predict_ml(df, live_only=True)
                        
                        current_atr = df['ATR'].iloc[-1]
                        current_adx = df['ADX'].iloc[-1]
                        
                        if isinstance(df.columns, pd.MultiIndex):
                            current_price = df['Close'].iloc[-1, 0]
                        else:
                            current_price = df['Close'].iloc[-1]
                            
                        _, _, sl_pct, target_pct = calculate_dynamic_risk(current_price, current_atr)
                        
                        option_strategy = get_option_suggestion(current_price, ml_latest_signal)
                        
                        regime = "Trending 📈" if current_adx > 25 else "Sideways ↕️"
                        
                        return {
                            "Stock": ticker.replace('.NS', ''),
                            "Regime": regime,
                            "Regime Signal": signal,
                            "XGBoost Signal": ml_latest_signal,
                            "XGB Prob": f"{ml_prob*100:.1f}%",
                            "Option Strategy": option_strategy,
                            "Stoploss": f"-{sl_pct:.2f}%",
                            "Target": f"+{target_pct:.2f}%"
                        }
                    except Exception as e:
                        print(f"Exception in process_stock for {ticker}: {e}")
                        import traceback
                        traceback.print_exc()
                        return None
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    futures = [executor.submit(process_stock, ticker, df, market_context) for ticker, df in stock_data.items()]
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res is not None:
                            results.append(res)
                
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
                    
                    st.info("ℹ️ **Note on Option Strategy:** The suggested Strike Prices (e.g., 3500 CE) are mathematically calculated based on the live stock price. Please search for these exact contracts in your broker app to see their live premium cost.")
                    st.success("Scan complete! Look for the green BUY signals or red SELL signals.")
                else:
                    st.warning("Could not fetch enough data for the stocks.")
                    
            except Exception as e:
                st.error(f"An error occurred during scanning: {e}")

with tab3:
    st.header("Backtest & Strategy Compare")
    st.write("Test multiple strategies over historical data to see their mathematical edge.")
    
    universe_options_bt = list(MARKET_UNIVERSES.keys()) + ["Nifty 500 (Official Full NSE)"]
    
    col_b0, col_b1, col_b2 = st.columns(3)
    with col_b0:
        selected_universe_bt = st.selectbox("Market Universe", universe_options_bt, index=0, key="univ_bt")
        
    current_list_bt = get_nifty_500_live() if selected_universe_bt == "Nifty 500 (Official Full NSE)" else MARKET_UNIVERSES[selected_universe_bt]
    
    with col_b1:
        bt_stock = st.selectbox("Select Stock to Backtest", current_list_bt, index=0, key="bt_stock")
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
                    coil_signals = get_coil_strategy_signal(df_bt, market_ctx_bt)
                    ml_signals, _, _ = train_and_predict_ml(df_bt)
                    
                    # Run Backtests
                    regime_metrics, _, regime_equity = run_backtest(df_bt, regime_signals)
                    ema_metrics, _, ema_equity = run_backtest(df_bt, ema_signals)
                    macd_metrics, _, macd_equity = run_backtest(df_bt, macd_signals)
                    coil_metrics, _, coil_equity = run_backtest(df_bt, coil_signals)
                    ml_metrics, _, ml_equity = run_backtest(df_bt, ml_signals)
                    
                    st.subheader("Equity Curve (Portfolio Balance)")
                    if not regime_equity.empty:
                        st.line_chart(regime_equity.set_index("Date")["Equity"])
                        min_date = regime_equity["Date"].min().strftime('%Y-%m-%d')
                        max_date = regime_equity["Date"].max().strftime('%Y-%m-%d')
                        st.caption(f"Chart Date Range: {min_date} to {max_date}")
                    
                    # Display Comparison
                    st.subheader("Strategy Comparison Metrics")
                    
                    comp_data = {
                        "Strategy": ["Regime+Context (Pro)", "EMA + RSI (Trend)", "MACD (Momentum)", "Coil+SMI+MFI (Mama)", "Machine Learning"],
                        "Total Trades": [regime_metrics["Total Trades"], ema_metrics["Total Trades"], macd_metrics["Total Trades"], coil_metrics["Total Trades"], ml_metrics["Total Trades"]],
                        "Win Rate (%)": [f"{regime_metrics['Win Rate (%)']:.2f}%", f"{ema_metrics['Win Rate (%)']:.2f}%", f"{macd_metrics['Win Rate (%)']:.2f}%", f"{coil_metrics['Win Rate (%)']:.2f}%", f"{ml_metrics['Win Rate (%)']:.2f}%"],
                        "Profit Factor": [f"{regime_metrics['Profit Factor']:.2f}", f"{ema_metrics['Profit Factor']:.2f}", f"{macd_metrics['Profit Factor']:.2f}", f"{coil_metrics['Profit Factor']:.2f}", f"{ml_metrics['Profit Factor']:.2f}"],
                        "Sharpe Ratio": [f"{regime_metrics['Sharpe Ratio']:.2f}", f"{ema_metrics['Sharpe Ratio']:.2f}", f"{macd_metrics['Sharpe Ratio']:.2f}", f"{coil_metrics['Sharpe Ratio']:.2f}", f"{ml_metrics['Sharpe Ratio']:.2f}"],
                        "Total Return (%)": [f"{regime_metrics['Total Return (%)']:.2f}%", f"{ema_metrics['Total Return (%)']:.2f}%", f"{macd_metrics['Total Return (%)']:.2f}%", f"{coil_metrics['Total Return (%)']:.2f}%", f"{ml_metrics['Total Return (%)']:.2f}%"],
                        "Max Drawdown (%)": [f"{regime_metrics['Max Drawdown (%)']:.2f}%", f"{ema_metrics['Max Drawdown (%)']:.2f}%", f"{macd_metrics['Max Drawdown (%)']:.2f}%", f"{coil_metrics['Max Drawdown (%)']:.2f}%", f"{ml_metrics['Max Drawdown (%)']:.2f}%"]
                    }
                    
                    comp_df = pd.DataFrame(comp_data)
                    st.dataframe(comp_df, use_container_width=True)
                    
                    st.info("Note: The Machine Learning strategy uses Walk-Forward Validation. It trains on expanding windows of historical data, eliminating look-ahead bias to simulate realistic trading performance.")
            except Exception as e:
                st.error(f"An error occurred during backtesting: {e}")
