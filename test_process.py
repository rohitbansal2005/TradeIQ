import pandas as pd
from data import fetch_multiple_stocks
from indicators import add_indicators
from ml_model import train_and_predict_ml
from strategy import get_regime_signal, calculate_dynamic_risk, get_option_suggestion

stock_data = fetch_multiple_stocks(['RELIANCE.NS'], period='2y')
df = stock_data['RELIANCE.NS']
market_context = 'Bullish'
ticker = 'RELIANCE.NS'

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
    
    regime = "Trending" if current_adx > 25 else "Sideways"
    
    option_strategy = get_option_suggestion(current_price, ml_latest_signal)
    
    print("Success:", option_strategy)
except Exception as e:
    print(f"Exception in process_stock for {ticker}: {e}")
    import traceback
    traceback.print_exc()
