import pandas as pd

def get_ema_rsi_signal(df):
    signals = pd.Series("HOLD", index=df.index)
    buy_condition = (df['EMA20'] > df['EMA50']) & (df['RSI'] < 40)
    sell_condition = (df['EMA20'] < df['EMA50'])
    signals.loc[buy_condition] = "BUY"
    signals.loc[sell_condition] = "SELL"
    return signals

def get_macd_signal(df):
    signals = pd.Series("HOLD", index=df.index)
    buy_condition = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    sell_condition = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    signals.loc[buy_condition] = "BUY"
    signals.loc[sell_condition] = "SELL"
    return signals

def get_regime_signal(df, market_context="Neutral"):
    """
    Advanced strategy using Regime Detection (ADX).
    If ADX > 25 (Trending): Use EMA Crossover
    If ADX <= 25 (Sideways): Use RSI Mean Reversion
    """
    signals = pd.Series("HOLD", index=df.index)
    
    ema_buy = (df['EMA20'] > df['EMA50']) & (df['EMA20'].shift(1) <= df['EMA50'].shift(1))
    ema_sell = (df['EMA20'] < df['EMA50']) & (df['EMA20'].shift(1) >= df['EMA50'].shift(1))
    
    rsi_buy = df['RSI'] < 30
    rsi_sell = df['RSI'] > 70
    
    trending_mask = df['ADX'] > 25
    sideways_mask = df['ADX'] <= 25
    
    signals.loc[trending_mask & ema_buy] = "BUY"
    signals.loc[trending_mask & ema_sell] = "SELL"
    
    signals.loc[sideways_mask & rsi_buy] = "BUY"
    signals.loc[sideways_mask & rsi_sell] = "SELL"
    
    if market_context == "Bearish":
        signals.loc[signals == "BUY"] = "HOLD"
        
    return signals

def get_coil_strategy_signal(df, market_context="Neutral"):
    """
    Coil Pattern Strategy with MFI and SMI.
    Targeting 70% efficiency when combined with XGBoost.
    """
    signals = pd.Series("HOLD", index=df.index)
    
    coil_active = df['Coil_Squeeze'] == 1
    mfi_bullish = (df['MFI'] > 40) & (df['MFI'] > df['MFI'].shift(1))
    smi_bullish = (df['SMI'] > df['SMI_Signal']) & (df['SMI'] > df['SMI'].shift(1))
    
    buy_condition = coil_active & mfi_bullish & smi_bullish
    
    # Sell when momentum dies
    sell_condition = (df['SMI'] < df['SMI_Signal']) & (df['MFI'] < 50)
    
    signals.loc[buy_condition] = "BUY"
    signals.loc[sell_condition] = "SELL"
    
    if market_context == "Bearish":
        signals.loc[signals == "BUY"] = "HOLD"
        
    return signals

def get_signal(df):
    return get_regime_signal(df).iloc[-1]

def get_risk_params():
    return {
        "Risk_Per_Trade": 0.01,
        "ATR_Stoploss_Multiplier": 1.5,
        "ATR_Target_Multiplier": 3.0
    }

def calculate_dynamic_risk(current_price, current_atr):
    risk_params = get_risk_params()
    sl_distance = current_atr * risk_params['ATR_Stoploss_Multiplier']
    target_distance = current_atr * risk_params['ATR_Target_Multiplier']
    
    stoploss_price = current_price - sl_distance
    target_price = current_price + target_distance
    
    sl_pct = (sl_distance / current_price) * 100
    target_pct = (target_distance / current_price) * 100
    
    return stoploss_price, target_price, sl_pct, target_pct

def calculate_position_size(capital, current_price, current_atr, ml_probability=None):
    risk_params = get_risk_params()
    base_risk = risk_params['Risk_Per_Trade']
    
    if ml_probability is not None and ml_probability > 0:
        win_prob = ml_probability
        reward_risk_ratio = risk_params['ATR_Target_Multiplier'] / risk_params['ATR_Stoploss_Multiplier']
        # Kelly Formula
        kelly_fraction = win_prob - ((1 - win_prob) / reward_risk_ratio)
        # Half-Kelly for safety
        kelly_fraction = max(0, kelly_fraction * 0.5)
        # Cap risk at 3%
        adjusted_risk = min(kelly_fraction, 0.03)
    else:
        adjusted_risk = base_risk
        
    risk_amount = capital * adjusted_risk
    risk_per_share = current_atr * risk_params['ATR_Stoploss_Multiplier']
    
    if risk_per_share <= 0:
        return 0, adjusted_risk
        
    shares = int(risk_amount / risk_per_share)
    max_shares = int(capital / current_price)
    
    return min(shares, max_shares), adjusted_risk

def get_option_suggestion(current_price, signal):
    """
    Suggests the closest ATM option strike and type based on the signal.
    """
    if signal == "HOLD":
        return "No Trade"
        
    # Determine Strike Step roughly based on Indian Market Price Levels
    if current_price < 250:
        step = 2.5
    elif current_price < 1000:
        step = 5
    elif current_price < 3000:
        step = 10
    elif current_price < 10000:
        step = 50
    else:
        step = 100
        
    strike = round(current_price / step) * step
    
    if signal == "BUY":
        return f"Buy {int(strike) if strike == int(strike) else strike} CE"
    elif signal == "SELL":
        return f"Buy {int(strike) if strike == int(strike) else strike} PE"
    
    return "No Trade"
