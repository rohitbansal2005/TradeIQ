def get_signal(df):
    """
    Generate BUY, SELL, or HOLD signal based on EMA and RSI.
    """
    # Using the last available row
    if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] and df['RSI'].iloc[-1] < 40:
        return "BUY"
    elif df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

def get_risk_params():
    """
    Returns simple risk logic parameters: Risk per trade, Stoploss, Target.
    """
    return {
        "Risk_Per_Trade": 0.01, # 1%
        "Stoploss": -0.02,      # 2%
        "Target": 0.04          # 4%
    }
