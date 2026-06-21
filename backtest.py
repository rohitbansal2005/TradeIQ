from strategy import get_signal

def run_backtest(df, initial_balance=100000):
    """
    Simulate trades over the historical data.
    """
    balance = initial_balance
    
    # Needs at least 50 days of data to properly calculate EMA50
    if len(df) < 50:
        return balance
        
    for i in range(50, len(df)):
        # Provide data up to current point to simulate backtest properly
        current_data = df.iloc[:i]
        signal = get_signal(current_data)
        
        if signal == "BUY":
            balance *= 1.02   # mock profit of 2%
        elif signal == "SELL":
            balance *= 0.98   # mock loss of 2%
            
    return balance

if __name__ == "__main__":
    from data import fetch_data
    from indicators import add_indicators
    
    df = fetch_data("TCS.NS", period="1y", interval="1d")
    df = add_indicators(df)
    final_balance = run_backtest(df)
    print(f"Final Balance: {final_balance:.2f}")
