import pandas as pd
import numpy as np
from strategy import get_risk_params

def run_backtest(df, signals_series, initial_balance=100000):
    balance = initial_balance
    risk = get_risk_params()
    
    in_position = False
    buy_price = 0.0
    shares = 0
    current_atr = 0.0
    stoploss_price = 0.0
    target_price = 0.0
    
    trades = []
    equity_curve = []
    
    if isinstance(df.columns, pd.MultiIndex):
        close_prices = df['Close'].iloc[:, 0]
        high_prices = df['High'].iloc[:, 0]
        low_prices = df['Low'].iloc[:, 0]
        open_prices = df['Open'].iloc[:, 0]
    else:
        close_prices = df['Close']
        high_prices = df['High']
        low_prices = df['Low']
        open_prices = df['Open']
        
    atr_series = df['ATR']
    
    max_balance = initial_balance
    max_drawdown = 0.0
    
    slippage_pct = 0.001 # 0.1%
    brokerage_pct = 0.0005 # 0.05%
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        today_open = open_prices.iloc[i]
        today_high = high_prices.iloc[i]
        today_low = low_prices.iloc[i]
        today_close = close_prices.iloc[i]
        
        yesterday_signal = signals_series.iloc[i-1]
        
        if in_position:
            sell_price = 0
            reason = ""
            
            if today_low <= stoploss_price:
                sell_price = stoploss_price
                reason = "Stoploss"
            elif today_high >= target_price:
                sell_price = target_price
                reason = "Target"
            elif yesterday_signal == "SELL":
                sell_price = today_open
                reason = "Exit Signal"
                
            if sell_price > 0:
                sell_price = sell_price * (1 - slippage_pct)
                gross_value = sell_price * shares
                net_value = gross_value * (1 - brokerage_pct)
                
                profit = net_value - (buy_price * shares)
                balance += net_value
                in_position = False
                
                trades.append({
                    "Sell Date": current_date,
                    "Buy Price": buy_price,
                    "Sell Price": sell_price,
                    "Profit": profit,
                    "Reason": reason
                })
        else:
            if yesterday_signal == "BUY":
                buy_price_raw = today_open
                buy_price = buy_price_raw * (1 + slippage_pct)
                buy_price = buy_price * (1 + brokerage_pct)
                
                current_atr = atr_series.iloc[i-1]
                
                risk_amount = balance * risk['Risk_Per_Trade']
                sl_distance = current_atr * risk['ATR_Stoploss_Multiplier']
                
                if sl_distance > 0:
                    shares = int(risk_amount / sl_distance)
                    max_shares = int(balance / buy_price)
                    shares = min(shares, max_shares)
                    
                    if shares > 0:
                        in_position = True
                        balance -= (buy_price * shares)
                        stoploss_price = buy_price_raw - sl_distance
                        target_price = buy_price_raw + (current_atr * risk['ATR_Target_Multiplier'])
                        
        current_equity = balance + (today_close * shares if in_position else 0)
        equity_curve.append({"Date": current_date, "Equity": current_equity})
        
        max_balance = max(max_balance, current_equity)
        drawdown = (max_balance - current_equity) / max_balance
        max_drawdown = max(max_drawdown, drawdown)
        
    if in_position:
        final_close = close_prices.iloc[-1]
        sell_price = final_close * (1 - slippage_pct)
        net_value = (sell_price * shares) * (1 - brokerage_pct)
        profit = net_value - (buy_price * shares)
        balance += net_value
        trades.append({
            "Sell Date": df.index[-1],
            "Buy Price": buy_price,
            "Sell Price": sell_price,
            "Profit": profit,
            "Reason": "End of Data"
        })
        
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['Profit'] > 0]
    losing_trades = [t for t in trades if t['Profit'] <= 0]
    
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_return = ((balance - initial_balance) / initial_balance) * 100
    
    gross_profit = sum(t['Profit'] for t in winning_trades)
    gross_loss = abs(sum(t['Profit'] for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        returns_series = equity_df['Equity'].pct_change().dropna()
        if len(returns_series) > 0 and returns_series.std() != 0:
            daily_rf = 0.05 / 252
            sharpe_ratio = (returns_series.mean() - daily_rf) / returns_series.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0
    
    metrics = {
        "Total Trades": total_trades,
        "Win Rate (%)": win_rate,
        "Total Return (%)": total_return,
        "Max Drawdown (%)": max_drawdown * 100,
        "Profit Factor": profit_factor,
        "Sharpe Ratio": sharpe_ratio
    }
    
    return metrics, trades, equity_df
