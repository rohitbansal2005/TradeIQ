import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def train_and_predict_ml(df, prob_threshold=0.60, live_only=False):
    """
    Trains an XGBoost model using expanding window walk-forward validation 
    to eliminate lookahead bias and provide authentic backtesting signals.
    """
    data = df.copy()
    
    # Advanced Quant Features
    required_features = ['Return_Z_Score', 'Velocity', 'Acceleration', 'OBV_Pct', 'Volatility', 'ADX', 'MFI', 'SMI', 'Coil_Squeeze']
    
    if isinstance(data.columns, pd.MultiIndex):
        close_col = data['Close'].iloc[:, 0]
    else:
        close_col = data['Close']
        
    # Target: 1 if next day's Close > today's Close, else 0
    data['Target'] = np.where(close_col.shift(-1) > close_col, 1, 0)
    
    # Drop NaNs
    data = data.dropna()
    
    if len(data) < 200:
        return pd.Series("HOLD", index=df.index), "HOLD", 0.0
        
    X = data[required_features]
    y = data['Target']
    
    signals = pd.Series("HOLD", index=df.index)
    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
    
    if not live_only:
        # Walk-forward validation (expanding window)
        start_idx = 100
        step = 20
        
        for i in range(start_idx, len(X), step):
            end_train = i
            end_predict = min(i + step, len(X))
            
            X_train = X.iloc[:end_train]
            y_train = y.iloc[:end_train]
            X_test = X.iloc[end_train:end_predict]
            
            if len(y_train.unique()) < 2:
                continue
                
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1] # Probability of Class 1 (BUY)
            
            test_dates = X.index[end_train:end_predict]
            for j, date in enumerate(test_dates):
                prob = probs[j]
                if prob > prob_threshold:
                    signals.loc[date] = "BUY"
                elif prob < (1 - prob_threshold):
                    signals.loc[date] = "SELL"
    
    # Live prediction (Latest)
    X_latest = df[required_features].ffill().bfill().iloc[-1:]
    
    # Train on all available data for the most up-to-date live prediction
    model.fit(X.iloc[:-1], y.iloc[:-1])
    latest_prob = model.predict_proba(X_latest)[0][1]
    
    if latest_prob > prob_threshold:
        latest_signal = "BUY"
    elif latest_prob < (1 - prob_threshold):
        latest_signal = "SELL"
    else:
        latest_signal = "HOLD"
        
    return signals, latest_signal, latest_prob

