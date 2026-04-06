import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def train_and_predict(entire_features, returns_data, universe):
    # 1. Target (y) is tomorrow's return
    # If your 'returns_data' is already pct_change, just shift it
    y_wide = returns_data.shift(-1)
    
    # 2. Clean X: Handle Infinity and extreme Outliers
    # Replace Inf with NaN so we can drop them
    X_clean = entire_features.replace([np.inf, -np.inf], np.nan)
    
    # "Winsorize" / Clip outliers: 
    # This prevents one crazy data error from breaking the float64 limit
    # We clip at the 1st and 99th percentile
    lower = X_clean.quantile(0.01)
    upper = X_clean.quantile(0.99)
    X_clean = X_clean.clip(lower, upper, axis=1)

    # 3. Reshape to 'Long' Format
    # Assuming entire_features has a MultiIndex: [Feature, Ticker] or similar
    # We want rows = (Date, Ticker) and columns = Features
    X_stacked = X_clean.stack(level=-1) # Level depends on your DF structure
    y_stacked = y_wide.stack()
    
    # 4. Align and Drop NaNs
    # ML models cannot 'fit' if any value is NaN
    data = pd.concat([X_stacked, y_stacked.rename('target')], axis=1).dropna()
    X_train = data.drop(columns=['target'])
    y_train = data['target']

    # 5. SCALE THE DATA (Crucial for the float64 error)
    # This turns '10,000,000 Volume' into '1.5' (standard deviations)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 6. Train
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train) 
    
    # 7. Predict
    # Prepare current features for prediction
    X_current = X_clean.stack(level=-1).fillna(0)
    X_current_scaled = scaler.transform(X_current)
    
    preds = model.predict(X_current_scaled)
    
    # 8. Unstack back to Wide (Dates x Stocks)
    signal_matrix = pd.Series(preds, index=X_current.index).unstack()
    
    return signal_matrix.where(universe.astype(bool), np.nan)