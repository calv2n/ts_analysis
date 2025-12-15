import xgboost as xgb
import numpy as np

class xgb_model:
    def __init__(self, p):
        self.md = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        self.p = p
    
    def train_model(self, train):
        X_train, y_train = self._create_lag_features(data=train, n_lags=self.p)
        self.md.fit(X_train, y_train)
    
    def _create_lag_features(self, data, n_lags):
        X = np.array([
            data[i - n_lags:i].values for i in range(n_lags, len(data))
        ])
        y = data[n_lags:]
        return X, y

    def forecast(self, train, n_forecast):
        """
        Rolling predictions: similar to how forecast works 
        in autoregressive model.
        """
        preds = np.zeros(n_forecast)
        last_values = train[-self.p:].values.copy()

        for i in range(n_forecast):
            X_pred = last_values.reshape(1, -1)
            
            y_pred = self.md.predict(X_pred)[0]
            preds[i] = y_pred
            
            last_values = np.append(last_values[1:], y_pred)
        
        return preds

    def predict(self, full_series, train_size):
        """
        Up-to-date approach: uses true y values before y_t 
        instead of rolling predictions.
        """
        X_test, _ = self._create_lag_features(full_series[train_size - self.p:], self.p)
        return self.md.predict(X_test)
