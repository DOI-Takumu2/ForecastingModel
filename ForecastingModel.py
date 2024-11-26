import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

def forecast_sarimax(data, freq, future_periods):
    # 列名の前後の空白を削除
    data.columns = data.columns.str.strip()

    # 列名を英語に変換（必要に応じて）
    data.rename(columns={"日付": "Date", "値": "Value"}, inplace=True)
    
    # 必須列の確認
    required_columns = ['Date', 'Value']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Excel ファイルに以下の列名が必要です: {required_columns}")
    
    # データ前処理
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # 日付型に変換
    data.dropna(subset=['Date'], inplace=True)  # 欠損行を削除
    data.set_index('Date', inplace=True)  # 日付をインデックスに設定
    ts = data['Value']
    
    # リサンプリング
    ts = ts.resample(freq).mean().fillna(method='ffill')
    if ts.empty:
        raise ValueError("リサンプリング後のデータが空です。元データまたは頻度設定を確認してください。")
    
    # モデルの最適化
    model = auto_arima(ts, seasonal=True, m=12 if freq in ['M', 'ME'] else 1, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
    
    # SARIMAXのフィッティング
    sarimax_model = SARIMAX(ts, order=model.order, seasonal_order=model.seasonal_order)
    results = sarimax_model.fit(disp=False)
    
    # 未来予測
    forecast = results.get_forecast(steps=future_periods)
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # 結果をDataFrameで返却
    forecast_index = pd.date_range(start=ts.index[-1] + pd.offsets.DateOffset(1), periods=future_periods, freq=freq)
    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Forecast": forecast_values,
        "Lower CI": forecast_ci.iloc[:, 0],
        "Upper CI": forecast_ci.iloc[:, 1]
    })
    return forecast_df
