import streamlit as st
import pandas as pd
from ForecastingModel import forecast_sarimax

st.title("時系列予測アプリ")
st.write("アップロードしたデータを基にSARIMAXモデルを使って予測を行います。")

# ファイルアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])

if uploaded_file:
    try:
        # データの読み込み
        data = pd.read_excel(uploaded_file)
        st.write("アップロードされたデータ:")
        st.dataframe(data.head())  # データの先頭を表示

        # 頻度の選択
        freq = st.selectbox("時系列の頻度を選択してください", options=["ME", "D", "Y"], index=0)
        future_periods = st.slider("未来の予測期間（単位: 選択した頻度）", min_value=1, max_value=60, value=12)

        # 予測ボタン
        if st.button("予測を実行"):
            with st.spinner("予測中..."):
                # 予測関数の呼び出し
                forecast_df = forecast_sarimax(data, freq, future_periods)
                st.write("予測結果:")
                st.dataframe(forecast_df)

                # ダウンロードボタン
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="予測結果をダウンロード",
                    data=csv,
                    file_name="forecast_results.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
