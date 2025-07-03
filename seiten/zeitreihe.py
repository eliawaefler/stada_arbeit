# seiten/zeitreihe.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def show(mobility_agg, df):
    st.title("📈 Zeitreihenanalyse – TradingView Style")

    # Auswahl Bewegungsvariable & Einflussvariable
    target_var = st.selectbox("Zielvariable (y-Achse)", ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"])
    compare_var = st.selectbox("Einflussvariable zum Vergleich", ["temp", "humidity", "wind_speed", "clouds_all", "feels_like", "visibility"])

    # Zeitintervall wählen
    interval = st.selectbox("Intervall", ["H", "4H", "D", "W"], index=0)

    # Kopie zur Verarbeitung
    df_plot = df[["DATUM", target_var, compare_var]].copy()
    df_plot = df_plot.dropna()
    df_plot = df_plot.set_index("DATUM")

    # Resampling
    rule = {"H": "H", "4H": "4H", "D": "D", "W": "W"}[interval]
    resampled = df_plot.resample(rule).agg({
        target_var: ["first", "max", "min", "last"],
        compare_var: "mean"
    })

    resampled.columns = ["open", "high", "low", "close", "compare"]
    resampled = resampled.dropna()

    # Bollinger-Bänder berechnen
    resampled["sma20"] = resampled["close"].rolling(20).mean()
    resampled["upper"] = resampled["sma20"] + 2 * resampled["close"].rolling(20).std()
    resampled["lower"] = resampled["sma20"] - 2 * resampled["close"].rolling(20).std()

    st.subheader("📊 Candlestick Chart mit Bollinger Bändern")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=resampled.index,
        open=resampled["open"],
        high=resampled["high"],
        low=resampled["low"],
        close=resampled["close"],
        name=target_var
    ))

    fig.add_trace(go.Scatter(
        x=resampled.index, y=resampled["sma20"],
        mode="lines", line=dict(color="blue", width=1), name="SMA 20"
    ))

    fig.add_trace(go.Scatter(
        x=resampled.index, y=resampled["upper"],
        mode="lines", line=dict(color="lightgrey", width=1), name="Upper BB", opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=resampled.index, y=resampled["lower"],
        mode="lines", line=dict(color="lightgrey", width=1), name="Lower BB", opacity=0.5
    ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        title=f"{target_var} – Intervall: {interval}"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Vergleich mit Einflussvariable
    st.subheader(f"📉 Vergleich mit Einflussgröße: {compare_var}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=resampled.index, y=resampled["compare"],
                              name=compare_var, line=dict(color="orange")))

    fig2.update_layout(
        height=300,
        template="plotly_white",
        title=f"{compare_var} über Zeit ({interval})",
        margin=dict(t=30, b=20)
    )

    st.plotly_chart(fig2, use_container_width=True)
