# seiten/zeitreihe.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def show(mobility_df, df):

    st.title("📈 Zeitreihenanalyse")

    st.subheader("Theorie")
    st.write("""**Was ist eine Zeitreihe?**  
    Eine Zeitreihe ist eine Folge von Messwerten, die in zeitlicher Reihenfolge erfasst wurden (z.B. stündliche Fuss- oder Veloverkehrsdaten).  
    Ziel ist es, Entwicklungen und Muster im Zeitverlauf zu erkennen.
    
    *Typische Fragestellungen:*
    - Gibt es Trends oder wiederkehrende Muster (z.B. Tages- oder Wochenschwankungen)?
    - Wie stark wirkt sich das Wetter auf die Mobilität aus?
    - Welche Extremwerte oder Ausreisser gibt es?
    
    *Methoden in dieser Analyse:*
    - Glättung mit gleitendem Durchschnitt (SMA)
    - Bollinger-Bänder zur Visualisierung von Volatilität
    - Vergleich mit Wettermerkmalen (z.B. Temperatur, Wind)
    - Aggregation nach Stunden, Tagen oder Wochen
    """)

    # Ziel- und Vergleichsvariablen wählen
    target_var = st.selectbox("Zielvariable (Kerzen & Linie)", ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"])
    compare_var = st.selectbox("Vergleichsvariable (Wetter)", ["temp", "humidity", "wind_speed", "clouds_all", "feels_like", "visibility"])



    # Zeitintervall wählen
    interval = st.selectbox("Intervall", ["H4", "D", "W"], index=0)
    rule_map = {"H4": "4H", "D": "D", "W": "W"}
    rule = rule_map[interval]

    # Durchschnitt über alle Standorte pro Stunde berechnen
    avg = mobility_df.copy()
    avg["DATUM"] = pd.to_datetime(avg["DATUM"])
    avg = avg.groupby("DATUM")[["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]].mean().reset_index()

    # Merge mit Wetterdaten
    df_plot = pd.merge(avg, df[["DATUM", compare_var]], on="DATUM", how="inner").dropna()
    df_plot = df_plot.set_index("DATUM")

    # OHLC-Resampling
    resampled = df_plot.resample(rule).agg({
        target_var: ["first", "max", "min", "last"],
        compare_var: "mean"
    })
    resampled.columns = ["open", "high", "low", "close", "compare"]
    resampled = resampled.dropna()


    # -------- Vergleichsplot (Ziel + Einflussvariable) ----------
    st.subheader(f"📉 Vergleich mit Wetterfaktor: {compare_var}")
    st.write("""
    Hier siehst du die Zielvariable (z.B. Fussgänger:innen)  
    zusammen mit einem Wetterfaktor (z.B. Temperatur) im selben Zeitformat.
    """)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=resampled.index, y=resampled["compare"],
        name=compare_var, line=dict(color="orange")
    ))
    fig2.add_trace(go.Scatter(
        x=resampled.index, y=resampled["close"],
        name=target_var, line=dict(color="royalblue")
    ))

    fig2.update_layout(
        height=300,
        template="plotly_white",
        title=f"{target_var} & {compare_var} im Zeitvergleich ({interval})",
        margin=dict(t=30, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -------- Interpretation --------
    st.subheader("📘 Interpretation")
    st.write(f"""
    - **Kerzenchart**: Gibt dir sofort ein Gefühl für die Dynamik von {target_var} im Tages- oder Wochenverlauf.  
    - **Bollinger-Bänder**: Wenn {target_var} ausserhalb der Bänder liegt, könnte es ein "besonderer" Zeitpunkt sein (z.B. Event, Wetterextrem).
    - **Linienvergleich**: Wenn sich {target_var} und {compare_var} synchron verhalten, kann ein Wettereffekt angenommen werden.
    """)

    st.subheader("Trading-View Stil")
    st.write("""
        Diese Ansicht zeigt geglättete Zeitreihen als Kerzencharts (OHLC) und Bollinger-Bänder.  
        Grundlage sind die **durchschnittlichen Bewegungswerte über alle Standorte je Stunde
        Leider haben die Daten eine extrem hohe Varianz was die Darstellung unschön macht**.
        """)

    # Bollinger-Bänder berechnen
    resampled["sma20"] = resampled["close"].rolling(20).mean()
    resampled["upper"] = resampled["sma20"] + 2 * resampled["close"].rolling(20).std()
    resampled["lower"] = resampled["sma20"] - 2 * resampled["close"].rolling(20).std()

    # -------- Candlestick Chart ----------
    st.subheader("📊 Kerzenchart mit Bollinger-Bändern")
    st.write("""
        für die Daten in dieser Arbeit war die Auswertung und darstellung mit Kerzenchart und Bollinger Bändern 
        nicht gut geeignet aber für sonstige Zeitreihen wie FX Charts können
        diese sehr nützlich sein.
        
        Jede Kerze zeigt 4/24/168 Stunden Bewegung:
        - Open: Beginnwert des Zeitraums  
        - High/Low: Max/Min im Zeitraum  
        - Close: Endwert  

        Die Bollinger-Bänder helfen, ungewöhnliche Ausschläge zu erkennen.
        """)

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
        template="plotly_dark",
        title=f"{target_var} – {interval}-Chart"
    )
    st.plotly_chart(fig, use_container_width=True)

