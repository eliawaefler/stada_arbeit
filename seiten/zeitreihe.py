# seiten/zeitreihe.py
import streamlit as st

def show(mobility_agg):
    st.title("⏳ Zeitreihenanalyse")

    variable = st.selectbox("Variable wählen", ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"])
    st.line_chart(mobility_agg.set_index("DATUM")[variable])

    st.subheader("Gleitender 24h Durchschnitt")
    sma_col = f"{variable}_SMA24"
    mobility_agg[sma_col] = mobility_agg[variable].rolling(window=24).mean()
    st.line_chart(mobility_agg.set_index("DATUM")[sma_col])

    st.subheader("Tägliche Summe")
    daily = mobility_agg.copy()
    daily["DATUM"] = daily["DATUM"].dt.date
    daily_sum = daily.groupby("DATUM")[variable].sum()
    st.line_chart(daily_sum)
