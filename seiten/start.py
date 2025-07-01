# seiten/start.py
import streamlit as st

def show(mobility_df, wetter_df, standorte_df):
    st.title("Statistische Auswertung – Zürich Mobility & Wetter")
    st.write("""
    Willkommen!  
    Diese Anwendung zeigt statistische Methoden auf Grundlage von Mobilitäts- und Wetterdaten aus Zürich.
    """)

    st.subheader("🚲 Mobility-Daten (Auszug)")
    st.dataframe(mobility_df.head(100))

    st.subheader("🌦 Wetterdaten (Auszug)")
    st.dataframe(wetter_df.head(100))

    st.subheader("📍 Standorte der Zählstationen (Auszug)")
    st.dataframe(standorte_df.head(100))
