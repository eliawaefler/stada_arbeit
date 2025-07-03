# seiten/start.py
import streamlit as st

def show(mobility_df, wetter_df, standorte_df):
    st.title("Statistische Auswertung – Zürich Mobility & Wetter")

    st.subheader("https://stadaarbeit.streamlit.app/")

    st.write("""
    Diese Webseite zeigt statistische Methoden auf Grundlage von Mobilitäts- und Wetterdaten aus Zürich.
    CAS "Statistische Datenanalyse und Datenvisualisierung" an der FFHS
    Elia Wäfler
    
    ChatGPT wurde als Hiflsmittel eingesetzt.
    """)

    st.subheader("🚲 Mobility-Daten (Auszug)")
    st.dataframe(mobility_df.head(100))

    st.subheader("🌦 Wetterdaten (Auszug)")
    st.dataframe(wetter_df.head(100))
