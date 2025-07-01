# seiten/deskriptiv.py
import streamlit as st
import pandas as pd

def show(mobility_df, wetter_df, standorte_df, df):
    st.title("📊 Deskriptive Statistik")

    section = st.selectbox("Datensatz auswählen", [
        "🚲 Mobility-Daten",
        "🌦 Wetterdaten",
        "📍 Standortdaten",
        "🔀 Kombination: Wetter & Bewegung"
    ])

    if section == "🚲 Mobility-Daten":
        st.subheader("Grundstatistik – Mobility")
        cols = ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]
        st.dataframe(mobility_df[cols].describe())

        st.subheader("Fehlende Werte")
        st.dataframe(mobility_df[cols].isnull().sum().to_frame("Fehlend"))

        st.subheader("Histogramm")
        selected = st.selectbox("Spalte wählen", cols)
        st.bar_chart(mobility_df[selected].dropna().value_counts().sort_index())

        st.subheader("Korrelationen")
        st.dataframe(mobility_df[cols].corr())

    elif section == "🌦 Wetterdaten":
        st.subheader("Grundstatistik – Wetter")
        numeric = wetter_df.select_dtypes(include="number").columns
        st.dataframe(wetter_df[numeric].describe())

        st.subheader("Fehlende Werte")
        st.dataframe(wetter_df[numeric].isnull().sum().to_frame("Fehlend"))

        st.subheader("Histogramm")
        selected = st.selectbox("Wetterspalte wählen", list(numeric))
        st.bar_chart(wetter_df[selected].dropna().value_counts().sort_index())

        st.subheader("Korrelationen")
        st.dataframe(wetter_df[numeric].corr())

    elif section == "📍 Standortdaten":
        st.subheader("Standortübersicht")
        st.dataframe(standorte_df.head(100))

        st.write("Anzahl Standorte:", len(standorte_df))

        if "geometry" in standorte_df.columns:
            st.map(standorte_df.rename(columns={"geometry": "location"}))  # wenn als POINT vorliegt

    elif section == "🔀 Kombination: Wetter & Bewegung":
        st.subheader("Korrelation Wetter vs. Mobilität")

        mobility_cols = ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]
        wetter_cols = [
            "temp", "humidity", "wind_speed", "clouds_all",
            "dew_point", "feels_like", "pressure", "visibility"
        ]

        kombi_df = df[mobility_cols + wetter_cols].dropna()
        st.write("Korrelation (z. B. Temperatur zu Anzahl Fussgänger)")
        st.dataframe(kombi_df.corr())

        st.subheader("Streudiagramm")
        x = st.selectbox("X-Achse (z. B. Wetter)", wetter_cols)
        y = st.selectbox("Y-Achse (z. B. Bewegung)", mobility_cols)

        st.write(f"Scatterplot: {x} vs. {y}")
        st.scatter_chart(kombi_df[[x, y]])
