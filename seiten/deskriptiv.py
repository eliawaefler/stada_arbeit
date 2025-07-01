# seiten/deskriptiv.py
import streamlit as st
import pandas as pd


def highlight_corr(val):
    try:
        if val >= 0.75:
            return "background-color: lightgreen"
        elif val >= 0.5:
            return "background-color: turquoise"
        elif val >= 0.25:
            return "background-color: lightblue"
        elif val <= -0.75:
            return "background-color: red"
        elif val <= -0.5:
            return "background-color: orange"
        elif val <= -0.25:
            return "background-color: yellow"
        else:
            return ""
    except:
        return ""

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
        corr = mobility_df[cols].corr()
        st.dataframe(corr.style.applymap(highlight_corr).format("{:.2f}"))

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
        corr = wetter_df[numeric].corr()
        st.dataframe(corr.style.applymap(highlight_corr).format("{:.2f}"))
    elif section == "📍 Standortdaten":
        st.subheader("Standortübersicht")
        st.dataframe(standorte_df.head(100).style.applymap(highlight_corr).format("{:.2f}"))
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
        corr = kombi_df.corr()
        st.write("🔢 Farblich formatierte Korrelationsmatrix")

        st.dataframe(corr.style.applymap(highlight_corr).format("{:.2f}"))
        st.subheader("📈 Streudiagramm")
        x = st.selectbox("X-Achse (z. B. Wetter)", wetter_cols)
        y = st.selectbox("Y-Achse (z. B. Bewegung)", mobility_cols)
        st.write(f"Scatterplot: {x} vs. {y}")
        st.scatter_chart(kombi_df[[x, y]])