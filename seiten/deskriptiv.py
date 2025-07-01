# seiten/deskriptiv.py
import streamlit as st

def show(mobility_df):
    st.title("📊 Deskriptive Statistik")

    numeric_cols = ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]

    st.subheader("Numerische Grundauswertung")
    st.dataframe(mobility_df[numeric_cols].describe())

    st.subheader("Fehlende Werte je Kategorie")
    missing = mobility_df[numeric_cols].isnull().sum().to_frame(name="Fehlende Werte")
    st.dataframe(missing)

    st.subheader("Verteilung (Histogramm)")
    selected = st.selectbox("Wähle eine Spalte", numeric_cols)
    if selected:
        st.bar_chart(mobility_df[selected].dropna().value_counts().sort_index())

    st.subheader("Korrelation zwischen Bewegungsarten")
    st.dataframe(mobility_df[numeric_cols].corr())
