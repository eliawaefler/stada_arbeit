# app.py
import streamlit as st
from load_data import load_all_data

# Seitenmodule importieren
from seiten import start, deskriptiv, mlr, pca, zeitreihe, clustering

# Daten laden
mobility_df, wetter_df, standorte_df, df, mobility_agg = load_all_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite auswählen", [
    "Start",
    "Deskriptive Statistik",
    "Multiple Lineare Regression (MLR)",
    "PCA",
    "Zeitreihenanalyse",
    "Clustering (K-Means)"
])

# Seiten-Dispatch
if page == "Start":
    start.show(mobility_df, wetter_df, standorte_df)
elif page == "Deskriptive Statistik":
    deskriptiv.show(mobility_df, wetter_df, standorte_df, df)
elif page == "Multiple Lineare Regression (MLR)":
    mlr.show(df)
elif page == "PCA":
    pca.show(df)
elif page == "Zeitreihenanalyse":
    zeitreihe.show(mobility_df, df)
elif page == "Clustering (K-Means)":
    clustering.show(df)
