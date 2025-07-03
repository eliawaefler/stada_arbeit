# seiten/pca.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def show(df):
    st.title("🧮 PCA – Hauptkomponentenanalyse")

    # Zeitliche Zusatzfeatures
    df = df.copy()
    df["weekday"] = df["DATUM"].dt.weekday
    df["hour"] = df["DATUM"].dt.hour

    # Variablenauswahl
    default_features = [
        "VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT",
        "temp", "humidity", "wind_speed", "clouds_all",
        "dew_point", "feels_like", "pressure", "visibility",
        "weekday", "hour"
    ]

    selected = st.multiselect("Wähle Variablen für PCA", default_features, default=default_features)

    if len(selected) < 2:
        st.warning("Bitte mindestens zwei Variablen auswählen.")
        return

    df_pca = df[selected].dropna()
    X_scaled = StandardScaler().fit_transform(df_pca)

    # PCA-Komponentenanzahl
    n_components = st.slider("Anzahl Hauptkomponenten", 2, min(len(selected), 10), value=2)

    # PCA durchführen
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(components, columns=pc_names)

    # -------------------
    st.subheader("📊 Scatterplot (erste 2 PCs)")
    st.write("Jeder Punkt entspricht einem Zeitpunkt (Stunde).")
    st.scatter_chart(pca_df[["PC1", "PC2"]])

    st.subheader("📈 Erklärte Varianz")
    for i, var in enumerate(explained_var):
        st.write(f"{pc_names[i]}: {var:.2%}")

    # -------------------
    st.subheader("📐 Einflussrichtungen (Loadings) auf PC1 & PC2")

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, feature in enumerate(selected):
        ax.arrow(0, 0,
                 pca.components_[0, i],
                 pca.components_[1, i],
                 head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        ax.text(pca.components_[0, i]*1.1,
                pca.components_[1, i]*1.1,
                feature,
                color='black', ha='center', va='center')

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA – Einfluss der Merkmale auf die ersten zwei PCs")
    ax.grid(True)
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    st.pyplot(fig)
