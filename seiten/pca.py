# seiten/pca.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def show(df):
    st.title("🧮 PCA – Hauptkomponentenanalyse")

    st.write("""
    Die Hauptkomponentenanalyse (PCA) reduziert viele Variablen auf wenige Komponenten,  
    die möglichst viel Varianz der ursprünglichen Daten erklären.  
    Ziel: Muster erkennen, Dimension reduzieren, Visualisierung verbessern.
    """)

    # Zeitliche Zusatzfeatures
    df = df.copy()
    df["weekday"] = df["DATUM"].dt.weekday
    df["hour"] = df["DATUM"].dt.hour

    default_features = [
        "VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT",
        "temp", "humidity", "wind_speed", "clouds_all",
        "dew_point", "feels_like", "pressure", "visibility",
        "weekday", "hour"
    ]

    st.subheader("📌 Variablenauswahl")
    selected = st.multiselect("Variablen für PCA", default_features, default=default_features)

    if len(selected) < 2:
        st.warning("Bitte mindestens zwei Variablen auswählen.")
        return

    df_pca = df[selected].dropna()
    X_scaled = StandardScaler().fit_transform(df_pca)

    st.subheader("⚙️ Anzahl Hauptkomponenten")
    n_components = st.slider("Anzahl Hauptkomponenten", 2, min(len(selected), 10), value=2)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(components, columns=pc_names)

    # -------------------
    st.subheader("🔵 2D PCA Scatterplot")

    if st.checkbox("K-Means-Clustering aktivieren"):
        k = st.slider("Anzahl Cluster", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_df[["PC1", "PC2"]])
        pca_df["Cluster"] = cluster_labels.astype(str)

        st.write("Cluster werden farblich dargestellt.")
        st.scatter_chart(pca_df, x="PC1", y="PC2", color="Cluster")
    else:
        st.scatter_chart(pca_df[["PC1", "PC2"]])

    # -------------------
    st.subheader("📈 Erklärte Varianz")
    for i, var in enumerate(explained
