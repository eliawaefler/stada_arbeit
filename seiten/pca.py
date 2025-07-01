# seiten/pca.py
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def show(df):
    st.title("🧮 PCA – Hauptkomponentenanalyse")

    features = [
        "VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT",
        "temp", "humidity", "wind_speed", "clouds_all",
        "dew_point", "feels_like", "pressure", "visibility"
    ]
    df_pca = df[features].dropna()
    scaled = StandardScaler().fit_transform(df_pca)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])

    st.subheader("2D PCA Scatterplot")
    st.scatter_chart(pca_df)

    st.subheader("Erklärte Varianz")
    st.write(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
    st.write(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")
