# seiten/clustering.py
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def show(df):
    st.title("🔀 Clustering – K-Means")

    features = [
        "VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT",
        "temp", "humidity", "wind_speed", "clouds_all",
        "dew_point", "feels_like", "pressure", "visibility"
    ]
    df_cluster = df[features].dropna()
    scaled = StandardScaler().fit_transform(df_cluster)

    k = st.slider("Anzahl Cluster", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(scaled)
    labels = kmeans.labels_

    components = PCA(n_components=2).fit_transform(scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels.astype(str)

    st.subheader("Cluster in PCA-2D-Projektion")
    st.scatter_chart(pca_df, x="PC1", y="PC2", color="Cluster")
