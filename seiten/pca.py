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
    for i, var in enumerate(explained_var):
        st.write(f"{pc_names[i]}: {var:.2%}")

    # -------------------
    st.subheader("📐 Einflussrichtungen (Loadings) auf PC1 & PC2")
    st.write("""
    Die Richtung und Länge der Pfeile zeigen, wie stark ein ursprüngliches Merkmal  
    zu PC1 und PC2 beiträgt. Je länger ein Pfeil, desto mehr Varianz dieses Merkmals  
    wird durch diese Hauptkomponenten erklärt.
    """)

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

    st.write("""
    **Wie liest man die Pfeilgrafik?**

    - 🔵 Pfeile mit ähnlicher Richtung → Variablen sind stark **positiv korreliert**  
    - 🔴 Pfeile in entgegengesetzte Richtung → **negativ korreliert**
    - ⚪ Pfeile senkrecht zueinander → **unkorreliert**
    - 🔺 Punkte, die in Pfeilrichtung liegen → **werden stark durch dieses Merkmal beeinflusst**
    """)

