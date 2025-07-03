# seiten/clustering.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def show(df):
    st.title("🔀 Clustering – Vergleich von Methoden")
    st.subheader("Theorie")
    st.write("""
    **Was ist Clustering?**  
    Clustering ist ein Verfahren zur Gruppierung von Datenpunkten basierend auf Ähnlichkeit – ohne dass vorher Klassen vorgegeben werden.  
    Ziel ist es, **Muster oder Strukturen** in den Daten zu erkennen.

    **K-Means Clustering:**  
    - Teilt die Daten in *K Gruppen* ein.
    - Jeder Punkt wird dem nächstgelegenen Clusterzentrum zugeordnet.
    - Die Zentren werden so angepasst, dass die Abstände innerhalb eines Clusters minimiert werden.

    **Typische Anwendung hier:**  
    - Gruppierung von Zeitpunkten mit ähnlichen Bewegungsmustern
    - Erkennen von typischen Wetter-Mobilitäts-Kombinationen
    - Visualisierung von ungekannten Strukturen im Datensatz
    """)


    # Zeitliche Features
    df = df.copy()
    df["weekday"] = df["DATUM"].dt.weekday
    df["hour"] = df["DATUM"].dt.hour

    # Auswahl an Variablen
    features = [
        "VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT",
        "temp", "humidity", "wind_speed", "clouds_all",
        "dew_point", "feels_like", "pressure", "visibility",
        "weekday", "hour"
    ]

    selected = st.multiselect("Variablen fürs Clustering", features, default=features)

    if len(selected) < 2:
        st.warning("Bitte mindestens zwei Variablen auswählen.")
        return

    df_cluster = df[selected].dropna()
    X_scaled = StandardScaler().fit_transform(df_cluster)

    st.subheader("🔧 Cluster-Methode wählen")
    method = st.selectbox("Clustering-Methode", ["KMeans", "Hierarchical (Bottom-Up)", "Hierarchical (Top-Down – simuliert)", "MeanShift"])

    if method == "KMeans":
        k = st.slider("Anzahl Cluster (k)", 2, 10, 4)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)

    elif method == "Hierarchical (Bottom-Up)":
        k = st.slider("Anzahl Cluster (k)", 2, 10, 4)
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_scaled)

    elif method == "Hierarchical (Top-Down – simuliert)":
        # Simuliere durch invertiertes Agglomerative Clustering (nicht perfekt)
        k = st.slider("Anzahl Cluster (k)", 2, 10, 4)
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_scaled[::-1])  # Reverse für simulierten "Top-Down-Effekt"

    elif method == "MeanShift":
        model = MeanShift()
        labels = model.fit_predict(X_scaled)
        k = len(np.unique(labels))
        st.write(f"→ Anzahl automatisch erkannter Cluster: {k}")

    else:
        st.error("Unbekannte Methode.")
        return

    # PCA zur Darstellung
    components = PCA(n_components=2).fit_transform(X_scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels.astype(str)

    st.subheader(f"📍 Cluster Visualisierung ({method})")
    fig, ax = plt.subplots(figsize=(6, 6))
    for cluster in sorted(pca_df["Cluster"].unique()):
        cluster_data = pca_df[pca_df["Cluster"] == cluster]
        ax.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"Cluster {cluster}", alpha=0.6)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Cluster in PCA-2D-Projektion")
    ax.legend()
    st.pyplot(fig)

    st.write("""
    **Interpretation:**  
    Ähnliche Stunden (z. B. "Rush Hour bei Regen", "Wochenende bei Sonne") werden durch das Clustering automatisch gruppiert.  
    Die PCA-Projektion erlaubt die visuelle Trennung der Cluster im 2D-Raum.
    """)
