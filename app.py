import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Dateien einlesen
mobility_files = [
    "zurich_mobility_1.csv",
    "zurich_mobility_2.csv",
    "zurich_mobility_3.csv"
]

mobility_dfs = []
for file in mobility_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        mobility_dfs.append(df)
    else:
        st.warning(f"Datei nicht gefunden: {file}")

# Mobility-Daten zusammenführen
if mobility_dfs:
    mobility_df = pd.concat(mobility_dfs, ignore_index=True)
else:
    mobility_df = pd.DataFrame()

# Wetter- und Standortdaten laden
try:
    wetter_df = pd.read_csv("zurich_wetter.csv")
except FileNotFoundError:
    wetter_df = pd.DataFrame()
    st.warning("zurich_wetter.csv nicht gefunden.")

try:
    standorte_df = pd.read_csv("zurich_standorte.csv")
except FileNotFoundError:
    standorte_df = pd.DataFrame()
    st.warning("zurich_standorte.csv nicht gefunden.")


# Zeitspalten vereinheitlichen
mobility_df["DATUM"] = pd.to_datetime(mobility_df["DATUM"])
wetter_df["dt_iso"] = pd.to_datetime(wetter_df["dt_iso"])
mobility_df["DATUM"] = mobility_df["DATUM"].dt.floor("H")
wetter_df["dt_iso"] = wetter_df["dt_iso"].dt.floor("H")

# Aggregation über alle Standorte
mobility_agg = mobility_df.groupby("DATUM")[["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]].sum().reset_index()
df = pd.merge(mobility_agg, wetter_df, left_on="DATUM", right_on="dt_iso", how="inner")


# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite auswählen", [
    "Start",
    "Deskriptive Statistik",
    "Multiple Lineare Regression (MLR)",
    "PCA",
    "Zeitreihenanalyse",
    "Clustering (K-Means)"
])

# Startseite
if page == "Start":
    st.title("Statistische Auswertung – Zürich Mobility & Wetter")
    st.write("""
    Willkommen!  
    Diese Anwendung zeigt statistische Methoden auf Grundlage von Mobilitäts- und Wetterdaten aus Zürich.  
    Links findest du die Methoden:
    - Deskriptive Statistik
    - Multiple Lineare Regression
    - PCA (Hauptkomponentenanalyse)
    - Zeitreihenanalyse
    - Clustering (K-Means)
    """)

    st.subheader("🚲 Zusammengesetzte Mobility-Daten (Auszug)")
    st.dataframe(mobility_df.head(100))  # Anzeige auf 100 Zeilen begrenzt

    st.subheader("🌦 Wetterdaten Zürich (Auszug)")
    st.dataframe(wetter_df.head(100))

    st.subheader("📍 Standorte der Zählstationen (Auszug)")
    st.dataframe(standorte_df.head(100))

# Deskriptive Statistik
elif page == "Deskriptive Statistik":
    st.title("Deskriptive Statistik")
    st.write("""
    Beschreibt Daten durch Kennzahlen wie:
    - Mittelwert, Median, Modus
    - Standardabweichung, Varianz, Spannweite
    - Histogramme, Boxplots

    Ziel: **Verständliche Zusammenfassung der Daten** ohne Modelle.
    """)

    st.subheader("Numerische Grundauswertung")
    numeric_cols = ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]
    st.dataframe(mobility_df[numeric_cols].describe())

    st.subheader("Fehlende Werte je Kategorie")
    missing = mobility_df[numeric_cols].isnull().sum().to_frame(name="Anzahl fehlender Werte")
    st.dataframe(missing)

    st.subheader("Verteilung (Histogramm)")

    selected_column = st.selectbox("Wähle eine Spalte für das Histogramm", numeric_cols)

    if selected_column:
        st.bar_chart(mobility_df[selected_column].dropna().value_counts().sort_index())

    st.subheader("Korrelation zwischen Bewegungsarten")
    correlation = mobility_df[numeric_cols].corr()
    st.dataframe(correlation)

# MLR
elif page == "Multiple Lineare Regression (MLR)":
    st.title("Multiple Lineare Regression (MLR)")
    st.write("""
    Modelliert den Zusammenhang zwischen einer Zielvariable \( Y \) und mehreren Einflussgrößen \( X_1, X_2, \dots \):

    \[
    Y = \beta_0 + \beta_1 X_1 + \dots + \beta_n X_n + \epsilon
    \]

    **Residuenanalyse** prüft, ob das Modell passend ist (z. B. Normalverteilung der Fehler, konstante Varianz).
    """)


    st.write("Diese Analyse versucht, Mobilitätsverhalten anhand von Wetterdaten vorherzusagen.")

    target = st.selectbox("Zu prognostizierende Zielvariable", ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"])

    wetter_features = [
        "temp", "humidity", "wind_speed", "clouds_all",
        "dew_point", "feels_like", "pressure", "visibility"
    ]

    # Drop Zeilen mit fehlenden Werten
    df_ml = df[[target] + wetter_features].dropna()

    # Input & Target definieren
    X = df_ml[wetter_features]
    y = df_ml[target]

    # Split in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell trainieren
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Vorhersagen
    y_pred = model.predict(X_test)

    st.subheader("Modellgüte")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
    st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred, squared=False):.2f}")

    st.subheader("Modellkoeffizienten")
    coeff_df = pd.DataFrame({
        "Merkmal": wetter_features,
        "Koeffizient": model.coef_
    })
    st.dataframe(coeff_df)

    st.subheader("Vorhergesagte vs. echte Werte (Streudiagramm)")
    st.scatter_chart(pd.DataFrame({"Echt": y_test, "Vorhersage": y_pred}))

# PCA
elif page == "PCA":
    st.title("PCA – Hauptkomponentenanalyse")
    st.write("""
    PCA reduziert die Dimension der Daten durch Transformation in neue Achsen (Hauptkomponenten), die möglichst viel Varianz erhalten.

    Vorteile:
    - Weniger Variablen
    - Bessere Visualisierbarkeit
    - Vorbereitung für ML oder Clustering
    """)

    st.write("""
        Diese Seite führt eine PCA durch, um Mobilitäts- und Wettermerkmale auf 2 Dimensionen zu reduzieren.  
        So lässt sich z. B. erkennen, ob ähnliche Tage/Wetterlagen ähnliche Mobilitätsmuster erzeugen.
        """)

    # Auswahl der Features für PCA
    features = [
        "VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT",
        "temp", "humidity", "wind_speed", "clouds_all",
        "dew_point", "feels_like", "pressure", "visibility"
    ]

    df_pca = df[features].dropna()

    # Standardisierung
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_pca)

    # PCA durchführen
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)

    # Ergebnis als DataFrame
    pca_df = pd.DataFrame(data=components, columns=["PC1", "PC2"])

    st.subheader("Scatterplot der Hauptkomponenten")
    st.write("Jeder Punkt ist ein Zeitpunkt (eine Stunde).")
    st.scatter_chart(pca_df)

    st.subheader("Erklärte Varianz")
    st.write(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
    st.write(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")

# Zeitreihenanalyse
elif page == "Zeitreihenanalyse":
    st.title("Zeitreihenanalyse")
    st.write("""
    Analyse von Daten über die Zeit, z. B. Wetter oder Mobilitätszahlen je Stunde/Tag.

    Ziel: Muster erkennen (Trend, Saisonalität, Zyklen) und **Vorhersagen** treffen.  
    Typische Modelle: AR, MA, ARIMA
    """)
    st.write("Hier wird der zeitliche Verlauf der Bewegungen in Zürich visualisiert.")

    variable = st.selectbox("Wähle eine Variable für den Zeitverlauf", ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"])

    st.line_chart(mobility_agg.set_index("DATUM")[variable])

    st.subheader("Gleitender Durchschnitt (24h)")
    mobility_agg[f"{variable}_SMA24"] = mobility_agg[variable].rolling(window=24).mean()
    st.line_chart(mobility_agg.set_index("DATUM")[f"{variable}_SMA24"])

    st.subheader("Tägliche Gesamtsumme")
    daily = mobility_agg.copy()
    daily["DATUM"] = daily["DATUM"].dt.date
    daily_sum = daily.groupby("DATUM")[variable].sum()
    st.line_chart(daily_sum)

# Clustering
elif page == "Clustering (K-Means)":
    st.title("Clustering – K-Means")
    st.write("""
    Unüberwachtes Verfahren zur Gruppierung ähnlicher Datenpunkte in **\( k \)** Cluster.

    Schritte:
    1. Wähle \( k \)
    2. Initialisiere Clusterzentren
    3. Weise Punkte zu, berechne neue Zentren
    4. Wiederhole bis Konvergenz

    Ziel: **Ähnliche Daten gruppieren** (z. B. Verkehrsorte mit ähnlichem Muster)
    """)


    st.write("""
    Hier werden Zeitpunkte (Stunden) anhand von Wetter- und Mobilitätsdaten in Cluster gruppiert.  
    Dies kann typische Mobilitätsmuster (z. B. „Rush Hour bei Regen“) aufdecken.
    """)

    # Features wählen
    features = [
        "VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT",
        "temp", "humidity", "wind_speed", "clouds_all",
        "dew_point", "feels_like", "pressure", "visibility"
    ]

    df_cluster = df[features].dropna()

    # Standardisieren
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Cluster-Anzahl auswählen
    k = st.slider("Anzahl Cluster (k)", 2, 10, 4)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # PCA zur Darstellung
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels.astype(str)

    st.subheader("Visualisierung der Cluster (PCA-Projektion)")
    st.write("Jeder Punkt ist ein Zeitpunkt, die Farben zeigen Clusterzugehörigkeit.")

    # Interaktiver Scatterplot
    st.scatter_chart(pca_df, x="PC1", y="PC2", color="Cluster")
