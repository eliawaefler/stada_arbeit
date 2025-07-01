import streamlit as st

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

# Zeitreihenanalyse
elif page == "Zeitreihenanalyse":
    st.title("Zeitreihenanalyse")
    st.write("""
    Analyse von Daten über die Zeit, z. B. Wetter oder Mobilitätszahlen je Stunde/Tag.

    Ziel: Muster erkennen (Trend, Saisonalität, Zyklen) und **Vorhersagen** treffen.  
    Typische Modelle: AR, MA, ARIMA
    """)

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
