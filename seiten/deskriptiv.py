# seiten/deskriptiv.py
import streamlit as st
import pandas as pd

def highlight_corr(val):
    try:
        if val >= 0.75:
            return "background-color: lightgreen"
        elif val >= 0.5:
            return "background-color: turquoise"
        elif val >= 0.25:
            return "background-color: lightblue"
        elif val <= -0.75:
            return "background-color: red"
        elif val <= -0.5:
            return "background-color: orange"
        elif val <= -0.25:
            return "background-color: yellow"
        else:
            return ""
    except:
        return ""

def show(mobility_df, wetter_df, df):
    st.title("📊 Deskriptive Statistik")

    st.subheader("Theorie")
    st.write("""
    Die deskriptive Statistik dient dazu, Daten durch Kennzahlen und Grafiken zu beschreiben. Typische Methoden sind:
    Mittelwert, Median, Modus: Lage der Werte
    Standardabweichung, Varianz: Streuung der Daten
    Histogramme: Verteilung sichtbar machen
    Korrelation: Zusammenhang zwischen zwei Variablen (z.B. Temperatur und Fussgängerzahl)

    """)

    section = st.selectbox("Datensatz auswählen", [
        "🔀 Kombination: Wetter & Bewegung",
        "🚲 Mobility-Daten",
        "🌦 Wetterdaten"
    ])

    if section == "🚲 Mobility-Daten":
        st.subheader("Grundstatistik – Mobility")
        cols = ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]

        # Nur Zeilen, in denen mindestens eine Zielvariable gültig ist
        mobility_valid = mobility_df.dropna(subset=cols, how="all")

        st.write(f"Anzahl gültiger Zeitpunkte: {len(mobility_valid)} von {len(mobility_df)}")

        st.dataframe(mobility_valid[cols].describe())

        st.write("""
        **Interpretation:**  
        Nur Zeitpunkte, in denen mindestens eine Bewegung gemessen wurde, werden berücksichtigt.  
        Die Tabelle zeigt Mittelwert, Standardabweichung, Minimum, Maximum usw.  
        Diese helfen, das typische Bewegungsverhalten zu verstehen.
        """)

        st.subheader("Fehlende Werte")
        st.dataframe(mobility_df[cols].isnull().sum().to_frame("Fehlend"))

        st.write("""
        **Interpretation:**  
        Viele Zeilen enthalten nur **eine Bewegungsart** (z.B. nur VELO oder nur FUSS).  
        Das ist normal, da jede Zählstelle auf einen Typ spezialisiert ist.  
        Daher sieht man pro Spalte viele NaNs – aber die Analyse filtert diese jetzt korrekt.
        """)

        st.subheader("Histogramm")
        selected = st.selectbox("Spalte wählen", cols)
        st.bar_chart(mobility_valid[selected].dropna().value_counts().sort_index())

        st.write("""
        **Interpretation:**  
        Verteilung der Bewegungen pro Stunde.  
        Häufigkeiten bestimmter Werte (z.B. viele Zeitpunkte mit genau 5 Velos).
        """)

        st.subheader("Korrelationen")
        corr = mobility_valid[cols].corr()
        st.dataframe(corr.style.applymap(highlight_corr).format("{:.2f}"))

        st.write("""
        **Interpretation:**  
        Zeigt, wie stark die Bewegungsarten miteinander korrelieren.  
        Hohe Werte bedeuten, dass z.B. VELO_IN oft mit FUSS_IN zusammen auftritt.
        """)

    elif section == "🌦 Wetterdaten":
        st.subheader("Grundstatistik – Wetter")
        numeric = wetter_df.select_dtypes(include="number").columns
        st.dataframe(wetter_df[numeric].describe())

        st.write("""
        **Interpretation:**  
        Temperatur, Luftfeuchtigkeit, Wind etc. werden als Überblick dargestellt.  
        Extremwerte oder Ausreisser (z.B. hoher Wind oder Druck) können sichtbar werden.
        """)

        st.subheader("Fehlende Werte")
        st.dataframe(wetter_df[numeric].isnull().sum().to_frame("Fehlend"))

        st.subheader("Histogramm")
        selected = st.selectbox("Wetterspalte wählen", list(numeric))
        st.bar_chart(wetter_df[selected].dropna().value_counts().sort_index())

        st.subheader("Korrelationen")
        corr = wetter_df[numeric].corr()
        st.dataframe(corr.style.applymap(highlight_corr).format("{:.2f}"))

        st.write("""
        **Interpretation:**  
        Zeigt Zusammenhänge zwischen Wettergrössen.  
        z.B. hoher Taupunkt und hohe Temperatur korrelieren oft stark.
        """)

    elif section == "🔀 Kombination: Wetter & Bewegung":
        st.subheader("Korrelation Wetter vs. Mobilität")
        mobility_cols = ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]
        wetter_cols = [
            "temp", "humidity", "wind_speed", "clouds_all",
            "dew_point", "feels_like", "pressure", "visibility"
        ]

        kombi_df = df[mobility_cols + wetter_cols].dropna(how="any")
        corr = kombi_df.corr()

        st.write("🔢 Farblich formatierte Korrelationsmatrix")
        st.dataframe(corr.style.applymap(highlight_corr).format("{:.2f}"))

        st.write("""
        **Interpretation:**  
        Erfasst Zusammenhänge zwischen Wetterbedingungen und Bewegungsverhalten.  
        z.B. bei Hitze weniger Velofahrer? Bei Nebel weniger Fussgänger?
        """)

        import plotly.graph_objects as go

        st.subheader("📈 Vergleich: Wetterverlauf und Bewegungsaktivität")
        x_var = st.selectbox("Wettervariable (als Linie)", wetter_cols, key="wetterlinie")
        y_var = st.selectbox("Bewegungsvariable (als Punkte)", mobility_cols, key="bewegungspunkte")

        df_plot = df[["DATUM", x_var, y_var]].dropna()

        fig = go.Figure()

        # Bewegungspunkte
        fig.add_trace(go.Scatter(
            x=df_plot["DATUM"],
            y=df_plot[y_var],
            mode="markers",
            name=y_var,
            marker=dict(color="blue", size=4),
            yaxis="y1"
        ))

        # Wetterlinie (eigene Achse rechts)
        fig.add_trace(go.Scatter(
            x=df_plot["DATUM"],
            y=df_plot[x_var],
            mode="lines",
            name=x_var,
            line=dict(color="orange", width=2),
            yaxis="y2"
        ))

        fig.update_layout(
            title=f"{y_var} (Punkte) vs. {x_var} (Linie)",
            xaxis=dict(title="Zeitpunkt"),
            yaxis=dict(title=y_var, side="left"),
            yaxis2=dict(title=x_var, overlaying="y", side="right", showgrid=False),
            height=500,
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(f"""
              **Interpretation:**  
              Die Punkte zeigen die Bewegungsaktivität ({y_var}) zu verschiedenen Zeitpunkten.  
              Die orange Linie zeigt den Verlauf der Wettervariable ({x_var}) mit eigener Skala rechts.  
              So erkennst du z.B., ob mehr Bewegung bei gutem Wetter stattfindet.
              """)