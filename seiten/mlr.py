# seiten/mlr.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def show(df):
    st.title("📈 Multiple Lineare Regression (MLR)")

    # Zielvariable
    target = st.selectbox("Zielvariable wählen", ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"])

    # Wettermerkmale (Checkbox-Auswahl)
    all_features = ["temp", "humidity", "wind_speed", "clouds_all",
                    "dew_point", "feels_like", "pressure", "visibility"]
    selected_features = st.multiselect("Wähle die Wetter-Variablen aus", all_features, default=all_features)

    if len(selected_features) < 1:
        st.warning("Bitte mindestens eine Variable auswählen.")
        return

    # Daten vorbereiten
    df_ml = df[[target] + selected_features].dropna()
    X = df_ml[selected_features]
    y = df_ml[target]

    # -------------------
    st.subheader("🧮 Korrelation der unabhängigen Variablen")

    if len(selected_features) >= 2:
        corr = X.corr()

        def highlight_corr(val):
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

        st.dataframe(corr.style.applymap(highlight_corr).format("{:.2f}"))

        st.write("""
        **Hinweis:**  
        Sehr hohe Korrelationen zwischen den unabhängigen Variablen (Multikollinearität)  
        können das Modell instabil machen und Interpretationen verzerren.
        """)
    else:
        st.info("Mindestens 2 unabhängige Variablen auswählen, um Korrelationen zu sehen.")

    # -------------------
    # Modelltraining
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuen = y_test - y_pred

    # -------------------
    st.subheader("📊 Modellgüte")
    st.write(f"**R²:** {r2_score(y_test, y_pred):.3f}")
    st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred):.2f}")

    # -------------------
    st.subheader("📉 Koeffizienten")
    coeff_df = pd.DataFrame({
        "Merkmal": selected_features,
        "Koeffizien
