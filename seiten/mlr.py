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

    target = st.selectbox("Zielvariable wählen", ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"])
    features = ["temp", "humidity", "wind_speed", "clouds_all",
                "dew_point", "feels_like", "pressure", "visibility"]

    df_ml = df[[target] + features].dropna()
    X = df_ml[features]
    y = df_ml[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuen = y_test - y_pred

    st.subheader("📊 Modellgüte")
    st.write(f"R²: {r2_score(y_test, y_pred):.3f}")
    st.write(f"RMSE: {mean_squared_error(y_test, y_pred, False):.2f}")  # <- FIXED HERE

    st.subheader("📉 Koeffizienten")
    st.dataframe(pd.DataFrame({"Merkmal": features, "Koeffizient": model.coef_}))

    st.subheader("📈 Vorhersage vs. Echtdaten")
    scatter_df = pd.DataFrame({"Echt": y_test, "Vorhersage": y_pred})
    st.scatter_chart(scatter_df)

    # --- Residuenanalyse ---
    st.subheader("🔍 Residuenanalyse")

    # Histogramm + Normalverteilungskurve
    fig, ax = plt.subplots()
    ax.hist(residuen, bins=30, alpha=0.6, density=True, label="Residuen")

    # Glockenkurve
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, residuen.mean(), residuen.std())
    ax.plot(x, p, "k", linewidth=2, label="Normalverteilung")

    ax.set_title("Histogramm der Residuen mit Glockenkurve")
    ax.legend()
    st.pyplot(fig)

    # Q-Q-Plot
    st.subheader("📐 Q-Q-Plot der Residuen")
    fig2, ax2 = plt.subplots()
    stats.probplot(residuen, dist="norm", plot=ax2)
    ax2.set_title("Q-Q-Plot")
    st.pyplot(fig2)
