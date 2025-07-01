# seiten/mlr.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def show(df):
    st.title("📈 Multiple Lineare Regression (MLR)")

    target = st.selectbox("Zielvariable wählen", ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"])
    features = ["temp", "humidity", "wind_speed", "clouds_all", "dew_point", "feels_like", "pressure", "visibility"]

    df_ml = df[[target] + features].dropna()
    X = df_ml[features]
    y = df_ml[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Modellgüte")
    st.write(f"R²: {r2_score(y_test, y_pred):.3f}")
    st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

    st.subheader("Koeffizienten")
    st.dataframe(pd.DataFrame({"Merkmal": features, "Koeffizient": model.coef_}))

    st.subheader("Vorhersage vs. Ist")
    st.scatter_chart(pd.DataFrame({"Echt": y_test, "Vorhersage": y_pred}))
