# load_data.py
import pandas as pd
import os

def load_all_data():
    # Mobility
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

    mobility_df = pd.concat(mobility_dfs, ignore_index=True)
    mobility_df["DATUM"] = pd.to_datetime(mobility_df["DATUM"]).dt.floor("H")

    # Wetterdaten sicher parsen
    wetter_df = pd.read_csv("zurich_wetter.csv")
    wetter_df["dt_iso"] = pd.to_datetime(wetter_df["dt_iso"], errors="coerce").dt.floor("H")
    wetter_df = wetter_df.dropna(subset=["dt_iso"])

    # Standorte
    try:
        standorte_df = pd.read_csv("zurich_standorte.csv")
    except FileNotFoundError:
        standorte_df = pd.DataFrame()

    # Aggregation: Stundenwerte summieren
    mobility_agg = mobility_df.groupby("DATUM")[["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]].sum().reset_index()

    # Merge mit Wetterdaten
    df = pd.merge(mobility_agg, wetter_df, left_on="DATUM", right_on="dt_iso", how="inner")

    return mobility_df, wetter_df, standorte_df, df, mobility_agg
