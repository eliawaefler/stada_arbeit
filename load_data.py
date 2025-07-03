# load_data.py
import pandas as pd
import os

def load_all_data():
    # === Mobility-Daten laden ===
    mobility_files = [
        "data/zurich_mobility_1.csv",
        "data/zurich_mobility_2.csv",
        "data/zurich_mobility_3.csv"
    ]

    mobility_dfs = []
    for file in mobility_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            mobility_dfs.append(df)
        else:
            print(f"❌ Datei nicht gefunden: {file}")

    if mobility_dfs:
        mobility_df = pd.concat(mobility_dfs, ignore_index=True)
        mobility_df["DATUM"] = pd.to_datetime(mobility_df["DATUM"], errors="coerce").dt.floor("h")
        mobility_df = mobility_df.dropna(subset=["DATUM"])
    else:
        mobility_df = pd.DataFrame()
        print("⚠️ Keine Mobility-Daten gefunden.")

    # === Wetterdaten laden (über Unix-Timestamp dt) ===
    try:
        wetter_df = pd.read_csv("data/zurich_wetter.csv")

        # Verwende Unix-Zeitstempel für robustes Parsing
        wetter_df["dt_iso"] = pd.to_datetime(wetter_df["dt"], unit="s").dt.floor("H")

        # Entferne Zeilen mit ungültigen Zeitstempeln
        wetter_df = wetter_df.dropna(subset=["dt_iso"])

    except Exception as e:
        print("❌ Fehler beim Laden der Wetterdaten:", e)
        wetter_df = pd.DataFrame()

    # === Standortdaten laden ===
    try:
        standorte_df = pd.read_csv("data/zurich_standorte.csv")
    except Exception as e:
        print("❌ Fehler beim Laden der Standortdaten:", e)
        standorte_df = pd.DataFrame()

    # === Aggregation der Bewegungsdaten ===
    if not mobility_df.empty:
        mobility_agg = mobility_df.groupby("DATUM")[["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]].sum().reset_index()
    else:
        mobility_agg = pd.DataFrame()

    # === Merge Mobility + Wetter ===
    if not mobility_agg.empty and not wetter_df.empty:
        df = pd.merge(mobility_agg, wetter_df, left_on="DATUM", right_on="dt_iso", how="inner")
    else:
        df = pd.DataFrame()
        print("⚠️ Kein gemeinsamer Datensatz für Analyse verfügbar (leerer Merge).")

    return mobility_df, wetter_df, standorte_df, df, mobility_agg
