from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_matches(path = None, since = "2022-08-01"):
    path = Path(path) if path else DATA_DIR / "laliga.csv"
    df = pd.read_csv(path)

    if "Division" in df.columns:
        df = df[df["Division"] == "SP1"].copy()

    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df[df["MatchDate"] >= since].copy()
    df = df.dropna(subset=["FTHome", "FTAway"])

    return df.sort_values("MatchDate").reset_index(drop = True)