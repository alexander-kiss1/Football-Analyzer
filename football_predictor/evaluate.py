import numpy as np
import pandas as pd
from football_predictor.model import fit_ratings, predict

OUTCOME = { "H": 0, "D": 1, "A": 2}
ODDS = ["OddHome", "OddDraw", "OddAway"]
ELO = ["HomeElo", "AwayElo"]

def log_loss(frame):
    p = frame[["p_H", "p_D", "p_A"]].to_numpy()
    ix = frame["actual"].map(OUTCOME).to_numpy()
    return -np.log(p[np.arange(len(p)), ix]).mean()

def rps(frame):
    p = frame[["p_H", "p_D", "p_A"]].to_numpy()
    oh = np.eye(3)[frame["actual"].map(OUTCOME).to_numpy()]
    cp, co = np.cumsum(p, axis = 1), np.cumsum(oh, axis = 1)
    return (((cp - co) ** 2).sum(axis = 1) / 2 ).mean()

def split(matches, n_test = 150):
    m = matches.sort_values("MatchDate").reset_index(drop = True)
    return m.iloc[:-n_test], m.iloc[-n_test:]

def predictions(model, test, extra = ()):
    rows = []
    for _, r in test.iterrows():
        p = predict(model, r["HomeTeam"], r["AwayTeam"])
        if p is None:
            continue
        row =  {"p_H": p["H"], "p_D": p["D"], "p_A": p["A"], "actual": r["FTResult"]}
        for col in extra:
            row[col] = r[col]
        rows.append(row)
    return pd.DataFrame(rows)

def evaluate(model, test):
    f = predictions(model, test)
    return {"n": len(f), "log_loss": log_loss(f), "rps": rps(f)}

def benchmark(model, train, test):
    f = predictions(model, test, extra = ODDS + ELO).dropna(subset = ODDS + ELO).reset_index(drop = True)
    base = train["FTResult"].value_counts(normalize = True)
    f_base = f.assign(p_H = base ["H"], p_D = base ["D"], p_A = base ["A"])

    inv = 1 / f[ODDS].to_numpy()
    book = inv / inv.sum(axis = 1, keepdims = True)
    f_book = f.assign(p_H = book[:, 0], p_D = book[:, 1], p_A = book[:, 2])

    d = (f["HomeElo"] - f["AwayElo"]).to_numpy() + 65
    ph = 1 / (1 + 10 ** (-d / 400))
    f_elo = f.assign(p_D = 0.26, p_H = (1 - .26) * ph, p_A = (1 - .26) * (1 - ph))

    rows = [("model", f), ("base rate", f_base), ("Elo + 65", f_elo), ("bookmaker", f_book)]
    return pd.DataFrame(
        [{"method": name, "n" : len(g), "log_loss": log_loss(g), "rps": rps(g)} for name, g in rows],
    ).set_index("method")
