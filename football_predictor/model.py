# Fit team ratings and predict scorelines
import numpy as np
import pandas as pd
from scipy.stats import poisson

FINAL = {"k": 10, "half_life": 90 }

def score_grid(lh, la, max_goals = 10):
    h = poisson.pmf(np.arange(max_goals + 1 ), lh)
    a = poisson.pmf(np.arange(max_goals + 1 ), la)
    grid = np.outer(h,a)
    return grid / grid.sum()

def fit_ratings(matches, k=10, half_life = None, ref_date = None):
    m = matches.copy()
    if half_life:
        ref = ref_date if ref_date is not None else m["MatchDate"].max()
        age = (ref - m["MatchDate"]).dt.days.clip(lower=0)
        m["w"] = 0.5 ** (age / half_life)
    else:
        m["w"] = 1.0

    def wavg(group_col, val_col):
        num = (m["w"] * m[val_col]).groupby(m[group_col]).sum()
        den = m["w"].groupby(m[group_col]).sum()
        return num / den

    ha = (m["w"] * m["FTHome"]).sum() / m["w"].sum()
    aa = (m["w"] * m["FTAway"]).sum() / m["w"].sum()

    raw = pd.DataFrame({
        "home_attack" : wavg("HomeTeam", "FTHome") / ha,
        "home_defense" : wavg("HomeTeam", "FTAway") / aa,
        "away_attack" : wavg("AwayTeam", "FTAway") / aa,
        "away_defense" : wavg("AwayTeam", "FTHome") / ha,
    })

    n = (m["w"].groupby(m["HomeTeam"]).sum().add(m["w"].groupby(m["AwayTeam"]).sum(), fill_value=0).reindex(raw.index).fillna(0))
    w = n / (n + k)
    adj = raw.mul(w, axis = 0).add(1 - w, axis = 0)
    return {"home_avg": ha, "away_avg": aa, "ratings": adj}

def predict(model, home, away, max_goals = 10):
    R, ha, aa = model["ratings"], model["home_avg"], model["away_avg"]
    if home not in R.index or away not in R.index:
        return None
    lh = ha * R.loc[home, "home_attack"] * R.loc[away, "away_defense"]
    la = aa * R.loc[away, "away_attack"] * R.loc[home, "home_defense"]
    g = score_grid(lh, la, max_goals)
    return {
        "Score": (int(round(lh)), int(round(la))),
        "H": np.tril(g, -1).sum(),
        "D": np.trace(g),
        "A": np.triu(g, 1).sum(),
        "xg": (lh, la),
    }
