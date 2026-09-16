import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from football_predictor.evaluate import ODDS, ELO, log_loss, rps, predictions
from football_predictor.data import load_matches
from football_predictor.features import build_match_features, current_team_form
from football_predictor.model import FINAL, fit_ratings

FEATURES = ["home_form_pts", "home_form_gd", "home_rest_days",
            "away_form_pts","away_form_gd", "away_rest_days"]

def load_training_table(window = 5, matches = None):
    if matches is None:
        matches = load_matches()
    resp = build_match_features(matches, window)

    return resp

def load_xy(window = 5, matches = None):
    m = load_training_table(window, matches = matches)

    X = m[FEATURES]
    y = m["FTResult"]

    return X, y

def cross_val_predictions(window = 5, n_splits = 5):
    m = load_training_table(window)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, test_idx, in tscv.split(m):
        extra_cols = ["HomeTeam", "AwayTeam", "MatchDate"] + ODDS + ELO
        extras = m.iloc[test_idx][extra_cols]
        X_train, X_test = m.iloc[train_idx][FEATURES], m.iloc[test_idx][FEATURES]
        y_train, y_test = m.iloc[train_idx]["FTResult"], m.iloc[test_idx]["FTResult"]

        res = LogisticRegression(max_iter=1000)
        pred = res.fit(X_train, y_train).predict_proba(X_test)

        result = pd.DataFrame({"p_A": pred[:,0], "p_D": pred[:,1], "p_H": pred[:,2], "actual": y_test})
        fold_frame = pd.concat([result, extras], axis=1)
        folds.append(fold_frame)

    return pd.concat(folds).reset_index(drop=True)


def build_comparisons(frame):
    base = frame["actual"].value_counts(normalize = True)
    frame_base = frame.assign(p_H = base["H"], p_D = base["D"], p_A = base["A"])

    inv = 1 / frame[ODDS].to_numpy()
    book = inv / inv.sum(axis=1, keepdims=True)
    frame_book = frame.assign(p_H=book[:, 0], p_D=book[:, 1], p_A=book[:, 2])

    d = (frame["HomeElo"] - frame["AwayElo"]).to_numpy() + 65
    ph = 1 / (1 + 10 ** (-d / 400))
    frame_elo = frame.assign(p_D=0.26, p_H=(1 - .26) * ph, p_A=(1 - .26) * (1 - ph))

    return [("model", frame), ("base rate", frame_base), ("Elo + 65", frame_elo), ("bookmaker", frame_book)]

def score_comparisons(rows):
    return pd.DataFrame(
        [{"method": name, "n": len(g), "log_loss": log_loss(g), "rps": rps(g)} for name, g in rows],
    ).set_index("method")

def cross_val_poisson_predictions(window = 5, n_splits = 5):
    m = load_training_table(window)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, test_idx, in tscv.split(m):
        train_m, test_m = m.iloc[train_idx], m.iloc[test_idx]

        model = fit_ratings(train_m, **FINAL, ref_date = train_m["MatchDate"].max())
        fold_frame = predictions(model, test_m)
        folds.append(fold_frame)

    return pd.concat(folds, axis = 0).reset_index(drop=True)

def train_classifier(window=5, matches = None):
    X, y = load_xy(window, matches = matches)
    res = LogisticRegression(max_iter=1000)
    res.fit(X, y)

    return res

def predict_classifier(res, form_table, home, away, fixture_date):
    if home not in form_table.index or away not in form_table.index:
        return None

    home_row = form_table.loc[home]
    away_row = form_table.loc[away]
    home_rest_days = (fixture_date - home_row["MatchDate"]).days
    away_rest_days = (fixture_date - away_row["MatchDate"]).days

    row = pd.DataFrame([{
        "home_form_pts": home_row["form_pts"],
        "home_form_gd": home_row["form_gd"],
        "home_rest_days": home_rest_days,
        "away_form_pts": away_row["form_pts"],
        "away_form_gd": away_row["form_gd"],
        "away_rest_days": away_rest_days,
    }])

    proba = res.predict_proba(row)
    return {"A": proba[0, 0], "D": proba[0, 1], "H": proba[0, 2]}


