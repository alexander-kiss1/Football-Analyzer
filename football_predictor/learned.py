import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

from football_predictor.data import load_matches
from football_predictor.features import build_match_features

def load_training_table(window = 5):
    matches = load_matches()
    resp = build_match_features(matches, window)

    return resp

def load_xy(window = 5):
    m = load_training_table(window)

    feats = ["home_form_pts", "home_form_gd", "home_rest_days",
              "away_form_pts", "away_form_gd", "away_rest_days"]

    X = m[feats]
    y = m["FTResult"]

    return X, y

def cross_val_predictions(window = 5, n_splits = 5):
    X, y = load_xy(window)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, test_idx, in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        res = LogisticRegression(max_iter=1000)
        pred = res.fit(X_train, y_train).predict_proba(X_test)

        result = pd.DataFrame({"p_A": pred[:,0], "p_D": pred[:,1], "p_H": pred[:,2], "actual": y.iloc[test_idx]})
        folds.append(result)

    return pd.concat(folds).reset_index(drop=True)

