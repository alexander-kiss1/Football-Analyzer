import pandas as pd
from football_predictor.data import load_matches
from football_predictor.fixtures import load_fixtures, fixtures_to_matches, thin_teams
from football_predictor.learned import predict_classifier, train_classifier
from football_predictor.model import fit_ratings, FINAL, predict
from football_predictor.features import current_team_form

def training_matches(refresh = False):
    matches = load_matches()
    live = fixtures_to_matches(load_fixtures(refresh = refresh))
    new = live[live["MatchDate"] > matches["MatchDate"].max()]
    return pd.concat([matches, new], ignore_index=True).sort_values("MatchDate").reset_index(drop=True)

def season_table(refresh = False, team = None):
    matches = training_matches(refresh)
    model = fit_ratings(matches, **FINAL, ref_date = matches["MatchDate"].max())
    fixtures = load_fixtures(refresh = refresh)
    clf = train_classifier(matches = matches)
    form_table = current_team_form(matches).set_index("team")
    if team:
        fixtures = fixtures[(fixtures["home_r"] == team) | (fixtures["away_r"] == team)]
    thin = thin_teams(matches)

    rows = []
    for _, m in fixtures.sort_values("matchday").iterrows():
        p = predict(model, m["home_r"], m["away_r"])
        c = predict_classifier(clf, form_table, m["home_r"], m["away_r"], m["utc"].tz_localize(None))
        if p is None:
            continue
        if c is None:
            continue
        ph, pa = p["Score"]
        pred_outcome = "H" if ph > pa else ("A" if ph < pa else "D")
        if pd.isna(m["fh"]):
            correct = ""
        else:
            actual_outcome = "H" if m["fh"] > m["fa"] else ("A" if m["fh"] < m["fa"] else "D")
            correct = "✔" if pred_outcome == actual_outcome else "X"
        rows.append({
            "matchday": m["matchday"],
            "home": m["home"],
            "away": m["away"],
            "status": m["status"],
            "pred": f'{p["Score"][0]}-{p["Score"][1]}',
            "H": p["H"], "D": p["D"], "A": p["A"],
            "actual": "" if pd.isna(m["fh"]) else f'{int(m["fh"])}-{int(m["fa"])}',
            "flag": "⚠️" if {m["home_r"], m["away_r"]} & thin else "",
            "correct" : correct,
            "clf_H": c["H"],
            "clf_D": c["D"],
            "clf_A": c["A"]
        })
    return pd.DataFrame(rows)

