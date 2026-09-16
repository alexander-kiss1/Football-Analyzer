import pandas as pd
import numpy as np

def _team_matches(matches):
    home = matches[["MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"]].rename(
        columns = {"HomeTeam": "team", "AwayTeam": "opponent", "FTHome" : "goals_for", "FTAway": "goals_against" }
    )
    away = matches[["MatchDate", "AwayTeam", "HomeTeam", "FTAway", "FTHome"]].rename(
        columns = {"AwayTeam": "team", "HomeTeam": "opponent", "FTAway": "goals_for", "FTHome": "goals_against"}
    )
    home["venue"] = "H"
    away["venue"] = "A"

    m = pd.concat([home, away], ignore_index=True).sort_values(["team", "MatchDate"]).reset_index(drop=True)

    return m

def add_points(team_matches):
    team_matches["points"] = np.select([team_matches["goals_for"] > team_matches["goals_against"], team_matches["goals_for"] == team_matches["goals_against"]],
              [3, 1],
              default = 0
              )
    return team_matches

def add_form(team_matches, window):
    team_matches["form_pts"] = team_matches.groupby("team")["points"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    team_matches["goal_diff"] = team_matches["goals_for"] - team_matches["goals_against"]
    team_matches["form_gd"] = team_matches.groupby("team")["goal_diff"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    return team_matches

def add_rest_days(team_matches):
    team_matches["rest_days"] = team_matches.groupby("team")["MatchDate"].diff().dt.days

    return team_matches

def current_team_form(matches, window=5):
    tm = _team_matches(matches)
    tm = add_points(tm)
    tm["goal_diff"] = tm["goals_for"] - tm["goals_against"]

    tm["form_pts"] = tm.groupby("team")["points"].transform(
        lambda s: s.rolling( window = window, min_periods = 1).mean())
    tm["form_gd"] = tm.groupby("team")["goal_diff"].transform(
        lambda s: s.rolling(window, min_periods=1).mean())

    return tm.groupby(["team"]).tail(1)

def build_team_features(matches, window):
    tm = _team_matches(matches)
    tm = add_points(tm)
    tm = add_form(tm, window)
    tm = add_rest_days(tm)

    return tm

def build_match_features(matches, window):
    tm = build_team_features(matches, window)
    home = tm[["team", "MatchDate", "form_pts", "form_gd", "rest_days"]].rename(
        columns = {"team": "HomeTeam", "form_pts": "home_form_pts", "form_gd": "home_form_gd", "rest_days": "home_rest_days" }
    )

    away = tm[["team", "MatchDate", "form_pts", "form_gd", "rest_days"]].rename(
        columns = {"team": "AwayTeam", "form_pts": "away_form_pts", "form_gd": "away_form_gd", "rest_days": "away_rest_days"}
    )

    result = matches.merge(home, on=["HomeTeam", "MatchDate"], how = "left") \
            .merge(away, on = ["AwayTeam", "MatchDate"], how = "left")

    resp = result.dropna(subset=["home_form_pts","home_form_gd", "home_rest_days",
                                 "away_form_pts", "away_form_gd", "away_rest_days"])

    return resp
