import pandas as pd

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
