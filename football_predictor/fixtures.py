import json
import os
import pandas as pd
from football_predictor.data import DATA_DIR
from collections import Counter

FIXTURES_JSON = DATA_DIR / "raw" / "fixtures_PD.json"
API_URL = "https://api.football-data.org/v4/competitions/PD/matches"

NAME_MAP = {
    "Athletic Club": "Ath Bilbao",
    "CA Osasuna": "Osasuna",
    "Club Atlético de Madrid": "Ath Madrid",
    "Deportivo Alavés": "Alaves",
    "Elche CF": "Elche",
    "FC Barcelona": "Barcelona",
    "Getafe CF": "Getafe",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "RC Celta de Vigo": "Celta",
    "RC Deportivo La Coruña": "La Coruna",
    "Real Betis Balompié": "Betis",
    "Real Madrid CF": "Real Madrid",
    "Real Sociedad de Fútbol": "Sociedad",
    "Rayo Vallecano de Madrid": "Vallecano",
    "RCD Espanyol de Barcelona": "Espanol",
    "Real Racing Club de Santander": "Santander",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal"
}

def fetch_fixtures():
    import requests

    resp = requests.get(API_URL, headers={"X-Auth-Token": os.environ["FOOTBALL_API_KEY"]})
    resp.raise_for_status()
    data = resp.json()
    FIXTURES_JSON.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES_JSON.write_text(json.dumps(data))
    return data

def load_fixtures(refresh = False):
    if refresh or not FIXTURES_JSON.exists():
        data = fetch_fixtures()
    else:
        data = json.loads(FIXTURES_JSON.read_text())

    fixtures = pd.DataFrame([
        {
            "matchday": m["matchday"],
            "utc": m["utcDate"],
            "status": m["status"],
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "fh": m["score"]["fullTime"]["home"],
            "fa": m["score"]["fullTime"]["away"],
        }
        for m in data["matches"]
    ])
    fixtures["utc"] = pd.to_datetime(fixtures["utc"])
    fixtures["home_r"] = fixtures["home"].map(NAME_MAP)
    fixtures["away_r"] = fixtures["away"].map(NAME_MAP)

    unmapped = set(fixtures.loc[fixtures["home_r"].isna(), "home"]) | set(fixtures.loc[fixtures["away_r"].isna(), "away"])
    assert not unmapped, f"unmapped API team names: {unmapped}"

    return fixtures

def thin_teams(matches, threshold = 10):
    played = Counter(matches["HomeTeam"]) + Counter(matches["AwayTeam"])
    return {t for t in set(NAME_MAP.values()) if played[t] < threshold}

def fixtures_to_matches(fixtures):
    played = fixtures[fixtures["status"] == "FINISHED"].copy()
    played["MatchDate"] = played["utc"].dt.tz_localize(None)
    played["HomeTeam"] = played["home_r"]
    played["AwayTeam"] = played["away_r"]
    played["FTHome"] = played["fh"]
    played["FTAway"] = played["fa"]
    played["FTResult"] = played.apply(lambda r: "H" if r["fh"] > r["fa"] else ("A" if r["fh"] < r["fa"] else "D"), axis=1)
    return played[["MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway", "FTResult"]]

