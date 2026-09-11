from flask import Flask, render_template, request
from football_predictor.season import season_table
from football_predictor.fixtures import NAME_MAP
app = Flask(__name__)

def row_class(status):
    if status == "FINISHED":
        return "finished"
    if status in ("IN_PLAY", "PAUSED"):
        return "live"
    return "upcoming"

@app.route("/")
def index():
    refresh = request.args.get("refresh") == "1"
    team = request.args.get("team") or None
    table = season_table(refresh=refresh, team = team)
    rows = table.to_dict(orient = "records")
    for r in rows:
        r["row_class"] = row_class(r["status"])
    teams = sorted(set(NAME_MAP.values()))
    return render_template("index.html", rows = rows, teams = teams, selected = team)


if __name__ == '__main__':
    app.run(debug=True)
