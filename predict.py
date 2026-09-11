import argparse
from football_predictor.data import load_matches
from football_predictor.model import fit_ratings, predict, FINAL

def main():
    parser = argparse.ArgumentParser(description = "Predict a La Liga score for a match")
    parser.add_argument("team1")
    parser.add_argument("team2")
    parser.add_argument("--home", help ="which team is at home (default: team1)")
    args = parser.parse_args()

    if args.home:
        if args.home not in (args.team1, args.team2):
            parser.error(f"--home must be one of {args.team1!r}, {args.team2!r}")
        home = args.home
        away = args.team2 if args.home == args.team1 else args.team1
    else:
        home, away = args.team1, args.team2

    matches = load_matches()
    model = fit_ratings(matches, **FINAL, ref_date = matches["MatchDate"].max())
    result = predict(model, home, away)
    if result is None:
        parser.error(f"unknown team: {home!r} or {away!r}")

    h, a = result["Score"]
    print(f"{home} {h} - {a} {away}")
    print(f"H {result['H']:.2f} D {result['D']:.2f} A {result['A']:.2f}")

if __name__ == "__main__":
    main()

