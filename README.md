# Football Predictor
- Project by Alex K.
- Despite the name, this project is strictly for La Liga matches in the 2026-27 season. It predicts La Liga outcomes using two independent models, a closed-form statistical model and a trained machine learning classifier, benchmarked honestly against each other and against market and Elo baselines.

## Overview

I wanted to build something that combined a statistical approach with a machine learning approach, just to see which one performs better on my favorite team's league, La Liga. This is a personal project, and it tackles the same problem from two different angles.

- Poisson ratings model (football_predictor/model.py): fits time-decayed, shrinkage-regularized attack and defense ratings for every team from historical results, then predicts a full scoreline distribution, not just win, draw, or loss, using independent Poisson goal models. This is the original model behind the CLI and the Flask app.
- Logistic regression classifier (football_predictor/learned.py): trained on rolling form features like recent points, goal difference, rest days to predict the match outcome, evaluated with time aware cross validation.

These two models aren't competing for the same job. The Poisson model owns the full scoreline, while the classifier is benchmarked specifically on the win, draw, or loss outcome, the same thing market bookmakers and Elo ratings compete on. Both are scored against the same baselines, so the comparison stays honest instead of cherry picked.

## Results

I benchmarked both models on pooled, out-of-fold predictions across a 5-fold time aware walk forward split of about 4 seasons of La Liga matches (2022 to present). Lower is better for both metrics is what I found. Log loss and RPS (Ranked Probability Score) score the full predicted probability distribution, not just whether the top pick was correct. Below is a screenshot of the table results from my notebook:

<img width="447" height="167" alt="Screenshot 2026-09-16 at 4 41 37 PM" src="https://github.com/user-attachments/assets/a9da779a-f9af-41b6-9239-8c427c60e4f1" />

Both models comfortably beat the baseline of always guessing the historical win, draw, or loss split, and the Poisson model edges out the trained classifier, but both still trail Elo ratings and, especially, the bookmaker's own odds. This was expected and not a failure. Bookmaker odds have information neither model has access to, like injuries, lineup news, recent player form, etc. and Elo compresses a team's entire long-run history into one well-tuned number, while the classifier's features are a much shorter, noisier 5-match rolling window.

## Setup

- pip install -e .
- Requires Python 3.11+. You'll need a FOOTBALL_API_KEY environment variable (a free football-data.org API key upon a simple email signup) the first time app.py or season.py fetches live fixtures.

## Usage

CLI, predict a single match:
- python predict.py Barcelona "Real Madrid"
- python predict.py Barcelona "Real Madrid" --home "Real Madrid"

Web app, full season prediction table:
- python app.py

## Known limitations

- The classifier's feature set is limited to rolling form, goal difference, and rest days. No player-level data, injuries, or competition context yet, which is probably why it hasn't closed the gap to Elo and the bookmakers.
- No automated tests yet.
- The model isn't persisted with joblib, so it retrains from scratch every run.

## Possible next steps

- Add richer features that play into predicting match results such as head-to-head history, standings position, injuries.
- Persist the trained classifier and add a test suite.
