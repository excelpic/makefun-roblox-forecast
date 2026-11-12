# Predict Roblox's Up-and-Coming top 10 — entry

A forecaster for the make.fun bounty `roblox-up-and-coming-forecast-01`.
It reads the chart history in `/input` (`history.ndjson`, one read per line,
plus `task.json` for the scored window) and writes one static forecast to
`/output/forecast.json`: for each of the top 10 places, a probability for
each game named being at that place, and a guessed top 10 in order. The
forecast is scored by log score against every hourly read of the 72 hours
after the history ends.

## Run

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Deterministic: two runs on
the same history write byte-identical forecasts.

## How it works

Every game in the history gets a rank path: a recency-weighted straight line
of rank against time, its slope shrunk towards zero and capped, projected
across the scored window. Around the projected rank we spread a Gaussian
bell of uncertainty that widens with the horizon, sample the window every
six hours, and average the bell's mass on places 1–10. That average is the
game's raw share of each place.

On top of that:

- the raw shares are blended 50/50 with a plain kernel centred on each
  game's rank in the final read, so a noisy trend cannot drag the forecast
  away from where the chart actually stood;
- where the history's details give a game's creation date, its path stops
  adding mass once it is 28 days old — games age out of Up-and-Coming —
  which also shrinks its share of the window;
- games missing from the last read are damped, the more the longer they
  have been gone;
- a small floor, spread by a broad prior over plausible candidates, is
  mixed in, and each place keeps 10% of its mass unnamed. Log score punishes
  a confident miss far more than it rewards a confident hit, so no place is
  ever bet too narrowly.

The tunables at the top of `main.py` were chosen on backtest folds replayed
from the practice history (train up to a cut, forecast the reads that
followed, score with the official formula): pooled log score −39.8 against
the house persistence baseline's −50.9 on the same folds, ahead in every
fold. They are frozen in the file.

## License

MIT — see `LICENSE`.
