"""Forecaster for Roblox's Up-and-Coming top 10 (make.fun program/v1 entry).

Reads the chart history in $MAKEFUN_INPUT_DIR (default /input): history.ndjson
(one read per line, oldest first) and task.json (the scored window). Writes one
static forecast to $MAKEFUN_OUTPUT_DIR (default /output): for each of the top
10 places, a probability for each game named being at that place, plus the
guessed top 10 in order. The same forecast is scored against every hourly
read of the 72 hours after the history ends, by log score, so what we want is
each game's honest share of each place over the whole window.

Model, in plain terms:

* Every game gets a rank path. From its appearances in the history we fit a
  recency-weighted straight line of rank against time (recent reads count
  more), shrink the slope towards zero (a few reads prove little), and project
  that line across the scored window.
* A game's future place is not a point but a smear: at each sampled hour of
  the window we spread a Gaussian bell of rank uncertainty around the
  projected rank, wider the further ahead we look, and average the mass that
  falls on places 1..10 over the window. That average is the game's raw share
  of each place.
* Where a read's details name a game's creation date, the game leaves the
  chart 28 days after publication, so its path adds no mass at samples after
  that moment, which also shrinks its share of the whole window.
* Games absent from the last read are damped: leaving the chart is sticky,
  and a quick return to the top 10 is rare.
* A floor of probability is mixed in, spread by a broad prior over every
  plausible candidate, because log score punishes a confident miss far more
  than it rewards a confident hit; and each place keeps a slice of its mass
  unnamed, which the scorer spreads over the whole chart of the read, as
  insurance against games nobody could have named.

Deterministic: no randomness, no clock, no network, standard library only.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

# --------------------------------------------------------------------------
# Tunables. Chosen on backtest folds replayed from the practice history
# (train up to a cut, forecast the reads that followed), then frozen.
# --------------------------------------------------------------------------
PARAMS = {
    "recency_tau_h": 18.0,    # e-folding age for weighting observations
    "slope_shrink": 6.0,      # pseudo-weight pulling a rank slope to zero
    "slope_cap_per_h": 0.6,   # fastest sustained climb/fall we will project
    "intercept_blend": 0.55,  # weight of the raw last rank vs the fitted one
    "sigma0": 3.0,            # rank uncertainty (places) at the window's start
    "sigma_growth": 0.35,     # extra uncertainty per sqrt(hour ahead)
    "absent_damp": 0.25,      # share kept by a game missing from the last read
    "stale_tau_h": 72.0,      # extra damping by hours since a game was last seen
    "floor_level": 0.005,     # floor mass mixed in, scaled by the broad prior
    "named_sum": 0.90,        # total named probability per place (rest unnamed)
    "sample_step_h": 6.0,     # spacing of the window samples along the paths
    "play_pull": 0.0,         # ranks/hour of climb credited per unit of
                              # log-playing growth per hour (0 = off; it
                              # backtested as noise, kept for the record)
    "lastker_mix": 0.5,       # share of the last-chart kernel in the raw mix
    "lastker_sigma0": 3.0,    # the kernel's uncertainty at the window's start
    "lastker_growth": 0.6,    # the kernel's uncertainty growth per sqrt(hour)
}

SECONDS_PER_HOUR = 3600.0
ELIGIBILITY_DAYS = 28.0  # games leave Up-and-Coming 28 days after publication


def parse_time(text):
    """ISO 8601 UTC text -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, sum_w)."""
    sw = sum(ws)
    if sw <= 0.0:
        return (ys[-1] if ys else 0.0), 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-9:
        return my, 0.0, sw
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return my - b * mx, b, sw


def game_paths(reads, params):
    """Fit one rank path per game.

    reads: list of read dicts, oldest first, each with 'at' and
    'up_and_coming'. Returns (paths, last_chart_ids, created): paths maps
    universe_id -> (rank_at_end, slope_per_hour, hours_since_last_seen);
    created maps universe_id -> publish datetime from the most recent
    details that carry one (games without details are absent).
    """
    p = params
    t_end = parse_time(reads[-1]["at"])
    obs = {}        # universe_id -> list of (hours_rel_end, rank); rel <= 0
    play = {}       # universe_id -> list of (hours_rel_end, log playing)
    last_seen = {}  # universe_id -> hours before the end of its last sighting
    created = {}    # universe_id -> publish datetime (latest details win)
    for read in reads:
        t = parse_time(read["at"])
        age_h = (t_end - t).total_seconds() / SECONDS_PER_HOUR
        for row in read.get("up_and_coming") or []:
            uid = row["universe_id"]
            obs.setdefault(uid, []).append((-age_h, float(row["rank"])))
            if row.get("playing"):
                play.setdefault(uid, []).append((-age_h, math.log(row["playing"])))
            last_seen[uid] = age_h
        details = read.get("details") or {}
        if isinstance(details, dict):
            for uid_text, det in details.items():
                if isinstance(det, dict) and det.get("created"):
                    try:
                        created[int(uid_text)] = parse_time(det["created"])
                    except (ValueError, TypeError):
                        pass
    last_chart = {row["universe_id"]
                  for row in reads[-1].get("up_and_coming") or []}

    paths = {}
    for uid, points in obs.items():
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        ws = [math.exp(x / p["recency_tau_h"]) for x in xs]  # newer weighs more
        a, b, sw = weighted_line(xs, ys, ws)
        if len(points) == 1:
            a, b = ys[0], 0.0
        shrink = sw / (sw + p["slope_shrink"])
        b = b * shrink
        # Player-count momentum: the chart is sorted by user growth, so a
        # game whose crowd is swelling tends to keep climbing even where its
        # rank line alone looks flat. Credit its rank slope accordingly.
        if p["play_pull"] and uid in play and len(play[uid]) >= 2:
            pxs = [pt[0] for pt in play[uid]]
            pys = [pt[1] for pt in play[uid]]
            pws = [math.exp(x / p["recency_tau_h"]) for x in pxs]
            _pa, pb, psw = weighted_line(pxs, pys, pws)
            pb *= psw / (psw + p["slope_shrink"])
            b -= p["play_pull"] * pb
        b = max(-p["slope_cap_per_h"], min(p["slope_cap_per_h"], b))
        last_rank = points[-1][1]
        a = (1.0 - p["intercept_blend"]) * a + p["intercept_blend"] * last_rank
        a = max(1.0, min(140.0, a))
        paths[uid] = (a, b, last_seen[uid])
    return paths, last_chart, created


def build_forecast(reads, window_from, window_until, params=None):
    """Build the forecast dict for a history and a scored window."""
    p = dict(PARAMS)
    if params:
        p.update(params)
    t_end = parse_time(reads[-1]["at"])
    places = 10

    paths, last_chart, created = game_paths(reads, p)

    # Broad prior for the floor: games near the top of the last chart, and
    # recently seen games generally, are the plausible surprise holders.
    prior = {}
    for uid, (a, _b, seen_age) in paths.items():
        base = math.exp(-(a - 1.0) / 45.0)
        if uid not in last_chart:
            base *= 0.4
        prior[uid] = base * math.exp(-seen_age / 96.0)

    # Sample the window; each game's bell of rank uncertainty, averaged over
    # the samples, is its raw share of every place. Samples at which a game
    # has aged out add nothing, so its average (taken over all samples) is
    # already its share of the whole window.
    total_h = (window_until - window_from).total_seconds() / SECONDS_PER_HOUR
    n_steps = max(1, int(math.ceil(total_h / p["sample_step_h"])))
    sample_times = [window_from + timedelta(hours=total_h * (i + 0.5) / n_steps)
                    for i in range(n_steps)]
    raw = {uid: [0.0] * places for uid in paths}
    for uid, (a, b, seen_age) in paths.items():
        damp = 1.0
        if uid not in last_chart:
            damp = p["absent_damp"] * math.exp(-seen_age / p["stale_tau_h"])
        age_out = None
        if uid in created:
            age_out = created[uid] + timedelta(days=ELIGIBILITY_DAYS)
        acc = raw[uid]
        for t in sample_times:
            if age_out is not None and t >= age_out:
                continue
            s = (t - t_end).total_seconds() / SECONDS_PER_HOUR
            r = max(1.0, min(160.0, a + b * s))
            sigma = p["sigma0"] + p["sigma_growth"] * math.sqrt(max(0.0, s))
            for k in range(1, places + 1):
                acc[k - 1] += (phi((k + 0.5 - r) / sigma)
                               - phi((k - 0.5 - r) / sigma))
        for k in range(places):
            acc[k] = acc[k] / float(n_steps) * damp

    # Optional blend with a plain last-chart kernel: the same window
    # sampling, but centred on each game's rank in the final read instead of
    # its fitted path. The path model can over-read a noisy trend; the kernel
    # is the humble version, and a mixture hedges the two.
    if p["lastker_mix"]:
        m = p["lastker_mix"]
        last_rank = {}
        for row in reads[-1].get("up_and_coming") or []:
            last_rank[row["universe_id"]] = float(row["rank"])
        for uid, r0 in last_rank.items():
            ker = [0.0] * places
            for t in sample_times:
                s = (t - t_end).total_seconds() / SECONDS_PER_HOUR
                sig = p["lastker_sigma0"] + p["lastker_growth"] * math.sqrt(max(0.0, s))
                for k in range(1, places + 1):
                    ker[k - 1] += (phi((k + 0.5 - r0) / sig)
                                   - phi((k - 0.5 - r0) / sig))
            acc = raw.setdefault(uid, [0.0] * places)
            for k in range(places):
                acc[k] = (1.0 - m) * acc[k] + m * ker[k] / float(n_steps)
        for uid, acc in raw.items():
            if uid not in last_rank:
                for k in range(places):
                    acc[k] *= (1.0 - m)

    # Mix the floor in, scaled by the broad prior, then normalise each place
    # to the named sum; the remainder stays unnamed.
    mixed = {}
    totals = [0.0] * places
    for uid, acc in raw.items():
        row = [acc[k] + p["floor_level"] * prior.get(uid, 0.0)
               for k in range(places)]
        mixed[uid] = row
        for k in range(places):
            totals[k] += row[k]
    probabilities = []
    for uid, row in mixed.items():
        for k in range(places):
            if totals[k] > 0.0:
                val = p["named_sum"] * row[k] / totals[k]
                if val >= 0.0005:
                    probabilities.append({"universe_id": uid, "place": k + 1,
                                          "p": round(val, 6)})
    # At most 100 rows per place (the rules' cap); keep the largest.
    probabilities.sort(key=lambda r: -r["p"])
    per_place = {}
    trimmed = []
    for row in probabilities:
        n = per_place.get(row["place"], 0)
        if n < 100:
            trimmed.append(row)
            per_place[row["place"]] = n + 1
    probabilities = trimmed

    # Top 10 guess: a game's places weighted by 11 - place over its shares.
    scored = []
    for uid, row in mixed.items():
        s = sum(row[k] * (places - k) for k in range(places))
        scored.append((-s, paths[uid][0], uid))
    scored.sort()
    top10 = [uid for _s, _a, uid in scored[:places]]
    if len(top10) < places:  # pad from the last chart, in its order
        for row in sorted(reads[-1].get("up_and_coming") or [],
                          key=lambda r: r["rank"]):
            if row["universe_id"] not in top10:
                top10.append(row["universe_id"])
            if len(top10) == places:
                break
    return {"probabilities": probabilities, "top10": top10[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda r: r["at"])
    window = task["window"]
    forecast = build_forecast(reads, parse_time(window["from"]),
                              parse_time(window["until"]))
    os.makedirs(output_dir, exist_ok=True)
    out_name = task.get("output", "forecast.json")
    with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
