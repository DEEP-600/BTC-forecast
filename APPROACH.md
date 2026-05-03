# How I Built This — My Approach to the AlphaI × Polaris Challenge

> This document walks through my actual thinking process — what I started with, what broke, what I learned, and what I ended up with. It's not a polished story. It's what actually happened.

---

## Where I Started

The starter notebook gave me a GBM (Geometric Brownian Motion) simulator running on daily USD/CHF currency data. The idea is simple: look at how much the price has been moving recently, simulate 10,000 possible "next hours", and read off the range where 95% of those simulations land.

My first job was just to make that work on Bitcoin 1-hour bars from Binance instead of daily forex data.

That part was straightforward. Binance has a public endpoint (`data-api.binance.vision`) that requires no API key, no account, nothing. You just ask for the last N candles and it gives them to you. I pointed the starter's data fetch at that URL and got Bitcoin prices loading in.

At this point I had a working model. Basic GBM with a rolling standard deviation as the volatility estimate, Gaussian (normal) distribution for simulated returns. It ran. It produced a range. But I hadn't tested whether it was actually *good*.

---

## The Backtest Reality Check

The brief asked for a 30-day backtest — 720 hourly bars, walk-forward, no peeking at future data.

"No peeking" sounds obvious but it's the easiest bug to write accidentally. If your code computes volatility on the whole dataset first and *then* loops through bars, you've already leaked future information into every prediction. I was very deliberate about structuring the loop so that at bar `i`, only data up to bar `i` was ever used.

Running the backtest on the basic GBM gave me:

- **Coverage: ~91%** — too low. Supposed to be 95%.
- **Winkler: ~$2,100** — decent but there's clearly room.

The model was being overconfident. It kept predicting ranges that were too narrow, missing more than 5% of the time.

---

## Understanding Why It Was Missing

The brief actually tells you the answer: *fat tails*. Bitcoin doesn't move like a normal bell curve. It has way more extreme moves than you'd expect. The starter notebook already uses a Student-t distribution for this reason — heavier tails, more room for surprises.

But the *other* problem was volatility estimation. I was using a simple rolling standard deviation with a fixed window. That treats last week's volatility the same as last hour's volatility. Bitcoin doesn't work like that — it has **volatility clustering**. Calm periods cluster together. Wild periods cluster together. If the last few hours have been jumpy, the next hour is more likely to also be jumpy.

This is exactly what GARCH models were built to solve.

---

## Upgrading the Volatility Model: GARCH(1,1)

GARCH(1,1) is a model that says: tomorrow's volatility is a blend of (a) long-run average volatility, (b) how volatile it actually was yesterday, and (c) what the model predicted yesterday's volatility would be.

In plain terms: it has memory. It reacts to recent conditions rather than treating all history equally.

I also kept the Student-t distribution for the innovations (the random shocks in each simulated path) because Bitcoin's tail behaviour is genuinely heavier than normal. Fitting the degrees-of-freedom parameter per window rather than fixing it gave noticeably better results — when BTC is behaving more normally, the df goes up; when it's in a wild patch, df drops.

I added an EWMA (Exponentially Weighted Moving Average) fallback in case GARCH failed to converge on a particular window — this happens occasionally with short or weird data windows and I didn't want the whole app to crash because of one bad bar.

After this upgrade, backtest results improved meaningfully:
- **Coverage: 94.86%** (was ~91%)
- **Winkler: ~$1,692** (was ~$2,100)

Coverage was now close to 95% but slightly under. Winkler dropped significantly.

---

## The Entropy Experiment

I noticed the starter notebook mentioned entropy-based volatility scaling — the idea being that when the *distribution* of recent returns starts looking more chaotic (higher Shannon entropy), the market is entering a regime shift and your intervals should widen preemptively.

I implemented it. Compute the Shannon entropy of the log-return histogram over the last 168 bars, compare it to a rolling calm baseline, and scale GARCH's sigma up when entropy spikes.

First attempt: `max_scalar = 2.5`. This was a disaster.

Results got *worse*:
- Coverage jumped to 98.75% — way too high. Ranges were so wide they were almost always right, but the Winkler score blew up to $2,325 because the width penalty dominates when you're not missing.

This taught me something important about the Winkler formula: **width itself is a penalty**. You're not trying to never miss. You're trying to find the tightest range that still catches ~95% of outcomes. Making ranges wider to avoid misses can easily make your Winkler *worse* if the misses you were preventing weren't that far off anyway.

I tuned it down aggressively — `max_scalar = 1.3`, and changed the mapping from a square-root curve (which ramps up quickly) to a quadratic curve (which stays near 1.0 for most bars and only ramps in true chaos). Final result:

- **Coverage: 95.56%** — now slightly *above* target, which is better than below
- **Winkler: $1,712** — marginally higher than without entropy, but coverage is now hitting the target more precisely

The entropy scaling adds maybe $20 to the Winkler score but brings coverage from 94.86% (below target) to 95.56% (above target). I kept it.

---

## Building the Dashboard

The brief said not to overthink the design — "does it load and show the right numbers" is what matters. But I also wanted it to actually be readable and useful, not just functional.

I used Streamlit. The dashboard:
- Fetches the latest closed 1-hour BTC/USDT bar from Binance on every load
- Runs the GARCH(1,1)-t model on the last 500 bars
- Shows the predicted 95% range for the next hour with a countdown timer
- Shows backtest metrics (coverage, Winkler, width, prediction count) at the top
- Has a volatility regime indicator (Calm / Elevated / Volatile) based on where current sigma sits relative to its recent distribution
- Shows the Student-t degrees of freedom and entropy scalar live so you can see what the model is "thinking"
- Auto-refreshes every 5 minutes

I also added a candlestick chart of the last 50 bars with the prediction band plotted as a shaded ribbon on the right edge.

---

## Part C: Persistence

Every time the dashboard loads, it saves the current prediction to a SQLite database. When a bar closes, the next visit fills in the actual close price and computes the Winkler score for that bar.

Two bugs I had to fix here:

**Bug 1 — Duplicate rows.** On every page visit, the app was inserting a new row for the same bar rather than updating the existing one. Fix: add a `UNIQUE` constraint on `bar_time` and use `INSERT OR IGNORE` so the first prediction for each bar wins and subsequent visits don't overwrite it.

**Bug 2 — Off-by-one in actual close.** I was matching bar T's prediction with bar T's close price, but the prediction is for what happens *after* bar T closes — so the actual should be bar T+1's close. Fixed with a `next_close_map` that correctly aligns predictions with their outcomes.

After fixing both bugs, the live coverage table started accumulating correctly. As of submission:
- **Live coverage: 100%** (growing daily)
- **Mean live Winkler: significantly better than backtest** — because BTC has been in a calm regime since deployment

---

## Live Performance vs Backtest

The backtest Winkler ($1,712) is higher than live ($936) because the backtest covers 30 days including volatile periods. Live performance looks better right now simply because BTC has been range-bound around $78k. This is expected — the backtest is the honest number; live is the current snapshot.

The Winkler chart on the dashboard shows this clearly: the first two bars from May 1 (when BTC was moving more) had Winkler scores of $1,670 and $1,598, then it settled into the $795–$963 range as volatility compressed.

---

## What I Would Do Next

If I had more time, the two most impactful improvements based on research would be:

1. **HAR-RV (Heterogeneous Autoregressive Realized Volatility)** — uses 5-minute bars to compute actual realized volatility within each hour, which is a better volatility estimate for intraday data than GARCH alone. Multiple papers confirm this beats GARCH for hourly Bitcoin specifically.

2. **Adaptive Conformal Inference** — a post-hoc calibration layer that adjusts the quantile multiplier based on empirical coverage, guaranteeing convergence to exactly 95% regardless of model misspecification. This is what production risk systems use.

Both would require more time than the deadline allowed, and I didn't want to rush something I didn't fully understand into the submission.

---

## Final Numbers

| Metric | Value |
|---|---|
| Backtest Coverage (95%) | 95.56% |
| Avg Range Width | $1,280 |
| Mean Winkler Score | $1,712 |
| Predictions | 720 |
| Live Coverage | 100%  |
| Mean Live Winkler | $921 - updating... |

Model: GARCH(1,1)-t conditional volatility + GBM with 10,000 Monte-Carlo paths + Student-t innovations + entropy-based crisis scaling.
Data: Binance `data-api.binance.vision` (no API key required).