# BTC/USDT · 1-Hour Price Range Forecaster
### AlphaI × Polaris Build Challenge

> Predicts the 95% confidence interval for Bitcoin's next 1-hour close using GARCH(1,1)-t volatility modelling + Monte-Carlo GBM simulation.

**[Live Dashboard →](https://btc-forecast-production.up.railway.app/)** &nbsp;|&nbsp; **[Approach & Journey →](./APPROACH.md)**

---

## Results

| Metric | Value |
|---|---|
| Backtest Coverage (95%) | **95.56%** ✅ |
| Avg Range Width | $1,280 |
| Mean Winkler Score | $1,712 |
| Predictions (Part A) | 720 bars |
| Live Coverage | 100%  |
| Mean Live Winkler | $921-updating.... |

Backtest computed: `2026-05-02T12:23:27 UTC` over 720 hourly bars (Apr 2 – May 2, 2026).

---

## What It Does

Every hour, a new Bitcoin candle closes. This model predicts the price range where BTC will land at the *next* close — with 95% confidence.

```
Current price:  $78,363
Predicted range: $77,999 – $78,761  (next 1-hour close)
Confidence:      95%
```

The dashboard auto-refreshes every 5 minutes, always showing the latest prediction.

---

## Model

The core stack, in order of application:

```
1. Data         Binance public API (data-api.binance.vision, no key needed)
                → Last 500 1-hour BTCUSDT bars

2. Volatility   GARCH(1,1) with Student-t innovations
                → Fits per rolling window, EWMA fallback if GARCH fails to converge
                → Student-t df fitted per window (captures fat tails)

3. Scaling      Entropy-based crisis detection
                → Shannon entropy of log-return distribution
                → Scales sigma up (max 1.3×) when market regime turns chaotic
                → Quadratic mapping: stays near 1.0 in calm, ramps only in extremes

4. Simulation   Geometric Brownian Motion — 10,000 Monte-Carlo paths
                → dS = S · (μ dt + σ · Student-t noise · √dt)

5. Interval     2.5th and 97.5th percentile of simulated terminal prices
                → = 95% prediction interval
```

**Why GARCH over simple rolling std?**
Bitcoin has volatility clustering — calm hours cluster together, wild hours cluster together. GARCH has memory: it weights recent volatility more than old volatility. A simple rolling std treats last week's move the same as last hour's, which systematically under-reacts to regime changes.

**Why Student-t over Gaussian?**
Bitcoin has fat tails — extreme moves happen far more often than a normal bell curve predicts. A Gaussian model systematically sets intervals too narrow and misses too often. Student-t with fitted degrees of freedom handles this correctly.

---

## Project Structure

```
├── app.py                  # Streamlit dashboard (Part B + C)
├── model.py                # GARCH(1,1)-t + GBM forecaster
├── data.py                 # Binance data fetcher
├── backtest.py             # 720-bar walk-forward backtest (Part A)
├── backtest_results.jsonl  # Part A output — 720 predictions
├── backtest_summary.json   # Part A metrics summary (dashboard reads this)
├── predictions_seed.db     # Seed DB with initial live history
├── seed_db.py              # Seeds volume DB on first Railway deploy
├── requirements.txt
├── Procfile                # Railway start command
├── railway.toml            # Railway deployment config
├── .streamlit/
│   └── config.toml         # Streamlit server config
└── APPROACH.md             # Full journey — decisions, bugs, experiments
```

---

## Running Locally

```bash
git clone https://github.com/DEEP-600/BTC-forecast
cd BTC-forecast

pip install -r requirements.txt

# Run the dashboard
streamlit run app.py

# Re-run the backtest (takes ~2 min)
python backtest.py
```

No API keys needed. No `.env` file. Just Python.

---

## Dashboard Features

**Part A — Backtest metrics** shown at top: coverage, avg width, Winkler, prediction count.

**Live Performance section** — real-time stats from resolved predictions:
- Live coverage % with delta vs 95% target
- Mean live Winkler with delta vs backtest baseline
- Best single-bar Winkler
- Winkler-per-bar chart with backtest baseline reference line

**Next 1-Hour Prediction** — current price, lower/upper bounds, range width, countdown to next candle.

**Volatility regime indicator** — Calm / Elevated / Volatile, with live σ, Student-t df, and entropy scalar displayed.

**Candlestick chart** — last 50 bars + prediction band as shaded ribbon.

**Part C — Prediction History table** — full SQLite-backed history of all predictions with actual closes filled in as bars resolve, Inside? flag, and Winkler score per bar.

---

## Deployment

Deployed on **Railway** with a persistent Volume so the SQLite prediction history survives redeploys.

Key config:
- `Procfile`: `web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- `railway.toml`: start command + healthcheck path
- `RAILWAY_VOLUME_MOUNT_PATH` env var tells the app where the volume is mounted
- `RAILWAY_RUN_UID=0` set to handle volume permission correctly
- `seed_db.py` copies `predictions_seed.db` into the volume on first deploy only

---

## What I Changed From the Starter

The starter notebook ran GBM on daily USD/CHF data with a fixed rolling standard deviation and Gaussian noise. Here's what changed and why:

| Change | Why |
|---|---|
| Data source → Binance BTCUSDT 1-hour | Assignment requirement |
| Rolling std → GARCH(1,1) | Captures volatility clustering, reacts to regime changes |
| Gaussian → Student-t with fitted df | Bitcoin has fat tails; Gaussian misses too often |
| Fixed window → Per-window df fitting | df varies across regimes; fitting it live is more accurate |
| Added EWMA fallback | GARCH occasionally fails to converge; prevents crashes on degenerate windows |
| Added entropy scaling | Widens intervals proactively in chaotic regimes (max 1.3×, quadratic mapping) |
| Built Streamlit dashboard | Part B requirement |
| Added SQLite persistence | Part C bonus |
| Walk-forward backtest with zero leakage | Correct evaluation methodology |

Full reasoning for each decision is in [APPROACH.md](./APPROACH.md).

---

## Submission

- **coverage_95**: `0.9556`
- **mean_winkler_95**: `1712`
- **Dashboard**: https://btc-forecast-production.up.railway.app/
- **Repo**: https://github.com/DEEP-600/BTC-forecast

*Model: GARCH(1,1)-t + GBM · 10,000 Monte-Carlo paths · Student-t innovations · Entropy crisis scaling · Data: Binance data-api.binance.vision*