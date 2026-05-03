"""
backtest.py — 30-day walk-forward backtest for the AlphaI × Polaris challenge

Strategy
--------
• Fetch 720 TEST bars  (≈ 30 days of hourly BTCUSDT)
  + LOOKBACK_BARS of warm-up data that precede the test window.
• For each bar i in [0, 720):
    - Use only bars[0 : LOOKBACK_BARS + i]  →  NO peeking at bar i+1.
    - Fit GARCH(1,1)-t on the last ROLLING_WINDOW log-returns.
    - Simulate 10 000 GBM paths, read off [2.5%, 97.5%] as the 95% range.
    - Record (bar_time, lower_95, upper_95, actual_close_of_bar_i+1).
• Save all 720 records to backtest_results.jsonl.
• Save summary metrics to backtest_summary.json for the dashboard.

No-peeking guarantee
--------------------
At step i the slice ends at index (LOOKBACK_BARS + i), so the ACTUAL close
of bar (i+1) is NEVER visible when we make the prediction.  This is enforced
by construction — there is no conditional branch or index arithmetic where
future data could accidentally leak in.

Run
---
    python backtest.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data  import fetch_klines
from model import predict_95_range, evaluate_predictions

# ── Config ───────────────────────────────────────────────────────────────────
TEST_BARS     = 720       # ≈ 30 days
LOOKBACK_BARS = 500       # warm-up data before test window (≈ 21 days)
ROLLING_WIN   = 500       # GARCH rolling window within lookback
N_SIMS        = 10_000
OUTPUT_FILE   = Path("backtest_results.jsonl")
SUMMARY_FILE  = Path("backtest_summary.json")


def run_backtest() -> list[dict]:
    total_bars = LOOKBACK_BARS + TEST_BARS + 1   # +1 so bar TEST_BARS has an actual

    print(f"[1/3] Fetching {total_bars} hours of BTCUSDT data from Binance …")
    df = fetch_klines(n_bars=total_bars)

    if len(df) < TEST_BARS + 2:
        raise RuntimeError(
            f"Only got {len(df)} bars — need at least {TEST_BARS + 2}."
        )

    closes     = df["close"].values          # shape (total_bars,)
    timestamps = df.index.tolist()

    print(f"    → Got {len(closes)} bars "
          f"({timestamps[0]} … {timestamps[-1]})")
    print(f"[2/3] Running {TEST_BARS}-step walk-forward backtest …")
    print("    (GARCH is refitted on every step — this takes ~3–5 min)\n")

    records = []
    for i in tqdm(range(TEST_BARS), unit="bar", ncols=72):
        # ── Prediction step ──────────────────────────────────────────────────
        # Available data: closes[0 : LOOKBACK_BARS + i]
        # We predict the close of closes[LOOKBACK_BARS + i + 1] (bar i+1 in test)
        history_end  = LOOKBACK_BARS + i          # EXCLUSIVE upper bound on history
        actual_index = LOOKBACK_BARS + i          # the bar whose close we predict
        # Wait — let's be precise:
        #   closes[LOOKBACK_BARS + i]     = the bar AT step i (the "current" bar)
        #   closes[LOOKBACK_BARS + i + 1] = the bar we are predicting
        #   history slice for fitting: closes[0 : LOOKBACK_BARS + i + 1]
        #   (includes the current bar's close, does NOT include the next bar)

        history = closes[: LOOKBACK_BARS + i + 1]   # ← no peeking

        # Limit GARCH window to ROLLING_WIN most recent returns
        window = history[-ROLLING_WIN:]

        result = predict_95_range(window, n_sims=N_SIMS)

        actual_close = float(closes[LOOKBACK_BARS + i + 1])   # revealed after prediction

        record = {
            "bar_time":       timestamps[LOOKBACK_BARS + i].isoformat(),
            "lower_95":       result["lower_95"],
            "upper_95":       result["upper_95"],
            "actual_price":   actual_close,
            "current_price":  result["current_price"],
            "sigma":          result["sigma"],
            "mu":             result["mu"],
            "t_df":           result["t_df"],
            "entropy_scalar": result["entropy_scalar"],
            "inside":         result["lower_95"] <= actual_close <= result["upper_95"],
        }
        records.append(record)

    return records


def save_results(records: list[dict]) -> dict:
    print(f"\n[3/3] Saving results …")

    OUTPUT_FILE.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    print(f"    → {OUTPUT_FILE}  ({len(records)} lines)")

    metrics = evaluate_predictions(records)

    # Round for readability
    summary = {
        "coverage_95":     round(metrics["coverage_95"],     4),
        "mean_width":      round(metrics["mean_width"],      2),
        "mean_winkler_95": round(metrics["mean_winkler_95"], 2),
        "n_predictions":   metrics["n_predictions"],
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(f"    → {SUMMARY_FILE}")

    return summary


def print_report(summary: dict):
    w = 54
    print("\n" + "═" * w)
    print("  AlphaI × Polaris — Backtest Report")
    print("═" * w)
    print(f"  Predictions     : {summary['n_predictions']}")
    print(f"  Coverage (95%)  : {summary['coverage_95']:.4f}  "
          f"{'✓ good' if abs(summary['coverage_95'] - 0.95) < 0.03 else '⚠ off'}")
    print(f"  Avg Range Width : ${summary['mean_width']:,.0f}")
    print(f"  Winkler Score   : ${summary['mean_winkler_95']:,.0f}  (lower = better)")
    print("═" * w)
    print()


if __name__ == "__main__":
    t0      = time.time()
    records = run_backtest()
    summary = save_results(records)
    print_report(summary)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s")
    print(f"\n  Submit these numbers in the form:")
    print(f"    coverage_95     = {summary['coverage_95']}")
    print(f"    mean_winkler_95 = {summary['mean_winkler_95']}")