"""
app.py — AlphaI × Polaris BTC Forecast Dashboard

Live dashboard that:
  1. Fetches the latest closed BTCUSDT 1-hour bar from Binance
  2. Runs GARCH(1,1)-t + GBM on the last 500 bars
  3. Displays the current price and 95% prediction range for the next hour
  4. Shows a Plotly candlestick chart of the last 50 bars with a shaded ribbon
  5. Shows Part-A backtest metrics at the top
  6. Part C: persists every prediction to SQLite and shows a growing history table

Deploy:
    pip install -r requirements.txt
    streamlit run app.py

Host on Streamlit Community Cloud (free, stays alive):
    1. Push this repo to GitHub (public)
    2. Go to share.streamlit.io → New app → select repo → main file: app.py
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from data  import fetch_klines
from model import predict_95_range, winkler_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Forecast | AlphaI × Polaris",
    page_icon="₿",
    layout="wide",
)

# Auto-refresh every 5 minutes (next candle logic)
st_autorefresh(interval=5 * 60 * 1000, key="auto_refresh")

# ── Constants ─────────────────────────────────────────────────────────────────
LOOKBACK        = 500          # bars used for model fitting
CHART_BARS      = 50           # bars shown on chart
SUMMARY_FILE    = Path("backtest_summary.json")

# DB path: Railway volume (/data) if available, else local
VOLUME_DB = "/data/predictions.db"
LOCAL_DB  = os.path.join(os.path.dirname(__file__), "predictions.db")
DB_FILE   = Path(VOLUME_DB if os.path.exists("/data") else LOCAL_DB)

# Auto-seed volume DB on first Railway deploy
if os.path.exists("/data") and not os.path.exists(VOLUME_DB):
    import seed_db
    seed_db.seed()


# ── SQLite persistence (Part C) ───────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at    TEXT,
            bar_time      TEXT UNIQUE,
            current_price REAL,
            lower_95      REAL,
            upper_95      REAL,
            sigma         REAL,
            t_df          REAL,
            entropy_scalar REAL,
            actual_price  REAL,
            inside        INTEGER
        )
    """)
    con.commit()
    return con


def save_prediction(con, pred: dict, bar_time: str):
    con.execute("""
        INSERT OR IGNORE INTO predictions
          (fetched_at, bar_time, current_price, lower_95, upper_95, sigma, t_df, entropy_scalar)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        bar_time,
        pred["current_price"],
        pred["lower_95"],
        pred["upper_95"],
        pred["sigma"],
        pred["t_df"],
        pred["entropy_scalar"],
    ))
    con.commit()


def fill_actuals(con, df_prices: pd.DataFrame):
    """
    Back-fill actual_price for past predictions once the NEXT bar has closed.
    A prediction made at bar_time T is predicting the close of bar T+1.
    So we look up the bar that comes AFTER bar_time in the price series.
    """
    rows = pd.read_sql("SELECT id, bar_time FROM predictions WHERE actual_price IS NULL", con)
    if rows.empty:
        return

    # Build a map: bar_time → next bar's close
    times  = list(df_prices.index)
    closes = list(df_prices["close"])
    next_close_map = {}
    for i in range(len(times) - 1):
        next_close_map[times[i].isoformat()] = closes[i + 1]

    for _, row in rows.iterrows():
        actual = next_close_map.get(row["bar_time"])
        if actual is not None:
            lo, hi = con.execute(
                "SELECT lower_95, upper_95 FROM predictions WHERE id=?", (row["id"],)
            ).fetchone()
            inside = int(lo <= actual <= hi)
            con.execute(
                "UPDATE predictions SET actual_price=?, inside=? WHERE id=?",
                (actual, inside, row["id"]),
            )
    con.commit()


def load_history(con, limit: int = 200) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT * FROM predictions ORDER BY bar_time DESC LIMIT ?",
        con,
        params=(limit,),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)   # re-fetch at most every 5 min
def get_data(n: int) -> pd.DataFrame:
    return fetch_klines(n_bars=n)


def load_backtest_summary() -> dict | None:
    if SUMMARY_FILE.exists():
        return json.loads(SUMMARY_FILE.read_text())
    return None


def vol_regime_label(sigma: float) -> tuple[str, str]:
    """Map hourly sigma (log-return) to a human label + colour."""
    annualised = sigma * (24 * 365) ** 0.5   # rough annualised vol
    if annualised < 0.50:
        return "🟢 Calm", "#2ecc71"
    elif annualised < 1.00:
        return "🟡 Normal", "#f1c40f"
    else:
        return "🔴 Volatile", "#e74c3c"


def minutes_to_next_close() -> int:
    """Minutes remaining until the top of the next UTC hour."""
    now = datetime.now(timezone.utc)
    return 59 - now.minute


# ── Main dashboard ────────────────────────────────────────────────────────────
def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:0'>₿ BTC/USDT · 1-Hour Forecast</h1>"
        "<p style='color:#888;margin-top:0'>AlphaI × Polaris Challenge · GARCH(1,1)-t + GBM Model</p>",
        unsafe_allow_html=True,
    )

    # ── Load Part-A summary ───────────────────────────────────────────────────
    summary = load_backtest_summary()

    if summary:
        c1, c2, c3, c4 = st.columns(4)
        coverage = summary["coverage_95"]
        cov_delta = f"{(coverage - 0.95)*100:+.2f}pp vs target 95%"
        c1.metric("Backtest Coverage (95%)", f"{coverage:.2%}", cov_delta)
        c2.metric("Avg Range Width",  f"${summary['mean_width']:,.0f}")
        c3.metric("Mean Winkler Score", f"${summary['mean_winkler_95']:,.0f}", "lower = better")
        c4.metric("Predictions", f"{summary['n_predictions']:,}")
        st.caption(f"Part-A backtest computed at {summary.get('generated_at','N/A')}")
    else:
        st.info(
            "Backtest summary not found. Run `python backtest.py` once "
            "to generate `backtest_summary.json`."
        )

    st.divider()

    # ── Live Performance (resolved bars) ────────────────────────────────────────
    st.subheader("📡 Live Performance (Resolved Bars)")

    # Open a read-only DB connection for live performance metrics
    con_live = sqlite3.connect(DB_FILE)
    resolved = pd.read_sql(
        "SELECT * FROM predictions WHERE actual_price IS NOT NULL ORDER BY bar_time ASC",
        con_live,
    )

    if len(resolved) < 3:
        st.info("⏳ Waiting for more resolved predictions…")
    else:
        # Compute per-row winkler
        resolved["winkler"] = resolved.apply(
            lambda r: winkler_score(r["lower_95"], r["upper_95"], r["actual_price"]),
            axis=1,
        )
        live_coverage = resolved["inside"].mean()
        mean_live_winkler = resolved["winkler"].mean()
        best_winkler = resolved["winkler"].min()
        n_resolved = len(resolved)

        st.caption(
            f"Live coverage over {n_resolved} resolved predictions: "
            f"**{live_coverage:.2%}** (target 95%)"
        )

        # Backtest baseline for deltas
        bt_winkler = summary["mean_winkler_95"] if summary else None

        cov_delta = f"{(live_coverage - 0.95)*100:+.1f}pp vs 95%"
        cov_delta_color = "normal" if live_coverage >= 0.95 else "inverse"

        winkler_delta = None
        winkler_delta_label = ""
        if bt_winkler is not None:
            winkler_delta = bt_winkler - mean_live_winkler
            winkler_delta_label = f"vs backtest ${bt_winkler:,.0f}"

        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Live Coverage", f"{live_coverage:.1%}", cov_delta, delta_color=cov_delta_color)
        lc2.metric("Mean Live Winkler", f"${mean_live_winkler:,.0f}", f"+${winkler_delta:,.0f} {winkler_delta_label}" if winkler_delta is not None else None, delta_color="normal")
        lc3.metric("Best Winkler Bar", f"${best_winkler:,.0f}")
        lc4.metric("Resolved Bars", f"{n_resolved}")

        # Winkler per bar line chart
        fig_live = go.Figure()

        # Color dots by inside/outside
        colors = ["#2ecc71" if ins == 1 else "#e74c3c" for ins in resolved["inside"]]

        fig_live.add_trace(go.Scatter(
            x       = resolved["bar_time"],
            y       = resolved["winkler"],
            mode    = "lines+markers",
            marker  = dict(color=colors, size=7),
            line    = dict(color="#00d4ff", width=1.5),
            name    = "Winkler",
        ))

        # Backtest baseline
        if bt_winkler is not None:
            fig_live.add_hline(
                y               = bt_winkler,
                line_dash       = "dash",
                line_color      = "#f39c12",
                annotation_text = f"Backtest baseline ${bt_winkler:,.0f}",
                annotation_position = "right",
            )

        # 95th percentile of live winklers
        pct95 = float(np.percentile(resolved["winkler"], 95))
        fig_live.add_hline(
            y               = pct95,
            line_dash       = "dot",
            line_color      = "#e74c3c",
            annotation_text = f"95th pct ${pct95:,.0f}",
            annotation_position = "right",
        )

        fig_live.update_layout(
            title       = "Winkler Score per Resolved Bar",
            xaxis_title = "Bar Time (UTC)",
            yaxis_title = "Winkler Score",
            template    = "plotly_dark",
            height      = 380,
            showlegend  = False,
            margin      = dict(l=10, r=10, t=50, b=10),
        )

        st.plotly_chart(fig_live, use_container_width=True)

    con_live.close()

    st.divider()

    # ── Fetch live data ───────────────────────────────────────────────────────
    with st.spinner("Fetching latest data from Binance …"):
        try:
            df = get_data(LOOKBACK + 1)
        except Exception as exc:
            st.error(f"Binance API error: {exc}")
            st.stop()

    # Run model on last LOOKBACK bars
    prices   = df["close"].values[-LOOKBACK:]
    bar_time = df.index[-1].isoformat()

    with st.spinner("Running GARCH(1,1)-t model …"):
        pred = predict_95_range(prices)

    # ── Persist prediction (Part C) ───────────────────────────────────────────
    con = init_db()
    save_prediction(con, pred, bar_time)
    fill_actuals(con, df)

    # ── Live prediction cards ─────────────────────────────────────────────────
    lo, hi      = pred["lower_95"], pred["upper_95"]
    curr        = pred["current_price"]
    width       = hi - lo
    mins_left   = minutes_to_next_close()
    vol_label, vol_color = vol_regime_label(pred["sigma"])

    st.subheader("Next 1-Hour Prediction")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current BTC Price",  f"${curr:,.2f}")
    col2.metric("Lower Bound (2.5%)", f"${lo:,.2f}")
    col3.metric("Upper Bound (97.5%)",f"${hi:,.2f}")
    col4.metric("Range Width",        f"${width:,.2f}")
    col5.metric("Next candle in",     f"{mins_left} min")

    st.markdown(
        f"<p style='color:{vol_color};font-size:1.1rem'>"
        f"Volatility Regime: <b>{vol_label}</b> "
        f"&nbsp;|&nbsp; Hourly σ = {pred['sigma']*100:.4f}% "
        f"&nbsp;|&nbsp; Student-t df = {pred['t_df']:.1f} "
        f"&nbsp;|&nbsp; Entropy scalar = {pred['entropy_scalar']:.2f}x"
        f"</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='background:#1a1a2e;border-radius:8px;padding:16px;"
        f"text-align:center;font-size:1.6rem;font-weight:bold;color:#00d4ff'>"
        f"95% confident BTC will be between "
        f"<span style='color:#2ecc71'>${lo:,.0f}</span> and "
        f"<span style='color:#e74c3c'>${hi:,.0f}</span> at the next close"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Candlestick chart + ribbon ────────────────────────────────────────────
    chart_df = df.iloc[-CHART_BARS:]

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x       = chart_df.index,
        open    = chart_df["open"],
        high    = chart_df["high"],
        low     = chart_df["low"],
        close   = chart_df["close"],
        name    = "BTCUSDT",
        increasing_line_color = "#2ecc71",
        decreasing_line_color = "#e74c3c",
    ))

    # Prediction ribbon for the NEXT bar
    next_time = chart_df.index[-1] + pd.Timedelta(hours=1)
    ribbon_x  = [chart_df.index[-1], next_time, next_time, chart_df.index[-1]]
    ribbon_y  = [lo, lo, hi, hi]

    fig.add_trace(go.Scatter(
        x       = ribbon_x,
        y       = ribbon_y,
        fill    = "toself",
        fillcolor = "rgba(0, 212, 255, 0.15)",
        line    = dict(color="rgba(0, 212, 255, 0.6)", width=1.5, dash="dot"),
        name    = "95% Prediction Band",
    ))

    # Current price line
    fig.add_hline(
        y           = curr,
        line_dash   = "dash",
        line_color  = "#f39c12",
        annotation_text = f"Current ${curr:,.0f}",
        annotation_position = "right",
    )

    fig.update_layout(
        title            = "Last 50 BTCUSDT 1-Hour Candles + Next-Hour 95% Range",
        xaxis_title      = "Time (UTC)",
        yaxis_title      = "Price (USDT)",
        xaxis_rangeslider_visible = False,
        template         = "plotly_dark",
        height           = 520,
        legend           = dict(orientation="h", y=1.05),
        margin           = dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Part C: prediction history ────────────────────────────────────────────
    st.subheader("📋 Prediction History (Part C)")
    history = load_history(con, limit=100)

    if history.empty:
        st.info("No history yet — revisit after a few hours.")
    else:
        # Compute per-row winkler where actual is known
        def row_winkler(row):
            if pd.isna(row["actual_price"]):
                return np.nan
            return winkler_score(row["lower_95"], row["upper_95"], row["actual_price"])

        history["winkler"]  = history.apply(row_winkler, axis=1)
        history["hit"]      = history["inside"].map(
            lambda x: "✅" if x == 1 else ("❌" if x == 0 else "⏳")
        )

        display_cols = [
            "bar_time", "current_price", "lower_95", "upper_95",
            "actual_price", "hit", "winkler",
        ]
        rename = {
            "bar_time":      "Bar Time (UTC)",
            "current_price": "Price at Prediction",
            "lower_95":      "Lower 95%",
            "upper_95":      "Upper 95%",
            "actual_price":  "Actual Close",
            "hit":           "Inside?",
            "winkler":       "Winkler",
        }
        disp = history[display_cols].rename(columns=rename)

        # Rolling coverage (rows where actual is known)
        known = history[history["inside"].notna()]
        if len(known) > 0:
            rolling_cov = known["inside"].mean()
            st.caption(
                f"Live coverage over {len(known)} resolved predictions: "
                f"**{rolling_cov:.2%}** (target 95%)"
            )

        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Lower 95%":           st.column_config.NumberColumn(format="$%.2f"),
                "Upper 95%":           st.column_config.NumberColumn(format="$%.2f"),
                "Price at Prediction": st.column_config.NumberColumn(format="$%.2f"),
                "Actual Close":        st.column_config.NumberColumn(format="$%.2f"),
                "Winkler":             st.column_config.NumberColumn(format="%.1f"),
            },
        )

    con.close()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Model: GARCH(1,1)-t conditional volatility + Geometric Brownian Motion "
        "| 10,000 Monte-Carlo paths | Student-t innovations "
        "| Data: Binance data-api.binance.vision (no API key) "
        "| Refreshes every 5 min automatically"
    )


if __name__ == "__main__":
    main()