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
    page_title="₿ BTC Forecast | AlphaI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Aesthetic Injection ───────────────────────────────────────────────────────
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,300;1,400&family=DM+Sans:wght@200;400&family=Space+Mono&display=swap" rel="stylesheet">
    
    <style>
        /* Global Reset & Background */
        .stApp {
            background-color: #000000;
            color: #ffffff;
            font-family: 'DM Sans', sans-serif;
        }
        
        /* Hide Streamlit Chrome */
        header, footer, #MainMenu {visibility: hidden; height: 0;}
        [data-testid="stHeader"] {display: none;}
        [data-testid="stToolbar"] {display: none;}
        
        /* Main Container Padding */
        .block-container {
            padding-top: 4rem !important;
            padding-bottom: 5rem !important;
            max-width: 1400px;
        }

        /* Typography */
        h1, h2, h3 {
            font-family: 'DM Sans', sans-serif;
            font-weight: 200;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            color: #ffffff;
        }
        
        /* Section Labels - Terminal Style */
        .section-label {
            font-family: 'Space Mono', monospace;
            font-size: 0.8rem;
            letter-spacing: 0.35em;
            text-transform: uppercase;
            color: #AAAAAA; 
            margin-bottom: 2rem;
            margin-top: 3rem;
            display: flex;
            align-items: center;
        }
        .section-label::before {
            content: "> ";
            margin-right: 15px;
            color: #00ff87;
            font-weight: bold;
        }

        /* Metric Cards - AlphaI Style */
        [data-testid="stMetric"] {
            background-color: #0A0A0A;
            border: 1px solid #1A1A1A;
            padding: 2rem !important;
            border-radius: 0px !important; 
            border-left: 3px solid #00ff87;
        }
        
        [data-testid="stMetricLabel"] {
            font-family: 'Space Mono', monospace !important;
            font-size: 0.75rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.25em !important;
            color: #AAAAAA !important; 
            margin-bottom: 1rem !important;
        }
        
        [data-testid="stMetricValue"] {
            font-family: 'Space Mono', monospace !important;
            font-size: 2.2rem !important;
            color: #ffffff !important;
            font-weight: 400 !important;
        }
        
        /* DataFrames & Tables */
        [data-testid="stDataFrame"] {
            border: 1px solid #1A1A1A !important;
            background-color: #000 !important;
        }
        
        /* Prediction Box */
        .prediction-box {
            background-color: #0A0A0A;
            border: 1px solid #1A1A1A;
            padding: 3rem;
            text-align: center;
            border-radius: 0px;
            margin: 2.5rem 0;
            border-left: 4px solid #00ff87;
        }
        .prediction-title {
            font-family: 'Space Mono', monospace;
            color: #AAAAAA;
            font-size: 0.85rem;
            letter-spacing: 0.4em;
            text-transform: uppercase;
            margin-bottom: 2rem;
        }
        .prediction-range {
            font-family: 'Space Mono', monospace;
            font-size: 3rem;
            color: #ffffff;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 2rem;
        }
        
        /* Page Header - Manifesto Style */
        .page-header {
            margin-bottom: 5rem;
            border-bottom: 1px solid #1A1A1A;
            padding-bottom: 3rem;
            text-align: center;
        }
        .page-header h1 {
            font-size: 3.5rem;
            margin: 0;
            line-height: 1.2;
            font-weight: 200;
        }
        .page-header .italic {
            font-family: 'Cormorant Garamond', serif;
            font-style: italic;
            text-transform: none;
            font-weight: 300;
            font-size: 4rem;
            color: #ffffff;
            display: block;
            margin-top: -0.5rem;
        }
        
        /* Divider */
        hr {
            border-top: 1px solid #1A1A1A !important;
            margin: 5rem 0 !important;
        }
        
        /* Status Dot */
        .status-dot {
            height: 10px;
            width: 10px;
            background-color: #00ff87;
            border-radius: 50%;
            display: inline-block;
            margin-right: 15px;
            box-shadow: 0 0 15px #00ff87;
        }

        /* Footer */
        .footer-text {
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            color: #555555;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            text-align: center;
            padding: 4rem 0;
            border-top: 1px solid #1A1A1A;
        }
    </style>
""", unsafe_allow_html=True)

def ms_until_next_candle(buffer_sec=30):
    now = datetime.now(timezone.utc)
    secs_past_hour = now.minute * 60 + now.second
    secs_until_close = 3600 - secs_past_hour
    return (secs_until_close + buffer_sec) * 1000

st_autorefresh(interval=ms_until_next_candle(), key="candle_sync")

# ── Constants ─────────────────────────────────────────────────────────────────
LOOKBACK        = 500
CHART_BARS      = 50
SUMMARY_FILE    = Path("backtest_summary.json")
DB_PATH         = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", os.path.dirname(__file__))
DB_PATH         = os.path.join(DB_PATH, "predictions.db")
DB_FILE         = Path(DB_PATH)

if os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") and not os.path.exists(DB_PATH):
    import seed_db
    seed_db.seed()

# ── SQLite persistence ────────────────────────────────────────────────────────
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

def save_prediction(con, pred, bar_time):
    con.execute("""
        INSERT OR IGNORE INTO predictions
          (fetched_at, bar_time, current_price, lower_95, upper_95, sigma, t_df, entropy_scalar)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(timezone.utc).isoformat(), bar_time, pred["current_price"], pred["lower_95"], pred["upper_95"], pred["sigma"], pred["t_df"], pred["entropy_scalar"]))
    con.commit()

def fill_actuals(con, df_prices):
    rows = pd.read_sql("SELECT id, bar_time FROM predictions WHERE actual_price IS NULL", con)
    if rows.empty: return
    times, closes = list(df_prices.index), list(df_prices["close"])
    m = {times[i].isoformat(): closes[i+1] for i in range(len(times)-1)}
    for _, r in rows.iterrows():
        a = m.get(r["bar_time"])
        if a is not None:
            lo, hi = con.execute("SELECT lower_95, upper_95 FROM predictions WHERE id=?", (r["id"],)).fetchone()
            con.execute("UPDATE predictions SET actual_price=?, inside=? WHERE id=?", (a, int(lo <= a <= hi), r["id"]))
    con.commit()

def load_history(con, limit=100):
    return pd.read_sql("SELECT * FROM predictions ORDER BY bar_time DESC LIMIT ?", con, params=(limit,))

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=55)
def get_data(n: int) -> pd.DataFrame:
    return fetch_klines(n_bars=n)

def load_backtest_summary() -> dict | None:
    if SUMMARY_FILE.exists(): return json.loads(SUMMARY_FILE.read_text())
    return None

def section_header(label, is_live=False):
    dot = '<span class="status-dot"></span>' if is_live else ""
    st.markdown(f'<div class="section-label">{dot}{label}</div>', unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Header
    st.markdown("""
        <div class="page-header">
            <h1>₿ BTC/USDT</h1>
            <span class="italic">Range Forecaster</span>
        </div>
    """, unsafe_allow_html=True)

    # Fetch & Predict
    try:
        df = get_data(LOOKBACK + 1)
    except Exception as e:
        st.error(f"Binance Error: {e}")
        st.stop()

    p, t = df["close"].values[-LOOKBACK:], df.index[-1].isoformat()
    pred = predict_95_range(p)
    con = init_db()
    save_prediction(con, pred, t)
    fill_actuals(con, df)
    summary = load_backtest_summary()

    # 2. Live Performance
    section_header("LIVE PERFORMANCE", is_live=True)
    res = pd.read_sql("SELECT * FROM predictions WHERE actual_price IS NOT NULL ORDER BY bar_time ASC", con)
    
    if len(res) >= 3:
        res["winkler"] = res.apply(lambda r: winkler_score(r["lower_95"], r["upper_95"], r["actual_price"]), axis=1)
        c = st.columns(4)
        c[0].metric("LIVE COVERAGE", f"{res['inside'].mean():.1%}")
        c[1].metric("MEAN WINKLER", f"${res['winkler'].mean():,.0f}")
        c[2].metric("BEST BAR", f"${res['winkler'].min():,.0f}")
        c[3].metric("RESOLVED", len(res))
        
        # Winkler chart with high visibility
        fig_live = go.Figure()
        colors = ["#00ff87" if i==1 else "#ff6b35" for i in res["inside"]]
        fig_live.add_trace(go.Scatter(
            x=res["bar_time"], y=res["winkler"], 
            mode="lines+markers", 
            marker=dict(color=colors, size=8, line=dict(width=1, color="#000")), 
            line=dict(color="#222222", width=1.5)
        ))
        
        if summary:
            fig_live.add_hline(y=summary["mean_winkler_95"], line_dash="dash", line_color="#444444", 
                               annotation_text="BT BASELINE", annotation_font=dict(family="Space Mono", size=10, color="#888"))
        
        fig_live.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
            height=350, margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=True, gridcolor="#111111", tickfont=dict(family="Space Mono", color="#AAAAAA", size=11)),
            yaxis=dict(showgrid=True, gridcolor="#111111", tickfont=dict(family="Space Mono", color="#AAAAAA", size=11), title="Winkler Score")
        )
        st.plotly_chart(fig_live, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("⏳ SYNCING RESOLVED STREAM...")

    st.divider()

    # 3. Backtest Metrics
    section_header("BACKTEST METRICS")
    if summary:
        c = st.columns(4)
        c[0].metric("COVERAGE", f"{summary['coverage_95']:.2%}")
        c[1].metric("MEAN WIDTH", f"${summary['mean_width']:,.0f}")
        c[2].metric("MEAN WINKLER", f"${summary['mean_winkler_95']:,.0f}")
        c[3].metric("SAMPLES (N)", f"{summary['n_predictions']:,}")
    
    st.divider()

    # 4. Active Prediction Box
    section_header("ACTIVE PREDICTION WINDOW")
    lo, hi = pred["lower_95"], pred["upper_95"]
    mins_left = 59 - datetime.now(timezone.utc).minute

    st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-title">Target Range (95% Confidence)</div>
            <div class="prediction-range">
                <span style="color:#00ff87">${lo:,.0f}</span>
                <span style="color:#222; font-size:2rem">—</span>
                <span style="color:#ff6b35">${hi:,.0f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CURRENT PRICE", f"${pred['current_price']:,.2f}")
    c2.metric("VOLATILITY", f"{pred['sigma']*100:.3f}%")
    c3.metric("ENTROPY SCALAR", f"{pred['entropy_scalar']:.2f}X")
    c4.metric("T-CLOSE", f"{mins_left} MIN")

    st.divider()

    # 5. Price Action Chart
    section_header("PRICE ACTION + PREDICTION BAND")
    cdf = df.iloc[-CHART_BARS:]
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=cdf.index, open=cdf["open"], high=cdf["high"], low=cdf["low"], close=cdf["close"], 
        increasing_line_color="#00ff87", decreasing_line_color="#ff6b35",
        increasing_fillcolor="#00ff87", decreasing_fillcolor="#ff6b35",
        name="BTC/USDT"
    ))
    
    # Prediction Band
    nt = cdf.index[-1] + pd.Timedelta(hours=1)
    fig.add_trace(go.Scatter(
        x=[cdf.index[-1], nt, nt, cdf.index[-1]], y=[lo, lo, hi, hi], 
        fill="toself", fillcolor="rgba(0,255,135,0.05)", 
        line=dict(color="#00ff87", width=1, dash="dot"), 
        name="Forecast Band"
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
        height=600, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor="#111111", rangeslider_visible=False, tickfont=dict(family="Space Mono", color="#AAAAAA")),
        yaxis=dict(showgrid=True, gridcolor="#111111", tickfont=dict(family="Space Mono", color="#AAAAAA")),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.divider()

    # 6. History Table
    section_header("PREDICTION HISTORY")
    h = load_history(con, 500)
    
    if not h.empty:
        h["INSIDE?"] = h["inside"].map({1: "✅", 0: "❌"}).fillna("⏳")
        h["WINKLER"] = h.apply(lambda r: winkler_score(r["lower_95"], r["upper_95"], r["actual_price"]) if not pd.isna(r["actual_price"]) else np.nan, axis=1)
        
        display_cols = ["bar_time", "current_price", "lower_95", "upper_95", "actual_price", "INSIDE?", "WINKLER"]
        rename_map = {
            "bar_time": "TIME (UTC)", 
            "current_price": "ENTRY PRICE", 
            "lower_95": "LOWER 95%", 
            "upper_95": "UPPER 95%", 
            "actual_price": "RESOLVED PRICE"
        }
        
        disp = h[display_cols].rename(columns=rename_map)
        
        st.dataframe(
            disp, 
            use_container_width=True, 
            hide_index=True, 
            column_config={
                "TIME (UTC)": st.column_config.TextColumn("TIME (UTC)"),
                "LOWER 95%": st.column_config.NumberColumn(format="$%.2f"), 
                "UPPER 95%": st.column_config.NumberColumn(format="$%.2f"),
                "ENTRY PRICE": st.column_config.NumberColumn(format="$%.2f"), 
                "RESOLVED PRICE": st.column_config.NumberColumn(format="$%.2f"),
                "WINKLER": st.column_config.NumberColumn(format="%.1f")
            }
        )
    
    # 7. Footer
    st.markdown("""
        <div class="footer-text">
           GARCH(1,1)-t + GBM | 10k Path Monte-Carlo | Live Binance Data Feed
        </div>
    """, unsafe_allow_html=True)
    
    con.close()

if __name__ == "__main__": main()
