"""
data.py — Binance public API wrapper for BTCUSDT 1-hour candles

We use data-api.binance.vision (not api.binance.com) because:
  - No API key or account required
  - No geo-block in India (critical for the submission)
  - Identical endpoint schema to the main API

Binance kline response columns (0-indexed):
  0  open_time (ms), 1 open, 2 high, 3 low, 4 close, 5 volume,
  6  close_time (ms), 7 quote_asset_volume, 8 num_trades,
  9  taker_buy_base, 10 taker_buy_quote, 11 ignore
"""

import time
import requests
import pandas as pd

BASE_URL  = "https://data-api.binance.vision"
KLINE_EP  = "/api/v3/klines"
SYMBOL    = "BTCUSDT"
INTERVAL  = "1h"
MAX_LIMIT = 1000   # Binance hard limit per request


def _parse_klines(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("open_time").sort_index()
    return df


def fetch_klines(n_bars: int = 750, retries: int = 3) -> pd.DataFrame:
    """
    Fetch the most recent `n_bars` closed 1-hour candles for BTCUSDT.

    Binance limits each request to 1000 bars.  If n_bars > 1000 we page
    backward with multiple calls.

    Returns a DataFrame indexed by open_time (UTC), sorted oldest→newest.
    The most recent bar may still be open; we drop it so every bar is closed.
    """
    all_frames = []
    end_time   = None                 # None → "up to now" on first call

    remaining = n_bars + 1            # +1 so we can drop the open bar at the end
    while remaining > 0:
        limit  = min(remaining, MAX_LIMIT)
        params = {
            "symbol":   SYMBOL,
            "interval": INTERVAL,
            "limit":    limit,
        }
        if end_time is not None:
            params["endTime"] = end_time   # exclusive upper bound (ms)

        for attempt in range(retries):
            try:
                resp = requests.get(
                    BASE_URL + KLINE_EP,
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
                batch = resp.json()
                break
            except Exception as exc:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Binance API failed after {retries} retries: {exc}"
                    ) from exc
                time.sleep(1.5 * (attempt + 1))

        if not batch:
            break

        frame = _parse_klines(batch)
        all_frames.append(frame)

        # Move end_time to just before the oldest bar in this batch
        oldest_open_ms = int(batch[0][0])
        end_time       = oldest_open_ms - 1

        remaining -= len(batch)
        if len(batch) < limit:
            break                          # Binance returned fewer than asked → done

    if not all_frames:
        raise RuntimeError("No kline data returned from Binance.")

    df = pd.concat(all_frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Drop the last bar if it is still open (close_time > now)
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"].astype("int64") // 1_000_000 < now_ms]

    # Keep only the most recent n_bars closed bars
    return df.iloc[-n_bars:]


def get_latest_closed_bar() -> dict:
    """
    Return the single most recent *closed* 1-hour bar as a dict.
    Used by the live dashboard.
    """
    df  = fetch_klines(n_bars=2)      # 2 so we're sure the last is closed
    row = df.iloc[-1]
    return {
        "open_time":  row.name.isoformat(),
        "open":       float(row["open"]),
        "high":       float(row["high"]),
        "low":        float(row["low"]),
        "close":      float(row["close"]),
        "volume":     float(row["volume"]),
    }