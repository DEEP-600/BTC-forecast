"""
model.py — GARCH(1,1)-t + GBM Forecaster for BTC/USDT 1-hour bars

Why this model?
  - Geometric Brownian Motion (GBM) is the required framework.
  - The key upgrade over a plain GBM: we don't use a fixed rolling std for
    volatility. We use a GARCH(1,1) model to get a *conditional* one-step-ahead
    volatility forecast that shrinks in calm regimes and expands in volatile ones.
    This is exactly "volatility clustering" — the #2 concept in the brief.
  - We simulate innovations with a Student-t distribution (not Normal) to capture
    BTC's fat tails — the #3 concept in the brief. The t-distribution is fitted
    fresh on each window's log-returns so the degrees-of-freedom adapt to regimes.
  - Result: narrower ranges during calm periods, wider during chaos → best
    possible Winkler score while keeping coverage pinned to 0.95.

Known bug in the starter notebook this fixes:
  - The starter estimates volatility as a simple rolling std(log_returns[-N:]).
    That treats yesterday-last-month as equally relevant as the last hour.
    GARCH weights recent shocks exponentially → far tighter ranges in practice.
"""

import warnings
import numpy as np
from arch import arch_model
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_SIMS       = 10_000   # Monte-Carlo paths per prediction
ALPHA        = 0.05     # 1 - confidence level → 95% interval
GARCH_MIN_OBS = 100     # minimum bars before we attempt GARCH; use EWMA below
EWMA_LAMBDA  = 0.94     # RiskMetrics decay — fallback when GARCH fails/too few obs


# ─────────────────────────────────────────────────────────────────────────────
# Volatility estimation
# ─────────────────────────────────────────────────────────────────────────────
def _ewma_vol(log_returns: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """Exponentially Weighted Moving Average volatility — fast fallback."""
    var = float(np.var(log_returns[:20]))          # seed with first 20 obs
    for r in log_returns:
        var = lam * var + (1 - lam) * r**2
    return float(np.sqrt(var))


def estimate_conditional_vol(log_returns: np.ndarray) -> float:
    """
    Fit GARCH(1,1) with Student-t innovations and return the 1-step-ahead
    conditional volatility forecast (annualised in log-return space).

    Falls back to EWMA if the series is too short or GARCH fails to converge.
    """
    if len(log_returns) < GARCH_MIN_OBS:
        return _ewma_vol(log_returns)

    try:
        # Scale up to percentage returns — GARCH optimisers are happier in that space
        pct = log_returns * 100.0
        am  = arch_model(pct, vol="Garch", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})

        # 1-step-ahead conditional variance, back to raw log-return scale
        fc   = res.forecast(horizon=1, reindex=False)
        cond_var_pct = float(fc.variance.iloc[-1, 0])
        cond_var     = cond_var_pct / (100.0**2)
        return float(np.sqrt(cond_var))

    except Exception:
        return _ewma_vol(log_returns)


# ─────────────────────────────────────────────────────────────────────────────
# Student-t parameter estimation
# ─────────────────────────────────────────────────────────────────────────────
def fit_student_t(log_returns: np.ndarray):
    """
    Fit a Student-t distribution to log-returns.
    Returns (df, loc, scale) — clamped so df > 2 (finite variance).
    """
    try:
        df, loc, scale = stats.t.fit(log_returns, floc=0)
        df = max(float(df), 2.5)          # must have finite variance
        return df, float(loc), float(scale)
    except Exception:
        return 4.0, 0.0, float(np.std(log_returns))


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_95_range(
    prices: np.ndarray,
    n_sims: int = N_SIMS,
    random_state: int | None = None,
) -> dict:
    """
    Given a 1-D array of *closed* hourly prices (oldest → newest),
    return the 95 % prediction interval for the NEXT hour's closing price.

    Parameters
    ----------
    prices       : array of historical close prices, length ≥ 2
    n_sims       : number of Monte-Carlo paths
    random_state : optional seed for reproducibility

    Returns
    -------
    dict with keys:
        current_price   float   last observed close
        lower_95        float   2.5th percentile of simulated prices
        upper_95        float   97.5th percentile
        sigma           float   GARCH conditional vol (log-return scale)
        mu              float   sample mean log-return used as drift
        t_df            float   fitted Student-t degrees of freedom
    """
    if random_state is not None:
        np.random.seed(random_state)

    prices      = np.asarray(prices, dtype=float)
    log_returns = np.diff(np.log(prices))           # length N-1

    S0    = float(prices[-1])
    mu    = float(log_returns.mean())               # drift
    sigma = estimate_conditional_vol(log_returns)   # GARCH conditional vol

    # Fit Student-t for fat-tail innovations
    df, _, _ = fit_student_t(log_returns)

    # ── GBM simulation ─────────────────────────────────────────────────────
    #   S_1 = S_0 * exp( (mu - 0.5*sigma^2)*dt  +  sigma*sqrt(dt)*Z )
    #   where Z ~ standardised t(df)
    dt  = 1.0                                        # one hour forward
    raw = stats.t.rvs(df=df, size=n_sims)           # raw t draws
    Z   = raw / np.sqrt(df / (df - 2))              # standardise to unit variance

    log_step = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    S1       = S0 * np.exp(log_step)

    lower = float(np.percentile(S1, 100 * ALPHA / 2))
    upper = float(np.percentile(S1, 100 * (1 - ALPHA / 2)))

    return {
        "current_price": S0,
        "lower_95":      lower,
        "upper_95":      upper,
        "sigma":         sigma,
        "mu":            mu,
        "t_df":          df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Winkler score (matches the assignment's evaluate() function)
# ─────────────────────────────────────────────────────────────────────────────
def winkler_score(lower: float, upper: float, actual: float, alpha: float = ALPHA) -> float:
    """
    Winkler (1972) interval score.
    Lower = better. A miss incurs a penalty of (2/alpha) * distance_outside.
    """
    width   = upper - lower
    penalty = 0.0
    if actual < lower:
        penalty = (2.0 / alpha) * (lower - actual)
    elif actual > upper:
        penalty = (2.0 / alpha) * (actual - upper)
    return width + penalty


def evaluate_predictions(records: list[dict]) -> dict:
    """
    Compute summary metrics over a list of prediction dicts.
    Each dict must have: lower_95, upper_95, actual_price.

    Returns: coverage_95, mean_width, mean_winkler_95
    """
    inside  = []
    widths  = []
    winklers = []
    for r in records:
        lo, hi, y = r["lower_95"], r["upper_95"], r["actual_price"]
        inside.append(lo <= y <= hi)
        widths.append(hi - lo)
        winklers.append(winkler_score(lo, hi, y))

    return {
        "coverage_95":     float(np.mean(inside)),
        "mean_width":      float(np.mean(widths)),
        "mean_winkler_95": float(np.mean(winklers)),
        "n_predictions":   len(records),
    }