"""
Portfolio construction for the long-short strategy.

The simplest book is long the top 10% of the score and short the bottom 10%, equal
weights, +1 / -1 (the default PortfolioConfig). This module builds the weights in steps,
each optional, and hands them to evaluation.backtest.run_backtest_on_weights, which
computes returns, drift-aware turnover and costs.

Steps, in the order they are applied (fields of PortfolioConfig):

1. Ranking            percentile rank of the score each month, or within each
                      industry (industry_neutral=True): every industry contributes
                      its own top and bottom names, so the book is not a sector bet;
                      or within 5 equal-size groups of 12-month volatility
                      (vol_neutral=True): every volatility group contributes its own
                      top and bottom 10%, so the book is not a "long calm / short
                      volatile" bet.
2. Membership         top / bottom bucket (scheme="bucket", long_q / short_q), with
                      an optional buffer (buffer_exit=0.20: a name enters at the top
                      10% and is only sold once it leaves the top 20%), or the whole
                      cross-section with weights proportional to the centred rank
                      (scheme="rank").
3. Weights in a leg   equal, or inverse volatility (leg_weighting="inverse_vol",
                      weight ~ 1 / raw_vol_12m): a stock with twice the volatility
                      gets half the weight, which tames the high-vol short leg; or
                      liquidity (leg_weighting="sqrt_adv", weight ~ sqrt of the daily
                      dollar volume), which keeps trading costs down (cost_study.py).
4. Per-name cap       max_weight, as a fraction of capital (before the vol target).
5. Leg sizes          dollar-neutral (+1 long / -1 short), or beta-neutral
                      (beta_neutral=True): leg sizes chosen so that
                      size_long * beta_long = size_short * beta_short, total gross 2.
6. Volatility target  the whole book is multiplied by a leverage L_t chosen so that
                      its forecast annual volatility is vol_target (10%). L_t is only
                      changed when the forecast at the current leverage leaves the
                      band vol_band (9%-11%); inside the band the leverage is kept,
                      which avoids trading on noise in the forecast.

Short-sale constraint (short_allowed_col): a stock whose column is False is never shorted
(it may still be held long); a short already held is closed when its stock becomes
unshortable.

Volatility forecast (step 6), made at the end of month t with data known then:

- "holdings": historical simulation of the *current* book. Take this month's
  weights w_t and the last vol_lookback months of each stock's realised monthly
  returns, compute the returns this exact book would have had, and take an
  exponentially weighted RMS (half-life vol_halflife months). Reacts at once when
  the book moves into riskier names. A stock without history in a month gets that
  month's cross-sectional mean return.
- "realized": exponentially weighted RMS of the strategy's own past (unscaled)
  monthly returns. Captures volatility regimes, ignores the book's composition.
  Falls back to "holdings" until 6 months of history exist.
- "blend": square root of the average of the two variances.
- "daily": historical simulation of the current book on its stocks' last
  vol_lookback_days DAILY returns (exponentially weighted, half-life vol_halflife_days),
  scaled to a year with 252 days. Reacts within weeks instead of months; needs the daily
  returns matrix (make_daily_returns_wide).

Forecasts from monthly data are biased (fat tails, volatility clustering, and the
holdings replay overstates risk because extreme-decile stocks were picked for their
big past moves). Two corrections:
- vol_adaptive (default): the forecast is multiplied by sqrt(mean z^2) over the
  last 24 months, with z = realised unlevered book return / its forecast, using only
  months already realised. The scale follows the book as its behaviour changes.
- vol_scale: one fixed multiplier, calibrated on validation (calibrate_vol_scale);
  used alone when vol_adaptive=False, and for the first 12 months otherwise.

All the inputs are columns of the predictions frame: month, permno, score,
ret_fwd1m, industry_id, and the unscaled raw_vol_12m / raw_beta_12m that the
preprocessing keeps in the panel (merge them in with add_risk_columns()).
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from evaluation.backtest import (  # noqa: E402
    portfolio_returns_and_turnover,
    run_backtest_on_weights,
)


@dataclass(frozen=True)
class PortfolioConfig:
    name: str = "Equal-weight deciles"
    # 1-2. ranking and membership
    scheme: str = "bucket"            # "bucket" (top / bottom quantile) or "rank" (all stocks)
    long_q: float = 0.90
    short_q: float = 0.10
    buffer_exit: float | None = None  # e.g. 0.20: keep a long while rank >= 0.80, a short while <= 0.20
    industry_neutral: bool = False
    min_industry_size: int = 10       # smaller industry-months fall back to the global rank
    vol_neutral: bool = False         # rank inside equal-size volatility groups (see _rank_scores)
    vol_buckets: int = 5
    vol_bucket_col: str = "raw_vol_12m"
    # 3-5. weights
    leg_weighting: str = "equal"      # "equal", "inverse_vol" or "sqrt_adv" (~ sqrt of daily $ volume)
    liquidity_col: str = "adv_usd"    # for "sqrt_adv": dollar volume known at t (cost_inputs.parquet)
    max_weight: float | None = None   # per-name cap, fraction of capital
    short_allowed_col: str | None = None  # bool column; False = the stock cannot be shorted
    beta_neutral: bool = False
    beta_shrink: float = 0.67         # Blume: beta_used = 0.67 * beta + 0.33
    leg_size_bounds: tuple = (0.5, 1.5)
    # 6. volatility target
    vol_target: float | None = None   # annual, e.g. 0.10
    vol_band: tuple = (0.09, 0.11)
    # Chosen on validation data only: calibrate on 2005-09, measure on 2010-14
    # (sizing_comparison.py --vol-method-study). blend / 6 months and holdings / 12 months
    # tied on error (0.26 vs 0.24 pp); blend re-levers less often.
    vol_method: str = "blend"         # "holdings", "realized", "blend" or "daily"
    vol_lookback: int = 36            # months of stock history for "holdings"
    vol_halflife: float = 6.0         # months
    vol_lookback_days: int = 126      # trading days of history for "daily"
    vol_halflife_days: float = 21.0   # trading days (about one month)
    vol_scale: float = 1.0            # multiplies the forecast (calibrated on validation)
    # Adaptive correction: multiply the forecast by sqrt(mean of (realised / forecast)^2)
    # over the book's own past months (only months < t), so the scale follows the book.
    # Window chosen on validation only (error on 2010-14 summed over three books:
    # 24m 2.2 pp, 36m / 60m / expanding 5.7 pp).
    vol_adaptive: bool = True
    vol_adapt_window: int | None = 24  # months; None = all past out-of-sample months
    vol_adapt_min_months: int = 12     # before that, vol_scale is used
    max_leverage: float = 2.0         # cap on L_t; the book's gross exposure is 2 * L_t
    # columns
    vol_col: str = "raw_vol_12m"
    beta_col: str = "raw_beta_12m"


# -------- data helpers --------


def add_risk_columns(pred, panel, cols=("raw_vol_12m", "raw_beta_12m")):
    """Merge the unscaled risk columns of the processed panel into a predictions frame."""
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise ValueError(
            f"{missing} not in the panel. Rebuild it "
            "(python src/preprocessing/build_processed_data.py)."
        )
    keep = [c for c in cols if c not in pred.columns]
    if not keep:
        return pred
    right = panel[["permno", "month"] + keep].copy()
    right["month"] = pd.to_datetime(right["month"])
    out = pred.copy()
    out["month"] = pd.to_datetime(out["month"])
    return out.merge(right, on=["permno", "month"], how="left")


def make_returns_wide(panel, return_col="ret_fwd1m"):
    """Month x permno matrix of ret_fwd1m (row t = return realised during month t+1)."""
    df = panel[["month", "permno", return_col]].copy()
    df["month"] = pd.to_datetime(df["month"])
    return df.set_index(["month", "permno"])[return_col].unstack().sort_index()


def add_trade_timing(pred, ret_day1, timing="day1", return_col="ret_fwd1m"):
    """Set the return the book earns, by when it trades.

    timing "close": at the month-end close the last input is measured at (return of month
    t+1). timing "day1": one day later, at the close of the first trading day of month t+1,
    held to the close of the first trading day of month t+2:
        (1 + r[t+1]) / (1 + d1[t+1]) * (1 + d1[t+2]) - 1
    with d1 the first-day return. ret_day1: Series indexed by (permno, month) (month t
    holds the first-day return of month t). A missing first-day return counts as 0. Keeps
    the month-end-close return in ret_close.
    """
    out = pred.copy()
    out["ret_close"] = out[return_col]
    if timing == "close":
        return out
    d1 = ret_day1[~ret_day1.index.duplicated()]
    nxt = out["month"] + pd.offsets.MonthEnd(1)
    nxt2 = out["month"] + pd.offsets.MonthEnd(2)
    r1 = d1.reindex(pd.MultiIndex.from_arrays([out["permno"], nxt])).to_numpy(dtype=float)
    r2 = d1.reindex(pd.MultiIndex.from_arrays([out["permno"], nxt2])).to_numpy(dtype=float)
    last_allowed = out["month"].max() + pd.offsets.MonthEnd(1)   # never use days after the period
    r2 = np.where(nxt2 <= last_allowed, r2, np.nan)
    r1, r2 = np.nan_to_num(r1), np.nan_to_num(r2)
    ok = (1 + r1) > 0.05          # a first day that wipes the stock out: keep the month's return
    out[return_col] = np.where(ok, (1 + out["ret_close"]) / np.where(ok, 1 + r1, 1.0) * (1 + r2) - 1, out["ret_close"])
    return out


# -------- steps 1-5: weights before the vol target (long leg +1, short leg -1) --------


def vol_bucket(df, n_buckets=5, vol_col="raw_vol_12m"):
    """0 (calmest) .. n_buckets-1 (most volatile), equal-size groups each month.

    Stocks with no volatility history (mostly recent listings) go into the most
    volatile group.
    """
    vol = pd.to_numeric(df[vol_col], errors="coerce")
    pct = vol.groupby(df["month"]).rank(pct=True, method="first")
    bucket = np.ceil(pct * n_buckets) - 1
    return bucket.fillna(n_buckets - 1).clip(0, n_buckets - 1).astype(int)


def _rank_scores(df, cfg, score_col):
    if cfg.vol_neutral:
        # Volatility-neutral book: the score is ranked inside each volatility group,
        # so every group supplies its own top / bottom names and the long and short
        # legs hold the same mix of calm and volatile stocks. Without this the MLP
        # book is mostly "long calm stocks, short volatile ones".
        if cfg.industry_neutral:
            raise ValueError("vol_neutral and industry_neutral cannot be combined (groups get too small)")
        groups = vol_bucket(df, cfg.vol_buckets, cfg.vol_bucket_col)
        return df.groupby([df["month"], groups])[score_col].rank(pct=True)
    rank = df.groupby("month")[score_col].rank(pct=True)
    if cfg.industry_neutral:
        g = df.groupby(["month", "industry_id"])[score_col]
        rank_ind = g.rank(pct=True)
        size = g.transform("size")
        rank = rank_ind.where(size >= cfg.min_industry_size, rank)
    return rank


def _bucket_positions(df, rank, cfg, can_short=None):
    """+1 / -1 / 0 per row; with a buffer the decision depends on last month's position.
    can_short: bool Series (same index as df) or None (every stock can be shorted)."""
    if can_short is None:
        can_short = pd.Series(True, index=df.index)
    if cfg.buffer_exit is None:
        pos = np.where(rank >= cfg.long_q, 1, np.where((rank <= cfg.short_q) & can_short, -1, 0))
        return pd.Series(pos, index=df.index, dtype="int8")

    exit_long, exit_short = 1.0 - cfg.buffer_exit, cfg.buffer_exit
    R = pd.Series(rank.values, index=pd.MultiIndex.from_arrays([df["month"], df["permno"]]))
    R = R.unstack().sort_index()
    vals = R.to_numpy()
    S = pd.Series(can_short.to_numpy(dtype=bool), index=pd.MultiIndex.from_arrays([df["month"], df["permno"]]))
    shortable = S.unstack().reindex_like(R).fillna(False).to_numpy(dtype=bool)
    out = np.zeros(vals.shape, dtype=np.int8)
    prev = np.zeros(vals.shape[1], dtype=np.int8)
    for t in range(vals.shape[0]):
        r = vals[t]
        ok = ~np.isnan(r)
        is_long = ok & ((r >= cfg.long_q) | ((prev == 1) & (r >= exit_long)))
        is_short = ok & ~is_long & shortable[t] & ((r <= cfg.short_q) | ((prev == -1) & (r <= exit_short)))
        cur = np.zeros_like(prev)
        cur[is_long] = 1
        cur[is_short] = -1
        out[t] = cur
        prev = cur
    stacked = pd.Series(out.ravel(), index=pd.MultiIndex.from_product([R.index, R.columns]))
    rows = pd.MultiIndex.from_arrays([df["month"], df["permno"]])
    return pd.Series(stacked.reindex(rows).fillna(0).to_numpy(dtype="int8"), index=df.index)


def _normalise_legs(w, months):
    """Scale each month's long leg to +1 and short leg to -1."""
    long_sum = w.clip(lower=0).groupby(months).transform("sum")
    short_sum = (-w.clip(upper=0)).groupby(months).transform("sum")
    out = pd.Series(0.0, index=w.index)
    out[w > 0] = w[w > 0] / long_sum[w > 0]
    out[w < 0] = w[w < 0] / short_sum[w < 0]
    return out


def _cap_weights(w, months, cap, n_iter=20):
    """Cap |w| at `cap` inside each leg, giving the excess to the uncapped names pro rata."""
    w = w.copy()
    for sign in (1, -1):
        leg = (np.sign(w) == sign)
        x = (w.abs()).where(leg, 0.0)
        for _ in range(n_iter):
            over = x > cap + 1e-12
            if not over.any():
                break
            excess = (x - cap).where(over, 0.0).groupby(months).transform("sum")
            x = x.where(~over, cap)
            room = x.where(leg & ~over & (x > 0), 0.0)
            room_sum = room.groupby(months).transform("sum")
            x = x + np.where(room_sum > 0, excess * room / room_sum.replace(0, np.nan), 0.0)
            x = x.fillna(0.0)
        w = w.where(~leg, sign * x)
    return w


def _beta_neutral_leg_sizes(df, w, cfg):
    beta = pd.to_numeric(df[cfg.beta_col], errors="coerce")
    beta = (cfg.beta_shrink * beta + (1 - cfg.beta_shrink)).fillna(1.0)
    months = df["month"]
    beta_long = (w.clip(lower=0) * beta).groupby(months).transform("sum")
    beta_short = (-w.clip(upper=0) * beta).groupby(months).transform("sum")
    total = beta_long + beta_short
    lo, hi = cfg.leg_size_bounds
    size_long = (2 * beta_short / total).where(total > 0, 1.0).clip(lo, hi)
    size_short = (2 * beta_long / total).where(total > 0, 1.0).clip(lo, hi)
    return pd.Series(np.where(w > 0, w * size_long, w * size_short), index=w.index)


def build_unscaled_weights(pred, cfg, score_col="score"):
    """Steps 1-5. Returns a copy of pred with rank_score, position and weight_unscaled."""
    df = pred.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.dropna(subset=[score_col]).sort_values(["month", "permno"]).reset_index(drop=True)
    months = df["month"]

    df["rank_score"] = _rank_scores(df, cfg, score_col)
    can_short = None
    if cfg.short_allowed_col is not None:
        if cfg.short_allowed_col not in df.columns:
            raise ValueError(f"short_allowed_col={cfg.short_allowed_col!r} is not a column")
        can_short = df[cfg.short_allowed_col].fillna(False).astype(bool)

    if cfg.scheme == "rank":
        centred = df["rank_score"] - df["rank_score"].groupby(months).transform("mean")
        magnitude, sign = centred.abs(), np.sign(centred)
        if can_short is not None:
            magnitude = magnitude.where(~((sign < 0) & ~can_short), 0.0)
    elif cfg.scheme == "bucket":
        df["position"] = _bucket_positions(df, df["rank_score"], cfg, can_short)
        magnitude, sign = (df["position"] != 0).astype(float), df["position"].astype(float)
    else:
        raise ValueError("scheme must be 'bucket' or 'rank'")

    if cfg.leg_weighting == "inverse_vol":
        vol = pd.to_numeric(df[cfg.vol_col], errors="coerce")
        vol = vol.fillna(vol.groupby(months).transform("median"))
        floor = vol.groupby(months).transform(lambda x: x.quantile(0.05))
        magnitude = magnitude / vol.clip(lower=floor)
    elif cfg.leg_weighting == "sqrt_adv":
        # Liquidity weights: weight ~ sqrt(average daily dollar volume), so a name that
        # trades 100x less gets 10x less weight. Missing volume -> the month's 10th
        # percentile (treated as illiquid).
        if cfg.liquidity_col not in df.columns:
            raise ValueError(f"leg_weighting='sqrt_adv' needs a {cfg.liquidity_col} column")
        adv = pd.to_numeric(df[cfg.liquidity_col], errors="coerce")
        adv = adv.fillna(adv.groupby(months).transform(lambda x: x.quantile(0.10)))
        magnitude = magnitude * np.sqrt(adv.clip(lower=0).fillna(0))
    elif cfg.leg_weighting != "equal":
        raise ValueError("leg_weighting must be 'equal', 'inverse_vol' or 'sqrt_adv'")

    w = _normalise_legs(sign * magnitude, months)
    if cfg.max_weight is not None:
        w = _cap_weights(w, months, cfg.max_weight)
    if cfg.beta_neutral:
        w = _beta_neutral_leg_sizes(df, w, cfg)
    df["weight_unscaled"] = w.astype(float)
    if "position" not in df.columns:
        df["position"] = np.sign(df["weight_unscaled"]).astype("int8")
    return df


# -------- step 6: volatility forecast and target --------


def _ewma_weights(n, halflife):
    # k = 0 is the most recent month
    return 0.5 ** (np.arange(n) / halflife)


def forecast_vol_holdings(df, returns_wide, lookback=36, halflife=12.0, weight_col="weight_unscaled"):
    """Annualised forecast vol of each month's book by historical simulation."""
    rw = returns_wide.sort_index()
    rw_months = rw.index
    rw_vals = rw.to_numpy(dtype=float)
    row_mean = np.nanmean(rw_vals, axis=1)
    col_pos = pd.Series(np.arange(rw.shape[1]), index=rw.columns)

    out = {}
    held = df.loc[df[weight_col] != 0, ["month", "permno", weight_col]]
    for month, g in held.groupby("month"):
        end = rw_months.searchsorted(month, side="left")  # rows strictly before `month`
        start = max(0, end - lookback)
        if end - start < 12:
            out[month] = np.nan
            continue
        cols = col_pos.reindex(g["permno"].to_numpy())
        known = cols.notna().to_numpy()
        hist = np.full((end - start, len(g)), np.nan)
        hist[:, known] = rw_vals[start:end][:, cols[known].astype(int).to_numpy()]
        hist = np.where(np.isnan(hist), row_mean[start:end, None], hist)
        sim = hist @ g[weight_col].to_numpy(dtype=float)        # oldest -> newest
        lam = _ewma_weights(len(sim), halflife)[::-1]           # newest gets weight 1
        out[month] = np.sqrt(12 * np.sum(lam * sim**2) / np.sum(lam))
    return pd.Series(out).sort_index()


def make_daily_returns_wide(daily):
    """Trading day x permno matrix of daily returns (float32) from a long frame with
    permno, dlycaldt, dlyret (data/raw_wrds/crsp_daily_returns.parquet)."""
    d = daily[["permno", "dlycaldt", "dlyret"]].copy()
    d["dlycaldt"] = pd.to_datetime(d["dlycaldt"])
    d["permno"] = d["permno"].astype("int64")
    d = d.drop_duplicates(["dlycaldt", "permno"])
    return d.pivot(index="dlycaldt", columns="permno", values="dlyret").sort_index().astype("float32")


def forecast_vol_daily(df, daily_wide, lookback=126, halflife=21.0, weight_col="weight_unscaled"):
    """Annualised forecast vol of each month's book: the current weights replayed on the
    last `lookback` trading days of daily returns up to the month-end (EWMA, half-life in
    days). A stock with no return on a day gets that day's cross-sectional mean."""
    days = daily_wide.index
    vals = daily_wide.to_numpy(dtype="float32")
    col_pos = pd.Series(np.arange(daily_wide.shape[1]), index=daily_wide.columns)
    out = {}
    held = df.loc[df[weight_col] != 0, ["month", "permno", weight_col]]
    for month, g in held.groupby("month"):
        end = days.searchsorted(pd.Timestamp(month), side="right")   # days <= month-end
        start = max(0, end - lookback)
        if end - start < 40:
            out[month] = np.nan
            continue
        block = vals[start:end]
        cols = col_pos.reindex(g["permno"].to_numpy())
        known = cols.notna().to_numpy()
        hist = np.full((end - start, len(g)), np.nan, dtype="float64")
        hist[:, known] = block[:, cols[known].astype(int).to_numpy()]
        day_mean = np.nanmean(np.where(np.isnan(block), np.nan, block), axis=1)
        hist = np.where(np.isnan(hist), day_mean[:, None], hist)
        sim = np.nan_to_num(hist) @ g[weight_col].to_numpy(dtype=float)
        lam = _ewma_weights(len(sim), halflife)[::-1]
        out[month] = np.sqrt(252 * np.sum(lam * sim**2) / np.sum(lam))
    return pd.Series(out).sort_index()


def forecast_vol_realized(book_returns, halflife=12.0, min_months=6):
    """Annualised EWMA RMS of the book's own past returns, using months < t only."""
    r = book_returns.sort_index()
    out = {}
    vals = r.to_numpy(dtype=float)
    for i, month in enumerate(r.index):
        past = vals[:i]
        if len(past) < min_months:
            out[month] = np.nan
            continue
        lam = _ewma_weights(len(past), halflife)[::-1]
        out[month] = np.sqrt(12 * np.sum(lam * past**2) / np.sum(lam))
    return pd.Series(out)


def choose_leverage(forecast, target=0.10, band=(0.09, 0.11), max_leverage=2.0):
    """L_t = target / forecast_t, re-set only when L_{t-1} * forecast_t leaves the band."""
    lev, rebalanced = [], []
    prev = None
    for f in forecast.to_numpy(dtype=float):
        if not np.isfinite(f) or f <= 0:
            cur = prev if prev is not None else 1.0
            reset = prev is None
        elif prev is None:
            cur, reset = target / f, True
        else:
            in_band = band[0] <= prev * f <= band[1]
            cur, reset = (prev, False) if in_band else (target / f, True)
        cur = min(cur, max_leverage)
        lev.append(cur)
        rebalanced.append(reset)
        prev = cur
    return pd.DataFrame({"leverage": lev, "leverage_reset": rebalanced}, index=forecast.index)


def build_portfolio(pred, cfg, returns_wide=None, score_col="score", post_unscaled=None, daily_wide=None):
    """Steps 1-6. Returns (weights frame with a `weight` column, monthly diagnostics or None).

    post_unscaled: optional function df -> df applied after steps 1-5 and before the vol
    target (cost_study.py uses it for liquidity caps and the small-cap index hedge).
    daily_wide: daily returns matrix (make_daily_returns_wide), for vol_method="daily".
    """
    df = build_unscaled_weights(pred, cfg, score_col=score_col)
    if post_unscaled is not None:
        df = post_unscaled(df)
    if cfg.vol_target is None:
        df["weight"] = df["weight_unscaled"]
        return df, None

    if cfg.vol_method == "daily":
        if daily_wide is None:
            raise ValueError("vol_method='daily' needs daily_wide (see make_daily_returns_wide)")
        f_hold = forecast_vol_daily(df, daily_wide, lookback=cfg.vol_lookback_days,
                                    halflife=cfg.vol_halflife_days)
    elif cfg.vol_method in ("holdings", "blend", "realized"):
        if returns_wide is None:
            raise ValueError("vol targeting needs returns_wide (see make_returns_wide)")
        f_hold = forecast_vol_holdings(
            df, returns_wide, lookback=cfg.vol_lookback, halflife=cfg.vol_halflife
        )
    else:
        raise ValueError("vol_method must be 'holdings', 'realized', 'blend' or 'daily'")

    book = portfolio_returns_and_turnover(df, weight_col="weight_unscaled")
    if cfg.vol_method in ("holdings", "daily"):
        forecast = f_hold
    else:
        f_real = forecast_vol_realized(book["gross_return"], halflife=cfg.vol_halflife)
        f_real = f_real.reindex(f_hold.index)
        if cfg.vol_method == "realized":
            forecast = f_real.fillna(f_hold)
        else:
            forecast = np.sqrt(0.5 * f_hold**2 + 0.5 * f_real**2).fillna(f_hold)

    if cfg.vol_adaptive:
        # z[s] = unlevered book return / its monthly forecast; E[z^2] = 1 if unbiased.
        # The return indexed s is realised during month s+1, so at the end of month t
        # only z[s] for s <= t-1 is known: hence the shift(1).
        realised = book["gross_return"].reindex(forecast.index)
        z2 = (realised / (forecast / np.sqrt(12))) ** 2
        past = z2.shift(1)
        if cfg.vol_adapt_window is None:
            mean_z2 = past.expanding(min_periods=cfg.vol_adapt_min_months).mean()
        else:
            mean_z2 = past.rolling(cfg.vol_adapt_window, min_periods=cfg.vol_adapt_min_months).mean()
        forecast = forecast * np.sqrt(mean_z2).fillna(cfg.vol_scale)
    else:
        forecast = forecast * cfg.vol_scale
    diag = choose_leverage(forecast, cfg.vol_target, cfg.vol_band, cfg.max_leverage)
    diag.insert(0, "forecast_vol_unlevered", forecast)
    diag["forecast_vol_levered"] = diag["forecast_vol_unlevered"] * diag["leverage"]
    df["weight"] = df["weight_unscaled"] * df["month"].map(diag["leverage"]).fillna(1.0)
    return df, diag


def backtest_portfolio(pred, cfg, returns_wide=None, transaction_cost=0.001, score_col="score",
                       cost_model=None, post_unscaled=None, daily_wide=None):
    """build_portfolio + the shared backtest engine. Adds 'diagnostics' and 'config'.

    cost_model: None for the flat transaction_cost, or evaluation.costs.RealisticCosts.
    The weights never depend on it (the vol target works on gross returns).
    """
    df, diag = build_portfolio(pred, cfg, returns_wide=returns_wide, score_col=score_col,
                               post_unscaled=post_unscaled, daily_wide=daily_wide)
    res = run_backtest_on_weights(df, weight_col="weight", transaction_cost=transaction_cost,
                                  cost_model=cost_model)
    res["diagnostics"] = diag
    res["config"] = cfg
    return res


def calibrate_vol_scale(pred_val, cfg, returns_wide, transaction_cost=0.001, n_iter=2, post_unscaled=None,
                        daily_wide=None):
    """vol_scale such that the vol-targeted book hits the target on validation data.

    Run on validation months only, then reuse the returned config on the test period.
    Always uses the flat cost, so a book's weights are the same under every cost model.
    """
    scale = cfg.vol_scale
    for _ in range(n_iter):
        res = backtest_portfolio(pred_val, replace(cfg, vol_scale=scale), returns_wide, transaction_cost,
                                 post_unscaled=post_unscaled, daily_wide=daily_wide)
        realised = np.sqrt(12) * res["returns_with_fees"].std()
        scale *= realised / cfg.vol_target
    return replace(cfg, vol_scale=float(scale))


# -------- trading-cost tools (used by cost_study.py) --------


def cap_weights_by_name(w, months, caps, n_iter=50):
    """Cap |w| at a per-row limit `caps` (same index as w), giving the excess to the other
    names of the same leg and month pro rata to their size. Names pushed over their own
    limit are capped again at the next pass. If a whole leg is at its limits, the leg
    ends up smaller than 1 (the excess is dropped)."""
    w = w.copy()
    caps = caps.reindex(w.index).clip(lower=0).fillna(0.0)
    for sign in (1, -1):
        leg = np.sign(w) == sign
        x = w.abs().where(leg, 0.0)
        c = caps.where(leg, np.inf)
        for _ in range(n_iter):
            over = x > c + 1e-12
            if not over.any():
                break
            excess = (x - c).where(over, 0.0).groupby(months).transform("sum")
            x = x.where(~over, c)
            free = leg & (x > 0) & (x < c - 1e-12)          # names with spare capacity
            share = x.where(free, 0.0)
            share_sum = share.groupby(months).transform("sum").replace(0, np.nan)
            x = x + (excess * share / share_sum).fillna(0.0)
        x = np.minimum(x, c)
        w = w.where(~leg, sign * x)
    return w


def apply_no_trade_band(df, band, leverage=None, weight_col="weight", return_col="ret_fwd1m"):
    """Skip small trades. A name already held on the right side keeps its drifted weight
    when that is within `band` (relative, e.g. 0.25 = +-25%) of its new target; entries,
    exits and sign changes always trade. Leverage changes of the vol target are always
    applied (every held weight is rescaled by L_t / L_{t-1}), so the target still works.

    df: one row per (month, permno) with weight_col and return_col (as build_portfolio
    returns it). Returns a copy with weight_col replaced and the target in weight_target.
    """
    out = df.copy()
    out["weight_target"] = out[weight_col]
    W = out.pivot(index="month", columns="permno", values=weight_col).sort_index()
    R = out.pivot(index="month", columns="permno", values=return_col).reindex_like(W)
    tgt = np.nan_to_num(W.to_numpy(dtype=float))
    in_univ = W.notna().to_numpy()
    ret = np.nan_to_num(R.to_numpy(dtype=float))
    if leverage is not None:
        lev = leverage.reindex(W.index).ffill().fillna(1.0).to_numpy(dtype=float)
    else:
        lev = np.ones(len(W))
    new = np.zeros_like(tgt)
    held = np.zeros(tgt.shape[1])
    for t in range(len(W)):
        if t == 0:
            cur = tgt[t]
        else:
            r = ret[t - 1]
            book = float(np.sum(held * r))
            drifted = held * (1 + r) / (1 + book) * (lev[t] / lev[t - 1])
            keep = (in_univ[t] & (tgt[t] != 0) & (drifted != 0) & (np.sign(tgt[t]) == np.sign(drifted))
                    & (np.abs(tgt[t] - drifted) <= band * np.abs(tgt[t])))
            cur = np.where(keep, drifted, tgt[t])
        cur = np.where(in_univ[t], cur, 0.0)
        new[t] = cur
        held = cur
    stacked = pd.DataFrame(new, index=W.index, columns=W.columns).stack()
    idx = pd.MultiIndex.from_arrays([out["month"], out["permno"]])
    out[weight_col] = stacked.reindex(idx).to_numpy()
    return out


def apply_partial_trading(df, kappa, unit_cost_fn, leverage=None, weight_col="weight", return_col="ret_fwd1m",
                          short_allowed_col=None, close_below=0.25):
    """Cost-aware trading (Garleanu and Pedersen 2013, in its simplest form): each month a
    stock moves only part of the way from its drifted weight to its target,

        w_new = w_drifted + rate * (w_target - w_drifted),   rate = 1 / (1 + kappa * c),

    with c the expected cost per dollar of that trade (commission + half spread + impact
    at its size). Cheap stocks trade almost fully, expensive ones slowly: costs fall, at
    the price of holding a slightly stale book. kappa = 0 is full trading. Stocks that
    leave the universe are sold in full, and so are shorts in stocks that can no longer be
    shorted (short_allowed_col False). A position that is no longer in the target is closed
    in full once it is smaller than close_below x the month's average target position:
    without this, expensive stocks are sold by a fraction each month and never quite reach
    zero (thousands of tiny positions). Each leg is then rescaled to its target size, so the
    book keeps the vol target's leverage and stays dollar-neutral.

    unit_cost_fn(t, trade_abs) -> cost per dollar for every column of the month x permno
    matrix (evaluation.costs.RealisticCosts.unit_cost_fn). Returns a copy of df with
    weight_col replaced, the target in weight_target and the trade rate in trade_rate.
    """
    out = df.copy()
    out["weight_target"] = out[weight_col]
    W = out.pivot(index="month", columns="permno", values=weight_col).sort_index()
    R = out.pivot(index="month", columns="permno", values=return_col).reindex_like(W)
    tgt = np.nan_to_num(W.to_numpy(dtype=float))
    in_univ = W.notna().to_numpy()
    ret = np.nan_to_num(R.to_numpy(dtype=float))
    if short_allowed_col is not None:
        S = out.pivot(index="month", columns="permno", values=short_allowed_col).reindex_like(W)
        shortable = S.fillna(False).to_numpy(dtype=bool)
    else:
        shortable = np.ones_like(tgt, dtype=bool)
    lev = (leverage.reindex(W.index).ffill().fillna(1.0).to_numpy(dtype=float)
           if leverage is not None else np.ones(len(W)))
    new = np.zeros_like(tgt)
    rates = np.ones_like(tgt)
    held = np.zeros(tgt.shape[1])
    for t in range(len(W)):
        if t == 0:
            cur = tgt[t].copy()
        else:
            r = ret[t - 1]
            book = float(np.sum(held * r))
            drifted = held * (1 + r) / (1 + book) * (lev[t] / lev[t - 1])
            gap = tgt[t] - drifted
            c = unit_cost_fn(t, np.abs(gap))
            rate = 1.0 / (1.0 + kappa * c)
            rate = np.where((drifted < 0) & ~shortable[t], 1.0, rate)   # recalled shorts close in full
            cur = drifted + rate * gap
            cur = np.where(in_univ[t], cur, 0.0)
            in_tgt = tgt[t] != 0
            if close_below > 0 and in_tgt.any():
                small = ~in_tgt & (np.abs(cur) < close_below * np.abs(tgt[t][in_tgt]).mean())
                cur = np.where(small, 0.0, cur)
                rate = np.where(small, 1.0, rate)
            rates[t] = rate
            # rescale each leg to its target size (leverage and dollar neutrality kept)
            for sign in (1, -1):
                leg_t = np.sum(np.abs(tgt[t][np.sign(tgt[t]) == sign]))
                leg_c = np.sum(np.abs(cur[np.sign(cur) == sign]))
                if leg_c > 0:
                    cur = np.where(np.sign(cur) == sign, cur * leg_t / leg_c, cur)
        new[t] = cur
        held = cur
    idx = pd.MultiIndex.from_arrays([out["month"], out["permno"]])
    out[weight_col] = pd.DataFrame(new, index=W.index, columns=W.columns).stack().reindex(idx).to_numpy()
    out["trade_rate"] = pd.DataFrame(rates, index=W.index, columns=W.columns).stack().reindex(idx).to_numpy()
    return out


# -------- summary statistics --------


def max_drawdown(returns):
    nav = (1 + returns.fillna(0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


def summarize_backtest(res, months=None, market_fwd=None, band=(0.09, 0.11)):
    """One row of headline numbers for the months in `months` (all months if None)."""
    gross, net = res["returns_no_fees"], res["returns_with_fees"]
    turnover, exposure = res["turnover"], res["gross_exposure"]
    breakdown = res.get("cost_breakdown")
    if months is not None:
        idx = gross.index.intersection(pd.DatetimeIndex(months))
        gross, net, turnover, exposure = gross[idx], net[idx], turnover[idx], exposure.reindex(idx)
        if breakdown is not None:
            breakdown = breakdown.reindex(idx)
    vol = np.sqrt(12) * net.std()
    roll = net.rolling(12).std() * np.sqrt(12)
    row = {
        "ann_return_gross": 12 * gross.mean(),
        "ann_return_net": 12 * net.mean(),
        "ann_vol_net": vol,
        "sharpe_gross": 12 * gross.mean() / (np.sqrt(12) * gross.std()),
        "sharpe_net": 12 * net.mean() / vol,
        "max_drawdown_net": max_drawdown(net),
        "turnover_per_month": turnover.mean(),
        # one-way cost per unit traded at which the net return is zero
        "breakeven_cost_bps": 1e4 * gross.mean() / turnover.mean() if turnover.mean() > 0 else np.nan,
        "avg_gross_exposure": exposure.mean(),
        "rolling12m_vol_in_band": float(roll.dropna().between(*band).mean()) if roll.notna().any() else np.nan,
        "worst_month_net": net.min(),
        "n_months": len(net),
    }
    if breakdown is not None:
        # realistic costs: what they cost per year, and per dollar traded
        for part in ("spread", "impact", "borrow", "total"):
            row[f"cost_{part}_per_year"] = 12 * breakdown[part].mean()
        trading = breakdown["spread"] + breakdown["impact"]
        row["avg_trading_cost_bps"] = 1e4 * trading.mean() / turnover.mean() if turnover.mean() > 0 else np.nan
        row["avg_participation_of_adv"] = breakdown["participation_weighted"].mean()
    if market_fwd is not None:
        m = market_fwd.reindex(net.index)
        ok = m.notna()
        if ok.sum() > 12:
            row["beta_vs_market"] = net[ok].cov(m[ok]) / m[ok].var()
    return row
