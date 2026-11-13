"""
Build the inputs of the realistic cost model: data/processed/cost_inputs.parquet.

One row per (permno, month), from 2003, for every CRSP security (the backtest picks
the ones it trades). Nothing here is a model input: these are the inputs of the cost
model (evaluation/costs.py) and the price / size filters of the cost study.

Columns
    permno, month          month = calendar month end, as in the panels
    prc                    |month-end price|
    mktcap_usd             market cap in dollars
    is_common              passes the CRSP common-stock filter of build_processed_data.py
    size_nyse_pct          share of NYSE common stocks with a market cap <= this one
                           (0.2 = the NYSE 20th percentile, the usual microcap cut)
    size_decile            1..10 by NYSE breakpoints
    half_spread            half the bid-ask spread, as a fraction of the price
    spread_source          quoted | high_low | size_median | month_median
    sigma_daily            standard deviation of daily returns in the month
    adv_usd                average daily dollar volume in the month
    adv_source             measured | size_p10 | month_p10
    adv_usd_6m             average of adv_usd over the last 6 months (t-5..t): a steadier
                           liquidity measure for position sizing (known at t)
    sir                    short interest / shares outstanding (NaN without the data)
    io_13f                 share of the shares held by 13F institutions, as known at t
                           (0 when no institution reports the stock; NaN without 13F data)
    hard_to_borrow         borrow-fee proxy flag (see below)
    borrow_tier            0 = general collateral, 1 = hard to borrow, 2 = special (the
                           fees per tier are in evaluation/costs.py)
    ret_day1               return on the stock's first trading day of the month (for the
                           one-day trading-delay test; NaN without crsp_daily_first_day.parquet)

Measures, all from the month's own trading days (the trade happens at its close):
- half_spread: the average daily closing quoted spread (ask - bid) / mid, halved,
  if at least MIN_QUOTE_DAYS days have quotes. Otherwise the Abdi-Ranaldo (2017)
  estimate from daily close, high and low, if positive. Otherwise the median of the
  month's stocks in the same NYSE size decile.
- sigma_daily: measured if MIN_RET_DAYS daily returns, else the size-decile median.
- adv_usd: measured if positive, else the 10th percentile of the size decile
  (deliberately on the illiquid side).
- hard_to_borrow: short interest in the month's top decile (HTB_SIR_QUANTILE), or
  price below HTB_PRICE, or, for stocks without short-interest data, a market cap
  below the NYSE HTB_NO_SI_SIZE_PCT percentile. Fees are set in evaluation/costs.py.
- borrow_tier 2 ("special"): heavily shorted (top decile of short interest) AND few
  shares to lend (13F ownership below SPECIAL_IO). Lendable supply comes mostly from
  institutions, so short demand against a small institutional float is what makes a
  stock expensive or impossible to borrow (D'Avolio 2002; Asquith, Pathak and Ritter
  2005). Stocks with almost no institutional ownership cannot be shorted at all: the
  cost study removes them from the short leg (--min-short-io, 5% by default).
  These tiers are a proxy: securities-lending data (Markit) would replace them if your
  WRDS account has it (download_wrds_extra.py --only probe tells you).

Usage from the project root, after src/data/download_wrds_costs.py:
    python src/preprocessing/build_cost_inputs.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from preprocessing import features as ft  # noqa: E402
from preprocessing.build_processed_data import CRSP_COMMON_STOCK  # noqa: E402

FIRST_MONTH = "2003-01-31"
MIN_QUOTE_DAYS = 5
MIN_HL_DAYS = 10
MIN_RET_DAYS = 10
MAX_HALF_SPREAD = 0.25       # daily quotes wider than 50% are already dropped in the query
HTB_SIR_QUANTILE = 0.90      # about the share of "special" stocks in lending-market studies
HTB_PRICE = 5.0
HTB_NO_SI_SIZE_PCT = 0.10
SPECIAL_IO = 0.25
REPORT_YEARS = [2005, 2010, 2015, 2020, 2024]


def month_end(s):
    return pd.to_datetime(s) + pd.offsets.MonthEnd(0)


def load_monthly(raw_dir):
    path = raw_dir / "crsp_msf_v2.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run src/data/download_wrds.py first")
    want = ["permno", "mthcaldt", "mthprc", "mthcap", "shrout", "mthvol", "cusip", "hdrcusip"] + list(CRSP_COMMON_STOCK)
    m = pd.read_parquet(path, columns=[c for c in want if c in _parquet_columns(path)])
    m["month"] = month_end(m["mthcaldt"])
    m = m[m["month"] >= FIRST_MONTH].copy()
    m["permno"] = pd.to_numeric(m["permno"], errors="coerce").astype("int64")
    for col in ["mthprc", "mthcap", "shrout", "mthvol"]:
        m[col] = pd.to_numeric(m[col], errors="coerce").astype(float)
    m["prc"] = m["mthprc"].abs()

    # CIZ market cap is in $ thousands (price x shares outstanding in thousands); check it.
    ratio = (m["mthcap"] / (m["prc"] * m["shrout"])).replace([np.inf, -np.inf], np.nan).median()
    if 0.5 < ratio < 2:
        m["mktcap_usd"] = m["mthcap"] * 1e3
        shares_mult = 1e3
    elif 500 < ratio < 2000:
        m["mktcap_usd"] = m["mthcap"]
        shares_mult = 1.0
    else:
        print(f"WARNING: mthcap / (price x shrout) = {ratio:.3g}; using price x shrout x 1000.")
        m["mktcap_usd"] = m["prc"] * m["shrout"] * 1e3
        shares_mult = 1e3
    m["shares"] = m["shrout"] * shares_mult

    ok = pd.Series(True, index=m.index)
    for col, allowed in CRSP_COMMON_STOCK.items():
        if col not in m:
            raise ValueError(f"{col} missing from crsp_msf_v2.parquet")
        ok &= m[col].astype("string").str.strip().str.upper().isin(allowed).fillna(False)
    m["is_common"] = ok
    m["is_nyse"] = ok & (m["primaryexch"].astype("string").str.strip().str.upper() == "N")
    m = m.sort_values(["permno", "month"]).drop_duplicates(["permno", "month"])
    cus = m["cusip"] if "cusip" in m else pd.Series(pd.NA, index=m.index)
    hdr = m["hdrcusip"] if "hdrcusip" in m else pd.Series(pd.NA, index=m.index)
    m["cusip8_hist"] = cus.astype("string").str.strip().str.upper().fillna(
        hdr.astype("string").str.strip().str.upper()).str[:8]
    return m[["permno", "month", "prc", "mktcap_usd", "shares", "mthvol", "is_common", "is_nyse", "cusip8_hist"]]


def _parquet_columns(path):
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


def add_nyse_size(m):
    """size_nyse_pct and size_decile from each month's NYSE common-stock market caps."""
    pct = pd.Series(np.nan, index=m.index)
    for month, g in m.groupby("month"):
        nyse = np.sort(g.loc[g["is_nyse"], "mktcap_usd"].dropna().to_numpy())
        if len(nyse) < 50:
            continue
        cap = g["mktcap_usd"].to_numpy()
        p = np.searchsorted(nyse, cap, side="right") / len(nyse)
        pct.loc[g.index] = np.where(np.isfinite(cap), p, np.nan)
    m["size_nyse_pct"] = pct
    m["size_decile"] = np.clip(np.ceil(pct * 10), 1, 10)
    m.loc[pct.isna(), "size_decile"] = np.nan
    return m


def load_daily(raw_dir):
    path = raw_dir / "crsp_daily_costs.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run src/data/download_wrds_costs.py first")
    d = pd.read_parquet(path)
    d["permno"] = pd.to_numeric(d["permno"], errors="coerce").astype("int64")
    d["month"] = month_end(d["month_start"])
    d = d.drop(columns=["month_start"]).drop_duplicates(["permno", "month"])
    out = pd.DataFrame({"permno": d["permno"], "month": d["month"]})
    qs = d["quoted_spread"].where(d["n_quote_days"] >= MIN_QUOTE_DAYS)
    out["half_spread_quoted"] = qs / 2
    hl_full = np.sqrt((4 * d["hl_prod_mean"]).clip(lower=0))
    out["half_spread_hl"] = (hl_full / 2).where((d["n_hl_days"] >= MIN_HL_DAYS) & (hl_full > 0))
    out["sigma_measured"] = d["sigma_daily"].where(d["n_ret_days"] >= MIN_RET_DAYS)
    out["adv_measured"] = d["adv_mean_usd"].where(d["adv_mean_usd"] > 0)
    out["vol_sum_shares"] = d["vol_sum_shares"]
    return out


def load_short_interest(raw_dir):
    path = raw_dir / "short_interest.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path)
    s["permno"] = pd.to_numeric(s["permno"], errors="coerce")
    s = s.dropna(subset=["permno", "shortint"])
    s["permno"] = s["permno"].astype("int64")
    s["month"] = month_end(s["datadate"])
    # last report in the month; the same permno can appear through two links: keep the larger
    s = s.sort_values(["permno", "month", "datadate", "shortint"])
    s = s.groupby(["permno", "month"], as_index=False).last()
    return s[["permno", "month", "shortint"]]


def fill_by_group(df, col, how, groups=("month", "size_decile")):
    """Fill NaNs of col with a statistic of the month x size-decile group, then of the month."""
    filled = df[col].copy()
    source = pd.Series(np.where(filled.notna(), "measured", None), index=df.index, dtype=object)
    stat = (lambda x: x.median()) if how == "median" else (lambda x: x.quantile(0.10))
    common = df["is_common"]
    for keys, label in [(list(groups), "size"), (["month"], "month")]:
        ref = df.loc[common].groupby(keys)[col].agg(stat).rename("ref")
        ref_vals = df[keys].merge(ref, left_on=keys, right_index=True, how="left")["ref"].to_numpy()
        need = filled.isna() & pd.notna(ref_vals)
        filled[need] = ref_vals[need]
        source[need] = f"{label}_{'median' if how == 'median' else 'p10'}"
    return filled, source


def main():
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw_wrds"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading monthly CRSP...")
    m = add_nyse_size(load_monthly(raw_dir))
    print(f"  {len(m):,} stock-months from {m['month'].min():%Y-%m}, {m['is_common'].mean():.0%} common stocks")

    print("Loading the daily summaries...")
    d = load_daily(raw_dir)
    df = m.merge(d, on=["permno", "month"], how="left")
    print(f"  daily summary found for {df['vol_sum_shares'].notna().mean():.0%} of stock-months "
          f"({df.loc[df['is_common'], 'vol_sum_shares'].notna().mean():.0%} of common-stock months)")

    fd_path = raw_dir / "crsp_daily_first_day.parquet"
    if fd_path.exists():
        fd = pd.read_parquet(fd_path, columns=["permno", "month_start", "ret_day1"])
        fd["permno"] = pd.to_numeric(fd["permno"], errors="coerce").astype("int64")
        fd["month"] = month_end(fd["month_start"])
        fd = fd.drop(columns=["month_start"]).drop_duplicates(["permno", "month"])
        df = df.merge(fd, on=["permno", "month"], how="left")
        print(f"  first-day return found for {df.loc[df['is_common'], 'ret_day1'].notna().mean():.0%} of common-stock months")
    else:
        df["ret_day1"] = np.nan
        print("  no crsp_daily_first_day.parquet: the trading-delay test will be skipped "
              "(python src/data/download_wrds_costs.py --username <you> --first-day-only)")

    # Unit check: shares traded in the month, daily file vs monthly file.
    both = df["is_common"] & (df["mthvol"] > 0) & (df["vol_sum_shares"] > 0)
    vol_ratio = float((df.loc[both, "vol_sum_shares"] / df.loc[both, "mthvol"]).median()) if both.any() else np.nan
    print(f"  unit check: daily volume summed over the month / monthly volume = {vol_ratio:.3f} (should be ~1)")
    if np.isfinite(vol_ratio) and 0.005 < vol_ratio < 0.02:
        print("  -> the daily file counts volume in hundreds of shares: dollar volume x 100")
        df["adv_measured"] *= 100
    elif np.isfinite(vol_ratio) and not 0.5 < vol_ratio < 2:
        print("  WARNING: daily and monthly volume units disagree; dollar volume may be off.")

    # Spread: quoted -> high-low -> size-decile median -> month median
    hs = df["half_spread_quoted"].copy()
    src = pd.Series(np.where(hs.notna(), "quoted", None), index=df.index, dtype=object)
    use_hl = hs.isna() & df["half_spread_hl"].notna()
    hs[use_hl] = df.loc[use_hl, "half_spread_hl"]
    src[use_hl] = "high_low"
    df["half_spread"] = hs.clip(upper=MAX_HALF_SPREAD)
    filled, fsrc = fill_by_group(df, "half_spread", "median")
    df["spread_source"] = np.where(src.notna(), src, fsrc)
    df["half_spread"] = filled

    df["sigma_daily"], _ = fill_by_group(df.assign(sigma_daily=df["sigma_measured"]), "sigma_daily", "median")
    df["adv_usd"], df["adv_source"] = fill_by_group(df.assign(adv_usd=df["adv_measured"]), "adv_usd", "p10")
    df = df.sort_values(["permno", "month"])
    df["adv_usd_6m"] = df.groupby("permno")["adv_usd"].transform(lambda x: x.rolling(6, min_periods=1).mean())

    # Borrow-fee proxy
    si = load_short_interest(raw_dir)
    if si is not None:
        df = df.merge(si, on=["permno", "month"], how="left")
        df = df.sort_values(["permno", "month"])
        df["shortint"] = df.groupby("permno")["shortint"].ffill(limit=2)
        df["sir"] = (df["shortint"] / df["shares"]).where(df["shares"] > 0)
        cut = df.loc[df["is_common"]].groupby("month")["sir"].quantile(HTB_SIR_QUANTILE)
        high_si = df["sir"] >= df["month"].map(cut)
        no_si = df["sir"].isna()
        print(f"  short interest found for {(~no_si[df['is_common']]).mean():.0%} of common-stock months")
    else:
        print("  no short_interest.parquet: hard-to-borrow from price and size only")
        df["sir"] = np.nan
        high_si = pd.Series(False, index=df.index)
        no_si = pd.Series(True, index=df.index)
    small = df["size_nyse_pct"] < HTB_NO_SI_SIZE_PCT
    df["hard_to_borrow"] = (high_si.fillna(False) | (df["prc"] < HTB_PRICE) | (no_si & small)).astype(bool)
    df["_high_si"] = high_si.fillna(False).astype(bool)

    # Institutional ownership as known at t (13F, quarter end + 45 days), for the
    # lendable-supply tiers and the no-short rule of the cost study
    df = df.sort_values(["permno", "month"]).reset_index(drop=True)
    ev = ft.institutional_events(raw_dir, df[["permno", "month", "cusip8_hist", "shares"]])
    if ev is not None:
        hit = ft.latest_event(df, df["permno"], ft.month_num(df["month"]).to_numpy(), ev, ["io"],
                              ft.INST_MAX_AGE_MONTHS)
        df["io_13f"] = df.pop("io").fillna(0.0).astype("float32")   # no 13F holder -> 0
        print(f"  13F ownership found for {hit[df['is_common'].to_numpy()].mean():.0%} of common-stock months "
              "(the rest: no institution reports the stock -> 0)")
    else:
        df["io_13f"] = np.float32(np.nan)
        print("  no inst_13f_holdings.parquet: no lendable-supply tiers, no no-short rule")
    special = df.pop("_high_si") & (df["io_13f"] < SPECIAL_IO)
    df["borrow_tier"] = np.where(special, 2, np.where(df["hard_to_borrow"], 1, 0)).astype("int8")

    keep = ["permno", "month", "prc", "mktcap_usd", "is_common", "size_nyse_pct", "size_decile",
            "half_spread", "spread_source", "sigma_daily", "adv_usd", "adv_source", "adv_usd_6m", "sir",
            "io_13f", "hard_to_borrow", "borrow_tier", "ret_day1"]
    out = df[keep].sort_values(["month", "permno"]).reset_index(drop=True)
    for col in ["prc", "mktcap_usd", "size_nyse_pct", "size_decile", "half_spread", "sigma_daily", "adv_usd", "sir",
                "ret_day1"]:
        out[col] = out[col].astype("float64" if col in ("mktcap_usd", "adv_usd") else "float32")
    out["adv_usd_6m"] = out["adv_usd_6m"].astype("float64")
    for col in ["spread_source", "adv_source"]:
        out[col] = out[col].astype("category")
    out.to_parquet(out_dir / "cost_inputs.parquet", index=False)

    # ---- report ----
    c = out[out["is_common"]].copy()
    c["year"] = c["month"].dt.year
    rows = []
    for year in REPORT_YEARS:
        y = c[c["year"] == year]
        if y.empty:
            continue
        for dec, g in y.groupby("size_decile"):
            rows.append({
                "year": year, "size_decile": int(dec), "stock_months": len(g),
                "median_half_spread_bps": 1e4 * g["half_spread"].median(),
                "median_adv_musd": g["adv_usd"].median() / 1e6,
                "median_sigma_daily": g["sigma_daily"].median(),
                "share_hard_to_borrow": g["hard_to_borrow"].mean(),
                "share_special": (g["borrow_tier"] == 2).mean(),
                "share_io_below_5pct": (g["io_13f"] < 0.05).mean(),
                "share_spread_quoted": (g["spread_source"] == "quoted").mean(),
            })
    rep = pd.DataFrame(rows)
    rep.to_csv(out_dir / "cost_inputs_report.csv", index=False)
    coverage = c.groupby("year")["spread_source"].value_counts(normalize=True).unstack().fillna(0)
    coverage.to_csv(out_dir / "cost_inputs_spread_sources.csv")
    with open(out_dir / "cost_inputs_settings.json", "w") as f:
        json.dump({"first_month": FIRST_MONTH, "min_quote_days": MIN_QUOTE_DAYS,
                   "min_hl_days": MIN_HL_DAYS, "min_ret_days": MIN_RET_DAYS,
                   "htb_sir_quantile": HTB_SIR_QUANTILE, "htb_price": HTB_PRICE,
                   "htb_no_si_size_pct": HTB_NO_SI_SIZE_PCT, "special_io": SPECIAL_IO,
                   "volume_unit_ratio": vol_ratio,
                   "short_interest": si is not None}, f, indent=4)

    print(f"\nSaved {len(out):,} rows to {out_dir / 'cost_inputs.parquet'}")
    if not rep.empty:
        print("\nMedian half-spread (bps) of common stocks, by NYSE size decile (1 = smallest):")
        print(rep.pivot(index="size_decile", columns="year", values="median_half_spread_bps")
              .to_string(float_format=lambda x: f"{x:7.1f}"))
        print("\nMedian daily dollar volume ($ million):")
        print(rep.pivot(index="size_decile", columns="year", values="median_adv_musd")
              .to_string(float_format=lambda x: f"{x:9.2f}"))
        print("\nShare of stocks flagged hard to borrow:")
        print(rep.pivot(index="size_decile", columns="year", values="share_hard_to_borrow")
              .to_string(float_format=lambda x: f"{x:6.0%}"))
        print("\nShare of stocks with less than 5% institutional ownership (not shortable):")
        print(rep.pivot(index="size_decile", columns="year", values="share_io_below_5pct")
              .to_string(float_format=lambda x: f"{x:6.0%}"))
    print("\nSpread source by year (common stocks):")
    print(coverage.loc[[y for y in coverage.index if y in REPORT_YEARS]].to_string(float_format=lambda x: f"{x:6.1%}"))


if __name__ == "__main__":
    main()
