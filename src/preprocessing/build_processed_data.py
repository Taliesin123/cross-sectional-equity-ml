"""
Build the model panel: one row per US common stock and month, with every feature known at
that month's close, next month's return and its class label.

Run from the project root (after the downloads in src/data/):
    python src/preprocessing/build_processed_data.py

Inputs (data/raw_wrds/, from the scripts in src/data/):
    crsp_msf_v2.parquet, ccm_link.parquet, ff_factors_monthly.parquet,
    comp_fundq.parquet, comp_fundq_extra.parquet, filing_dates.parquet,
    crsp_daily_features.parquet, ff_factors_daily.parquet,
    earnings_announcement_days.parquet, short_interest_full.parquet,
    inst_13f_holdings.parquet, inst_13f_managers.parquet,
    ibes_link.parquet, ibes_statsum.parquet, ibes_statsum_unadj.parquet, ibes_actuals.parquet,
    ibes_recommendations.parquet, macro_fred.parquet (each optional: without it, its
    inputs are left out)

Outputs (data/processed/):
    panel_model_filing.parquet   the panel (features scaled, targets, splits, ids)
    feature_groups.json          the model inputs and the settings of the build
    industry_mapping.json        2-digit SIC -> industry id (learned on train)
    split_dates.json             train / validation / test / holdout months
    feature_selection_table.csv  the missing-rate rule, input by input
    preprocessing_report.csv     every logged step and count

Steps
    1. CRSP monthly: clean, monthly return features (lags, momentum, vol, beta), then the
       market features of features.py on each security's full history, next month's
       return as target, and the common-stock universe filter.
    2. Compustat quarterly: clean, two availability dates per filing (earnings release
       for income-statement features, 10-Q / 10-K filing for cash-flow and balance-sheet
       features), year-to-date items turned into quarterly flows, ratios and growth rates,
       and the balance-sheet / earnings-surprise features of features.py.
    3. As-of merge CRSP <- Compustat through the CCM link (permno -> gvkey), one merge per
       availability date, so no item is used before it was public; ratios to market cap.
    4. Labels (bottom 20% / middle / top 20% of next month's return, across all stocks and
       inside the 5 volatility groups), split by date, inputs chosen on training data
       only, missing flags, monthly winsorising, median imputation and robust scaling;
       the market-regime inputs (one value per month) are scaled over time instead.
"""

import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as ft  # noqa: E402

warnings.filterwarnings("ignore")


# -------- settings --------

PANEL_NAME = "filing"   # the panel's name: features are timed with the filing dates

# Without an SEC filing date, a filing is public at the filing deadline plus the
# late-filing extension (NT 10-Q +5 days, NT 10-K +15 days) on the pre-2003 deadlines
# (45 / 90 days after quarter end).
FILING_FALLBACK_DAYS = {"Q": 50, "K": 105}
# Features built from cash-flow-statement items. The cash-flow statement is often only
# published with the 10-Q / 10-K, weeks after the earnings release, so these features use
# the filing date. The balance sheet is treated the same way (features.FILING_TIMED).
# Everything else from Compustat (income statement) uses the earnings release.
CASH_FLOW_FEATURES = [
    "capex_to_sales_qtr", "operating_cf_to_sales_qtr", "investing_cf_to_sales_qtr",
    "financing_cf_to_sales_qtr", "dividends_to_sales_qtr", "cash_change_to_sales_qtr",
    "working_capital_change_to_sales_qtr", "receivables_change_to_sales_qtr",
    "inventory_change_to_sales_qtr", "capxy_qtr_yoy_growth", "oancfy_qtr_yoy_growth",
    "flag_dvy_not_reported",
]
FILING_DATE_FEATURES = CASH_FLOW_FEATURES + ft.FILING_TIMED
DATE_COLS = ["datadate", "availability_date_is", "availability_date_cf", "rdq", "filedate"]
# Accounting data older than this (months between quarter end and the month of use) is
# treated as missing: 1 to 15 months.
ACCOUNTING_AGE = (1, 15)

MODEL_START = "1980-01-31"
SPLITS = {
    "train": ("1980-01-31", "2004-12-31"),
    "val": ("2005-01-31", "2014-12-31"),
    "test": ("2015-01-31", "2024-11-30"),
    # the v4 holdout year: opened once for v4 (2026-10-19), so no longer a clean holdout;
    # reported as "recent", never used for a choice
    "recent": ("2024-12-31", "2025-11-30"),
}
# The new holdout: every month from here on, once CRSP has its next-month return.
HOLDOUT_START = "2025-12-31"

# US common stocks on NYSE / NYSE American / NASDAQ: the CRSP CIZ equivalent of
# SHRCD in (10, 11) and EXCHCD in (1, 2, 3), checked every month.
CRSP_COMMON_STOCK = {
    "sharetype": ["NS"],
    "securitytype": ["EQTY"],
    "securitysubtype": ["COM"],
    "usincflg": ["Y"],
    "issuertype": ["ACOR", "CORP"],
    "primaryexch": ["N", "A", "Q"],
    "conditionaltype": ["RW"],
    "tradingstatusflg": ["A"],
}
CRSP_FLAG_COLS = list(CRSP_COMMON_STOCK)

# Compustat year-to-date items turned into quarterly flows.
YTD_ITEMS = [
    "saley", "revty", "cogsy", "niy", "niity", "oibdpy", "oiadpy", "xsgay", "xrdy",
    "capxy", "xinty", "dpcy", "oancfy", "ivncfy", "fincfy", "dvy", "chechy",
    "wcapchy", "recchy", "invchy",
] + ft.EXTRA_YTD_ITEMS
# True quarterly item used in place of the differenced year-to-date item when present.
QUARTERLY_ITEM_FOR = {
    "saley": "saleq", "cogsy": "cogsq", "xsgay": "xsgaq", "oiadpy": "oiadpq",
    "oibdpy": "oibdpq", "niy": "niq", "xrdy": "xrdq", "xinty": "xintq", "dpcy": "dpq",
}
# Missing in a filing that reports sales -> 0, with a 0/1 flag (structural zeros).
STRUCTURAL_ZERO_ITEMS = ["xrdy", "xsgay", "xinty", "dvy"]

# Input selection, on training data only: drop an input more than 70% missing over the
# last ten training years (Compustat coverage in the early 1980s is structurally low).
SELECTION_WINDOW = ("1995-01-31", "2004-12-31")
MAX_MISSING_RATE = 0.70
# Unscaled copies kept for portfolio construction (vol groups, vol target); not inputs.
RAW_COPY_FEATURES = ["vol_6m", "vol_12m", "beta_12m", "beta_24m", "idio_vol_12m", "vol_d_3m"]

RETURN_FEATURES = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag6", "ret_lag12", "reversal_1m",
    "mom6m1", "mom12m1", "vol_6m", "vol_12m", "beta_12m", "beta_24m", "idio_vol_12m",
]
# Compustat columns carried into the panel that are not inputs.
COMP_METADATA_COLS = [
    "comp_gvkey", "comp_cusip", "comp_tic", "comp_conm", "comp_datadate",
    "comp_availability_date_is", "comp_availability_date_cf", "comp_rdq", "comp_filedate",
    "comp_cf_datadate", "comp_cf_availability_date", "comp_curcdq", "comp_fyearq",
    "comp_fqtr", "comp_fyr", "comp_merge_date",
]
ID_COLUMNS = ["permno", "permco", "month", "target_month", "cusip8", "ticker", "siccd",
              "sic2", "industry_code", "split"]
TARGET_COLUMNS = ["ret_fwd1m", "label20", "label10", "label20_vg", "rank_vg", "rank_all", "vol_group"]
# Volatility groups of the target "inside vol groups" (the groups the vol-neutral books trade)
VOL_GROUPS = 5


# -------- small utils --------

preprocessing_log = []


def log_step(step, message, value=None):
    """Print a step and keep it for preprocessing_report.csv."""
    preprocessing_log.append({"step": step, "message": message, "value": value})
    if value is None:
        print(f"[{step}] {message}")
    else:
        print(f"[{step}] {message}: {value}")


def find_project_root(start_path=None):
    """The first folder, from the current one upwards, with data/ and src/."""
    start_path = Path(start_path or Path.cwd()).resolve()
    for path in [start_path] + list(start_path.parents):
        if (path / "data").exists() and (path / "src").exists():
            return path
    raise FileNotFoundError("Could not find the project root (a folder with data/ and src/).")


def month_end(date_series):
    return pd.to_datetime(date_series, errors="coerce") + pd.offsets.MonthEnd(0)


def safe_divide(numerator, denominator, positive_denominator=True, eps=1e-8):
    """Ratio, NaN when the denominator is (close to) zero, or not positive."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    if positive_denominator:
        valid = denominator > eps
    else:
        valid = denominator.abs() > eps
    out = numerator / denominator
    out = out.where(valid)
    return out.replace([np.inf, -np.inf], np.nan)


def _to_float(s):
    """Nullable WRDS numbers (Float64 / Int64 with pd.NA) -> plain float64 with NaN."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


def downcast_panel(panel):
    """Features as float32 (the model uses float32 anyway; float64 would double memory).
    Returns, prices and sizes stay float64."""
    keep_float64 = {"ret", "retx", "ret_fwd1m", "mkt_ret", "mktcap", "prc", "shrout", "vol"}
    for c in panel.columns:
        if panel[c].dtype == "float64" and c not in keep_float64:
            panel[c] = panel[c].astype("float32")
    gc.collect()
    return panel


# -------- 1. CRSP --------


def read_market_return(raw_dir):
    """Monthly CRSP value-weighted market return by month-end (Fama-French mktrf + rf)."""
    path = raw_dir / "ff_factors_monthly.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run src/data/download_wrds.py to the end")
    ff = pd.read_parquet(path)
    ff["month"] = pd.to_datetime(ff["date"]) + pd.offsets.MonthEnd(0)
    mkt = _to_float(ff["mktrf"]) + _to_float(ff["rf"])
    if mkt.abs().median() > 0.5:  # percent units -> decimals
        mkt = mkt / 100.0
    return pd.Series(mkt.to_numpy(), index=ff["month"]).groupby(level=0).last()


def load_crsp(raw_dir):
    """crsp.msf_v2 (CIZ) -> one clean row per permno and month.

    CIZ returns already include delisting returns: nothing is added. Units: mthcap in
    $ thousands, shrout in thousands of shares, mthvol in shares.
    """
    path = raw_dir / "crsp_msf_v2.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run src/data/download_wrds.py first")
    raw = pd.read_parquet(path)
    log_step("load_crsp", "raw shape (crsp.msf_v2)", raw.shape)
    crsp = pd.DataFrame({
        "permno": pd.to_numeric(raw["permno"], errors="coerce").astype("Int64"),
        "permco": pd.to_numeric(raw["permco"], errors="coerce").astype("Int64"),
        "month": month_end(raw["mthcaldt"]),   # 1986-12-31 = the return of December 1986
        "ret": _to_float(raw["mthret"]),
        "retx": _to_float(raw["mthretx"]),
        "cusip": raw["cusip"].astype("string").str.strip().str.upper(),
        "hdrcusip": raw["hdrcusip"].astype("string").str.strip().str.upper(),
        "ticker": raw["ticker"].astype("string").str.strip().str.upper(),
        "siccd": raw["siccd"],
        "prc": _to_float(raw["mthprc"]),
        "shrout": _to_float(raw["shrout"]),
        "mktcap": _to_float(raw["mthcap"]),
        "vol": _to_float(raw["mthvol"]),
    })
    for col in CRSP_FLAG_COLS:
        crsp[col] = raw[col].astype("string").str.strip().str.upper()
    del raw
    crsp["mkt_ret"] = crsp["month"].map(read_market_return(raw_dir))
    log_step("load_crsp", "months without a market return", int(crsp["mkt_ret"].isna().sum()))
    crsp = crsp.dropna(subset=["month", "permno"])

    # CUSIPs: the header CUSIP is kept as an identifier (cusip8); the historical one
    # (cusip8_hist) links 13F holdings, which are reported under the CUSIP of the day.
    for col in ["cusip", "hdrcusip"]:
        crsp.loc[crsp[col].isin(["", "nan", "NaN", "None"]), col] = pd.NA
    crsp["cusip8_hist"] = crsp["cusip"].fillna(crsp["hdrcusip"]).str[:8]
    crsp["cusip8"] = crsp["hdrcusip"].fillna(crsp["cusip"]).str[:8]

    log_step("clean_crsp_returns", "missing stock returns before drop", int(crsp["ret"].isna().sum()))
    log_step("clean_crsp_returns", "returns below -100%", int((crsp["ret"] < -1).sum()))
    crsp = crsp.dropna(subset=["ret"])
    crsp = crsp[crsp["ret"] >= -1].copy()   # -1 is a complete loss; below is impossible

    # Industry: 2-digit SIC code of the month (CRSP SIC codes change over time).
    crsp["siccd"] = pd.to_numeric(crsp["siccd"], errors="coerce")
    crsp.loc[crsp["siccd"] <= 0, "siccd"] = np.nan
    crsp["siccd"] = crsp["siccd"].astype("Int64")
    crsp["sic2"] = crsp["siccd"].astype("string").str.zfill(4).str[:2]
    crsp.loc[crsp["siccd"].isna(), "sic2"] = pd.NA
    crsp["industry_code"] = crsp["sic2"]
    log_step("clean_crsp_industry", "missing industry code", int(crsp["industry_code"].isna().sum()))

    crsp = crsp.sort_values(["permno", "month"])
    n_dups = int(crsp.duplicated(["permno", "month"]).sum())
    crsp = crsp.drop_duplicates(["permno", "month"], keep="first")
    log_step("clean_crsp_duplicates", "duplicate permno-month rows removed", n_dups)
    crsp = crsp.sort_values(["permno", "month"]).reset_index(drop=True)
    log_step("load_crsp", "clean shape", crsp.shape)
    return crsp


# Calendar-window checks: no window may span a gap in a stock's history.


def valid_lag_window(df, first_lag, last_lag):
    first_m = df.groupby("permno")["month_num"].shift(first_lag)
    last_m = df.groupby("permno")["month_num"].shift(last_lag)
    return (
        first_m.notna()
        & last_m.notna()
        & ((df["month_num"] - first_m) == first_lag)
        & ((first_m - last_m) == (last_lag - first_lag))
    )


def valid_rolling_window(df, window):
    first_m = df.groupby("permno")["month_num"].shift(window - 1)
    return first_m.notna() & ((df["month_num"] - first_m) == (window - 1))


def rolling_mean_by_stock(df, col, window):
    return (
        df.groupby("permno")[col]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def build_crsp_return_features(crsp):
    """Lagged returns, momentum, volatility, beta and idiosyncratic volatility."""
    crsp = crsp.sort_values(["permno", "month"]).reset_index(drop=True)
    crsp["month_num"] = crsp["month"].dt.year * 12 + crsp["month"].dt.month

    for lag in [1, 2, 3, 6, 12]:
        lag_ret = crsp.groupby("permno")["ret"].shift(lag)
        lag_month = crsp.groupby("permno")["month_num"].shift(lag)
        valid_lag = lag_month.notna() & ((crsp["month_num"] - lag_month) == lag)
        crsp[f"ret_lag{lag}"] = lag_ret.where(valid_lag)

    crsp["reversal_1m"] = -crsp["ret"]

    # Momentum skips the most recent month (short-term reversal): 6-1 sums the returns of
    # months t-5 .. t-1 counted from the row's month as t+1, i.e. lags 2 to 6 of the row;
    # 12-1 uses lags 2 to 12. Sums of log returns.
    ret_shift2 = crsp.groupby("permno")["ret"].shift(2)
    crsp["_log_ret_shift2"] = np.log1p(ret_shift2)
    mom6 = (
        crsp.groupby("permno")["_log_ret_shift2"]
        .rolling(window=5, min_periods=5)
        .sum()
        .reset_index(level=0, drop=True)
    )
    crsp["mom6m1"] = np.expm1(mom6).where(valid_lag_window(crsp, 2, 6))
    mom12 = (
        crsp.groupby("permno")["_log_ret_shift2"]
        .rolling(window=11, min_periods=11)
        .sum()
        .reset_index(level=0, drop=True)
    )
    crsp["mom12m1"] = np.expm1(mom12).where(valid_lag_window(crsp, 2, 12))

    for window in [6, 12]:
        vol = (
            crsp.groupby("permno")["ret"]
            .rolling(window=window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )
        crsp[f"vol_{window}m"] = vol.where(valid_rolling_window(crsp, window))

    # Beta and idiosyncratic volatility from rolling moments (faster than regressions).
    crsp["_ret_sq"] = crsp["ret"] ** 2
    crsp["_mkt_sq"] = crsp["mkt_ret"] ** 2
    crsp["_ret_mkt"] = crsp["ret"] * crsp["mkt_ret"]
    for window in [12, 24]:
        print(f"rolling beta over {window} months...")
        mean_ret = rolling_mean_by_stock(crsp, "ret", window)
        mean_mkt = rolling_mean_by_stock(crsp, "mkt_ret", window)
        mean_ret_sq = rolling_mean_by_stock(crsp, "_ret_sq", window)
        mean_mkt_sq = rolling_mean_by_stock(crsp, "_mkt_sq", window)
        mean_ret_mkt = rolling_mean_by_stock(crsp, "_ret_mkt", window)

        cov_rm = mean_ret_mkt - mean_ret * mean_mkt
        var_m = mean_mkt_sq - mean_mkt**2
        var_r = mean_ret_sq - mean_ret**2
        ok = valid_rolling_window(crsp, window) & (var_m > 1e-12)
        crsp[f"beta_{window}m"] = (cov_rm / var_m).where(ok)
        if window == 12:
            # residual variance = Var(r) - Cov(r, m)^2 / Var(m)
            resid_var = var_r - (cov_rm**2 / var_m)
            crsp["idio_vol_12m"] = np.sqrt(resid_var.clip(lower=0)).where(ok)

    crsp = crsp.drop(columns=["month_num", "_log_ret_shift2", "_ret_sq", "_mkt_sq", "_ret_mkt"])
    log_step("crsp_features", "number of return features", len(RETURN_FEATURES))
    return crsp


def build_crsp_target(crsp):
    """Target = next month's return; kept only when the next row is exactly next month."""
    crsp = crsp.sort_values(["permno", "month"]).reset_index(drop=True)
    crsp["ret_fwd1m"] = crsp.groupby("permno")["ret"].shift(-1)
    crsp["target_month"] = crsp.groupby("permno")["month"].shift(-1)
    valid = crsp["target_month"].notna() & (crsp["target_month"] == crsp["month"] + pd.offsets.MonthEnd(1))
    log_step("crsp_target", "rows before target filtering", len(crsp))
    log_step("crsp_target", "valid next-month targets", int(valid.sum()))
    return crsp.loc[valid].reset_index(drop=True)


def apply_universe_filter(crsp):
    """Keep US common stocks on NYSE / NYSE American / NASDAQ, month by month.

    Applied after the features and the target are built, so rolling windows and the
    next-month return use each security's full history.
    """
    log_step("universe_filter", "rows before", len(crsp))
    keep = pd.Series(True, index=crsp.index)
    for col, allowed in CRSP_COMMON_STOCK.items():
        ok = crsp[col].isin(allowed).fillna(False)
        log_step("universe_filter", f"rows failing {col} in {allowed}", int((~ok).sum()))
        keep &= ok
    crsp = crsp.loc[keep].reset_index(drop=True)
    log_step("universe_filter", "rows after (US common stocks, NYSE/AMEX/NASDAQ)", len(crsp))
    return crsp


# -------- 2. Compustat --------


def load_filing_dates(raw_dir):
    """SEC filing dates (src/data/download_filing_dates.py), or None if not downloaded.
    Columns: gvkey, period (period of report, may be missing), filedate, kind (Q / K)."""
    path = raw_dir / "filing_dates.parquet"
    if not path.exists():
        return None
    f = pd.read_parquet(path)
    f["gvkey"] = f["gvkey"].astype("string").str.strip()
    f["filedate"] = pd.to_datetime(f["filedate"], errors="coerce")
    f["period"] = pd.to_datetime(f["period"], errors="coerce") if "period" in f else pd.NaT
    f["kind"] = f["kind"].astype("string").str.upper()
    return f.dropna(subset=["gvkey", "filedate", "kind"])


def match_filing_dates(comp, filings):
    """Filing date of each Compustat row: the 10-Q (fiscal Q1-Q3) or 10-K (Q4) whose
    period of report is the row's datadate. Where the source has no period of report,
    the first such filing within 120 days (10-Q) / 180 days (10-K) after datadate, each
    filing going to one quarter only (the earliest), so a late 10-Q of the previous
    quarter cannot be taken for this quarter's. Matches before quarter end + 10 days or
    more than 15 days before the earnings release are dropped as implausible."""
    fq = pd.to_numeric(comp["fqtr"], errors="coerce").astype("float64")
    kind = pd.Series(np.where(fq.eq(4), "K", "Q"), index=comp.index)
    out = pd.Series(pd.NaT, index=comp.index, dtype="datetime64[ns]")
    rows = comp[["gvkey", "datadate"]].assign(kind=kind.to_numpy(), _row=np.arange(len(comp)))
    rows = rows.dropna(subset=["gvkey", "datadate"])
    rows["gvkey"] = rows["gvkey"].astype("string")
    rows["datadate"] = rows["datadate"].astype("datetime64[ns]")
    by_period = 0
    p = filings.dropna(subset=["period"])
    if len(p):
        p = p.groupby(["gvkey", "period", "kind"], as_index=False)["filedate"].min()
        p["period"] = p["period"].astype("datetime64[ns]")
        m = rows.merge(p, left_on=["gvkey", "datadate", "kind"], right_on=["gvkey", "period", "kind"], how="inner")
        out.iloc[m["_row"].to_numpy()] = m["filedate"].to_numpy()
        by_period = len(m)
    # window matching only with filings that carry no period of report
    no_period = filings[filings["period"].isna()]
    rest = rows[out.iloc[rows["_row"].to_numpy()].isna().to_numpy()]
    by_window = 0
    for k, window in (("Q", 120), ("K", 180)):
        pending = rest[rest["kind"] == k]
        right = no_period.loc[no_period["kind"] == k, ["gvkey", "filedate"]].drop_duplicates()
        right = right.assign(filedate=right["filedate"].astype("datetime64[ns]"))
        used = pd.Index([])
        for _ in range(6):
            if pending.empty or right.empty:
                break
            avail = right[~pd.MultiIndex.from_frame(right[["gvkey", "filedate"]]).isin(used)]
            m = pd.merge_asof(pending.sort_values("datadate"), avail.sort_values("filedate"),
                              left_on="datadate", right_on="filedate", by="gvkey", direction="forward",
                              tolerance=pd.Timedelta(days=window), allow_exact_matches=False)
            m = m.dropna(subset=["filedate"]).sort_values(["gvkey", "filedate", "datadate"])
            first = ~m.duplicated(["gvkey", "filedate"])
            win = m[first]
            out.iloc[win["_row"].to_numpy()] = win["filedate"].to_numpy()
            by_window += len(win)
            used = used.append(pd.MultiIndex.from_frame(win[["gvkey", "filedate"]])) if len(used) else \
                pd.MultiIndex.from_frame(win[["gvkey", "filedate"]])
            pending = pending[pending["_row"].isin(m.loc[~first, "_row"])]
    lag = (out - comp["datadate"]).dt.days
    ok = lag.between(10, 400)
    if "rdq" in comp:
        rdq = pd.to_datetime(comp["rdq"], errors="coerce")
        ok &= ~((out - rdq).dt.days < -15)
    log_step("filing_dates", "Compustat rows matched by period of report", by_period)
    log_step("filing_dates", "Compustat rows matched by first filing after quarter end", by_window)
    log_step("filing_dates", "matches dropped as implausible", int((out.notna() & ~ok).sum()))
    return out.where(ok)


def add_filing_availability(comp, raw_dir):
    """Two availability dates per filing (the first month-end on or after):
    - availability_date_is (income-statement features): the day after the earnings
      release (rdq) or the filing, whichever came first; without either, quarter end
      + 50 days (10-Q) / 105 days (10-K).
    - availability_date_cf (cash-flow and balance-sheet features): the day after the SEC
      filing; without a filing date, the later of the release and that deadline. Never
      before the income-statement date.
    """
    filings = load_filing_dates(raw_dir)
    if filings is None:
        print("No data/raw_wrds/filing_dates.parquet: cash-flow timing uses the filing "
              "deadlines (run src/data/download_filing_dates.py for the actual dates).")
        comp["filedate"] = pd.NaT
    else:
        comp["filedate"] = match_filing_dates(comp, filings)
    dd = comp["datadate"]
    fq = pd.to_numeric(comp["fqtr"], errors="coerce").astype("float64")
    is_q4 = fq.eq(4) | fq.isna()
    deadline = dd + pd.to_timedelta(np.where(is_q4, FILING_FALLBACK_DAYS["K"], FILING_FALLBACK_DAYS["Q"]), unit="D")
    rdq = pd.to_datetime(comp["rdq"], errors="coerce")
    rdq_ok = rdq.notna() & (rdq - dd).dt.days.between(0, 365)
    fd = comp["filedate"]
    has_f = fd.notna()
    one = pd.Timedelta(days=1)
    first_public = pd.concat([rdq.where(rdq_ok), fd], axis=1).min(axis=1)       # NaT if neither
    is_date = (first_public + one).where(rdq_ok | has_f, deadline)
    cf_date = (fd + one).where(has_f, pd.concat([(rdq + one).where(rdq_ok), deadline], axis=1).max(axis=1))
    cf_date = pd.concat([cf_date, is_date], axis=1).max(axis=1)
    comp["availability_date_is"] = is_date + pd.offsets.MonthEnd(0)
    comp["availability_date_cf"] = cf_date + pd.offsets.MonthEnd(0)

    recent = dd >= "1996-01-01"
    log_step("filing_dates", "rows with a filing date (all / since 1996)",
             f"{has_f.mean():.3f} / {has_f[recent].mean():.3f}")
    for k, m in (("10-Q (Q1-Q3)", ~is_q4), ("10-K (Q4)", is_q4)):
        lag = (fd - dd).dt.days[m & has_f]
        if len(lag):
            log_step("filing_dates", f"median days quarter end -> {k}", float(lag.median()))
    both = has_f & rdq_ok
    if both.any():
        gap = (fd - rdq).dt.days[both]
        log_step("filing_dates", "median days earnings release -> filing", float(gap.median()))
        log_step("filing_dates", "share of filings before the release date", float((gap < 0).mean()))
    src = np.select([rdq_ok & has_f, rdq_ok, has_f], ["release+filing", "release only", "filing only"], "deadline")
    for name, share in pd.Series(src[recent.to_numpy()]).value_counts(normalize=True).items():
        log_step("filing_dates", f"date source since 1996: {name}", float(share))
    return comp


def load_and_clean_compustat(raw_dir):
    """comp.fundq (+ the extra items), standard filters (INDL / C / STD / D), two
    availability dates per filing, one row per gvkey and datadate."""
    path = raw_dir / "comp_fundq.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run src/data/download_wrds.py first")
    comp = pd.read_parquet(path)
    log_step("load_compustat", "raw shape (comp.fundq)", comp.shape)
    comp.columns = comp.columns.str.strip().str.lower()
    text_cols = ["gvkey", "cusip", "tic", "conm", "datacqtr", "datafqtr", "indfmt", "consol",
                 "popsrc", "datafmt", "curcdq"]
    for col in comp.columns:
        if col not in text_cols and col not in ("datadate", "rdq") and pd.api.types.is_numeric_dtype(comp[col]):
            comp[col] = _to_float(comp[col])
    for col in text_cols:
        if col in comp.columns:
            comp[col] = comp[col].astype("string").str.strip()
    for col in ["indfmt", "consol", "popsrc", "datafmt", "curcdq", "cusip", "tic"]:
        if col in comp.columns:
            comp[col] = comp[col].str.upper()
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    comp["rdq"] = pd.to_datetime(comp["rdq"], errors="coerce")
    comp = ft.merge_fundq_extra(comp, raw_dir, log_step)

    log_step("clean_compustat_dates", "missing datadate before drop", int(comp["datadate"].isna().sum()))
    comp = comp.dropna(subset=["datadate"]).copy()
    lag_days = (comp["rdq"] - comp["datadate"]).dt.days
    log_step("clean_compustat_dates", "filings with a usable rdq", float(lag_days.between(0, 365).mean()))
    log_step("clean_compustat_dates", "median rdq - quarter end (days)", float(lag_days[lag_days.between(0, 365)].median()))
    comp.loc[comp["gvkey"].isin(["", "nan", "NaN", "None"]), "gvkey"] = pd.NA

    # Standard Compustat filters (the download already applies them; checked again here).
    n_before = len(comp)
    comp = comp[comp["indfmt"].eq("INDL") & comp["consol"].eq("C") & comp["datafmt"].eq("STD")
                & comp["popsrc"].eq("D")].copy()
    log_step("clean_compustat_filters", "rows removed by format filters", int(n_before - len(comp)))
    comp = add_filing_availability(comp, raw_dir)

    # One row per firm and date: keep the most complete row.
    numeric_cols = comp.select_dtypes(include=[np.number]).columns.tolist()
    n_dups = int(comp.duplicated(["gvkey", "datadate"]).sum())
    comp["_n_non_missing"] = comp[numeric_cols].notna().sum(axis=1)
    comp = (
        comp.sort_values(["gvkey", "datadate", "_n_non_missing"], ascending=[True, True, False])
        .drop_duplicates(["gvkey", "datadate"], keep="first")
        .drop(columns=["_n_non_missing"])
    )
    log_step("clean_compustat_duplicates", "duplicate firm-date rows removed", n_dups)
    log_step("load_compustat", "clean shape", comp.shape)
    return comp


def build_compustat_features(comp):
    """Compustat features. Returns (comp, info), info listing the features and flags.

    Most items are year-to-date (Q4 holds about 12 months of activity, Q1 three), so they
    are turned into quarterly flows before any ratio or growth rate is computed.
    """
    comp = comp.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    log_step("compustat_features", "loaded clean Compustat shape", comp.shape)

    created_features = []
    quarterly_flow_cols = []

    def has_col(col):
        return col in comp.columns

    def to_num(col):
        return pd.to_numeric(comp[col], errors="coerce")

    def add_feature(name, values):
        comp[name] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        created_features.append(name)

    # ---- year-to-date -> quarterly flows ----
    # Sorted by firm, fiscal year and quarter, the previous row is the previous fiscal
    # quarter when firm and fiscal year are the same and the quarter number is one less.
    comp = comp.sort_values(["gvkey", "fyearq", "fqtr", "datadate"])
    for col in ["fyearq", "fqtr"]:
        comp[col] = pd.to_numeric(comp[col], errors="coerce")
    ytd_cols = [c for c in YTD_ITEMS if c in comp.columns]
    comp = ft.structural_zero_extra_ytd(comp, log_step)

    valid_diff = (comp["gvkey"].eq(comp["gvkey"].shift(1))
                  & comp["fyearq"].eq(comp["fyearq"].shift(1))
                  & comp["fqtr"].eq(comp["fqtr"].shift(1) + 1))
    for col in ytd_cols:
        current_ytd = pd.to_numeric(comp[col], errors="coerce")
        previous_ytd = current_ytd.shift(1)
        qtr_flow = pd.Series(np.nan, index=comp.index, dtype="float64")
        q1 = comp["fqtr"].eq(1)                           # Q1: the YTD value is the flow
        qtr_flow.loc[q1] = current_ytd.loc[q1]
        mask = comp["fqtr"].gt(1) & valid_diff            # Q2-Q4: difference of YTD values
        qtr_flow.loc[mask] = current_ytd.loc[mask] - previous_ytd.loc[mask]
        comp[f"{col}_qtr"] = qtr_flow.replace([np.inf, -np.inf], np.nan)
        quarterly_flow_cols.append(f"{col}_qtr")
    log_step("compustat_ytd_to_qtr", "YTD variables converted", len(quarterly_flow_cols))

    # ---- true quarterly items ----
    # Where comp.fundq has the quarterly item (saleq, ...) it replaces the differenced YTD
    # value: same quantity, measured directly, and available when the previous quarter is not.
    for ytd_col, q_col in QUARTERLY_ITEM_FOR.items():
        qtr_col = f"{ytd_col}_qtr"
        if q_col in comp.columns and qtr_col in comp.columns:
            q = to_num(q_col)
            before = comp[qtr_col].notna().mean()
            comp[qtr_col] = q.where(q.notna(), comp[qtr_col])
            log_step("compustat_quarterly_items", f"{qtr_col}: share available, YTD-only -> with {q_col}",
                     f"{before:.3f} -> {comp[qtr_col].notna().mean():.3f}")

    # ---- structural zeros ----
    # A firm that reports sales but no R&D / SG&A / interest / dividends has, in most
    # cases, none to report: the value is 0, not unknown (Koh & Reeb 2015); imputing the
    # cross-sectional median would invent spending. The flow is set to 0 only when the item
    # is missing in a filing that reports sales, and a 0/1 flag tells a reported 0 from a
    # filled one. A YTD value that exists but cannot be differenced stays NaN (unknown).
    flag_features = []
    reports_sales = to_num("saley").notna() | to_num("saleq").notna()
    for col in STRUCTURAL_ZERO_ITEMS:
        qtr_col = f"{col}_qtr"
        not_reported = reports_sales & to_num(col).isna()
        q_col = QUARTERLY_ITEM_FOR.get(col)
        if q_col in comp.columns:
            not_reported &= to_num(q_col).isna()
        comp.loc[not_reported, qtr_col] = 0.0
        flag = f"flag_{col}_not_reported"
        comp[flag] = not_reported.astype("int8")
        flag_features.append(flag)
        log_step("compustat_structural_zeros", f"{col}: share of sales-reporting filings set to 0",
                 float(not_reported[reports_sales].mean()))

    # ---- size, margins and ratios to quarterly sales ----
    s = to_num("saley_qtr")
    add_feature("log_sales_qtr", np.log1p(s.where(s > 0)))
    r = to_num("revty_qtr")
    add_feature("log_revenue_qtr", np.log1p(r.where(r > 0)))
    # About 11% of Compustat filings are in CAD. Ratios and growth rates are currency-free,
    # log levels are not: they are blanked for non-USD filings.
    non_usd = comp["curcdq"].notna() & comp["curcdq"].ne("USD")
    comp.loc[non_usd, ["log_sales_qtr", "log_revenue_qtr"]] = np.nan
    log_step("compustat_currency", "non-USD filings: log levels set to NaN", int(non_usd.sum()))

    sb = to_num("saley_qtr")
    add_feature("gross_margin_qtr", safe_divide(sb - to_num("cogsy_qtr"), sb))
    for name, item in [
        ("oibdp_margin_qtr", "oibdpy_qtr"), ("operating_margin_qtr", "oiadpy_qtr"),
        ("net_margin_qtr", "niy_qtr"), ("net_income_total_margin_qtr", "niity_qtr"),
        ("sga_to_sales_qtr", "xsgay_qtr"), ("rd_to_sales_qtr", "xrdy_qtr"),
        ("capex_to_sales_qtr", "capxy_qtr"), ("interest_to_sales_qtr", "xinty_qtr"),
        ("depreciation_to_sales_qtr", "dpcy_qtr"), ("operating_cf_to_sales_qtr", "oancfy_qtr"),
        ("investing_cf_to_sales_qtr", "ivncfy_qtr"), ("financing_cf_to_sales_qtr", "fincfy_qtr"),
        ("dividends_to_sales_qtr", "dvy_qtr"), ("cash_change_to_sales_qtr", "chechy_qtr"),
        ("working_capital_change_to_sales_qtr", "wcapchy_qtr"),
        ("receivables_change_to_sales_qtr", "recchy_qtr"), ("inventory_change_to_sales_qtr", "invchy_qtr"),
    ]:
        add_feature(name, safe_divide(to_num(item), sb))

    # ---- year-over-year growth: same fiscal quarter one year earlier ----
    comp = comp.sort_values(["gvkey", "datadate"])
    comp["_month_num"] = comp["datadate"].dt.year * 12 + comp["datadate"].dt.month
    month_gap = comp["_month_num"] - comp.groupby("gvkey")["_month_num"].shift(4)
    valid_lag4 = month_gap.between(11, 13) & comp["fqtr"].eq(comp.groupby("gvkey")["fqtr"].shift(4))
    for col in ["saley_qtr", "revty_qtr", "oibdpy_qtr", "oiadpy_qtr", "niy_qtr", "cogsy_qtr",
                "xsgay_qtr", "xrdy_qtr", "capxy_qtr", "oancfy_qtr"]:
        lag4 = pd.to_numeric(comp.groupby("gvkey")[col].shift(4), errors="coerce")
        g = safe_divide(to_num(col), lag4, positive_denominator=True) - 1
        add_feature(f"{col}_yoy_growth", g.where(valid_lag4))

    # ---- balance sheet, rest of the cash-flow statement, earnings surprises ----
    # (+ the amounts behind the ratios to market value, formed after the merge)
    comp = ft.build_accounting_features(comp, "gvkey", add_feature, log_step)
    # amounts in CAD cannot be divided by a market cap in USD
    non_usd = comp["curcdq"].notna() & comp["curcdq"].ne("USD")
    helpers = [c for c in ft.HELPERS_IS + ft.HELPERS_FILING if c in comp.columns]
    comp.loc[non_usd, helpers] = np.nan
    comp = comp.drop(columns=["_month_num"])

    identifier_cols = ["gvkey", "cusip", "tic", "conm", "datadate", "availability_date_is",
                       "availability_date_cf", "rdq", "filedate", "curcdq", "fyearq", "fqtr", "fyr"]
    log_step("compustat_features", "number of quarterly flow variables", len(quarterly_flow_cols))
    log_step("compustat_features", "number of engineered Compustat features", len(created_features))
    comp = comp[list(dict.fromkeys([c for c in identifier_cols if c in comp.columns]
                                   + created_features + flag_features))]
    for col in created_features:
        if comp[col].dtype == "float64":
            comp[col] = comp[col].astype("float32")
    gc.collect()
    return comp, {"features": created_features, "flags": flag_features}


# -------- 3. Merge CRSP + Compustat --------


def attach_gvkey(crsp, raw_dir):
    """Compustat gvkey of each CRSP stock-month from the CCM link history.

    A link applies from linkdt to linkenddt (open-ended links run to today). When a
    permno has several valid links in a month, the primary one (P) wins over C, and
    LC (researched) over LU.
    """
    link = pd.read_parquet(raw_dir / "ccm_link.parquet")
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce").astype("Int64")
    link = link.dropna(subset=["permno", "gvkey"]).copy()
    link["gvkey"] = link["gvkey"].astype("string").str.strip()
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
    link["prio"] = link["linkprim"].eq("P").astype(int) * 2 + link["linktype"].eq("LC").astype(int)

    crsp = crsp.reset_index(drop=True)
    keys = crsp[["permno", "month"]].copy()
    keys["_row"] = np.arange(len(crsp))
    m = keys.merge(link[["permno", "gvkey", "linkdt", "linkenddt", "prio"]], on="permno", how="inner")
    valid = (m["month"] >= m["linkdt"]) & (m["month"] <= m["linkenddt"] + pd.offsets.MonthEnd(0))
    m = m.loc[valid].sort_values(["_row", "prio"], ascending=[True, False]).drop_duplicates("_row")

    crsp["gvkey"] = pd.Series(pd.NA, index=crsp.index, dtype="string")
    crsp.loc[m["_row"].to_numpy(), "gvkey"] = m["gvkey"].to_numpy()
    linked = crsp["gvkey"].notna()
    log_step("ccm_link", "stock-months with a valid CCM link", float(linked.mean()))
    log_step("ccm_link", "... since 1980", float(linked[crsp["month"] >= MODEL_START].mean()))
    return crsp


def prepare_compustat_for_asof(comp, availability_col, feature_cols, flag_cols=()):
    """Right-hand table of an as-of merge: one row per (gvkey, month-end it becomes usable).

    Filings with none of these features are dropped first: the as-of merge takes the latest
    filing, so an empty filing would otherwise hide the previous good one. Two filings of
    a firm usable in the same month: the later quarter wins.
    """
    meta_cols = ["gvkey", "cusip", "tic", "conm", "datadate", "availability_date_is",
                 "availability_date_cf", "rdq", "filedate", "curcdq", "fyearq", "fqtr", "fyr"]
    meta_cols = [c for c in meta_cols if c in comp.columns]
    feature_cols = [c for c in feature_cols if c in comp.columns]
    flag_cols = [c for c in flag_cols if c in comp.columns]

    right = comp[meta_cols + feature_cols + flag_cols].dropna(subset=["gvkey", "datadate", availability_col]).copy()
    empty = right[feature_cols].isna().all(axis=1)
    log_step("merge_prepare", f"empty filings dropped ({availability_col})", int(empty.sum()))
    right = right.loc[~empty].copy()
    right["merge_date"] = pd.to_datetime(right[availability_col]) + pd.offsets.MonthEnd(0)
    right["_n_nn"] = right[feature_cols].notna().sum(axis=1)
    right = (
        right.sort_values(["gvkey", "merge_date", "datadate", "_n_nn"])
        .drop_duplicates(["gvkey", "merge_date"], keep="last")
        .drop(columns=["_n_nn"])
    )
    # prefix every Compustat column but the key
    return right.rename(columns={c: ("comp_merge_date" if c == "merge_date" else f"comp_{c}")
                                 for c in right.columns if c != "gvkey"})


def merge_crsp_compustat_asof(crsp, comp_right, setup_name):
    """Latest usable filing for each stock-month (stock-months without a link are kept,
    with empty accounting columns)."""
    has_key = crsp["gvkey"].notna() & crsp["month"].notna()
    unlinked = crsp.loc[~has_key].copy()
    left = crsp.loc[has_key].copy()
    right = comp_right.dropna(subset=["gvkey", "comp_merge_date"]).copy()
    log_step(f"merge_{setup_name}", "stock-months without a gvkey", int((~has_key).sum()))

    # merge_asof needs exactly the same dtypes on both sides
    left["month"] = pd.to_datetime(left["month"]).astype("datetime64[ns]")
    right["comp_merge_date"] = pd.to_datetime(right["comp_merge_date"]).astype("datetime64[ns]")
    for col in [f"comp_{c}" for c in DATE_COLS]:
        if col in right.columns:
            right[col] = pd.to_datetime(right[col]).astype("datetime64[ns]")
    left["gvkey"] = left["gvkey"].astype("string").str.strip().str.upper()
    right["gvkey"] = right["gvkey"].astype("string").str.strip().str.upper()
    left = left.sort_values(["month", "gvkey", "permno"]).reset_index(drop=True)
    right = right.sort_values(["comp_merge_date", "gvkey"]).reset_index(drop=True)

    merged = pd.merge_asof(left, right, left_on="month", right_on="comp_merge_date", by="gvkey",
                           direction="backward", allow_exact_matches=True)
    if len(unlinked):
        unlinked["month"] = pd.to_datetime(unlinked["month"]).astype("datetime64[ns]")
        merged = pd.concat([merged, unlinked], ignore_index=True)
    merged["accounting_age_months"] = (
        (merged["month"].dt.year - merged["comp_datadate"].dt.year) * 12
        + (merged["month"].dt.month - merged["comp_datadate"].dt.month)
    )
    return merged


def apply_staleness_filter(panel, min_age, max_age, setup_name):
    """Accounting data more than max_age months old (or younger than min_age) -> NaN; the
    CRSP row is kept."""
    matched = panel["comp_datadate"].notna()
    valid_age = panel["accounting_age_months"].between(min_age, max_age)
    stale = matched & ~valid_age
    comp_cols = [c for c in panel.columns if c.startswith("comp_")] + ["accounting_age_months"]
    panel.loc[stale, comp_cols] = np.nan
    matched_after = panel["comp_datadate"].notna()
    log_step(f"staleness_{setup_name}", "matched after staleness filter", int(matched_after.sum()))
    log_step(f"staleness_{setup_name}", "match rate after filter", float(matched_after.mean()))
    return panel


def merge_panels(crsp, comp, comp_info, raw_dir):
    """Income-statement features as of their release date, cash-flow and balance-sheet
    features as of their filing date: one as-of merge and staleness filter per group.
    Then the ratios to market value, with the market cap of the month of use."""
    features, flags = comp_info["features"], comp_info["flags"]
    log_step("merge_features", "Compustat features to merge", len(features))
    crsp = attach_gvkey(crsp, raw_dir)

    late = set(FILING_DATE_FEATURES)
    is_feats = [c for c in features if c not in late]
    cf_feats = [c for c in features if c in late]
    is_flags = [c for c in flags if c not in late]
    cf_flags = [c for c in flags if c in late]
    log_step("merge_filing", "release-date / filing-date features", f"{len(is_feats)} / {len(cf_feats)}")

    right_is = prepare_compustat_for_asof(comp, "availability_date_is", is_feats, is_flags)
    panel = merge_crsp_compustat_asof(crsp, right_is, "filing")
    del right_is
    panel = apply_staleness_filter(panel, *ACCOUNTING_AGE, "filing_income")

    slim = crsp[["permno", "month", "gvkey"]].copy()
    right_cf = prepare_compustat_for_asof(comp, "availability_date_cf", cf_feats, cf_flags)
    cf = merge_crsp_compustat_asof(slim, right_cf, "filing_cf")
    del slim, right_cf
    cf = apply_staleness_filter(cf, *ACCOUNTING_AGE, "filing_cashflow")
    cf_cols = [f"comp_{c}" for c in cf_feats + cf_flags if f"comp_{c}" in cf.columns]
    # Both merges start from the same CRSP rows sorted the same way, so the rows line up:
    # the columns are attached in place (a merge would copy the whole panel).
    same_rows = (len(cf) == len(panel)
                 and np.array_equal(cf["permno"].to_numpy(dtype="int64"), panel["permno"].to_numpy(dtype="int64"))
                 and np.array_equal(cf["month"].to_numpy(), panel["month"].to_numpy()))
    if same_rows:
        for c in cf_cols:
            panel[c] = cf[c].to_numpy()
        panel["comp_cf_datadate"] = cf["comp_datadate"].to_numpy()
        panel["comp_cf_availability_date"] = cf["comp_availability_date_cf"].to_numpy()
    else:
        cf = cf[["permno", "month"] + cf_cols + ["comp_datadate", "comp_availability_date_cf"]].rename(
            columns={"comp_datadate": "comp_cf_datadate", "comp_availability_date_cf": "comp_cf_availability_date"})
        n = len(panel)
        panel = panel.merge(cf, on=["permno", "month"], how="left")
        assert len(panel) == n, "duplicate permno-months in the cash-flow merge"
    del cf
    gc.collect()
    has_both = panel["comp_cf_datadate"].notna() & panel["comp_datadate"].notna()
    older = has_both & (panel["comp_cf_datadate"] != panel["comp_datadate"])
    log_step("merge_filing", "stock-months whose cash-flow features come from an older filing "
             "than their income features", float(older.sum() / max(has_both.sum(), 1)))
    return downcast_panel(ft.add_market_ratio_features(panel, log_step, PANEL_NAME))


# -------- 4. labels, split, input selection, transformations --------


def add_cross_sectional_labels(panel):
    """label20: 0 = bottom 20% of next month's return that month, 2 = top 20%, 1 = rest.
    label10: the same with 10%."""
    def label_one_month(x, bottom_q, top_q):
        lo, hi = x.quantile(bottom_q), x.quantile(top_q)
        out = pd.Series(1, index=x.index, dtype="Int64")
        out[x <= lo] = 0
        out[x >= hi] = 2
        out[x.isna()] = pd.NA
        return out

    for name, q in (("label20", 0.20), ("label10", 0.10)):
        panel[name] = (panel.groupby("month")["ret_fwd1m"]
                       .transform(lambda x: label_one_month(x, q, 1 - q)).astype("Int64"))

    # The same target inside the 5 volatility groups the vol-neutral books rank in
    # (12-month volatility known at t; stocks without it go into the most volatile group,
    # as in evaluation/portfolio.vol_bucket).
    vol = pd.to_numeric(panel["vol_12m"], errors="coerce")
    pct = vol.groupby(panel["month"]).rank(pct=True, method="first")
    panel["vol_group"] = (np.ceil(pct * VOL_GROUPS) - 1).fillna(VOL_GROUPS - 1).clip(0, VOL_GROUPS - 1).astype("int8")
    grp = [panel["month"], panel["vol_group"]]
    panel["label20_vg"] = (panel.groupby(grp)["ret_fwd1m"]
                           .transform(lambda x: label_one_month(x, 0.20, 0.80)).astype("Int64"))
    # continuous targets for the tree model: percentile rank - 0.5, inside the group / month
    panel["rank_vg"] = (panel.groupby(grp)["ret_fwd1m"].rank(pct=True) - 0.5).astype("float32")
    panel["rank_all"] = (panel.groupby("month")["ret_fwd1m"].rank(pct=True) - 0.5).astype("float32")
    return panel


def make_labels_and_split(panel, processed_dir):
    panel["month"] = pd.to_datetime(panel["month"])
    panel["target_month"] = pd.to_datetime(panel["target_month"])
    panel = panel[panel["month"] >= pd.Timestamp(MODEL_START)].copy()
    log_step("model_sample", "panel shape from " + MODEL_START, panel.shape)
    panel = add_cross_sectional_labels(panel)

    split_dates = {f"{name}_{edge}": d for name, (start, end) in SPLITS.items()
                   for edge, d in (("start", start), ("end", end))}
    last_month = panel["month"].max()
    if last_month >= pd.Timestamp(HOLDOUT_START):
        split_dates["holdout_start"] = HOLDOUT_START
        split_dates["holdout_end"] = str(last_month.date())
    panel["split"] = pd.NA
    bounds = dict(SPLITS)
    if "holdout_start" in split_dates:
        bounds["holdout"] = (split_dates["holdout_start"], split_dates["holdout_end"])
    for name, (start, end) in bounds.items():
        panel.loc[panel["month"].between(pd.Timestamp(start), pd.Timestamp(end)), "split"] = name
    panel = panel.dropna(subset=["split"]).copy()
    for name in ("train", "val", "test", "recent", "holdout"):
        split_dates[f"n_{name}_months"] = int(panel.loc[panel["split"] == name, "month"].nunique())
    with open(processed_dir / "split_dates.json", "w") as f:
        json.dump(split_dates, f, indent=4)
    log_step("labels_split", "labelled panel shape", panel.shape)
    return panel


def select_features_using_train(panel, candidate_features, group_name, min_std=1e-12):
    """Keep an input unless it is more than MAX_MISSING_RATE missing over SELECTION_WINDOW
    (training months) or constant in training."""
    train_mask = panel["split"] == "train"
    train = panel.loc[train_mask, candidate_features]
    win_mask = train_mask & panel["month"].between(pd.Timestamp(SELECTION_WINDOW[0]),
                                                   pd.Timestamp(SELECTION_WINDOW[1]))
    window_missing = panel.loc[win_mask, candidate_features].isna().mean()
    table = pd.DataFrame({
        "feature": candidate_features,
        "group": group_name,
        "train_missing_rate": train.isna().mean().reindex(candidate_features).values,
        "selection_window_missing_rate": window_missing.reindex(candidate_features).values,
        "train_std": train.std(skipna=True).reindex(candidate_features).values,
    })
    table["drop_missing"] = table["selection_window_missing_rate"] > MAX_MISSING_RATE
    table["drop_low_variance"] = table["train_std"].fillna(0) <= min_std
    table["keep"] = ~table["drop_missing"] & ~table["drop_low_variance"]
    kept = table.loc[table["keep"], "feature"].tolist()
    dropped = table.loc[~table["keep"], "feature"].tolist()
    log_step(f"feature_selection_{group_name}", "candidate / kept / dropped",
             f"{len(candidate_features)} / {len(kept)} / {len(dropped)}")
    return kept, dropped, table


def select_features_and_add_flags(panel, processed_dir):
    """Choose the inputs on training data, add the missing flags (0/1 inputs)."""
    flag_features = [c for c in panel.columns if c.startswith("comp_flag_")]
    compustat_candidates = [c for c in panel.columns if c.startswith("comp_")
                            and c not in COMP_METADATA_COLS and c not in flag_features]
    market_candidates = [c for c in ft.MARKET_FEATURES if c in panel.columns]

    kept_ret, dropped_ret, t1 = select_features_using_train(panel, RETURN_FEATURES, "returns")
    kept_mkt, dropped_mkt, t2 = select_features_using_train(panel, market_candidates, "market")
    kept_comp, dropped_comp, t3 = select_features_using_train(panel, compustat_candidates, "compustat")
    pd.concat([t1, t2, t3], ignore_index=True).to_csv(processed_dir / "feature_selection_table.csv", index=False)
    if dropped_mkt or dropped_comp:
        log_step("feature_selection", "dropped", ", ".join(dropped_mkt + dropped_comp))
    numeric_features = kept_ret + kept_mkt + kept_comp
    log_step("feature_selection", "numeric inputs", len(numeric_features))

    # Missing flags for every input with at least 5% missing in training.
    train_missing = panel.loc[panel["split"] == "train", numeric_features].isna().mean()
    with_flags = train_missing[train_missing >= 0.05].index.tolist()
    missing_indicator_features = [f"miss_{c}" for c in with_flags]
    for col in with_flags:
        panel[f"miss_{col}"] = panel[col].isna().astype("int8")
    for col in flag_features:   # structural-zero flags: NaN (no fresh filing) -> 0
        panel[col] = panel[col].fillna(0).astype("int8")
    log_step("missing_indicators", "inputs with a missing flag", len(missing_indicator_features))

    feature_groups = {
        "return_features": kept_ret,
        "market_features": kept_mkt,
        "compustat_features": kept_comp,
        "numeric_features": numeric_features,
        "missing_indicator_features": missing_indicator_features,
        "flag_features": flag_features,
        "binary_features": missing_indicator_features + flag_features,
        "categorical_features": ["industry_id"],
        "analyst_features": [c for c in kept_mkt if c in ft.ANALYST_FEATURES],
        "dropped_features": dropped_ret + dropped_mkt + dropped_comp,
    }
    return panel, feature_groups


def transform_numeric_features(panel, features, lower_q=0.01, upper_q=0.99, min_scale=1e-4, final_clip=10.0):
    """Each input, month by month: winsorise at 1% / 99%, fill missing values with the
    month's median (then the training median, then 0), scale robustly ((x - median) / IQR,
    z-score if the IQR is too small), clip at +-10. Transforms the panel in place."""
    train_medians = panel.loc[panel["split"].eq("train"), features].median(skipna=True)
    for i, col in enumerate(features, start=1):
        print(f"[transform] {i}/{len(features)}: {col}")
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

        q_low = panel.groupby("month")[col].quantile(lower_q)
        q_high = panel.groupby("month")[col].quantile(upper_q)
        panel[col] = panel[col].clip(lower=panel["month"].map(q_low), upper=panel["month"].map(q_high))

        m_med = panel.groupby("month")[col].median()
        panel[col] = panel[col].fillna(panel["month"].map(m_med))
        panel[col] = panel[col].fillna(train_medians[col]).fillna(0.0)

        g = panel.groupby("month")[col]
        med = panel["month"].map(g.median())
        iqr = panel["month"].map(g.quantile(0.75) - g.quantile(0.25))
        mean = panel["month"].map(g.mean())
        std = panel["month"].map(g.std())
        robust = (panel[col] - med) / iqr
        zscore = (panel[col] - mean) / std
        scaled = robust.where(iqr.abs() >= min_scale, zscore.where(std.abs() >= min_scale, 0.0))
        scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        panel[col] = scaled.clip(-final_clip, final_clip).astype("float32")

    log_step("transform", "inputs transformed", len(features))
    log_step("transform", "NaNs left in the inputs", int(panel[features].isna().sum().sum()))
    log_step("transform", "max absolute transformed value", float(panel[features].abs().max().max()))
    return panel


def transform_and_save(panel, feature_groups, processed_dir, regime=None):
    # unscaled copies for portfolio construction (never model inputs)
    raw_cols = []
    for col in RAW_COPY_FEATURES:
        if col in panel.columns:
            panel[f"raw_{col}"] = pd.to_numeric(panel[col], errors="coerce").astype("float32")
            raw_cols.append(f"raw_{col}")

    numeric_features = feature_groups["numeric_features"]
    panel = transform_numeric_features(panel, numeric_features)

    # industry ids learned on the training set; 0 = an industry unseen in training
    train_industries = (panel.loc[panel["split"] == "train", "industry_code"].dropna()
                        .astype(str).sort_values().unique())
    industry_mapping = {"__UNKNOWN__": 0}
    for idx, code in enumerate(train_industries, start=1):
        industry_mapping[code] = idx
    panel["industry_id"] = (panel["industry_code"].astype("string").map(industry_mapping)
                            .fillna(0).astype("int16"))
    log_step("industry_encoding", "number of train industries", len(train_industries))
    with open(processed_dir / "industry_mapping.json", "w") as f:
        json.dump(industry_mapping, f, indent=4)

    # market-regime inputs: one value per month, already scaled over time (not across
    # stocks: the same value for every stock); months before their history exists -> 0
    regime_cols = []
    if regime is not None and len(regime.columns):
        regime_cols = list(regime.columns)
        for c in regime_cols:
            panel[c] = panel["month"].map(regime[c]).fillna(0.0).astype("float32")
    log_step("regime", "regime inputs", len(regime_cols))

    model_inputs = numeric_features + feature_groups["binary_features"] + regime_cols
    analyst = set(feature_groups.get("analyst_features", []))
    v4_inputs = [c for c in numeric_features + feature_groups["binary_features"]
                 if c not in analyst and not (c.startswith("miss_") and c[5:] in analyst)]
    feature_groups = {
        **feature_groups,
        "raw_risk_columns": raw_cols,
        "target_columns": TARGET_COLUMNS,
        "id_columns": [c for c in ID_COLUMNS if c in panel.columns],
        "model_features_numeric_binary": model_inputs,
        "model_features_mlp": model_inputs + ["industry_id"],
        "regime_features": regime_cols,
        "v4_inputs": v4_inputs,
        "settings": {
            "panel": PANEL_NAME,
            "splits": SPLITS,
            "holdout_start": HOLDOUT_START,
            "filing_fallback_days": FILING_FALLBACK_DAYS,
            "accounting_age_months": list(ACCOUNTING_AGE),
            "cash_flow_features": CASH_FLOW_FEATURES,
            "filing_date_features": FILING_DATE_FEATURES,
            "structural_zero_items": STRUCTURAL_ZERO_ITEMS,
            "selection_window": list(SELECTION_WINDOW),
            "max_missing_rate": MAX_MISSING_RATE,
            "short_interest_publication_days": ft.SI_PUBLICATION_DAYS,
            "inst_13f_filing_days": ft.INST_FILING_DAYS,
            "nasdaq_volume_divisor": ft.NASDAQ_VOLUME_DIVISOR,
            "ew_half_life_months": ft.EW_HALF_LIFE_MONTHS,
            "ibes_max_age_days": ft.IBES_MAX_AGE_DAYS,
            "ibes_sue_max_age_months": ft.IBES_SUE_MAX_AGE_MONTHS,
            "ibes_link_max_score": ft.IBES_LINK_MAX_SCORE,
            "regime_min_months": ft.REGIME_MIN_MONTHS,
            "vol_groups": VOL_GROUPS,
        },
        "transformations": {
            "compustat_ytd_to_quarterly": "YTD items turned into quarterly flows before any ratio",
            "winsorization": "cross-sectional by month, 1% and 99%",
            "imputation": "monthly median, then training median, then 0",
            "standardization": "cross-sectional robust scaling by month (median / IQR); "
                               "regime inputs: expanding z-score over time",
            "final_clip": "inputs clipped to [-10, 10]",
            "industry_encoding": "mapping learned on train only; unknown = 0",
        },
    }
    with open(processed_dir / "feature_groups.json", "w") as f:
        json.dump(feature_groups, f, indent=4)

    assert panel[model_inputs].isna().sum().sum() == 0
    assert panel["industry_id"].isna().sum() == 0
    assert not np.isinf(panel[model_inputs].select_dtypes(include=[np.number])).any().any()
    panel.to_parquet(processed_dir / f"panel_model_{PANEL_NAME}.parquet", index=False)
    log_step("save", "model panel shape", panel.shape)
    return panel, feature_groups


# -------- main --------


def main():
    root = find_project_root()
    raw_dir = root / "data" / "raw_wrds"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    print("project root:", root)

    # 1. CRSP
    crsp = build_crsp_return_features(load_crsp(raw_dir))
    crsp = ft.add_market_features(crsp, raw_dir, log_step)   # size, liquidity, daily, SI, 13F ...
    crsp = downcast_panel(crsp)
    crsp = build_crsp_target(crsp)
    crsp = apply_universe_filter(crsp)
    crsp = ft.add_industry_momentum(crsp, log_step)
    regime = ft.regime_table(raw_dir, crsp[["month", "ret"]], log_step)
    gc.collect()

    # 2. Compustat
    comp, comp_info = build_compustat_features(load_and_clean_compustat(raw_dir))

    # 3-4. merge, labels, split, inputs, transformations
    panel = merge_panels(crsp, comp, comp_info, raw_dir)
    del crsp, comp
    gc.collect()
    panel = make_labels_and_split(panel, processed_dir)
    panel, feature_groups = select_features_and_add_flags(panel, processed_dir)
    transform_and_save(panel, feature_groups, processed_dir, regime)

    pd.DataFrame(preprocessing_log).to_csv(processed_dir / "preprocessing_report.csv", index=False)
    print("\ndone.")


if __name__ == "__main__":
    main()
