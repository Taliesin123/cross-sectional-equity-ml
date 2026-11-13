"""
Feature definitions used by build_processed_data.py, beyond the monthly return features
and the income-statement / cash-flow ratios that build_processed_data.py builds itself.

Market side (one value per stock-month, known at the month-end close):
    size / liquidity / issuance from the monthly CRSP file, daily-return features from
    the CRSP daily file (summarised per stock-month on the WRDS server), short interest,
    the stock's return around its latest earnings release, institutional ownership (13F),
    analyst forecasts (IBES).
Market regime (one value per month, the same for every stock; scaled over time, not
    across stocks): market return, market and momentum volatility, cross-sectional
    dispersion, credit and term spreads (regime_table).
Accounting side (called from build_compustat_features):
    balance-sheet value / quality / investment features, the rest of the cash-flow
    statement, earnings surprises; ratios to market value are formed after the merge
    with the market cap of the month the features are used (add_market_ratio_features).

Timing rules (no feature is used before it was public):
    short interest     settlement date + 14 days (published about 8 business days after
                       the settlement date)
    announcement ret.  the third trading day of the window (day +1) must be on or before
                       the month-end
    13F holdings       quarter end + 45 days (the 13F filing deadline)
    balance sheet      the 10-Q / 10-K filing date (availability_date_cf, like the
                       cash-flow features; the balance sheet is not always in the release)
    earnings surprise  the earnings release (availability_date_is)
    IBES consensus     the IBES statistical period date (statpers, mid-month) must be on or
                       before the month-end, and at most 50 days old
    IBES surprise      the announcement date + 1 day must be on or before the month-end
    market regime      month t's own data (index returns, yields averaged over the month)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings

SI_PUBLICATION_DAYS = 14        # short interest usable from settlement date + 14 days
SI_MAX_AGE_MONTHS = 2           # older short interest is treated as missing
EAR_MAX_AGE_MONTHS = 4          # the latest release only (one per quarter)
INST_FILING_DAYS = 45           # 13F deadline after quarter end
INST_MAX_AGE_MONTHS = 6
EW_HALF_LIFE_MONTHS = 1.0       # exponentially weighted daily variance
MIN_DAYS_1M, MIN_DAYS_3M, MIN_DAYS_12M = 10, 30, 120

# NASDAQ volume was double counted (dealer market) until the 2000s; Gao & Ritter (2010)
# divisors put it on the NYSE / AMEX basis before turnover / dollar volume / Amihud.
NASDAQ_VOLUME_DIVISOR = [("2001-01-31", 2.0), ("2001-12-31", 1.8), ("2003-12-31", 1.6)]

CRSP_MONTHLY_FEATURES = [
    "log_mktcap", "log_price", "turnover_1m", "turnover_12m", "log_dollar_volume_3m",
    "share_issuance_12m", "dividend_yield_12m", "price_to_12m_high", "ret_36_13",
    "seasonal_1y", "seasonal_2_5y", "industry_mom_6m",
]
DAILY_FEATURES = [
    "vol_d_1m", "vol_d_3m", "vol_d_ew", "vol_d_last5", "vol_ratio_1m_12m", "ivol_d_1m",
    "ivol_d_3m", "beta_d_12m", "max_ret_1m", "min_ret_1m", "skew_d_3m",
    "zero_ret_share_3m", "log_amihud_3m", "bid_ask_spread",
]
SHORT_INTEREST_FEATURES = ["short_interest_ratio", "short_interest_change_3m", "days_to_cover"]
ANNOUNCEMENT_FEATURES = ["earnings_ann_return"]
INSTITUTIONAL_FEATURES = ["inst_ownership", "inst_ownership_change", "inst_breadth_change"]
ANALYST_FEATURES = [
    "an_coverage", "an_dispersion", "an_fep", "an_fy2_growth", "an_ltg", "an_updown",
    "an_rev_1m", "an_rev_3m", "an_sue", "an_rec_mean", "an_rec_change_3m",
]
MARKET_FEATURES = (CRSP_MONTHLY_FEATURES + DAILY_FEATURES + SHORT_INTEREST_FEATURES
                   + ANNOUNCEMENT_FEATURES + INSTITUTIONAL_FEATURES + ANALYST_FEATURES)
# metadata kept in the panel so check_processed_data.py can verify the timing
MARKET_DATE_COLUMNS = ["si_settlement_date", "ear_end_date", "inst_report_date", "an_statpers",
                       "an_sue_ann_date"]

IBES_MAX_AGE_DAYS = 50          # a consensus older than this (no update this month) is stale
IBES_SUE_MAX_AGE_MONTHS = 4     # the latest quarter's surprise only
IBES_LINK_MAX_SCORE = 2         # wrdsapps.ibcrsphist match quality (lower = better)

# Market-regime inputs: one value per month, z-scored over time with the history known at
# t (expanding mean / std, at least REGIME_MIN_MONTHS), clipped at +-4.
REGIME_FEATURES = [
    "reg_mkt_ret_1m", "reg_mkt_ret_12m", "reg_mkt_ret_24m", "reg_mkt_vol_1m", "reg_mkt_vol_ratio",
    "reg_umd_ret_12m", "reg_umd_vol_6m", "reg_xs_dispersion", "reg_credit_spread", "reg_term_spread",
]
REGIME_MIN_MONTHS = 60

# Compustat items from comp_fundq_extra.parquet that the features use (the download
# keeps more items).
FUNDQ_EXTRA_USED = ["spiq", "intanq", "mibtq", "mibq", "sstky", "prstkcy", "dltisy", "dltry", "aqcy"]
EXTRA_YTD_ITEMS = ["sstky", "prstkcy", "dltisy", "dltry", "aqcy"]
FUNDQ_KEYS = ["gvkey", "datadate", "fyearq", "fqtr", "fyr"]

# Accounting features timed with the filing (balance sheet involved) ...
BALANCE_SHEET_FEATURES = [
    "roa_qtr", "roe_qtr", "gross_profit_to_assets", "operating_profitability",
    "asset_growth_yoy", "investment_to_assets_yoy", "book_leverage", "liabilities_to_assets",
    "current_ratio", "quick_ratio", "cash_to_assets", "ppe_to_assets", "accruals_to_assets",
    "net_operating_assets", "asset_turnover", "roa_change_yoy", "roa_vol_8q",
    "intangibles_to_assets", "special_items_to_assets", "net_debt_issuance_to_assets",
    "net_equity_issuance_to_assets", "acquisitions_to_assets", "capex_to_assets",
]
# ... and with the earnings release (income statement only).
EARNINGS_FEATURES = ["sue_earnings", "sue_sales"]
# Raw amounts carried through the merge to form ratios to market value (h_ = helper,
# never a model input). Income-statement amounts follow the release, the rest the filing.
HELPERS_IS = ["h_ib_ttm", "h_sale_ttm", "h_xrd_ttm", "h_oibdp_ttm"]
HELPERS_FILING = ["h_be", "h_debt", "h_cash", "h_pstk", "h_mib", "h_oancf_ttm", "h_payout_ttm",
                  "h_net_payout_ttm"]
MARKET_RATIO_FEATURES = [
    "book_to_market", "earnings_to_price", "sales_to_price", "cashflow_to_price",
    "rd_to_market", "payout_yield", "net_payout_yield", "ebitda_to_ev", "market_leverage",
]
FILING_TIMED = BALANCE_SHEET_FEATURES + HELPERS_FILING


# ---------------------------------------------------------------- helpers


def month_num(month):
    month = pd.to_datetime(month)
    return (month.dt.year * 12 + month.dt.month).astype("int64")


def stock_key(permno, mnum):
    return pd.to_numeric(permno, errors="coerce").astype("int64").to_numpy() * 100_000 + np.asarray(mnum)


def attach_by_key(df, df_key, src, src_key, cols):
    """df[cols] = src[cols] matched on an int64 key (src unique on its key), without
    copying df the way a merge would."""
    idx = pd.Index(src_key)
    if not idx.is_unique:
        raise ValueError("attach_by_key: source keys are not unique")
    pos = idx.get_indexer(df_key)
    hit = pos >= 0
    for c in cols:
        vals = src[c].to_numpy()
        if np.issubdtype(vals.dtype, np.datetime64):
            out = np.full(len(df), np.datetime64("NaT"), dtype=vals.dtype)
        else:
            vals = vals.astype("float64")
            out = np.full(len(df), np.nan)
        out[hit] = vals[pos[hit]]
        df[c] = out.astype("float32") if out.dtype == "float64" else out
    return hit


def latest_event(df, df_permno, df_mnum, events, value_cols, max_age):
    """For each row of df: the event (permno, usable month number) with the largest usable
    month <= the row's month, at most max_age months old. events: columns permno, umnum
    and value_cols, one row per (permno, umnum)."""
    ev_key = stock_key(events["permno"], events["umnum"])
    idx = pd.Index(ev_key)
    if not idx.is_unique:
        raise ValueError("latest_event: duplicate (permno, month) events")
    base = pd.to_numeric(df_permno, errors="coerce").astype("int64").to_numpy() * 100_000
    pos = np.full(len(df), -1)
    for age in range(max_age + 1):
        need = pos < 0
        if not need.any():
            break
        p = idx.get_indexer(base[need] + (np.asarray(df_mnum)[need] - age))
        pos[np.flatnonzero(need)[p >= 0]] = p[p >= 0]
    hit = pos >= 0
    for c in value_cols:
        vals = events[c].to_numpy()
        if np.issubdtype(vals.dtype, np.datetime64):
            out = np.full(len(df), np.datetime64("NaT"), dtype=vals.dtype)
            out[hit] = vals[pos[hit]]
            df[c] = out
        else:
            out = np.full(len(df), np.nan)
            out[hit] = vals.astype("float64")[pos[hit]]
            df[c] = out.astype("float32")
    return hit


def consecutive(mnum, permno, k):
    """True where the row k rows back is the same stock exactly k months earlier (rows
    sorted by stock, then month)."""
    m = pd.Series(np.asarray(mnum))
    p = pd.Series(np.asarray(permno))
    return ((p.shift(k) == p) & (m - m.shift(k) == k)).to_numpy()


def rolling_sum(values, window):
    return pd.Series(np.asarray(values, dtype="float64")).rolling(window, min_periods=window).sum().to_numpy()


def nasdaq_divisor(month, exch):
    month = pd.to_datetime(month)
    div = np.ones(len(month))
    is_q = (pd.Series(exch).astype("string").str.strip().str.upper() == "Q").fillna(False).to_numpy()
    prev = pd.Timestamp("1900-01-01")
    for end, d in NASDAQ_VOLUME_DIVISOR:
        end = pd.Timestamp(end)
        m = is_q & (month > prev).to_numpy() & (month <= end).to_numpy()
        div[m] = d
        prev = end
    return div


def log_positive(x):
    x = np.asarray(x, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def ratio(num, den, positive_den=True):
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    ok = den > 0 if positive_den else den != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
    out[~np.isfinite(out)] = np.nan
    return out


# ---------------------------------------------------------------- CRSP monthly


def add_crsp_monthly_features(crsp, log):
    """Size, liquidity, issuance, payout and long-horizon return features from the
    monthly file. crsp must be sorted by permno, month (all securities, full history)."""
    mn = month_num(crsp["month"]).to_numpy()
    pno = crsp["permno"].to_numpy()
    cap = np.where(crsp["mktcap"] > 0, crsp["mktcap"], np.nan)
    prc = np.abs(crsp["prc"].to_numpy(dtype="float64"))
    prc = np.where(prc > 0, prc, np.nan)
    shares = np.where(crsp["shrout"] > 0, crsp["shrout"] * 1000.0, np.nan)
    vol = crsp["vol"].to_numpy(dtype="float64")
    vol_adj = np.where(vol >= 0, vol, np.nan) / nasdaq_divisor(crsp["month"], crsp["primaryexch"])
    ret = crsp["ret"].to_numpy(dtype="float64")
    retx = crsp["retx"].to_numpy(dtype="float64") if "retx" in crsp else np.full(len(crsp), np.nan)

    out = {}
    out["log_mktcap"] = log_positive(cap)
    out["log_price"] = log_positive(prc)
    turn = ratio(vol_adj, shares)
    out["turnover_1m"] = turn
    out["turnover_12m"] = np.where(consecutive(mn, pno, 11), rolling_sum(turn, 12) / 12, np.nan)
    dvol = vol_adj * prc
    out["log_dollar_volume_3m"] = log_positive(np.where(consecutive(mn, pno, 2), rolling_sum(dvol, 3) / 3, np.nan))
    lrx = np.log1p(np.where(retx > -1, retx, np.nan))
    sum12x = np.where(consecutive(mn, pno, 11), rolling_sum(lrx, 12), np.nan)
    # growth of split-adjusted shares: market-cap growth minus price-only return
    cap_lag12 = pd.Series(cap).shift(12).to_numpy()
    out["share_issuance_12m"] = np.where(consecutive(mn, pno, 12),
                                         log_positive(cap) - log_positive(cap_lag12) - sum12x, np.nan)
    out["dividend_yield_12m"] = np.where(consecutive(mn, pno, 11), rolling_sum(ret - retx, 12), np.nan)
    # split-adjusted price relative to its 12-month high (monthly closes), in logs (<= 0)
    c = pd.Series(np.nan_to_num(lrx, nan=0.0)).groupby(pno).cumsum().to_numpy()
    cmax = pd.Series(c).rolling(12, min_periods=12).max().to_numpy()
    ok_px = consecutive(mn, pno, 11) & np.isfinite(lrx)
    out["price_to_12m_high"] = np.where(ok_px, c - cmax, np.nan)
    # long-term reversal: months t-36 .. t-13
    lr = np.log1p(np.where(ret > -1, ret, np.nan))
    lr13 = pd.Series(lr).shift(13).to_numpy()
    ok_lt = consecutive(mn, pno, 13) & consecutive(mn, pno, 36)
    out["ret_36_13"] = np.where(ok_lt, np.expm1(rolling_sum(lr13, 24)), np.nan)
    # seasonality (Heston-Sadka): the target month's calendar month in earlier years.
    # The target is t+1, so one year back is lag 11, two to five years back lags 23-59.
    out["seasonal_1y"] = np.where(consecutive(mn, pno, 11), pd.Series(ret).shift(11).to_numpy(), np.nan)
    acc = np.zeros(len(crsp))
    cnt = np.zeros(len(crsp))
    for lag in (23, 35, 47, 59):
        v = pd.Series(ret).shift(lag).to_numpy()
        # lag-th row back is the same stock exactly lag months earlier (gaps allowed between)
        okl = (pd.Series(pno).shift(lag).eq(pd.Series(pno)) & (pd.Series(mn) - pd.Series(mn).shift(lag) == lag)).to_numpy()
        okl = okl & np.isfinite(v)
        acc += np.where(okl, v, 0.0)
        cnt += okl
    out["seasonal_2_5y"] = np.where(cnt >= 2, acc / np.maximum(cnt, 1), np.nan)
    # 6-month return t-5..t, for the industry momentum built after the universe filter
    out["_ret_6m"] = np.where(consecutive(mn, pno, 5), np.expm1(rolling_sum(lr, 6)), np.nan)
    out["_cap_lag1"] = np.where(consecutive(mn, pno, 1), pd.Series(cap).shift(1).to_numpy(), np.nan)
    out["_vol_adj"] = vol_adj
    for k, v in out.items():
        crsp[k] = np.asarray(v, dtype="float64").astype("float32")
    for k in CRSP_MONTHLY_FEATURES:
        if k in crsp:
            log("crsp_monthly_features", f"{k}: share available", float(np.isfinite(crsp[k]).mean()))
    return crsp


def add_industry_momentum(crsp, log):
    """Value-weighted 6-month return of the stock's industry (2-digit SIC), among the
    stocks in the universe that month (Moskowitz-Grinblatt)."""
    if "_ret_6m" not in crsp:
        return crsp
    w = crsp["_cap_lag1"].astype("float64")
    r = crsp["_ret_6m"].astype("float64")
    ok = w.notna() & r.notna() & crsp["industry_code"].notna()
    grp = [crsp["month"], crsp["industry_code"]]
    num = (r * w).where(ok).groupby(grp).transform("sum")
    den = w.where(ok).groupby(grp).transform("sum")
    n = ok.groupby(grp).transform("sum")
    crsp["industry_mom_6m"] = (num / den).where((n >= 3) & (den > 0)).astype("float32")
    log("crsp_monthly_features", "industry_mom_6m: share available", float(crsp["industry_mom_6m"].notna().mean()))
    return crsp.drop(columns=["_ret_6m", "_cap_lag1"])


# ---------------------------------------------------------------- daily


def load_daily_summary(raw_dir):
    path = raw_dir / "crsp_daily_features.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d["permno"] = pd.to_numeric(d["permno"], errors="coerce")
    d = d.dropna(subset=["permno", "month_start"])
    d["permno"] = d["permno"].astype("int64")
    d["month"] = pd.to_datetime(d["month_start"]) + pd.offsets.MonthEnd(0)
    d = d.drop(columns=["month_start"]).drop_duplicates(["permno", "month"])
    return d.sort_values(["permno", "month"]).reset_index(drop=True)


def daily_features_table(d):
    """Daily-return features per (permno, month) from the server-side sums."""
    mn = month_num(d["month"]).to_numpy()
    pno = d["permno"].to_numpy()
    f = lambda c: d[c].to_numpy(dtype="float64") if c in d else np.full(len(d), np.nan)

    def pooled(c, w):
        return np.where(consecutive(mn, pno, w - 1), rolling_sum(f(c), w), np.nan) if w > 1 else f(c)

    def var_from(n, s, ss):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (ss - s * s / n) / (n - 1)
        return np.where(n > 1, np.maximum(v, 0.0), np.nan)

    def reg(w):
        n, sr, srr = pooled("n_rm", w), pooled("s_r_m", w), pooled("s_rr_m", w)
        sm, smm, srm = pooled("s_m", w), pooled("s_mm", w), pooled("s_rm", w)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = (srm - sr * sm / n) / (n - 1)
        vm = var_from(n, sm, smm)
        vr = var_from(n, sr, srr)
        beta = ratio(cov, vm)
        resid = np.maximum(vr - np.where(vm > 0, cov * cov / np.where(vm > 0, vm, 1.0), np.nan), 0.0)
        return n, beta, resid

    out = pd.DataFrame({"permno": d["permno"], "month": d["month"]})
    n1, n3, n12 = f("n_ret"), pooled("n_ret", 3), pooled("n_ret", 12)
    var_1m = var_from(n1, f("s_r"), f("s_rr"))
    var_3m = var_from(n3, pooled("s_r", 3), pooled("s_rr", 3))
    var_12m = var_from(n12, pooled("s_r", 12), pooled("s_rr", 12))
    out["vol_d_1m"] = np.where(n1 >= MIN_DAYS_1M, np.sqrt(var_1m), np.nan)
    out["vol_d_3m"] = np.where(n3 >= MIN_DAYS_3M, np.sqrt(var_3m), np.nan)
    vol12 = np.where(n12 >= MIN_DAYS_12M, np.sqrt(var_12m), np.nan)
    out["vol_ratio_1m_12m"] = ratio(out["vol_d_1m"], vol12)
    # exponentially weighted mean of each month's mean squared daily return (6 months)
    ms = ratio(f("s_rr"), n1)
    lam = 0.5 ** (1.0 / EW_HALF_LIFE_MONTHS)
    num = np.zeros(len(d))
    den = np.zeros(len(d))
    for k in range(6):
        v = pd.Series(ms).shift(k).to_numpy() if k else ms
        ok = (pd.Series(pno).shift(k).eq(pd.Series(pno)) & (pd.Series(mn) - pd.Series(mn).shift(k) == k)).to_numpy() if k else np.ones(len(d), bool)
        ok = ok & np.isfinite(v)
        num += np.where(ok, lam ** k * v, 0.0)
        den += np.where(ok, lam ** k, 0.0)
    out["vol_d_ew"] = np.where((n1 >= MIN_DAYS_1M) & (den > 0), np.sqrt(num / np.maximum(den, 1e-12)), np.nan)
    out["vol_d_last5"] = np.where(f("n_last5") >= 4, np.sqrt(np.maximum(ratio(f("s_rr_last5"), f("n_last5")), 0)), np.nan)
    nr1, _, res1 = reg(1)
    nr3, _, res3 = reg(3)
    out["ivol_d_1m"] = np.where(nr1 >= MIN_DAYS_1M, np.sqrt(res1), np.nan)
    out["ivol_d_3m"] = np.where(nr3 >= MIN_DAYS_3M, np.sqrt(res3), np.nan)
    # Dimson beta (today's + yesterday's market), 12 months of daily returns
    n12r, b0, _ = reg(12)
    nl, srl, sml = pooled("n_rml", 12), pooled("s_r_ml", 12), pooled("s_ml", 12)
    smlml, srml = pooled("s_mlml", 12), pooled("s_rml", 12)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov_l = (srml - srl * sml / nl) / (nl - 1)
    b1 = ratio(cov_l, var_from(nl, sml, smlml))
    out["beta_d_12m"] = np.where((n12r >= MIN_DAYS_12M) & (nl >= MIN_DAYS_12M), b0 + b1, np.nan)
    out["max_ret_1m"] = np.where(n1 >= MIN_DAYS_1M, f("max_r"), np.nan)
    out["min_ret_1m"] = np.where(n1 >= MIN_DAYS_1M, f("min_r"), np.nan)
    s1, s2, s3 = pooled("s_r", 3), pooled("s_rr", 3), pooled("s_rrr", 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = s1 / n3
        m2 = s2 / n3 - mu ** 2
        m3 = s3 / n3 - 3 * mu * s2 / n3 + 2 * mu ** 3
        skew = m3 / np.power(np.where(m2 > 0, m2, np.nan), 1.5)
    out["skew_d_3m"] = np.where(n3 >= MIN_DAYS_3M, skew, np.nan)
    out["zero_ret_share_3m"] = np.where(n3 >= MIN_DAYS_3M, ratio(pooled("n_zero", 3), n3), np.nan)
    # Amihud: mean |r| / dollar volume, per $ million; NASDAQ adjustment applied later
    na3 = pooled("n_amihud", 3)
    out["_amihud_3m"] = np.where(na3 >= MIN_DAYS_1M, ratio(pooled("s_amihud", 3), na3) * 1e6, np.nan)
    qs = np.where(f("n_quote_days") >= 5, f("quoted_spread"), np.nan)
    hl = np.sqrt(np.maximum(4 * f("hl_prod_mean"), 0))
    hl = np.where((f("n_hl_days") >= 5) & (hl > 0), hl, np.nan)
    out["bid_ask_spread"] = np.where(np.isfinite(qs), qs, hl)
    for c in out.columns[2:]:
        out[c] = np.asarray(out[c], dtype="float64")
        out.loc[~np.isfinite(out[c]), c] = np.nan
        out[c] = out[c].astype("float32")
    return out


def add_daily_features(crsp, raw_dir, log):
    d = load_daily_summary(raw_dir)
    if d is None:
        print("No data/raw_wrds/crsp_daily_features.parquet: daily-return features left out "
              "(run src/data/download_wrds_features.py).")
        return crsp, []
    t = daily_features_table(d)
    del d
    cols = [c for c in t.columns if c not in ("permno", "month")]
    key_c = stock_key(crsp["permno"], month_num(crsp["month"]))
    key_t = stock_key(t["permno"], month_num(t["month"]))
    hit = attach_by_key(crsp, key_c, t, key_t, cols)
    log("daily_features", "stock-months with a daily summary", float(hit.mean()))
    am = crsp.pop("_amihud_3m").to_numpy(dtype="float64")
    crsp["log_amihud_3m"] = log_positive(am * nasdaq_divisor(crsp["month"], crsp["primaryexch"])).astype("float32")
    made = [c for c in DAILY_FEATURES if c in crsp]
    for c in made:
        log("daily_features", f"{c}: share available", float(crsp[c].notna().mean()))
    return crsp, made


# ---------------------------------------------------------------- short interest


def add_short_interest_features(crsp, raw_dir, log):
    path = raw_dir / "short_interest_full.parquet"
    if not path.exists():
        path = raw_dir / "short_interest.parquet"
        if not path.exists():
            print("No short-interest file: short-interest features left out.")
            return crsp, []
        print("Using short_interest.parquet (July 2006 onwards only; download_wrds_features.py "
              "fetches 1973 onwards): the feature will be missing in most training years.")
    s = pd.read_parquet(path)
    s["permno"] = pd.to_numeric(s["permno"], errors="coerce")
    s["shortint"] = pd.to_numeric(s["shortint"], errors="coerce")
    s = s.dropna(subset=["permno", "shortint", "datadate"])
    s["permno"] = s["permno"].astype("int64")
    s["datadate"] = pd.to_datetime(s["datadate"])
    s = s.groupby(["permno", "datadate"], as_index=False)["shortint"].max()
    # shares outstanding and volume of the settlement month (same split basis as shortint)
    s_key = stock_key(s["permno"], month_num(s["datadate"] + pd.offsets.MonthEnd(0)))
    c_key = stock_key(crsp["permno"], month_num(crsp["month"]))
    src = pd.DataFrame({"shares": crsp["shrout"].to_numpy(dtype="float64") * 1000.0,
                        "vol_adj": crsp["_vol_adj"].to_numpy(dtype="float64")})
    ok_src = pd.Index(c_key).is_unique
    if not ok_src:
        raise ValueError("duplicate permno-months in CRSP")
    pos = pd.Index(c_key).get_indexer(s_key)
    s["shares"] = np.where(pos >= 0, src["shares"].to_numpy()[np.maximum(pos, 0)], np.nan)
    s["vol_adj"] = np.where(pos >= 0, src["vol_adj"].to_numpy()[np.maximum(pos, 0)], np.nan)
    s["sir"] = ratio(s["shortint"], s["shares"])
    s["dtc"] = ratio(s["shortint"], s["vol_adj"] / 21.0)
    s["usable"] = (s["datadate"] + pd.Timedelta(days=SI_PUBLICATION_DAYS)) + pd.offsets.MonthEnd(0)
    s["umnum"] = month_num(s["usable"])
    s = s.sort_values(["permno", "umnum", "datadate"]).drop_duplicates(["permno", "umnum"], keep="last")
    s = s.rename(columns={"datadate": "si_settlement_date"})
    mn = month_num(crsp["month"]).to_numpy()
    hit = latest_event(crsp, crsp["permno"], mn, s, ["sir", "dtc", "si_settlement_date"], SI_MAX_AGE_MONTHS)
    crsp["short_interest_ratio"] = crsp.pop("sir").clip(upper=1.0)
    crsp["days_to_cover"] = crsp.pop("dtc")
    # change over three months (the value known three months ago, same rule)
    lag3 = crsp["short_interest_ratio"].shift(3).where(consecutive(mn, crsp["permno"].to_numpy(), 3))
    crsp["short_interest_change_3m"] = (crsp["short_interest_ratio"] - lag3).astype("float32")
    log("short_interest", "stock-months with short interest", float(hit.mean()))
    log("short_interest", "first settlement date", str(s["si_settlement_date"].min().date()))
    return crsp, SHORT_INTEREST_FEATURES


# ---------------------------------------------------------------- earnings announcement return


def announcement_returns(a, ffd):
    """Market-adjusted return on trading days -1, 0, +1 around each earnings release
    (day 0 = first trading day on or after rdq)."""
    a = a.dropna(subset=["permno", "rdq", "dlycaldt"]).copy()
    a["permno"] = pd.to_numeric(a["permno"], errors="coerce").astype("int64")
    a["rdq"] = pd.to_datetime(a["rdq"]).dt.normalize()
    a["dlycaldt"] = pd.to_datetime(a["dlycaldt"]).dt.normalize()
    a = a.drop_duplicates(["permno", "rdq", "dlycaldt"]).sort_values(["permno", "rdq", "dlycaldt"])
    if ffd is not None:
        m = ffd.set_index("date")["mkt"]
        a["mkt"] = a["dlycaldt"].map(m).fillna(0.0)
    else:
        a["mkt"] = 0.0
    g = a.groupby(["permno", "rdq"], sort=False)
    k = g.cumcount()
    after = a["dlycaldt"] >= a["rdq"]
    k0 = k.where(after).groupby([a["permno"], a["rdq"]]).transform("min")
    off = k - k0
    w = a[off.between(-1, 1)].copy()
    w["off"] = off[off.between(-1, 1)].astype(int)
    w["lr"] = np.log1p(pd.to_numeric(w["dlyret"], errors="coerce").clip(lower=-0.99))
    w["lm"] = np.log1p(w["mkt"])
    agg = w.groupby(["permno", "rdq"]).agg(
        n=("off", "size"), n_ret=("lr", "count"), lr=("lr", "sum"), lm=("lm", "sum"),
        first=("dlycaldt", "min"), last=("dlycaldt", "max"))
    agg = agg.reset_index()
    ok = (agg["n"] == 3) & (agg["n_ret"] == 3) \
        & ((agg["rdq"] - agg["first"]).dt.days <= 7) & ((agg["last"] - agg["rdq"]).dt.days <= 10)
    agg = agg[ok]
    return pd.DataFrame({"permno": agg["permno"], "rdq": agg["rdq"], "ear_end_date": agg["last"],
                         "ear": np.expm1(agg["lr"]) - np.expm1(agg["lm"])})


def load_ff_daily(raw_dir):
    path = raw_dir / "ff_factors_daily.parquet"
    if not path.exists():
        return None
    f = pd.read_parquet(path)
    f["date"] = pd.to_datetime(f["date"])
    mkt = pd.to_numeric(f["mktrf"], errors="coerce") + pd.to_numeric(f["rf"], errors="coerce")
    if mkt.abs().median() > 0.5:
        mkt = mkt / 100.0
    return pd.DataFrame({"date": f["date"], "mkt": mkt})


def add_announcement_return(crsp, raw_dir, log):
    path = raw_dir / "earnings_announcement_days.parquet"
    if not path.exists():
        print("No earnings_announcement_days.parquet: announcement-return feature left out.")
        return crsp, []
    e = announcement_returns(pd.read_parquet(path), load_ff_daily(raw_dir))
    e["umnum"] = month_num(e["ear_end_date"] + pd.offsets.MonthEnd(0))
    e = e.sort_values(["permno", "umnum", "ear_end_date"]).drop_duplicates(["permno", "umnum"], keep="last")
    hit = latest_event(crsp, crsp["permno"], month_num(crsp["month"]).to_numpy(), e,
                       ["ear", "ear_end_date"], EAR_MAX_AGE_MONTHS)
    crsp["earnings_ann_return"] = crsp.pop("ear")
    log("announcement_return", "earnings releases with a 3-day return", int(len(e)))
    log("announcement_return", "stock-months with a recent announcement return", float(hit.mean()))
    return crsp, ANNOUNCEMENT_FEATURES


# ---------------------------------------------------------------- 13F


def institutional_events(raw_dir, ids):
    """13F ownership per stock and report quarter, with the month it becomes usable.

    ids: one row per (permno, month) with cusip8_hist (CRSP's CUSIP of the day) and shares
    (shares outstanding). Returns permno, umnum (first month-end after quarter end + 45
    days), io, io_change, breadth_change, inst_report_date; None without the 13F file."""
    hp, mp = raw_dir / "inst_13f_holdings.parquet", raw_dir / "inst_13f_managers.parquet"
    if not hp.exists():
        return None
    h = pd.read_parquet(hp)
    h["rdate"] = pd.to_datetime(h["rdate"]) + pd.offsets.MonthEnd(0)
    h["cusip8"] = h["cusip"].astype("string").str.strip().str.upper().str[:8]
    h = h.groupby(["cusip8", "rdate"], as_index=False).agg(inst_shares=("inst_shares", "sum"), n_inst=("n_inst", "sum"))
    # permno and shares outstanding at the report date, through CRSP's historical CUSIP
    look = pd.DataFrame({"cusip8": ids["cusip8_hist"].astype("string"), "rdate": pd.to_datetime(ids["month"]),
                         "permno": ids["permno"].astype("int64"),
                         "shares": ids["shares"].to_numpy(dtype="float64")})
    look = look.dropna(subset=["cusip8"]).drop_duplicates(["cusip8", "rdate", "permno"])
    h = h.merge(look, on=["cusip8", "rdate"], how="inner")
    h = h.groupby(["permno", "rdate"], as_index=False).agg(inst_shares=("inst_shares", "sum"),
                                                           n_inst=("n_inst", "sum"), shares=("shares", "first"))
    h["io"] = np.minimum(ratio(h["inst_shares"], h["shares"]), 1.0)
    if mp.exists():
        m = pd.read_parquet(mp)
        m["rdate"] = pd.to_datetime(m["rdate"]) + pd.offsets.MonthEnd(0)
        h["n_managers"] = h["rdate"].map(m.groupby("rdate")["n_managers"].sum())
    else:
        h["n_managers"] = np.nan
    h = h.sort_values(["permno", "rdate"])
    prev_ok = (h["permno"].shift(1) == h["permno"]) & ((month_num(h["rdate"]) - month_num(h["rdate"]).shift(1)) == 3)
    h["io_change"] = (h["io"] - h["io"].shift(1)).where(prev_ok)
    h["breadth_change"] = ((h["n_inst"] - h["n_inst"].shift(1)) / h["n_managers"].shift(1)).where(prev_ok)
    h["usable"] = (h["rdate"] + pd.Timedelta(days=INST_FILING_DAYS)) + pd.offsets.MonthEnd(0)
    h["umnum"] = month_num(h["usable"])
    return h.drop_duplicates(["permno", "umnum"], keep="last").rename(columns={"rdate": "inst_report_date"})


def add_institutional_features(crsp, raw_dir, log):
    ids = pd.DataFrame({"permno": crsp["permno"], "month": crsp["month"], "cusip8_hist": crsp["cusip8_hist"],
                        "shares": crsp["shrout"].to_numpy(dtype="float64") * 1000.0})
    h = institutional_events(raw_dir, ids)
    if h is None:
        print("No inst_13f_holdings.parquet: institutional-ownership features left out.")
        return crsp, []
    hit = latest_event(crsp, crsp["permno"], month_num(crsp["month"]).to_numpy(), h,
                       ["io", "io_change", "breadth_change", "inst_report_date"], INST_MAX_AGE_MONTHS)
    crsp["inst_ownership"] = crsp.pop("io")
    crsp["inst_ownership_change"] = crsp.pop("io_change")
    crsp["inst_breadth_change"] = crsp.pop("breadth_change")
    log("inst_13f", "stock-months with 13F holdings", float(hit.mean()))
    return crsp, INSTITUTIONAL_FEATURES


# ---------------------------------------------------------------- IBES analyst features


def _ibes_read(raw_dir, name):
    path = raw_dir / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "ticker" in df:
        df["ticker"] = df["ticker"].astype("string").str.strip()
    for c in ("statpers", "fpedats", "anndats_act", "anndats", "pends", "sdate", "edate"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "fpi" in df:
        df["fpi"] = df["fpi"].astype("string").str.strip()
    return df


def _link_ibes(df, link, date_col):
    """IBES ticker -> permno valid at date_col (best link score, then the first)."""
    x = df.merge(link, on="ticker", how="inner")
    edate = x["edate"].fillna(pd.Timestamp("2100-12-31"))
    x = x[(x[date_col] >= x["sdate"]) & (x[date_col] <= edate)]
    if "score" in x:
        x = x.sort_values("score", kind="stable")
    return x.drop(columns=["sdate", "edate"] + [c for c in ("score", "ncusip") if c in x])


def analyst_events(raw_dir, log):
    """Monthly analyst variables per (permno, statpers month) from the IBES summaries.

    Per-share forecasts are split-adjusted to today's share basis in statsum_epsus. To put
    them on the same basis as the CRSP price of the month, they are divided by the IBES
    adjustment factor of that date, f = adjusted / unadjusted forecast (statsumu_epsus).
    Without the unadjusted file the price-scaled inputs are left out."""
    link = _ibes_read(raw_dir, "ibes_link")
    s = _ibes_read(raw_dir, "ibes_statsum")
    if link is None or s is None:
        return None, None, None
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce")
    link = link.dropna(subset=["permno", "ticker"])
    if "score" in link:
        link = link[pd.to_numeric(link["score"], errors="coerce").fillna(0) <= IBES_LINK_MAX_SCORE]
    link["permno"] = link["permno"].astype("int64")
    link = link[[c for c in ("ticker", "permno", "sdate", "edate", "score") if c in link]]

    s = s.dropna(subset=["ticker", "statpers"])
    s["month"] = s["statpers"] + pd.offsets.MonthEnd(0)
    s = s.sort_values("statpers").drop_duplicates(["ticker", "month", "fpi"], keep="last")
    num = ["numest", "numup", "numdown", "meanest", "medest", "stdev", "highest", "lowest"]
    for c in num:
        if c in s:
            s[c] = pd.to_numeric(s[c], errors="coerce").astype("float64")
    wide = None
    for fpi, tag in (("1", "fy1"), ("2", "fy2"), ("0", "ltg"), ("6", "q1")):
        part = s.loc[s["fpi"] == fpi, ["ticker", "month", "statpers", "fpedats"] + [c for c in num if c in s]]
        part = part.rename(columns={c: f"{tag}_{c}" for c in part.columns if c not in ("ticker", "month")})
        wide = part if wide is None else wide.merge(part, on=["ticker", "month"], how="outer")
    if "fy1_statpers" not in wide:
        return None, None, None
    wide["statpers"] = wide["fy1_statpers"]
    for tag in ("q1", "fy2", "ltg"):
        if f"{tag}_statpers" in wide:
            wide["statpers"] = wide["statpers"].fillna(wide[f"{tag}_statpers"])

    # IBES adjustment factor of the date (adjusted / unadjusted), from the FY1 forecasts
    u = _ibes_read(raw_dir, "ibes_statsum_unadj")
    if u is not None and {"meanest", "fpi"} <= set(u.columns):
        u = u.dropna(subset=["ticker", "statpers"])
        u["month"] = u["statpers"] + pd.offsets.MonthEnd(0)
        ratios = []
        for fpi, tag in (("1", "fy1"), ("6", "q1")):
            uu = u[u["fpi"] == fpi].drop_duplicates(["ticker", "month"], keep="last")
            x = wide[["ticker", "month", f"{tag}_meanest"]].merge(uu[["ticker", "month", "meanest"]],
                                                                   on=["ticker", "month"], how="inner")
            r = (x[f"{tag}_meanest"] / x["meanest"]).where(x["meanest"].abs() > 1e-3)
            ratios.append(pd.Series(r.to_numpy(), index=pd.MultiIndex.from_frame(x[["ticker", "month"]])))
        f = pd.concat(ratios, axis=1).median(axis=1) if ratios else pd.Series(dtype=float)
        f = f[(f > 1e-4) & (f < 1e4)]
        wide["f_adj"] = pd.MultiIndex.from_frame(wide[["ticker", "month"]]).map(f)
        has_unadj = True
    else:
        wide["f_adj"] = np.nan
        has_unadj = False
        print("No ibes_statsum_unadj.parquet: the price-scaled analyst inputs are left out.")

    wide = _link_ibes(wide, link, "statpers")
    # several tickers on one permno-month: keep the most-followed one
    wide = wide.sort_values("fy1_numest", ascending=False).drop_duplicates(["permno", "month"])
    log("analyst", "permno-months with an FY1 consensus", int(wide["fy1_numest"].notna().sum()))
    log("analyst", "adjustment factor found (price-scaled inputs)", float(wide["f_adj"].notna().mean()))
    return wide, link, has_unadj


def analyst_surprises(raw_dir, link, s_wide):
    """Quarterly surprises against the last consensus before the announcement:
    (actual - consensus) / price, both per share on the consensus date's basis."""
    a = _ibes_read(raw_dir, "ibes_actuals")
    if a is not None and {"pends", "anndats", "value"} <= set(a.columns):
        a = a.rename(columns={"pends": "fpedats", "anndats": "ann_date", "value": "actual"})
    else:   # fall back on the actuals carried by the summaries
        s = _ibes_read(raw_dir, "ibes_statsum")
        if s is None or not {"actual", "anndats_act"} <= set(s.columns):
            return None
        a = s.loc[s["fpi"] == "6", ["ticker", "fpedats", "actual", "anndats_act"]].rename(
            columns={"anndats_act": "ann_date"})
    a["actual"] = pd.to_numeric(a["actual"], errors="coerce")
    a = a.dropna(subset=["ticker", "fpedats", "ann_date", "actual"]).drop_duplicates(["ticker", "fpedats"], keep="last")
    s = _ibes_read(raw_dir, "ibes_statsum")
    q = s.loc[s["fpi"] == "6", ["ticker", "statpers", "fpedats", "meanest"]].dropna()
    q["meanest"] = pd.to_numeric(q["meanest"], errors="coerce")
    q = q.merge(a[["ticker", "fpedats", "ann_date", "actual"]], on=["ticker", "fpedats"], how="inner")
    q = q[q["statpers"] < q["ann_date"]].sort_values("statpers").drop_duplicates(["ticker", "fpedats"], keep="last")
    q["month"] = q["statpers"] + pd.offsets.MonthEnd(0)
    # price and adjustment factor of the consensus month
    q = q.merge(s_wide[["ticker", "month", "f_adj"]].drop_duplicates(["ticker", "month"]), on=["ticker", "month"], how="left")
    q = _link_ibes(q, link, "statpers")
    return q.drop_duplicates(["permno", "fpedats"])


def add_analyst_features(crsp, raw_dir, log):
    """IBES consensus, revisions, surprise and recommendations for each stock-month."""
    wide, link, has_unadj = analyst_events(raw_dir, log)
    if wide is None:
        print("No IBES files (ibes_link / ibes_statsum): analyst features left out.")
        return crsp, []
    mn = month_num(crsp["month"]).to_numpy()
    price = np.abs(crsp["prc"].to_numpy(dtype="float64"))
    price = np.where(price > 0, price, np.nan)

    w = wide.copy()
    w["umnum"] = month_num(w["month"])
    w = w.sort_values(["permno", "umnum"])
    # revision of the FY1 consensus since last month, same fiscal year only (adjusted basis)
    same = ((w["permno"].shift(1) == w["permno"]) & (w["umnum"].shift(1) == w["umnum"] - 1)
            & (w["fy1_fpedats"].shift(1) == w["fy1_fpedats"]))
    w["d_fy1"] = (w["fy1_meanest"] - w["fy1_meanest"].shift(1)).where(same)
    vals = {
        "a_numest": w["fy1_numest"],
        "a_disp": np.where(w["fy1_numest"] >= 2, ratio(w.get("fy1_stdev", np.nan), w["fy1_meanest"].abs()), np.nan),
        "a_fy1": w["fy1_meanest"], "a_fy2": w.get("fy2_meanest", pd.Series(np.nan, index=w.index)),
        "a_ltg": w.get("ltg_meanest", pd.Series(np.nan, index=w.index)),
        "a_updown": ratio(w["fy1_numup"] - w["fy1_numdown"], w["fy1_numest"]) if "fy1_numup" in w else np.nan,
        "a_dfy1": w["d_fy1"], "a_f": w["f_adj"], "an_statpers": w["statpers"],
    }
    ev = pd.DataFrame({"permno": w["permno"].to_numpy(), "umnum": w["umnum"].to_numpy()})
    for k, v in vals.items():
        ev[k] = v.to_numpy() if hasattr(v, "to_numpy") else v
    ev = ev.drop_duplicates(["permno", "umnum"], keep="last")
    # the statpers of month t, or of month t-1 if t has none (then at most 50 days old)
    hit = latest_event(crsp, crsp["permno"], mn, ev, [c for c in ev.columns if c not in ("permno", "umnum")], 1)
    age = (crsp["month"] - pd.to_datetime(crsp["an_statpers"])).dt.days
    stale = hit & ~(age <= IBES_MAX_AGE_DAYS).to_numpy()
    for c in [c for c in ev.columns if c not in ("permno", "umnum")]:
        crsp.loc[stale, c] = pd.NaT if c == "an_statpers" else np.nan
    covered = hit & ~stale
    p_adj = price * crsp["a_f"].to_numpy(dtype="float64")       # the price on the forecasts' basis

    crsp["an_coverage"] = np.where(covered, np.log1p(crsp["a_numest"].fillna(0).to_numpy(dtype="float64")), 0.0).astype("float32")
    crsp["an_dispersion"] = crsp["a_disp"]
    crsp["an_fep"] = ratio(crsp["a_fy1"].to_numpy(dtype="float64"), p_adj, positive_den=True)
    crsp["an_fy2_growth"] = ratio((crsp["a_fy2"] - crsp["a_fy1"]).to_numpy(dtype="float64"), p_adj)
    crsp["an_ltg"] = crsp["a_ltg"]
    crsp["an_updown"] = crsp["a_updown"]
    crsp["an_rev_1m"] = ratio(crsp["a_dfy1"].to_numpy(dtype="float64"), p_adj)
    # 3-month revision: this month's and the two previous months' revisions (at least 2);
    # crsp is sorted by permno, month (add_market_features)
    r = pd.Series(crsp["an_rev_1m"].to_numpy(dtype="float64"))
    pno = crsp["permno"].to_numpy()
    mm = month_num(crsp["month"]).to_numpy()
    lag = lambda k: r.shift(k).where(consecutive(mm, pno, k))  # noqa: E731
    stack = pd.concat([r, lag(1), lag(2)], axis=1)
    crsp["an_rev_3m"] = stack.sum(axis=1, min_count=2).astype("float32")
    for c in ("an_fep", "an_fy2_growth", "an_rev_1m", "an_dispersion", "an_ltg", "an_updown"):
        crsp[c] = pd.to_numeric(crsp[c], errors="coerce").astype("float32")

    # earnings surprise against the consensus, usable from the announcement date + 1 day
    q = analyst_surprises(raw_dir, link, wide) if has_unadj else None
    if q is not None and len(q):
        pq = crsp[["permno", "month"]].assign(prc=price)
        q = q.merge(pq.rename(columns={"prc": "p_cons"}), on=["permno", "month"], how="left")
        q["sue"] = ratio(q["actual"] - q["meanest"], q["p_cons"] * q["f_adj"])
        q["umnum"] = month_num(q["ann_date"] + pd.Timedelta(days=1) + pd.offsets.MonthEnd(0))
        q = (q.dropna(subset=["sue"]).sort_values(["permno", "umnum", "ann_date"])
             .drop_duplicates(["permno", "umnum"], keep="last")
             .rename(columns={"ann_date": "an_sue_ann_date"}))
        hit_q = latest_event(crsp, crsp["permno"], mm, q, ["sue", "an_sue_ann_date"], IBES_SUE_MAX_AGE_MONTHS)
        crsp["an_sue"] = crsp.pop("sue")
        log("analyst", "stock-months with a recent surprise against the consensus", float(hit_q.mean()))
    else:
        crsp["an_sue"] = np.float32(np.nan)
        crsp["an_sue_ann_date"] = pd.NaT

    # recommendations (1 = strong buy ... 5 = sell)
    rec = _ibes_read(raw_dir, "ibes_recommendations")
    if rec is not None and {"meanrec", "statpers"} <= set(rec.columns):
        rec["meanrec"] = pd.to_numeric(rec["meanrec"], errors="coerce")
        rec = _link_ibes(rec.dropna(subset=["ticker", "statpers", "meanrec"]), link, "statpers")
        rec["umnum"] = month_num(rec["statpers"] + pd.offsets.MonthEnd(0))
        rec = rec.sort_values("statpers").drop_duplicates(["permno", "umnum"], keep="last")
        latest_event(crsp, crsp["permno"], mm, rec.rename(columns={"statpers": "_rec_date"}),
                     ["meanrec", "_rec_date"], 1)
        rec_age = (crsp["month"] - pd.to_datetime(crsp["_rec_date"])).dt.days
        crsp["an_rec_mean"] = crsp.pop("meanrec").where(rec_age <= IBES_MAX_AGE_DAYS).astype("float32")
        lag3 = crsp["an_rec_mean"].shift(3).where(consecutive(mm, pno, 3))
        crsp["an_rec_change_3m"] = (crsp["an_rec_mean"] - lag3).astype("float32")
        crsp = crsp.drop(columns=["_rec_date"])
        log("analyst", "stock-months with a recommendation consensus", float(crsp["an_rec_mean"].notna().mean()))
    else:
        crsp["an_rec_mean"] = np.float32(np.nan)
        crsp["an_rec_change_3m"] = np.float32(np.nan)
    crsp = crsp.drop(columns=[c for c in ("a_numest", "a_disp", "a_fy1", "a_fy2", "a_ltg", "a_updown", "a_dfy1", "a_f")
                              if c in crsp])
    log("analyst", "stock-months covered by IBES", float(covered.mean()))
    for c in ANALYST_FEATURES:
        log("analyst", f"{c}: share available", float(crsp[c].notna().mean()))
    return crsp, ANALYST_FEATURES


# ---------------------------------------------------------------- market regime


def _expanding_z(s, min_months=REGIME_MIN_MONTHS, clip=4.0):
    """(x_t - mean of x up to t) / std of x up to t: uses only months <= t."""
    s = s.astype("float64")
    mean = s.expanding(min_periods=min_months).mean()
    std = s.expanding(min_periods=min_months).std()
    return ((s - mean) / std).clip(-clip, clip)


def regime_table(raw_dir, universe, log):
    """One row per month-end: the market-regime inputs (z-scored over time).

    universe: the panel's stock-months (month, ret) for the cross-sectional dispersion."""
    out = {}
    ffm = raw_dir / "ff_factors_monthly.parquet"
    if ffm.exists():
        f = pd.read_parquet(ffm)
        f["month"] = pd.to_datetime(f["date"]) + pd.offsets.MonthEnd(0)
        f = f.set_index("month").sort_index()
        mkt = pd.to_numeric(f["mktrf"], errors="coerce") + pd.to_numeric(f["rf"], errors="coerce")
        if mkt.abs().median() > 0.5:
            mkt = mkt / 100.0
        lm = np.log1p(mkt)
        out["reg_mkt_ret_1m"] = mkt
        out["reg_mkt_ret_12m"] = np.expm1(lm.rolling(12, min_periods=12).sum())
        out["reg_mkt_ret_24m"] = np.expm1(lm.rolling(24, min_periods=24).sum())
        if "umd" in f:
            umd = pd.to_numeric(f["umd"], errors="coerce")
            if umd.abs().median() > 0.5:
                umd = umd / 100.0
            out["reg_umd_ret_12m"] = np.expm1(np.log1p(umd).rolling(12, min_periods=12).sum())
            out["reg_umd_vol_6m"] = umd.rolling(6, min_periods=6).std() * np.sqrt(12)
    ffd = raw_dir / "ff_factors_daily.parquet"
    if ffd.exists():
        d = pd.read_parquet(ffd)
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date").sort_index()
        dm = pd.to_numeric(d["mktrf"], errors="coerce") + pd.to_numeric(d["rf"], errors="coerce")
        if dm.abs().median() > 0.05:
            dm = dm / 100.0
        g = dm.groupby(dm.index + pd.offsets.MonthEnd(0))
        var = g.apply(lambda x: (x ** 2).mean())
        n = g.count()
        vol_1m = np.sqrt(252 * var).where(n >= 10)
        vol_12m = np.sqrt(252 * (var * n).rolling(12, min_periods=12).sum() / n.rolling(12, min_periods=12).sum())
        out["reg_mkt_vol_1m"] = vol_1m
        out["reg_mkt_vol_ratio"] = vol_1m / vol_12m
        if "umd" in d:
            du = pd.to_numeric(d["umd"], errors="coerce")
            if du.abs().median() > 0.05:
                du = du / 100.0
            gu = du.groupby(du.index + pd.offsets.MonthEnd(0))
            vu, nu = gu.apply(lambda x: (x ** 2).mean()), gu.count()
            out["reg_umd_vol_6m"] = np.sqrt(252 * (vu * nu).rolling(6, min_periods=6).sum()
                                            / nu.rolling(6, min_periods=6).sum())
    # cross-sectional dispersion of this month's returns (winsorised 1% / 99%)
    u = universe[["month", "ret"]].dropna()
    lo = u.groupby("month")["ret"].transform(lambda x: x.quantile(0.01))
    hi = u.groupby("month")["ret"].transform(lambda x: x.quantile(0.99))
    out["reg_xs_dispersion"] = u["ret"].clip(lo, hi).groupby(u["month"]).std()
    mac = raw_dir / "macro_fred.parquet"
    if mac.exists():
        m = pd.read_parquet(mac)
        m["month"] = pd.to_datetime(m["date"]) + pd.offsets.MonthEnd(0)
        m = m.set_index("month").sort_index()
        if {"baa", "aaa"} <= set(m.columns):
            out["reg_credit_spread"] = m["baa"] - m["aaa"]
        if {"gs10", "tb3ms"} <= set(m.columns):
            out["reg_term_spread"] = m["gs10"] - m["tb3ms"]
    else:
        print("No macro_fred.parquet: credit and term spreads left out of the regime inputs.")
    t = pd.DataFrame({k: pd.Series(v) for k, v in out.items()}).sort_index()
    t = t[[c for c in REGIME_FEATURES if c in t.columns]]
    z = t.apply(_expanding_z)
    for c in z.columns:
        log("regime", f"{c}: first month", str(t[c].first_valid_index())[:10])
    return z.astype("float32")


def add_market_features(crsp, raw_dir, log):
    """All market-side features (before the universe filter, on full histories)."""
    crsp = crsp.sort_values(["permno", "month"]).reset_index(drop=True)
    crsp = add_crsp_monthly_features(crsp, log)
    crsp, _ = add_daily_features(crsp, raw_dir, log)
    crsp, _ = add_short_interest_features(crsp, raw_dir, log)
    crsp, _ = add_announcement_return(crsp, raw_dir, log)
    crsp, _ = add_institutional_features(crsp, raw_dir, log)
    crsp, _ = add_analyst_features(crsp, raw_dir, log)
    crsp = crsp.drop(columns=["_vol_adj"])
    return crsp


# ---------------------------------------------------------------- Compustat


def merge_fundq_extra(comp, raw_dir, log):
    """Add the comp_fundq_extra.parquet items the features use (same rows as
    comp_fundq.parquet; matched on gvkey, datadate, fiscal year / quarter / year end)."""
    path = raw_dir / "comp_fundq_extra.parquet"
    if not path.exists():
        print("No comp_fundq_extra.parquet: the features that need the extra Compustat items "
              "(special items, intangibles, share / debt issuance, acquisitions) are left out.")
        return comp
    x = pd.read_parquet(path)
    x.columns = [c.lower() for c in x.columns]
    keep = [c for c in FUNDQ_EXTRA_USED if c in x.columns and c not in comp.columns]
    if not keep:
        return comp
    x = x[FUNDQ_KEYS + keep].copy()
    x["gvkey"] = x["gvkey"].astype("string").str.strip()
    x["datadate"] = pd.to_datetime(x["datadate"], errors="coerce")
    for c in ["fyearq", "fqtr", "fyr"] + keep:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype("float64")
    x = x.drop_duplicates(FUNDQ_KEYS)
    left = comp[FUNDQ_KEYS].copy()
    left["gvkey"] = left["gvkey"].astype("string").str.strip()
    left["datadate"] = pd.to_datetime(left["datadate"], errors="coerce")
    for c in ["fyearq", "fqtr", "fyr"]:
        left[c] = pd.to_numeric(left[c], errors="coerce").astype("float64")
    left["_row"] = np.arange(len(left))
    m = left.merge(x, on=FUNDQ_KEYS, how="left")
    assert len(m) == len(comp), "duplicate keys in comp_fundq_extra"
    m = m.sort_values("_row")
    for c in keep:
        comp[c] = m[c].to_numpy()
    matched = m[keep].notna().any(axis=1).mean()
    log("fundq_extra", "items added", ", ".join(keep))
    log("fundq_extra", "filings with at least one extra item", float(matched))
    return comp


def structural_zero_extra_ytd(comp, log):
    """Buybacks, share issuance, debt issuance / repayment and acquisitions not reported by
    a firm that reports its cash-flow statement are 0 (as the structural zeros of
    build_compustat_features)."""
    if "oancfy" not in comp:
        return comp
    reports_cf = pd.to_numeric(comp["oancfy"], errors="coerce").notna()
    for c in EXTRA_YTD_ITEMS:
        if c in comp:
            v = pd.to_numeric(comp[c], errors="coerce")
            fill = reports_cf & v.isna()
            comp[c] = v.where(~fill, 0.0)
            log("fundq_extra", f"{c}: share of cash-flow filings set to 0", float(fill[reports_cf].mean()))
    return comp


def build_accounting_features(comp, firm_key, add_feature, log):
    """Balance-sheet, cash-flow and earnings-surprise features (comp sorted by firm,
    datadate; every feature uses this filing and earlier filings of the same firm)."""
    keys = comp[firm_key]
    mnum = (comp["datadate"].dt.year * 12 + comp["datadate"].dt.month).to_numpy()
    _add = add_feature

    def add_feature(name, values):  # noqa: F811 (ndarray or Series -> Series on comp's index)
        _add(name, pd.Series(np.asarray(values, dtype="float64"), index=comp.index))

    num = lambda c: pd.to_numeric(comp[c], errors="coerce").astype("float64") if c in comp else pd.Series(np.nan, index=comp.index)
    gap = {k: (pd.Series(mnum, index=comp.index) - pd.Series(mnum, index=comp.index).groupby(keys).shift(k)) for k in (1, 3, 4, 7)}

    def lag(s, k, lo, hi):
        return s.groupby(keys).shift(k).where(gap[k].between(lo, hi))

    def ttm(s):
        tot = s + s.groupby(keys).shift(1) + s.groupby(keys).shift(2) + s.groupby(keys).shift(3)
        return tot.where(gap[3].between(8, 10))

    def zero_if_missing(s, ok):
        return s.where(s.notna() | ~ok, 0.0)

    at, ceq, seq, lt = num("atq"), num("ceqq"), num("seqq"), num("ltq")
    act, lct, che, invt = num("actq"), num("lctq"), num("cheq"), num("invtq")
    ppent, dltt, dlc = num("ppentq"), num("dlttq"), num("dlcq")
    txditc, pstk = num("txditcq"), num("pstkq")
    mib = num("mibtq").fillna(num("mibq"))
    at_ok = at > 0
    at = at.where(at_ok)
    has_bs = at.notna()
    ib = num("ibq").fillna(num("niy_qtr"))
    sale = num("saley_qtr")
    cogs, xsga, xint = num("cogsy_qtr"), num("xsgay_qtr"), num("xinty_qtr")
    xrd, oibdp, oancf = num("xrdy_qtr"), num("oibdpy_qtr"), num("oancfy_qtr")
    dv, capx = num("dvy_qtr"), num("capxy_qtr")
    prstkc, sstk = num("prstkcy_qtr"), num("sstky_qtr")
    dltis, dltr, aqc = num("dltisy_qtr"), num("dltry_qtr"), num("aqcy_qtr")

    # book equity (Hou-Xue-Zhang quarterly): stockholders' equity + deferred taxes - preferred
    se = seq.fillna(ceq + pstk.fillna(0)).fillna(at - lt)
    be = se + txditc.fillna(0) - pstk.fillna(0)
    debt = dltt.fillna(0) + dlc.fillna(0)
    debt = debt.where(dltt.notna() | dlc.notna())
    at_l1, at_l4 = lag(at, 1, 2, 4), lag(at, 4, 11, 13)
    be_l1 = lag(be, 1, 2, 4)
    sale_ttm, cogs_ttm, xsga_ttm, xint_ttm = ttm(sale), ttm(cogs), ttm(xsga), ttm(xint)
    ib_ttm, oancf_ttm, capx_ttm = ttm(ib), ttm(oancf), ttm(capx)

    roa = pd.Series(ratio(ib, at_l1), index=comp.index)
    add_feature("roa_qtr", roa)
    add_feature("roe_qtr", ratio(ib, be_l1))
    add_feature("gross_profit_to_assets", ratio(sale_ttm - cogs_ttm, at))
    add_feature("operating_profitability", ratio(sale_ttm - cogs_ttm - xsga_ttm.fillna(0) - xint_ttm.fillna(0), be))
    add_feature("asset_growth_yoy", ratio(at, at_l4) - 1)
    inv = (ppent - lag(ppent, 4, 11, 13)) + (invt.fillna(0) - lag(invt, 4, 11, 13).fillna(0))
    add_feature("investment_to_assets_yoy", ratio(inv, at_l4))
    add_feature("book_leverage", ratio(debt, at))
    add_feature("liabilities_to_assets", ratio(lt, at))
    add_feature("current_ratio", ratio(act, lct))
    add_feature("quick_ratio", ratio(act - invt.fillna(0), lct))
    add_feature("cash_to_assets", ratio(che, at))
    add_feature("ppe_to_assets", ratio(ppent, at))
    add_feature("accruals_to_assets", ratio(ib_ttm - oancf_ttm, at))
    # net operating assets (Hirshleifer et al.): operating assets - operating liabilities
    oa = at - che.fillna(0)
    ol = at - dlc.fillna(0) - dltt.fillna(0) - mib.fillna(0) - pstk.fillna(0) - ceq
    add_feature("net_operating_assets", ratio(oa - ol, at_l1))
    add_feature("asset_turnover", ratio(sale_ttm, at))
    add_feature("roa_change_yoy", roa - lag(roa, 4, 11, 13))
    roa_sd = roa.rolling(8, min_periods=8).std().where(gap[7].between(20, 22))
    add_feature("roa_vol_8q", roa_sd)
    add_feature("intangibles_to_assets", ratio(num("intanq"), at))
    add_feature("special_items_to_assets", ratio(num("spiq"), at_l1))
    add_feature("net_debt_issuance_to_assets", ratio(ttm(dltis) - ttm(dltr), at))
    add_feature("net_equity_issuance_to_assets", ratio(ttm(sstk) - ttm(prstkc), at))
    add_feature("acquisitions_to_assets", ratio(ttm(aqc), at))
    add_feature("capex_to_assets", ratio(capx_ttm, at))

    # earnings surprises: seasonal random walk, scaled by the last 8 quarters' surprises
    for name, s in (("sue_earnings", ib), ("sue_sales", sale)):
        d4 = s - lag(s, 4, 11, 13)
        sd = d4.rolling(8, min_periods=6).std().where(gap[7].between(20, 24))
        add_feature(name, ratio(d4, sd))

    # helpers for ratios to market value (formed after the merge)
    add_feature("h_ib_ttm", ib_ttm)
    add_feature("h_sale_ttm", sale_ttm.where(sale_ttm > 0))
    add_feature("h_xrd_ttm", ttm(xrd))
    add_feature("h_oibdp_ttm", ttm(oibdp))
    add_feature("h_be", be)
    add_feature("h_debt", debt)
    add_feature("h_cash", che)
    add_feature("h_pstk", pstk)
    add_feature("h_mib", mib)
    add_feature("h_oancf_ttm", oancf_ttm)
    payout = ttm(dv) + ttm(prstkc)
    add_feature("h_payout_ttm", payout)
    add_feature("h_net_payout_ttm", payout - ttm(sstk))
    log("balance_sheet", "filings with total assets", float(has_bs.mean()))
    log("balance_sheet", "filings with book equity > 0", float((be > 0).mean()))
    return comp


def add_market_ratio_features(panel, log, name="panel"):
    """Accounting amounts over the market value of the month the feature is used."""
    me = pd.to_numeric(panel["mktcap"], errors="coerce").astype("float64") / 1000.0   # $ millions
    me = me.where(me > 0)
    g = lambda c: pd.to_numeric(panel[f"comp_{c}"], errors="coerce").astype("float64") if f"comp_{c}" in panel else pd.Series(np.nan, index=panel.index)
    be = g("h_be")
    out = {
        "book_to_market": ratio(be.where(be > 0), me),
        "earnings_to_price": ratio(g("h_ib_ttm"), me),
        "sales_to_price": ratio(g("h_sale_ttm"), me),
        "cashflow_to_price": ratio(g("h_oancf_ttm"), me),
        "rd_to_market": ratio(g("h_xrd_ttm"), me),
        "payout_yield": ratio(g("h_payout_ttm"), me),
        "net_payout_yield": ratio(g("h_net_payout_ttm"), me),
    }
    debt = g("h_debt")
    ev = me + debt.fillna(0) + g("h_pstk").fillna(0) + g("h_mib").fillna(0) - g("h_cash").fillna(0)
    out["ebitda_to_ev"] = ratio(g("h_oibdp_ttm"), ev.where(debt.notna() | g("h_cash").notna()))
    out["market_leverage"] = ratio(debt, debt + me)
    for k, v in out.items():
        panel[f"comp_{k}"] = np.asarray(v, dtype="float64").astype("float32")
        log("market_ratios", f"{name} comp_{k}: share available", float(np.isfinite(panel[f"comp_{k}"]).mean()))
    panel["comp_flag_negative_book_equity"] = (be <= 0).astype("int8").where(be.notna())
    helpers = [c for c in panel.columns if c.startswith("comp_h_")]
    panel.drop(columns=helpers, inplace=True)
    return panel
