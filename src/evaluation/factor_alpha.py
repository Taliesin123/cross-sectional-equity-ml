"""
Alpha after known factors: is the book's return more than exposure to published factors?

Each book's monthly net return (from cost_study.py, book_returns.csv) is regressed on
    market (mktrf), size (smb), value (hml), profitability (rmw), investment (cma)
        Fama-French five factors, ff.fivefactors_monthly
    momentum (umd)                      from the same table
    short-term reversal (str)           built here from CRSP, as Fama and French build
                                        theirs: six value-weighted portfolios of US common
                                        stocks on size (NYSE median) and last month's
                                        return (NYSE 30th / 70th percentiles);
                                        str = 1/2 (small low + big low) - 1/2 (small high + big high)
r_book[t] = alpha + b' f[t+1] + e: the book's weights are set at the end of month t, its
return is earned in month t+1 (from the first trading day of t+1 with the one-day delay),
so it is matched with the factors of month t+1. The intercept is the return the factors
do not explain; its t-statistic uses Newey-West standard errors (3 lags).

Run from the project root (after cost_study.py):
    python src/evaluation/factor_alpha.py --run wf_full
Outputs: results/cost_study_<run>/factor_alpha.csv (+ printed table).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from preprocessing.build_processed_data import CRSP_COMMON_STOCK  # noqa: E402

FACTORS = ["mktrf", "smb", "hml", "rmw", "cma", "umd", "str"]
NW_LAGS = 3


def short_term_reversal(raw_dir):
    """Monthly short-term reversal factor from CRSP (factor month = index)."""
    cols = ["permno", "mthcaldt", "mthret", "mthcap", "primaryexch"] + [c for c in CRSP_COMMON_STOCK if c != "primaryexch"]
    m = pd.read_parquet(raw_dir / "crsp_msf_v2.parquet", columns=cols)
    m["month"] = pd.to_datetime(m["mthcaldt"]) + pd.offsets.MonthEnd(0)
    ok = pd.Series(True, index=m.index)
    for col, allowed in CRSP_COMMON_STOCK.items():
        ok &= m[col].astype("string").str.strip().str.upper().isin(allowed).fillna(False)
    m = m.loc[ok, ["permno", "month", "mthret", "mthcap", "primaryexch"]].copy()
    m["permno"] = pd.to_numeric(m["permno"], errors="coerce")
    m["ret"] = pd.to_numeric(m["mthret"], errors="coerce")
    m["cap"] = pd.to_numeric(m["mthcap"], errors="coerce")
    m = m.dropna(subset=["permno"]).sort_values(["permno", "month"])
    prev = m.groupby("permno").shift(1)
    consecutive = (m["month"] - pd.offsets.MonthEnd(1)) == prev["month"]
    m["prior_ret"] = prev["ret"].where(consecutive)
    m["prior_cap"] = prev["cap"].where(consecutive)
    m["nyse"] = m["primaryexch"].astype("string").str.strip().str.upper() == "N"
    m = m.dropna(subset=["ret", "prior_ret", "prior_cap"])
    m = m[m["prior_cap"] > 0]
    g = m[m["nyse"]].groupby("month")
    bp = pd.DataFrame({"size": g["prior_cap"].median(), "lo": g["prior_ret"].quantile(0.3),
                       "hi": g["prior_ret"].quantile(0.7)})
    m = m.join(bp, on="month").dropna(subset=["size", "lo", "hi"])
    m["big"] = m["prior_cap"] > m["size"]
    m["bucket"] = np.select([m["prior_ret"] <= m["lo"], m["prior_ret"] >= m["hi"]], ["low", "high"], "mid")
    m["wr"] = m["ret"] * m["prior_cap"]
    p = m.groupby(["month", "big", "bucket"]).agg(wr=("wr", "sum"), w=("prior_cap", "sum"))
    p = (p["wr"] / p["w"]).unstack(["big", "bucket"])
    return 0.5 * (p[(False, "low")] + p[(True, "low")]) - 0.5 * (p[(False, "high")] + p[(True, "high")])


def load_factors(raw_dir):
    path = raw_dir / "ff_fivefactors_monthly.parquet"
    if not path.exists():
        sys.exit(f"{path} not found: run  python src/data/download_wrds_extra.py --username <you> --only ff5")
    f = pd.read_parquet(path)
    f["month"] = pd.to_datetime(f["date"]) + pd.offsets.MonthEnd(0)
    f = f.set_index("month").sort_index()
    out = pd.DataFrame(index=f.index)
    for c in ("mktrf", "smb", "hml", "rmw", "cma", "umd"):
        if c in f:
            x = pd.to_numeric(f[c], errors="coerce")
            out[c] = x / 100.0 if x.abs().median() > 0.2 else x
    out["str"] = short_term_reversal(raw_dir).reindex(out.index)
    return out


def newey_west(X, e, lags=NW_LAGS):
    """Newey-West covariance of OLS coefficients."""
    n = len(e)
    xe = X * e[:, None]
    s = xe.T @ xe
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = xe[lag:].T @ xe[:-lag]
        s += w * (g + g.T)
    xtx_inv = np.linalg.inv(X.T @ X)
    return xtx_inv @ s @ xtx_inv * n / max(n - X.shape[1], 1)


def regress(r, F):
    """OLS of r on the factors F (aligned, no NaN). Returns a dict."""
    X = np.column_stack([np.ones(len(r)), F.to_numpy(dtype=float)])
    y = r.to_numpy(dtype=float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    se = np.sqrt(np.diag(newey_west(X, e)))
    out = {"alpha_per_year": 12 * b[0], "alpha_t": b[0] / se[0],
           "r2": 1 - e.var() / y.var() if y.var() > 0 else np.nan, "months": len(y),
           "raw_return_per_year": 12 * y.mean(),
           "residual_vol_per_year": np.sqrt(12) * e.std()}
    for name, coef, s in zip(F.columns, b[1:], se[1:]):
        out[f"beta_{name}"] = coef
        out[f"t_{name}"] = coef / s
    out["information_ratio"] = out["alpha_per_year"] / out["residual_vol_per_year"] if out["residual_vol_per_year"] > 0 else np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="Alpha of the books after FF5 + momentum + short-term reversal.")
    parser.add_argument("--run", default="wf_full", help="results/cost_study_<run>/book_returns.csv")
    parser.add_argument("--books", nargs="+", default=None, help="book letters (default: all)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    study = root / "results" / f"cost_study_{args.run}"
    path = study / "book_returns.csv"
    if not path.exists():
        sys.exit(f"{path} not found: run  python src/evaluation/cost_study.py --run {args.run}  first")
    br = pd.read_csv(path, parse_dates=["month"])
    factors = load_factors(root / "data" / "raw_wrds")
    factors.index = factors.index - pd.offsets.MonthEnd(1)       # factors of t+1 on the book's row t
    cols = [c for c in FACTORS if c in factors.columns and factors[c].notna().any()]
    rows = []
    for (book, costs), g in br.groupby(["book", "costs"], sort=False):
        if args.books and book.split(".")[0] not in args.books:
            continue
        g = g.set_index("month").sort_index()
        for split in ("val", "test", "recent", "all"):
            r = g["net"] if split == "all" else g.loc[g["split"] == split, "net"]
            d = pd.concat([r.rename("r"), factors[cols]], axis=1, join="inner").dropna()
            if len(d) < 24:
                continue
            rows.append({"book": book, "costs": costs, "split": split, **regress(d["r"], d[cols])})
    table = pd.DataFrame(rows)
    table.to_csv(study / "factor_alpha.csv", index=False)
    show = ["book", "costs", "split", "raw_return_per_year", "alpha_per_year", "alpha_t", "r2",
            "information_ratio"] + [f"beta_{c}" for c in cols]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(f"Net returns regressed on {', '.join(cols)} (Newey-West t, {NW_LAGS} lags):")
        print(table[[c for c in show if c in table]].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nSaved", study / "factor_alpha.csv")


if __name__ == "__main__":
    main()
