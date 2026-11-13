"""
Download the raw data of the market, daily, short-interest, announcement and 13F features,
and the Compustat items the core download leaves out (step 4 of 5).

Run on your own computer, from the project root, after the other downloads:
    python src/data/download_wrds_features.py --username YOUR_WRDS_USERNAME

Steps, in this order (run one step only with  --only <step>):
  fundq_extra     comp.fundq items the core download leaves out: more income-statement
                  items (special items, pretax income, EPS ...), more balance-sheet items
                  (payables, goodwill, intangibles, minority interest ...) and the rest of
                  the cash-flow statement (share issuance and buybacks, debt issuance and
                  repayment, acquisitions ...). Same rows and filters as comp_fundq.parquet.
  ff_daily        ff.factors_daily: daily market return (daily betas, announcement returns)
  daily_features  crsp.dsf_v2 summarised per stock-month ON THE WRDS SERVER (one row per
                  stock-month comes back), 1978 onwards: sums of daily returns, squares and
                  cubes, products with the market return (today's and yesterday's), max /
                  min daily return, zero-return days, the last five days' squared returns,
                  Amihud illiquidity, dollar volume, quoted spread and high-low spread
                  inputs. (crsp_daily_costs.parquet, for the cost model, starts in 2003:
                  too late for training features.)
  announcements   daily returns from 10 days before to 10 days after each earnings release
                  (comp.fundq rdq, linked to permnos with the CCM table on the server), 1979
                  onwards, for the earnings-announcement return.
  short_interest  comp.sec_shortint, plus comp.sec_shortint_legacy if your account has it,
                  from 1973 (comp.sec_shortint alone starts in July 2006; without older
                  data the build drops the short-interest features, missing in every
                  training year).
  inst_13f        Thomson-Reuters 13F institutional holdings (tr_13f.s34 or tfn.s34),
                  summed per stock and quarter on the server: shares held by institutions,
                  number of institutions, and the number of filing institutions per quarter.
                  Skipped with a message if your subscription does not include it (or with
                  --skip-13f). This is the slowest step.
  + checks which other libraries your account can read (IBES analyst forecasts ...) and
    writes it to the report.

Outputs in data/raw_wrds/ (licensed data, never commit them):
  comp_fundq_extra.parquet, ff_factors_daily.parquet, crsp_daily_features.parquet,
  earnings_announcement_days.parquet, short_interest_full.parquet,
  inst_13f_holdings.parquet, inst_13f_managers.parquet, download_report_features.json

Every step is fetched in chunks saved as they arrive (data/raw_wrds/chunks/); running the
command again resumes from the first missing chunk (the current year is refreshed).
Column names are looked up, not assumed: a column the table does not have is skipped and
listed in the report.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STEPS = ["fundq_extra", "ff_daily", "daily_features", "announcements", "short_interest", "inst_13f"]

FUNDQ_KEYS = ["gvkey", "datadate", "fyearq", "fqtr", "fyr"]
FUNDQ_FILTER = "indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'"
FUNDQ_EXTRA = [
    # income statement (quarterly)
    "revtq", "piq", "spiq", "xoprq", "nopiq", "xidoq", "epspxq", "epsfxq", "ajexq",
    "cshfdq", "dvpsxq",
    # balance sheet (quarterly)
    "apq", "txpq", "intanq", "gdwlq", "mibq", "mibtq", "teqq", "wcapq", "aoq", "loq",
    "lltq", "ivltq", "ivstq", "ivaoq", "acoq", "lcoq", "ppegtq", "dpactq", "lseq",
    # cash-flow statement (year-to-date, like the core download's items)
    "ibcy", "sstky", "prstkcy", "dltisy", "dltry", "dlcchy", "aqcy", "sppey", "ivchy",
    "sivy", "cdvcy", "txdcy", "apalchy", "txachy", "aolochy", "exrey", "fiaoy", "esubcy",
    "xidocy", "intpny", "txpdy",
]

DAILY_OPTIONAL = ["dlyprc", "dlyvol", "dlyprcvol", "dlybid", "dlyask", "dlyhigh", "dlylow"]
SHORTINT_FIRST_YEAR = 1973
OTHER_LIBRARIES = {
    "ibes": ["statsum_epsus", "act_epsus", "det_epsus"],
    "wrdsapps": ["ibcrsphist"],
    "tr_13f": ["s34"],
    "tfn": ["s34"],
    "comp": ["sec_shortint", "co_filedate"],
}


def rollback(db):
    try:
        db.connection.rollback()
    except Exception:
        pass


def columns_or_none(db, library, table):
    from download_wrds import available_columns
    try:
        return set(available_columns(db, library, table))
    except Exception:
        rollback(db)
        return None


def to_float32(df, keep=("permno",)):
    """Halve the memory of a downloaded chunk: float64 -> float32 (sums of daily returns
    and their squares keep ~7 significant digits, plenty for these features)."""
    for c in df.columns:
        if c not in keep and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def year_chunks(db, name, years, make_sql, date_cols, chunk_dir, refresh_current=True, levels=None,
                compact=False):
    """Run make_sql(year, level) for each year, saving chunks/<name>_<year>.parquet.
    levels: query versions to try in order on the first year that is fetched (the first
    that works is kept for the following years). compact: store floats as float32."""
    import pandas as pd

    this_year = int(time.strftime("%Y"))
    chosen = None
    t_all = time.time()
    for i, year in enumerate(years, 1):
        path = chunk_dir / f"{name}_{year}.parquet"
        if path.exists() and not (refresh_current and year >= this_year):
            print(f"  [{i}/{len(years)}] {year}: already downloaded, skipped")
            continue
        t0 = time.time()
        tries = [chosen] if chosen else (levels or [None])
        df = None
        for lv in tries:
            try:
                df = db.raw_sql(make_sql(year, lv), date_cols=date_cols)
                if chosen is None and levels and lv != levels[0]:
                    print(f"  note: the fuller query failed, using the '{lv}' version")
                chosen = lv
                break
            except Exception as exc:
                rollback(db)
                if lv == tries[-1]:
                    raise
                print(f"  query version '{lv}' failed ({str(exc).splitlines()[0][:140]}), trying a simpler one")
        if compact:
            df = to_float32(df)
        df.to_parquet(path, index=False)
        print(f"  [{i}/{len(years)}] {year}: {len(df):,} rows in {time.time() - t0:.0f}s "
              f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)
        del df
    parts = [pd.read_parquet(p) for p in sorted(chunk_dir.glob(f"{name}_*.parquet"))]
    parts = [p for p in parts if len(p)]
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), chosen


# ---------------------------------------------------------------- 1. fundq extra


def step_fundq_extra(db, out, report):
    from download_wrds import pick, download_in_chunks, save

    cols = pick(db, "comp", "fundq", FUNDQ_KEYS + FUNDQ_EXTRA, report)
    missing_keys = [k for k in FUNDQ_KEYS if k not in cols]
    if missing_keys:
        raise RuntimeError(f"comp.fundq has no {missing_keys}")
    print(f"  {len(cols) - len(FUNDQ_KEYS)} extra items")
    df = download_in_chunks(db, "fqextra",  # not "fundq_...": download_wrds.py reads chunks/fundq_*
                            f"SELECT {', '.join(cols)} FROM comp.fundq WHERE {FUNDQ_FILTER}",
                            "datadate", out, report)
    save(df, out / "comp_fundq_extra.parquet", report, "comp_fundq_extra", "datadate")
    for c in ["atq", "spiq", "sstky", "prstkcy", "dltisy", "apq", "gdwlq", "epspxq"]:
        if c in df:
            print(f"  {c}: present on {df[c].notna().mean():.0%} of filings")


# ---------------------------------------------------------------- 2. ff daily


def step_ff_daily(db, out, report):
    from download_wrds import save

    have = columns_or_none(db, "ff", "factors_daily")
    if not have or not {"date", "mktrf", "rf"} <= have:
        print("  ff.factors_daily not available: daily betas and announcement returns will use "
              "raw returns (no market adjustment).")
        report["ff_daily"] = "not available"
        return
    keep = [c for c in ["date", "mktrf", "smb", "hml", "rf", "umd"] if c in have]
    df = db.raw_sql(f"SELECT {', '.join(keep)} FROM ff.factors_daily WHERE date >= '1970-01-01'",
                    date_cols=["date"])
    save(df, out / "ff_factors_daily.parquet", report, "ff_factors_daily", "date")


# ---------------------------------------------------------------- 3. daily features


def daily_features_sql(have, has_ff, year, level):
    """One year of crsp.dsf_v2 -> one row per (permno, month) of sufficient statistics.

    level "full": everything; "no_hl": no high-low spread input (no LEAD); "minimal": no
    last-five-days and no high-low (no window functions at all)."""
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    prc = "ABS(s.dlyprc)"
    if {"dlyprc", "dlyvol"} <= have:
        dv = f"{prc} * s.dlyvol"
    elif "dlyprcvol" in have:
        dv = "s.dlyprcvol"
    else:
        dv = "CAST(NULL AS double precision)"
    if {"dlybid", "dlyask"} <= have:
        qs = ("CASE WHEN s.dlybid > 0 AND s.dlyask > s.dlybid "
              "THEN (s.dlyask - s.dlybid) / ((s.dlyask + s.dlybid) / 2.0) END")
    else:
        qs = "CAST(NULL AS double precision)"
    use_hl = level == "full" and {"dlyhigh", "dlylow", "dlyprc"} <= have
    if use_hl:
        c = f"CASE WHEN {prc} > 0 THEN LN({prc}) END"
        eta = ("CASE WHEN s.dlyhigh > 0 AND s.dlylow > 0 AND s.dlyhigh >= s.dlylow "
               "THEN (LN(s.dlyhigh) + LN(s.dlylow)) / 2.0 END")
    else:
        c = eta = "CAST(NULL AS double precision)"
    null = "CAST(NULL AS double precision)"
    if has_ff:
        mkt_cte = f"""m AS (
            SELECT date, mktrf + rf AS mkt, LAG(mktrf + rf) OVER (ORDER BY date) AS mkt_lag
            FROM ff.factors_daily
            WHERE date >= DATE '{start}' - 15 AND date < '{end}'
        ),"""
        mkt_join = "LEFT JOIN m ON m.date = s.dlycaldt"
        mkt, mkt_lag = "m.mkt", "m.mkt_lag"
    else:
        mkt_cte, mkt_join, mkt, mkt_lag = "", "", null, null
    rn = ("ROW_NUMBER() OVER (PARTITION BY s.permno, date_trunc('month', s.dlycaldt) "
          "ORDER BY s.dlycaldt DESC)") if level != "minimal" else "CAST(NULL AS integer)"
    eta_next = ("LEAD(eta) OVER (PARTITION BY permno ORDER BY dlycaldt)" if use_hl else null)
    return f"""
        WITH {mkt_cte}
        d AS (
            SELECT s.permno, s.dlycaldt,
                   CAST(date_trunc('month', s.dlycaldt) AS date) AS month_start,
                   s.dlyret AS r, {mkt} AS mkt, {mkt_lag} AS mkt_lag,
                   {dv} AS dv, {qs} AS qs, {c} AS c, {eta} AS eta,
                   {rn} AS rn_desc
            FROM crsp.dsf_v2 AS s
            {mkt_join}
            WHERE s.dlycaldt >= '{start}' AND s.dlycaldt < '{end}'
        ), e AS (
            SELECT d.*, {eta_next} AS eta_next FROM d
        )
        SELECT permno, month_start,
               COUNT(*) AS n_days,
               COUNT(r) AS n_ret,
               SUM(r) AS s_r, SUM(r * r) AS s_rr, SUM(r * r * r) AS s_rrr,
               MAX(r) AS max_r, MIN(r) AS min_r,
               SUM(CASE WHEN r = 0 THEN 1 ELSE 0 END) AS n_zero,
               COUNT(r * mkt) AS n_rm,
               SUM(CASE WHEN mkt IS NOT NULL THEN r END) AS s_r_m,
               SUM(CASE WHEN mkt IS NOT NULL THEN r * r END) AS s_rr_m,
               SUM(CASE WHEN r IS NOT NULL THEN mkt END) AS s_m,
               SUM(CASE WHEN r IS NOT NULL THEN mkt * mkt END) AS s_mm,
               SUM(r * mkt) AS s_rm,
               COUNT(r * mkt_lag) AS n_rml,
               SUM(CASE WHEN mkt_lag IS NOT NULL THEN r END) AS s_r_ml,
               SUM(CASE WHEN r IS NOT NULL THEN mkt_lag END) AS s_ml,
               SUM(CASE WHEN r IS NOT NULL THEN mkt_lag * mkt_lag END) AS s_mlml,
               SUM(r * mkt_lag) AS s_rml,
               SUM(CASE WHEN rn_desc <= 5 THEN r * r END) AS s_rr_last5,
               COUNT(CASE WHEN rn_desc <= 5 THEN r END) AS n_last5,
               SUM(CASE WHEN dv > 0 THEN ABS(r) / dv END) AS s_amihud,
               COUNT(CASE WHEN dv > 0 THEN r END) AS n_amihud,
               SUM(dv) AS s_dv,
               SUM(CASE WHEN dv > 0 THEN 1 ELSE 0 END) AS n_traded,
               AVG(CASE WHEN qs < 0.5 THEN qs END) AS quoted_spread,
               COUNT(CASE WHEN qs < 0.5 THEN qs END) AS n_quote_days,
               AVG((c - eta) * (c - eta_next)) AS hl_prod_mean,
               COUNT((c - eta) * (c - eta_next)) AS n_hl_days
        FROM e
        GROUP BY permno, month_start
    """


def step_daily_features(db, out, report, first_year):
    from download_wrds import save

    have = columns_or_none(db, "crsp", "dsf_v2")
    if not have or not {"permno", "dlycaldt", "dlyret"} <= have:
        raise RuntimeError("crsp.dsf_v2 (permno, dlycaldt, dlyret) not available")
    missing = [c for c in DAILY_OPTIONAL if c not in have]
    report["missing_columns"]["crsp.dsf_v2"] = missing
    if missing:
        print(f"  crsp.dsf_v2: not available, the measures that need them are skipped: {missing}")
    ffc = columns_or_none(db, "ff", "factors_daily")
    has_ff = bool(ffc) and {"date", "mktrf", "rf"} <= ffc
    if not has_ff:
        print("  no ff.factors_daily: market sums left empty (idiosyncratic vol = total vol)")
    years = list(range(first_year, int(time.strftime("%Y")) + 1))
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    df, level = year_chunks(db, "daily_feat", years,
                            lambda y, lv: daily_features_sql(have, has_ff, y, lv),
                            ["month_start"], chunk_dir, levels=["full", "no_hl", "minimal"],
                            compact=True)
    report["daily_features_query_version"] = level or "already downloaded"
    df = df[df["month_start"].dt.year >= first_year]
    save(df, out / "crsp_daily_features.parquet", report, "crsp_daily_features", "month_start")
    for col in ["s_rr", "s_rm", "s_rr_last5", "s_amihud", "quoted_spread", "hl_prod_mean"]:
        if col in df:
            print(f"  {col}: present on {df[col].notna().mean():.0%} of stock-months")


# ---------------------------------------------------------------- 4. announcements


def announcements_sql(year):
    return f"""
        WITH ann AS (
            SELECT DISTINCT l.lpermno AS permno, CAST(f.rdq AS date) AS rdq
            FROM comp.fundq AS f
            JOIN crsp.ccmxpf_lnkhist AS l
              ON f.gvkey = l.gvkey
             AND l.linktype IN ('LU', 'LC') AND l.linkprim IN ('P', 'C')
             AND f.rdq >= l.linkdt AND (l.linkenddt IS NULL OR f.rdq <= l.linkenddt)
            WHERE f.indfmt = 'INDL' AND f.datafmt = 'STD' AND f.popsrc = 'D' AND f.consol = 'C'
              AND f.rdq >= '{year}-01-01' AND f.rdq < '{year + 1}-01-01'
        )
        SELECT a.permno, a.rdq, d.dlycaldt, d.dlyret
        FROM ann AS a
        JOIN crsp.dsf_v2 AS d
          ON d.permno = a.permno
         AND d.dlycaldt >= a.rdq - 10 AND d.dlycaldt <= a.rdq + 10
        WHERE d.dlycaldt >= DATE '{year}-01-01' - 15 AND d.dlycaldt < DATE '{year + 1}-01-01' + 15
    """


def step_announcements(db, out, report, first_year):
    from download_wrds import save

    years = list(range(first_year, int(time.strftime("%Y")) + 1))
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    df, _ = year_chunks(db, "ann_days", years, lambda y, lv: announcements_sql(y),
                        ["rdq", "dlycaldt"], chunk_dir, compact=True)
    save(df, out / "earnings_announcement_days.parquet", report, "earnings_announcement_days", "rdq")
    if len(df):
        n = df[["permno", "rdq"]].drop_duplicates().shape[0]
        print(f"  {n:,} earnings releases with daily returns")


# ---------------------------------------------------------------- 5. short interest


SHORTINT_TABLES = ["sec_shortint", "sec_shortint_legacy"]  # comp.<table>; the second may not exist


def shortint_sql(table, adj, first_year):
    return f"""
        SELECT s.gvkey, s.iid, s.datadate, s.shortint{adj}, l.lpermno AS permno
        FROM comp.{table} AS s
        JOIN crsp.ccmxpf_lnkhist AS l
          ON s.gvkey = l.gvkey AND s.iid = l.liid
         AND l.linktype IN ('LU', 'LC') AND l.linkprim IN ('P', 'C')
         AND s.datadate >= l.linkdt
         AND (l.linkenddt IS NULL OR s.datadate <= l.linkenddt)
        WHERE s.datadate >= '{first_year}-01-01'
    """


def step_short_interest(db, out, report):
    """comp.sec_shortint, plus comp.sec_shortint_legacy if your account has it (the main
    table starts in July 2006, too late for a training feature)."""
    import pandas as pd
    from download_wrds import save

    link_cols = columns_or_none(db, "crsp", "ccmxpf_lnkhist") or set()
    parts = []
    for table in SHORTINT_TABLES:
        have = columns_or_none(db, "comp", table)
        if not have:
            print(f"  comp.{table}: not available")
            continue
        if not {"gvkey", "iid", "datadate", "shortint"} <= have or "liid" not in link_cols:
            print(f"  comp.{table}: no gvkey / iid / datadate / shortint columns, skipped")
            continue
        adj = ", s.shortintadj" if "shortintadj" in have else ""
        t0 = time.time()
        df = db.raw_sql(shortint_sql(table, adj, SHORTINT_FIRST_YEAR), date_cols=["datadate"])
        df["source"] = table
        print(f"  comp.{table}: {len(df):,} rows in {time.time() - t0:.0f}s"
              + (f", {df['datadate'].min().date()} -> {df['datadate'].max().date()}" if len(df) else ""))
        parts.append(df)
    si = pd.concat(parts, ignore_index=True) if parts else None
    if si is not None:  # the main table wins where both report the same date
        si = si.drop_duplicates(["gvkey", "iid", "datadate", "permno"], keep="first")
    report["short_interest_tables"] = [p["source"].iloc[0] for p in parts if len(p)]
    if si is None or si.empty:
        print("  no short interest: the short-interest features are left out.")
        return
    save(si, out / "short_interest_full.parquet", report, "short_interest_full", "datadate")
    years = si["datadate"].dt.year
    print("  stocks with short interest per year (first / 1990 / 2000 / 2010 / last):",
          ", ".join(f"{y}: {si.loc[years == y, 'permno'].nunique():,}"
                    for y in sorted({years.min(), 1990, 2000, 2010, years.max()}) if (years == y).any()))


# ---------------------------------------------------------------- 6. 13F


def find_13f_table(db):
    for lib in ("tr_13f", "tfn"):
        cols = columns_or_none(db, lib, "s34")
        if cols and {"fdate", "mgrno", "rdate", "cusip", "shares"} <= cols:
            return f"{lib}.s34"
    return None


def holdings_sql(table, year):
    return f"""
        WITH v AS (
            SELECT mgrno, rdate, MIN(fdate) AS fdate
            FROM {table}
            WHERE rdate >= '{year}-01-01' AND rdate < '{year + 1}-01-01'
            GROUP BY mgrno, rdate
        )
        SELECT h.cusip, h.rdate, SUM(h.shares) AS inst_shares, COUNT(DISTINCT h.mgrno) AS n_inst
        FROM {table} AS h
        JOIN v ON h.mgrno = v.mgrno AND h.rdate = v.rdate AND h.fdate = v.fdate
        WHERE h.rdate >= '{year}-01-01' AND h.rdate < '{year + 1}-01-01' AND h.shares > 0
        GROUP BY h.cusip, h.rdate
    """


def managers_sql(table, year):
    return f"""
        SELECT rdate, COUNT(DISTINCT mgrno) AS n_managers
        FROM {table}
        WHERE rdate >= '{year}-01-01' AND rdate < '{year + 1}-01-01'
        GROUP BY rdate
    """


def step_inst_13f(db, out, report, first_year):
    from download_wrds import save

    table = find_13f_table(db)
    if table is None:
        print("  no 13F holdings table (tr_13f.s34 / tfn.s34) in your subscription: skipped; "
              "the institutional-ownership features are left out.")
        report["inst_13f"] = "not available"
        return
    print(f"  using {table}")
    years = list(range(first_year, int(time.strftime("%Y")) + 1))
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    hold, _ = year_chunks(db, "13f_hold", years, lambda y, lv: holdings_sql(table, y), ["rdate"], chunk_dir)
    mgr, _ = year_chunks(db, "13f_mgr", years, lambda y, lv: managers_sql(table, y), ["rdate"], chunk_dir)
    save(hold, out / "inst_13f_holdings.parquet", report, "inst_13f_holdings", "rdate")
    save(mgr, out / "inst_13f_managers.parquet", report, "inst_13f_managers", "rdate")
    report["inst_13f"] = table


# ---------------------------------------------------------------- other libraries


def check_other_libraries(db, report):
    found = {}
    for lib, tables in OTHER_LIBRARIES.items():
        for t in tables:
            found[f"{lib}.{t}"] = columns_or_none(db, lib, t) is not None
    report["readable_tables"] = found
    print("  " + ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in found.items()))


def main():
    parser = argparse.ArgumentParser(description="Download the raw data of the market, daily, "
                                     "short-interest, announcement and 13F features from WRDS.")
    parser.add_argument("--username", required=True, help="your WRDS username")
    parser.add_argument("--out", default="data/raw_wrds")
    parser.add_argument("--only", choices=STEPS, nargs="+", help="run only these steps")
    parser.add_argument("--first-year-daily", type=int, default=1978,
                        help="first year of daily data (features need a year of history before 1980)")
    parser.add_argument("--first-year-13f", type=int, default=1980)
    parser.add_argument("--skip-13f", action="store_true", help="skip the (slow) 13F step")
    args = parser.parse_args()

    try:
        import wrds
    except ImportError:
        sys.exit("The wrds package is missing: run  pip install wrds  first.")

    root = Path.cwd()
    if not (root / "src").exists():
        sys.exit("Run this from the project root (the folder that contains src/).")
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "download_report_features.json"
    report = {"tables": {}, "missing_columns": {}}
    if report_path.exists():
        try:
            report.update(json.loads(report_path.read_text()))
        except ValueError:
            pass
    report["started"] = time.strftime("%Y-%m-%d %H:%M")

    steps = args.only or [s for s in STEPS if not (s == "inst_13f" and args.skip_13f)]
    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=args.username)
    runners = {
        "fundq_extra": lambda: step_fundq_extra(db, out, report),
        "ff_daily": lambda: step_ff_daily(db, out, report),
        "daily_features": lambda: step_daily_features(db, out, report, args.first_year_daily),
        "announcements": lambda: step_announcements(db, out, report, args.first_year_daily + 1),
        "short_interest": lambda: step_short_interest(db, out, report),
        "inst_13f": lambda: step_inst_13f(db, out, report, args.first_year_13f),
    }
    failed = []
    for i, step in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {step}")
        t0 = time.time()
        try:
            runners[step]()
        except Exception as exc:  # one step failing should not lose the others
            rollback(db)
            failed.append(step)
            report.setdefault("errors", {})[step] = str(exc)[:2000]
            print(f"  FAILED: {str(exc).splitlines()[0][:300]}")
        print(f"  ({(time.time() - t0) / 60:.1f} min)")
        report_path.write_text(json.dumps(report, indent=4, default=str))
    print("\nOther libraries your account can read:")
    check_other_libraries(db, report)
    db.close()
    report["finished"] = time.strftime("%Y-%m-%d %H:%M")
    report_path.write_text(json.dumps(report, indent=4, default=str))
    if failed:
        print(f"\nDone, but these steps failed: {failed}. Run again to retry them "
              f"(finished chunks are kept), or send me the error above / {report_path.name}.")
    else:
        print(f"\nDone. Files in {out}. Next: python src/preprocessing/build_cost_inputs.py")


if __name__ == "__main__":
    main()
