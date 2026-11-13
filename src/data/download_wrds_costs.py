"""
Download the trading-cost inputs from WRDS (step 2 of 5).

Run on your own computer, from the project root, after download_wrds.py:
    python src/data/download_wrds_costs.py --username YOUR_WRDS_USERNAME

What it downloads (to data/raw_wrds/, parquet; licensed data, never commit it):

  crsp_daily_costs.parquet   The CRSP daily stock file (crsp.dsf_v2) summarised per
                             stock and month *on the WRDS server*, so only one row per
                             stock-month comes back (about the size of the monthly file):
                               quoted_spread   average daily closing (ask - bid) / mid
                               hl_prod_mean    input of the Abdi-Ranaldo high-low spread
                                               estimate (fallback when quotes are missing)
                               sigma_daily     standard deviation of daily returns
                               adv_mean_usd    average daily dollar volume (price x shares)
                               adv_median_usd  median daily dollar volume
                               vol_sum_shares  shares traded in the month (unit check
                                               against the monthly file)
                               n_days, n_ret_days, n_quote_days, n_hl_days, n_traded_days
  crsp_daily_first_day.parquet  each stock's return on its first trading day of every
                             month (for the one-day trading-delay test of cost_study.py)
  short_interest.parquet     Compustat short interest (comp.sec_shortint, from July 2006),
                             linked to CRSP permnos with the CCM link table on the server.
                             Used for the borrow-fee proxy of the cost model only (the
                             short-interest features use download_wrds_features.py's longer
                             file). Skipped (with a message) if your subscription does not
                             include it; costs then use a size / price rule for
                             hard-to-borrow stocks.
  download_report_costs.json what was downloaded, and any column WRDS did not have.

Only 2003 onwards is downloaded: costs are needed for the validation, test and holdout
months (2005 onwards), plus a year of history for the fallbacks.

To fetch only the first-day returns:
    python src/data/download_wrds_costs.py --username YOUR_WRDS_USERNAME --first-day-only

Like download_wrds.py, the daily file is fetched one year at a time; each year is saved
as soon as it arrives (data/raw_wrds/chunks/costs_daily_<year>.parquet), and running the
command again resumes from the first missing year.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Daily columns we can use. Only permno, dlycaldt and dlyret are required; each optional
# group switches one measure on (the query is built from what the table has).
DAILY_REQUIRED = ["permno", "dlycaldt", "dlyret"]
DAILY_OPTIONAL = ["dlyprc", "dlyvol", "dlyprcvol", "dlybid", "dlyask", "dlyhigh", "dlylow"]

SHORTINT_COLUMNS = ["gvkey", "iid", "datadate", "shortint", "shortintadj"]


def available_columns(db, library, table):
    desc = db.describe_table(library=library, table=table)
    return {c.lower() for c in desc["name"]}


def daily_sql(have, start, end, level="full"):
    """SQL that turns one period of crsp.dsf_v2 into one row per (permno, month).

    level: "full" (all measures), "no_hl" (no high-low spread, no median), "minimal"
    (only what any version of the table supports). Missing columns become NULLs.
    """
    prc = "ABS(dlyprc)"
    if {"dlyprc", "dlyvol"} <= have:
        dv = f"{prc} * dlyvol"
    elif "dlyprcvol" in have:
        dv = "dlyprcvol"
    else:
        dv = "CAST(NULL AS double precision)"
    vol = "dlyvol" if "dlyvol" in have else "CAST(NULL AS double precision)"
    if {"dlybid", "dlyask"} <= have:
        qs = ("CASE WHEN dlybid > 0 AND dlyask > dlybid "
              "THEN (dlyask - dlybid) / ((dlyask + dlybid) / 2.0) END")
    else:
        qs = "CAST(NULL AS double precision)"
    use_hl = level == "full" and {"dlyhigh", "dlylow", "dlyprc"} <= have
    if use_hl:
        c = f"CASE WHEN {prc} > 0 THEN LN({prc}) END"
        eta = ("CASE WHEN dlyhigh > 0 AND dlylow > 0 AND dlyhigh >= dlylow "
               "THEN (LN(dlyhigh) + LN(dlylow)) / 2.0 END")
    else:
        c = eta = "CAST(NULL AS double precision)"
    median = ("PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dv)" if level == "full"
              else "CAST(NULL AS double precision)")
    return f"""
        WITH d AS (
            SELECT permno, dlycaldt,
                   CAST(date_trunc('month', dlycaldt) AS date) AS month_start,
                   dlyret,
                   {dv} AS dv,
                   {vol} AS vol_shares,
                   {qs} AS qs,
                   {c} AS c,
                   {eta} AS eta
            FROM crsp.dsf_v2
            WHERE dlycaldt >= '{start}' AND dlycaldt < '{end}'
        ), e AS (
            SELECT d.*, LEAD(eta) OVER (PARTITION BY permno ORDER BY dlycaldt) AS eta_next
            FROM d
        )
        SELECT permno, month_start,
               COUNT(*) AS n_days,
               COUNT(dlyret) AS n_ret_days,
               STDDEV_SAMP(dlyret) AS sigma_daily,
               AVG(dv) AS adv_mean_usd,
               {median} AS adv_median_usd,
               SUM(CASE WHEN dv > 0 THEN 1 ELSE 0 END) AS n_traded_days,
               SUM(vol_shares) AS vol_sum_shares,
               AVG(CASE WHEN qs < 0.5 THEN qs END) AS quoted_spread,
               COUNT(CASE WHEN qs < 0.5 THEN qs END) AS n_quote_days,
               AVG((c - eta) * (c - eta_next)) AS hl_prod_mean,
               COUNT((c - eta) * (c - eta_next)) AS n_hl_days
        FROM e
        GROUP BY permno, month_start
    """


def download_daily(db, out, report, first_year):
    import pandas as pd

    have = available_columns(db, "crsp", "dsf_v2")
    missing_req = [c for c in DAILY_REQUIRED if c not in have]
    if missing_req:
        sys.exit(f"crsp.dsf_v2 has no {missing_req}: cannot build the cost inputs.")
    missing_opt = [c for c in DAILY_OPTIONAL if c not in have]
    report["missing_columns"]["crsp.dsf_v2"] = missing_opt
    if missing_opt:
        print(f"  crsp.dsf_v2: not available, the measures that need them are skipped: {missing_opt}")

    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    this_year = int(time.strftime("%Y"))
    years = list(range(first_year, this_year + 1))
    level = None  # chosen on the first year that has to be downloaded
    t_all = time.time()
    for i, year in enumerate(years, 1):
        path = chunk_dir / f"costs_daily_{year}.parquet"
        if path.exists() and year < this_year:  # the current year is always refreshed
            print(f"  [{i}/{len(years)}] {year}: already downloaded, skipped")
            continue
        t0 = time.time()
        levels = [level] if level else ["full", "no_hl", "minimal"]
        df = None
        for lv in levels:
            try:
                df = db.raw_sql(daily_sql(have, f"{year}-01-01", f"{year + 1}-01-01", lv),
                                date_cols=["month_start"])
                if level is None and lv != "full":
                    print(f"  note: the full query failed, using the '{lv}' version")
                level = lv
                break
            except Exception as exc:  # e.g. an SQL feature the server rejects
                if lv == levels[-1]:
                    raise
                print(f"  query version '{lv}' failed ({str(exc).splitlines()[0][:120]}), trying a simpler one")
                try:
                    db.connection.rollback()
                except Exception:
                    pass
        df.to_parquet(path, index=False)
        print(f"  [{i}/{len(years)}] {year}: {len(df):,} stock-months in {time.time() - t0:.0f}s "
              f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)
    report["daily_query_version"] = level or "already downloaded"

    parts = [pd.read_parquet(p) for p in sorted(chunk_dir.glob("costs_daily_*.parquet"))]
    parts = [p for p in parts if len(p)]
    full = pd.concat(parts, ignore_index=True)
    full = full[full["month_start"].dt.year >= first_year]
    return full


def first_day_sql(start, end):
    """First trading day of each stock-month and its return (one row per permno-month)."""
    return f"""
        SELECT DISTINCT ON (permno, date_trunc('month', dlycaldt))
               permno,
               CAST(date_trunc('month', dlycaldt) AS date) AS month_start,
               dlycaldt AS first_day,
               dlyret AS ret_day1
        FROM crsp.dsf_v2
        WHERE dlycaldt >= '{start}' AND dlycaldt < '{end}'
        ORDER BY permno, date_trunc('month', dlycaldt), dlycaldt
    """


def download_first_day(db, out, first_year):
    import pandas as pd

    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    this_year = int(time.strftime("%Y"))
    years = list(range(first_year, this_year + 1))
    t_all = time.time()
    for i, year in enumerate(years, 1):
        path = chunk_dir / f"first_day_{year}.parquet"
        if path.exists() and year < this_year:
            print(f"  [{i}/{len(years)}] {year}: already downloaded, skipped")
            continue
        t0 = time.time()
        df = db.raw_sql(first_day_sql(f"{year}-01-01", f"{year + 1}-01-01"),
                        date_cols=["month_start", "first_day"])
        df.to_parquet(path, index=False)
        print(f"  [{i}/{len(years)}] {year}: {len(df):,} stock-months in {time.time() - t0:.0f}s "
              f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)
    parts = [pd.read_parquet(p) for p in sorted(chunk_dir.glob("first_day_*.parquet"))]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def download_short_interest(db, out, report, first_year):
    try:
        have = available_columns(db, "comp", "sec_shortint")
        link_have = available_columns(db, "crsp", "ccmxpf_lnkhist")
    except Exception as exc:
        report["short_interest"] = f"not downloaded: {exc}"
        print(f"  comp.sec_shortint not available ({str(exc).splitlines()[0][:120]}): skipped.")
        print("  Costs will use the size / price rule for hard-to-borrow stocks instead.")
        return None
    need = {"gvkey", "iid", "datadate", "shortint"}
    if not need <= have or "liid" not in link_have:
        report["short_interest"] = f"not downloaded: missing {sorted(need - have)} or ccm liid"
        print("  comp.sec_shortint lacks the needed columns: skipped.")
        return None
    adj = ", s.shortintadj" if "shortintadj" in have else ""
    sql = f"""
        SELECT s.gvkey, s.iid, s.datadate, s.shortint{adj}, l.lpermno AS permno
        FROM comp.sec_shortint AS s
        JOIN crsp.ccmxpf_lnkhist AS l
          ON s.gvkey = l.gvkey AND s.iid = l.liid
         AND l.linktype IN ('LU', 'LC') AND l.linkprim IN ('P', 'C')
         AND s.datadate >= l.linkdt
         AND (l.linkenddt IS NULL OR s.datadate <= l.linkenddt)
        WHERE s.datadate >= '{first_year}-01-01'
    """
    t0 = time.time()
    df = db.raw_sql(sql, date_cols=["datadate"])
    print(f"  {len(df):,} rows in {time.time() - t0:.0f}s")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download the trading-cost inputs from WRDS.")
    parser.add_argument("--username", required=True, help="your WRDS username")
    parser.add_argument("--out", default="data/raw_wrds")
    parser.add_argument("--first-year", type=int, default=2003)
    parser.add_argument("--skip-short-interest", action="store_true")
    parser.add_argument("--first-day-only", action="store_true",
                        help="only the first-trading-day returns (if the rest is already downloaded)")
    args = parser.parse_args()

    try:
        import wrds
    except ImportError:
        sys.exit("The wrds package is missing: run  pip install wrds  first.")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_wrds import save  # same helper as the main download

    root = Path.cwd()
    if not (root / "src").exists():
        sys.exit("Run this from the project root (the folder that contains src/).")
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"tables": {}, "missing_columns": {}, "started": time.strftime("%Y-%m-%d %H:%M")}

    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=args.username)

    if args.first_day_only:
        print(f"\nFirst-trading-day returns (crsp.dsf_v2), {args.first_year} onwards")
        fd = download_first_day(db, out, args.first_year)
        save(fd, out / "crsp_daily_first_day.parquet", report, "crsp_daily_first_day", "month_start")
        db.close()
        print("\nDone. Next: python src/data/download_filing_dates.py --username <you>")
        return

    print(f"\n[1/3] CRSP daily (crsp.dsf_v2) summarised per stock-month on the server, {args.first_year} onwards")
    daily = download_daily(db, out, report, args.first_year)
    save(daily, out / "crsp_daily_costs.parquet", report, "crsp_daily_costs", "month_start")
    for col in ["quoted_spread", "hl_prod_mean", "sigma_daily", "adv_mean_usd"]:
        if col in daily:
            print(f"  {col}: present on {daily[col].notna().mean():.0%} of stock-months")
    del daily

    if args.skip_short_interest:
        print("\n[2/3] Short interest: skipped (--skip-short-interest)")
    else:
        print("\n[2/3] Compustat short interest (comp.sec_shortint), linked to permnos")
        si = download_short_interest(db, out, report, args.first_year)
        if si is not None:
            save(si, out / "short_interest.parquet", report, "short_interest", "datadate")

    print("\n[3/3] First-trading-day returns (for the trading-delay test)")
    fd = download_first_day(db, out, args.first_year)
    save(fd, out / "crsp_daily_first_day.parquet", report, "crsp_daily_first_day", "month_start")

    db.close()
    report["finished"] = time.strftime("%Y-%m-%d %H:%M")
    with open(out / "download_report_costs.json", "w") as f:
        json.dump(report, f, indent=4)
    print("\nDone. Next: python src/data/download_filing_dates.py --username <you>")


if __name__ == "__main__":
    main()
