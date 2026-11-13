"""
Download the SEC filing date of every 10-Q / 10-K (step 3 of 5), for the report timing
(income-statement features from the earnings release, cash-flow and balance-sheet features
from the filing).

Run on your own computer, from the project root:
    python src/data/download_filing_dates.py --username YOUR_WRDS_USERNAME

Output (licensed data, never commit it):
    data/raw_wrds/filing_dates.parquet
        gvkey, period (period of report when the source has it), filedate,
        kind (Q = 10-Q, K = 10-K), form, source

Sources, tried in this order (column names are looked up, not assumed):
    1. comp.co_filedate                     Compustat's filing dates, by gvkey and datadate
    2. wrdssec.wrds_forms (or wrdssec.forms) the WRDS SEC filings index, linked to gvkey with
       + wrdssec.wciklink_gvkey             the WRDS CIK-GVKEY link table
If your subscription has neither, the script says so; the build then times cash-flow
features with the filing deadlines (10-Q: quarter end + 50 days, 10-K: + 105 days).
Amendments (10-Q/A, 10-K/A) are left out: the first filing is what the market saw.
"""

import argparse
import re
import sys
import time
from pathlib import Path

FORMS_Q = ["10-Q", "10-Q405", "10-QSB", "10QSB", "10-QT", "10-QSB40"]
FORMS_K = ["10-K", "10-K405", "10-KSB", "10KSB", "10-KSB40", "10-KT", "10-K405T", "10-KT405"]


def columns(db, library, table):
    desc = db.describe_table(library=library, table=table)
    return [c.lower() for c in desc["name"]]


def first_of(cols, candidates):
    return next((c for c in candidates if c in cols), None)


def kind_of(form):
    """Q / K / None from a form or source-type code (10-Q, 10Q, 10-K405, 10KSB ...)."""
    f = re.sub(r"[\s-]", "", str(form).upper())
    if "/A" in f or f in ("10QA", "10KA"):
        return None
    if f.startswith("10Q"):
        return "Q"
    if f.startswith("10K"):
        return "K"
    return None


def rollback(db):
    try:
        db.connection.rollback()
    except Exception:
        pass


def from_compustat(db, first_year):
    cols = columns(db, "comp", "co_filedate")
    date_col = first_of(cols, ["filedate", "fdate", "filingdate", "file_date"])
    type_col = first_of(cols, ["srctype", "form", "formtype", "src"])
    if not {"gvkey", "datadate"} <= set(cols) or date_col is None or type_col is None:
        raise ValueError(f"comp.co_filedate has columns {cols}: no gvkey / datadate / filing date / type")
    print(f"  comp.co_filedate: using {date_col} and {type_col}")
    df = db.raw_sql(f"""
        SELECT gvkey, datadate AS period, {date_col} AS filedate, {type_col} AS form
        FROM comp.co_filedate
        WHERE datadate >= '{first_year}-01-01'
    """, date_cols=["period", "filedate"])
    df["source"] = "comp.co_filedate"
    return df


def from_wrdssec(db, first_year):
    forms_table = None
    for t in ("wrds_forms", "forms"):
        try:
            fcols = columns(db, "wrdssec", t)
            forms_table = t
            break
        except Exception:
            rollback(db)
    if forms_table is None:
        raise ValueError("no wrdssec.wrds_forms / wrdssec.forms table")
    lcols = columns(db, "wrdssec", "wciklink_gvkey")
    date_col = first_of(fcols, ["fdate", "filedate", "date_filed", "filingdate"])
    period_col = first_of(fcols, ["secpdate", "period", "pdate", "rdate", "periodofreport", "conformed_period"])
    if "cik" not in fcols or "form" not in fcols or date_col is None:
        raise ValueError(f"wrdssec.{forms_table} has columns {fcols}: no cik / form / filing date")
    if not {"gvkey", "cik"} <= set(lcols):
        raise ValueError(f"wrdssec.wciklink_gvkey has columns {lcols}: no gvkey / cik")
    print(f"  wrdssec.{forms_table}: using {date_col}" + (f", period of report {period_col}" if period_col else ", no period column"))
    forms = ", ".join(f"'{f}'" for f in FORMS_Q + FORMS_K)
    period_sql = f"f.{period_col} AS period" if period_col else "CAST(NULL AS date) AS period"
    df = db.raw_sql(f"""
        SELECT DISTINCT l.gvkey, f.form, f.{date_col} AS filedate, {period_sql}
        FROM wrdssec.{forms_table} AS f
        JOIN wrdssec.wciklink_gvkey AS l
          ON LTRIM(CAST(f.cik AS text), '0') = LTRIM(CAST(l.cik AS text), '0')
        WHERE f.form IN ({forms}) AND f.{date_col} >= '{first_year}-01-01'
    """, date_cols=["filedate", "period"])
    df["source"] = f"wrdssec.{forms_table}"
    return df


def main():
    parser = argparse.ArgumentParser(description="Download SEC filing dates of 10-Q / 10-K reports.")
    parser.add_argument("--username", required=True, help="your WRDS username")
    parser.add_argument("--out", default="data/raw_wrds")
    parser.add_argument("--first-year", type=int, default=1980)
    args = parser.parse_args()

    try:
        import wrds
    except ImportError:
        sys.exit("The wrds package is missing: run  pip install wrds  first.")
    import pandas as pd

    root = Path.cwd()
    if not (root / "src").exists():
        sys.exit("Run this from the project root (the folder that contains src/).")
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=args.username)
    df = None
    for name, fetch in (("Compustat filing dates (comp.co_filedate)", from_compustat),
                        ("WRDS SEC filings index (wrdssec)", from_wrdssec)):
        print(f"\nTrying {name}")
        t0 = time.time()
        try:
            df = fetch(db, args.first_year)
            print(f"  {len(df):,} rows in {time.time() - t0:.0f}s")
            break
        except Exception as exc:
            rollback(db)
            print(f"  not available: {str(exc).splitlines()[0][:160]}")
    db.close()
    if df is None or df.empty:
        sys.exit("\nNo filing-date source in your subscription. The build will use the filing deadlines "
                 "for cash-flow features (nothing else to do).")

    df["kind"] = df["form"].map(kind_of)
    df = df.dropna(subset=["gvkey", "filedate", "kind"])
    df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
    df["filedate"] = pd.to_datetime(df["filedate"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df = df.dropna(subset=["filedate"]).drop_duplicates(["gvkey", "period", "filedate", "kind"])
    df = df[["gvkey", "period", "filedate", "kind", "form", "source"]].sort_values(["gvkey", "filedate"])
    path = out / "filing_dates.parquet"
    df.to_parquet(path, index=False)

    years = df["filedate"].dt.year
    print(f"\nSaved {len(df):,} filings to {path}")
    print(f"  {df['gvkey'].nunique():,} companies, {years.min()}-{years.max()}; "
          f"10-Q {int((df['kind'] == 'Q').sum()):,}, 10-K {int((df['kind'] == 'K').sum()):,}; "
          f"period of report on {df['period'].notna().mean():.0%} of rows")
    print(df.groupby(years.floordiv(5).mul(5))["gvkey"].size().rename("filings per 5 years").to_string())
    print("\nNext: python src/data/download_wrds_features.py --username <you>")


if __name__ == "__main__":
    main()
