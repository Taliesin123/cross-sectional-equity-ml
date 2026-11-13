"""
Download the core raw data from WRDS (step 1 of 5; see the README for the others).

Run on your own computer, from the project root:
    pip install wrds
    python src/data/download_wrds.py --username YOUR_WRDS_USERNAME

The first connection asks for your WRDS password (and a Duo approval if WRDS asks
for one) and offers to save it in ~/.pgpass so later runs do not ask again.

What it downloads (to data/raw_wrds/, parquet; nothing here may be committed to GitHub):

  crsp_msf_v2.parquet      CRSP monthly stock file, new CIZ format (crsp.msf_v2), all
                           securities, 1960 onwards: returns (delisting returns are
                           already included in CIZ returns: do NOT add them again),
                           price, shares outstanding, volume, identifiers, SIC / NAICS,
                           and the share-type / exchange flags that replace the old
                           SHRCD / EXCHCD codes.
  ccm_link.parquet         CRSP-Compustat link history (crsp.ccmxpf_lnkhist), links of
                           type LU / LC, primary links P / C, with their validity dates.
  comp_fundq.parquet       Compustat quarterly fundamentals (comp.fundq), standard
                           filters, with the report date rdq, 20 year-to-date income and
                           cash-flow items, their quarterly versions, and the balance sheet.
  ff_factors_monthly.parquet  Fama-French monthly factors and risk-free rate (the market
                           return mktrf + rf, for betas: crsp.msf_v2 has no index column).
  download_report.json     row counts, date ranges, and any requested column that the
                           WRDS table did not have.

Column names in the CIZ tables have changed over time, so the script asks WRDS which
columns exist and downloads the ones on its list that are there; anything missing is
printed and written to the report instead of failing.

CRSP and Compustat are downloaded in 5-year chunks, each saved as soon as it arrives
(data/raw_wrds/chunks/). Progress is printed per chunk, memory stays low, and if the
run is interrupted, running the same command again resumes from the first missing chunk.
"""

import argparse
import json
import sys
import time
from pathlib import Path

START_DATE = "1960-01-01"

# --- what we ask for --------------------------------------------------------------

CRSP_COLUMNS = [
    # identifiers and dates
    "permno", "permco", "mthcaldt", "hdrcusip", "cusip", "ticker",
    # returns, price, size, liquidity
    "mthret", "mthretx", "mthprc", "mthcap", "shrout", "mthvol",
    # industry
    "siccd",
    # replacements for SHRCD in (10, 11) and EXCHCD in (1, 2, 3)
    "sharetype", "securitytype", "securitysubtype", "usincflg", "issuertype",
    "primaryexch", "conditionaltype", "tradingstatusflg",
]

CCM_QUERY = """
    SELECT gvkey, lpermno AS permno, lpermco AS permco, linktype, linkprim,
           linkdt, linkenddt
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
"""

# Year-to-date income-statement and cash-flow items (turned into quarterly flows).
FUNDQ_YTD = [
    "saley", "revty", "cogsy", "niy", "niity", "oibdpy", "oiadpy", "xsgay", "xrdy",
    "capxy", "xinty", "dpcy", "oancfy", "ivncfy", "fincfy", "dvy", "chechy",
    "wcapchy", "recchy", "invchy",
]
FUNDQ_IDS = [
    "gvkey", "datadate", "rdq", "fyearq", "fqtr", "fyr", "cusip", "tic", "conm",
    "curcdq", "indfmt", "consol", "popsrc", "datafmt", "datacqtr", "datafqtr",
]
# Quarterly income items and the balance sheet.
FUNDQ_QUARTERLY = [
    "saleq", "cogsq", "xsgaq", "oiadpq", "oibdpq", "niq", "ibq", "dpq", "xintq", "txtq",
    "xrdq", "atq", "ceqq", "seqq", "ltq", "lctq", "actq", "cheq", "invtq", "rectq",
    "ppentq", "dlttq", "dlcq", "txditcq", "pstkq", "cshoq", "prccq", "cshprq",
]

FF_TABLES = {
    "factors_monthly": ["date", "mktrf", "smb", "hml", "rf", "umd"],
}


# --- helpers ----------------------------------------------------------------------


def available_columns(db, library, table):
    desc = db.describe_table(library=library, table=table)
    return [c.lower() for c in desc["name"]]


def pick(db, library, table, wanted, report):
    have = set(available_columns(db, library, table))
    keep = [c for c in wanted if c in have]
    missing = [c for c in wanted if c not in have]
    report["missing_columns"][f"{library}.{table}"] = missing
    if missing:
        print(f"  {library}.{table}: not available, skipped: {missing}")
    return keep


def save(df, path, report, name, date_col=None):
    df.to_parquet(path, index=False)
    info = {"rows": int(len(df)), "columns": list(df.columns), "file": path.name}
    if date_col and date_col in df.columns and len(df):
        info["first_date"] = str(df[date_col].min())[:10]
        info["last_date"] = str(df[date_col].max())[:10]
    report["tables"][name] = info
    print(f"  saved {path.name}: {len(df):,} rows"
          + (f", {info.get('first_date')} -> {info.get('last_date')}" if "first_date" in info else ""))


def download_in_chunks(db, name, select_sql, date_col, out, report, years=5, first_year=1960):
    """Run select_sql once per `years`-year window of date_col, saving each window to
    out/chunks/<name>_<start>.parquet; existing chunk files are skipped (resume)."""
    import pandas as pd

    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    last_year = int(time.strftime("%Y"))
    starts = list(range(first_year, last_year + 1, years))
    t_all = time.time()
    for i, y0 in enumerate(starts, 1):
        path = chunk_dir / f"{name}_{y0}.parquet"
        if path.exists():
            print(f"  [{i}/{len(starts)}] {y0}-{y0 + years - 1}: already downloaded, skipped")
            continue
        t0 = time.time()
        sql = (f"{select_sql} AND {date_col} >= '{y0}-01-01' "
               f"AND {date_col} < '{y0 + years}-01-01'")
        df = db.raw_sql(sql, date_cols=[date_col])
        df.to_parquet(path, index=False)
        print(f"  [{i}/{len(starts)}] {y0}-{y0 + years - 1}: {len(df):,} rows "
              f"in {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.1f} min)", flush=True)
        del df
    parts = [pd.read_parquet(p) for p in sorted(chunk_dir.glob(f"{name}_*.parquet"))]
    full = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return full


# --- main -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Download the core raw data from WRDS.")
    parser.add_argument("--username", required=True, help="your WRDS username")
    parser.add_argument("--out", default="data/raw_wrds", help="output folder (relative to the project root)")
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
    report = {"tables": {}, "missing_columns": {}, "started": time.strftime("%Y-%m-%d %H:%M")}

    print("Connecting to WRDS (password, and possibly a Duo approval, will be asked)...")
    db = wrds.Connection(wrds_username=args.username)
    try:
        db.create_pgpass_file()
    except Exception:
        pass  # already exists or not writable: not needed for this run

    # 1. CRSP monthly, CIZ
    print("\n[1/4] CRSP monthly stock file (crsp.msf_v2), in 5-year chunks")
    cols = pick(db, "crsp", "msf_v2", CRSP_COLUMNS, report)
    crsp = download_in_chunks(
        db, "crsp", f"SELECT {', '.join(cols)} FROM crsp.msf_v2 WHERE TRUE",
        "mthcaldt", out, report,
    )
    save(crsp, out / "crsp_msf_v2.parquet", report, "crsp_msf_v2", "mthcaldt")
    del crsp

    # 2. CCM link
    print("\n[2/4] CRSP-Compustat link history (crsp.ccmxpf_lnkhist)")
    link = db.raw_sql(CCM_QUERY, date_cols=["linkdt", "linkenddt"])
    save(link, out / "ccm_link.parquet", report, "ccm_link", "linkdt")

    # 3. Compustat quarterly
    print("\n[3/4] Compustat quarterly fundamentals (comp.fundq), in 5-year chunks")
    cols = pick(db, "comp", "fundq", FUNDQ_IDS + FUNDQ_YTD + FUNDQ_QUARTERLY, report)
    fundq = download_in_chunks(
        db, "fundq",
        f"""SELECT {', '.join(cols)} FROM comp.fundq
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'""",
        "datadate", out, report,
    )
    if "rdq" in fundq:
        import pandas as pd
        fundq["rdq"] = pd.to_datetime(fundq["rdq"], errors="coerce")
    rdq_share = float(fundq["rdq"].notna().mean()) if "rdq" in fundq else 0.0
    report["rdq_available_share"] = rdq_share
    print(f"  rdq (report date) present on {rdq_share:.0%} of filings")
    save(fundq, out / "comp_fundq.parquet", report, "comp_fundq", "datadate")
    del fundq

    # 4. Fama-French
    print("\n[4/4] Fama-French factors (ff library)")
    for table, wanted in FF_TABLES.items():
        try:
            cols = pick(db, "ff", table, wanted, report)
            ff = db.raw_sql(f"SELECT {', '.join(cols)} FROM ff.{table}", date_cols=["date"])
            save(ff, out / f"ff_{table}.parquet", report, f"ff_{table}", "date")
        except Exception as exc:  # the subscription may not include every FF table
            report["missing_columns"][f"ff.{table}"] = f"not downloaded: {exc}"
            print(f"  ff.{table}: not downloaded ({exc})")

    db.close()
    report["finished"] = time.strftime("%Y-%m-%d %H:%M")
    with open(out / "download_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nDone. Files in {out}. Next: python src/data/download_wrds_costs.py --username <you>")


if __name__ == "__main__":
    main()
