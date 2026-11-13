"""
Download the data added in v5 (step 5 of 5; run it after the four downloads of v4).

Run on your own computer, from the project root:
    python src/data/download_wrds_extra.py --username YOUR_WRDS_USERNAME

Steps, in this order (run one step only with  --only <step>):
  probe          what your WRDS account can read, written to the report:
                   - securities-lending data (Markit): real borrow fees and availability;
                   - Compustat Snapshot / point-in-time fundamentals (values as first
                     published, before restatements);
                   - the IBES tables the analyst features use.
                 Nothing is downloaded by this step.
  ff5            Fama-French five factors + momentum (ff.fivefactors_monthly), for the
                 factor regressions of src/evaluation/factor_alpha.py.
  daily_returns  daily returns (crsp.dsf_v2) of every stock that is a US common stock in
                 some month since 2003, for the daily volatility forecast of the
                 volatility target. About 25M rows; saved per year (resumable).
  ibes           IBES (Refinitiv) analyst data, US firms, EPS:
                   - the IBES ticker -> CRSP permno link (wrdsapps.ibcrsphist);
                   - monthly consensus summaries (ibes.statsum_epsus): long-term
                     growth, fiscal years 1 and 2, next quarter; number of estimates,
                     up / down revisions, mean / median / dispersion, the actual and its
                     announcement date;
                   - the same summaries unadjusted for splits (ibes.statsumu_epsus), to
                     put forecasts on the same share basis as the CRSP price;
                   - quarterly actuals with announcement dates (ibes.act_epsus);
                   - the recommendation summaries (ibes.recdsum), if readable.
                 Saved per year (resumable).
  macro          credit and term spreads from FRED (public, no WRDS needed): Moody's
                 BAA and AAA yields, 10-year Treasury, 3-month T-bill (monthly).

Outputs in data/raw_wrds/ (licensed data, never commit them):
  ff_fivefactors_monthly.parquet, crsp_daily_returns.parquet, ibes_link.parquet,
  ibes_statsum.parquet, ibes_statsum_unadj.parquet, ibes_actuals.parquet,
  ibes_recommendations.parquet, macro_fred.parquet, download_report_extra.json

A step that fails (a table your account cannot read, for example) is reported and the
others still run. Column names are looked up, not assumed.

The connection survives drops: a lost link (Wi-Fi, a server restart, the computer going
to sleep) is detected within about a minute, and the query is retried on a new
connection. Keep the computer awake while it runs:
    caffeinate -i python src/data/download_wrds_extra.py --username YOUR_WRDS_USERNAME
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STEPS = ["probe", "ff5", "daily_returns", "ibes", "macro"]
DAILY_FIRST_YEAR = 2003          # the backtests start in 2005; vol forecasts look back 6 months
IBES_FIRST_YEAR = 1976
RECS_FIRST_YEAR = 1993           # IBES recommendations start in late 1993
FRED_SERIES = {"baa": "BAA", "aaa": "AAA", "gs10": "GS10", "tb3ms": "TB3MS"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# Library-name fragments that identify the optional data sets in the probe. Libraries
# whose name contains "samp" are WRDS's free samples (a few firms or one year), not a
# subscription.
LENDING_HINTS = ("markit", "msf", "seclend", "lending", "dataexplorers")
SNAPSHOT_HINTS = ("snap", "pit", "compsamp", "comp_point")
SAMPLE_HINT = "samp"

# Connection: TCP keepalives detect a dead link in about a minute (instead of hours), a
# query may run at most QUERY_TIMEOUT on the server, and a lost connection is retried.
KEEPALIVES = {"keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 6}
QUERY_TIMEOUT = "60min"
RETRIES = 4
IBES_TABLES = ["statsum_epsus", "statsumu_epsus", "act_epsus", "actu_epsus", "recdsum", "det_epsus"]

STATSUM_WANTED = ["ticker", "cusip", "statpers", "measure", "fpi", "fpedats", "numest", "numup",
                  "numdown", "meanest", "medest", "stdev", "highest", "lowest", "usfirm",
                  "actual", "anndats_act"]
STATSUMU_WANTED = ["ticker", "statpers", "measure", "fpi", "fpedats", "meanest", "medest",
                   "highest", "lowest", "usfirm"]
ACT_WANTED = ["ticker", "measure", "pdicity", "pends", "anndats", "value", "usfirm"]
RECS_WANTED = ["ticker", "statpers", "meanrec", "medrec", "numrec", "numup", "numdown",
               "buypct", "holdpct", "sellpct", "usfirm"]
LINK_WANTED = ["ticker", "permno", "ncusip", "sdate", "edate", "score"]


class Wrds:
    """wrds.Connection that reconnects and retries a query when the connection drops."""

    RETRIED = ("raw_sql", "describe_table", "list_libraries", "list_tables")

    def __init__(self, username):
        self.username = username
        self.db = None
        self.connect()

    def connect(self):
        import wrds
        args = {"sslmode": "require", "application_name": "cross_sectional_ml", **KEEPALIVES}
        self.db = wrds.Connection(wrds_username=self.username, wrds_connect_args=args)
        try:   # a server-side limit per query; ignored if the server does not allow it
            import sqlalchemy as sa
            self.db.connection.execute(sa.text(f"SET statement_timeout = '{QUERY_TIMEOUT}'"))
        except Exception:
            rollback(self.db)

    def reconnect(self):
        try:
            self.db.close()
        except Exception:
            pass
        self.connect()

    def __getattr__(self, name):
        attr = getattr(self.db, name)
        if name not in self.RETRIED:
            return attr

        def call(*args, **kwargs):
            for attempt in range(1, RETRIES + 1):
                try:
                    return getattr(self.db, name)(*args, **kwargs)
                except Exception as exc:
                    if not connection_lost(exc) or attempt == RETRIES:
                        raise
                    wait = 30 * attempt
                    print(f"\n  connection lost ({str(exc).splitlines()[0][:120]}); reconnecting in {wait}s "
                          f"(attempt {attempt + 1} of {RETRIES})", flush=True)
                    time.sleep(wait)
                    try:
                        self.reconnect()
                    except Exception as exc2:
                        print(f"  reconnection failed: {str(exc2).splitlines()[0][:120]}", flush=True)
        return call


def connection_lost(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("operationalerror", "server closed the connection", "timed out",
                                   "connection reset", "ssl syscall", "could not receive data",
                                   "connection not open", "terminating connection", "broken pipe",
                                   "closed the connection", "resourceclosederror"))


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


def year_chunks(db, name, years, make_sql, date_cols, chunk_dir):
    """make_sql(year) for each year, saved as chunks/<name>_<year>.parquet (resumable;
    the current year is always fetched again)."""
    import pandas as pd

    this_year = int(time.strftime("%Y"))
    t_all = time.time()
    for i, year in enumerate(years, 1):
        path = chunk_dir / f"{name}_{year}.parquet"
        if path.exists() and year < this_year:
            continue
        t0 = time.time()
        print(f"  [{i}/{len(years)}] {year} ...", end="", flush=True)
        df = db.raw_sql(make_sql(year), date_cols=date_cols)
        for c in df.columns:
            if df[c].dtype == "float64" and c not in ("permno",):
                df[c] = df[c].astype("float32")
        df.to_parquet(path, index=False)
        print(f" {len(df):,} rows in {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.1f} min)",
              flush=True)
    parts = [pd.read_parquet(p) for p in sorted(chunk_dir.glob(f"{name}_*.parquet"))]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def sql_list(values):
    return ", ".join(f"'{v}'" for v in values)


# ---------------------------------------------------------------- 1. probe


def list_libraries(db):
    try:
        return sorted(str(x).lower() for x in db.list_libraries())
    except Exception:
        rollback(db)
        return []


def list_tables(db, library):
    try:
        return sorted(str(x).lower() for x in db.list_tables(library=library))
    except Exception:
        rollback(db)
        return []


def step_probe(db, report):
    libs = list_libraries(db)
    report["probe"] = {"n_libraries": len(libs)}

    def describe(hints, key, what):
        found = [lib for lib in libs if any(h in lib for h in hints)]
        samples = [lib for lib in found if SAMPLE_HINT in lib]
        info = {}
        for lib in found:
            if lib in samples:
                continue
            tables = list_tables(db, lib)
            info[lib] = {t: sorted(columns_or_none(db, lib, t) or []) for t in tables[:40]}
        report["probe"][key] = info
        report["probe"][key + "_samples_only"] = samples
        if info:
            print(f"  {what}: FOUND in {', '.join(info)} ({sum(len(v) for v in info.values())} tables). "
                  f"The table and column names are in the report: send it to me to wire them in.")
        else:
            print(f"  {what}: not in your subscription"
                  + (f" (only WRDS's free samples: {', '.join(samples)})." if samples else "."))
        return bool(info)

    has_lending = describe(LENDING_HINTS, "securities_lending", "securities-lending data (Markit)")
    has_pit = describe(SNAPSHOT_HINTS, "point_in_time_fundamentals", "Compustat Snapshot / point-in-time")
    readable = {t: columns_or_none(db, "ibes", t) is not None for t in IBES_TABLES}
    readable["wrdsapps.ibcrsphist"] = columns_or_none(db, "wrdsapps", "ibcrsphist") is not None
    report["probe"]["ibes_tables"] = readable
    print("  IBES: " + ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in readable.items()))
    report["probe"]["summary"] = {
        "securities_lending": has_lending,
        "point_in_time_fundamentals": has_pit,
        "borrow_cost_model": "Markit (to wire in)" if has_lending else
                             "proxy: institutional ownership + short interest (build_cost_inputs.py)",
        "accounting_values": "point-in-time available (to wire in)" if has_pit else
                             "Compustat as of today (restated); IBES actuals are as first announced",
    }


# ---------------------------------------------------------------- 2. FF5


def step_ff5(db, out, report):
    from download_wrds import save

    have = columns_or_none(db, "ff", "fivefactors_monthly")
    if not have or not {"date", "mktrf", "smb", "hml", "rf"} <= have:
        raise RuntimeError("ff.fivefactors_monthly is not readable")
    keep = [c for c in ["date", "mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"] if c in have]
    df = db.raw_sql(f"SELECT {', '.join(keep)} FROM ff.fivefactors_monthly", date_cols=["date"])
    save(df, out / "ff_fivefactors_monthly.parquet", report, "ff_fivefactors_monthly", "date")
    if "umd" not in df:
        print("  no momentum column (umd): factor_alpha.py builds momentum from CRSP instead")


# ---------------------------------------------------------------- 3. daily returns


def common_stock_permnos(out, first_year):
    """Permnos that pass the common-stock filter in some month since first_year, from the
    monthly file already downloaded (keeps the daily download to the stocks we can hold)."""
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
    from build_processed_data import CRSP_COMMON_STOCK

    path = out / "crsp_msf_v2.parquet"
    if not path.exists():
        raise RuntimeError(f"{path} not found: run src/data/download_wrds.py first")
    m = pd.read_parquet(path, columns=["permno", "mthcaldt"] + list(CRSP_COMMON_STOCK))
    m = m[pd.to_datetime(m["mthcaldt"]) >= f"{first_year}-01-01"]
    ok = pd.Series(True, index=m.index)
    for col, allowed in CRSP_COMMON_STOCK.items():
        ok &= m[col].astype("string").str.strip().str.upper().isin(allowed).fillna(False)
    return sorted(pd.to_numeric(m.loc[ok, "permno"], errors="coerce").dropna().astype(int).unique())


def step_daily_returns(db, out, report, first_year):
    from download_wrds import save

    have = columns_or_none(db, "crsp", "dsf_v2")
    if not have or not {"permno", "dlycaldt", "dlyret"} <= have:
        raise RuntimeError("crsp.dsf_v2 (permno, dlycaldt, dlyret) not readable")
    permnos = common_stock_permnos(out, first_year)
    print(f"  {len(permnos):,} stocks")
    plist = ", ".join(str(p) for p in permnos)
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    years = list(range(first_year, int(time.strftime("%Y")) + 1))
    df = year_chunks(db, "daily_ret", years, lambda y: f"""
        SELECT permno, dlycaldt, dlyret FROM crsp.dsf_v2
        WHERE dlycaldt >= '{y}-01-01' AND dlycaldt < '{y + 1}-01-01'
          AND dlyret IS NOT NULL AND permno IN ({plist})""", ["dlycaldt"], chunk_dir)
    if df.empty:
        raise RuntimeError("crsp.dsf_v2 returned no rows for these stocks")
    df["permno"] = df["permno"].astype("int32")
    save(df, out / "crsp_daily_returns.parquet", report, "crsp_daily_returns", "dlycaldt")


# ---------------------------------------------------------------- 4. IBES


def ibes_filter(have, fpis):
    parts = []
    if "measure" in have:
        parts.append("measure = 'EPS'")
    if "usfirm" in have:
        parts.append("usfirm = 1")
    if fpis is not None and "fpi" in have:
        parts.append(f"fpi IN ({sql_list(fpis)})")
    return " AND ".join(parts) or "TRUE"


def step_ibes(db, out, report):
    from download_wrds import save

    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    years = list(range(IBES_FIRST_YEAR, int(time.strftime("%Y")) + 1))
    done = {}

    # link IBES ticker -> CRSP permno
    have = columns_or_none(db, "wrdsapps", "ibcrsphist")
    if not have or not {"ticker", "permno", "sdate", "edate"} <= have:
        raise RuntimeError("wrdsapps.ibcrsphist (the IBES-CRSP link) is not readable")
    cols = [c for c in LINK_WANTED if c in have]
    link = db.raw_sql(f"SELECT {', '.join(cols)} FROM wrdsapps.ibcrsphist WHERE permno IS NOT NULL",
                      date_cols=["sdate", "edate"])
    save(link, out / "ibes_link.parquet", report, "ibes_link", "sdate")

    # consensus summaries, split-adjusted (all features) and unadjusted (share basis)
    for table, wanted, fpis, name, file in (
            ("statsum_epsus", STATSUM_WANTED, ["0", "1", "2", "6"], "ibes_sum", "ibes_statsum.parquet"),
            ("statsumu_epsus", STATSUMU_WANTED, ["1", "6"], "ibes_sumu", "ibes_statsum_unadj.parquet")):
        have = columns_or_none(db, "ibes", table)
        if not have or not {"ticker", "statpers", "meanest"} <= have:
            print(f"  ibes.{table}: not readable, skipped")
            report.setdefault("ibes_missing", []).append(table)
            continue
        cols = [c for c in wanted if c in have]
        where = ibes_filter(have, fpis)
        print(f"  ibes.{table}: {len(cols)} columns, per year from {IBES_FIRST_YEAR}")
        df = year_chunks(db, name, years, lambda y: f"""
            SELECT {', '.join(cols)} FROM ibes.{table}
            WHERE {where} AND statpers >= '{y}-01-01' AND statpers < '{y + 1}-01-01'""",
                         [c for c in ("statpers", "fpedats", "anndats_act") if c in cols], chunk_dir)
        save(df, out / file, report, name, "statpers")
        done[table] = len(df)

    # quarterly actuals with announcement dates (as first announced)
    have = columns_or_none(db, "ibes", "act_epsus")
    if have and {"ticker", "pends", "anndats", "value"} <= have:
        cols = [c for c in ACT_WANTED if c in have]
        where = ibes_filter(have, None) + (" AND pdicity = 'QTR'" if "pdicity" in have else "")
        print(f"  ibes.act_epsus: per year from {IBES_FIRST_YEAR}")
        df = year_chunks(db, "ibes_act", years, lambda y: f"""
            SELECT {', '.join(cols)} FROM ibes.act_epsus WHERE {where}
            AND anndats >= '{y}-01-01' AND anndats < '{y + 1}-01-01'""", ["pends", "anndats"], chunk_dir)
        save(df, out / "ibes_actuals.parquet", report, "ibes_actuals", "anndats")
        done["act_epsus"] = len(df)
    else:
        print("  ibes.act_epsus: not readable, skipped (the summaries' actuals are used if present)")

    # recommendations
    have = columns_or_none(db, "ibes", "recdsum")
    if have and {"ticker", "statpers", "meanrec"} <= have:
        cols = [c for c in RECS_WANTED if c in have]
        where = "usfirm = 1" if "usfirm" in have else "TRUE"
        print(f"  ibes.recdsum: per year from {RECS_FIRST_YEAR}")
        df = year_chunks(db, "ibes_rec", [y for y in years if y >= RECS_FIRST_YEAR], lambda y: f"""
            SELECT {', '.join(cols)} FROM ibes.recdsum WHERE {where}
            AND statpers >= '{y}-01-01' AND statpers < '{y + 1}-01-01'""", ["statpers"], chunk_dir)
        save(df, out / "ibes_recommendations.parquet", report, "ibes_recommendations", "statpers")
        done["recdsum"] = len(df)
    else:
        print("  ibes.recdsum: not readable, the recommendation features are left out")
    report["ibes_rows"] = done


# ---------------------------------------------------------------- 5. macro (FRED)


def step_macro(out, report):
    import io
    import urllib.request

    import pandas as pd
    from download_wrds import save

    frames = []
    for col, series in FRED_SERIES.items():
        with urllib.request.urlopen(FRED_URL.format(series), timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        f = pd.read_csv(io.StringIO(raw))
        date_col = [c for c in f.columns if "date" in c.lower()][0]
        f = pd.DataFrame({"date": pd.to_datetime(f[date_col]),
                          col: pd.to_numeric(f[series], errors="coerce")}).set_index("date")
        frames.append(f)
        print(f"  {series}: {f[col].notna().sum()} months, {f.index.min():%Y-%m} -> {f.index.max():%Y-%m}")
    df = pd.concat(frames, axis=1).reset_index()
    save(df, out / "macro_fred.parquet", report, "macro_fred", "date")


def main():
    parser = argparse.ArgumentParser(description="Download the v5 data: probe, FF5, daily returns, "
                                     "IBES and macro series.")
    parser.add_argument("--username", required=True, help="your WRDS username")
    parser.add_argument("--out", default="data/raw_wrds")
    parser.add_argument("--only", choices=STEPS, nargs="+", help="run only these steps")
    parser.add_argument("--first-year-daily", type=int, default=DAILY_FIRST_YEAR)
    args = parser.parse_args()

    root = Path.cwd()
    if not (root / "src").exists():
        sys.exit("Run this from the project root (the folder that contains src/).")
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "download_report_extra.json"
    report = {"tables": {}, "missing_columns": {}}
    if report_path.exists():
        try:
            report.update(json.loads(report_path.read_text()))
        except ValueError:
            pass
    report["started"] = time.strftime("%Y-%m-%d %H:%M")
    steps = args.only or STEPS

    db = None
    if any(s != "macro" for s in steps):
        try:
            import wrds  # noqa: F401
        except ImportError:
            sys.exit("The wrds package is missing: run  pip install wrds  first.")
        print("Connecting to WRDS...")
        db = Wrds(args.username)
    runners = {
        "probe": lambda: step_probe(db, report),
        "ff5": lambda: step_ff5(db, out, report),
        "daily_returns": lambda: step_daily_returns(db, out, report, args.first_year_daily),
        "ibes": lambda: step_ibes(db, out, report),
        "macro": lambda: step_macro(out, report),
    }
    failed = []
    for i, step in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {step}")
        t0 = time.time()
        try:
            runners[step]()
            report.get("errors", {}).pop(step, None)   # an error from an earlier run is resolved
        except Exception as exc:  # one step failing should not lose the others
            if db is not None:
                rollback(db)
            failed.append(step)
            report.setdefault("errors", {})[step] = str(exc)[:2000]
            print(f"  FAILED: {str(exc).splitlines()[0][:300]}")
        print(f"  ({(time.time() - t0) / 60:.1f} min)")
        report_path.write_text(json.dumps(report, indent=4, default=str))
    if db is not None:
        db.close()
    report["finished"] = time.strftime("%Y-%m-%d %H:%M")
    report_path.write_text(json.dumps(report, indent=4, default=str))
    if failed:
        print(f"\nDone, but these steps failed: {failed}. Run again to retry them (finished "
              f"chunks are kept), or send me the error above / {report_path.name}.")
    else:
        print(f"\nDone. Files in {out}. Next: python src/preprocessing/build_cost_inputs.py")


if __name__ == "__main__":
    main()
