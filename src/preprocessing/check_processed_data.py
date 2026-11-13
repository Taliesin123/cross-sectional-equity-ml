"""
Checks on the model panel built by build_processed_data.py.

Run from the project root:
    python src/preprocessing/check_processed_data.py

Fails (AssertionError) if any of these does not hold:
    - no feature is used before it was public: income-statement items before the earnings
      release / filing, cash-flow and balance-sheet items before the 10-Q / 10-K filing,
      short interest before settlement + 14 days, the announcement return before the
      close of day +1, 13F holdings before quarter end + 45 days, an IBES consensus before
      its statistical date (or more than 50 days old), an IBES surprise before the
      announcement date + 1 day
    - accounting data is 1 to 15 months old; the target is exactly one month ahead
    - one row per stock and month; every row is a US common stock on NYSE / AMEX / NASDAQ
    - no missing or infinite model input, every scaled input within +-10
    - splits in date order; labels close to 20 / 60 / 20 in every split
    - industry_id and the unscaled risk columns are not numeric inputs
    - each market-regime input has one value per month (the same for every stock)
    - the targets inside volatility groups are 20 / 60 / 20 in every group
and prints, for each feature, the share of stock-months where it had a value.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as ft  # noqa: E402
from build_processed_data import (  # noqa: E402
    ACCOUNTING_AGE,
    CRSP_COMMON_STOCK,
    PANEL_NAME,
    find_project_root,
)


def main():
    processed = find_project_root() / "data" / "processed"
    panel_path = processed / f"panel_model_{PANEL_NAME}.parquet"
    files = [panel_path, processed / "feature_groups.json", processed / "split_dates.json",
             processed / "industry_mapping.json", processed / "preprocessing_report.csv"]
    print("--- files ---")
    for f in files:
        print(f.name, ":", "ok" if f.exists() else "MISSING")
        assert f.exists(), f"missing {f}"

    panel = pd.read_parquet(panel_path)
    fg = json.loads((processed / "feature_groups.json").read_text())
    industry_mapping = json.loads((processed / "industry_mapping.json").read_text())
    numeric_features = fg["numeric_features"]
    num_bin = fg["model_features_numeric_binary"]
    cats = fg["categorical_features"]
    panel["month"] = pd.to_datetime(panel["month"])
    panel["target_month"] = pd.to_datetime(panel["target_month"])
    month = panel["month"]

    print("\n--- timing ---")
    print("panel:", panel.shape)
    age = panel["accounting_age_months"].dropna()
    print(f"accounting age: {age.min():.0f}-{age.max():.0f} months, median {age.median():.0f}")
    assert age.between(*ACCOUNTING_AGE).all()
    for col, what in (("comp_availability_date_is", "income-statement features before their release / filing"),
                      ("comp_cf_availability_date", "cash-flow / balance-sheet features before their filing")):
        used = panel[col].notna()
        early = int((pd.to_datetime(panel.loc[used, col]) > month[used]).sum())
        print(f"{what}: {early}")
        assert early == 0
    both = panel["comp_cf_datadate"].notna() & panel["comp_datadate"].notna()
    older = (panel.loc[both, "comp_cf_datadate"] < panel.loc[both, "comp_datadate"]).mean()
    print(f"stock-months whose cash-flow features come from an older filing than the income "
          f"features (released, not yet filed): {older:.1%}")
    rules = {
        "si_settlement_date": ("short interest", pd.Timedelta(days=ft.SI_PUBLICATION_DAYS)),
        "ear_end_date": ("announcement return (day +1)", pd.Timedelta(0)),
        "inst_report_date": ("13F holdings", pd.Timedelta(days=ft.INST_FILING_DAYS)),
        "an_statpers": ("IBES consensus", pd.Timedelta(0)),
        "an_sue_ann_date": ("IBES surprise (announcement + 1 day)", pd.Timedelta(days=1)),
    }
    for col, (what, delay) in rules.items():
        if col in panel.columns:
            d = pd.to_datetime(panel[col])
            used = d.notna()
            early = int(((d[used] + delay) > month[used]).sum())
            age_m = (month[used] - d[used]).dt.days / 30.44
            print(f"{what}: used on {used.mean():.1%} of stock-months, median age {age_m.median():.1f} months, "
                  f"used before public: {early}")
            assert early == 0
    if "an_statpers" in panel.columns:
        d = pd.to_datetime(panel["an_statpers"])
        too_old = int(((month - d).dt.days > ft.IBES_MAX_AGE_DAYS).sum())
        print(f"IBES consensus older than {ft.IBES_MAX_AGE_DAYS} days: {too_old}")
        assert too_old == 0
    gap = (panel["target_month"].dt.year - month.dt.year) * 12 + (panel["target_month"].dt.month - month.dt.month)
    assert gap.eq(1).all(), "target not exactly 1 month ahead"
    assert panel.duplicated(["permno", "month"]).sum() == 0

    print("\n--- inputs ---")
    n_miss = int(panel[num_bin].isna().sum().sum())
    n_inf = int(np.isinf(panel[num_bin].select_dtypes(include=[np.number])).sum().sum())
    max_abs = float(panel[numeric_features].abs().max().max())
    print(f"missing={n_miss} missing categorical={int(panel[cats].isna().sum().sum())} inf={n_inf} max_abs={max_abs:.3f}")
    assert n_miss == 0 and n_inf == 0 and max_abs <= 10.000001
    assert panel[cats].isna().sum().sum() == 0
    assert "industry_id" not in num_bin and "industry_id" in cats
    for col in fg.get("raw_risk_columns", []):
        assert col in panel.columns and col not in num_bin, col
    for col in fg.get("flag_features", []):
        assert set(panel[col].unique()) <= {0, 1}, f"{col} is not 0/1"
    assert "__UNKNOWN__" in industry_mapping and industry_mapping["__UNKNOWN__"] == 0
    for col in fg.get("regime_features", []):
        per_month = panel.groupby("month")[col].nunique()
        assert (per_month <= 1).all(), f"{col} differs across stocks within a month"
        assert col in num_bin and col not in numeric_features
    if fg.get("regime_features"):
        print("regime inputs (one value per month):", ", ".join(fg["regime_features"]))

    print("\n--- universe and split ---")
    for col, values in CRSP_COMMON_STOCK.items():
        bad = ~panel[col].isin(values).fillna(False)
        assert not bad.any(), f"{int(bad.sum())} rows with {col} not in {values}"
    print("all rows are US common stocks on NYSE / AMEX / NASDAQ (CRSP flags)")
    print(panel.groupby("split")["month"].agg(["min", "max", "nunique"]))
    order = panel.groupby("split")["month"].agg(["min", "max"]).sort_values("min")
    assert (order["min"].shift(-1).dropna() > order["max"].iloc[:-1]).all(), "splits overlap"
    assert {"train", "val", "test"} <= set(panel["split"].unique())
    labels = panel.groupby("split")["label20"].value_counts(normalize=True).unstack()
    print("\nlabel20 shares by split:\n", labels.round(3))
    assert labels[0].between(0.19, 0.21).all() and labels[1].between(0.58, 0.62).all()
    assert labels[2].between(0.19, 0.21).all()
    if "label20_vg" in panel.columns:
        vg = panel.groupby("vol_group")["label20_vg"].value_counts(normalize=True).unstack()
        print("\nlabel20_vg shares by volatility group:\n", vg.round(3))
        assert vg[0].between(0.19, 0.22).all() and vg[2].between(0.19, 0.22).all()
        assert panel["rank_vg"].between(-0.5, 0.5).all()
    print("\nstock-months with a CCM link / a Compustat filing, by split:")
    print(pd.DataFrame({
        "ccm_link": panel.groupby("split")["gvkey"].apply(lambda s: s.notna().mean()),
        "compustat": panel.groupby("split")["comp_datadate"].apply(lambda s: s.notna().mean()),
    }).round(3))

    print("\n--- share of stock-months with a value (before imputation), by split ---")
    cov = {}
    for f in numeric_features:
        flag = f"miss_{f}"
        cov[f] = (1 - panel.groupby("split")[flag].mean()) if flag in panel.columns else \
            pd.Series(1.0, index=sorted(panel["split"].unique()))
    with pd.option_context("display.max_rows", 500):
        print(pd.DataFrame(cov).T.round(2).to_string())
    if fg.get("dropped_features"):
        print("dropped by the training missing-rate rule:", fg["dropped_features"])

    print("\n--- input counts ---")
    print("numeric:", len(numeric_features), f"({len(fg['return_features'])} monthly return, "
          f"{len(fg['market_features'])} market incl. {len(fg.get('analyst_features', []))} analyst, "
          f"{len(fg['compustat_features'])} Compustat)")
    print("regime:", len(fg.get("regime_features", [])))
    print("missing flags:", len(fg["missing_indicator_features"]))
    print("structural-zero flags:", len(fg["flag_features"]))
    print("model inputs:", len(num_bin), "+ industry id")
    print("\nall checks passed.")


if __name__ == "__main__":
    main()
