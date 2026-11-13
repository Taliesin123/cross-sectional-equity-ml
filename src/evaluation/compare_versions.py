"""
v5 against v4, under the rule written before the v5 results exist.

Both versions are evaluated in this folder with the same, more realistic backtest
(trading one day later, the short-sale limit, borrow tiers, the vol forecast chosen on
validation): v4 = results/mlp_filing (one fit on 1980-2004, copied from the v4 folder),
v5 = the walk-forward ensemble.

Pre-registered rules (README section 6, rules 3 and 4):
  3. Market-regime inputs stay only if, on validation, scrambling them lowers the network's
     IC with a t-statistic above 2 (walk_forward.py --importance). Otherwise the v5
     candidate is wf_full_noregime (python src/models/walk_forward.py --variants full_noregime).
  4. v5 replaces v4 only if, on VALIDATION (2005-2014):
       (a) its monthly rank IC on the stock-months both score is higher, with a t-statistic
           of the monthly difference above 2, and
       (b) the book its cost study chooses has a higher net Sharpe at $100M (realistic costs).
  Test (2015-Nov 2024) and recent (Dec 2024-Nov 2025) numbers are reported and decide nothing.

Run from the project root (after walk_forward.py, walk_forward.py --importance and
cost_study.py for both runs):
    python src/evaluation/compare_versions.py
Output: results/walk_forward/version_comparison.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from models.walk_forward import group_spearman  # noqa: E402

LADDER = ["mlp_filing", "wf_base_seed1", "wf_base", "wf_full_mlp", "wf_full_gbm", "wf_full",
          "wf_full_noregime", "wf_full_noibes", "wf_full_label20"]
LABELS = {
    "mlp_filing": "v4: one fit 1980-2004, v4 inputs",
    "wf_base_seed1": "+ refit every year (1 seed)",
    "wf_base": "+ 5 seeds averaged",
    "wf_full_mlp": "+ analyst + regime inputs, target inside vol groups",
    "wf_full_gbm": "trees alone (all v5 inputs)",
    "wf_full": "v5: MLP seeds + trees",
    "wf_full_noregime": "v5 without the regime inputs",
    "wf_full_noibes": "MLP without the analyst inputs",
    "wf_full_label20": "MLP, all-stock target",
}


def load_pred(root, run):
    f = root / "results" / run / "predictions.parquet"
    if not f.exists():
        return None
    p = pd.read_parquet(f, columns=["permno", "month", "score"])
    p["month"] = pd.to_datetime(p["month"])
    return p


def regime_verdict(root):
    f = root / "results" / "walk_forward" / "importance_full_validation.csv"
    if not f.exists():
        return None, "not tested yet (run: python src/models/walk_forward.py --importance)"
    t = pd.read_csv(f, index_col=0)
    if "market regime" not in t.index:
        return True, "no regime inputs in the panel"
    r = t.loc["market regime"]
    keep = bool(r["ic_drop"] > 0 and r["t"] > 2)
    return keep, f"IC drop {r['ic_drop']:.4f} (t {r['t']:.1f}) when scrambled -> {'kept' if keep else 'dropped'}"


def main():
    parser = argparse.ArgumentParser(description="v5 against v4 under the pre-registered rule.")
    parser.add_argument("--baseline", default="mlp_filing")
    parser.add_argument("--candidate", default=None, help="default: wf_full, or wf_full_noregime per rule 3")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    panel = pd.read_parquet(root / "data" / "processed" / "panel_model_filing.parquet",
                            columns=["permno", "month", "split", "ret_fwd1m"])
    panel["month"] = pd.to_datetime(panel["month"])

    keep_regime, why = regime_verdict(root)
    candidate = args.candidate or ("wf_full" if keep_regime in (True, None) else "wf_full_noregime")
    print(f"Rule 3, market-regime inputs: {why}. Candidate: {candidate}")

    runs = {r: load_pred(root, r) for r in LADDER}
    runs = {r: p for r, p in runs.items() if p is not None}
    if args.baseline not in runs or candidate not in runs:
        sys.exit(f"Need results/{args.baseline}/predictions.parquet and results/{candidate}/predictions.parquet.")
    common = None
    for p in runs.values():
        k = p[["permno", "month"]]
        common = k if common is None else common.merge(k, on=["permno", "month"])
    base = panel.merge(common, on=["permno", "month"])
    ics = {}
    for r, p in runs.items():
        d = base.merge(p, on=["permno", "month"])
        ics[r] = group_spearman(d, ["month"], "score", "ret_fwd1m")
    split = base.drop_duplicates("month").set_index("month")["split"]

    rows = []
    for r in runs:
        row = {"run": r, "description": LABELS.get(r, r)}
        for s in ("val", "test", "recent"):
            m = split[split == s].index
            x = ics[r].reindex(m).dropna()
            row[f"ic_{s}"] = x.mean() if len(x) else np.nan
        cs = root / "results" / f"cost_study_{r}"
        if (cs / "chosen_book.json").exists() and (cs / "cost_study.csv").exists():
            plan = json.loads((cs / "chosen_book.json").read_text())
            t = pd.read_csv(cs / "cost_study.csv")
            aum = plan["cost_model"]["aum"]
            t = t[(t["book"] == plan["book"]["name"]) & (t["aum"] == aum)]
            row["chosen_book"] = plan["book"]["name"]
            for s in ("val", "test", "recent"):
                x = t[t["split"] == s]
                if len(x):
                    row[f"sharpe_{s}"] = float(x["sharpe_net"].iloc[0])
                    row[f"return_{s}"] = float(x["ann_return_net"].iloc[0])
                    row[f"vol_{s}"] = float(x["ann_vol_net"].iloc[0])
                    row[f"maxdd_{s}"] = float(x["max_drawdown_net"].iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    out = root / "results" / "walk_forward"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "version_comparison.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 55):
        print("\nMonthly rank IC on the same stock-months, and the chosen book at $100M (net, realistic costs):")
        show = ["description", "ic_val", "ic_test", "ic_recent", "chosen_book", "sharpe_val", "sharpe_test",
                "sharpe_recent", "maxdd_test"]
        print(table[[c for c in show if c in table]].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # IC by size group (NYSE percentiles from cost_inputs.parquet): where does each version rank well?
    ci = root / "data" / "processed" / "cost_inputs.parquet"
    if ci.exists():
        size = pd.read_parquet(ci, columns=["permno", "month", "size_nyse_pct"])
        size["month"] = pd.to_datetime(size["month"])
        size["size_group"] = pd.cut(size["size_nyse_pct"], [-0.01, 0.2, 0.5, 1.01],
                                    labels=["micro (<20% NYSE)", "small (20-50%)", "large (>50%)"])
        srows = []
        for r in (args.baseline, candidate):
            d = base.merge(runs[r], on=["permno", "month"]).merge(size[["permno", "month", "size_group"]],
                                                                 on=["permno", "month"], how="inner")
            d["size_group"] = d["size_group"].astype(str)
            ic = group_spearman(d, ["month", "size_group"], "score", "ret_fwd1m", min_n=20)
            for s in ("val", "test", "recent"):
                m = split[split == s].index
                x = ic[ic.index.get_level_values(0).isin(m)]
                for g, v in x.groupby(level=1):
                    srows.append({"run": r, "split": s, "size_group": g, "ic": v.mean()})
        st = pd.DataFrame(srows).pivot_table(index=["size_group", "split"], columns="run", values="ic")
        st.to_csv(out / "ic_by_size.csv")
        print("\nRank IC by size group:")
        print(st.to_string(float_format=lambda x: f"{x:7.3f}"))

    # rule 4
    m = split[split == "val"].index
    diff = (ics[candidate] - ics[args.baseline]).reindex(m).dropna()
    t_ic = diff.mean() / diff.std() * np.sqrt(len(diff))
    a = bool(diff.mean() > 0 and t_ic > 2)
    rb, rc = table.set_index("run").loc[args.baseline], table.set_index("run").loc[candidate]
    have_books = "sharpe_val" in table and pd.notna(rb.get("sharpe_val")) and pd.notna(rc.get("sharpe_val"))
    b = bool(have_books and rc["sharpe_val"] > rb["sharpe_val"])
    print(f"\nRule 4 (validation only): (a) IC {rc['ic_val']:.4f} vs {rb['ic_val']:.4f}, difference "
          f"{diff.mean():+.4f} (t {t_ic:.1f}): {'yes' if a else 'no'}")
    if have_books:
        print(f"                          (b) chosen book's net Sharpe {rc['sharpe_val']:.2f} vs "
              f"{rb['sharpe_val']:.2f}: {'yes' if b else 'no'}")
    else:
        print("                          (b) run cost_study.py for both runs first")
    verdict = candidate if (a and b) else args.baseline
    print(f"\nVerdict: {'v5 (' + candidate + ')' if verdict == candidate else 'v4 stays'}"
          + ("" if have_books else " (incomplete: cost studies missing)"))
    with open(out / "verdict.json", "w") as f:
        json.dump({"regime_rule": why, "candidate": candidate, "baseline": args.baseline,
                   "ic_difference_val": float(diff.mean()), "ic_difference_t": float(t_ic),
                   "rule_a": a, "rule_b": b, "verdict": verdict}, f, indent=4)


if __name__ == "__main__":
    main()
