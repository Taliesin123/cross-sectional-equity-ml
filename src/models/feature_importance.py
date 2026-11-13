"""
Which feature groups does the trained MLP rely on?

Permutation importance by group: within every month, the values of one group of features
(with their missing flags) are shuffled across stocks, the trained model scores the
shuffled data, and the drop in the monthly rank IC is the group's importance. Shuffling
within a month keeps each feature's cross-section intact and only breaks its link with
the stock; a group is shuffled as a block, so correlated features in the same group
cannot stand in for each other.

Run from the project root after training (src/models/mlp_model.py):
    python src/models/feature_importance.py
    python src/models/feature_importance.py --split test --repeats 3

Uses validation months by default (analysis only: nothing is chosen from it).
Output: results/mlp_filing/feature_importance_<split>.csv (+ printed table).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

import features as ft  # noqa: E402
from build_processed_data import CASH_FLOW_FEATURES  # noqa: E402

MOMENTUM = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag6", "ret_lag12", "reversal_1m", "mom6m1",
            "mom12m1", "ret_36_13", "seasonal_1y", "seasonal_2_5y", "industry_mom_6m", "price_to_12m_high"]
MONTHLY_RISK = ["vol_6m", "vol_12m", "beta_12m", "beta_24m", "idio_vol_12m"]
SIZE_TRADING = ["log_mktcap", "log_price", "turnover_1m", "turnover_12m", "log_dollar_volume_3m"]
GROUP_OF = {
    **{f: "momentum / reversal (monthly returns)" for f in MOMENTUM},
    **{f: "risk (monthly returns)" for f in MONTHLY_RISK},
    **{f: "risk / lottery / liquidity (daily returns)" for f in ft.DAILY_FEATURES},
    **{f: "size / trading" for f in SIZE_TRADING},
    **{f: "issuance / payout (prices)" for f in ["share_issuance_12m", "dividend_yield_12m"]},
    **{f: "short interest" for f in ft.SHORT_INTEREST_FEATURES},
    **{f: "earnings-announcement return" for f in ft.ANNOUNCEMENT_FEATURES},
    **{f: "institutional ownership (13F)" for f in ft.INSTITUTIONAL_FEATURES},
    **{f"comp_{f}": "balance sheet" for f in ft.BALANCE_SHEET_FEATURES},
    **{f"comp_{f}": "value ratios to market cap" for f in ft.MARKET_RATIO_FEATURES},
    "comp_flag_negative_book_equity": "value ratios to market cap",
    **{f"comp_{f}": "earnings surprise" for f in ft.EARNINGS_FEATURES},
    **{f: "analyst forecasts (IBES)" for f in ft.ANALYST_FEATURES},
    **{f: "market regime" for f in ft.REGIME_FEATURES},
}


def group_of(feature, cash_flow):
    base = feature[5:] if feature.startswith("miss_") else feature
    if base in GROUP_OF:
        return GROUP_OF[base]
    name = base[5:] if base.startswith("comp_") else base
    if base.startswith("comp_"):
        return "cash-flow statement" if name in cash_flow else "income statement"
    return "other"


def monthly_ic(score, ret, month):
    df = pd.DataFrame({"s": score, "r": ret, "m": month})
    ics = df.groupby("m")[["s", "r"]].apply(
        lambda g: g["s"].corr(g["r"], method="spearman") if len(g) >= 50 else np.nan)
    return ics.dropna()


def main():
    parser = argparse.ArgumentParser(description="Permutation importance of feature groups.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--repeats", type=int, default=1, help="shuffles per group (averaged)")
    parser.add_argument("--run-name", default="mlp_filing", help="results folder of the trained model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from mlp_model import get_device, get_project_root, load_mlp_checkpoint, load_panel_and_features, \
        predict_proba_numpy

    root = get_project_root()
    run_dir = root / "results" / args.run_name
    ckpt = run_dir / "best_model.pt"
    if not ckpt.exists():
        sys.exit(f"{ckpt} not found: train first (python src/models/mlp_model.py)")
    panel, features = load_panel_and_features()
    cash_flow = set(CASH_FLOW_FEATURES)
    device = get_device()
    model, _ = load_mlp_checkpoint(ckpt, device=device, expected_features=features)

    sub = panel[panel["split"] == args.split].sort_values(["month", "permno"]).reset_index(drop=True)
    del panel
    x = sub[features].to_numpy(dtype=np.float32)
    ind = sub["industry_id"].to_numpy(dtype=np.int64)
    ret = sub["ret_fwd1m"].to_numpy()
    month = sub["month"].to_numpy()
    starts = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])
    ends = np.r_[starts[1:], len(month)]

    def score(xm, im):
        p, _ = predict_proba_numpy(model, xm, im, 65536, device)
        return p[:, 2] - p[:, 0]

    base_ic = monthly_ic(score(x, ind), ret, month)
    print(f"{args.split}: {len(sub):,} stock-months, {len(starts)} months, base IC {base_ic.mean():.4f}")

    groups = {}
    for i, f in enumerate(features):
        groups.setdefault(group_of(f, cash_flow), []).append(i)
    groups["industry (embedding)"] = None
    rng = np.random.default_rng(args.seed)
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(args.repeats):
            perm = np.arange(len(sub))
            for a, b in zip(starts, ends):
                perm[a:b] = a + rng.permutation(b - a)
            if cols is None:
                xs, inds = x, ind[perm]
            else:
                xs = x.copy()
                xs[:, cols] = x[np.ix_(perm, cols)]
                inds = ind
            ic = monthly_ic(score(xs, inds), ret, month)
            drops.append((base_ic - ic.reindex(base_ic.index)).dropna())
        d = pd.concat(drops, axis=1).mean(axis=1)
        rows.append({"group": name, "n_inputs": 0 if cols is None else len(cols),
                     "ic_drop": d.mean(), "ic_drop_t": d.mean() / d.std(ddof=1) * np.sqrt(len(d)),
                     "share_of_base_ic": d.mean() / base_ic.mean()})
        print(f"  {name:48s} IC drop {d.mean():+.4f} (t {rows[-1]['ic_drop_t']:5.1f})")
    out = pd.DataFrame(rows).sort_values("ic_drop", ascending=False)
    out.insert(0, "base_ic", base_ic.mean())
    path = run_dir / f"feature_importance_{args.split}.csv"
    out.to_csv(path, index=False)
    print("\n" + out.round(4).to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
