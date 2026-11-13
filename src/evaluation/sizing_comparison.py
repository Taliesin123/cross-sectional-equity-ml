"""
Position sizing: the same score traded with different portfolios.

Every variant is backtested with the shared engine (drift-aware turnover, 10 bps one-way
by default) over 2005-2025 in one continuous run, and reported separately for validation
(2005-2014), test (2015-Nov 2024) and the recent months (Dec 2024-Nov 2025). Choices (the
vol forecast method and its calibration) use validation data only. As in cost_study.py,
the book trades one day after the month-end close by default (--trade-at day1, with the
first-day returns of cost_inputs.parquet).

Variants
    1. Equal-weight deciles                 long top 10% / short bottom 10%, equal weights
    2. Rank weights                         all stocks, weight ~ centred rank
    3. Inverse-vol deciles                  deciles, weight ~ 1 / raw_vol_12m within each leg
    4. + beta-neutral                       leg sizes set so the book's beta is 0
    5. + industry-neutral                   deciles taken within each industry
    6. Full construction                    5 + buffer (enter 10%, exit 20%) + 1% cap per name
    7. Vol-neutral deciles                  each month, 5 equal-size groups by 12-month vol;
                                            long top 10% / short bottom 10% of each group
    8. Vol-neutral deciles + buffer         7 with the 10% / 20% buffer
    9. Equal-weight deciles, 10% vol target 1 scaled to a 10% annual vol (re-levered outside 9-11%)
   10. Full construction, 10% vol target    6 scaled the same way
   11. Vol-neutral deciles, 10% vol target  7 scaled the same way (book A of cost_study.py)

The vol target uses the adaptive correction (PortfolioConfig.vol_adaptive): the
forecast is rescaled by how much the book's own past 24 months over- or under-shot
it, using realised months only. vol_scale, calibrated on validation, is only used
for the first 12 months.

Usage from the project root (after mlp_model.py):
    python src/evaluation/sizing_comparison.py
    python src/evaluation/sizing_comparison.py --vol-method-study    # also re-run the method choice
    python src/evaluation/sizing_comparison.py --costs realistic --aum 1e8
                         # every variant under realistic costs (needs cost_inputs.parquet);
                         # -> results/sizing_<run>_costs_100M/
Outputs: results/sizing_mlp_filing/ (CSV tables + figures). The holdout is not used here:
it is opened once, for the one book chosen on validation, by cost_study.py --holdout.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from evaluation.backtest import add_forward_market_return  # noqa: E402
from evaluation.portfolio import (  # noqa: E402
    PortfolioConfig,
    add_risk_columns,
    add_trade_timing,
    backtest_portfolio,
    calibrate_vol_scale,
    make_returns_wide,
    summarize_backtest,
)

# Reference palette, light mode (dataviz skill): slots 1-2 + neutrals.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb", "#d9d8d3"


def make_variants(target=0.10, band=(0.09, 0.11)):
    ew = PortfolioConfig()
    inv = replace(ew, name="Inverse-vol deciles", leg_weighting="inverse_vol")
    beta = replace(inv, name="+ beta-neutral", beta_neutral=True)
    ind = replace(beta, name="+ industry-neutral", industry_neutral=True)
    full = replace(ind, name="Full construction (+ buffer 10/20%, cap 1%)", buffer_exit=0.20, max_weight=0.01)
    vn = replace(ew, name="Vol-neutral deciles (5 vol groups)", vol_neutral=True)
    vt = dict(vol_target=target, vol_band=tuple(band))
    return [
        ew,
        replace(ew, name="Rank weights (all stocks)", scheme="rank"),
        inv,
        beta,
        ind,
        full,
        vn,
        replace(vn, name="Vol-neutral deciles + buffer 10/20%", buffer_exit=0.20),
        replace(ew, name=f"Equal-weight deciles, {target:.0%} vol target", **vt),
        replace(full, name=f"Full construction, {target:.0%} vol target", **vt),
        replace(vn, name=f"Vol-neutral deciles, {target:.0%} vol target", **vt),
    ]


def vol_method_study(val, returns_wide, books, target, band, cost):
    """Choose the forecast method on validation only: calibrate on 2005-09, measure on 2010-14."""
    first = val[val["month"] <= "2009-12-31"]
    second = pd.DatetimeIndex(val.loc[val["month"] >= "2010-01-01", "month"].unique())
    rows = []
    for book in books:
        for method in ("holdings", "realized", "blend"):
            for halflife in (6.0, 12.0):
                cfg = replace(book, vol_target=target, vol_band=tuple(band),
                              vol_method=method, vol_halflife=halflife)
                cfg = calibrate_vol_scale(first, cfg, returns_wide, cost)
                res = backtest_portfolio(val, cfg, returns_wide, cost)
                net = res["returns_with_fees"].reindex(second)
                vol = np.sqrt(12) * net.std()
                rows.append({
                    "book": book.name, "method": method, "halflife": halflife,
                    "scale_2005_09": cfg.vol_scale, "vol_2010_14": vol,
                    "abs_error": abs(vol - target),
                    "leverage_resets": int(res["diagnostics"].reindex(second)["leverage_reset"].sum()),
                })
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which="both", colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figures(results, out_dir, target, band, test_start):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    pct = FuncFormatter(lambda y, _: f"{y:.0%}")
    ew = "Equal-weight deciles"
    vn = "Vol-neutral deciles (5 vol groups)"
    ew_vt = f"Equal-weight deciles, {target:.0%} vol target"
    vn_vt = f"Vol-neutral deciles, {target:.0%} vol target"

    def finish(fig, ax, title, ylabel, name, legend_loc="upper left"):
        ax.axvline(test_start, color=INK_2, linewidth=1, linestyle=":")
        ax.text(test_start, ax.get_ylim()[1], " test →", color=INK_2, fontsize=9, va="top", ha="left")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc=legend_loc)
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=200)
        plt.close(fig)

    # 1. Drawdown of the equal-weight book vs the vol-neutral book (no vol target, net of costs)
    fig, ax = plt.subplots(figsize=(9, 4))
    _style(ax)
    for key, color, label in [(ew, ORANGE, "equal-weight deciles"), (vn, BLUE, "vol-neutral deciles")]:
        nav = (1 + results[key]["returns_with_fees"]).cumprod()
        dd = nav / nav.cummax() - 1
        ax.plot(dd.index, dd.values, color=color, linewidth=1.8, label=label)
    ax.yaxis.set_major_formatter(pct)
    finish(fig, ax, "Drawdown from the previous peak, net of costs, no vol target", "drawdown",
           "drawdown_equal_weight_vs_vol_neutral.png", legend_loc="lower left")

    # 2. Rolling 12-month realised vol of the vol-neutral book, with and without the target
    fig, ax = plt.subplots(figsize=(9, 4))
    _style(ax)
    ax.axhspan(band[0], band[1], color=BAND, alpha=0.6, linewidth=0, label=f"target band {band[0]:.0%}-{band[1]:.0%}")
    for key, color, label in [(vn, ORANGE, "vol-neutral, no target"),
                              (vn_vt, BLUE, f"vol-neutral, {target:.0%} target")]:
        roll = results[key]["returns_with_fees"].rolling(12).std() * np.sqrt(12)
        ax.plot(roll.index, roll.values, color=color, linewidth=1.8, label=label)
    ax.yaxis.set_major_formatter(pct)
    finish(fig, ax, "Rolling 12-month realised volatility (net returns)", "annualised vol", "rolling_vol_vs_target.png")

    # 3. Growth of 1 for the two vol-targeted books
    fig, ax = plt.subplots(figsize=(9, 4))
    _style(ax)
    for key, color in [(ew_vt, ORANGE), (vn_vt, BLUE)]:
        nav = (1 + results[key]["returns_with_fees"]).cumprod()
        ax.plot(nav.index, nav.values, color=color, linewidth=1.8, label=key)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}" if f"{y:g}"[0] in "25" else ""))
    finish(fig, ax, f"Growth of 1, net of costs, both books at a {target:.0%} vol target", "value (log scale)", "cumulative_vol_targeted.png")

    # 4. Leverage L_t
    fig, ax = plt.subplots(figsize=(9, 4))
    _style(ax)
    for key, color in [(ew_vt, ORANGE), (vn_vt, BLUE)]:
        lev = results[key]["diagnostics"]["leverage"]
        ax.step(lev.index, lev.values, where="post", color=color, linewidth=1.8, label=key)
    finish(fig, ax, "Leverage applied to the book (gross exposure = 2 × leverage)", "leverage", "leverage.png")


def main():
    parser = argparse.ArgumentParser(description="Compare position-sizing schemes on one model's predictions.")
    parser.add_argument("--run", default="wf_full", help="results/<run>/predictions.parquet")
    parser.add_argument("--trade-at", choices=["day1", "close"], default="day1")
    parser.add_argument("--predictions", default=None, help="explicit predictions file (overrides --run)")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--target", type=float, default=0.10)
    parser.add_argument("--band", type=float, nargs=2, default=[0.09, 0.11])
    parser.add_argument("--vol-method-study", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--costs", choices=["flat", "realistic"], default="flat",
                        help="flat --cost-bps (default) or the per-stock model of evaluation/costs.py")
    parser.add_argument("--aum", type=float, default=1e8, help="fund size for --costs realistic ($)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"
    pred_path = Path(args.predictions) if args.predictions else root / "results" / args.run / "predictions.parquet"
    run_name = pred_path.parent.name if args.predictions else args.run
    suffix = f"_costs_{args.aum / 1e6:g}M" if args.costs == "realistic" else ""
    out_dir = root / "results" / f"sizing_{run_name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cost = args.cost_bps / 1e4
    cost_model = None
    if args.costs == "realistic":
        from evaluation.costs import CostModel, RealisticCosts, load_cost_inputs

        cost_model = RealisticCosts(load_cost_inputs(root), CostModel(aum=args.aum))
        print(f"Costs: {cost_model.name} (vol scales still calibrated with the flat cost, so weights are unchanged)")

    print("predictions:", pred_path)
    pred = pd.read_parquet(pred_path)
    pred["month"] = pd.to_datetime(pred["month"])
    panel = pd.read_parquet(
        processed / "panel_model_filing.parquet",
        columns=["permno", "month", "split", "ret_fwd1m", "mkt_ret", "raw_vol_12m", "raw_beta_12m"],
    )
    panel["month"] = pd.to_datetime(panel["month"])
    returns_wide = make_returns_wide(panel)          # all months, for the vol forecast history
    pred = pred.drop(columns=[c for c in ("split", "ret_fwd1m") if c in pred.columns]).merge(
        panel[["permno", "month", "split", "ret_fwd1m"]], on=["permno", "month"], how="inner")
    pred = add_risk_columns(pred, panel)
    if "industry_id" not in pred.columns:
        raise ValueError("predictions need industry_id for the industry-neutral variants")
    timing = args.trade_at
    ci = processed / "cost_inputs.parquet"
    if timing == "day1":
        day1 = pd.read_parquet(ci, columns=["permno", "month", "ret_day1"]) if ci.exists() else None
        if day1 is None or day1["ret_day1"].notna().sum() == 0:
            print("No first-day returns (cost_inputs.parquet): trading at the month-end close.")
            timing = "close"
        else:
            day1["month"] = pd.to_datetime(day1["month"])
            pred = add_trade_timing(pred, day1.set_index(["permno", "month"])["ret_day1"], "day1")
    print("Trading at:", "the close of the first day of the next month" if timing == "day1" else "the month-end close")

    splits = [s for s in ("val", "test", "recent") if (pred["split"] == s).any()]
    oos = pred[pred["split"].isin(splits)].copy()
    val = oos[oos["split"] == "val"].copy()
    test_months = pd.DatetimeIndex(oos.loc[oos["split"] == "test", "month"].unique())
    split_months = {name: pd.DatetimeIndex(oos.loc[oos["split"] == name, "month"].unique()) for name in splits}
    market_fwd = add_forward_market_return(
        oos[["month"]].drop_duplicates(), panel
    ).set_index("month")["mkt_fwd1m"]

    variants = make_variants(args.target, args.band)
    if args.vol_method_study:
        books = [variants[0], variants[5], variants[6]]
        study = vol_method_study(val, returns_wide, books, args.target, args.band, cost)
        study.to_csv(out_dir / "vol_method_study_validation.csv", index=False)
        print("\nVol forecast methods (validation only: calibrate 2005-09, measure 2010-14):")
        print(study.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    rows, results = [], {}
    for cfg in variants:
        if cfg.vol_target is not None:
            cfg = calibrate_vol_scale(val, cfg, returns_wide, cost)
            note = " (used for the first 12 months; adaptive correction after)" if cfg.vol_adaptive else ""
            print(f"{cfg.name}: vol_scale calibrated on validation = {cfg.vol_scale:.3f}{note}")
        res = backtest_portfolio(oos, cfg, returns_wide, cost, cost_model=cost_model)
        results[cfg.name] = res
        for split_name, months in split_months.items():
            row = summarize_backtest(res, months, market_fwd, band=tuple(args.band))
            diag = res["diagnostics"]
            if diag is not None:
                d = diag.reindex(months)
                row["avg_leverage"] = d["leverage"].mean()
                row["leverage_resets"] = int(d["leverage_reset"].sum())
                row["vol_scale"] = cfg.vol_scale
            rows.append({"variant": cfg.name, "split": split_name, **row})
        if diag is not None:
            safe = "".join(ch if ch.isalnum() else "_" for ch in cfg.name).strip("_")
            out = diag.copy()
            out["net_return"] = res["returns_with_fees"]
            out["gross_exposure"] = res["gross_exposure"]
            out.to_csv(out_dir / f"vol_target_monthly_{safe}.csv")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "sizing_comparison.csv", index=False)

    cols = ["ann_return_net", "ann_vol_net", "sharpe_gross", "sharpe_net", "max_drawdown_net",
            "turnover_per_month", "breakeven_cost_bps", "avg_gross_exposure", "beta_vs_market"]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        for split_name in splits:
            t = table[table["split"] == split_name].set_index("variant")[cols]
            label = cost_model.name if cost_model is not None else f"{args.cost_bps:g} bps one-way"
            print(f"\n{split_name.upper()}  ({label})")
            print(t.to_string(float_format=lambda x: f"{x:8.3f}"))

    lo, hi = args.band
    print(f"\nVol target check: realised annual vol of the net returns, target {args.target:.0%} "
          f"(band {lo:.0%}-{hi:.0%}); test is the real check.")
    vt = table[table["avg_leverage"].notna()] if "avg_leverage" in table else table.iloc[0:0]
    for _, r in vt.iterrows():
        flag = "inside" if lo <= r["ann_vol_net"] <= hi else "OUTSIDE"
        print(f"  {r['variant']:<40} {r['split']:<5} vol {r['ann_vol_net']:.1%}  ({flag} the band), "
              f"rolling-12m vol in band {r['rolling12m_vol_in_band']:.0%} of months")

    if not args.no_plots:
        save_figures(results, out_dir, args.target, args.band, test_months.min())
    print("\nSaved to", out_dir)


if __name__ == "__main__":
    main()
