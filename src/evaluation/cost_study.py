"""
Realistic trading costs for eleven versions of the book, the choice of one on validation,
and the final holdout run.

Same predictions; only the portfolio rules and the cost charged change. Costs per stock
and month (evaluation/costs.py): per-share commission + half bid-ask spread +
square-root market impact (depends on the fund size) + a borrow fee on shorts by borrow
tier (general collateral / hard to borrow / special).

Realism settings (v5 defaults):
    --trade-at day1      the book trades at the close of the first trading day of month
                         t+1, one day after the month-end close the inputs are measured at
                         (the headline). --trade-at close gives the old timing; the delay
                         table compares the two.
    --min-short-io 0.05  no short position in a stock with less than 5% of its shares held
                         by 13F institutions (no lendable supply); a short already held is
                         closed when the stock drops below. 0 switches the rule off.
    --vol-method auto    the volatility forecast of the 10% target: the monthly "blend" of
                         v4 or the "daily" replay of the book on its stocks' daily returns
                         (crsp_daily_returns.parquet), whichever tracks the target better on
                         validation (RMSE of the rolling 12-month vol against 10%).

Books (all vol-neutral deciles with the 10% vol target; filters use the price and size at
the end of month t, known when trading):
    A. Vol-neutral deciles          long top 10% / short bottom 10% of each of 5 vol groups
    B. + buffer 10/20%              a name enters at the top / bottom 10% of its vol
                                    group and is only sold once it leaves the 20%
    C. Price >= $5                  stocks under $5 removed before ranking
    D. Price >= $5 + buffer
    E. No microcaps + buffer        also removes stocks below the NYSE 20th size percentile
    F. Liquidity weights + buffer   inside each leg weight ~ sqrt(daily $ volume, 6-month
                                    average), capped at 2% of capital per name
    G. $5 + buffer + liquidity      C's filter with F's weights
    H. G + volume cap               no position above 5% of the stock's daily $ volume
                                    (6-month average) at the fund size being tested
    I. H + no-trade band            a held name is not traded while its drifted weight is
                                    within 25% of its target (vol-target re-levering is
                                    still applied)
    J. I + index hedge              no single-name shorts in microcaps; their long
                                    exposure is hedged with a short in a small-cap index
                                    (value-weighted CRSP stocks below the NYSE median,
                                    a stand-in for a Russell 2000 future / ETF)
    K. F + cost-aware trading       F, but each stock moves only part of the way to its
                                    target each month, the more slowly the more it costs
                                    to trade (Garleanu-Pedersen); a position no longer in
                                    the target is closed once below a quarter of an average
                                    position; the speed parameter kappa is chosen on
                                    validation (partial_trading_grid.csv)

Cost scenarios: a flat 10 bps, and realistic costs for funds of $10M to $3B (in
end-of-2024 dollars, scaled with the market before), plus sensitivities at the decision
size.

The choice is made on VALIDATION only: the book with the highest validation net
Sharpe under realistic costs at the decision fund size (--decision-aum, $100M by
default). Test and recent months are printed for information. The choice, its settings
and the pre-registered holdout checks are written to chosen_book.json.

Usage from the project root (after build_cost_inputs.py):
    python src/evaluation/cost_study.py --run wf_full     # validation + test + recent
    python src/evaluation/cost_study.py --run wf_full --holdout       # ONCE, at the very end

--holdout reads chosen_book.json, checks that the validation / test numbers are
unchanged (nothing was modified after the choice), evaluates that one book on the
holdout months, and writes holdout_result.json. It refuses to run a second time. Once
holdout_result.json exists, a new run of the study leaves chosen_book.json as it is.

Outputs: results/cost_study_<run>/ (tables, monthly book returns, figures, chosen_book.json).
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from evaluation.backtest import add_forward_market_return, portfolio_returns_and_turnover  # noqa: E402
from evaluation.costs import CostModel, RealisticCosts, load_cost_inputs  # noqa: E402
from evaluation.portfolio import (  # noqa: E402
    PortfolioConfig,
    add_risk_columns,
    add_trade_timing,
    apply_no_trade_band,
    apply_partial_trading,
    build_portfolio,
    calibrate_vol_scale,
    cap_weights_by_name,
    make_daily_returns_wide,
    make_returns_wide,
    summarize_backtest,
    vol_bucket,
)

FLAT_COST = 0.0010
TABLE_AUMS = (1e7, 1e8, 1e9)
CURVE_AUMS = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9)
HEDGE_ID = -1          # permno used for the small-cap index hedge
MICROCAP = 0.20        # NYSE size percentile below which a stock is a microcap
HEDGE_SIZE = 0.50      # the hedge index: value-weighted stocks below the NYSE median
# The hedge instrument's trading inputs (a liquid small-cap ETF / future)
HEDGE_INPUTS = {"half_spread": 0.5e-4, "sigma_daily": 0.015, "adv_usd": 5e9,
                "hard_to_borrow": False, "prc": 200.0}

# Reference palette, light mode (dataviz skill): categorical slots 1-8, fixed order, + neutrals.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
PLOT_BOOKS = ("A", "C", "E", "F", "G", "I", "J", "K")   # 8 lines at most on one chart
KAPPA_GRID = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0)   # cost-aware trading speeds tried on validation
REPORT_SPLITS = ("val", "test", "recent")


@dataclass(frozen=True)
class Book:
    name: str
    cfg: PortfolioConfig
    min_price: float | None = None
    min_size_pct: float | None = None    # NYSE size percentile, 0.20 = no microcaps
    adv_cap: float | None = None         # max position, as a fraction of daily $ volume
    trade_band: float | None = None      # no-trade band, relative to the target weight
    hedge_microcap_shorts: bool = False  # replace microcap shorts by a small-cap index short
    partial_kappa: float | None = None   # cost-aware partial trading speed (None = trade fully)

    @property
    def letter(self):
        return self.name.split(".")[0]

    def to_json(self):
        d = asdict(self)
        d["cfg"] = asdict(self.cfg)
        return d

    @staticmethod
    def from_json(d):
        cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in d["cfg"].items()}
        rest = {k: v for k, v in d.items() if k not in ("name", "cfg")}
        return Book(d["name"], PortfolioConfig(**cfg), **rest)


def make_books(target=0.10, band=(0.09, 0.11), short_col=None, vol_method="blend", kappa=None):
    base = PortfolioConfig(name="Vol-neutral deciles, 10% vol target", vol_neutral=True,
                           vol_target=target, vol_band=tuple(band), short_allowed_col=short_col,
                           vol_method=vol_method)
    buf = replace(base, buffer_exit=0.20)
    liq = replace(buf, leg_weighting="sqrt_adv", liquidity_col="adv_usd_6m", max_weight=0.02)
    return [
        Book("A. Vol-neutral deciles", base),
        Book("B. + buffer 10/20%", buf),
        Book("C. Price >= $5", base, min_price=5.0),
        Book("D. Price >= $5 + buffer", buf, min_price=5.0),
        Book("E. No microcaps + buffer", buf, min_price=5.0, min_size_pct=MICROCAP),
        Book("F. Liquidity weights + buffer", liq),
        Book("G. $5 + buffer + liquidity weights", liq, min_price=5.0),
        Book("H. G + position <= 5% of daily volume", liq, min_price=5.0, adv_cap=0.05),
        Book("I. H + no-trade band 25%", liq, min_price=5.0, adv_cap=0.05, trade_band=0.25),
        Book("J. I + microcap shorts -> index hedge", liq, min_price=5.0, adv_cap=0.05, trade_band=0.25,
             hedge_microcap_shorts=True),
        Book("K. F + cost-aware trading", liq, partial_kappa=kappa if kappa is not None else 0.0),
    ]


def apply_filters(pred, book):
    keep = pd.Series(True, index=pred.index)
    if book.min_price is not None:
        keep &= pred["prc"] >= book.min_price            # NaN price -> dropped
    if book.min_size_pct is not None:
        keep &= pred["size_nyse_pct"] >= book.min_size_pct
    return pred.loc[keep].copy()


# -------- data --------


@dataclass
class Context:
    pred: pd.DataFrame
    returns_wide: pd.DataFrame
    inputs: pd.DataFrame
    costs: RealisticCosts
    market_fwd: pd.Series
    hedge_ret: pd.Series
    split_of_month: pd.Series
    has_day1: bool
    hedge_ret_close: pd.Series = None
    daily_wide: pd.DataFrame = None
    short_col: str | None = None


def small_cap_index(panel, inputs):
    """Monthly return of the hedge index: value-weighted ret_fwd1m of the panel's stocks
    below the NYSE median (weights = market cap at the end of month t), and its return on
    the first trading day of each month (for the delay test)."""
    x = panel[["permno", "month", "ret_fwd1m"]].merge(
        inputs[["permno", "month", "mktcap_usd", "size_nyse_pct", "ret_day1"]], on=["permno", "month"], how="inner")
    x = x[(x["size_nyse_pct"] < HEDGE_SIZE) & (x["mktcap_usd"] > 0)]
    w = x["mktcap_usd"]
    ok = x["ret_fwd1m"].notna()
    ret = (w * x["ret_fwd1m"]).where(ok, 0).groupby(x["month"]).sum() / w.where(ok, 0).groupby(x["month"]).sum()
    ok1 = x["ret_day1"].notna()
    day1 = (w * x["ret_day1"]).where(ok1, 0).groupby(x["month"]).sum() / w.where(ok1, 0).groupby(x["month"]).sum()
    return ret.sort_index(), day1.sort_index()


def load_data(root, args, splits):
    processed = root / "data" / "processed"
    pred_path = Path(args.predictions) if args.predictions else root / "results" / args.run / "predictions.parquet"
    print("predictions:", pred_path)
    pred = pd.read_parquet(pred_path)
    pred["month"] = pd.to_datetime(pred["month"])
    panel = pd.read_parquet(processed / "panel_model_filing.parquet",
                            columns=["permno", "month", "split", "ret_fwd1m", "mkt_ret", "raw_vol_12m", "raw_beta_12m"])
    panel["month"] = pd.to_datetime(panel["month"])
    # the split of each month as the current panel defines it (older prediction files
    # may call the months after the test period "holdout")
    pred = pred.drop(columns=[c for c in ("split",) if c in pred.columns]).merge(
        panel[["permno", "month", "split"]], on=["permno", "month"], how="inner")
    pred = pred[pred["split"].isin(splits)].copy()   # holdout rows dropped unless --holdout
    if "holdout" not in splits:
        panel = panel[panel["month"] <= pred["month"].max()]   # nothing from the holdout months
    inputs = load_cost_inputs(root)
    if "ret_day1" not in inputs:
        inputs["ret_day1"] = np.nan
    hedge_ret, hedge_day1 = small_cap_index(panel, inputs)
    hedge_ret = hedge_ret[hedge_ret.index <= pred["month"].max()]
    has_day1 = bool(inputs["ret_day1"].notna().any())
    timing = args.trade_at if has_day1 else "close"
    if args.trade_at == "day1" and not has_day1:
        print("WARNING: no first-day returns in cost_inputs.parquet: trading at the month-end close instead.")

    returns_wide = make_returns_wide(panel)          # month-end returns: history for the vol forecast
    pred = add_risk_columns(pred, panel)
    pred = pred.drop(columns=[c for c in ("ret_fwd1m",) if c in pred.columns]).merge(
        panel[["permno", "month", "ret_fwd1m"]], on=["permno", "month"], how="left")
    if "adv_usd_6m" not in inputs:   # cost inputs built before the 6-month average existed
        inputs["adv_usd_6m"] = inputs["adv_usd"]
    if "io_13f" not in inputs:
        inputs["io_13f"] = np.nan
    pred = pred.merge(inputs[["permno", "month", "prc", "size_nyse_pct", "adv_usd", "adv_usd_6m", "io_13f"]],
                      on=["permno", "month"], how="left")
    day1 = inputs.set_index(["permno", "month"])["ret_day1"]
    delayed = add_trade_timing(pred, day1, "day1")["ret_fwd1m"].to_numpy() if has_day1 else None
    pred = add_trade_timing(pred, day1, timing)     # ret_fwd1m = the return earned, ret_close = month-end timing
    pred["ret_day1_timing"] = delayed if delayed is not None else pred["ret_fwd1m"]

    # short-sale constraint: no lendable supply without institutional owners
    short_col = None
    if args.min_short_io > 0 and inputs["io_13f"].notna().any():
        pred["can_short"] = pred["io_13f"].fillna(0.0) >= args.min_short_io
        short_col = "can_short"
        sh = pred.loc[pred["split"] == "val", "can_short"].mean()
        print(f"Short-sale rule: no shorts below {args.min_short_io:.0%} institutional ownership "
              f"({1 - sh:.1%} of validation stock-months cannot be shorted)")
    elif args.min_short_io > 0:
        print("No 13F ownership in cost_inputs.parquet: every stock can be shorted.")

    # the small-cap index (hedge) with the same timing
    h = pd.DataFrame({"permno": HEDGE_ID, "month": hedge_ret.index, "ret_fwd1m": hedge_ret.to_numpy()})
    hd1 = pd.Series(hedge_day1.to_numpy(), index=pd.MultiIndex.from_arrays(
        [np.full(len(hedge_day1), HEDGE_ID), hedge_day1.index]))
    h = add_trade_timing(h, hd1, timing)
    hedge_ret_t = h.set_index("month")["ret_fwd1m"]

    returns_wide[HEDGE_ID] = hedge_ret.reindex(returns_wide.index)

    # the hedge instrument as one more "stock" for the cost model
    months = pd.DatetimeIndex(sorted(inputs["month"].unique()))
    hedge_rows = pd.DataFrame({"permno": HEDGE_ID, "month": months, "is_common": False, "mktcap_usd": np.nan,
                               "size_nyse_pct": np.nan, "size_decile": np.nan,
                               "ret_day1": hedge_day1.reindex(months).to_numpy(), **HEDGE_INPUTS})
    hedge_rows = hedge_rows.dropna(axis=1, how="all")   # all-NA columns are filled by concat anyway
    inputs = pd.concat([inputs, hedge_rows], ignore_index=True)

    model = CostModel(aum=args.decision_aum, impact_k=args.impact_k, spread_mult=args.spread_mult,
                      commission_per_share=args.commission_per_share)
    costs = RealisticCosts(inputs, model)
    market_fwd = add_forward_market_return(pred[["month"]].drop_duplicates(), panel).set_index("month")["mkt_fwd1m"]
    split_of_month = pred.drop_duplicates("month").set_index("month")["split"]

    daily_wide = None
    daily_path = root / "data" / "raw_wrds" / "crsp_daily_returns.parquet"
    if args.vol_method in ("auto", "daily"):
        if daily_path.exists():
            d = pd.read_parquet(daily_path)
            d = d[pd.to_datetime(d["dlycaldt"]) <= pred["month"].max()]
            daily_wide = make_daily_returns_wide(d)
            del d
            print(f"Daily returns: {daily_wide.shape[0]:,} days x {daily_wide.shape[1]:,} stocks")
        elif args.vol_method == "daily":
            sys.exit(f"--vol-method daily needs {daily_path} (src/data/download_wrds_extra.py)")
        else:
            print("No crsp_daily_returns.parquet: the vol target uses the monthly forecast (blend).")
    print(f"Trading at: {'the close of the first day of the next month' if timing == 'day1' else 'the month-end close'}")
    return Context(pred, returns_wide, inputs, costs, market_fwd, hedge_ret_t, split_of_month, has_day1,
                   hedge_ret_close=hedge_ret, daily_wide=daily_wide, short_col=short_col)


# -------- one book --------


def match_legs(w, months):
    """Scale the larger leg down to the size of the smaller one, month by month, so the
    book stays dollar-neutral when caps shrink one side more than the other."""
    longs = w.clip(lower=0).groupby(months).transform("sum")
    shorts = (-w.clip(upper=0)).groupby(months).transform("sum")
    target = np.minimum(longs, shorts)
    f_long = (target / longs.replace(0, np.nan)).fillna(0.0)
    f_short = (target / shorts.replace(0, np.nan)).fillna(0.0)
    return w.where(w <= 0, w * f_long).where(w >= 0, w * f_short)


def make_hook(book, ctx, aum_t=None, leverage=None):
    """Steps applied to the unscaled weights before the vol target: volume caps, hedge."""
    if book.adv_cap is None and not book.hedge_microcap_shorts:
        return None

    def hook(df):
        df = df.copy()
        months = df["month"]
        w = df["weight_unscaled"].astype(float)
        if book.adv_cap is not None:
            adv = df["adv_usd_6m"].astype(float)
            adv = adv.fillna(adv.groupby(months).transform(lambda x: x.quantile(0.10)))
            lev = (leverage.reindex(months).fillna(1.0).clip(lower=0.25).to_numpy()
                   if leverage is not None else 1.0)
            caps = book.adv_cap * adv.to_numpy() / (aum_t.reindex(months).to_numpy() * lev)
            w = cap_weights_by_name(w, months, pd.Series(caps, index=df.index))
            if not book.hedge_microcap_shorts:
                w = match_legs(w, months)
        hedge = None
        if book.hedge_microcap_shorts:
            micro = ~(df["size_nyse_pct"] >= MICROCAP)            # unknown size counts as micro
            w = w.where(~(micro & (w < 0)), 0.0)
            micro_long = w.where(micro & (w > 0), 0.0).groupby(months).transform("sum")
            longs = w.clip(lower=0).groupby(months).transform("sum")
            shorts = (-w.clip(upper=0)).groupby(months).transform("sum")
            # single-name shorts cover at most (longs - microcap longs); only scaled down, so
            # volume caps still hold; the index short covers the rest
            scale = ((longs - micro_long) / shorts.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
            w = w.where(w >= 0, w * scale)
            covered = (-w.clip(upper=0)).groupby(months).transform("sum")
            m = (longs - covered).groupby(months).first()
            m = m[m > 1e-12]
            hedge = pd.DataFrame({"month": m.index, "permno": HEDGE_ID, "weight_unscaled": -m.to_numpy(),
                                  "ret_fwd1m": ctx.hedge_ret.reindex(m.index).fillna(0.0).to_numpy(),
                                  "ret_close": ctx.hedge_ret_close.reindex(m.index).fillna(0.0).to_numpy(),
                                  "ret_day1_timing": ctx.hedge_ret.reindex(m.index).fillna(0.0).to_numpy(),
                                  "position": -1,
                                  "split": ctx.split_of_month.reindex(m.index).to_numpy()})
        df["weight_unscaled"] = w
        df["position"] = np.sign(w).astype("int8")
        if hedge is not None and len(hedge):
            df = pd.concat([df, hedge], ignore_index=True).sort_values(["month", "permno"]).reset_index(drop=True)
        return df

    return hook


@dataclass
class BuiltBook:
    book: Book
    cfg: PortfolioConfig
    aum: float | None
    weights_df: pd.DataFrame
    diag: pd.DataFrame
    gross: pd.Series
    turnover: pd.Series
    exposure: pd.Series
    W: pd.DataFrame
    pre: pd.DataFrame
    extra: dict = field(default_factory=dict)


def build_book(book, ctx, aum=None, cfg=None):
    """Weights of one book (at fund size `aum` when it has volume caps). cfg: reuse a
    calibrated config (vol_scale) instead of calibrating again."""
    oos_f = apply_filters(ctx.pred, book)
    val_f = oos_f[oos_f["split"] == "val"]
    aum_t = None
    if book.adv_cap is not None:
        months = pd.DatetimeIndex(sorted(oos_f["month"].unique()))
        aum_t = ctx.costs.with_model(aum=aum).aum_by_month(months)
    hook = make_hook(book, ctx, aum_t)
    if cfg is None:
        cfg = calibrate_vol_scale(val_f, book.cfg, ctx.returns_wide, FLAT_COST, post_unscaled=hook,
                                  daily_wide=ctx.daily_wide)
    df, diag = build_portfolio(oos_f, cfg, returns_wide=ctx.returns_wide, post_unscaled=hook,
                               daily_wide=ctx.daily_wide)
    if book.adv_cap is not None and diag is not None:
        # second pass: dollar caps use the leverage the vol target actually applies
        hook = make_hook(book, ctx, aum_t, leverage=diag["leverage"])
        df, diag = build_portfolio(oos_f, cfg, returns_wide=ctx.returns_wide, post_unscaled=hook,
                                   daily_wide=ctx.daily_wide)
    if book.partial_kappa:
        df = partial_trade(df, book.partial_kappa, ctx, aum, diag)
    if book.trade_band is not None:
        df = apply_no_trade_band(df, book.trade_band, leverage=diag["leverage"] if diag is not None else None)
    if book.adv_cap is not None:
        # final hard limit on the positions actually held (after leverage and the band):
        # clip, then shrink the larger leg so the book stays dollar-neutral. When the caps
        # bind the book is smaller and its volatility falls below the target: that is the
        # capacity limit.
        adv = df["adv_usd_6m"].astype(float)
        adv = adv.fillna(adv.groupby(df["month"]).transform(lambda x: x.quantile(0.10)))
        cap = book.adv_cap * adv / df["month"].map(aum_t)
        stock = df["permno"] != HEDGE_ID
        w = df["weight"].astype(float)
        df["weight"] = w.where(~stock, np.sign(w) * np.minimum(w.abs(), cap.fillna(0.0)))
    if book.adv_cap is not None or book.trade_band is not None:
        # positions left untouched by the band drift (winners up, losers down): re-balance the legs
        df["weight"] = match_legs(df["weight"].astype(float), df["month"])
    m = portfolio_returns_and_turnover(df, weight_col="weight", return_matrices=True)
    gross = m["gross_return"].sort_index()
    return BuiltBook(book, cfg, aum, df, diag, gross, m["turnover"].reindex(gross.index).fillna(0),
                     m["gross_exposure"], m["weights"], m["pre_trade"])


def partial_trade(df, kappa, ctx, aum, diag):
    """Cost-aware trading on top of a built book (book K): unit costs at fund size aum."""
    months = pd.DatetimeIndex(sorted(df["month"].unique()))
    permnos = pd.Index(sorted(df["permno"].unique()))
    fn = ctx.costs.with_model(aum=aum or ctx.costs.model.aum).unit_cost_fn(months, permnos)
    return apply_partial_trading(df, kappa, fn, leverage=diag["leverage"] if diag is not None else None,
                                 short_allowed_col=ctx.short_col if ctx.short_col in df.columns else None)


def result(bb, cost_model=None, ret_col=None):
    """The dict summarize_backtest expects, for the flat cost (None) or a cost model.
    ret_col: evaluate the same weights with another return column (the delay test)."""
    gross, turnover, exposure, W, pre = bb.gross, bb.turnover, bb.exposure, bb.W, bb.pre
    if ret_col is not None:
        df = bb.weights_df.assign(ret_fwd1m=bb.weights_df[ret_col])
        m = portfolio_returns_and_turnover(df, weight_col="weight", return_matrices=True)
        gross = m["gross_return"].sort_index()
        turnover, exposure, W, pre = m["turnover"].reindex(gross.index).fillna(0), m["gross_exposure"], m["weights"], m["pre_trade"]
    if cost_model is None:
        costs, breakdown = FLAT_COST * turnover, None
    else:
        breakdown = cost_model.monthly_costs(W, pre).reindex(gross.index)
        costs = breakdown["total"].fillna(0)
    return {"returns_no_fees": gross, "returns_with_fees": gross - costs,
            "turnover": turnover, "gross_exposure": exposure, "cost_breakdown": breakdown}


def summary_row(bb, res, months, market_fwd, band):
    row = summarize_backtest(res, months, market_fwd, band=band)
    d = bb.diag.reindex(months)
    row["avg_leverage"] = d["leverage"].mean()
    W = bb.W.reindex(months)
    row["avg_names"] = (W.drop(columns=[HEDGE_ID], errors="ignore") != 0).sum(axis=1).mean()
    row["avg_hedge_weight"] = W[HEDGE_ID].mean() if HEDGE_ID in W.columns else 0.0
    return row


# -------- where the costs come from --------


def cost_breakdown_tables(bb, cost_model, inputs, split_months):
    s = cost_model.stock_costs(bb.W, bb.pre)
    s["traded"] = s["trade"].abs()
    s["trading_cost"] = s["spread_cost"] + s["impact_cost"]
    df = bb.weights_df[["month", "permno", "raw_vol_12m", "weight", "ret_fwd1m"]].copy()
    stocks = df[df["permno"] != HEDGE_ID].copy()
    stocks["vol_group"] = vol_bucket(stocks, 5, "raw_vol_12m") + 1          # 1 = calmest
    s = s.merge(stocks[["month", "permno", "vol_group"]], on=["month", "permno"], how="left")
    s["vol_group"] = s["vol_group"].fillna(0).astype(int)                    # 0 = left the universe
    size = inputs.loc[inputs["permno"] != HEDGE_ID, ["month", "permno", "size_decile"]]
    s = s.merge(size, on=["month", "permno"], how="left")
    s["size_decile"] = s["size_decile"].fillna(0).astype(int)                # 1 = smallest, 0 = unknown
    s["side"] = s["side"].map({1: "long", -1: "short"}).fillna("none")
    is_hedge = s["permno"] == HEDGE_ID
    s.loc[is_hedge, ["vol_group", "size_decile"]] = -1                       # -1 = the index hedge
    s.loc[is_hedge, "side"] = "hedge"

    tables = []
    for split_name, months in split_months.items():
        ss = s[s["month"].isin(months)]
        n_years = len(months) / 12
        for key in ("vol_group", "size_decile", "side"):
            g = ss.groupby(key)
            t = pd.DataFrame({
                "share_of_traded": g["traded"].sum() / ss["traded"].sum(),
                "share_of_cost": (g["trading_cost"].sum() + g["borrow_cost"].sum())
                / (ss["trading_cost"].sum() + ss["borrow_cost"].sum()),
                "trading_cost_bps_per_dollar": 1e4 * g["trading_cost"].sum() / g["traded"].sum(),
                "spread_bps_per_dollar": 1e4 * g["spread_cost"].sum() / g["traded"].sum(),
                "impact_bps_per_dollar": 1e4 * g["impact_cost"].sum() / g["traded"].sum(),
                "cost_per_year": (g["trading_cost"].sum() + g["borrow_cost"].sum()) / n_years,
                "median_participation_of_adv": g["participation"].median(),
            }).reset_index().rename(columns={key: "group"})
            t.insert(0, "by", key)
            t.insert(0, "split", split_name)
            tables.append(t)
        # profit by side: gross contribution, costs, net (per year, fraction of capital)
        rows = df[df["month"].isin(months)]
        contrib = rows["weight"] * rows["ret_fwd1m"].fillna(0)
        hedge = rows["permno"] == HEDGE_ID
        for side, mask in (("long", (rows["weight"] > 0) & ~hedge), ("short", (rows["weight"] < 0) & ~hedge),
                           ("hedge", hedge)):
            c = ss[ss["side"] == side]
            if not mask.any() and c.empty:
                continue
            gross_y = contrib[mask].sum() / n_years
            cost_y = (c["trading_cost"].sum() + c["borrow_cost"].sum()) / n_years
            tables.append(pd.DataFrame([{
                "split": split_name, "by": "profit_by_side", "group": side,
                "gross_return_per_year": gross_y, "cost_per_year": cost_y,
                "net_return_per_year": gross_y - cost_y,
                "borrow_per_year": c["borrow_cost"].sum() / n_years,
            }]))
    return pd.concat(tables, ignore_index=True)


# -------- figures --------


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which="both", colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figures(curve, breakdown, out_dir, books, decision_aum, flat_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    money = FuncFormatter(lambda x, _: f"${x / 1e9:g}B" if x >= 1e9 else f"${x / 1e6:g}M")

    # 1. Capacity curve: net Sharpe vs fund size, validation and test side by side
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, split_name in zip(axes, ("val", "test")):
        _style(ax)
        shown = [b for b in books if b.letter in PLOT_BOOKS]
        for color, book in zip(SERIES, shown):
            c = curve[(curve["book"] == book.name) & (curve["split"] == split_name)].sort_values("aum")
            ax.plot(c["aum"], c["sharpe_net"], color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=book.name)
            ax.annotate(book.letter, (c["aum"].iloc[-1], c["sharpe_net"].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)
        flat = flat_rows[(flat_rows["split"] == split_name) & (flat_rows["book"] == books[0].name)]
        if len(flat):
            ax.axhline(flat["sharpe_net"].iloc[0], color=INK_2, linewidth=1, linestyle=":")
            ax.text(curve["aum"].min(), flat["sharpe_net"].iloc[0], " A at flat 10 bps",
                    color=INK_2, fontsize=8, va="bottom")
        ax.axvline(decision_aum, color=GRID, linewidth=1.2)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(money)
        ax.set_xlabel("fund size (end-2024 dollars)", color=INK_2, fontsize=9)
        ax.set_title("Validation 2005-14" if split_name == "val" else "Test 2015-24",
                     color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("net Sharpe, realistic costs", color=INK_2, fontsize=9)
    axes[0].axhline(0, color=INK_2, linewidth=0.8)
    axes[1].axhline(0, color=INK_2, linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(fig_dir / "capacity_curve.png", dpi=200)
    plt.close(fig)

    # 2. Trading cost per dollar traded by NYSE size decile (book A, test), spread vs impact
    b = breakdown[(breakdown["split"] == "test") & (breakdown["by"] == "size_decile")].copy()
    b = b[b["group"].astype(int) > 0]          # drop "unknown" (0) and the hedge (-1)
    if len(b):
        b["d"] = b["group"].astype(int)
        b = b.sort_values("d")
        fig, ax = plt.subplots(figsize=(9, 4))
        _style(ax)
        x = np.arange(len(b))
        ax.bar(x, b["spread_bps_per_dollar"], width=0.7, color=SERIES[0], label="commission + half spread",
               edgecolor=SURFACE, linewidth=2)
        ax.bar(x, b["impact_bps_per_dollar"], width=0.7, bottom=b["spread_bps_per_dollar"], color=SERIES[1],
               label=f"market impact at {money(decision_aum, None)}", edgecolor=SURFACE, linewidth=2)
        for xi, tot, share in zip(x, b["trading_cost_bps_per_dollar"], b["share_of_traded"]):
            ax.text(xi, tot, f"{tot:.0f}\n{share:.0%} of $", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(x, [f"{d}" for d in b["d"]])
        ax.set_xlabel("NYSE size decile (1 = smallest)", color=INK_2, fontsize=9)
        ax.set_ylabel("bps per dollar traded", color=INK_2, fontsize=9)
        ax.set_title("Book A, test: trading cost by company size (label: cost, share of dollars traded)",
                     color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
        ax.margins(y=0.2)
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(fig_dir / "cost_by_size_decile.png", dpi=200)
        plt.close(fig)


# -------- main --------


def vol_tracking_error(bb, months, target):
    """RMSE of the rolling 12-month realised vol (gross returns) against the target."""
    r = bb.gross.reindex(months)
    roll = r.rolling(12).std() * np.sqrt(12)
    return float(np.sqrt(((roll.dropna() - target) ** 2).mean()))


def choose_vol_method(ctx, args, split_months):
    """Monthly 'blend' (v4) or 'daily' forecast: the one whose rolling 12-month vol stays
    closer to the target on validation (book F, the flat cost: weights only)."""
    if ctx.daily_wide is None or args.vol_method == "blend":
        return "blend", None
    if args.vol_method == "daily":
        return "daily", None
    rows = []
    for method in ("blend", "daily"):
        book = next(b for b in make_books(args.target, tuple(args.band), ctx.short_col, method) if b.letter == "F")
        bb = build_book(book, ctx, aum=args.decision_aum)
        row = {"method": method, "vol_scale": bb.cfg.vol_scale}
        for s, m in split_months.items():
            row[f"rmse_rolling_vol_{s}"] = vol_tracking_error(bb, m, args.target)
            row[f"realised_vol_{s}"] = float(bb.gross.reindex(m).std() * np.sqrt(12))
        rows.append(row)
    t = pd.DataFrame(rows)
    chosen = t.loc[t["rmse_rolling_vol_val"].idxmin(), "method"]
    return chosen, t


def choose_kappa(ctx, args, split_months, f_book):
    """Speed of cost-aware trading (book K), chosen on validation at the decision size."""
    bb_f = build_book(f_book, ctx, aum=args.decision_aum)
    model = ctx.costs.with_model(aum=args.decision_aum)
    rows = []
    for kappa in KAPPA_GRID:
        if kappa == 0:
            bb = bb_f
        else:
            df = partial_trade(bb_f.weights_df, kappa, ctx, args.decision_aum, bb_f.diag)
            m = portfolio_returns_and_turnover(df, weight_col="weight", return_matrices=True)
            gross = m["gross_return"].sort_index()
            bb = BuiltBook(f_book, bb_f.cfg, args.decision_aum, df, bb_f.diag, gross,
                           m["turnover"].reindex(gross.index).fillna(0), m["gross_exposure"], m["weights"],
                           m["pre_trade"])
        res = result(bb, model)
        row = {"kappa": kappa}
        for s, months in split_months.items():
            x = summarize_backtest(res, months, ctx.market_fwd, band=tuple(args.band))
            row.update({f"sharpe_net_{s}": x["sharpe_net"], f"turnover_{s}": x["turnover_per_month"],
                        f"cost_per_year_{s}": x["cost_total_per_year"]})
        rows.append(row)
    grid = pd.DataFrame(rows)
    return float(grid.loc[grid["sharpe_net_val"].idxmax(), "kappa"]), grid


def run_study(args, root, out_dir):
    ctx = load_data(root, args, list(REPORT_SPLITS))
    assert not (ctx.pred["split"] == "holdout").any()
    split_months = {s: pd.DatetimeIndex(sorted(ctx.pred.loc[ctx.pred["split"] == s, "month"].unique()))
                    for s in REPORT_SPLITS if (ctx.pred["split"] == s).any()}
    band = tuple(args.band)
    costs, base = ctx.costs, ctx.costs.model

    # ---- settings chosen on validation: vol forecast, then the cost-aware trading speed ----
    vol_method, vol_table = choose_vol_method(ctx, args, split_months)
    if vol_table is not None:
        vol_table.to_csv(out_dir / "vol_method_choice.csv", index=False)
        print("\nVol forecast (book F, validation decides):")
        print(vol_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Vol forecast used: {vol_method}")
    f_book = next(b for b in make_books(args.target, band, ctx.short_col, vol_method) if b.letter == "F")
    kappa, kgrid = choose_kappa(ctx, args, split_months, f_book)
    kgrid.to_csv(out_dir / "partial_trading_grid.csv", index=False)
    print("\nCost-aware trading (book K): net Sharpe at the decision size by speed parameter kappa")
    print(kgrid.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"kappa chosen on validation: {kappa:g}" + ("  (0 = K is F)" if kappa == 0 else ""))
    if kappa == KAPPA_GRID[-1]:
        print("WARNING: kappa is the largest value tried: slower trading might be better still.")

    books = make_books(args.target, band, ctx.short_col, vol_method, kappa)
    sens = {
        "default": {},
        "impact x 0.5": {"impact_k": 0.5 * base.impact_k},
        "impact x 2": {"impact_k": 2.0 * base.impact_k},
        "half the spread": {"spread_mult": 0.5 * base.spread_mult},
        "patient execution (half spread, half impact)": {"spread_mult": 0.5 * base.spread_mult,
                                                         "impact_k": 0.5 * base.impact_k},
        "borrow fees x 2": {"borrow_fee_htb": 2 * base.borrow_fee_htb,
                            "borrow_fee_special": 2 * base.borrow_fee_special},
        "no AUM scaling": {"scale_aum_with_market": False},
    }
    aums = sorted(set(TABLE_AUMS) | set(CURVE_AUMS) | {args.decision_aum})
    alt_timing = "close" if args.trade_at == "day1" else "day1"
    alt_label = {"close": "month-end close (no delay)", "day1": "one day later"}

    rows, curve, sens_rows, delay_rows, ret_rows, built = [], [], [], [], [], {}
    for book in books:
        t0 = time.time()
        # the book at the decision size (calibrated there); books whose weights depend on the
        # fund size (volume caps, cost-aware trading) are rebuilt per size
        bb_dec = build_book(book, ctx, aum=args.decision_aum)
        if book is books[0]:            # kept for the cost breakdown; the chosen book is rebuilt below
            built[book.name] = bb_dec
        size_dependent = book.adv_cap is not None or bool(book.partial_kappa)
        for a in [None] + aums:
            if a is None:
                bb, model, label = bb_dec, None, "flat 10 bps"
            else:
                bb = bb_dec if (not size_dependent or a == args.decision_aum) else build_book(book, ctx, a, bb_dec.cfg)
                model, label = costs.with_model(aum=a), f"${a / 1e6:,.0f}M"
            res = result(bb, model)
            if a == args.decision_aum or a is None:
                ret_rows.append(pd.DataFrame({"book": book.name, "costs": label, "month": res["returns_with_fees"].index,
                                              "gross": res["returns_no_fees"].to_numpy(),
                                              "net": res["returns_with_fees"].to_numpy()}))
            for split_name, months in split_months.items():
                row = summary_row(bb, res, months, ctx.market_fwd, band)
                full = {"book": book.name, "costs": label, "aum": a if a is not None else np.nan,
                        "split": split_name, **row}
                if model is not None:
                    curve.append(full)
                if model is None or a in TABLE_AUMS or a == args.decision_aum:
                    rows.append(full)
        for label, change in sens.items():
            res = result(bb_dec, costs.with_model(aum=args.decision_aum, **change))
            for split_name, months in split_months.items():
                r = summarize_backtest(res, months, ctx.market_fwd, band=band)
                sens_rows.append({"book": book.name, "scenario": label, "split": split_name,
                                  "sharpe_net": r["sharpe_net"], "ann_return_net": r["ann_return_net"],
                                  "cost_total_per_year": r["cost_total_per_year"]})
        if ctx.has_day1:
            dec_model = costs.with_model(aum=args.decision_aum)
            for label, r in ((alt_label[args.trade_at] + " (headline)", result(bb_dec, dec_model)),
                             (alt_label[alt_timing], result(bb_dec, dec_model, ret_col="ret_close"
                                                            if alt_timing == "close" else "ret_day1_timing"))):
                for split_name, months in split_months.items():
                    x = summarize_backtest(r, months, ctx.market_fwd, band=band)
                    delay_rows.append({"book": book.name, "trade_at": label, "split": split_name,
                                       "ann_return_gross": x["ann_return_gross"], "ann_return_net": x["ann_return_net"],
                                       "sharpe_net": x["sharpe_net"]})
        print(f"{book.name}: {time.time() - t0:.0f}s (vol_scale {bb_dec.cfg.vol_scale:.3f}, "
              f"{bb_dec.weights_df['permno'].ne(HEDGE_ID).sum():,} stock-months in its universe)", flush=True)

    table, curve, sens_df = pd.DataFrame(rows), pd.DataFrame(curve), pd.DataFrame(sens_rows)
    table.to_csv(out_dir / "cost_study.csv", index=False)
    curve.to_csv(out_dir / "capacity_curve.csv", index=False)
    split_col = pd.Series(ctx.split_of_month)
    book_returns = pd.concat(ret_rows, ignore_index=True)
    book_returns["split"] = book_returns["month"].map(split_col)
    book_returns.to_csv(out_dir / "book_returns.csv", index=False)
    delay = pd.DataFrame(delay_rows)
    if len(delay):
        delay.to_csv(out_dir / "delay_test.csv", index=False)

    # ---- choice, on validation only ----
    dec_label = f"${args.decision_aum / 1e6:,.0f}M"
    dec = curve[(curve["aum"] == args.decision_aum) & (curve["split"] == "val")].set_index("book")
    chosen_name = dec["sharpe_net"].idxmax()
    chosen = next(b for b in books if b.name == chosen_name)
    if chosen_name not in built:    # books are not all kept in memory: rebuild the chosen one
        built[chosen_name] = build_book(chosen, ctx, aum=args.decision_aum)

    # ---- sensitivities that change the book itself (chosen book only) ----
    if ctx.short_col is not None:
        for label, thr in (("no short-sale limit", 0.0), ("no shorts below 10% institutional ownership", 0.10)):
            saved = ctx.pred["can_short"].copy()
            ctx.pred["can_short"] = ctx.pred["io_13f"].fillna(0.0) >= thr
            bb = build_book(chosen, ctx, aum=args.decision_aum)
            ctx.pred["can_short"] = saved
            res = result(bb, costs.with_model(aum=args.decision_aum))
            for split_name, months in split_months.items():
                r = summarize_backtest(res, months, ctx.market_fwd, band=band)
                sens_rows.append({"book": chosen.name, "scenario": label, "split": split_name,
                                  "sharpe_net": r["sharpe_net"], "ann_return_net": r["ann_return_net"],
                                  "cost_total_per_year": r["cost_total_per_year"]})
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(out_dir / "sensitivity.csv", index=False)

    # ---- breakdowns for A and the chosen book ----
    bd_tables = []
    for name in dict.fromkeys([books[0].name, chosen_name]):
        t = cost_breakdown_tables(built[name], costs.with_model(aum=args.decision_aum), ctx.inputs, split_months)
        t.insert(0, "book", name)
        bd_tables.append(t)
    breakdown = pd.concat(bd_tables, ignore_index=True)
    breakdown.to_csv(out_dir / f"cost_breakdown_{dec_label.replace('$', '').replace(',', '')}.csv", index=False)

    # ---- print ----
    cols = ["ann_return_gross", "ann_return_net", "ann_vol_net", "sharpe_net", "max_drawdown_net",
            "turnover_per_month", "avg_trading_cost_bps", "cost_total_per_year", "avg_names", "avg_hedge_weight"]
    fmt = lambda x: f"{x:8.3f}"  # noqa: E731
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        for split_name in split_months:
            t = table[table["split"] == split_name].set_index(["book", "costs"])
            print(f"\n{split_name.upper()}: net of costs (realistic = commission + half spread + impact + borrow)")
            print(t[[c for c in cols if c in t]].to_string(float_format=fmt))
        print(f"\nSensitivity at {dec_label}: net Sharpe")
        print(sens_df.pivot_table(index="book", columns=["split", "scenario"], values="sharpe_net")
              .to_string(float_format=lambda x: f"{x:6.2f}"))
        if len(delay):
            print(f"\nTrading time at {dec_label}: net Sharpe")
            print(delay.pivot_table(index="book", columns=["split", "trade_at"], values="sharpe_net")
                  .to_string(float_format=lambda x: f"{x:7.2f}"))
        for name in dict.fromkeys([books[0].name, chosen_name]):
            a = breakdown[(breakdown["book"] == name) & (breakdown["split"] == "test")]
            print(f"\n{name}, test, {dec_label}: where the costs come from "
                  "(vol group 1 = calmest, 0 = left the universe; size decile 1 = smallest, 0 = unknown; -1 = index hedge)")
            for key in ("vol_group", "size_decile", "side"):
                t = a[a["by"] == key].copy()
                if key != "side":
                    t = t.assign(k=t["group"].astype(int)).sort_values("k")
                print(t[["by", "group", "share_of_traded", "share_of_cost", "trading_cost_bps_per_dollar",
                         "spread_bps_per_dollar", "impact_bps_per_dollar", "median_participation_of_adv"]]
                      .to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
        p = breakdown[(breakdown["by"] == "profit_by_side")]
        print(f"\nProfit by side at {dec_label} (fraction of capital per year):")
        print(p[["book", "split", "group", "gross_return_per_year", "cost_per_year", "borrow_per_year",
                 "net_return_per_year"]].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    if dec["sharpe_net"].max() <= 0:
        pos = curve[(curve["split"] == "val") & (curve["sharpe_net"] > 0)]
        hint = (f"; the largest fund size with a positive validation Sharpe is ${pos['aum'].max() / 1e6:,.0f}M"
                if len(pos) else "")
        print(f"\nWARNING: no book has a positive validation net Sharpe at {dec_label}{hint}. "
              "Choose the fund size (--decision-aum) before looking at the holdout.")
    test_row = curve[(curve["book"] == chosen_name) & (curve["aum"] == args.decision_aum)
                     & (curve["split"] == "test")].iloc[0]
    val_row = dec.loc[chosen_name]
    print(f"\nChoice on validation at {dec_label}: {chosen_name} "
          f"(validation net Sharpe {val_row['sharpe_net']:.2f}; test, for information, {test_row['sharpe_net']:.2f})")
    ranking = dec["sharpe_net"].sort_values(ascending=False)
    print("Validation ranking: " + ", ".join(f"{n.split('.')[0]} {v:.2f}" for n, v in ranking.items()))

    # ---- pre-registration for the holdout ----
    sr = float(test_row["sharpe_net"])
    se_1y = float(np.sqrt(1 + sr ** 2 / 2))
    plan = {
        "written": time.strftime("%Y-%m-%d %H:%M"),
        "run": args.run, "panel": "filing",
        "book": chosen.to_json(),
        "vol_scale_calibrated": built[chosen_name].cfg.vol_scale,
        "trade_at": args.trade_at, "min_short_io": args.min_short_io, "vol_method": vol_method,
        "cost_model": asdict(costs.with_model(aum=args.decision_aum).model),
        "selection_rule": f"highest validation net Sharpe at {dec_label}, realistic costs, among {len(books)} books",
        "fingerprint": {"val_sharpe_net": float(val_row["sharpe_net"]), "test_sharpe_net": sr,
                        "val_ann_return_net": float(val_row["ann_return_net"]),
                        "test_ann_return_net": float(test_row["ann_return_net"])},
        "holdout_checks": {
            "sharpe_90pct_interval": [sr - 1.645 * se_1y, sr + 1.645 * se_1y],
            "sharpe_interval_note": "test net Sharpe +/- 1.645 x sqrt(1 + SR^2/2): the spread of a "
                                    "12-month Sharpe if the true Sharpe equals the test one",
            "ann_vol_range": [0.07, 0.13],
            "max_drawdown_not_worse_than": float(test_row["max_drawdown_net"]),
        },
    }
    if (out_dir / "holdout_result.json").exists():
        # the record of the choice made before the holdout is never overwritten
        print("\nThe holdout has been opened: chosen_book.json (the choice fixed before it) is kept as it is.")
    else:
        with open(out_dir / "chosen_book.json", "w") as f:
            json.dump(plan, f, indent=4)
        lo, hi = plan["holdout_checks"]["sharpe_90pct_interval"]
        print(f"Written to chosen_book.json, with the holdout checks fixed in advance: net Sharpe in "
              f"[{lo:.2f}, {hi:.2f}], vol in 7-13%, drawdown no worse than {test_row['max_drawdown_net']:.1%}.")
        print("If you change anything (books, costs, fund size), run this again before the holdout.")

    if not args.no_plots:
        flat_rows = table[table["costs"] == "flat 10 bps"]
        save_figures(curve, breakdown[breakdown["book"] == books[0].name], out_dir, books,
                     args.decision_aum, flat_rows)
    print("\nSaved to", out_dir)


def run_holdout(args, root, out_dir):
    plan_path = out_dir / "chosen_book.json"
    if not plan_path.exists():
        sys.exit("No chosen_book.json: run  python src/evaluation/cost_study.py  first.")
    done = out_dir / "holdout_result.json"
    if done.exists():
        sys.exit(f"The holdout was already opened ({done}). Reporting it again after changes would turn it "
                 "into a validation set. Delete that file only if you accept that.")
    plan = json.loads(plan_path.read_text())
    book = Book.from_json(plan["book"])
    model = CostModel(**plan["cost_model"])
    args.decision_aum = model.aum
    args.trade_at = plan.get("trade_at", args.trade_at)
    args.min_short_io = plan.get("min_short_io", args.min_short_io)
    args.vol_method = plan.get("vol_method", "blend")

    print("=" * 78 + "\nHOLDOUT RUN: one book, frozen settings, evaluated once.\n" + "=" * 78)
    ctx = load_data(root, args, [*REPORT_SPLITS, "holdout"])
    if not (ctx.pred["split"] == "holdout").any():
        sys.exit("No holdout rows in the predictions yet (the holdout starts with the months after "
                 "November 2025: download CRSP again once they are available, then rebuild and predict).")
    split_months = {s: pd.DatetimeIndex(sorted(ctx.pred.loc[ctx.pred["split"] == s, "month"].unique()))
                    for s in (*REPORT_SPLITS, "holdout") if (ctx.pred["split"] == s).any()}
    costs = ctx.costs.with_model(model)
    bb = build_book(book, ctx, aum=model.aum)
    res = result(bb, costs)

    # nothing may have changed since the choice: validation and test must reproduce
    fp = plan["fingerprint"]
    now = {s: summarize_backtest(res, split_months[s], ctx.market_fwd, band=tuple(args.band)) for s in ("val", "test")}
    diffs = {"val_sharpe_net": now["val"]["sharpe_net"] - fp["val_sharpe_net"],
             "test_sharpe_net": now["test"]["sharpe_net"] - fp["test_sharpe_net"]}
    if max(abs(v) for v in diffs.values()) > 1e-6 and not args.accept_changes:
        sys.exit(f"Validation / test numbers differ from chosen_book.json ({diffs}): something changed after "
                 "the choice. Re-run the study, or pass --accept-changes if you know why.")

    rows = []
    for split_name, months in split_months.items():
        for label, r in (("realistic", res), ("flat 10 bps", result(bb, None))):
            rows.append({"split": split_name, "costs": label,
                         **summary_row(bb, r, months, ctx.market_fwd, tuple(args.band))})
    table = pd.DataFrame(rows)
    h = table[(table["split"] == "holdout") & (table["costs"] == "realistic")].iloc[0]
    checks = plan["holdout_checks"]
    lo, hi = checks["sharpe_90pct_interval"]
    outcome = {
        "sharpe_in_interval": bool(lo <= h["sharpe_net"] <= hi),
        "vol_in_range": bool(checks["ann_vol_range"][0] <= h["ann_vol_net"] <= checks["ann_vol_range"][1]),
        "drawdown_ok": bool(h["max_drawdown_net"] >= checks["max_drawdown_not_worse_than"]),
    }
    table.to_csv(out_dir / "holdout_table.csv", index=False)
    monthly = pd.DataFrame({"gross": res["returns_no_fees"], "net": res["returns_with_fees"],
                            "leverage": bb.diag["leverage"]}).reindex(split_months["holdout"])
    monthly.to_csv(out_dir / "holdout_monthly.csv")
    with open(done, "w") as f:
        json.dump({"opened": time.strftime("%Y-%m-%d %H:%M"), "book": book.name,
                   "holdout": {k: float(v) for k, v in h.items()
                               if isinstance(v, (int, float, np.integer, np.floating))},
                   "checks": outcome}, f, indent=4)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.set_index(["split", "costs"])[["ann_return_net", "ann_vol_net", "sharpe_net",
                                                    "max_drawdown_net", "worst_month_net", "n_months"]]
              .to_string(float_format=lambda x: f"{x:8.3f}"))
    print(f"\nPre-registered checks: Sharpe in [{lo:.2f}, {hi:.2f}]: {outcome['sharpe_in_interval']}; "
          f"vol in 7-13%: {outcome['vol_in_range']}; drawdown no worse than "
          f"{checks['max_drawdown_not_worse_than']:.1%}: {outcome['drawdown_ok']}")
    print("Saved to", out_dir, "- report these numbers as they are.")


def main():
    parser = argparse.ArgumentParser(description="Realistic trading costs for the recommended book.")
    parser.add_argument("--run", default="wf_full", help="results/<run>/predictions.parquet")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--target", type=float, default=0.10)
    parser.add_argument("--band", type=float, nargs=2, default=[0.09, 0.11])
    parser.add_argument("--decision-aum", type=float, default=1e8, help="fund size for the choice ($)")
    parser.add_argument("--impact-k", type=float, default=0.5)
    parser.add_argument("--spread-mult", type=float, default=1.0)
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--trade-at", choices=["day1", "close"], default="day1",
                        help="day1: close of the first trading day of the next month (default); "
                             "close: the month-end close the inputs are measured at")
    parser.add_argument("--min-short-io", type=float, default=0.05,
                        help="no shorts in stocks with less 13F institutional ownership (0 = no limit)")
    parser.add_argument("--vol-method", choices=["auto", "blend", "daily"], default="auto")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="evaluate the frozen choice on the holdout, ONCE")
    parser.add_argument("--accept-changes", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    run_name = Path(args.predictions).parent.name if args.predictions else args.run
    out_dir = root / "results" / f"cost_study_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.holdout:
        run_holdout(args, root, out_dir)
    else:
        run_study(args, root, out_dir)


if __name__ == "__main__":
    main()
