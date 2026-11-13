"""
Backtest engine and shared evaluation helpers: monthly book returns, drift-aware turnover
and costs from per-stock weights, rank IC, decile returns, and the equal-weight decile
backtests of mlp_model.py --diagnostics. portfolio.py and cost_study.py build their
weights and hand them to the same engine.
"""

import numpy as np
import pandas as pd


# -------- core stats --------


def compute_stats(returns):
    ann_return = 12 * returns.mean()
    ann_vol = np.sqrt(12) * returns.std()

    if ann_vol == 0:
        sharpe = np.nan
    else:
        sharpe = ann_return / ann_vol

    pnl = (1 + returns).cumprod() - 1

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "pnl": pnl,
    }


# -------- the long-short backtest engine --------
#
# Turnover is measured against the previous book *after* it has drifted with one month of
# returns, and positions that disappear from the panel (delisting, missing next-month
# return) are charged for being closed.
#
# Notation: the row (i, t) holds the weight w[i,t] chosen at the end of month t and
# ret_fwd1m = r[i,t+1]. The book return indexed t is R[t] = sum_i w[i,t] * r[i,t+1].
# Just before rebalancing at t+1, stock i weighs w[i,t] * (1 + r[i,t+1]) / (1 + R[t]).
# turnover[t+1] = sum_i |w[i,t+1] - that drifted weight|  (first month: sum_i |w[i,t]|).


def portfolio_returns_and_turnover(data, weight_col="weight", date_col="month",
                                   id_col="permno", return_col="ret_fwd1m", return_matrices=False):
    """Gross monthly book return, turnover and exposures from per-row weights.

    data must have one row per (date, id). Returns a dict of monthly Series indexed
    by date: gross_return, turnover, gross_exposure (sum |w|), net_exposure (sum w).
    With return_matrices=True it also returns the date x id matrices `weights` (after
    trading) and `pre_trade` (the drifted weights just before trading), which the
    per-stock cost model (evaluation/costs.py) needs.
    """
    months = pd.DatetimeIndex(sorted(pd.to_datetime(data[date_col]).unique()))
    held = data.loc[data[weight_col].fillna(0) != 0, [date_col, id_col, weight_col, return_col]]
    held = held.assign(**{date_col: pd.to_datetime(held[date_col])})

    W = held.set_index([date_col, id_col])[weight_col].unstack().reindex(months).fillna(0.0)
    R = held.set_index([date_col, id_col])[return_col].unstack().reindex(months)
    R = R.reindex(columns=W.columns).fillna(0.0)  # a held stock with no return earns 0

    gross_return = (W * R).sum(axis=1)
    drifted = (W * (1.0 + R)).div(1.0 + gross_return, axis=0)
    pre_trade = drifted.shift(1).fillna(0.0)
    turnover = (W - pre_trade).abs().sum(axis=1)

    out = {
        "gross_return": gross_return,
        "turnover": turnover,
        "gross_exposure": W.abs().sum(axis=1),
        "net_exposure": W.sum(axis=1),
    }
    if return_matrices:
        out["weights"], out["pre_trade"] = W, pre_trade
    return out


def make_decile_weights(data, score_col="score", date_col="month",
                        long_quantile=0.9, short_quantile=0.1, gross_exposure=2.0):
    """Equal-weight book: long the top bucket, short the bottom bucket."""
    data = data.copy()
    data["rank_score"] = data.groupby(date_col)[score_col].rank(pct=True)

    data["position"] = 0
    data.loc[data["rank_score"] >= long_quantile, "position"] = 1
    data.loc[data["rank_score"] <= short_quantile, "position"] = -1

    long_count = data.groupby(date_col)["position"].transform(lambda x: (x == 1).sum())
    short_count = data.groupby(date_col)["position"].transform(lambda x: (x == -1).sum())

    data["weight"] = 0.0
    data.loc[data["position"] == 1, "weight"] = (gross_exposure / 2) / long_count
    data.loc[data["position"] == -1, "weight"] = -(gross_exposure / 2) / short_count
    return data


def run_backtest_on_weights(data, weight_col="weight", date_col="month", id_col="permno",
                            return_col="ret_fwd1m", transaction_cost=0.001, cost_model=None):
    """Shared engine: per-row weights in, the standard backtest dict out.

    cost_model=None: flat transaction_cost per unit traded. Otherwise an object
    with monthly_costs(weights, pre_trade) -> DataFrame with a `total` column, such as
    evaluation.costs.RealisticCosts; its breakdown is returned as `cost_breakdown`.
    """
    book = portfolio_returns_and_turnover(
        data, weight_col=weight_col, date_col=date_col, id_col=id_col, return_col=return_col,
        return_matrices=cost_model is not None,
    )
    returns_no_fees = book["gross_return"].sort_index()
    turnover = book["turnover"].reindex(returns_no_fees.index).fillna(0)
    if cost_model is None:
        costs, breakdown = transaction_cost * turnover, None
    else:
        breakdown = cost_model.monthly_costs(book["weights"], book["pre_trade"])
        costs = breakdown["total"].reindex(returns_no_fees.index).fillna(0)
    returns_with_fees = returns_no_fees - costs

    stats_no_fees = compute_stats(returns_no_fees)
    stats_with_fees = compute_stats(returns_with_fees)
    return {
        "data": data,
        "returns_no_fees": returns_no_fees,
        "returns_with_fees": returns_with_fees,
        "turnover": turnover,
        "costs": costs,
        "cost_breakdown": breakdown,
        "gross_exposure": book["gross_exposure"],
        "net_exposure": book["net_exposure"],
        "pnl_no_fees": stats_no_fees["pnl"],
        "pnl_with_fees": stats_with_fees["pnl"],
        "stats_no_fees": stats_no_fees,
        "stats_with_fees": stats_with_fees,
    }


def backtest_from_predictions(data, predictions, split=None, score_col="score", date_col="month",
                              id_col="permno", return_col="ret_fwd1m", long_quantile=0.9,
                              short_quantile=0.1, transaction_cost=0.001, gross_exposure=2.0):
    """Equal-weight decile book of a score, through the shared engine.

    predictions: a column name, a Series or an array. split: e.g. "val" to backtest the
    validation months only.
    """
    data = data.copy()
    if split is not None:
        data = data[data["split"] == split].copy()
    data[score_col] = data[predictions] if isinstance(predictions, str) else predictions
    data = make_decile_weights(data, score_col=score_col, date_col=date_col, long_quantile=long_quantile,
                               short_quantile=short_quantile, gross_exposure=gross_exposure)
    return run_backtest_on_weights(data, weight_col="weight", date_col=date_col, id_col=id_col,
                                   return_col=return_col, transaction_cost=transaction_cost)


# -------- market alignment helper --------


def add_forward_market_return(df, panel, date_col="month", market_col="mkt_ret",
                              output_col="mkt_fwd1m"):
    # Strategy returns use ret_fwd1m (t+1). For CAPM diagnostics the market
    # return has to be shifted the same way so the regression aligns.
    df = df.copy()

    if market_col not in panel.columns:
        raise ValueError(f"Market column '{market_col}' not found in panel.")

    market_by_month = (
        panel[[date_col, market_col]]
        .dropna()
        .drop_duplicates(date_col)
        .set_index(date_col)[market_col]
    )

    target_month = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(1)
    df[output_col] = target_month.map(market_by_month)
    return df


# -------- IC diagnostics --------


def compute_monthly_ic(df, score_col="score", return_col="ret_fwd1m",
                       date_col="month", min_n_stocks=50):
    # Spearman and Pearson cross-sectional correlation between the score and
    # next-month returns. Skip months with too few stocks (very noisy IC).
    rows = []
    for month, g in df.groupby(date_col):
        g = g[[score_col, return_col]].dropna()
        if len(g) < min_n_stocks:
            continue
        rows.append({
            "month": month,
            "spearman_ic": g[score_col].corr(g[return_col], method="spearman"),
            "pearson_ic": g[score_col].corr(g[return_col], method="pearson"),
            "n_stocks": len(g),
        })
    return pd.DataFrame(rows)


def summarize_ic(df, model_name, score_col="score", splits=("val", "test")):
    # Per-split IC summary with t-statistics.
    ic = compute_monthly_ic(df, score_col=score_col)
    if ic.empty:
        raise ValueError(f"No IC observations for {model_name}.")

    split_map = df[["month", "split"]].drop_duplicates()
    ic = ic.merge(split_map, on="month", how="left")

    rows = []
    for split_name in splits:
        tmp = ic[ic["split"] == split_name].copy()
        for ic_col in ["spearman_ic", "pearson_ic"]:
            mean_ic = tmp[ic_col].mean()
            std_ic = tmp[ic_col].std()
            n_months = tmp[ic_col].count()
            t_stat = mean_ic / (std_ic / np.sqrt(n_months)) if std_ic > 0 else np.nan
            rows.append({
                "model": model_name,
                "split": split_name,
                "ic_type": ic_col,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "n_months": n_months,
                "t_stat": t_stat,
            })
    return pd.DataFrame(rows)


# -------- decile diagnostic --------


def compute_decile_returns(df, score_col="score", return_col="ret_fwd1m",
                           date_col="month", split="test"):
    # Average next-month return by score decile. A monotone D1 -> D10 curve is
    # the textbook sign that the score captures cross-sectional spread.
    data = df[df["split"] == split].copy()

    def assign_deciles(x):
        return pd.qcut(x, 10, labels=False, duplicates="drop") + 1

    data["decile"] = data.groupby(date_col)[score_col].transform(assign_deciles)
    decile_ret = (
        data.groupby([date_col, "decile"])[return_col]
        .mean()
        .reset_index()
        .groupby("decile")[return_col]
        .mean()
        .reset_index(name="avg_next_month_return")
    )
    return decile_ret


# -------- equal-weight decile backtests (mlp_model.py --diagnostics) --------


def run_long_short_backtest(df, split="test", score_col="score",
                            long_quantile=0.90, short_quantile=0.10,
                            transaction_cost=0.0010, gross_exposure=2.0):
    """Equal-weight top / bottom decile book of one split."""
    return backtest_from_predictions(df, score_col, split=split, score_col=score_col,
                                     long_quantile=long_quantile, short_quantile=short_quantile,
                                     transaction_cost=transaction_cost, gross_exposure=gross_exposure)


def run_backtest_grid(pred_df, model_name=None, score_col="score",
                      splits=("val", "test"),
                      costs=(0.0, 0.0005, 0.0010, 0.0025, 0.0050),
                      quantile_pairs=((0.90, 0.10), (0.80, 0.20), (0.70, 0.30)),
                      gross_exposure=2.0):
    # Run the backtest across all (split, bucket size, cost) combinations.
    # Used for transaction-cost sensitivity and portfolio-size robustness.
    rows = []
    for split_name in splits:
        for long_q, short_q in quantile_pairs:
            bucket_pct = round((1 - long_q) * 100)
            for cost in costs:
                bt = backtest_from_predictions(pred_df, score_col, split=split_name, score_col=score_col,
                                               long_quantile=long_q, short_quantile=short_q,
                                               transaction_cost=cost, gross_exposure=gross_exposure)
                row = {
                    "split": split_name,
                    "bucket_pct": bucket_pct,
                    "transaction_cost_bps": int(cost * 10000),
                    "gross_ann_return": bt["stats_no_fees"]["ann_return"],
                    "gross_ann_vol": bt["stats_no_fees"]["ann_vol"],
                    "gross_sharpe": bt["stats_no_fees"]["sharpe"],
                    "net_ann_return": bt["stats_with_fees"]["ann_return"],
                    "net_ann_vol": bt["stats_with_fees"]["ann_vol"],
                    "net_sharpe": bt["stats_with_fees"]["sharpe"],
                    "avg_monthly_turnover": bt["turnover"].mean(),
                    "n_months": len(bt["returns_with_fees"]),
                }
                if model_name is not None:
                    row = {"model": model_name, **row}
                rows.append(row)
    return pd.DataFrame(rows)
