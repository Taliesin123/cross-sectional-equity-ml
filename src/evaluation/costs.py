"""
Realistic trading costs: per-stock commission, half spread, market impact and borrow fee.

For a trade in stock i at the end of month t, the cost per dollar traded is

    commission[i,t] + spread_mult * half_spread[i,t] + impact_k * sigma_daily[i,t] * sqrt(D / ADV[i,t])

with commission = max(commission_min_bps, commission_per_share / price): brokers charge per
share, so a $2 stock costs 25 bps per dollar where a $50 stock costs 1 bp.

where D = |w_new - w_drifted| * AUM_t is the dollar size of the trade and ADV the
stock's average daily dollar volume (the square-root impact law). Each short position
also pays a borrow fee, fee[i,t] / 12 per month on its size, by the borrow tier of
build_cost_inputs.py: borrow_fee_gc for most stocks (general collateral),
borrow_fee_htb for hard-to-borrow stocks, borrow_fee_special for heavily shorted stocks
with few institutional shares to lend.

The fund size matters only through the impact term, so the result is a capacity
curve. AUM is given in dollars of the reference month (end of the test period) and,
by default, scaled back in earlier months with the total market cap of US common
stocks: the fund is the same share of the market in 2005 as in 2024.

Inputs come from data/processed/cost_inputs.parquet (build_cost_inputs.py). A stock
traded in a month without cost data (for example the month it delisted) uses its last
values from up to 3 months before; failing that, the month's illiquid end (75th
percentile spread and volatility, 10th percentile dollar volume).

Usage: pass RealisticCosts(...) as cost_model to evaluation.backtest.run_backtest_on_weights
or evaluation.portfolio.backtest_portfolio; without it a flat cost per dollar traded is used.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    aum: float = 1e8                      # fund capital in $ of the reference month
    aum_reference_month: str = "2024-11-30"
    scale_aum_with_market: bool = True
    commission_per_share: float = 0.005   # $ per share traded
    commission_min_bps: float = 0.5       # floor per dollar traded (high-priced stocks)
    spread_mult: float = 1.0              # 1 = pay half the closing quoted spread
    # Square-root law: the price moves by Y * sigma * sqrt(D / ADV) by the end of an order,
    # Y of order 1 for stocks; the average price paid over the order is 2/3 of that move.
    # 0.5 ~ (2/3) x 0.75; cost_study.py also reports 0.25 and 1.0.
    impact_k: float = 0.5
    borrow_fee_gc: float = 0.0025         # annual fee, easy-to-borrow stocks
    borrow_fee_htb: float = 0.04          # annual fee, hard-to-borrow stocks
    borrow_fee_special: float = 0.10      # annual fee, heavily shorted + small lendable supply
    max_cost: float = 0.25                # cap per dollar traded (guards against bad data)

    def label(self):
        return f"${self.aum / 1e6:,.0f}M"


def load_cost_inputs(root):
    path = Path(root) / "data" / "processed" / "cost_inputs.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/data/download_wrds_costs.py, then "
            "src/preprocessing/build_cost_inputs.py."
        )
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df["permno"] = df["permno"].astype("int64")
    return df


class RealisticCosts:
    """Cost model + its inputs. monthly_costs() is what the backtest engine calls."""

    FIELDS = ("half_spread", "sigma_daily", "adv_usd", "borrow_tier", "prc")

    def __init__(self, inputs, model=None, _shared=None):
        self.model = model or CostModel()
        if _shared is not None:
            self._s = _shared
            return
        if "borrow_tier" not in inputs:   # cost inputs built before the tiers existed
            inputs = inputs.assign(borrow_tier=inputs["hard_to_borrow"].astype("int8"))
        inputs = inputs[["permno", "month", "is_common", "mktcap_usd", *self.FIELDS]].copy()
        inputs["prc"] = inputs["prc"].where(inputs["prc"] > 0)
        common = inputs[inputs["is_common"]]
        g = common.groupby("month")
        self._s = {
            "inputs": inputs.set_index(["month", "permno"]).sort_index(),
            "market_cap": g["mktcap_usd"].sum().sort_index(),
            # conservative fills: the illiquid end of each month's cross-section
            "fill": pd.DataFrame({
                "half_spread": g["half_spread"].quantile(0.75),
                "sigma_daily": g["sigma_daily"].quantile(0.75),
                "adv_usd": g["adv_usd"].quantile(0.10),
                "borrow_tier": 0.0,
                "prc": g["prc"].quantile(0.25),
            }).sort_index(),
            "cache": {},
        }

    def with_model(self, model=None, **changes):
        """Same inputs (and cached matrices), different parameters."""
        new = replace(model or self.model, **changes)
        return RealisticCosts(None, new, _shared=self._s)

    @property
    def name(self):
        return f"realistic costs, {self.model.label()}"

    def describe(self):
        return asdict(self.model)

    # ---- inputs aligned with a weights matrix ----

    def aum_by_month(self, months):
        m = self.model
        months = pd.DatetimeIndex(months)
        if not m.scale_aum_with_market:
            return pd.Series(m.aum, index=months)
        cap = self._s["market_cap"]
        ref_month = pd.Timestamp(m.aum_reference_month)
        ref = cap[cap.index <= ref_month]
        ref_value = ref.iloc[-1] if len(ref) else cap.iloc[-1]
        cap_t = cap.reindex(cap.index.union(months)).ffill().bfill().reindex(months)
        return m.aum * cap_t / ref_value

    def matrices(self, months, permnos):
        months = pd.DatetimeIndex(months)
        permnos = pd.Index(permnos).astype("int64")
        key = (hash(months.asi8.tobytes()), hash(permnos.to_numpy().tobytes()))
        cache = self._s["cache"]
        if key in cache:
            return cache[key]
        inp = self._s["inputs"]
        lo = months.min() - pd.offsets.MonthEnd(3)
        sub = inp.loc[(inp.index.get_level_values("month") >= lo)
                      & (inp.index.get_level_values("month") <= months.max())
                      & inp.index.get_level_values("permno").isin(permnos)]
        grid = pd.DatetimeIndex(sorted(set(months) | set(sub.index.get_level_values("month"))))
        fill = self._s["fill"].reindex(self._s["fill"].index.union(grid)).ffill().bfill().reindex(months)
        out = {}
        for field in self.FIELDS:
            mat = sub[field].astype(float).unstack("permno").reindex(index=grid, columns=permnos)
            mat = mat.ffill(limit=3).reindex(months)
            missing = mat.isna()
            out[field] = mat.where(~missing, np.broadcast_to(fill[field].to_numpy()[:, None], mat.shape))
            if field == "half_spread":
                out["half_spread_filled"] = missing
        if len(cache) >= 2:  # one book at a time: keeps memory low
            cache.clear()
        cache[key] = out
        return out

    # ---- costs ----

    def _unit_costs(self, trades_abs, mats, aum):
        m = self.model
        dollars = trades_abs * aum.to_numpy()[:, None]
        participation = dollars / mats["adv_usd"].to_numpy()
        commission = np.maximum(m.commission_min_bps / 1e4,
                                m.commission_per_share / np.abs(mats["prc"].to_numpy()))
        spread_unit = commission + m.spread_mult * mats["half_spread"].to_numpy()
        impact_unit = m.impact_k * mats["sigma_daily"].to_numpy() * np.sqrt(participation)
        spread_unit = np.minimum(spread_unit, m.max_cost)
        impact_unit = np.minimum(impact_unit, np.maximum(m.max_cost - spread_unit, 0.0))
        return spread_unit, impact_unit, participation

    def unit_cost_fn(self, months, permnos):
        """f(t, trade_abs) -> cost per dollar traded (commission + half spread + impact)
        for each of `permnos` in month index t, at trade sizes trade_abs (fractions of
        capital). Used by cost-aware trading to set how fast each stock is traded."""
        months = pd.DatetimeIndex(months)
        mats = self.matrices(months, permnos)
        aum = self.aum_by_month(months).to_numpy()
        m = self.model
        prc = np.abs(mats["prc"].to_numpy())
        commission = np.maximum(m.commission_min_bps / 1e4, m.commission_per_share / prc)
        spread = np.minimum(commission + m.spread_mult * mats["half_spread"].to_numpy(), m.max_cost)
        sig, adv = mats["sigma_daily"].to_numpy(), mats["adv_usd"].to_numpy()

        def f(t, trade_abs):
            impact = m.impact_k * sig[t] * np.sqrt(trade_abs * aum[t] / adv[t])
            return np.nan_to_num(spread[t] + np.minimum(impact, np.maximum(m.max_cost - spread[t], 0.0)),
                                 nan=m.max_cost)
        return f

    def stock_costs(self, weights, pre_trade):
        """One row per stock-month with a trade or a short position: the cost components."""
        W = weights.sort_index()
        P = pre_trade.reindex_like(W).fillna(0.0)
        mats = self.matrices(W.index, W.columns)
        trades = (W - P).to_numpy()
        tabs = np.abs(trades)
        spread_u, impact_u, part = self._unit_costs(tabs, mats, self.aum_by_month(W.index))
        w = W.to_numpy()
        tier = mats["borrow_tier"].to_numpy()
        fee = np.select([tier >= 2, tier >= 1], [self.model.borrow_fee_special, self.model.borrow_fee_htb],
                        self.model.borrow_fee_gc)
        borrow = np.clip(-w, 0, None) * fee / 12
        i, j = np.nonzero((tabs > 0) | (w < 0))
        side = np.sign(np.where(w[i, j] != 0, w[i, j], -trades[i, j]))
        return pd.DataFrame({
            "month": W.index[i], "permno": W.columns[j].to_numpy(),
            "weight": w[i, j], "trade": trades[i, j], "side": side.astype(int),
            "participation": part[i, j],
            "spread_cost": tabs[i, j] * spread_u[i, j],
            "impact_cost": tabs[i, j] * impact_u[i, j],
            "borrow_cost": borrow[i, j],
            "inputs_filled": mats["half_spread_filled"].to_numpy()[i, j],
        })

    def monthly_costs(self, weights, pre_trade):
        """Costs as a fraction of capital, per month (the index of the returns)."""
        s = self.stock_costs(weights, pre_trade)
        s["traded"] = s["trade"].abs()
        s["traded_dollars_x_part"] = s["traded"] * s["participation"]
        g = s.groupby("month")
        out = pd.DataFrame({
            "spread": g["spread_cost"].sum(),
            "impact": g["impact_cost"].sum(),
            "borrow": g["borrow_cost"].sum(),
            "traded": g["traded"].sum(),
            "participation_weighted": g["traded_dollars_x_part"].sum(),
        }).reindex(weights.sort_index().index).fillna(0.0)
        out["total"] = out["spread"] + out["impact"] + out["borrow"]
        out["participation_weighted"] = out["participation_weighted"] / out["traded"].replace(0, np.nan)
        out["aum"] = self.aum_by_month(out.index)
        return out
