# Methodology: cross-sectional US equity strategy (v5)

Each month, models rank every US common stock by how likely it is to beat, or to trail, the other
stocks over the next month. A long-short book then buys the best-ranked stocks and shorts the
worst-ranked ones. The sample runs from 1980 to 2025. The data are CRSP, Compustat, IBES, short interest
and 13F holdings, downloaded from WRDS. They are licensed, so they are **not** in the repository.

**What v5 changes.** v4 was one network fitted once, on 1980–2004. Its validation results (2005–2014)
were much better than its test results (2015–2024). v5 makes the backtest more realistic, then tries to
make the model stronger:

| Change | Why |
| --- | --- |
| **Realism** | |
| The book trades one day after the month-end close, by default | the inputs are measured at that close: trading on it is not realistic |
| No shorts in stocks with less than 5% institutional ownership; borrow fees in three tiers | institutions supply almost all the shares that can be borrowed |
| Volatility target with a forecast from daily returns, if it tracks the target better on validation | the monthly forecast reacts with a lag |
| Returns regressed on the Fama-French five factors, momentum and short-term reversal | to see what the known factors do not explain |
| Checked: securities-lending fees (Markit) and point-in-time accounting (Compustat Snapshot) | the WRDS account has only WRDS's free samples of both, so the borrow fees stay proxies and the accounting data stay restated |
| **Performance** | |
| The models are refitted every year on all the history to date (walk-forward) | a model fitted on 1980–2004 was predicting 2024 |
| 5 network seeds averaged, plus gradient-boosted trees (LightGBM) | less noise from one training run; a second kind of model |
| Analyst inputs from IBES: revisions, surprise against the consensus, dispersion, coverage, recommendations | the strongest known signals in large stocks, where the model was weakest |
| Market-regime inputs: market return and volatility, momentum's volatility, dispersion, credit and term spreads | the inputs can matter differently in different markets |
| The target ranked inside the 5 volatility groups the book trades | the model learns the ranking the book uses |
| Cost-aware trading (book K): each stock moves only part of the way to its target, the more slowly the more it costs | costs took a third or more of gross return |

Everything else is v4's: the same network and settings, the same inputs (plus the new ones), the same
timing rules, the same books A–J and cost model.

Contents:

1. The models
2. Data files
3. How each file is processed
4. The features
5. Trading: timing, costs, borrowing, volatility target
6. Splits, walk-forward retraining and the decision rules
7. Results
8. Known limits

Appendix: code layout and how to run it.

---

## 1. The models

### What they predict

Take one stock in one month t. The models predict where that stock's return in month t+1 will rank:

- **The network (MLP)** predicts three classes: bottom 20%, middle 60% or top 20%. In v4 the classes
  were taken across all stocks of the month. In v5 they are taken **inside each of the 5 volatility
  groups** of that month (the same groups the vol-neutral book ranks in). The score is
  P(top) − P(bottom).
- **The trees (LightGBM)** predict the stock's percentile rank of next month's return inside its
  volatility group (a number from −0.5 to +0.5).
- **The ensemble** is the average of the two scores' ranks within the month.

Why ranks and not returns:

- **Fat tails.** A few stocks that double or collapse would dominate a squared-error fit on raw returns.
- **Only relative performance matters.** Ranking inside each month removes the market's move, which a
  long-short book does not bet on anyway.
- **Equal months.** A calm month and a crash month weigh the same in training.

### The network (unchanged from v4)

```
inputs per stock-month                                   industry id (2-digit SIC; 81 values)
(scaled numbers + 0/1 flags + regime inputs, section 4)                |
                 \                                    embedding: 8 learned numbers per industry
                  \___________________ concatenate ___________________/
                                          |
                    Linear -> 128  ->  BatchNorm  ->  ReLU  ->  Dropout 0.2
                    Linear -> 64   ->  BatchNorm  ->  ReLU  ->  Dropout 0.2
                    Linear -> 3    ->  softmax: P(bottom), P(middle), P(top)
```

Two hidden layers of 128 and 64 neurons. Class-weighted cross-entropy, AdamW (learning rate 1e-3,
weight decay 1e-4), batches of 8,192.

**Why an MLP.** Hidden layers can learn interactions that a linear model cannot: momentum can matter more
for small stocks, profitability more for cheap ones, and (new in v5) a signal can matter differently in
calm and turbulent markets.

**Why an embedding layer.** Industry is a category with 81 values: 80 two-digit SIC codes seen in
training, plus "unknown". The embedding gives each industry 8 learned numbers instead of 81 inputs
that are almost always 0. Industries that react to the inputs in a similar way end up with similar
vectors.

### The trees (new)

LightGBM regression on the within-group return rank:

- 63 leaves per tree, learning rate 0.05, at least 1,000 stock-months per leaf;
- each tree sees half the stocks and half the inputs, L2 penalty 10;
- the industry id as a categorical input.

Trees split on thresholds, so they find interactions and thresholds the network may miss. The two make
different mistakes, which is why their average is more stable than either.

### Training: refitted every year (walk-forward)

For each year Y from 2005 to 2025, every model is fitted on all the stock-months up to November of Y−1.
That is the last month whose next-month return is known at the end of December. The models then
predict every month of year Y.

1. **How long to train.** The last 36 months of that history are held out. The network trains on the
   rest until the held-out loss stops improving for 5 epochs; the trees until the held-out error stops
   improving for 50 rounds.
2. **Refit.** Every model is then trained again on the **whole** history to November of Y−1, for that
   number of epochs (network) or trees, so the most recent three years are used too.
3. **Five seeds.** The network is trained with 5 random seeds; their probabilities are averaged.

So 2005 is predicted by models fitted on 1980–2004, 2015 by models fitted on 1980–2014, and 2025 by
models fitted on 1980–2024. Every prediction in 2005–2025 is out of sample, made with the history
available at the time. No setting is chosen on the predicted years.

The inputs, missing-value flags and industry ids are still chosen on 1980–2004 only, earlier than any
predicted year.

### From score to portfolio

As in v4 (section 5 has the details):

- **Long the top decile, short the bottom decile, inside 5 volatility groups.** The information is in
  the extremes, and ranking inside volatility groups stops the book from being "long calm stocks, short
  volatile ones".
- **A buffer:** a stock enters at the top / bottom 10% of its group and leaves only past 20%.
- **Liquidity weights** (√ of daily dollar volume, at most 2% per stock), so the money sits in stocks
  that are cheap to trade.
- **A 10% volatility target.**

New in v5: the book trades one day later, cannot short stocks without lendable supply, may use a daily
volatility forecast, and book K trades each stock only part of the way toward its target.

---

## 2. Data files

All files are in `data/raw_wrds/`, are written by the scripts in `src/data/`, and must never be
committed.

| File | WRDS source | Contents | Size / period | Script |
| --- | --- | --- | --- | --- |
| `crsp_msf_v2.parquet` | `crsp.msf_v2` (CRSP monthly, CIZ format) | monthly total and ex-dividend return, price, market cap, shares, volume, SIC, share-type and exchange flags | 4,891,218 rows, 1960–2025 | `download_wrds.py` |
| `ccm_link.parquet` | `crsp.ccmxpf_lnkhist` | CRSP permno ↔ Compustat gvkey links with their valid dates (types LU / LC, primary P / C) | 33,324 links | `download_wrds.py` |
| `comp_fundq.parquet` | `comp.fundq` (INDL / STD / D / C) | quarterly fundamentals: year-to-date and quarterly income and cash-flow items, balance sheet, earnings release date `rdq` | 2,139,515 filings, to Sep 2026 | `download_wrds.py` |
| `ff_factors_monthly.parquet` | `ff.factors_monthly` | market excess return and risk-free rate | 1,202 months | `download_wrds.py` |
| `crsp_daily_costs.parquet` | `crsp.dsf_v2`, summarised per stock-month on the WRDS server | daily-return volatility, average daily dollar volume, quoted spread, high-low spread inputs | 2,107,440 stock-months, 2003–2025 | `download_wrds_costs.py` |
| `short_interest.parquet` | `comp.sec_shortint` | short interest per settlement date, linked to permnos | 2,708,227 rows, Jul 2006–2026 | `download_wrds_costs.py` |
| `crsp_daily_first_day.parquet` | `crsp.dsf_v2` | each stock's return on the first trading day of each month | 2003–2025 | `download_wrds_costs.py` |
| `filing_dates.parquet` | `comp.co_filedate` | SEC filing date of each 10-Q / 10-K (amendments left out) | 1,175,531 filings | `download_filing_dates.py` |
| `comp_fundq_extra.parquet` | `comp.fundq` | 51 more quarterly items: special items, intangibles, minority interest, share and debt issuance, buybacks, acquisitions, ... | 2,139,515 filings (same rows) | `download_wrds_features.py` |
| `ff_factors_daily.parquet` | `ff.factors_daily` | daily market return | 14,287 days, 1970–2026 | `download_wrds_features.py` |
| `crsp_daily_features.parquet` | `crsp.dsf_v2`, summarised per stock-month on the WRDS server | daily-return sums for volatility, beta, skewness, max / min, zero-return days, Amihud illiquidity, spreads | 4,246,665 stock-months, 1978–2025 | `download_wrds_features.py` |
| `earnings_announcement_days.parquet` | `crsp.dsf_v2` around `comp.fundq.rdq` | daily returns from 10 days before to 10 days after each earnings release | 13,788,062 rows, 939,202 releases | `download_wrds_features.py` |
| `short_interest_full.parquet` | `comp.sec_shortint` + `comp.sec_shortint_legacy` | short interest per settlement date | 3,721,100 rows, 1973–2026 | `download_wrds_features.py` |
| `inst_13f_holdings.parquet` | `tr_13f.s34` (first vintage of each filing), summed per stock and quarter on the server | shares held by 13F institutions, number of holders | 1,832,948 stock-quarters, 1980–2026 | `download_wrds_features.py` |
| `inst_13f_managers.parquet` | `tr_13f.s34` | number of institutions filing a 13F each quarter | 185 quarters | `download_wrds_features.py` |
| `ff_fivefactors_monthly.parquet` | `ff.fivefactors_monthly` | Fama-French five factors and momentum | monthly, 1963– | `download_wrds_extra.py` (v5) |
| `crsp_daily_returns.parquet` | `crsp.dsf_v2` | daily returns of every stock that is a US common stock in some month since 2003 | about 25M rows, 2003–2025 | `download_wrds_extra.py` (v5) |
| `ibes_link.parquet` | `wrdsapps.ibcrsphist` | IBES ticker ↔ CRSP permno links with their valid dates and match quality | | `download_wrds_extra.py` (v5) |
| `ibes_statsum.parquet` | `ibes.statsum_epsus` (US firms, EPS) | monthly consensus: long-term growth, fiscal years 1 and 2, next quarter; number of estimates, up / down revisions, mean, median, dispersion, high, low; the actual and its announcement date | 1976–2025 | `download_wrds_extra.py` (v5) |
| `ibes_statsum_unadj.parquet` | `ibes.statsumu_epsus` | the same forecasts, not adjusted for later splits | 1976–2025 | `download_wrds_extra.py` (v5) |
| `ibes_actuals.parquet` | `ibes.act_epsus` | quarterly earnings per share as first announced, with the announcement date | 1976–2025 | `download_wrds_extra.py` (v5) |
| `ibes_recommendations.parquet` | `ibes.recdsum` | monthly consensus recommendation (1 = strong buy … 5 = sell), if the account can read it | 1993–2025 | `download_wrds_extra.py` (v5) |
| `macro_fred.parquet` | FRED (public, not WRDS) | Moody's BAA and AAA yields, 10-year Treasury, 3-month T-bill | monthly, 1953– | `download_wrds_extra.py` (v5) |

Files built from them, in `data/processed/` (also not committed, except the small JSON / CSV descriptions):

- **`panel_model_filing.parquet`:** the model panel, 2,785,393 stock-months.
- **`cost_inputs.parquet`:** the cost-model inputs per stock-month.
- **Small descriptions:**
  - `feature_groups.json`: the inputs and the build settings;
  - `industry_mapping.json`;
  - `split_dates.json`;
  - `feature_selection_table.csv`;
  - `preprocessing_report.csv`: every step of the build, with its counts.

---

## 3. How each file is processed

`src/preprocessing/build_processed_data.py` builds the model panel and `features.py` defines the
features. `build_cost_inputs.py` builds the cost inputs.

**The timing rule.** Every row is dated at a month-end t. Every feature in that row must have been
public at the close of t. `check_processed_data.py` asserts this for every dated source, and every
count of early use is 0.

**`crsp_msf_v2` (CRSP monthly)**

1. Identifiers are read as text (a CUSIP read as a number loses its leading zeros). Dates are moved to
   month-ends.
2. Rows without a return are dropped: 82,116 rows.
3. `industry_code` is the 2-digit SIC code.
4. The monthly return features (section 4) are computed on each security's full history, before any
   filter, so that lags and rolling windows exist. A lag is used only if the lagged row is exactly that
   many months back: a gap in a stock's history never shifts a window.
5. NASDAQ volume is divided by the Gao–Ritter factors (2.0 to Jan 2001, 1.8 to Dec 2001, 1.6 in 2002–03),
   because dealer trades were double-counted. Turnover, dollar volume and Amihud use the adjusted
   volume.
6. The target `ret_fwd1m` is the return of month t+1, kept only if the next row really is the next
   calendar month. This leaves 4,766,733 stock-months.
7. **Universe filter, applied each month on CRSP's own flags.** A stock is kept only if it is a US
   common stock (`sharetype NS`, `securitytype EQTY`, `securitysubtype COM`, `usincflg Y`, `issuertype
   ACOR/CORP`) listed on NYSE / AMEX / NASDAQ (`primaryexch N/A/Q`, `conditionaltype RW`,
   `tradingstatusflg A`). This is the CIZ equivalent of share codes 10/11 and exchange codes 1/2/3.
   3,488,879 stock-months pass. From 1980, 2,785,393 are kept: about 5,900 stocks a month in training
   and 3,800 in the test decade.
8. **Delistings:** CIZ returns already include the delisting return.

**`ccm_link` (CRSP ↔ Compustat link)**

- Each stock-month gets the gvkey whose link is valid that month (link start ≤ month ≤ link end).
- With several valid links, a primary link (P) wins over C, and LC (researched) wins over LU.
- 97% of stock-months since 1980 have a link. Unlinked stocks stay in the universe with empty
  accounting data.

**`comp_fundq` + `comp_fundq_extra` (Compustat quarterly)**

1. Industrial, standard-format, domestic, consolidated records only. 1,600 duplicate firm-quarters are
   removed. The extra items are matched on gvkey, quarter end and fiscal year / quarter.
2. Year-to-date items become single-quarter flows: Q1 as reported, Q2–Q4 = this YTD − last quarter's
   YTD. A true quarterly item (`saleq`, `cogsq`, `niq`, ...) is used where present. This raises
   quarterly sales coverage from 60% to 81% of filings.
3. **Structural zeros:**
   - R&D, SG&A, interest and dividends not reported by a firm that reports sales are set to 0, with a
     0/1 flag.
   - Buybacks, share issuance, debt issuance / repayment and acquisitions not reported by a firm that
     reports its cash-flow statement are set to 0.
4. **Non-USD filings (232,785, mostly CAD):** log levels and the amounts later divided by a USD market
   cap are blanked. Ratios and growth rates are kept, since they carry no currency.
5. The features are computed on each firm's own history. "Same quarter last year" comparisons check
   that the row four filings back is 11–13 months earlier. Trailing-twelve-month (TTM) sums require
   four consecutive quarters.
6. Filings with no data at all are dropped before the merge, so they cannot hide the previous good
   filing.

**`filing_dates` (10-Q / 10-K dates) and the two availability dates**

Each filing is matched to its quarter by period of report (1,099,206 Compustat rows). 8,224 matches are
dropped as implausible: less than 10 days after quarter end, or more than 15 days before the earnings
release.

Each Compustat filing then gets two dates, each rounded up to the next month-end:

| Features | Usable from | Without that date |
| --- | --- | --- |
| Income statement and earnings surprise | the earlier of the earnings release (`rdq`) and the SEC filing | quarter end + 50 days (10-Q) or + 105 days (10-K) |
| Cash-flow statement and balance sheet | the SEC filing: these are often published only with the 10-Q / 10-K, weeks after the release | the later of the release and those deadlines |

The 50 / 105 days are the 45 / 90-day filing deadlines plus the late-filing extension (5 / 15 days).

Two as-of merges attach, to each stock-month, the latest filing usable by then. Each merge keeps a
filing only if it is 1 to 15 months old. About 85% of stock-months get accounting data. In 17%, the
cash-flow and balance-sheet features come from an older filing than the income features: the release
is out, the filing is not.

**`ff_factors_monthly` and `ff_factors_daily`**

- Market return = `mktrf + rf`, the CRSP value-weighted market.
- The monthly series is used for the monthly betas.
- The daily series is used for the daily betas, idiosyncratic volatility and the market-adjusted
  announcement return.

**`crsp_daily_features` (daily returns, summarised on the server)**

- The WRDS server returns one row per stock-month of sums: count, Σr, Σr², Σr³, sums with today's and
  yesterday's market return, max, min, zero-return days, the last five days, Σ|r| / dollar volume,
  quoted-spread mean and the high-low spread term.
- Python adds the sums over 1, 3 or 12 consecutive months to form the 14 daily features.
- A feature needs at least 10, 30 or 120 trading days. 87% of CRSP stock-months have a daily summary.

**`earnings_announcement_days`**

- Day 0 is the first trading day on or after the release. The return over days −1, 0, +1, minus the
  market return over the same days, is the announcement return.
- It requires three days with returns. It is usable from the month-end after day +1 and is kept for up
  to 4 months.
- 930,540 releases have a 3-day return. 59% of stock-months have a recent one.

**`short_interest_full`**

- One value per stock and settlement date. It is divided by the shares outstanding of the settlement
  month (capped at 1) and by daily volume (monthly volume / 21) for days to cover.
- It is usable from settlement + 14 days (publication comes about 8 business days after settlement)
  and is kept for up to 2 months.
- Coverage: 31% of training stock-months (the legacy table thins out before 2006) and 98% of
  validation and test.

**`inst_13f_holdings` + `inst_13f_managers`**

- Shares held by all 13F institutions, per CUSIP and quarter (first vintage of each filing).
- Linked to permnos through CRSP's historical CUSIP at the report date, and divided by shares
  outstanding (capped at 1).
- Usable from quarter end + 45 days (the 13F deadline) and kept for up to 6 months. 77% of stock-months
  have it.

**IBES analyst data** (v5)

- **Link:** each IBES ticker gets the CRSP permno whose link is valid on the consensus date
  (`wrdsapps.ibcrsphist`, match quality score 0–2). With several tickers on one stock-month, the one with
  the most estimates is kept.
- **Timing:** the consensus is computed by IBES on its statistical date (`statpers`, mid-month). A
  stock-month uses the consensus of its own month, or of the month before if none, and only if it is at
  most 50 days old.
- **Share basis.** The split-adjusted file restates every past forecast on today's share count. Dividing
  it by the CRSP price of the time would be wrong for any stock that split since. The factor
  f = adjusted / unadjusted forecast (same date, `statsumu_epsus`) converts the price to the forecasts'
  basis: forecast / (price × f). Without the unadjusted file, the price-scaled inputs are left out.
- **Revisions** compare this month's FY1 consensus with last month's only when both forecast the same
  fiscal year (when the fiscal year rolls over, the change is not a revision).
- **Surprise against the consensus:** the quarter's actual (as first announced) minus the last
  consensus for that quarter dated before the announcement, over the price. It is usable from the
  month-end after the announcement date + 1 day and kept for up to 4 months.
- Stocks without analysts get coverage 0 and missing values (with their flags) for the rest.

**Market regime** (v5)

- From the Fama-French files: the market return of the month, of 12 and 24 months; the market's daily
  volatility this month and its ratio to the last 12 months; momentum's (UMD) return over 12 months and
  its volatility over 6 months.
- From CRSP: the cross-sectional dispersion of this month's returns (winsorised 1% / 99%).
- From FRED: credit spread (BAA − AAA) and term spread (10-year − 3-month), monthly averages.
- These are the same for every stock in a month, so scaling across stocks would turn them into 0. Each
  is instead z-scored over time with the history known at t (expanding mean and standard deviation, at
  least 60 months), clipped at ±4. Months before that history exists get 0.

**Ratios to market value.** These are formed after the merge, as an accounting amount (at its own date)
divided by the market cap of the month the feature is used. Book-to-market in March 2010 is therefore
the latest public book equity over the March 2010 market cap.

**From features to model inputs** (all on the merged panel, from January 1980)

1. **Labels:** `label20` comes from the ranks of `ret_fwd1m` inside each month. `label20_vg` (v5) does
   the same inside each month's 5 volatility groups (12-month volatility known at t; stocks without it
   go into the most volatile group), and `rank_vg` is the percentile rank inside the group minus 0.5,
   the trees' target. Splits are by date (section 6).
2. **Selection, on training data only:** an input is dropped if more than 70% of it is missing over
   1995–2004, the last ten training years (coverage in the early 1980s is structurally low). Three
   Compustat inputs were dropped this way.
3. **Missing flags:** a 0/1 flag `miss_<input>` is added for every input missing in at least 5% of
   training stock-months (87 for v4's inputs; the analyst inputs add their own).
4. **Per input and per month:**
   1. winsorise at the 1st / 99th percentile;
   2. fill missing values with the month's median (then the training median, then 0);
   3. scale as (x − median) / IQR (a z-score if the IQR is ~0);
   4. clip at ±10.

   Inputs are thereby comparable across stocks within a month, robust to extreme ratios, and free of
   look-ahead.
5. **Industry ids:** 2-digit SIC → 1…80 for the industries seen in training, 0 = unknown.
6. **Unscaled copies** of the 6- and 12-month vol, the 12- and 24-month beta, the idiosyncratic vol and
   the daily 3-month vol are kept for portfolio construction. They are never inputs.

**`crsp_daily_costs` + `short_interest` + `crsp_daily_first_day` → `cost_inputs.parquet`** (from 2003)

For each stock-month, `build_cost_inputs.py` computes the half spread, the daily volatility, the
average daily dollar volume (that month and the 6-month average), NYSE size percentile and decile, the
share held by 13F institutions as known at t, and a borrow tier (section 5). It also keeps the
first-day return: the book now trades at the close of that day.

**`crsp_daily_returns`** (v5) becomes a trading-day × stock matrix for the daily volatility forecast
(section 5). **`ff_fivefactors_monthly`** and a short-term reversal factor built from CRSP are the
factors of `factor_alpha.py` (section 7).

---

## 4. The features

**v4's 199 inputs + industry, plus v5's analyst and regime inputs:**

- scaled numbers: 13 monthly-return, 33 market, 61 Compustat (v4), 11 analyst (v5);
- missing flags (one for each input missing in at least 5% of training stock-months) and 5
  structural-zero / sign flags;
- 10 market-regime inputs (v5);
- the industry id, through the embedding.

The exact list is in `data/processed/feature_groups.json` (`v4_inputs` holds v4's).

"t" is the month of the row. Every feature is known at the close of t.

**Monthly returns (13; CRSP monthly; month-end t)**

| Input | Definition |
| --- | --- |
| `ret_lag1`, `ret_lag2`, `ret_lag3`, `ret_lag6`, `ret_lag12` | return of month t−1, t−2, t−3, t−6, t−12 |
| `reversal_1m` | −(return of month t): short-term reversal |
| `mom6m1`, `mom12m1` | compounded return over t−6…t−2 and t−12…t−2 (momentum, skipping the latest months) |
| `vol_6m`, `vol_12m` | standard deviation of monthly returns over the last 6 / 12 months |
| `beta_12m`, `beta_24m` | beta against the market over 12 / 24 months |
| `idio_vol_12m` | residual volatility of the 12-month market regression |

**Size, trading, issuance, long-horizon returns (12; CRSP monthly; month-end t)**

| Input | Definition |
| --- | --- |
| `log_mktcap` | log market cap |
| `log_price` | log price |
| `turnover_1m`, `turnover_12m` | volume / shares outstanding, this month and 12-month average |
| `log_dollar_volume_3m` | log of average monthly dollar volume over 3 months |
| `share_issuance_12m` | log growth of split-adjusted shares over 12 months |
| `dividend_yield_12m` | sum over 12 months of (total return − price return) |
| `price_to_12m_high` | log of price / highest month-end price of the last 12 months (split-adjusted) |
| `ret_36_13` | compounded return t−36…t−13 (long-term reversal) |
| `seasonal_1y` | the stock's return in the target's calendar month one year earlier |
| `seasonal_2_5y` | average return in the target's calendar month 2 to 5 years earlier |
| `industry_mom_6m` | value-weighted 6-month return of the stock's 2-digit SIC industry (at least 3 stocks) |

**Daily returns: risk, lottery, liquidity (14; CRSP daily; month-end t)**

| Input | Definition |
| --- | --- |
| `vol_d_1m`, `vol_d_3m` | standard deviation of daily returns over 1 / 3 months |
| `vol_d_ew` | exponentially weighted daily volatility (half-life 1 month, 6 months of data) |
| `vol_d_last5` | volatility of the last five trading days of the month |
| `vol_ratio_1m_12m` | 1-month daily vol / 12-month daily vol |
| `ivol_d_1m`, `ivol_d_3m` | idiosyncratic volatility: residual of the daily market regression, 1 / 3 months |
| `beta_d_12m` | Dimson beta: slopes on today's and yesterday's market, 12 months of daily returns |
| `max_ret_1m`, `min_ret_1m` | largest and smallest daily return of the month |
| `skew_d_3m` | skewness of daily returns over 3 months |
| `zero_ret_share_3m` | share of trading days with a zero return, 3 months |
| `log_amihud_3m` | log of mean \|daily return\| / dollar volume (per $M), 3 months (Amihud illiquidity) |
| `bid_ask_spread` | average closing quoted spread (ask − bid) / mid; Abdi–Ranaldo high-low estimate if fewer than 5 days of quotes |

**Short interest (3; settlement date + 14 days)**

| Input | Definition |
| --- | --- |
| `short_interest_ratio` | shares sold short / shares outstanding |
| `short_interest_change_3m` | change of that ratio over 3 months |
| `days_to_cover` | shares sold short / average daily volume |

**Earnings-announcement return (1; close of day +1)**

| Input | Definition |
| --- | --- |
| `earnings_ann_return` | stock return − market return over trading days −1, 0, +1 around the latest earnings release |

**Institutional ownership (3; quarter end + 45 days)**

| Input | Definition |
| --- | --- |
| `inst_ownership` | shares held by 13F institutions / shares outstanding |
| `inst_ownership_change` | change from the previous quarter |
| `inst_breadth_change` | change in the number of institutions holding the stock / number of 13F filers |

**Analyst forecasts (11; IBES consensus date, at most 50 days old)** (v5)

| Input | Definition |
| --- | --- |
| `an_coverage` | log(1 + number of analysts with an FY1 forecast); 0 without coverage |
| `an_dispersion` | standard deviation of the FY1 forecasts / \|mean forecast\| (at least 2 analysts) |
| `an_fep` | mean FY1 forecast / price (forecast earnings yield) |
| `an_fy2_growth` | (FY2 forecast − FY1 forecast) / price |
| `an_ltg` | mean long-term growth forecast (%) |
| `an_updown` | (upward − downward FY1 revisions this month) / number of estimates |
| `an_rev_1m`, `an_rev_3m` | change in the FY1 consensus since last month / price; sum over the last 3 months (same fiscal year only) |
| `an_sue` | (actual quarterly EPS − last consensus before the announcement) / price; from announcement + 1 day |
| `an_rec_mean`, `an_rec_change_3m` | mean recommendation (1 = strong buy … 5 = sell); its change over 3 months |

**Market regime (10; one value per month, z-scored over time)** (v5)

| Input | Definition |
| --- | --- |
| `reg_mkt_ret_1m`, `reg_mkt_ret_12m`, `reg_mkt_ret_24m` | market return over 1, 12, 24 months (a 24-month loss is when momentum crashes) |
| `reg_mkt_vol_1m`, `reg_mkt_vol_ratio` | market daily volatility this month; its ratio to the last 12 months |
| `reg_umd_ret_12m`, `reg_umd_vol_6m` | momentum factor's return over 12 months; its volatility over 6 months |
| `reg_xs_dispersion` | standard deviation of stock returns across stocks this month |
| `reg_credit_spread`, `reg_term_spread` | BAA − AAA yield; 10-year Treasury − 3-month T-bill |

**Income statement (19; earnings release)**

| Input | Definition |
| --- | --- |
| `comp_log_sales_qtr`, `comp_log_revenue_qtr` | log quarterly sales / revenue (USD filings) |
| `comp_gross_margin_qtr` | (sales − cost of goods sold) / sales |
| `comp_oibdp_margin_qtr`, `comp_operating_margin_qtr` | operating income before / after depreciation / sales |
| `comp_net_margin_qtr` | net income / sales |
| `comp_sga_to_sales_qtr`, `comp_rd_to_sales_qtr`, `comp_interest_to_sales_qtr`, `comp_depreciation_to_sales_qtr` | SG&A, R&D, interest expense, depreciation / sales |
| `comp_saley_qtr_yoy_growth`, `comp_revty_qtr_yoy_growth`, `comp_oibdpy_qtr_yoy_growth`, `comp_oiadpy_qtr_yoy_growth`, `comp_niy_qtr_yoy_growth`, `comp_cogsy_qtr_yoy_growth`, `comp_xsgay_qtr_yoy_growth` | growth against the same fiscal quarter a year earlier: sales, revenue, operating income before / after depreciation, net income, cost of goods sold, SG&A |
| `comp_sue_earnings`, `comp_sue_sales` | earnings surprise: (this quarter − same quarter last year) / standard deviation of that change over the last 8 quarters; on earnings and on sales |

**Cash-flow statement (10; 10-Q / 10-K filing)**

| Input | Definition |
| --- | --- |
| `comp_capex_to_sales_qtr`, `comp_operating_cf_to_sales_qtr`, `comp_investing_cf_to_sales_qtr`, `comp_financing_cf_to_sales_qtr`, `comp_dividends_to_sales_qtr`, `comp_cash_change_to_sales_qtr`, `comp_receivables_change_to_sales_qtr`, `comp_inventory_change_to_sales_qtr` | quarterly capex, operating / investing / financing cash flow, dividends, change in cash, in receivables, in inventory / sales |
| `comp_capxy_qtr_yoy_growth`, `comp_oancfy_qtr_yoy_growth` | growth of capex and of operating cash flow against the same quarter a year earlier |

**Balance sheet: profitability, investment, leverage, liquidity, accruals (23; 10-Q / 10-K filing)**

| Input | Definition |
| --- | --- |
| `comp_roa_qtr`, `comp_roe_qtr` | quarterly income / last quarter's total assets; / last quarter's book equity |
| `comp_gross_profit_to_assets` | TTM (sales − cost of goods sold) / assets |
| `comp_operating_profitability` | TTM (sales − COGS − SG&A − interest) / book equity |
| `comp_asset_growth_yoy` | assets / assets a year earlier − 1 |
| `comp_investment_to_assets_yoy` | (change in PP&E + change in inventory over a year) / assets a year earlier |
| `comp_book_leverage`, `comp_liabilities_to_assets` | debt / assets; total liabilities / assets |
| `comp_current_ratio`, `comp_quick_ratio` | current assets / current liabilities; (current assets − inventory) / current liabilities |
| `comp_cash_to_assets`, `comp_ppe_to_assets`, `comp_intangibles_to_assets` | cash, PP&E, intangibles / assets |
| `comp_accruals_to_assets` | TTM (income − operating cash flow) / assets |
| `comp_net_operating_assets` | (operating assets − operating liabilities) / last quarter's assets |
| `comp_asset_turnover` | TTM sales / assets |
| `comp_roa_change_yoy`, `comp_roa_vol_8q` | ROA − ROA a year earlier; standard deviation of ROA over 8 quarters |
| `comp_special_items_to_assets` | special items / last quarter's assets |
| `comp_net_debt_issuance_to_assets`, `comp_net_equity_issuance_to_assets` | TTM (debt issued − repaid) / assets; TTM (shares issued − bought back) / assets |
| `comp_acquisitions_to_assets`, `comp_capex_to_assets` | TTM acquisitions, TTM capex / assets |

Book equity = stockholders' equity + deferred taxes − preferred stock. TTM = sum of the last four
quarters.

**Value ratios to market cap (9; amount at its own date, market cap of month t)**

| Input | Definition |
| --- | --- |
| `comp_book_to_market` | book equity (if > 0) / market cap |
| `comp_earnings_to_price`, `comp_sales_to_price`, `comp_cashflow_to_price`, `comp_rd_to_market` | TTM income, sales, operating cash flow, R&D / market cap |
| `comp_payout_yield`, `comp_net_payout_yield` | TTM (dividends + buybacks) / market cap; minus share issuance |
| `comp_ebitda_to_ev` | TTM operating income before depreciation / (market cap + debt + preferred + minority interest − cash) |
| `comp_market_leverage` | debt / (debt + market cap) |

**Flags (5 + 87)**

- **Structural-zero flags:** `comp_flag_xrdy_not_reported`, `comp_flag_xsgay_not_reported`,
  `comp_flag_xinty_not_reported` and `comp_flag_dvy_not_reported` (the item was set to 0, section 3).
- **Sign flag:** `comp_flag_negative_book_equity`.
- **Missing flags:** `miss_<input>` = 1 when the input was missing before imputation. There are 87, one
  for each input missing in ≥ 5% of training stock-months.

**Dropped by the missing-rate rule** (more than 70% missing over 1995–2004):

- `comp_net_income_total_margin_qtr`;
- `comp_xrdy_qtr_yoy_growth`;
- `comp_working_capital_change_to_sales_qtr`.

The full table is in `data/processed/feature_selection_table.csv`.

**Which groups mattered in v4.** Permutation importance of the v4 network on validation: each group
is shuffled inside each month, and the table shows the drop in IC. The base IC is 0.111. (v5:
`walk_forward.py --importance` measures the same for the refitted models, analyst and regime groups
included; results in section 7.)

| Group | IC drop | t |
| --- | --- | --- |
| Risk / lottery / liquidity (daily returns) | 0.036 | 9.4 |
| Momentum / reversal (monthly returns) | 0.026 | 7.3 |
| Income statement | 0.016 | 9.6 |
| Balance sheet | 0.010 | 7.4 |
| Value ratios to market cap | 0.010 | 4.8 |
| Earnings surprise | 0.0045 | 6.3 |
| Risk (monthly returns) | 0.0036 | 4.3 |
| Size / trading | 0.0035 | 2.4 |
| Earnings-announcement return | 0.0028 | 6.6 |
| Cash-flow statement | 0.0026 | 3.9 |
| Industry (embedding) | 0.0020 | 3.6 |
| Institutional ownership (13F) | 0.0017 | 3.6 |
| Short interest | 0.0011 | 2.8 |
| Issuance / payout (prices) | 0.0005 | 1.8 |

Each group is shuffled together with its missing flags (`results/mlp_filing/feature_importance_val.csv`).

---

## 5. Trading: timing, costs, borrowing, volatility target

### When the book trades (v5: one day later)

The inputs are measured at the close of the last trading day of month t. Nobody can compute the scores
and trade at that same close. In v5 the book trades at the **close of the first trading day of month
t+1** and holds until the close of the first trading day of month t+2. Its monthly return is

```
(1 + r[t+1]) / (1 + r_day1[t+1]) × (1 + r_day1[t+2]) − 1
```

with r the stock's monthly return and r_day1 its first-day return (both from CRSP). Every number in this README is on this timing unless marked
otherwise. `--trade-at close` gives v4's timing; the delay table (`delay_test.csv`) compares the two
for every book. For v4's book F, the delay cost 0.36 of validation Sharpe and 0.09 of test Sharpe.

### Cost per dollar traded (unchanged from v4)

At each rebalance the book moves from last month's weights, drifted by one month of returns, to the new
weights. Each trade in stock i is D = |w_new − w_drifted| × AUM dollars. Positions that leave the panel
(a delisting, for instance) are charged for being closed. `src/evaluation/costs.py` charges, stock by
stock and month by month:

```
commission  +  half spread  +  0.5 × σ_daily × √( D / ADV )
```

| Term | Definition | Where it comes from |
| --- | --- | --- |
| Commission | max(0.5 bp, $0.005 per share ÷ price): 1 bp on a $50 stock, 25 bps on a $2 stock | brokers charge per share |
| Half spread | half the average daily closing quoted spread (ask − bid) / mid that month (≥ 5 days of quotes); else the Abdi–Ranaldo (2017) estimate from daily close / high / low; else the median of the stock's NYSE size decile that month | CRSP daily; quotes exist for 99.7% of common-stock months |
| Market impact | 0.5 × σ_daily × √(D / ADV): σ_daily = standard deviation of the stock's daily returns that month, ADV = its average daily dollar volume that month | square-root law: an order of size Q moves the price by about Y·σ·√(Q / V) with Y ≈ 1, and the average price paid is about 2/3 of the final move, so 0.5 ≈ 2/3 × 0.75 |
| Cap | at most 25% per dollar traded | guards against bad data |

**Fund size.** $100M in end-of-November-2024 dollars, scaled back each month with the total market cap
of US common stocks: the fund is the same share of the market in 2005 as in 2024. Only the impact term
depends on the fund size.

**Missing cost data** (for example, in the month a stock delists): the stock's values from up to 3
months before are used; failing that, the illiquid end of that month's cross-section (75th percentile
spread and volatility, 10th percentile volume).

### Borrowing (v5: a short-sale limit and three fee tiers)

The shares a short seller borrows come mostly from institutions' holdings. A stock that institutions
barely own often cannot be borrowed at all, and a heavily shorted one with few lenders costs far more
than the usual fee.

**Short-sale limit.** No short position in a stock whose 13F institutional ownership, as known at the
end of month t, is below **5%** of its shares (`--min-short-io`; 0 switches the rule off). A short
already held is closed in full when the stock falls below the limit. The ranks do not change: stocks
that cannot be shorted are left out of the short leg, and the other shorts are scaled up so the leg
keeps its size. Stocks no 13F filer holds count as 0%.

**Fee tiers** (annual fee / 12 × short size, charged monthly):

| Tier | Rule (known at t) | Fee / year |
| --- | --- | --- |
| General collateral | everything else | 0.25% |
| Hard to borrow (v4's rule) | short interest in the month's top 10%, or price below $5, or no short-interest data and market cap below the NYSE 10th percentile | 4% |
| Special (v5) | short interest in the month's top 10% **and** institutional ownership below 25% | 10% |

These are proxies. Real fees come from securities-lending data (Markit, also called IHS Markit or S&P
Global Securities Finance). The first step of `download_wrds_extra.py` lists the WRDS libraries the
account can read (`data/raw_wrds/download_report_extra.json`): this account has only WRDS's free Markit
sample (`mrktsamp_msf`, the year 2011), not the subscription, so the tiers stay.

### Volatility target (v5: daily forecast, if it tracks the target better)

The book is scaled each month to a forecast volatility of 10% a year (leverage at most 2, re-set only
when the forecast leaves 9–11%). Two forecasts of the scaled book's volatility are compared:

- **"blend"** (v4): the average of two variances, a replay of the current book on its stocks' last 36
  months of monthly returns and an EWMA of the book's own monthly returns (half-life 6 months).
- **"daily"** (v5): the current book replayed on its stocks' last **126 trading days** of daily returns
  (EWMA, half-life 21 trading days), × 252. A stock missing on a day gets that day's average return. It
  reacts within weeks instead of months.

Both are multiplied by the same correction, √(mean (realised / forecast)²) over the book's last 24
months, with only months already realised. `cost_study.py --vol-method auto` (the default) builds book F
with each, measures how far its rolling 12-month realised volatility stays from 10% on validation (root
mean square error), and keeps the closer one for every book (`vol_method_choice.csv`).

### Cost-aware trading (v5: book K)

Costs took a third or more of v4's gross return. Book K is book F, except that each stock moves only
part of the way from its drifted weight to its target each month (Gârleanu and Pedersen, 2013, in its
simplest form):

```
w_new = w_drifted + rate × (w_target − w_drifted),      rate = 1 / (1 + κ × c)
```

c is the stock's expected cost per dollar for that trade (commission + half spread + impact at its
size, at the fund size being tested). With κ = 50, a large stock with c = 5 bps trades 98% of the way;
a microcap with c = 100 bps trades 67% of the way. Then:

- stocks that leave the panel, and shorts in stocks that can no longer be shorted, are closed in full;
- a position no longer in the target is closed in full once it is below a quarter of the month's
  average position. Without this rule an expensive stock is sold by a fraction each month and never
  reaches zero: the book carried some 2,600 tiny leftover positions;
- each leg is rescaled to its target size, so the book keeps the vol target's leverage and stays
  dollar-neutral.

κ = 0 is book F. κ is chosen on validation, at $100M, from 0, 25, 50, 100, 200, 400, 800, 1600: the
highest net Sharpe (`partial_trading_grid.csv`).

### The books

All are vol-neutral deciles (long the top 10%, short the bottom 10% inside each of 5 volatility groups)
with the 10% vol target:

| Book | Rule |
| --- | --- |
| A | equal weights |
| B | A + buffer: a stock enters at the top / bottom 10% of its group, leaves only past 20% |
| C, D | A, B without stocks under $5 |
| E | B without stocks under $5 or below the NYSE 20th size percentile |
| F | B with liquidity weights: √(daily $ volume, 6-month average), at most 2% per stock |
| G | F without stocks under $5 |
| H | G + no position above 5% of the stock's daily $ volume |
| I | H + no trade while a held stock's drifted weight is within 25% of its target |
| J | I + no single-name shorts in microcaps, hedged with a short in a small-cap index |
| **K** (v5) | F + cost-aware trading |

Every book obeys the short-sale limit and pays the tiered fees. The book traded is the one with the
highest validation net Sharpe at $100M (section 6).

---

## 6. Splits, walk-forward retraining and the decision rules

| Split | Months | Stock-months | In v5 |
| --- | --- | --- | --- |
| Train | Jan 1980 – Dec 2004 (300) | 1,783,757 | chooses the inputs, missing flags and industry ids; the first walk-forward fit |
| Validation | Jan 2005 – Dec 2014 (120) | 500,596 | every choice (below) |
| Test | Jan 2015 – Nov 2024 (119) | 456,373 | reported only; decides nothing |
| Recent | Dec 2024 – Nov 2025 (12) | 44,667 | v4's holdout, already opened once: reported only, decides nothing |
| Holdout | Dec 2025 onward | none yet | opened once, for the one chosen book, when the data exist |

With the walk-forward, every month from 2005 is predicted by models fitted on the months before it
(section 1). "Validation" is therefore not a set of months the models never train on; it is the period
on which every choice is judged. Each of its months was still predicted out of sample.

**How clean is the test period?** No v5 setting is fitted or chosen on test. But v4's test results were
seen, and they shaped what v5 tries (the walk-forward, for instance, because test was much weaker than
validation). The recent year was v4's holdout. Test and recent are useful checks, not clean ones. Only
the months after November 2025 are a clean test, and `cost_study.py --holdout` opens them once.

**Fixed in advance, not tuned:** the network and its settings (v4's), the trees' settings, the 5 seeds,
the 36 months held out for early stopping, the equal-weight average of the two models' ranks, the IBES
age limit (50 days), the short-sale limit (5%), the fee tiers and book K's rule for closing leftovers.

**Decision rules,** written before any result of the v5 models existed. All on validation (Jan 2005 –
Dec 2014):

1. **Volatility forecast:** "daily" or "blend", whichever keeps book F's rolling 12-month volatility
   closer to 10% (root mean square error).
2. **Cost-aware trading speed κ:** book K's highest net Sharpe at $100M.
3. **Market-regime inputs:** they stay only if scrambling them (each month's values replaced by those of
   a random month before 2015) lowers the validation IC of the network (seeds averaged) with a
   t-statistic above 2 (`walk_forward.py --importance`). Otherwise the v5 model is `wf_full_noregime`, the same ensemble
   without them.
4. **v5 against v4:** v5 replaces v4 only if both:
   - (a) its monthly rank IC on the stock-months both versions score is higher, with a t-statistic of the
     monthly difference above 2;
   - (b) the book its cost study chooses has a higher net Sharpe at $100M than the book v4's study
     chooses.

   Both versions go through the same v5 backtest (one-day delay, short-sale limit, fee tiers, the vol
   forecast and κ each chosen on validation). `compare_versions.py` applies the rule.
5. **The traded book:** the highest net Sharpe at $100M among A–K, for the version rule 4 keeps.
6. **Holdout checks,** written to `results/cost_study_<run>/chosen_book.json` by the study: net Sharpe
   within the 90% range of a 12-month Sharpe if the true Sharpe equals the test one; realised vol 7–13%;
   drawdown no worse than test's.

---

## 7. Results

### a) v4's predictions under the v5 backtest

A first check, run before the new data exist: v4's real predictions through the v5 backtest. The 13F
ownership and the daily returns were not available for it, so it has **no short-sale limit, no special
fee tier, and the "blend" vol forecast**. The full run on the Mac adds them. Fund size $100M, realistic
costs, net of everything.

**Book F, one change at a time** (net Sharpe, validation / test / Dec 2024 – Nov 2025):

| Backtest | Validation | Test | Recent |
| --- | --- | --- | --- |
| v4's (trades at the month-end close) | 2.29 | 1.04 | 0.84 |
| + trades one day later (the v5 headline) | 1.92 | 0.95 | 0.89 |
| Book K: + cost-aware trading (κ = 50, chosen on validation) | 2.01 | 0.97 | 0.73 |

The first line is close to v4's published 2.31 / 1.05 / 0.85.

**Books F and K, trading one day later:**

| | F validation | F test | K validation | K test |
| --- | --- | --- | --- | --- |
| Gross return / year | 24.5% | 14.8% | 24.0% | 13.7% |
| Cost per dollar traded | 30.3 bps | 25.9 bps | 24.5 bps | 20.3 bps |
| Turnover per month | 1.82 | 1.54 | 1.76 | 1.50 |
| Total cost / year (incl. borrow) | 8.2% | 6.0% | 6.8% | 4.8% |
| **Net return / year** | **16.3%** | **8.8%** | **17.2%** | **8.9%** |
| Yearly vol | 8.5% | 9.3% | 8.6% | 9.1% |
| **Net Sharpe** | **1.92** | **0.95** | **2.01** | **0.97** |
| Max drawdown | −7.1% | −28.1% | −7.0% | −27.4% |
| Stocks held | 1,061 | 989 | 1,085 | 1,000 |

- **Cost-aware trading saves 1.2–1.4% a year of costs and gives up part of it in gross return**
  (0.5% on validation, 1.1% on test: the book is a little stale). The net gain at $100M is small: +0.09
  of Sharpe on validation, +0.02 on test. It grows with the fund size: at $1B, K's net Sharpe is 1.54 / 0.60 against F's 1.22 / 0.42.
- **The choice on validation** (rule 5) is K, at 2.01, ahead of F (1.92) and G and I (1.76).
- **The vol target still runs low:** realised vol 8.5–9.3% against 10%, and the test drawdown is −28%.
  The daily forecast, chosen only if it tracks the target better on validation, is the fix tested on
  the Mac.
- On the recent year K does worse than F (0.73 against 0.89). With 12 months a Sharpe has a standard
  error of about 1.2, and the year decides nothing.

**Alpha after the factors** (book K, $100M, net returns regressed on the Fama-French five factors,
momentum and short-term reversal; Newey-West t-statistics):

| | Net return / year | Alpha / year | t | R² | Main exposures (t > 2) |
| --- | --- | --- | --- | --- | --- |
| Validation | 17.2% | 15.9% | 5.1 | 0.21 | profitability +0.42 |
| Test | 8.9% | 7.1% | 2.6 | 0.30 | value +0.31, profitability +0.31, small-minus-big −0.26, momentum +0.19 |
| 2005 – Nov 2025 | 12.7% | 11.8% | 5.5 | 0.19 | profitability +0.37, small-minus-big −0.18 |

The known factors explain a fifth to a third of the book's month-to-month moves, and at most a fifth
of its average return. The comparison is conservative, since the factor returns
are before any trading cost and the book's are after. The book has no market or short-term-reversal
exposure.

### b) v5 against v4

Not run yet: the walk-forward needs the full panel with the new data, on the Mac (appendix). Once
`compare_versions.py` has run, its tables go here:

- `results/walk_forward/ic_comparison.csv`: IC by split for each step (v4 → refit every year → 5 seeds
  → new inputs and target → trees → ensemble);
- `results/walk_forward/importance_full_validation.csv`: what each input group adds, rule 3;
- `results/walk_forward/version_comparison.csv` and `verdict.json`: rule 4;
- `results/walk_forward/ic_by_size.csv`: IC by size group, to see if the analyst inputs help large
  stocks, where v4 was weakest (test IC 0.032);
- `results/cost_study_wf_full/factor_alpha.csv`: alpha after the factors.

### c) Previous versions

| Version | Change |
| --- | --- |
| v1 | CSV extracts of CRSP / Compustat; accounting data used 3 months after quarter end; 37 inputs |
| v2 | data fixes (CUSIP merge, empty filings, structural zeros, funds and REITs removed); vol-neutral book and 10% vol target |
| v3 | WRDS data (CRSP common-stock flags, CCM link, true quarterly items, report dates); realistic trading costs |
| v3.1 | income statement at the release, cash flow at the 10-Q / 10-K filing |
| v4 | +65 candidate inputs: balance sheet, value ratios, earnings surprise, size / liquidity, daily returns, short interest, announcement return, 13F (199 inputs instead of 77) |
| v5 | trading one day later, short-sale limit and fee tiers, daily vol forecast, factor alpha; yearly refits, 5 seeds + LightGBM, IBES and regime inputs, target inside vol groups, cost-aware trading |

v4's traded book (F, $100M, realistic costs, v4's backtest: trading at the month-end close, no
short-sale limit, two fee tiers):

| | IC | Net return / yr | Yearly vol | Net Sharpe | Max drawdown |
| --- | --- | --- | --- | --- | --- |
| Validation 2005–14 | 0.111 | 18.8% | 8.1% | 2.31 | −5.4% |
| Test 2015–Nov 2024 | 0.121 | 9.7% | 9.2% | 1.05 | −26.7% |
| Dec 2024 – Nov 2025 (v4's holdout) | — | 8.9% | 10.5% | 0.85 | −8.0% |

v4's IC by size group (validation / test): microcaps 0.135 / 0.162, small 0.052 / 0.057, large
0.038 / 0.032. The signal is strongest where trading costs most, which is why book F weights by
liquidity.

---

## 8. Known limits

- **Accounting data are restated.** Compustat holds today's values, restatements included; only their
  timing is point-in-time. Compustat's point-in-time product (Snapshot) would fix this, but this WRDS
  account has only its free sample (`compsamp_snapshot`, `snapsamp`).
- **Borrow fees are proxies** (section 5), and the 5% ownership limit is a rule of thumb. 13F covers
  institutions with more than $100M, and a stock whose CUSIP no filer reports counts as unowned.
- **The test period is not fully clean** (section 6). Only months after November 2025 are.
- **IBES consensus is mid-month.** It is computed around the middle of the month, so at the month-end
  it is about two weeks old. Before the mid-1980s few stocks are covered.
- **Executing at the close of day 1** assumes the whole trade gets that day's closing price at the
  modelled cost. A real fund would spread large trades over several days.
- **The cost model is a model.** Its impact coefficient comes from the literature, not from this
  strategy's own trades.
- **The walk-forward uses an expanding window** (all history since 1980). A rolling window was not
  tried.
- **Cost-aware trading is the simplest form** of Gârleanu–Pedersen: one speed for every stock, scaled
  by its cost. It does not model how fast each signal decays.

---

## Appendix: code layout and how to run it

```
src/
  data/
    download_wrds.py            1/5  CRSP monthly, CCM links, Compustat quarterly, Fama-French monthly
    download_wrds_costs.py      2/5  CRSP daily per stock-month (costs), short interest 2006+, first-day returns
    download_filing_dates.py    3/5  SEC filing dates of 10-Q / 10-K
    download_wrds_features.py   4/5  extra Compustat items, daily summaries 1978+, announcement days,
                                     FF daily, short interest 1973+, 13F holdings
    download_wrds_extra.py      5/5  (v5) lists the libraries the account can read (Markit, Snapshot),
                                     FF five factors + momentum, CRSP daily returns 2003+, IBES, FRED rates
  preprocessing/
    build_cost_inputs.py        cost-model inputs per stock-month (+ 13F ownership, borrow tiers)
    build_processed_data.py     the model panel (+ analyst and regime inputs, targets inside vol groups)
    features.py                 feature definitions
    check_processed_data.py     assertions on the panel (timing, universe, inputs, splits)
  models/
    mlp_model.py                v4's network, one fit on 1980-2004 (the baseline)
    feature_importance.py       v4's permutation importance by input group
    walk_forward.py             (v5) yearly refits, 5 seeds, LightGBM, the ensemble, importance
  evaluation/
    backtest.py                 backtest engine: returns, drift-aware turnover, flat or per-stock costs, IC
    portfolio.py                portfolio construction (+ trade timing, short-sale limit, daily vol, book K)
    costs.py                    the per-stock cost model (+ fee tiers)
    sizing_comparison.py        portfolio-construction variants, flat costs
    cost_study.py               books A–K under realistic costs; vol forecast, κ and book chosen on validation
    factor_alpha.py             (v5) alpha after FF5 + momentum + short-term reversal
    compare_versions.py         (v5) v5 against v4 under the rule of section 6
results/                        tables and figures (predictions and data are not committed)
```

Run from the project root, one line at a time (`pip install -r requirements.txt` first). The four v4
downloads come first if `data/raw_wrds/` is empty:

```bash
caffeinate -i python src/data/download_wrds_extra.py --username YOUR_WRDS_USERNAME
python src/preprocessing/build_cost_inputs.py
python src/preprocessing/build_processed_data.py
python src/preprocessing/check_processed_data.py
caffeinate -i python src/models/walk_forward.py
python src/models/walk_forward.py --importance
python src/evaluation/cost_study.py --run mlp_filing
python src/evaluation/cost_study.py --run wf_full
python src/evaluation/factor_alpha.py --run mlp_filing
python src/evaluation/factor_alpha.py --run wf_full
python src/evaluation/sizing_comparison.py --run wf_full
python src/evaluation/compare_versions.py
```

- `results/mlp_filing/predictions.parquet` must hold v4's predictions (copied from the v4 folder, or
  rebuilt with `python src/models/mlp_model.py`).
- If rule 3 drops the regime inputs, run `python src/models/walk_forward.py --variants full_noregime`,
  then the cost study and factor alpha with `--run wf_full_noregime`, then `compare_versions.py` again.
- `walk_forward.py` takes several hours (21 yearly fits of 5 networks and the trees, for two variants):
  run it overnight. Each finished year is saved, so it resumes where it stopped. `--quick` is a short
  test run.
- If LightGBM cannot load (on a Mac it needs `brew install libomp`), the trees use scikit-learn's
  histogram boosting instead, and the training log says so.
- The new download takes a few hours (IBES and the daily returns). `caffeinate -i` (macOS) keeps the
  computer awake; keep it plugged in with the lid open. A dropped connection is retried, and a rerun
  resumes from the years already saved. Use `caffeinate -i` for `walk_forward.py` too.
- The full walk-forward variants refuse to run on a panel without the IBES inputs. Years already
  saved are reused only if they were made with the same inputs; otherwise they are set aside and
  redone.

Outputs:

- `results/walk_forward/`: yearly predictions and models, IC tables, importance, the v5 / v4 verdict.
- `results/wf_*/`: each walk-forward model's predictions (the input of the cost study).
- `results/cost_study_<run>/`: the books under realistic costs, capacity, sensitivities, the delay
  table, κ and vol choices, monthly book returns, factor alpha, `chosen_book.json`.

The holdout step (`python src/evaluation/cost_study.py --run <the chosen run> --holdout`) is for the
months after November 2025, once they are downloaded. It runs once.
