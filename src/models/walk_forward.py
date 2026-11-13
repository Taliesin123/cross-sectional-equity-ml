"""
Walk-forward retraining: one fit per year instead of one fit on 1980-2004.

For each year Y from 2005 to the last year of data, the models are fitted on every month up
to November of Y-1 (the last month whose next-month return is known at the end of
December) and predict the stocks of every month of year Y. Within each fit, the last 36
months are first held out to set the number of epochs (MLP) and trees (LightGBM) by early
stopping; the models are then refitted on the whole history with those numbers, so the
most recent years are used too (--no-refit keeps the early-stopped models instead). No
other choice is made, and nothing after the fit date is ever used. Predictions for 2005-2014 (validation), 2015-2024 (test) and the
recent months are therefore all out of sample, each made by a model trained on the
history available at that time.

Variants (each writes results/<run>/predictions.parquet, the input of
sizing_comparison.py and cost_study.py --run <run>):
    base   the v4 inputs and target (bottom / middle / top 20% of next month's return),
           MLP with 5 seeds:
             wf_base_seed1   the first seed alone: what retraining alone changes
             wf_base         the 5 seeds' probabilities averaged
    full   all v5 inputs (v4 + analyst forecasts + market regime), the target ranked
           inside the 5 volatility groups the book trades, MLP with 5 seeds +
           gradient-boosted trees (LightGBM) on the within-group return rank:
             wf_full_mlp     MLP seeds averaged
             wf_full_gbm     trees alone
             wf_full         MLP and trees combined (average of their monthly ranks)
    full_noregime (optional, --variants): full without the market-regime inputs, MLP and
           trees (wf_full_noregime). Run it if the regime inputs fail their validation
           test (--importance): it then replaces full.
    full_noibes / full_label20 (optional): the MLP without the analyst inputs / with the
           all-stock target, to isolate those changes.

Run from the project root (after build_processed_data.py):
    python src/models/walk_forward.py                    # base + full (several hours; resumable)
    python src/models/walk_forward.py --variants full    # one variant
    python src/models/walk_forward.py --summary-only     # IC tables from the saved years
    python src/models/walk_forward.py --importance       # input-group importance (validation)
    python src/models/walk_forward.py --quick            # smoke test: 2 years, 1 seed, 2 epochs

Each finished year is saved (results/walk_forward/<variant>/years/), so an interrupted
run resumes where it stopped. The MLP is the v4 network (MLPWithIndustryEmbedding, same settings);
batches are drawn directly from one in-memory array, which is several times faster than
the per-row DataLoader of mlp_model.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from models.mlp_model import MLPWithIndustryEmbedding, get_device, get_project_root  # noqa: E402

FIRST_YEAR = 2005
INNER_VAL_MONTHS = 36
SEEDS = (42, 43, 44, 45, 46)
MLP = dict(hidden_dims=(128, 64), industry_emb_dim=8, dropout=0.20, lr=1e-3, weight_decay=1e-4,
           batch_size=8192, max_epochs=30, patience=5)
GBM = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=1000,
           feature_fraction=0.5, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, max_bin=63,
           n_estimators=2000, early_stopping_rounds=50)
ID_COLS = ["permno", "month", "split", "ret_fwd1m", "label20", "industry_id", "mkt_ret", "vol_group"]

VARIANTS = {
    "base": dict(inputs="v4", mlp_target="label20", gbm=False,
                 runs={"wf_base_seed1": "mlp_seed1", "wf_base": "mlp"}),
    "full": dict(inputs="all", mlp_target="label20_vg", gbm=True, gbm_target="rank_vg",
                 runs={"wf_full_mlp": "mlp", "wf_full_gbm": "gbm", "wf_full": "ensemble"}),
    "full_noregime": dict(inputs="no_regime", mlp_target="label20_vg", gbm=True, gbm_target="rank_vg",
                          runs={"wf_full_noregime_mlp": "mlp", "wf_full_noregime": "ensemble"}),
    "full_noibes": dict(inputs="no_analyst", mlp_target="label20_vg", gbm=False,
                        runs={"wf_full_noibes": "mlp"}),
    "full_label20": dict(inputs="all", mlp_target="label20", gbm=False,
                         runs={"wf_full_label20": "mlp"}),
}


# -------- data --------


def input_columns(fg, which):
    all_inputs = fg["model_features_numeric_binary"]
    analyst = set(fg.get("analyst_features", []))
    regime = set(fg.get("regime_features", []))
    is_analyst = lambda c: c in analyst or (c.startswith("miss_") and c[5:] in analyst)  # noqa: E731
    if which == "v4":
        return list(fg.get("v4_inputs") or [c for c in all_inputs if not is_analyst(c) and c not in regime])
    if which == "all":
        return list(all_inputs)
    if which == "no_regime":
        return [c for c in all_inputs if c not in regime]
    if which == "no_analyst":
        return [c for c in all_inputs if not is_analyst(c)]
    raise ValueError(which)


def load_data(root):
    """Inputs as one float32 matrix (rows sorted by month, permno), the industry ids, and
    the other columns. The inputs are read a few columns at a time, so the peak memory is
    about the matrix itself (the full panel would take several times more as a DataFrame)."""
    import pyarrow.parquet as pq
    processed = root / "data" / "processed"
    fg = json.loads((processed / "feature_groups.json").read_text())
    feats = fg["model_features_numeric_binary"]
    path = processed / "panel_model_filing.parquet"
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [c for c in ("label20_vg", "rank_vg") if c not in names]
    if missing:
        sys.exit(f"{missing} not in the panel: rebuild it (python src/preprocessing/build_processed_data.py)")
    targets = ["label20", "label20_vg", "rank_vg", "rank_all", "vol_group"]
    meta = pd.read_parquet(path, columns=[c for c in dict.fromkeys(ID_COLS + targets) if c in names])
    meta["month"] = pd.to_datetime(meta["month"])
    keep = np.flatnonzero(meta["label20"].notna().to_numpy())
    meta = meta.iloc[keep]
    order = np.lexsort((meta["permno"].to_numpy(), meta["month"].to_numpy()))   # month, then permno
    meta = meta.iloc[order].reset_index(drop=True)
    rows = keep[order]
    X = np.empty((len(rows), len(feats)), dtype=np.float32)
    step = 16
    for a in range(0, len(feats), step):
        part = pq.read_table(path, columns=feats[a:a + step])
        for j, c in enumerate(feats[a:a + step]):
            X[:, a + j] = part.column(c).to_numpy().astype(np.float32, copy=False)[rows]
        del part
    ind = np.array(meta["industry_id"].to_numpy(dtype=np.int64))
    return X, ind, meta, fg, feats


# -------- MLP --------


def class_weights(y, n=3):
    counts = np.bincount(y, minlength=n).astype(np.float32)
    return torch.tensor(counts.sum() / (n * np.maximum(counts, 1)), dtype=torch.float32)


@torch.no_grad()
def predict_proba(model, Xt, It, cols, rows, device, batch=65536):
    model.eval()
    out = []
    for i in range(rows.start, rows.stop, batch):
        j = min(i + batch, rows.stop)
        logits = model(Xt[i:j][:, cols].to(device), It[i:j].to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)


@torch.no_grad()
def eval_loss(model, Xt, It, yt, cols, rows, loss_fn, device, batch=65536):
    model.eval()
    tot, n = 0.0, 0
    for i in range(rows.start, rows.stop, batch):
        j = min(i + batch, rows.stop)
        loss = loss_fn(model(Xt[i:j][:, cols].to(device), It[i:j].to(device)), yt[i:j].to(device))
        tot += float(loss) * (j - i)
        n += j - i
    return tot / max(n, 1)


def train_mlp(Xt, It, yt, cols, fit, val, n_ind, seed, device, max_epochs, patience, log=None):
    """Fit on rows fit (a range). With val (a range): early stopping on its loss, the best
    epoch's weights are returned. With val None: exactly max_epochs epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPWithIndustryEmbedding(len(cols), n_ind, industry_emb_dim=MLP["industry_emb_dim"],
                                     hidden_dims=MLP["hidden_dims"], dropout=MLP["dropout"]).to(device)
    w = class_weights(yt[fit.start:fit.stop].numpy()).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=MLP["lr"], weight_decay=MLP["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    cols_t = torch.as_tensor(cols)
    all_cols = len(cols) == Xt.shape[1] and list(cols) == list(range(Xt.shape[1]))
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(fit.stop - fit.start, generator=gen) + fit.start
        for i in range(0, len(perm), MLP["batch_size"]):
            idx = perm[i:i + MLP["batch_size"]]
            xb = Xt.index_select(0, idx)
            xb = (xb if all_cols else xb.index_select(1, cols_t)).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, It[idx].to(device)), yt[idx].to(device))
            loss.backward()
            opt.step()
        if val is None:
            if log is not None:
                log.append({"epoch": epoch, "val_loss": np.nan, "seconds": round(time.time() - t0, 1)})
            continue
        v = eval_loss(model, Xt, It, yt, cols_t, val, loss_fn, device)
        if log is not None:
            log.append({"epoch": epoch, "val_loss": v, "seconds": round(time.time() - t0, 1)})
        if v < best - 1e-7:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if val is None:
        return model, max_epochs, np.nan
    model.load_state_dict(best_state)
    return model, best_epoch, best


# -------- trees --------


def _lightgbm():
    """The lightgbm module, or None if it cannot be loaded (not installed, or on a Mac
    without the OpenMP library: brew install libomp). The trees then use sklearn."""
    try:
        import lightgbm
        return lightgbm
    except (ImportError, OSError) as e:
        if not getattr(_lightgbm, "warned", False):
            print(f"LightGBM unavailable ({type(e).__name__}: {str(e).splitlines()[0][:120]}); "
                  "using sklearn's HistGradientBoostingRegressor instead.", flush=True)
            _lightgbm.warned = True
        return None


def _stack(X, ind, cols, rows, chunk=250_000):
    """float32 array of X[rows, cols] with the industry id as a last column, filled in
    chunks (no full-size temporary)."""
    out = np.empty((rows.stop - rows.start, len(cols) + 1), dtype=np.float32)
    all_cols = list(cols) == list(range(X.shape[1]))
    for a in range(rows.start, rows.stop, chunk):
        b = min(a + chunk, rows.stop)
        out[a - rows.start:b - rows.start, :-1] = X[a:b] if all_cols else X[a:b][:, cols]
        out[a - rows.start:b - rows.start, -1] = ind[a:b]
    return out


def train_gbm(X, ind, target, cols, fit, val, n_threads, quick=False, refit=True):
    """LightGBM on the within-group return rank (sklearn's histogram boosting if
    lightgbm cannot be loaded): the number of trees is set by early stopping on val, then
    (refit) the model is fitted again on fit + val with that number. Returns a
    predict(rows) function, the number of trees and the engine.

    Memory: LightGBM reads the history in batches and bins it once (no full copy of the
    inputs, which take several GB); the fit and val parts are subsets of the binned data."""
    assert fit.stop == val.start
    n_fit = fit.stop - fit.start
    rows = range(fit.start, val.stop)
    ya = np.ascontiguousarray(target[rows.start:rows.stop], dtype=np.float32)
    cat = len(cols)                    # the industry id, after the inputs
    rounds = 30 if quick else GBM["n_estimators"]

    def predict_rows(rows_p):
        return _stack(X, ind, cols, rows_p)

    lgb = _lightgbm()
    if lgb is not None:
        class Rows(lgb.Sequence):
            """The history read in batches: LightGBM bins it without a full copy."""
            batch_size = 65536

            def __len__(self):
                return len(rows)

            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    a, b, _ = idx.indices(len(rows))
                    return _stack(X, ind, cols, range(rows.start + a, rows.start + b))
                # single rows and lists are used to sample the bin edges: LightGBM wants float64
                if isinstance(idx, (list, np.ndarray)):
                    ii = np.asarray(idx, dtype=np.int64) + rows.start
                    return np.column_stack([X[ii][:, cols], ind[ii]]).astype(np.float64)
                i = int(idx) + rows.start
                return np.append(X[i, cols].astype(np.float64), float(ind[i]))

        params = {k: v for k, v in GBM.items() if k not in ("n_estimators", "early_stopping_rounds")}
        params.update(verbose=-1, num_threads=n_threads, seed=42,
                      max_bin_by_feature=[GBM["max_bin"]] * len(cols) + [255])   # 81 industries
        full = lgb.Dataset([Rows()], ya, categorical_feature=[cat], free_raw_data=True, params=params)
        full.construct()
        dtr = full.subset(np.arange(n_fit, dtype=np.int32))
        dva = full.subset(np.arange(n_fit, len(ya), dtype=np.int32))
        booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(GBM["early_stopping_rounds"], verbose=False)])
        best = booster.best_iteration or rounds
        del dtr, dva
        if refit:
            booster = lgb.train(params, full, num_boost_round=best)
        del full

        def predict(rows_p):
            return booster.predict(predict_rows(rows_p), num_iteration=best)
        return predict, best, "lightgbm"
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xa = _stack(X, ind, cols, rows)
        Xf, yf, Xv, yv = Xa[:n_fit], ya[:n_fit], Xa[n_fit:], ya[n_fit:]
        m = HistGradientBoostingRegressor(learning_rate=GBM["learning_rate"], max_leaf_nodes=GBM["num_leaves"],
                                          min_samples_leaf=GBM["min_data_in_leaf"], l2_regularization=GBM["lambda_l2"],
                                          max_bins=255, categorical_features=[cat], warm_start=True,
                                          early_stopping=False, max_iter=0, random_state=42)
        best, best_loss, it, bad = 0, np.inf, 0, 0
        step = 10 if quick else 50
        while it < rounds:
            it += step
            m.set_params(max_iter=it)
            m.fit(Xf, yf)
            loss = float(np.mean((m.predict(Xv) - yv) ** 2))
            if loss < best_loss - 1e-9:
                best_loss, best, bad = loss, it, 0
            else:
                bad += 1
                if bad >= 2:
                    break
        m.set_params(max_iter=max(best, step), warm_start=False)
        if refit:
            m.fit(Xa, ya)
        else:
            m.fit(Xf, yf)
        del Xa, Xf, Xv

        def predict(rows_p):
            return m.predict(predict_rows(rows_p))
        return predict, best, "sklearn-hgb"


# -------- one variant, year by year --------


def month_rows(meta):
    """month -> range of row positions (the data are sorted by month)."""
    m = meta["month"].to_numpy()
    starts = np.flatnonzero(np.r_[True, m[1:] != m[:-1]])
    ends = np.r_[starts[1:], len(m)]
    return pd.Series([range(s, e) for s, e in zip(starts, ends)], index=pd.DatetimeIndex(m[starts]))


def work_dir(out_root, name, quick=False):
    """Per-variant working folder (years, models, log); --quick runs never mix with real ones."""
    return out_root / ("walk_forward_quick" if quick else "walk_forward") / name


def saved_input_count(out):
    """Number of inputs of the networks already saved in out/models (None if none)."""
    files = sorted((out / "models").glob("*.pt"))
    if not files:
        return None
    state = torch.load(files[0], map_location="cpu")
    w = [v for k, v in state.items() if k.endswith("weight") and v.ndim == 2 and "emb" not in k]
    return int(w[0].shape[1]) - MLP["industry_emb_dim"] if w else None


def check_saved_years(out, signature, n_inputs):
    """Saved years are reused only if they were made with the same inputs and targets
    (after a rebuild of the panel with new inputs, they are set aside and redone)."""
    sig_path = out / "inputs.json"
    stale = None
    if sig_path.exists():
        if json.loads(sig_path.read_text()) != signature:
            stale = "the inputs or targets changed"
    else:
        n_saved = saved_input_count(out)
        if n_saved is not None and n_saved != n_inputs:
            stale = f"the saved networks have {n_saved} inputs, the panel now gives {n_inputs}"
    if stale:
        moved = out.parent / f"{out.name}_stale_{time.strftime('%Y%m%d-%H%M')}"
        out.rename(moved)
        print(f"  {stale}: the years already done were moved to {moved.name} and are redone")
        (out / "years").mkdir(parents=True)
        (out / "models").mkdir()
    sig_path.write_text(json.dumps(signature, indent=1))


def run_variant(name, spec, X, ind, meta, fg, feats, out_root, args, device):
    if spec["inputs"] in ("all", "no_regime") and not fg.get("analyst_features") and not args.allow_no_ibes:
        print(f"\n=== {name}: SKIPPED. The panel has no analyst (IBES) inputs, so this would not be the v5 "
              "model. Download IBES (python src/data/download_wrds_extra.py --username <you> --only ibes), "
              "rebuild the panel, then run this again (or pass --allow-no-ibes).")
        return
    out = work_dir(out_root, name, args.quick)
    (out / "years").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    pos = {f: i for i, f in enumerate(feats)}
    names = input_columns(fg, spec["inputs"])
    cols = [pos[c] for c in names]
    check_saved_years(out, {"inputs": names, "mlp_target": spec["mlp_target"],
                            "gbm_target": spec.get("gbm_target") if spec["gbm"] else None}, len(cols))
    rows_of = month_rows(meta)
    months = rows_of.index
    y_mlp = np.array(meta[spec["mlp_target"]].astype("int64").to_numpy())
    Xt, It, yt = torch.from_numpy(X), torch.from_numpy(ind), torch.from_numpy(y_mlp)
    n_ind = int(ind.max()) + 1
    seeds = SEEDS[:args.seeds]
    last_year = months.max().year if args.last_year is None else args.last_year
    years = [y for y in range(args.first_year, last_year + 1) if (months.year == y).any()]
    if args.quick:
        years = years[:2]
    print(f"\n=== {name}: {len(cols)} inputs, MLP target {spec['mlp_target']}, {len(seeds)} seeds"
          + (f", trees on {spec['gbm_target']}" if spec["gbm"] else "") + f", years {years[0]}-{years[-1]}")
    for year in years:
        path = out / "years" / f"{year}.parquet"
        if path.exists():
            print(f"{year}: done already")
            continue
        t0 = time.time()
        cutoff = pd.Timestamp(f"{year - 1}-11-30")
        fit_months = months[months <= cutoff]
        if len(fit_months) < INNER_VAL_MONTHS + 60:
            print(f"{year}: not enough history, skipped")
            continue
        val_start = fit_months[-INNER_VAL_MONTHS]
        fit = range(0, rows_of[val_start].start)
        val = range(rows_of[val_start].start, rows_of[fit_months[-1]].stop)
        pm = months[months.year == year]
        pred_rows = range(rows_of[pm[0]].start, rows_of[pm[-1]].stop)
        res = meta.iloc[pred_rows.start:pred_rows.stop][["permno", "month"]].copy()
        info = {"year": year, "fit_rows": len(fit), "val_rows": len(val), "pred_rows": len(pred_rows),
                "fit_end": str(fit_months[-INNER_VAL_MONTHS - 1].date()), "val_end": str(fit_months[-1].date())}
        probs = []
        max_ep = 2 if args.quick else MLP["max_epochs"]
        if not args.no_refit:
            # 1) the number of epochs: early stopping on the last 36 months (first seed);
            # 2) every seed refitted on all the history (to November of Y-1) for that many
            # epochs, so the most recent three years are used too
            log = []
            _, ep_star, vl = train_mlp(Xt, It, yt, cols, fit, val, n_ind, seeds[0], device, max_ep,
                                       MLP["patience"], log)
            info["early_stop"] = {"best_epoch": ep_star, "val_loss": vl, "epochs": len(log),
                                  "sec_per_epoch": float(np.mean([e["seconds"] for e in log]))}
        for k, seed in enumerate(seeds, 1):
            log = []
            if args.no_refit:
                model, ep, vl = train_mlp(Xt, It, yt, cols, fit, val, n_ind, seed, device, max_ep,
                                          MLP["patience"], log)
            else:
                model, ep, vl = train_mlp(Xt, It, yt, cols, range(fit.start, val.stop), None, n_ind, seed,
                                          device, max(ep_star, 1), MLP["patience"], log)
            p = predict_proba(model, Xt, It, torch.as_tensor(cols), pred_rows, device)
            probs.append(p)
            res[f"score_s{k}"] = (p[:, 2] - p[:, 0]).astype("float32")
            torch.save(model.state_dict(), out / "models" / f"{year}_seed{seed}.pt")
            info[f"seed{seed}"] = {"epochs": ep, "val_loss": vl,
                                   "sec_per_epoch": float(np.mean([e["seconds"] for e in log]))}
        pm_ = np.mean(probs, axis=0)
        res["p_bottom"], res["p_middle"], res["p_top"] = pm_[:, 0], pm_[:, 1], pm_[:, 2]
        res["score_mlp"] = (pm_[:, 2] - pm_[:, 0]).astype("float32")
        if spec["gbm"]:
            tgt = meta[spec["gbm_target"]].to_numpy(dtype=np.float32)
            predict, rounds, engine = train_gbm(X, ind, tgt, cols, fit, val, args.threads, args.quick,
                                                refit=not args.no_refit)
            res["score_gbm"] = predict(pred_rows).astype("float32")
            info["gbm"] = {"rounds": int(rounds), "engine": engine}
        res.to_parquet(path, index=False)
        info["minutes"] = round((time.time() - t0) / 60, 1)
        with open(out / "training_log.jsonl", "a") as f:
            f.write(json.dumps(info) + "\n")
        eps = [info[f"seed{s}"]["epochs"] for s in seeds]
        how = (f"early stop on {len(val):,} rows to {info['val_end']}, then {len(seeds)} seeds refitted on "
               f"{len(fit) + len(val):,} rows for {eps[0]} epochs" if not args.no_refit else
               f"early stop on {len(val):,} rows, best epochs {eps}")
        print(f"{year}: fit {len(fit):,} rows to {info['fit_end']}; {how}"
              + (f"; trees {info['gbm']['rounds']}" if spec["gbm"] else "") + f"; {info['minutes']} min", flush=True)
    write_runs(name, spec, meta, out_root, args.quick)


def month_rank(df, col):
    return df.groupby("month")[col].rank(pct=True)


def write_runs(name, spec, meta, out_root, quick=False):
    files = sorted((work_dir(out_root, name, quick) / "years").glob("*.parquet"))
    if not files:
        return
    pred = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    pred = pred.merge(meta[[c for c in ID_COLS if c in meta.columns]], on=["permno", "month"], how="left")
    if "score_gbm" in pred:
        pred["score_ensemble"] = (month_rank(pred, "score_mlp") + month_rank(pred, "score_gbm")) / 2
    source = {"mlp": "score_mlp", "mlp_seed1": "score_s1", "gbm": "score_gbm", "ensemble": "score_ensemble"}
    for run, kind in spec["runs"].items():
        col = source[kind]
        if col not in pred:
            continue
        d = out_root / (f"quick_{run}" if quick else run)
        d.mkdir(parents=True, exist_ok=True)
        keep = [c for c in ID_COLS + ["p_bottom", "p_middle", "p_top", "score_mlp", "score_gbm", "score_s1"]
                if c in pred.columns]
        out = pred[keep].copy()
        out["score"] = pred[col].astype("float32")
        out.to_parquet(d / "predictions.parquet", index=False)
        ic_table(out).to_csv(d / "ic_summary.csv", index=False)
        print(f"  wrote results/{d.name}/predictions.parquet ({len(out):,} rows, "
              f"{out['month'].min():%Y-%m} -> {out['month'].max():%Y-%m})")


# -------- IC summaries --------


def group_spearman(df, keys, a, b, min_n=30):
    """Spearman correlation of columns a and b inside each group of keys (vectorised)."""
    d = df[keys + [a, b]].dropna()
    g = d.groupby(keys)
    ra, rb = g[a].rank(), g[b].rank()
    ra = ra - ra.groupby([d[k] for k in keys]).transform("mean")
    rb = rb - rb.groupby([d[k] for k in keys]).transform("mean")
    grp = [d[k] for k in keys]
    num = (ra * rb).groupby(grp).sum()
    den = np.sqrt((ra ** 2).groupby(grp).sum() * (rb ** 2).groupby(grp).sum())
    n = ra.groupby(grp).size()
    return (num / den).where(n >= min_n)


def monthly_ic(df, score="score", by_group=False):
    if not by_group:
        return group_spearman(df, ["month"], score, "ret_fwd1m")
    # rank IC inside each volatility group, averaged over groups (what the book trades)
    return group_spearman(df, ["month", "vol_group"], score, "ret_fwd1m").groupby(level=0).mean()


def ic_table(pred):
    rows = []
    has_vg = "vol_group" in pred.columns and pred["vol_group"].notna().any()
    ic = monthly_ic(pred)
    ic_vg = monthly_ic(pred, by_group=True) if has_vg else None
    split = pred.drop_duplicates("month").set_index("month")["split"]
    for s in ("val", "test", "recent", "holdout"):
        m = split[split == s].index
        if not len(m):
            continue
        x = ic.reindex(m).dropna()
        row = {"split": s, "months": len(x), "ic": x.mean(), "ic_t": x.mean() / x.std() * np.sqrt(len(x))}
        if ic_vg is not None:
            v = ic_vg.reindex(m).dropna()
            row.update({"ic_in_vol_groups": v.mean(), "ic_in_vol_groups_t": v.mean() / v.std() * np.sqrt(len(v))})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(out_root, meta, compare):
    runs = []
    if compare is not None and compare.exists():
        p = pd.read_parquet(compare, columns=["permno", "month", "score"])
        p["month"] = pd.to_datetime(p["month"])
        p = p.merge(meta[["permno", "month", "split", "ret_fwd1m", "vol_group"]], on=["permno", "month"], how="inner")
        runs.append(("v4 (one fit, 1980-2004)", p))
    for run in ("wf_base_seed1", "wf_base", "wf_full_mlp", "wf_full_gbm", "wf_full",
                "wf_full_noregime_mlp", "wf_full_noregime", "wf_full_noibes", "wf_full_label20"):
        f = out_root / run / "predictions.parquet"
        if f.exists():
            runs.append((run, pd.read_parquet(f, columns=["permno", "month", "split", "ret_fwd1m", "vol_group", "score"])))
    if not runs:
        return None
    # the same stock-months for every run (the months all of them predict)
    common = None
    for _, p in runs:
        k = set(zip(p["permno"], p["month"]))
        common = k if common is None else common & k
    rows = []
    for name, p in runs:
        p = p[[(a, b) in common for a, b in zip(p["permno"], p["month"])]]
        t = ic_table(p)
        t.insert(0, "run", name)
        rows.append(t)
    table = pd.concat(rows, ignore_index=True)
    (out_root / "walk_forward").mkdir(exist_ok=True)
    table.to_csv(out_root / "walk_forward" / "ic_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nRank IC on the same stock-months (t-statistics over months):")
        print(table.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    return table


# -------- importance of the input groups (validation years) --------


def importance(out_root, X, ind, meta, fg, feats, device, variant="full"):
    """Permutation importance of the walk-forward MLP (seeds averaged) on validation
    years, each year scored by its own models: drop in the monthly rank IC when a group of
    inputs is shuffled across stocks within each month (as feature_importance.py), and,
    for the market-regime inputs (the same for every stock), when each month gets the
    regime values of a randomly drawn earlier month (only their timing can matter)."""
    from models.feature_importance import group_of
    from preprocessing.build_processed_data import CASH_FLOW_FEATURES

    spec = VARIANTS[variant]
    names = input_columns(fg, spec["inputs"])
    pos = {f: i for i, f in enumerate(feats)}
    cols = [pos[c] for c in names]
    groups = {}
    for j, c in enumerate(names):
        groups.setdefault(group_of(c, set(CASH_FLOW_FEATURES)), []).append(j)
    groups["industry (embedding)"] = None
    rows_of = month_rows(meta)
    first_rows = np.asarray([rows_of[m].start for m in rows_of.index[rows_of.index.year < 2015]])
    regime_pool = X[first_rows][:, cols]
    rng = np.random.default_rng(0)
    drops = []
    for year in range(2005, 2015):
        files = sorted((work_dir(out_root, variant) / "models").glob(f"{year}_seed*.pt"))
        if not files:
            continue
        pm = rows_of.index[rows_of.index.year == year]
        rows = range(rows_of[pm[0]].start, rows_of[pm[-1]].stop)
        Xy = X[rows.start:rows.stop][:, cols].copy()
        Iy = ind[rows.start:rows.stop]
        sub = meta.iloc[rows.start:rows.stop][["month", "ret_fwd1m"]].reset_index(drop=True)
        month_idx = sub["month"].to_numpy()
        starts = np.flatnonzero(np.r_[True, month_idx[1:] != month_idx[:-1]])
        ends = np.r_[starts[1:], len(month_idx)]
        models = []
        for f in files:
            m = MLPWithIndustryEmbedding(len(cols), int(ind.max()) + 1, industry_emb_dim=MLP["industry_emb_dim"],
                                         hidden_dims=MLP["hidden_dims"], dropout=MLP["dropout"]).to(device)
            m.load_state_dict(torch.load(f, map_location=device))
            models.append(m)
        all_cols = torch.arange(len(cols))

        def score(Xs, Is):
            Xt, It = torch.from_numpy(np.ascontiguousarray(Xs)), torch.from_numpy(np.ascontiguousarray(Is))
            p = np.mean([predict_proba(m, Xt, It, all_cols, range(0, len(Xs)), device) for m in models], axis=0)
            return p[:, 2] - p[:, 0]

        base = group_spearman(sub.assign(s=score(Xy, Iy)), ["month"], "s", "ret_fwd1m")
        for gname, gcols in groups.items():
            perm = np.arange(len(sub))
            for a, b in zip(starts, ends):
                perm[a:b] = a + rng.permutation(b - a)
            Xs, Is = Xy, Iy
            if gname == "industry (embedding)":
                Is = Iy[perm]
            elif gname == "market regime":
                Xs = Xy.copy()
                for a, b in zip(starts, ends):
                    Xs[a:b, gcols] = regime_pool[rng.integers(len(regime_pool)), gcols]
            else:
                Xs = Xy.copy()
                Xs[:, gcols] = Xy[perm][:, gcols]
            ic = group_spearman(sub.assign(s=score(Xs, Is)), ["month"], "s", "ret_fwd1m")
            drops.append(pd.DataFrame({"group": gname, "month": base.index, "ic_drop": (base - ic).values,
                                       "base_ic": base.values}))
        print(f"  {year}: {len(groups)} groups scored", flush=True)
    if not drops:
        print(f"No saved models of the {variant} variant for 2005-2014: run the walk-forward first.")
        return None
    d = pd.concat(drops)
    t = d.groupby("group").agg(ic_drop=("ic_drop", "mean"), sd=("ic_drop", "std"), months=("ic_drop", "count"))
    t["t"] = t["ic_drop"] / t["sd"] * np.sqrt(t["months"])
    t["base_ic"] = d.groupby("month")["base_ic"].first().mean()
    t = t.drop(columns="sd").sort_values("ic_drop", ascending=False)
    (out_root / "walk_forward").mkdir(exist_ok=True)
    t.to_csv(out_root / "walk_forward" / f"importance_{variant}_validation.csv")
    print(f"\nValidation 2005-2014, walk-forward MLP ({variant}): drop in monthly rank IC when the group is scrambled")
    print(t.to_string(float_format=lambda x: f"{x:8.4f}"))
    return t


def main():
    parser = argparse.ArgumentParser(description="Walk-forward retraining (one fit per year).")
    parser.add_argument("--variants", nargs="+", default=["base", "full"], choices=list(VARIANTS))
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--last-year", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto (GPU / Apple MPS if available), cpu, mps, cuda")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
    parser.add_argument("--compare", default=None,
                        help="predictions of a fixed model to compare with "
                             "(default: results/mlp_filing/predictions.parquet if it exists)")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--importance", action="store_true")
    parser.add_argument("--quick", action="store_true", help="smoke test: 2 years, 1 seed, 2 epochs")
    parser.add_argument("--allow-no-ibes", action="store_true",
                        help="run the full variants even if the panel has no analyst inputs")
    parser.add_argument("--no-refit", action="store_true",
                        help="keep each seed's early-stopped model (trained without the last 36 months)")
    args = parser.parse_args()
    if args.quick:
        args.seeds = 1
    import os
    args.threads = args.threads or os.cpu_count() or 4
    torch.set_num_threads(args.threads)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    root = get_project_root()
    out_root = root / "results"
    print("device:", device, "| threads:", args.threads)
    X, ind, meta, fg, feats = load_data(root)
    print(f"panel: {len(meta):,} stock-months, {len(feats)} inputs available")
    if args.importance:
        for name in args.variants:
            if name != "base":
                importance(out_root, X, ind, meta, fg, feats, device, name)
        return
    if not args.summary_only:
        for name in args.variants:
            run_variant(name, VARIANTS[name], X, ind, meta, fg, feats, out_root, args, device)
    else:
        for name in args.variants:
            write_runs(name, VARIANTS[name], meta, out_root)
    if args.quick:
        print("\n--quick: smoke test only; outputs in results/walk_forward_quick and results/quick_*")
        return
    compare = Path(args.compare) if args.compare else out_root / "mlp_filing" / "predictions.parquet"
    summarize(out_root, meta, compare)


if __name__ == "__main__":
    main()
