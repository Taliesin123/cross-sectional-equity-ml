"""
The model: an MLP classifier with an industry embedding.

Inputs: the scaled numeric features and 0/1 flags of data/processed/panel_model_filing.parquet
(v4's list: "v4_inputs" in feature_groups.json) and the stock's industry id (2-digit SIC).
In v5 this script is the baseline: v4's single fit on 1980-2004. The v5 models (yearly refits,
seeds, trees, analyst and regime inputs) are in walk_forward.py. Output: the
probabilities that next month's return falls in the bottom 20%, the middle 60% or the top
20% of that month's cross-section. The trading score is p_top - p_bottom.

Architecture: industry id -> 8-dimensional embedding, concatenated with the inputs ->
Linear(128) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(64) -> BatchNorm -> ReLU ->
Dropout(0.2) -> Linear(3). Cross-entropy with class weights (inverse frequency), AdamW
(lr 1e-3, weight decay 1e-4), batches of 8,192, at most 30 epochs; trained on the training
months only, and the epoch with the lowest validation loss is kept (early stopping,
patience 5).

Run from the project root:
    python src/models/mlp_model.py --diagnostics       # train, predict, save diagnostics
    python src/models/mlp_model.py --load-existing     # reuse results/mlp_filing/best_model.pt

Outputs in results/mlp_filing/: best_model.pt, predictions.parquet (every stock-month,
holdout included; p_bottom, p_middle, p_top, pred_class, score), training_history.csv,
metrics.json, monthly_ic.csv, ic_summary.csv; with --diagnostics also the equal-weight
decile backtests at 0-50 bps and their figures.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

# mlp_model.py is in src/models/ and imports from src/evaluation/. We add src/
# to sys.path so the absolute imports below work whether the script is run
# directly or imported by another script.
_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from evaluation.backtest import (  # noqa: E402
    compute_decile_returns,
    compute_monthly_ic,
    run_backtest_grid,
    run_long_short_backtest,
    summarize_ic,
)


# ---- paths / reproducibility ----


def get_project_root(start_path=None):
    # walk up until we find both data/ and src/
    if start_path is None:
        start_path = Path.cwd()
    start_path = Path(start_path).resolve()

    for path in [start_path] + list(start_path.parents):
        if (path / "data").exists() and (path / "src").exists():
            return path

    # fallback when this file is in src/models/
    try:
        candidate = Path(__file__).resolve().parents[2]
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    except NameError:
        pass

    raise FileNotFoundError("Could not find project root.")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---- dataset / data loading ----


class EquityDataset(Dataset):
    """Numeric features + industry id + class label."""

    def __init__(self, x_num, industry_id, y):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.industry_id = torch.tensor(industry_id, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.industry_id[idx], self.y[idx]


PANEL_NAME = "filing"


def load_panel_and_features(
    panel_name=PANEL_NAME,
    panel_path=None,
    feature_groups_path=None,
    features=None,
    target_class="label20",
):
    """
    Load the processed panel and the feature list.

    Preprocessing already handled: missing values, winsorization, scaling,
    chronological split, and industry mapping.
    """
    root = get_project_root()
    processed = root / "data" / "processed"

    if panel_path is None:
        panel_path = processed / f"panel_model_{panel_name}.parquet"
        if not panel_path.exists():
            raise FileNotFoundError(f"{panel_path} not found: run src/preprocessing/build_processed_data.py")
    if feature_groups_path is None:
        feature_groups_path = processed / "feature_groups.json"

    panel = pd.read_parquet(panel_path)

    if features is None:
        with open(feature_groups_path, "r") as f:
            fg = json.load(f)
        # v5: the v4 inputs, so this script still reproduces v4's single fit (the baseline);
        # the v5 models with every input are in walk_forward.py
        features = fg.get("v4_inputs") or fg["model_features_numeric_binary"]
    else:
        features = list(features)

    required_cols = list(features) + [
        "industry_id",
        target_class,
        "split",
        "permno",
        "month",
        "ret_fwd1m",
    ]
    missing_cols = [c for c in required_cols if c not in panel.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in panel: {missing_cols}")

    panel = panel.dropna(subset=[target_class]).copy()
    panel["month"] = pd.to_datetime(panel["month"])

    # industry_id must stay categorical, never in numeric features
    if "industry_id" in features:
        raise ValueError("industry_id should not be in numeric features.")

    # The preprocessing pipeline guarantees no NaN / inf, but we check once
    # because a broken upstream change is hard to debug later.
    assert panel[features].isna().sum().sum() == 0, "NaNs in numeric features"
    assert panel["industry_id"].isna().sum() == 0, "NaNs in industry_id"

    return panel, features


def make_arrays(panel, features, target_class="label20"):
    x_num = panel[list(features)].to_numpy(dtype=np.float32)
    industry_id = panel["industry_id"].to_numpy(dtype=np.int64)
    y = panel[target_class].to_numpy(dtype=np.int64)
    split = panel["split"].to_numpy()

    return {
        "x_num": x_num,
        "industry_id": industry_id,
        "y": y,
        "split": split,
        "train_mask": split == "train",
        "val_mask": split == "val",
        "test_mask": split == "test",
        "n_numeric_features": x_num.shape[1],
        "n_industries": int(panel["industry_id"].max()) + 1,
    }


def make_loaders(arrays, batch_size=8192):
    tr = arrays["train_mask"]
    va = arrays["val_mask"]
    te = arrays["test_mask"]

    train_ds = EquityDataset(
        arrays["x_num"][tr], arrays["industry_id"][tr], arrays["y"][tr]
    )
    val_ds = EquityDataset(
        arrays["x_num"][va], arrays["industry_id"][va], arrays["y"][va]
    )
    test_ds = EquityDataset(
        arrays["x_num"][te], arrays["industry_id"][te], arrays["y"][te]
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


# ---- model ----


class MLPWithIndustryEmbedding(nn.Module):
    """
    MLP classifier with a learned industry embedding.

    Output classes: 0 = bottom, 1 = middle, 2 = top return bin.
    """

    def __init__(
        self,
        n_numeric_features,
        n_industries,
        industry_emb_dim=8,
        hidden_dims=(128, 64),
        dropout=0.20,
        n_classes=3,
    ):
        super().__init__()

        self.industry_embedding = nn.Embedding(
            num_embeddings=n_industries,
            embedding_dim=industry_emb_dim,
        )

        input_dim = n_numeric_features + industry_emb_dim
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, industry_id):
        emb = self.industry_embedding(industry_id)
        x = torch.cat([x_num, emb], dim=1)
        return self.net(x)


def build_model(
    n_numeric_features,
    n_industries,
    industry_emb_dim=8,
    hidden_dims=(128, 64),
    dropout=0.20,
    n_classes=3,
    device=None,
):
    if device is None:
        device = get_device()
    model = MLPWithIndustryEmbedding(
        n_numeric_features=n_numeric_features,
        n_industries=n_industries,
        industry_emb_dim=industry_emb_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        n_classes=n_classes,
    ).to(device)
    return model


# ---- training / evaluation ----


def compute_class_weights(y_train, n_classes, device):
    # inverse frequency weighting for the cross-entropy
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"At least one class has zero observations: {counts}")
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_obs = 0

    for x_num_batch, industry_batch, y_batch in loader:
        x_num_batch = x_num_batch.to(device)
        industry_batch = industry_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_num_batch, industry_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        bs = y_batch.size(0)
        total_loss += loss.item() * bs
        n_obs += bs

    return total_loss / n_obs


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_obs = 0
    all_preds, all_targets = [], []

    for x_num_batch, industry_batch, y_batch in loader:
        x_num_batch = x_num_batch.to(device)
        industry_batch = industry_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_num_batch, industry_batch)
        loss = loss_fn(logits, y_batch)
        preds = logits.argmax(dim=1)

        bs = y_batch.size(0)
        total_loss += loss.item() * bs
        n_obs += bs

        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y_batch.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    return {
        "loss": float(total_loss / n_obs),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_mlp(
    arrays,
    features,
    panel_name,
    target_class,
    results_dir,
    device,
    epochs=30,
    batch_size=8192,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5,
    industry_emb_dim=8,
    hidden_dims=(128, 64),
    dropout=0.20,
    n_classes=3,
    use_class_weights=True,
    seed=42,
    verbose=True,
):
    """Train the MLP and save the best checkpoint based on validation loss."""
    set_seed(seed)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = results_dir / "best_model.pt"

    train_loader, val_loader, test_loader = make_loaders(arrays, batch_size=batch_size)

    model = build_model(
        n_numeric_features=arrays["n_numeric_features"],
        n_industries=arrays["n_industries"],
        industry_emb_dim=industry_emb_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        n_classes=n_classes,
        device=device,
    )

    if use_class_weights:
        class_weights = compute_class_weights(
            arrays["y"][arrays["train_mask"]],
            n_classes=n_classes,
            device=device,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        class_weights = None
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = np.inf
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_metrics['loss']:.5f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": list(features),
                    "n_numeric_features": arrays["n_numeric_features"],
                    "n_industries": arrays["n_industries"],
                    "industry_emb_dim": industry_emb_dim,
                    "hidden_dims": list(hidden_dims),
                    "dropout": dropout,
                    "n_classes": n_classes,
                    "target_class": target_class,
                    "panel_name": panel_name,
                    "use_class_weights": use_class_weights,
                    "class_weights": None
                    if class_weights is None
                    else class_weights.detach().cpu().numpy().tolist(),
                    "seed": seed,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "best_epoch": best_epoch,
                    "best_val_loss": float(best_val_loss),
                },
                best_model_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose:
                print("Early stopping.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "training_history.csv", index=False)

    # reload best checkpoint so we evaluate the best model, not the last one
    checkpoint = torch.load(best_model_path, map_location=device)
    model = build_model(
        n_numeric_features=checkpoint["n_numeric_features"],
        n_industries=checkpoint["n_industries"],
        industry_emb_dim=checkpoint["industry_emb_dim"],
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=checkpoint["dropout"],
        n_classes=checkpoint["n_classes"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = evaluate(model, val_loader, loss_fn, device)
    test_metrics = evaluate(model, test_loader, loss_fn, device)

    metrics = {
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "validation": val_metrics,
        "test": test_metrics,
        "training_args": {
            "panel_name": panel_name,
            "target_class": target_class,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "industry_emb_dim": industry_emb_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
            "n_classes": n_classes,
            "use_class_weights": use_class_weights,
            "seed": seed,
        },
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return model, history_df, metrics


def load_mlp_checkpoint(
    checkpoint_path,
    device=None,
    expected_features=None,
    expected_panel_name=None,
    expected_target_class=None,
):
    """Load a checkpoint and fail with a clear error if it is stale.

    A checkpoint trained on another feature list, panel or target is refused rather
    than silently applied to the wrong inputs.
    """
    if device is None:
        device = get_device()

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    required_keys = [
        "model_state_dict",
        "features",
        "n_numeric_features",
        "n_industries",
        "industry_emb_dim",
        "hidden_dims",
        "dropout",
        "n_classes",
    ]
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is incompatible with the current "
            f"mlp_model.py. Missing keys: {missing_keys}. Delete this "
            "checkpoint or rerun with retrain=True / without --load-existing."
        )

    if expected_features is not None and list(checkpoint["features"]) != list(
        expected_features
    ):
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained with a different feature list. "
            "Delete it or rerun with retrain=True."
        )

    if (
        expected_panel_name is not None
        and checkpoint.get("panel_name") != expected_panel_name
    ):
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained on panel "
            f"{checkpoint.get('panel_name')!r}, not {expected_panel_name!r}."
        )

    if (
        expected_target_class is not None
        and checkpoint.get("target_class") != expected_target_class
    ):
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained on target "
            f"{checkpoint.get('target_class')!r}, not {expected_target_class!r}."
        )

    model = build_model(
        n_numeric_features=checkpoint["n_numeric_features"],
        n_industries=checkpoint["n_industries"],
        industry_emb_dim=checkpoint["industry_emb_dim"],
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=checkpoint["dropout"],
        n_classes=checkpoint["n_classes"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# ---- prediction / diagnostics ----


@torch.no_grad()
def predict_proba_numpy(model, x_num, industry_id, batch_size, device):
    model.eval()
    probs_list, preds_list = [], []
    n = len(x_num)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = torch.tensor(x_num[start:end], dtype=torch.float32, device=device)
        ind_batch = torch.tensor(
            industry_id[start:end], dtype=torch.long, device=device
        )

        logits = model(x_batch, ind_batch)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        probs_list.append(probs.detach().cpu().numpy())
        preds_list.append(preds.detach().cpu().numpy())

    return np.vstack(probs_list), np.concatenate(preds_list)


def build_prediction_dataframe(
    panel, probs, pred_class, target_class="label20", keep_only_val_test=False
):
    base_cols = ["permno", "month", "split", "ret_fwd1m", target_class]
    optional_cols = [
        "ticker",
        "trading_symbol",
        "cusip8",
        "mkt_ret",
        "industry_id",
        "mom12m1",
        "reversal_1m",
        "ret_lag1",
    ]
    cols = base_cols + [c for c in optional_cols if c in panel.columns]
    pred_df = panel[cols].copy()

    pred_df["p_bottom"] = probs[:, 0]
    pred_df["p_middle"] = probs[:, 1]
    pred_df["p_top"] = probs[:, 2]
    pred_df["pred_class"] = pred_class.astype(int)
    # main score for the long-short strategy
    pred_df["score"] = pred_df["p_top"] - pred_df["p_bottom"]

    if keep_only_val_test:
        pred_df = pred_df[pred_df["split"].isin(["val", "test"])].copy()

    pred_df = pred_df.sort_values(["month", "permno"]).reset_index(drop=True)
    return pred_df


def compute_mlp_ic_tables(pred_df):
    # Convenience wrapper: combine the two centralized IC helpers into the
    # (ic_df_with_split, summary) pair that the rest of this file expects.
    ic_df = compute_monthly_ic(pred_df)
    if ic_df.empty:
        return ic_df, pd.DataFrame()

    split_map = pred_df[["month", "split"]].drop_duplicates()
    ic_df = ic_df.merge(split_map, on="month", how="left")
    ic_summary = summarize_ic(pred_df, model_name="MLP", score_col="score")
    return ic_df, ic_summary


# ---- main entry point ----


def run_mlp(
    panel_name=PANEL_NAME,
    panel_path=None,
    feature_groups_path=None,
    features=None,
    target_class="label20",
    run_name=None,
    results_dir=None,
    retrain=True,
    use_existing_checkpoint=True,
    save_outputs=True,
    include_train_predictions=True,
    epochs=30,
    batch_size=8192,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5,
    industry_emb_dim=8,
    hidden_dims=(128, 64),
    dropout=0.20,
    n_classes=3,
    use_class_weights=True,
    seed=42,
    device=None,
    verbose=True,
):
    """
    Train or load the MLP and return predictions ready for the backtest.

    Set retrain=False with use_existing_checkpoint=True to skip training
    when iterating on downstream analysis.
    """
    set_seed(seed)

    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if results_dir is None:
        root = get_project_root()
        if run_name is None:
            run_name = f"mlp_{panel_name}"
        results_dir = root / "results" / run_name
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    panel, features = load_panel_and_features(
        panel_name=panel_name,
        panel_path=panel_path,
        feature_groups_path=feature_groups_path,
        features=features,
        target_class=target_class,
    )

    arrays = make_arrays(panel, features, target_class=target_class)

    checkpoint_path = results_dir / "best_model.pt"

    if verbose:
        print("Panel:", panel_name)
        print("Results dir:", results_dir)
        print("Device:", device)
        print("Panel shape:", panel.shape)
        print("Number of features:", len(features))
        print("Number of industries:", arrays["n_industries"])
        print("Train obs:", int(arrays["train_mask"].sum()))
        print("Val obs:", int(arrays["val_mask"].sum()))
        print("Test obs:", int(arrays["test_mask"].sum()))

    should_load = use_existing_checkpoint and checkpoint_path.exists() and not retrain

    if should_load:
        if verbose:
            print(f"Loading existing checkpoint: {checkpoint_path}")
        model, checkpoint = load_mlp_checkpoint(
            checkpoint_path,
            device=device,
            expected_features=features,
            expected_panel_name=panel_name,
            expected_target_class=target_class,
        )

        history_path = results_dir / "training_history.csv"
        history_df = (
            pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        )

        metrics_path = results_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            # recompute val/test metrics if metrics.json is missing
            train_loader, val_loader, test_loader = make_loaders(
                arrays, batch_size=batch_size
            )
            loss_fn = nn.CrossEntropyLoss()
            metrics = {
                "validation": evaluate(model, val_loader, loss_fn, device),
                "test": evaluate(model, test_loader, loss_fn, device),
            }
    else:
        model, history_df, metrics = train_mlp(
            arrays=arrays,
            features=features,
            panel_name=panel_name,
            target_class=target_class,
            results_dir=results_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            industry_emb_dim=industry_emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            n_classes=n_classes,
            use_class_weights=use_class_weights,
            seed=seed,
            verbose=verbose,
        )

    probs, pred_class = predict_proba_numpy(
        model=model,
        x_num=arrays["x_num"],
        industry_id=arrays["industry_id"],
        batch_size=batch_size,
        device=device,
    )

    pred_df = build_prediction_dataframe(
        panel=panel,
        probs=probs,
        pred_class=pred_class,
        target_class=target_class,
        keep_only_val_test=not include_train_predictions,
    )

    ic_df, ic_summary = compute_mlp_ic_tables(pred_df)

    if save_outputs:
        pred_df.to_parquet(results_dir / "predictions.parquet", index=False)
        ic_df.to_csv(results_dir / "monthly_ic.csv", index=False)
        ic_summary.to_csv(results_dir / "ic_summary.csv", index=False)

        run_summary = {
            "panel_name": panel_name,
            "target_class": target_class,
            "results_dir": str(results_dir),
            "n_features": len(features),
            "n_industries": arrays["n_industries"],
            "include_train_predictions": include_train_predictions,
            "metrics": metrics,
            "ic_summary": ic_summary.to_dict(orient="records"),
        }
        with open(results_dir / "run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=4)

    return {
        "model": model,
        "features": list(features),
        "panel": panel,
        "predictions": pred_df,
        "train": pred_df[pred_df["split"] == "train"].copy(),
        "val": pred_df[pred_df["split"] == "val"].copy(),
        "test": pred_df[pred_df["split"] == "test"].copy(),
        "metrics": metrics,
        "history": history_df,
        "ic_df": ic_df,
        "ic_summary": ic_summary,
        "results_dir": results_dir,
        "checkpoint_path": checkpoint_path,
    }


# ---- standalone diagnostics / figures ----


def save_mlp_diagnostic_figures(pred_df, bt_grid, main_bt, results_dir):
    """Save the key MLP figures used in the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cumulative return.
    plt.figure(figsize=(10, 4))
    plt.plot(main_bt["pnl_no_fees"].index, main_bt["pnl_no_fees"], label="Gross")
    plt.plot(main_bt["pnl_with_fees"].index, main_bt["pnl_with_fees"], label="Net")
    plt.xlabel("Month")
    plt.ylabel("Cumulative return")
    plt.title("MLP cumulative return, test, top/bottom 10%, 10 bps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "mlp_cumulative_return_test_top10_10bps.png", dpi=300)
    plt.close()

    # 2. Decile returns.
    decile_ret = compute_decile_returns(pred_df)
    decile_ret.to_csv(results_dir / "mlp_decile_returns_test.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.bar(decile_ret["decile"].astype(str), decile_ret["avg_next_month_return"])
    plt.xlabel("MLP score decile")
    plt.ylabel("Average next-month return")
    plt.title("MLP test decile returns")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(figures_dir / "mlp_decile_returns_test.png", dpi=300)
    plt.close()

    # 3. Transaction-cost sensitivity.
    plt.figure(figsize=(8, 5))
    for bucket, tmp in bt_grid.groupby("bucket_pct"):
        tmp = tmp.sort_values("transaction_cost_bps")
        plt.plot(
            tmp["transaction_cost_bps"],
            tmp["net_sharpe"],
            marker="o",
            label=f"Top/bottom {bucket}%",
        )
    plt.xlabel("Transaction cost, bps")
    plt.ylabel("Net Sharpe ratio")
    plt.title("MLP transaction-cost sensitivity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "mlp_transaction_cost_sensitivity.png", dpi=300)
    plt.close()

    # 4. Portfolio-size robustness at 10 bps.
    sharpe_10bps = bt_grid[bt_grid["transaction_cost_bps"] == 10].sort_values(
        "bucket_pct"
    )
    plt.figure(figsize=(8, 5))
    plt.plot(
        sharpe_10bps["bucket_pct"],
        sharpe_10bps["gross_sharpe"],
        marker="o",
        label="Gross",
    )
    plt.plot(
        sharpe_10bps["bucket_pct"],
        sharpe_10bps["net_sharpe"],
        marker="o",
        label="Net",
    )
    plt.xlabel("Long/short bucket size, %")
    plt.ylabel("Sharpe ratio")
    plt.title("MLP bucket robustness, 10 bps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "mlp_bucket_robustness_10bps.png", dpi=300)
    plt.close()

    # 5. Net drawdown.
    net_returns = main_bt["returns_with_fees"].copy()
    equity_curve = (1 + net_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    drawdown.to_frame("drawdown").to_csv(
        results_dir / "mlp_drawdown_test_top10_10bps.csv"
    )
    plt.figure(figsize=(10, 4))
    plt.plot(drawdown.index, drawdown)
    plt.xlabel("Month")
    plt.ylabel("Drawdown")
    plt.title("MLP net drawdown, test, top/bottom 10%, 10 bps")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "mlp_drawdown_test_top10_10bps.png", dpi=300)
    plt.close()


def run_mlp_diagnostics(
    panel_name=PANEL_NAME,
    target_class="label20",
    run_name=None,
    retrain=False,
    use_existing_checkpoint=True,
    epochs=30,
    batch_size=8192,
    seed=42,
    gross_exposure=2.0,
    save_plots=True,
    verbose=True,
):
    """Train / load the MLP, then save the equal-weight decile backtests (test period, 0 to
    50 bps, top/bottom 10 / 20 / 30%) and their figures."""
    if run_name is None:
        run_name = f"mlp_{panel_name}"

    res = run_mlp(
        panel_name=panel_name,
        target_class=target_class,
        run_name=run_name,
        retrain=retrain,
        use_existing_checkpoint=use_existing_checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        save_outputs=True,
        include_train_predictions=True,
        seed=seed,
        verbose=verbose,
    )

    pred_df = res["predictions"].copy()
    pred_df["month"] = pd.to_datetime(pred_df["month"])
    results_dir = Path(res["results_dir"])

    bt_grid = run_backtest_grid(
        pred_df=pred_df,
        splits=("test",),
        gross_exposure=gross_exposure,
    )
    bt_grid.to_csv(results_dir / "mlp_backtest_grid.csv", index=False)

    main_bt = run_long_short_backtest(pred_df, gross_exposure=gross_exposure)
    main_stats = pd.DataFrame(
        [
            {
                "gross_ann_return": main_bt["stats_no_fees"]["ann_return"],
                "gross_ann_vol": main_bt["stats_no_fees"]["ann_vol"],
                "gross_sharpe": main_bt["stats_no_fees"]["sharpe"],
                "net_ann_return": main_bt["stats_with_fees"]["ann_return"],
                "net_ann_vol": main_bt["stats_with_fees"]["ann_vol"],
                "net_sharpe": main_bt["stats_with_fees"]["sharpe"],
                "avg_monthly_turnover": main_bt["turnover"].mean(),
                "n_months": len(main_bt["returns_with_fees"]),
            }
        ]
    )
    main_stats.to_csv(results_dir / "mlp_main_backtest_stats.csv", index=False)

    with open(results_dir / "metrics_readable.json", "w") as f:
        json.dump(res["metrics"], f, indent=4)

    if save_plots:
        save_mlp_diagnostic_figures(pred_df, bt_grid, main_bt, results_dir)

    if verbose:
        print("\nMLP diagnostics saved.")
        print("Results dir:", results_dir)
        print("Main backtest stats:")
        print(main_stats)

    return {
        "mlp_result": res,
        "backtest_grid": bt_grid,
        "main_backtest": main_bt,
        "main_stats": main_stats,
        "results_dir": results_dir,
    }


# ---- CLI ----


def main():
    parser = argparse.ArgumentParser(
        description="Train the MLP with industry embeddings and optionally save diagnostics."
    )
    parser.add_argument("--target-class", default="label20")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Skip training and load an existing checkpoint if it is compatible.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also save the equal-weight decile backtests and their figures.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="With --diagnostics, save tables but skip PNG figures.",
    )
    parser.add_argument(
        "--gross-exposure",
        type=float,
        default=2.0,
        help="Gross exposure used in the MLP diagnostic backtests.",
    )

    args = parser.parse_args()

    if args.diagnostics:
        res = run_mlp_diagnostics(
            target_class=args.target_class,
            retrain=not args.load_existing,
            use_existing_checkpoint=args.load_existing,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            gross_exposure=args.gross_exposure,
            save_plots=not args.no_plots,
            verbose=True,
        )
        print("\nDone.")
        print("Results dir:", res["results_dir"])
    else:
        res = run_mlp(
            target_class=args.target_class,
            retrain=not args.load_existing,
            use_existing_checkpoint=args.load_existing,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=True,
        )
        print("\nDone.")
        print("Results dir:", res["results_dir"])
        print("Validation metrics:", res["metrics"].get("validation"))
        print("Test metrics:", res["metrics"].get("test"))


if __name__ == "__main__":
    main()
