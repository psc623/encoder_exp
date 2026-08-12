"""Exact linear classification channel, ported from bsnip2-smri-classification's
classify.py (github.com/uchicago-wjsh/bsnip2-smri-classification).

Why this exists: AttentionPoolHead pools the frozen encoder's 64 spatial tokens
through a learned softmax attention before the classifier ever sees them -- a
lossy compression step. This module instead flattens all 64*width pooled
features into one vector and fits an *exact* L2 logistic regression directly
on it, using the same row-space (Gram-eigendecomposition) trick
bsnip2-smri-classification/classification/classify.py uses for its ~5e5-voxel
features: the fit is mathematically identical to ordinary sklearn
LogisticRegression on the full P-dimensional vector, just solved in the
n-sample space, which is exact (not an approximation) whenever P > n.

One deliberate protocol difference from classify.py: that script tunes C by
k-fold CV on the train split because it has no separate validation split.
This framework already carves out a dedicated validation split (same one
AttentionPoolHead's weight_decay grid uses), so C is tuned against that split
instead -- consistent with the rest of encoderbench's model-selection
protocol, not a downgrade of classify.py's method.

The fitted (w, b) is exposed as a LinearPoolHead so finetune_variants.py can
warm-start from it exactly the way run_finetune warm-starts AttentionPoolHead
from run_probe's checkpoint.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from encoderbench.cache import FeatureCache, load_cache
from encoderbench.training import _class_weights, _labels, _summary
from encoderbench.utils import ensure_parent, set_seed, write_json

DEFAULT_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


class LinearPoolHead:
    """Flatten (n_tokens, width) tokens -> exact linear score. Same forward
    contract as AttentionPoolHead (tokens in, 2-class logits out) so it drops
    into the same loss / _infer / _summary machinery, but has no hidden layer:
    the positive-class logit is w . flatten(z-scored tokens) + b; the negative
    logit is pinned at 0, which makes the 2-class softmax reproduce
    sigmoid(w.x+b) exactly (the standard binary-LR/2-class-softmax identity).

    n_tokens is 64 for the pooled_grid=[4,4,4] cache and e.g. 4096 for MASS's
    native (unpooled) layer-3 cache -- not hardcoded, since the whole point of
    the native path is to not silently re-impose a fixed token count.
    """

    def __init__(self, n_tokens: int, width: int, mean: "torch.Tensor", std: "torch.Tensor"):
        import torch
        from torch import nn

        self.n_tokens = n_tokens
        self.width = width
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)
        self.linear = nn.Linear(n_tokens * width, 2)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()

    def set_weights(self, w: np.ndarray, b: float) -> None:
        import torch

        with torch.no_grad():
            self.linear.weight[1].copy_(torch.from_numpy(w.astype(np.float32)))
            self.linear.bias[1].fill_(float(b))
            self.linear.weight[0].zero_()
            self.linear.bias[0].zero_()

    def to_module(self) -> "torch.nn.Module":
        """Wrap as a proper nn.Module (mean/std as buffers) for finetune_variants.py."""
        import torch
        from torch import nn

        width, mean, std, linear = self.width, self.mean, self.std, self.linear

        class _Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", mean)
                self.register_buffer("std", std)
                self.linear = linear

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                normalized = (tokens.float() - self.mean) / self.std
                flat = normalized.reshape(normalized.shape[0], -1)
                return self.linear(flat)

        return _Wrapped()


def trainable_parameter_count_linear(n_tokens: int, width: int) -> int:
    return n_tokens * width * 2 + 2


def _flatten_and_zscore(features: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    normalized = (features - mean) / sd
    return normalized.reshape(normalized.shape[0], -1).astype(np.float32)


def _zfit(flat_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = flat_train.mean(0, dtype=np.float64).astype(np.float32)
    sd = flat_train.std(0, dtype=np.float64).astype(np.float32)
    sd[sd == 0] = 1.0
    return mu, sd


def _lr_fit_exact(Z: np.ndarray, y: np.ndarray, C: float) -> tuple[np.ndarray, float]:
    """Row-space-whitened exact L2 logistic regression -- identical weight
    vector to sklearn LogisticRegression(C=C) fit on the full P-dim Z, ported
    from bsnip2-smri-classification/classification/classify.py:lr_backend.
    """
    from sklearn.linear_model import LogisticRegression

    K = Z @ Z.T
    lam, U = np.linalg.eigh(K)
    lam, U = lam[::-1], U[:, ::-1]
    s = np.sqrt(np.clip(lam, 0, None))
    keep = s > (1e-6 * s[0] if s[0] > 0 else 0)
    U, s = U[:, keep], s[keep]
    W = (U / s).astype(np.float32)
    Ftr = (U * s).astype(np.float32)
    model = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=5000)
    model.fit(Ftr, y)
    # alpha lives in the whitened n-dim space; map back to the P-dim voxel/feature
    # space via W (= U / s), exactly as classify.py's `tf` transform does for new points.
    alpha = model.coef_[0] @ W.T  # (n,) dual-style coefficients in training-sample space
    w = alpha @ Z                  # (P,) primal weight vector -- exact, not approximate
    b = float(model.intercept_[0])
    return w.astype(np.float32), b


def _decision_scores(Z: np.ndarray, Ztr: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return Z @ w + b


def run_linear_probe(cache_path: str | Path, output_dir: str | Path, positive: str, seed: int,
                     evaluation: dict[str, Any], c_grid: tuple[float, ...] = DEFAULT_C_GRID,
                     shuffled: bool = False, splits_override: np.ndarray | None = None,
                     tag_suffix: str = "", inner_folds: int = 0) -> dict[str, Any]:
    """inner_folds > 0 switches C selection from "score the single held-out
    validation split" to smri's own protocol (`classify.py:tune_C`): stratified
    k-fold CV *inside* the training fold, averaging ROC-AUC over folds. That
    removes the dependence on one small validation subset, which is the whole
    point of adopting the repeated-split protocol.
    """
    cache = load_cache(cache_path)
    if splits_override is not None:
        if len(splits_override) != len(cache.splits):
            raise ValueError(f"splits_override has {len(splits_override)} rows, "
                             f"cache has {len(cache.splits)}")
        cache.splits = np.asarray(splits_override)
        cache.validate()
    set_seed(seed)
    y = _labels(cache, positive)
    split_indices = {split: np.flatnonzero(cache.splits == split)
                     for split in ("train", "validation", "test")}
    train_y = y[split_indices["train"]].copy()
    if shuffled:
        train_y = train_y[np.random.default_rng(seed).permutation(len(train_y))]
    val_y = y[split_indices["validation"]]

    n_tokens = cache.features.shape[1]
    width = cache.features.shape[-1]
    # Same token-level normalization AttentionPoolHead uses, so the two heads
    # see numerically comparable inputs -- only the architecture differs.
    from encoderbench.training import token_normalization

    if inner_folds > 0:
        # smri's selection method (classify.py:tune_C): C chosen by stratified
        # k-fold CV *inside the training fold*, scored by ROC-AUC averaged over
        # folds, ties broken toward stronger regularization -- no dependence on
        # one small held-out validation subset.
        #
        # Deliberately restricted to the `train` rows rather than train+validation:
        # every other mode trains on exactly those rows, so pooling the extra
        # validation rows here would hand the frozen-linear baseline ~25% more
        # training data than the fine-tuned modes it is being compared against,
        # and would also leak validation rows into the warm-start weights that
        # modes 4/6 inherit.
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        pool = np.asarray(split_indices["train"])
        pool_y = y[pool].copy()
        if shuffled:
            pool_y = pool_y[np.random.default_rng(seed).permutation(len(pool_y))]
        mean_t, std_t = token_normalization(cache.features[pool])
        mean, sd = mean_t.numpy(), std_t.numpy()
        flat_pool = _flatten_and_zscore(cache.features[pool], mean, sd)
        flat_test = _flatten_and_zscore(cache.features[split_indices["test"]], mean, sd)

        cv = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
        fold_scores = {float(C): [] for C in c_grid}
        for fit_index, score_index in cv.split(flat_pool, pool_y):
            for C in c_grid:
                w_fold, b_fold = _lr_fit_exact(flat_pool[fit_index], pool_y[fit_index], C)
                scores = _decision_scores(flat_pool[score_index], None, w_fold, b_fold)
                fold_scores[float(C)].append(roc_auc_score(pool_y[score_index], scores))
        mean_auc = {C: float(np.mean(v)) for C, v in fold_scores.items()}
        chosen_C = max(mean_auc, key=lambda C: (mean_auc[C], -C))
        w, b = _lr_fit_exact(flat_pool, pool_y, chosen_C)
        best = {"balanced_accuracy": None, "loss": None, "C": float(chosen_C), "w": w, "b": b,
                "inner_cv_mean_auc": mean_auc, "inner_folds": inner_folds,
                "selection": "inner k-fold CV on the training fold (smri protocol)"}
        flat_train = flat_pool
    else:
        mean_t, std_t = token_normalization(cache.features[split_indices["train"]])
        mean, sd = mean_t.numpy(), std_t.numpy()

        flat_train = _flatten_and_zscore(cache.features[split_indices["train"]], mean, sd)
        flat_val = _flatten_and_zscore(cache.features[split_indices["validation"]], mean, sd)
        flat_test = _flatten_and_zscore(cache.features[split_indices["test"]], mean, sd)

        best = None
        for C in c_grid:
            w, b = _lr_fit_exact(flat_train, train_y, C)
            val_scores = _decision_scores(flat_val, flat_train, w, b)
            val_pred = (val_scores >= 0.0).astype(np.int64)
            sensitivity = np.sum((val_y == 1) & (val_pred == 1)) / max(1, np.sum(val_y == 1))
            specificity = np.sum((val_y == 0) & (val_pred == 0)) / max(1, np.sum(val_y == 0))
            balanced = float((sensitivity + specificity) / 2)
            from sklearn.metrics import log_loss
            probs_val = 1.0 / (1.0 + np.exp(-val_scores))
            val_loss = float(log_loss(val_y, np.clip(probs_val, 1e-7, 1 - 1e-7)))
            candidate = {"balanced_accuracy": balanced, "loss": val_loss, "C": float(C), "w": w, "b": b}
            if best is None or (candidate["balanced_accuracy"], -candidate["loss"]) > (
                    best["balanced_accuracy"], -best["loss"]):
                best = candidate
    assert best is not None

    test_scores = _decision_scores(flat_test, flat_train, best["w"], best["b"])
    probability = 1.0 / (1.0 + np.exp(-test_scores))

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = f"seed_{seed}" + ("_shuffled" if shuffled else "") + tag_suffix

    import torch
    head = LinearPoolHead(n_tokens, width, torch.from_numpy(mean), torch.from_numpy(sd))
    head.set_weights(best["w"], best["b"])
    checkpoint = output / f"linear_{tag}.pt"
    torch.save({"w": best["w"], "b": best["b"], "mean": mean, "std": sd,
                "C": best["C"], "cache_metadata": cache.metadata, "seed": seed,
                "shuffled": shuffled}, checkpoint)

    predictions_path = ensure_parent(output / f"linear_{tag}_predictions.csv")
    test_index = split_indices["test"]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("file_id", "subject_id", "true", "pred", "positive_probability")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, prob in zip(test_index, probability):
            writer.writerow({"file_id": cache.file_ids[index], "subject_id": cache.subject_ids[index],
                             "true": cache.labels[index], "pred": positive if prob >= 0.5 else "CN",
                             "positive_probability": f"{float(prob):.10g}"})

    result = {"kind": "linear_probe", "disease": cache.metadata.get("disease"),
              "encoder": cache.metadata.get("encoder"), "seed": seed, "shuffled_labels": shuffled,
              "positive_label": positive,
              "selection": {"C": best["C"], "balanced_accuracy": best["balanced_accuracy"],
                            "loss": best["loss"],
                            "inner_folds": best.get("inner_folds", 0),
                            "inner_cv_mean_auc": best.get("inner_cv_mean_auc"),
                            "method": best.get("selection", "held-out validation split")},
              "trainable_parameters": trainable_parameter_count_linear(n_tokens, width),
              "token_shape": list(cache.features.shape[1:]), "checkpoint": str(checkpoint),
              "predictions": str(predictions_path),
              "metrics": _summary(cache, test_index, probability, positive,
                                  int(evaluation["bootstrap_samples"]), int(evaluation["bootstrap_seed"]))}
    write_json(output / f"linear_{tag}_summary.json", result)
    return result
