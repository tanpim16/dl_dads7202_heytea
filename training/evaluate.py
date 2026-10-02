import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from scipy.stats import ttest_ind

from config import CLASSES, RESULTS_DIR


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device):
    """Returns (true_labels, predicted_labels, softmax_probs)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
    return all_labels, all_preds, all_probs


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(labels, preds) -> dict:
    return {
        "accuracy":           accuracy_score(labels, preds),
        # macro = ทุกคลาสน้ำหนักเท่ากัน (สำคัญเพราะ cha_dam_yen มีรูปน้อย) / weighted = ตามจำนวนรูป
        "f1_macro":           f1_score(labels, preds, average="macro",  zero_division=0),
        "precision_macro":    precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":       recall_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted":        f1_score(labels, preds, average="weighted",  zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(labels, preds, average="weighted",    zero_division=0),
        # Per-class (list, index = class index matching CLASSES)
        "f1_per_class":        f1_score(labels, preds, average=None, zero_division=0).tolist(),
        "precision_per_class": precision_score(labels, preds, average=None, zero_division=0).tolist(),
        "recall_per_class":    recall_score(labels, preds, average=None, zero_division=0).tolist(),
    }


def summarize_runs(all_metrics: list) -> dict:
    """Compute mean±SD across a list of per-seed metric dicts."""
    keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro",
            "f1_weighted", "precision_weighted", "recall_weighted"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[f"{k}_mean"] = np.mean(vals)
        summary[f"{k}_std"]  = np.std(vals, ddof=1)

    # Per-class mean±SD
    for metric in ["f1", "precision", "recall"]:
        arr = np.array([m[f"{metric}_per_class"] for m in all_metrics])  # (n_seeds, n_classes)
        summary[f"{metric}_per_class_mean"] = arr.mean(axis=0).tolist()
        summary[f"{metric}_per_class_std"]  = arr.std(axis=0, ddof=1).tolist()

    return summary


# ── Statistical testing ───────────────────────────────────────────────────────

def hedges_g(a, b) -> float:
    """Effect size: Cohen's d (pooled SD) + small-sample correction (n=5 seeds ต่อกลุ่ม)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    return d * (1 - 3 / (4 * (na + nb) - 9))


def welch_ttest_all_pairs(arch_scores: dict, metric: str = "f1_macro") -> pd.DataFrame:
    """
    arch_scores: {"vgg16": [score_seed1, score_seed2, ...], ...}
    Pairwise Welch's t-test + Hedges' g (|g| ~0.2 small, ~0.5 medium, >=0.8 large).
    """
    rows = []
    for a1, a2 in combinations(arch_scores.keys(), 2):
        _, p = ttest_ind(arch_scores[a1], arch_scores[a2], equal_var=False)
        g = hedges_g(arch_scores[a1], arch_scores[a2])
        rows.append({
            "metric":        metric,
            "Model (1)":     a1,
            "Model (2)":     a2,
            "Mean (1)":      round(np.mean(arch_scores[a1]), 4),
            "Mean (2)":      round(np.mean(arch_scores[a2]), 4),
            "p-value":       round(p, 4),
            "Hedges g":      round(g, 3),
            "Significant (p<0.05)": p < 0.05,
        })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, arch, save_path=None):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    short = [c.replace("cha_", "") for c in CLASSES]
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {arch}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_mean_sd(summary_df: pd.DataFrame, metric="f1_weighted", save_path=None):
    """
    Bar chart + error bars (mean±SD).
    Teacher note: use strip/dot plot for cleaner visualization — see comment below.
    """
    archs = summary_df["arch"].tolist()
    means = summary_df[f"{metric}_mean"].tolist()
    stds  = summary_df[f"{metric}_std"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(archs, means, yerr=stds, capsize=6,
           color="steelblue", alpha=0.7, error_kw={"elinewidth": 1.5})
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison — mean±SD ({metric}, test set)")
    lo = max(0.0, min(means) - 0.15)
    hi = min(1.0, max(means) + 0.10)
    ax.set_ylim(lo, hi)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_strip(runs_df: pd.DataFrame, metric="f1_macro", save_path=None):
    """Dot/strip plot: 1 จุด = 1 seed + เส้น mean และ ±SD (เห็นการกระจายจริง ดีกว่า bar + error bar)."""
    archs = list(dict.fromkeys(runs_df["arch"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.stripplot(data=runs_df, x="arch", y=metric, order=archs, size=8,
                  jitter=0.08, color="steelblue", alpha=0.8, ax=ax)
    for i, a in enumerate(archs):
        v = runs_df.loc[runs_df["arch"] == a, metric]
        m, sd = v.mean(), v.std(ddof=1)
        ax.hlines(m, i - 0.25, i + 0.25, color="black", lw=2)
        ax.vlines(i, m - sd, m + sd, color="black", lw=1)
        ax.text(i + 0.28, m, f"{m:.3f}±{sd:.3f}", va="center", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_title(f"Model Comparison — {metric}, test set (dot = seed, bar = mean±SD)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_per_class(summary_df: pd.DataFrame, save_path=None):
    """Per-class F1 heatmap (archs × classes)."""
    data = []
    for _, row in summary_df.iterrows():
        for i, cls in enumerate(CLASSES):
            data.append({
                "arch":  row["arch"],
                "class": cls.replace("cha_", ""),
                "F1":    row["f1_per_class_mean"][i],
            })
    df = pd.DataFrame(data).pivot(index="arch", columns="class", values="F1")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title("Per-class F1 (mean across seeds) — test set")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def print_classification_report(labels, preds, arch):
    print(f"\n── Classification Report: {arch} ──")
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0))
