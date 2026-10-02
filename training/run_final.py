"""
Final evaluation: 5 seeds × 4 architectures.

Workflow:
  1. Fixed stratified split (SPLIT_SEED=42) — same for every run.
  2. For each arch × seed: train two-stage, evaluate on TEST set.
  3. Compute mean±SD across seeds per arch.
  4. Welch's t-test between every arch pair.
  5. Save CSVs + plots to results/.

Run:
    python run_final.py
"""
import torch
import numpy as np
import pandas as pd
import wandb
from pathlib import Path

from config import ARCHS, SEEDS, CHECKPOINT_DIR, RESULTS_DIR, BEST_HPARAMS, CLASSES
from dataset import make_splits, make_loaders, get_class_weights
from models import build_model
from trainer import train_two_stage
from evaluate import (
    predict, compute_metrics, summarize_runs,
    welch_ttest_all_pairs,
    plot_confusion_matrix, plot_mean_sd, plot_per_class,
    print_classification_report,
)
from gradcam import visualize_gradcam
from utils import set_seed, get_device

WANDB_PROJECT = "heytea-cnn"


def run_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}\n")

    # ── Fixed split ───────────────────────────────────────────────────────────
    train_df, val_df, test_df = make_splits()
    _, _, test_loader = make_loaders(train_df, val_df, test_df)
    class_weights     = get_class_weights(train_df)

    print("Split summary (test set is fixed, never touched during training):")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = " | ".join(f"{c}={len(df[df['class']==c])}" for c in CLASSES)
        print(f"  {name:5s}: {len(df):4d}  [{counts}]")

    # ── Main loop ─────────────────────────────────────────────────────────────
    all_summaries: list[dict] = []
    arch_f1s: dict[str, list[float]] = {a: [] for a in ARCHS}

    for arch in ARCHS:
        hparams      = BEST_HPARAMS[arch]
        arch_metrics = []

        for seed in SEEDS:
            run_name  = f"{arch}_seed{seed}"
            ckpt_path = CHECKPOINT_DIR / f"{run_name}.pt"

            wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                config={"arch": arch, "seed": seed, **hparams},
                reinit=True,
            )

            set_seed(seed)
            train_loader, val_loader, _ = make_loaders(train_df, val_df, test_df)
            model = build_model(arch)

            train_two_stage(
                model, arch, train_loader, val_loader,
                class_weights, hparams, device, ckpt_path, seed,
            )

            # Evaluate best checkpoint on TEST set
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)
            labels, preds, _ = predict(model, test_loader, device)
            metrics = compute_metrics(labels, preds)
            arch_metrics.append(metrics)
            arch_f1s[arch].append(metrics["f1_weighted"])

            wandb.log({
                "test/f1_weighted":        metrics["f1_weighted"],
                "test/accuracy":           metrics["accuracy"],
                "test/precision_weighted": metrics["precision_weighted"],
                "test/recall_weighted":    metrics["recall_weighted"],
            })
            wandb.finish()

            print(f"  {run_name:35s}  test F1={metrics['f1_weighted']:.4f}  "
                  f"acc={metrics['accuracy']:.4f}")

        # Per-architecture summary
        summary = summarize_runs(arch_metrics)
        summary["arch"] = arch
        all_summaries.append(summary)

        # Confusion matrix & GradCAM — use best-seed checkpoint
        best_seed = SEEDS[int(np.argmax(arch_f1s[arch]))]
        best_ckpt = CHECKPOINT_DIR / f"{arch}_seed{best_seed}.pt"
        model = build_model(arch)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        model.to(device)

        labels, preds, _ = predict(model, test_loader, device)
        plot_confusion_matrix(
            labels, preds, arch,
            save_path=RESULTS_DIR / f"cm_{arch}.png",
        )
        print_classification_report(labels, preds, arch)

        visualize_gradcam(
            model, arch, test_loader, device,
            n_correct=3, n_wrong=3,
            save_path=RESULTS_DIR / f"gradcam_{arch}.png",
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("\n" + "="*65)
    print("Model Comparison — test set, mean±SD (5 seeds)")
    print("="*65)
    metrics_to_show = ["f1_weighted", "accuracy", "precision_weighted", "recall_weighted"]
    for _, row in summary_df.iterrows():
        vals = "  ".join(
            f"{m.split('_')[0]}={row[f'{m}_mean']:.4f}±{row[f'{m}_std']:.4f}"
            for m in metrics_to_show
        )
        print(f"  {row['arch']:25s}  {vals}")

    # ── Statistical tests ─────────────────────────────────────────────────────
    ttest_df = welch_ttest_all_pairs(arch_f1s)
    ttest_df.to_csv(RESULTS_DIR / "ttest.csv", index=False)
    print("\nWelch's t-test (pairwise, weighted F1, test set):")
    print(ttest_df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_mean_sd(summary_df, "f1_weighted",
                 save_path=RESULTS_DIR / "comparison_f1.png")
    plot_mean_sd(summary_df, "accuracy",
                 save_path=RESULTS_DIR / "comparison_acc.png")
    plot_per_class(summary_df,
                   save_path=RESULTS_DIR / "per_class_f1.png")

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_all()
