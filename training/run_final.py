"""
Final evaluation: 5 seeds × 4 architectures.

Workflow:
  1. Fixed group split from tea_dataset/split.csv — same for every run.
  2. For each arch × seed: train two-stage, evaluate on TEST set.
     ผลแต่ละ run เขียนลง results/runs.csv ทันที -> session หลุดกลางทางก็รันต่อได้ (run ที่เสร็จแล้วจะถูกข้าม)
  3. Compute mean±SD across seeds per arch (macro + weighted).
  4. Welch's t-test + Hedges' g between every arch pair (macro F1 = primary).
  5. Save CSVs + plots (dot/strip plot) to results/.

Run:
    python run_final.py                          # ทุก arch ใน config.ARCHS
    python run_final.py --archs resnet50 vgg16   # บาง arch (แบ่งรันหลาย session ได้)
    python run_final.py --summary-only           # สรุปผลจาก runs.csv อย่างเดียว ไม่เทรน
"""
import argparse
import json
import os

import torch
import numpy as np
import pandas as pd
import wandb

from config import ARCHS, SEEDS, CHECKPOINT_DIR, RESULTS_DIR, BEST_HPARAMS, CLASSES
from dataset import make_splits, make_loaders, get_class_weights
from models import build_model
from trainer import train_two_stage
from evaluate import (
    predict, compute_metrics, summarize_runs,
    welch_ttest_all_pairs,
    plot_confusion_matrix, plot_strip, plot_per_class,
    print_classification_report,
)
from gradcam import visualize_gradcam
from utils import set_seed, get_device

WANDB_PROJECT = "heytea-cnn"
PRIMARY = "f1_macro"   # metric หลัก: macro เพราะคลาสไม่สมดุล (cha_dam_yen น้อย)


def runs_path():
    return RESULTS_DIR / "runs.csv"


def load_runs() -> pd.DataFrame:
    p = runs_path()
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["arch", "seed"])


def append_run(row: dict):
    df = pd.concat([load_runs(), pd.DataFrame([row])], ignore_index=True)
    df.to_csv(runs_path(), index=False)


def train_arch(arch, train_df, val_df, test_df, test_loader, class_weights, device):
    hparams = BEST_HPARAMS[arch]
    done = set(load_runs().query("arch == @arch")["seed"].astype(int))

    for seed in SEEDS:
        run_name = f"{arch}_seed{seed}"
        if seed in done:
            print(f"  skip {run_name} (มีผลใน runs.csv แล้ว)")
            continue
        ckpt_path = CHECKPOINT_DIR / f"{run_name}.pt"

        # W&B ไม่บังคับ: ไม่มี WANDB_API_KEY -> ปิด logging (ไม่ค้างรอ login)
        wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True,
                   config={"arch": arch, "seed": seed, **hparams},
                   mode="online" if os.environ.get("WANDB_API_KEY") else "disabled")

        set_seed(seed)
        train_loader, val_loader, _ = make_loaders(train_df, val_df, test_df)
        model = build_model(arch)
        train_two_stage(model, arch, train_loader, val_loader,
                        class_weights, hparams, device, ckpt_path, seed)

        # Evaluate best checkpoint on TEST set
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        labels, preds, probs = predict(model, test_loader, device)
        m = compute_metrics(labels, preds)
        # ผลทำนายรายรูป -> ใช้ทำ error analysis ภายหลัง (ไม่ต้องรันโมเดลใหม่)
        (RESULTS_DIR / "preds").mkdir(exist_ok=True)
        test_df.assign(pred=[CLASSES[k] for k in preds],
                       confidence=np.max(probs, axis=1).round(4)).to_csv(
            RESULTS_DIR / "preds" / f"{run_name}.csv", index=False)
        wandb.log({f"test/{k}": v for k, v in m.items() if not isinstance(v, list)})
        wandb.finish()

        append_run({"arch": arch, "seed": seed,
                    **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in m.items()}})
        print(f"  {run_name:35s}  test F1macro={m['f1_macro']:.4f}  "
              f"F1w={m['f1_weighted']:.4f}  acc={m['accuracy']:.4f}")

    # Confusion matrix & GradCAM — best seed (ตาม macro F1)
    runs = load_runs().query("arch == @arch")
    best_seed = int(runs.loc[runs[PRIMARY].idxmax(), "seed"])
    best_ckpt = CHECKPOINT_DIR / f"{arch}_seed{best_seed}.pt"
    if not best_ckpt.exists():
        print(f"  ! ไม่มี checkpoint {best_ckpt.name} (รันใน session อื่น) -> ข้าม CM/GradCAM")
        return
    # ภาพประกอบพังได้ แต่ห้ามทำให้ทั้ง run หยุด (Kaggle v1 ล้มเพราะ GradCAM ของ VGG)
    try:
        model = build_model(arch)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        model.to(device)
        labels, preds, _ = predict(model, test_loader, device)
        plot_confusion_matrix(labels, preds, arch, save_path=RESULTS_DIR / f"cm_{arch}.png")
        print_classification_report(labels, preds, arch)
        visualize_gradcam(model, arch, test_loader, device, n_correct=3, n_wrong=3,
                          save_path=RESULTS_DIR / f"gradcam_{arch}.png")
    except Exception as e:
        print(f"  ! CM/GradCAM ของ {arch} ล้ม ({type(e).__name__}: {e}) -> ข้าม, เทรนต่อ")


def summarize():
    runs = load_runs()
    if runs.empty:
        print("ยังไม่มีผลใน runs.csv")
        return
    for c in ["f1_per_class", "precision_per_class", "recall_per_class"]:
        runs[c] = runs[c].apply(json.loads)
    archs = [a for a in ARCHS if a in set(runs["arch"])] + sorted(set(runs["arch"]) - set(ARCHS))

    all_summaries = []
    for arch in archs:
        r = runs[runs["arch"] == arch]
        if len(r) < len(SEEDS):
            print(f"  ! {arch}: มีแค่ {len(r)}/{len(SEEDS)} seeds")
        s = summarize_runs(r.to_dict("records"))
        s["arch"], s["n_seeds"] = arch, len(r)
        all_summaries.append(s)
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("\n" + "=" * 75)
    print("Model Comparison — test set, mean±SD across seeds")
    print("=" * 75)
    show = ["f1_macro", "f1_weighted", "accuracy", "precision_macro", "recall_macro"]
    for _, row in summary_df.iterrows():
        vals = "  ".join(f"{m}={row[f'{m}_mean']:.4f}±{row[f'{m}_std']:.4f}" for m in show)
        print(f"  {row['arch']:20s}  {vals}")

    # ── Statistical tests (macro = primary, weighted = secondary) ────────────
    tests = []
    for metric in ["f1_macro", "f1_weighted"]:
        scores = {a: runs.loc[runs["arch"] == a, metric].tolist() for a in archs}
        tests.append(welch_ttest_all_pairs(scores, metric))
    ttest_df = pd.concat(tests, ignore_index=True)
    ttest_df.to_csv(RESULTS_DIR / "ttest.csv", index=False)
    print("\nWelch's t-test + Hedges' g (pairwise, test set):")
    print(ttest_df.to_string(index=False))

    # ── Plots: dot/strip (1 จุด = 1 seed) แทน bar + error bar ────────────────
    for metric in ["f1_macro", "f1_weighted", "accuracy"]:
        plot_strip(runs, metric, save_path=RESULTS_DIR / f"comparison_{metric}.png")
    plot_per_class(summary_df, save_path=RESULTS_DIR / "per_class_f1.png")
    print(f"\nAll results saved to {RESULTS_DIR}/")


def run_all(archs=None, summary_only=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not summary_only:
        device = get_device()
        print(f"Device: {device}\n")

        # ── Fixed split ───────────────────────────────────────────────────────
        train_df, val_df, test_df = make_splits()
        _, _, test_loader = make_loaders(train_df, val_df, test_df)
        class_weights = get_class_weights(train_df)

        print("Split summary (test set is fixed, never touched during training):")
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            counts = " | ".join(f"{c}={len(df[df['class']==c])}" for c in CLASSES)
            print(f"  {name:5s}: {len(df):4d}  [{counts}]")

        for arch in archs or ARCHS:
            train_arch(arch, train_df, val_df, test_df, test_loader, class_weights, device)

    summarize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", choices=ARCHS)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    run_all(a.archs, a.summary_only)
