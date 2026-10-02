"""
Class-imbalance experiment: เทียบ 4 วิธี บน architecture เดียว × 5 seeds (split เดียวกับ run_final)

    none          CrossEntropy ธรรมดา (baseline ไม่ชดเชย)
    class_weight  CrossEntropy + weight ตามความถี่คลาส (ค่าที่ใช้ใน run_final)
    sampler       WeightedRandomSampler (oversample คลาสเล็ก) + CrossEntropy ธรรมดา
    focal         Focal loss gamma=2 (ไม่มี class weight เพื่อแยกผล)

วัดผล: macro F1 (หลัก), weighted F1, accuracy + precision/recall ของ cha_dam_yen (คลาสเล็กสุด)
ผลแต่ละ run เขียนลง results/imbalance_runs.csv ทันที (รันต่อได้ถ้าหลุด)

Run:
    python run_imbalance.py                       # ResNet-50, ทั้ง 4 วิธี
    python run_imbalance.py --arch mobilenet_v3_large --methods none focal
    python run_imbalance.py --summary-only
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import wandb

import config
from config import ARCHS, SEEDS, CHECKPOINT_DIR, RESULTS_DIR, BEST_HPARAMS, CLASSES
from dataset import make_splits, make_loaders, get_class_weights
from evaluate import predict, compute_metrics, welch_ttest_all_pairs, plot_strip
from losses import FocalLoss
from models import build_model
from trainer import train_two_stage
from utils import set_seed, get_device

METHODS = ["none", "class_weight", "sampler", "focal"]
FOCAL_GAMMA = 2.0
SMALL = "cha_dam_yen"


def out_csv():
    return RESULTS_DIR / "imbalance_runs.csv"


def load_runs():
    p = out_csv()
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["arch", "method", "seed"])


def run(arch, methods):
    device = get_device()
    train_df, val_df, test_df = make_splits()
    print(f"Device: {device} | arch={arch} | methods={methods}")
    small = CLASSES.index(SMALL)

    for method in methods:
        # dataset.make_loaders / get_class_weights อ่าน config.IMBALANCE ตอนเรียก
        config.IMBALANCE = {"none": "none", "class_weight": "class_weight",
                            "sampler": "sampler", "focal": "none"}[method]
        done = set(load_runs().query("arch == @arch and method == @method")["seed"].astype(int))
        for seed in SEEDS:
            name = f"imb_{method}_{arch}_seed{seed}"
            if seed in done:
                print(f"  skip {name}")
                continue
            wandb.init(project="heytea-cnn", name=name, reinit=True,
                       config={"arch": arch, "method": method, "seed": seed},
                       mode="online" if os.environ.get("WANDB_API_KEY") else "disabled")
            set_seed(seed)
            train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df)
            model = build_model(arch)
            criterion = FocalLoss(FOCAL_GAMMA) if method == "focal" else None
            ckpt = CHECKPOINT_DIR / f"{name}.pt"
            train_two_stage(model, arch, train_loader, val_loader, get_class_weights(train_df),
                            BEST_HPARAMS[arch], device, ckpt, seed, criterion=criterion)

            model.load_state_dict(torch.load(ckpt, map_location=device))
            y, p, _ = predict(model, test_loader, device)
            m = compute_metrics(y, p)
            wandb.finish()
            row = {"arch": arch, "method": method, "seed": seed,
                   **{k: v for k, v in m.items() if not isinstance(v, list)},
                   f"recall_{SMALL}": m["recall_per_class"][small],
                   f"precision_{SMALL}": m["precision_per_class"][small],
                   "f1_per_class": json.dumps(m["f1_per_class"])}
            pd.concat([load_runs(), pd.DataFrame([row])], ignore_index=True).to_csv(out_csv(), index=False)
            print(f"  {name:45s} F1macro={m['f1_macro']:.4f} acc={m['accuracy']:.4f} "
                  f"{SMALL} R={row[f'recall_{SMALL}']:.3f} P={row[f'precision_{SMALL}']:.3f}")
            ckpt.unlink(missing_ok=True)   # ไม่เก็บ checkpoint (ประหยัดพื้นที่ Output ของ Kaggle)


def summarize():
    runs = load_runs()
    if runs.empty:
        print("ยังไม่มีผลใน imbalance_runs.csv")
        return
    cols = ["f1_macro", "f1_weighted", "accuracy", f"recall_{SMALL}", f"precision_{SMALL}"]
    for arch, r in runs.groupby("arch"):
        g = r.groupby("method")[cols].agg(["mean", lambda x: x.std(ddof=1), "count"])
        g.columns = [f"{c}_{'std' if s == '<lambda_0>' else s}" for c, s in g.columns]
        g = g.reindex([m for m in METHODS if m in g.index])
        g.insert(0, "arch", arch)
        g.to_csv(RESULTS_DIR / f"imbalance_summary_{arch}.csv")
        print(f"\n=== Imbalance methods — {arch} (test set, mean±SD over seeds) ===")
        for method, row in g.iterrows():
            print(f"  {method:13s} n={int(row['f1_macro_count'])}  " + "  ".join(
                f"{c}={row[f'{c}_mean']:.4f}±{row[f'{c}_std']:.4f}" for c in cols))

        tests = []
        for metric in ["f1_macro", f"recall_{SMALL}"]:
            scores = {m: r.loc[r["method"] == m, metric].tolist() for m in METHODS if m in set(r["method"])}
            tests.append(welch_ttest_all_pairs(scores, metric))
        t = pd.concat(tests, ignore_index=True).rename(columns={"Model (1)": "Method (1)", "Model (2)": "Method (2)"})
        t.to_csv(RESULTS_DIR / f"imbalance_ttest_{arch}.csv", index=False)
        print("\nWelch's t-test + Hedges' g:")
        print(t.to_string(index=False))

        plot_df = r.rename(columns={"method": "_m", "arch": "_a"}).assign(arch=lambda d: d["_m"])
        plot_df["arch"] = pd.Categorical(plot_df["arch"], [m for m in METHODS if m in set(plot_df["arch"])])
        plot_df = plot_df.sort_values("arch")
        plot_df["arch"] = plot_df["arch"].astype(str)
        for metric in ["f1_macro", f"recall_{SMALL}"]:
            plot_strip(plot_df, metric, save_path=RESULTS_DIR / f"imbalance_{metric}_{arch}.png")
    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50", choices=ARCHS)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        run(a.arch, a.methods)
    summarize()
