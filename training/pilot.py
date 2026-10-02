"""
Pilot: ResNet-50 seed เดียว เช็คว่า pipeline + group split สมเหตุสมผลก่อนรัน run_final.py

    python pilot.py                 # group split (split.csv) — ตัวจริง
    python pilot.py --split random  # stratified random split แบบเดิม — ไว้เทียบหา leakage

อ่านผล:
    test acc ~0.7-0.9         = ปกติ
    test acc ~0.99            = น่าสงสัยว่า leak
    random >> group (เกิน ~5-10 จุด) = มีรูปคล้ายกันข้าม split ตอน random -> group split จำเป็นจริง
"""
import argparse
import json

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from config import CLASSES, BEST_HPARAMS, CHECKPOINT_DIR, RESULTS_DIR, METADATA, SPLIT_SEED
from dataset import make_splits, make_loaders, get_class_weights
from evaluate import predict
from models import build_model
from trainer import train_two_stage
from utils import set_seed, get_device

ARCH, SEED = "resnet50", 11


def random_splits():
    df = pd.read_csv(METADATA)
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    tv, test = train_test_split(df, test_size=0.20, stratify=df["class"], random_state=SPLIT_SEED)
    train, val = train_test_split(tv, test_size=0.125, stratify=tv["class"], random_state=SPLIT_SEED)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["group", "random"], default="group")
    args = ap.parse_args()

    wandb.init(mode="disabled")
    device = get_device()
    train_df, val_df, test_df = make_splits() if args.split == "group" else random_splits()
    print(f"device={device} split={args.split} | train {len(train_df)} val {len(val_df)} test {len(test_df)}")

    set_seed(SEED)
    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df)
    model = build_model(ARCH)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"pilot_{ARCH}_{args.split}.pt"
    train_two_stage(model, ARCH, train_loader, val_loader, get_class_weights(train_df),
                    BEST_HPARAMS[ARCH], device, ckpt, SEED)

    import torch
    model.load_state_dict(torch.load(ckpt, map_location=device))
    y, p, _ = predict(model, test_loader, device)
    y, p = np.array(y), np.array(p)
    test_df = test_df.reset_index(drop=True).assign(correct=(y == p))

    res = {
        "split": args.split,
        "test_acc": accuracy_score(y, p),
        "f1_macro": f1_score(y, p, average="macro"),
        "f1_weighted": f1_score(y, p, average="weighted"),
        "recall_per_class": dict(zip(CLASSES, (confusion_matrix(y, p).diagonal()
                                               / np.bincount(y, minlength=len(CLASSES))).round(3))),
        "acc_per_source": test_df.groupby("engine")["correct"].mean().round(3).to_dict(),
    }
    print("\n" + "=" * 55)
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print("\nconfusion matrix (row = true, col = pred):", CLASSES)
    print(confusion_matrix(y, p))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"pilot_{args.split}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    test_df.to_csv(RESULTS_DIR / f"pilot_{args.split}_test_preds.csv", index=False)


if __name__ == "__main__":
    main()
