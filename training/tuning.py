"""
ส่วนกลางของ hyperparameter tuning — ใช้ร่วมกันระหว่าง Optuna (run_tune_optuna.py) และ W&B Sweep (run_sweep.py)
search space เดียวกันทั้งสองเครื่องมือ เพื่อเทียบกันได้ตรง ๆ

เกณฑ์เลือก: best validation weighted F1 (ตัวเดียวกับที่ใช้ early stopping / เลือก checkpoint)
ห้ามใช้ test set ตอน tuning — test ใช้ครั้งเดียวตอน run_final
"""
import csv
import json
import os
import time
from pathlib import Path

import config
from config import CHECKPOINT_DIR
from dataset import make_splits, make_loaders, get_class_weights
from models import build_model
from trainer import train_two_stage
from utils import set_seed, get_device

TUNE_SEED = 42          # seed เดียวทุก trial (seed ที่ใช้รายงานผลจริงคือ 11..55 ใน run_final)

SEARCH_SPACE = {        # ตรงกับ SWEEP_CONFIG เดิมของ run_sweep.py
    "stage1_lr":       [0.0005, 0.001],
    "stage1_epochs":   [5, 10],
    "stage2_lr":       [0.00001, 0.00005, 0.0001],
    "stage2_epochs":   [10, 15, 20],
    "label_smoothing": [0.0, 0.05, 0.1],
}


def tuning_dir() -> Path:
    d = config.RESULTS_DIR / "tuning"
    d.mkdir(parents=True, exist_ok=True)
    return d


_DATA = {}


def _data():
    if not _DATA:
        tr, va, te = make_splits()
        _DATA.update(train=tr, val=va, test=te, weights=get_class_weights(tr))
    return _DATA


def train_trial(arch: str, hp: dict, tag: str, epoch_callback=None) -> float:
    """เทรน 1 trial บน train, คืนค่า best val weighted F1 (ไม่แตะ test)"""
    d = _data()
    set_seed(TUNE_SEED)
    train_loader, val_loader, _ = make_loaders(d["train"], d["val"], d["test"])
    model = build_model(arch)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"tune_{tag}.pt"
    try:
        return train_two_stage(model, arch, train_loader, val_loader, d["weights"], hp,
                               get_device(), ckpt, TUNE_SEED, epoch_callback=epoch_callback)
    finally:
        ckpt.unlink(missing_ok=True)


def log_trial(tool: str, arch: str, trial_id, hp: dict, value, state: str, seconds: float):
    f = tuning_dir() / "trials.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["tool", "arch", "trial", *SEARCH_SPACE, "best_val_f1", "state", "seconds"])
        w.writerow([tool, arch, trial_id, *[hp[k] for k in SEARCH_SPACE],
                    "" if value is None else round(value, 4), state, round(seconds)])


def wandb_mode():
    return "online" if os.environ.get("WANDB_API_KEY") else "disabled"


def select_best(out_json: Path = None) -> dict:
    """รวมผลทั้ง Optuna + Sweep -> hparams ที่ดีที่สุดต่อ arch (ตาม val F1) -> best_hparams.json"""
    import pandas as pd
    t = pd.read_csv(tuning_dir() / "trials.csv")
    t = t[t["state"] == "complete"].dropna(subset=["best_val_f1"])
    best, rows = {}, []
    for arch, g in t.groupby("arch"):
        r = g.sort_values(["best_val_f1", "seconds"], ascending=[False, True]).iloc[0]
        best[arch] = {k: (int(r[k]) if k.endswith("epochs") else float(r[k])) for k in SEARCH_SPACE}
        for tool, gt in g.groupby("tool"):
            rows.append({"arch": arch, "tool": tool, "n_complete": len(gt),
                         "best_val_f1": gt["best_val_f1"].max(), "chosen": tool == r["tool"]})
    out_json = out_json or tuning_dir() / "best_hparams.json"
    out_json.write_text(json.dumps(best, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(best, indent=2))
    return best
