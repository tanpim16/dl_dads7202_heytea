"""
Hyperparameter tuning ด้วย W&B Sweep (Bayesian optimization) — แยก sweep ต่อ architecture
ต้องมี WANDB_API_KEY (Kaggle: Add-ons -> Secrets)

    python run_sweep.py                     # ทุก arch, 10 runs/arch
    python run_sweep.py --archs resnet50 --count 15

- search space เดียวกับ Optuna (tuning.SEARCH_SPACE), seed 42 ทุก run
- metric ที่ sweep optimize = best_val_f1 (best validation weighted F1 ของ run นั้น)
- ผลทุก run ถูกบันทึกลง results/tuning/trials.csv ด้วย (tool = wandb_sweep) เพื่อเทียบกับ Optuna
"""
import argparse
import os
import time

import wandb

from config import ARCHS
from tuning import SEARCH_SPACE, train_trial, log_trial, select_best

WANDB_PROJECT = "heytea-cnn"


def sweep_config(arch):
    return {
        "name": f"sweep_{arch}",
        "method": "bayes",
        "metric": {"name": "best_val_f1", "goal": "maximize"},
        "parameters": {"arch": {"value": arch}, **{k: {"values": v} for k, v in SEARCH_SPACE.items()}},
    }


def sweep_train():
    run = wandb.init()
    cfg = dict(run.config)
    arch = cfg["arch"]
    hp = {k: cfg[k] for k in SEARCH_SPACE}
    t0 = time.time()
    try:
        value = train_trial(arch, hp, f"sweep_{run.id}")
        wandb.log({"best_val_f1": value})
        log_trial("wandb_sweep", arch, run.id, hp, value, "complete", time.time() - t0)
    except Exception as e:
        log_trial("wandb_sweep", arch, run.id, hp, None, f"failed: {type(e).__name__}", time.time() - t0)
        raise
    finally:
        run.finish()


def main(archs=None, count=10):
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("W&B Sweep ต้องมี WANDB_API_KEY (Kaggle: Add-ons -> Secrets)")
    for arch in archs or ARCHS:
        sweep_id = wandb.sweep(sweep_config(arch), project=WANDB_PROJECT)
        print(f"\nW&B Sweep — {arch}: {count} runs (sweep id {sweep_id})")
        wandb.agent(sweep_id, function=sweep_train, count=count)
    select_best()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", choices=ARCHS)
    ap.add_argument("--count", type=int, default=10)
    a = ap.parse_args()
    main(a.archs, a.count)
