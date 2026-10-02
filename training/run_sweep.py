"""
W&B Bayesian hyperparameter sweep.

Run:
    python run_sweep.py

After the sweep completes, inspect W&B and fill in BEST_HPARAMS in config.py,
then run run_final.py.
"""
import wandb
from config import ARCHS, CHECKPOINT_DIR
from dataset import make_splits, make_loaders, get_class_weights
from models import build_model
from trainer import train_two_stage
from utils import set_seed, get_device

WANDB_PROJECT = "heytea-cnn"
SWEEP_COUNT   = 60   # total sweep runs (15 per arch on average)

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "stage2/val_f1", "goal": "maximize"},
    "parameters": {
        "arch":            {"values": ARCHS},
        "stage1_lr":       {"values": [0.0005, 0.001]},
        "stage1_epochs":   {"values": [5, 10]},
        "stage2_lr":       {"values": [0.00001, 0.00005, 0.0001]},
        "stage2_epochs":   {"values": [10, 15, 20]},
        "label_smoothing": {"values": [0.0, 0.05, 0.1]},
    },
}


def sweep_train():
    run = wandb.init()
    cfg = wandb.config

    device = get_device()
    set_seed(42)

    train_df, val_df, test_df = make_splits()
    train_loader, val_loader, _ = make_loaders(train_df, val_df, test_df)
    class_weights = get_class_weights(train_df)

    model    = build_model(cfg.arch)
    ckpt     = CHECKPOINT_DIR / f"sweep_{run.id}.pt"
    hparams  = {
        "stage1_lr":       cfg.stage1_lr,
        "stage1_epochs":   cfg.stage1_epochs,
        "stage2_lr":       cfg.stage2_lr,
        "stage2_epochs":   cfg.stage2_epochs,
        "label_smoothing": cfg.label_smoothing,
    }

    train_two_stage(model, cfg.arch, train_loader, val_loader,
                    class_weights, hparams, device, ckpt, seed=42)

    if ckpt.exists():
        ckpt.unlink()   # save disk space during sweep


if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=sweep_train, count=SWEEP_COUNT)
