"""
Hyperparameter tuning ด้วย Optuna (TPE sampler + Median pruning) — แยก study ต่อ architecture

    python run_tune_optuna.py                       # ทุก arch, 10 trials/arch
    python run_tune_optuna.py --archs resnet50 --trials 15

- search space เดียวกับ W&B Sweep (tuning.SEARCH_SPACE)
- pruning: ใน stage 2 ถ้า val F1 ที่ epoch k ต่ำกว่า median ของ trial ก่อน ๆ ที่ epoch เดียวกัน -> หยุด trial (ประหยัด GPU)
- ถ้ามี WANDB_API_KEY: ทุก trial ถูก log เข้า W&B project heytea-cnn (group = optuna-<arch>)
- ผล: results/tuning/trials.csv, optuna_<arch>.png (optimization history + param importance)
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import wandb

from config import ARCHS
from tuning import SEARCH_SPACE, train_trial, log_trial, tuning_dir, wandb_mode, select_best


def objective_for(arch):
    def objective(trial: optuna.Trial):
        hp = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}
        # search space เล็ก (108 แบบ) -> TPE เสนอค่าซ้ำได้: ใช้ผลเดิมแทนการเทรนซ้ำ (seed เดียวกัน = ผลเดิม)
        for t in trial.study.trials:
            if t.number != trial.number and t.state == optuna.trial.TrialState.COMPLETE and t.params == hp:
                print(f"  trial {trial.number}: ซ้ำกับ trial {t.number} -> ใช้ค่าเดิม {t.value:.4f}")
                return t.value
        t0 = time.time()
        run = wandb.init(project="heytea-cnn", group=f"optuna-{arch}", name=f"optuna_{arch}_t{trial.number}",
                         config={"tool": "optuna", "arch": arch, **hp}, reinit=True, mode=wandb_mode())

        def cb(stage, epoch, vl_f1):
            if stage == "stage2":
                trial.report(vl_f1, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        try:
            value = train_trial(arch, hp, f"optuna_{arch}_{trial.number}", epoch_callback=cb)
            log_trial("optuna", arch, trial.number, hp, value, "complete", time.time() - t0)
            wandb.log({"best_val_f1": value})
            return value
        except optuna.TrialPruned:
            log_trial("optuna", arch, trial.number, hp, None, "pruned", time.time() - t0)
            raise
        finally:
            run.finish()
    return objective


def plot_study(study, arch):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        vals = [(t.number, t.value) for t in study.trials if t.value is not None]
        axes[0].plot([n for n, _ in vals], [v for _, v in vals], "o", label="trial")
        best, cur = [], -1
        for _, v in vals:
            cur = max(cur, v)
            best.append(cur)
        axes[0].plot([n for n, _ in vals], best, "-", color="red", label="best so far")
        pruned = [t.number for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        for n in pruned:
            axes[0].axvline(n, color="gray", alpha=0.2)
        axes[0].set(title=f"Optuna — {arch} (grey = pruned)", xlabel="trial", ylabel="best val weighted F1")
        axes[0].legend()
        imp = optuna.importance.get_param_importances(study)
        axes[1].barh(list(imp)[::-1], list(imp.values())[::-1])
        axes[1].set(title="Hyperparameter importance (fANOVA)")
        plt.tight_layout()
        plt.savefig(tuning_dir() / f"optuna_{arch}.png", dpi=120)
        plt.close(fig)
    except Exception as e:   # importance ต้องมี trial ที่จบอย่างน้อย 2 ตัว
        print(f"  ! plot {arch}: {e}")


def main(archs=None, trials=10):
    for arch in archs or ARCHS:
        print(f"\n{'#' * 60}\nOptuna — {arch}: {trials} trials\n{'#' * 60}")
        study = optuna.create_study(
            study_name=f"heytea_{arch}", direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=4),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        )
        study.optimize(objective_for(arch), n_trials=trials)
        print(f"  best {arch}: {study.best_value:.4f} {study.best_params}")
        plot_study(study, arch)
    select_best()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", choices=ARCHS)
    ap.add_argument("--trials", type=int, default=10)
    a = ap.parse_args()
    main(a.archs, a.trials)
