import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
import wandb

from config import CHECKPOINT_DIR
from models import freeze_backbone, unfreeze_top


# ── Single epoch helpers ──────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        all_preds.extend(model(imgs.detach()).argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    # Avoid double forward: use logits cached above
    # (recalculated above only for metrics — acceptable for small datasets)
    n = len(loader.dataset)
    return (total_loss / n,
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="weighted", zero_division=0))


def _train_epoch_efficient(model, loader, optimizer, criterion, device):
    """Single forward pass per batch (stores preds from the same forward call)."""
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.detach().argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(loader.dataset)
    return (total_loss / n,
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="weighted", zero_division=0))


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * len(labels)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(loader.dataset)
    return (total_loss / n,
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="weighted", zero_division=0))


# ── Stage runner ──────────────────────────────────────────────────────────────

def _run_stage(model, train_loader, val_loader, optimizer, scheduler,
               criterion, device, max_epochs, patience, ckpt_path,
               stage_name, epoch_offset=0, epoch_callback=None):
    best_val_f1, no_improve = 0.0, 0

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc, tr_f1 = _train_epoch_efficient(
            model, train_loader, optimizer, criterion, device
        )
        vl_loss, vl_acc, vl_f1 = _eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        wandb.log({
            f"{stage_name}/train_loss": tr_loss,
            f"{stage_name}/train_acc":  tr_acc,
            f"{stage_name}/train_f1":   tr_f1,
            f"{stage_name}/val_loss":   vl_loss,
            f"{stage_name}/val_acc":    vl_acc,
            f"{stage_name}/val_f1":     vl_f1,
            "epoch": epoch_offset + epoch,
        })
        print(f"  [{stage_name}] E{epoch:02d} | "
              f"tr {tr_loss:.4f}/{tr_f1:.4f} | "
              f"vl {vl_loss:.4f}/{vl_f1:.4f}")
        if epoch_callback is not None:   # เช่น Optuna pruning (อาจ raise TrialPruned)
            epoch_callback(stage_name, epoch, vl_f1)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            no_improve  = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping (patience={patience})")
                break

    return best_val_f1


# ── Main entry point ──────────────────────────────────────────────────────────

def train_two_stage(model, arch, train_loader, val_loader,
                    class_weights, hparams, device, ckpt_path, seed, criterion=None,
                    epoch_callback=None):
    """
    Two-stage fine-tuning:
      Stage 1 — freeze backbone, train classifier head only
      Stage 2 — unfreeze top layers, fine-tune with lower LR
    Returns best val F1 achieved across both stages.
    criterion: loss ที่จะใช้ (เช่น FocalLoss); None = CrossEntropy + class_weights (เดิม)
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if criterion is None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=hparams.get("label_smoothing", 0.0),
        )
    criterion = criterion.to(device)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    freeze_backbone(model, arch)
    model.to(device)

    opt1 = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                lr=hparams["stage1_lr"])
    sch1 = CosineAnnealingLR(opt1, T_max=hparams["stage1_epochs"])

    print(f"\n{'='*55}\nStage 1 — {arch}  seed={seed}\n{'='*55}")
    best_s1 = _run_stage(
        model, train_loader, val_loader, opt1, sch1, criterion, device,
        max_epochs=hparams["stage1_epochs"], patience=5,
        ckpt_path=ckpt_path, stage_name="stage1", epoch_callback=epoch_callback,
    )

    # Restore best stage-1 weights before unfreezing
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    unfreeze_top(model, arch)

    opt2 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=hparams["stage2_lr"])
    sch2 = CosineAnnealingLR(opt2, T_max=hparams["stage2_epochs"])

    print(f"\n{'='*55}\nStage 2 — {arch}  seed={seed}\n{'='*55}")
    best_s2 = _run_stage(
        model, train_loader, val_loader, opt2, sch2, criterion, device,
        max_epochs=hparams["stage2_epochs"], patience=5,
        ckpt_path=ckpt_path, stage_name="stage2",
        epoch_offset=hparams["stage1_epochs"], epoch_callback=epoch_callback,
    )

    best = max(best_s1, best_s2)
    print(f"\n  Best val F1 → stage1={best_s1:.4f}  stage2={best_s2:.4f}  "
          f"overall={best:.4f}")
    return best
