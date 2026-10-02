"""
Error analysis จาก checkpoint 1 ตัว (รันในเครื่องได้ ไม่ต้องใช้ GPU)

    python error_analysis.py --arch resnet50 --ckpt ../checkpoints/resnet50_seed11.pt
    python error_analysis.py --arch resnet50 --ckpt ../checkpoints/pilot_resnet50_group.pt --name pilot

ผลลัพธ์ -> results/error_analysis/<name>/
    preds.csv          ทุกรูปใน test: true / pred / confidence / แหล่งรูป / group
    confusion_pairs    คู่คลาสที่สับสนบ่อยสุด (true -> pred)
    by_source.csv      accuracy แยกตามแหล่งรูป (bing / baidu / manual) — เช็คว่าโมเดลเรียนสไตล์ภาพไหม
    wrong_all.jpg      รูปที่ทายผิดทุกรูป (เรียงจากมั่นใจผิดมากสุด)
    gradcam_wrong.jpg  GradCAM ของรูปที่ทายผิดทุกรูป
    gradcam_correct.jpg GradCAM ของรูปที่ทายถูก (สุ่มคลาสละ 2 รูป) ไว้เทียบว่าโมเดลมองตรงไหน
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import CLASSES, DATA_DIR, RESULTS_DIR
from dataset import make_splits, TeaDataset, get_transforms
from gradcam import GradCAM, _denormalize, _overlay
from models import build_model, get_gradcam_layer
from utils import get_device

SHORT = [c.replace("cha_", "") for c in CLASSES]


@torch.no_grad()
def predict_df(model, df, device):
    ds = TeaDataset(df, get_transforms("test"))
    probs = []
    for i in range(0, len(ds), 32):
        x = torch.stack([ds[j][0] for j in range(i, min(i + 32, len(ds)))]).to(device)
        probs.append(torch.softmax(model(x), 1).cpu().numpy())
    P = np.concatenate(probs)
    out = df.reset_index(drop=True).copy()
    out["pred"] = [CLASSES[k] for k in P.argmax(1)]
    out["confidence"] = P.max(1).round(4)
    out["p_true"] = P[np.arange(len(out)), [CLASSES.index(c) for c in out["class"]]].round(4)
    out["correct"] = out["pred"] == out["class"]
    return out


def gradcam_grid(model, arch, rows, device, title, path, cols=4):
    if rows.empty:
        return
    gcam = GradCAM(model, get_gradcam_layer(model, arch))
    tf = get_transforms("test")
    n = len(rows)
    r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, cols * 2, figsize=(cols * 4, r * 2.2))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (_, row) in enumerate(rows.iterrows()):
        from PIL import Image
        x = tf(Image.open(DATA_DIR / row["relpath"]).convert("RGB"))
        cam, _ = gcam(x.to(device))
        img = _denormalize(x)
        a, b = axes[k // cols, (k % cols) * 2], axes[k // cols, (k % cols) * 2 + 1]
        a.imshow(img)
        b.imshow(_overlay(img, cam))
        color = "green" if row["correct"] else "red"
        a.set_title(f"true: {row['class'].replace('cha_', '')}", fontsize=7)
        b.set_title(f"pred: {row['pred'].replace('cha_', '')} ({row['confidence']:.2f})", fontsize=7, color=color)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    out = RESULTS_DIR / "error_analysis" / (a.name or a.ckpt.split("/")[-1].removesuffix(".pt"))
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = build_model(a.arch)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.to(device).eval()

    _, val_df, test_df = make_splits()
    df = predict_df(model, test_df if a.split == "test" else val_df, device)
    df.to_csv(out / "preds.csv", index=False)
    wrong = df[~df["correct"]].sort_values("confidence", ascending=False)

    pairs = (wrong.groupby(["class", "pred"]).size().rename("n").reset_index()
                  .sort_values("n", ascending=False))
    pairs.to_csv(out / "confusion_pairs.csv", index=False)
    src = df.groupby("engine")["correct"].agg(n="size", accuracy="mean").round(3)
    src.to_csv(out / "by_source.csv")

    print(f"{a.split}: {len(df)} images | wrong {len(wrong)} | acc {df['correct'].mean():.3f}")
    print("\nคู่ที่สับสน (true -> pred):")
    print(pairs.to_string(index=False))
    print("\naccuracy แยกตามแหล่งรูป:")
    print(src.to_string())
    print("\nทายผิดแบบมั่นใจสูง (>=0.8) = น่าจะ label ผิด หรือรูปกำกวม ควรเปิดดู:")
    print(wrong[wrong["confidence"] >= 0.8][["filename", "class", "pred", "confidence", "engine"]].to_string(index=False))

    gradcam_grid(model, a.arch, wrong, device,
                 f"{a.arch} — all {len(wrong)} misclassified (most confident first)", out / "gradcam_wrong.jpg")
    corr = pd.concat([g.sample(min(2, len(g)), random_state=0)
                      for _, g in df[df["correct"]].groupby("class")])
    gradcam_grid(model, a.arch, corr, device, f"{a.arch} — correct (2 per class)", out / "gradcam_correct.jpg")
    print(f"\nSaved -> {out}/")


if __name__ == "__main__":
    main()
