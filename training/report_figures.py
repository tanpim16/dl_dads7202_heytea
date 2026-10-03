"""
สร้างรูป / ตารางประกอบ README (รายงานส่งอาจารย์) -> report/

    python report_figures.py --results ../results_kaggle --log ../pilot_group.log

    eda_counts.png        จำนวนรูปต่อคลาส แยกแหล่งรูป (+ ตารางใน eda.json)
    eda_image_stats.png   ขนาดรูป / aspect ratio / ความสว่าง ต่อคลาส
    samples.jpg           ตัวอย่างรูปคลาสละ 5 รูป
    augmentation.png      ตัวอย่าง data augmentation ที่ใช้ตอนเทรน
    baseline_imagenet.png pre-trained ResNet-50 (ImageNet, ยังไม่ fine-tune) ทายรูปของเรา
    eyeball.png           ภาพเดียวกัน: ImageNet baseline vs VGG-16 vs ResNet-50 (fine-tuned, best seed)
    learning_curve.png    train/val loss + F1 ต่อ epoch (จาก log)
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageStat
from torchvision import models
from torchvision.models import ResNet50_Weights

from config import CLASSES, DATA_DIR, METADATA, SPLIT_FILE
from dataset import get_transforms

SHORT = [c.replace("cha_", "") for c in CLASSES]
SRC_NAME = {"bing": "Bing", "baidu": "Baidu", "manual": "Delivery apps / FB (manual)"}


def eda(out: Path):
    df = pd.read_csv(METADATA)
    sp = pd.read_csv(SPLIT_FILE)[["filename", "split"]]
    df = df.merge(sp, on="filename")

    ct = pd.crosstab(df["class"], df["engine"]).reindex(CLASSES)[["bing", "baidu", "manual"]]
    ax = ct.rename(columns=SRC_NAME).plot.barh(stacked=True, figsize=(8, 3.6),
                                                color=["#4C72B0", "#DD8452", "#55A868"])
    for i, tot in enumerate(ct.sum(1)):
        ax.text(tot + 3, i, str(tot), va="center", fontsize=9)
    ax.set_yticklabels(SHORT)
    ax.set_xlabel("images")
    ax.set_title(f"Images per class by source (total {len(df)})")
    plt.tight_layout()
    plt.savefig(out / "eda_counts.png", dpi=150)
    plt.close()

    bright = []
    for p in df["relpath"]:
        with Image.open(DATA_DIR / p) as im:
            bright.append(ImageStat.Stat(im.convert("L").resize((64, 64))).mean[0])
    df["brightness"] = bright

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, col, title in zip(axes, ["width", "aspect_ratio", "brightness"],
                              ["Image width (px, before resize)", "Aspect ratio (w/h)", "Mean brightness (0-255)"]):
        data = [df.loc[df["class"] == c, col] for c in CLASSES]
        ax.boxplot(data, labels=SHORT, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out / "eda_image_stats.png", dpi=150)
    plt.close()

    split = pd.crosstab(df["class"], df["split"]).reindex(CLASSES)[["train", "val", "test"]]
    stats = {
        "total": len(df),
        "per_class": ct.sum(1).to_dict(),
        "per_class_source": ct.to_dict(orient="index"),
        "split": split.to_dict(orient="index"),
        "imbalance_ratio": round(ct.sum(1).max() / ct.sum(1).min(), 2),
        "width_median": df.groupby("class")["width"].median().to_dict(),
        "height_median": df.groupby("class")["height"].median().to_dict(),
        "aspect_median": df.groupby("class")["aspect_ratio"].median().round(2).to_dict(),
        "brightness_mean": df.groupby("class")["brightness"].mean().round(1).to_dict(),
        "grayscale": int(df["is_grayscale"].sum()),
        "min_side": int(df[["width", "height"]].min(1).min()),
    }
    (out / "eda.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return df


def sheet(paths_titles, path, cols, S=200, title_size=8):
    rows = int(np.ceil(len(paths_titles) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * S / 80, rows * (S + 40) / 80))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (p, t, color) in zip(np.atleast_1d(axes).ravel(), paths_titles):
        ax.imshow(ImageOps.fit(Image.open(DATA_DIR / p).convert("RGB"), (S, S)))
        ax.set_title(t, fontsize=title_size, color=color)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def samples(df, out: Path, k=5):
    items = []
    for c in CLASSES:
        sub = df[df["class"] == c].sample(k, random_state=7)
        items += [(p, f"{c.replace('cha_', '')} ({e})", "black") for p, e in zip(sub["relpath"], sub["engine"])]
    sheet(items, out / "samples.jpg", cols=k)


def augmentation(df, out: Path):
    torch.manual_seed(0)
    p = df[df["class"] == "cha_thai"].sample(1, random_state=3)["relpath"].iloc[0]
    img = Image.open(DATA_DIR / p).convert("RGB")
    tf = get_transforms("train")
    mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.9))
    axes[0].imshow(ImageOps.fit(img, (224, 224)))
    axes[0].set_title("original")
    for ax in axes[1:]:
        ax.imshow((tf(img) * std + mean).clamp(0, 1).permute(1, 2, 0))
        ax.set_title("augmented")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / "augmentation.png", dpi=120)
    plt.close()


@torch.no_grad()
def imagenet_top1(paths):
    w = ResNet50_Weights.IMAGENET1K_V2
    m = models.resnet50(weights=w).eval()
    tf = w.transforms()
    res = []
    for p in paths:
        pr = m(tf(Image.open(DATA_DIR / p).convert("RGB"))[None]).softmax(1)[0]
        v, i = pr.max(0)
        res.append((w.meta["categories"][i], float(v)))
    return res


@torch.no_grad()
def baseline_all_models(out: Path):
    """ImageNet weights ทั้ง 4 arch (ยังไม่ตัดต่อ/ไม่เทรน) ทายรูปเดียวกัน คลาสละ 1 รูปจาก test set"""
    from torchvision.models import VGG16_Weights, EfficientNet_B3_Weights, MobileNet_V3_Large_Weights
    specs = [("VGG-16", models.vgg16, VGG16_Weights.IMAGENET1K_V1),
             ("ResNet-50", models.resnet50, ResNet50_Weights.IMAGENET1K_V2),
             ("EfficientNet-B3", models.efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
             ("MobileNet-V3-L", models.mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2)]
    test = pd.read_csv(SPLIT_FILE).query("split == 'test'")
    meta = pd.read_csv(METADATA).set_index("filename")
    picks = [test[test["class"] == c]["filename"].sample(1, random_state=1).iloc[0] for c in CLASSES]
    imgs = [Image.open(DATA_DIR / meta.at[f, "relpath"]).convert("RGB") for f in picks]
    fig, axes = plt.subplots(len(specs), len(picks), figsize=(3.1 * len(picks), 3.4 * len(specs)))
    table = {}
    for r, (name, fn, w) in enumerate(specs):
        m, tf = fn(weights=w).eval(), w.transforms()
        table[name] = []
        for c, (f, img) in enumerate(zip(picks, imgs)):
            pr = m(tf(img)[None]).softmax(1)[0]
            v, i = pr.max(0)
            lab = w.meta["categories"][i]
            table[name].append((lab, round(float(v), 3)))
            ax = axes[r, c]
            ax.imshow(ImageOps.fit(img, (224, 224)))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"true: {CLASSES[c].replace('cha_', '')}\npred: {lab[:20]} ({float(v):.0%})", fontsize=8, color="red")
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.suptitle("Pre-trained CNNs (ImageNet weights, no layer changes, no training) on our test images", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "baseline_all_models.png", dpi=110)
    plt.close()
    return table


def baseline_and_eyeball(results: Path, out: Path):
    test = pd.read_csv(SPLIT_FILE).query("split == 'test'")
    meta = pd.read_csv(METADATA).set_index("filename")
    runs = pd.read_csv(results / "runs.csv")
    best = {a: int(runs[runs.arch == a].sort_values("f1_macro").iloc[-1]["seed"]) for a in ["vgg16", "resnet50"]}
    P = {a: pd.read_csv(results / "preds" / f"{a}_seed{s}.csv").set_index("filename") for a, s in best.items()}

    # 2 รูป/คลาส: 1 รูปที่ทั้งคู่ถูก + 1 รูปที่ VGG ผิดแต่ ResNet ถูก (ถ้ามี) -> เห็นความต่างของโมเดล
    picks = []
    for c in CLASSES:
        t = test[test["class"] == c]["filename"]
        ok = [f for f in t if P["vgg16"].at[f, "pred"] == c and P["resnet50"].at[f, "pred"] == c]
        diff = [f for f in t if P["vgg16"].at[f, "pred"] != c and P["resnet50"].at[f, "pred"] == c]
        picks += ok[:1] + (diff[:1] or ok[1:2])
    paths = [meta.at[f, "relpath"] for f in picks]
    top1 = imagenet_top1(paths)

    items = [(p, f"true: {meta.at[f, 'class'].replace('cha_', '')}\nImageNet: {n[:22]} ({v:.0%})", "black")
             for f, p, (n, v) in zip(picks, paths, top1)]
    sheet(items, out / "baseline_imagenet.png", cols=5, title_size=7)

    items = []
    for f, p, (n, v) in zip(picks, paths, top1):
        c = meta.at[f, "class"]
        vg, rs = P["vgg16"].at[f, "pred"], P["resnet50"].at[f, "pred"]
        mark = lambda x: "OK" if x == c else "X"
        items.append((p, f"true: {c.replace('cha_', '')}\nImageNet: {n[:18]}\n"
                         f"VGG16: {vg.replace('cha_', '')} {mark(vg)}\nResNet50: {rs.replace('cha_', '')} {mark(rs)}",
                      "black" if vg == c else "red"))
    sheet(items, out / "eyeball.png", cols=5, title_size=7)
    return {"best_seed": best, "imagenet_top1": [(meta.at[f, "class"], n, round(v, 3)) for f, (n, v) in zip(picks, top1)]}


def learning_curve(log: Path, out: Path):
    rows, stage_off = [], 0
    for line in log.read_text().splitlines():
        m = re.search(r"\[(stage\d)\] E(\d+) \| tr ([\d.]+)/([\d.]+) \| vl ([\d.]+)/([\d.]+)", line)
        if m:
            st, e = m.group(1), int(m.group(2))
            rows.append({"stage": st, "epoch": e, "tr_loss": float(m.group(3)), "tr_f1": float(m.group(4)),
                         "vl_loss": float(m.group(5)), "vl_f1": float(m.group(6))})
    df = pd.DataFrame(rows)
    s1 = df[df["stage"] == "stage1"]["epoch"].max()
    df["x"] = np.where(df["stage"] == "stage1", df["epoch"], df["epoch"] + s1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, k, lab in zip(axes, ["loss", "f1"], ["loss", "weighted F1"]):
        ax.plot(df["x"], df[f"tr_{k}"], "o-", ms=3, label="train")
        ax.plot(df["x"], df[f"vl_{k}"], "o-", ms=3, label="val")
        ax.axvline(s1 + 0.5, color="gray", ls="--", lw=1)
        ax.text(s1 + 0.7, ax.get_ylim()[1] * 0.95, "stage 2 (unfreeze top)", fontsize=8, va="top")
        ax.set_xlabel("epoch")
        ax.set_ylabel(lab)
        ax.legend()
    fig.suptitle("Learning curve — ResNet-50, seed 11 (pilot run)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "learning_curve.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results_kaggle")
    ap.add_argument("--log", default="../pilot_group.log")
    ap.add_argument("--out", default="../report")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = eda(out)
    samples(df, out)
    augmentation(df, out)
    info = baseline_and_eyeball(Path(a.results), out)
    info["baseline_all_models"] = baseline_all_models(out)
    if Path(a.log).exists():
        learning_curve(Path(a.log), out)
    (out / "baseline.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=1, ensure_ascii=False))
    print(open(out / "eda.json").read())


if __name__ == "__main__":
    main()


def parse_kaggle_log(log: Path) -> pd.DataFrame:
    """แยก epoch ของทุก arch/seed จาก log ของ run_final (header 'Stage N — <arch>  seed=<s>')"""
    rows, arch, seed = [], None, None
    for line in log.read_text(errors="ignore").splitlines():
        h = re.search(r"Stage \d — (\S+)\s+seed=(\d+)", line)
        if h:
            arch, seed = h.group(1), int(h.group(2))
            continue
        m = re.search(r"\[(stage\d)\] E(\d+) \| tr ([\d.]+)/([\d.]+) \| vl ([\d.]+)/([\d.]+)", line)
        if m and arch:
            rows.append({"arch": arch, "seed": seed, "stage": m.group(1), "epoch": int(m.group(2)),
                         "tr_loss": float(m.group(3)), "tr_f1": float(m.group(4)),
                         "vl_loss": float(m.group(5)), "vl_f1": float(m.group(6))})
    df = pd.DataFrame(rows)
    s1 = df[df["stage"] == "stage1"].groupby(["arch", "seed"])["epoch"].max().rename("s1_len")
    df = df.join(s1, on=["arch", "seed"])
    df["x"] = np.where(df["stage"] == "stage1", df["epoch"], df["epoch"] + df["s1_len"])
    return df


def learning_curves_all(log: Path, runs_csv: Path, out: Path):
    """4 arch x (loss, F1): median seed (ตาม test macro F1) เส้นหนา + seed อื่นเส้นจาง (val)"""
    df = parse_kaggle_log(log)
    runs = pd.read_csv(runs_csv)
    archs = [a for a in ["vgg16", "resnet50", "efficientnet_b3", "mobilenet_v3_large"] if a in set(df["arch"])]
    fig, axes = plt.subplots(len(archs), 2, figsize=(12, 3.1 * len(archs)))
    med = {}
    for r, a in enumerate(archs):
        rr = runs[runs["arch"] == a].sort_values("f1_macro").reset_index(drop=True)
        ms = int(rr.loc[len(rr) // 2, "seed"])
        med[a] = ms
        for c, (k, lab) in enumerate([("loss", "loss"), ("f1", "weighted F1")]):
            ax = axes[r, c]
            for s, g in df[df["arch"] == a].groupby("seed"):
                if s != ms:
                    ax.plot(g["x"], g[f"vl_{k}"], color="tab:orange", alpha=0.18, lw=1)
            g = df[(df["arch"] == a) & (df["seed"] == ms)]
            ax.plot(g["x"], g[f"tr_{k}"], "o-", ms=3, color="tab:blue", label=f"train (seed {ms})")
            ax.plot(g["x"], g[f"vl_{k}"], "o-", ms=3, color="tab:orange", label=f"val (seed {ms})")
            ax.axvline(g["s1_len"].iloc[0] + 0.5, color="gray", ls="--", lw=1)
            ax.set_title(f"{a} — {lab}   (bold = seed {ms})", fontsize=10)
            ax.set_xlabel("epoch (dashed line = start of stage 2)")
            if r == 0:
                ax.plot([], [], color="tab:orange", alpha=0.3, label="val, other 4 seeds")
                ax.legend(fontsize=8)
    fig.suptitle("Learning curves — final Kaggle runs. Bold = median seed by test macro F1; faint = val of the other seeds",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "learning_curves_all.png", dpi=130)
    plt.close()
    # สรุปช่องว่าง train-val ตอนจบ (ไว้เขียนเรื่อง overfit)
    last = df.sort_values("x").groupby(["arch", "seed"]).tail(1)
    gap = last.assign(gap_f1=last["tr_f1"] - last["vl_f1"], gap_loss=last["vl_loss"] - last["tr_loss"]) \
              .groupby("arch")[["tr_f1", "vl_f1", "gap_f1", "tr_loss", "vl_loss", "gap_loss"]].agg(["mean", "std"]).round(3)
    epochs = df.groupby(["arch", "seed"])["x"].max().groupby("arch").agg(["mean", "min", "max"])
    return med, gap, epochs
