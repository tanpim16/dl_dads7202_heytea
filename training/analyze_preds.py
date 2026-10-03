"""
Error analysis จากผลทำนายรายรูปของทุก run (results/preds/<arch>_seed<N>.csv) — ไม่ต้องใช้ checkpoint / GPU

    python analyze_preds.py --results ../results_kaggle

ผลลัพธ์ -> <results>/analysis/
    cm_<arch>.png            confusion matrix รวม 5 seeds (normalize ตามแถว = recall)
    cm_all.png               4 arch ในรูปเดียว
    confusion_pairs.csv      คู่ true -> pred ที่ผิดบ่อย (ต่อ arch, เฉลี่ยต่อ seed)
    by_source.csv            accuracy แยกแหล่งรูป (bing / baidu / manual) ต่อ arch, mean±SD
    hard_images.csv          รูปที่ผิดใน run ส่วนใหญ่ (จาก 20 run) = รูปยาก / label อาจผิด
    hard_images.jpg          contact sheet ของรูปยาก พร้อม label จริง และคำทำนายที่พบบ่อยสุด
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import confusion_matrix

from config import ARCHS, CLASSES, DATA_DIR

SHORT = [c.replace("cha_", "") for c in CLASSES]


def load(results: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((results / "preds").glob("*_seed*.csv")):
        arch, seed = f.stem.rsplit("_seed", 1)
        rows.append(pd.read_csv(f).assign(arch=arch, seed=int(seed)))
    df = pd.concat(rows, ignore_index=True)
    df["correct"] = df["pred"] == df["class"]
    return df


def plot_cms(df, out: Path):
    archs = [a for a in ARCHS if a in set(df["arch"])]
    fig, axes = plt.subplots(1, len(archs), figsize=(4.6 * len(archs), 4.2))
    for ax, arch in zip(np.atleast_1d(axes), archs):
        d = df[df["arch"] == arch]
        cm = confusion_matrix(d["class"], d["pred"], labels=CLASSES, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, cbar=False,
                    xticklabels=SHORT, yticklabels=SHORT, ax=ax)
        ax.set_title(f"{arch} (5 seeds pooled)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        f1, a1 = plt.subplots(figsize=(5.5, 4.6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                    xticklabels=SHORT, yticklabels=SHORT, ax=a1)
        a1.set_title(f"Confusion matrix — {arch} (5 seeds pooled, row-normalized)")
        a1.set_xlabel("Predicted")
        a1.set_ylabel("Actual")
        f1.tight_layout()
        f1.savefig(out / f"cm_{arch}.png", dpi=150)
        plt.close(f1)
    fig.tight_layout()
    fig.savefig(out / "cm_all.png", dpi=150)
    plt.close(fig)


def hard_sheet(hard: pd.DataFrame, path: Path, S=240, cols=6):
    rows = int(np.ceil(len(hard) / cols))
    sheet = Image.new("RGB", (cols * S, rows * (S + 44)), "white")
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for i, (_, r) in enumerate(hard.iterrows()):
        x, y = (i % cols) * S, (i // cols) * (S + 44)
        try:
            sheet.paste(ImageOps.fit(Image.open(DATA_DIR / r["relpath"]).convert("RGB"), (S - 6, S - 6)), (x + 3, y + 3))
        except Exception:
            pass
        d.text((x + 4, y + S), f"true: {r['class'].replace('cha_', '')}", fill="black", font=font)
        d.text((x + 4, y + S + 20), f"pred: {r['top_pred'].replace('cha_', '')}  ({r['wrong_runs']}/{r['n_runs']} wrong)",
               fill="red", font=font)
    sheet.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--hard-min", type=float, default=0.5, help="ผิดอย่างน้อยกี่สัดส่วนของทุก run ถึงนับเป็นรูปยาก")
    a = ap.parse_args()
    results = Path(a.results)
    out = results / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = load(results)
    n_runs = df.groupby("filename").size().max()
    print(f"{df['arch'].nunique()} archs, {n_runs} runs, {df['filename'].nunique()} test images")

    plot_cms(df, out)

    # คู่ที่สับสน: จำนวนรูปผิดเฉลี่ยต่อ seed
    w = df[~df["correct"]]
    pairs = (w.groupby(["arch", "class", "pred"]).size() / df.groupby("arch")["seed"].nunique()).rename("wrong_per_seed")
    pairs = pairs.reset_index().sort_values(["arch", "wrong_per_seed"], ascending=[True, False])
    pairs.to_csv(out / "confusion_pairs.csv", index=False)
    print("\nคู่ที่สับสนบ่อยสุด (รวมทุก arch, รูปผิดเฉลี่ยต่อ run):")
    allp = (w.groupby(["class", "pred"]).size() / n_runs).rename("wrong_per_run").sort_values(ascending=False)
    print(allp.head(8).round(2).to_string())

    # แหล่งรูป: acc ต่อ run แล้วเฉลี่ยข้าม seed
    src = (df.groupby(["arch", "seed", "engine"])["correct"].mean().groupby(["arch", "engine"])
             .agg(["mean", lambda x: x.std(ddof=1)]).round(3))
    src.columns = ["acc_mean", "acc_std"]
    src = src.join(df[df["seed"] == df["seed"].min()].groupby(["arch", "engine"]).size().rename("n_test"))
    src.to_csv(out / "by_source.csv")
    print("\naccuracy แยกแหล่งรูป (mean±SD ข้าม seeds):")
    print(src.reset_index().pivot(index="arch", columns="engine", values="acc_mean").reindex(ARCHS).to_string())

    # รูปยาก: ผิดในหลาย run
    g = df.groupby("filename")
    hard = pd.DataFrame({
        "class": g["class"].first(), "relpath": g["relpath"].first(), "engine": g["engine"].first(),
        "n_runs": g.size(), "wrong_runs": g["correct"].apply(lambda s: int((~s).sum())),
        "top_pred": w.groupby("filename")["pred"].agg(lambda s: s.value_counts().index[0]),
        "archs_wrong": w.groupby("filename")["arch"].nunique(),
    }).fillna({"archs_wrong": 0})
    hard = hard[hard["wrong_runs"] >= a.hard_min * hard["n_runs"]].sort_values("wrong_runs", ascending=False)
    hard.reset_index().to_csv(out / "hard_images.csv", index=False)
    print(f"\nรูปยาก (ผิด >= {a.hard_min:.0%} ของ {n_runs} run): {len(hard)} รูป")
    print(hard.reset_index()[["filename", "class", "top_pred", "wrong_runs", "archs_wrong", "engine"]].to_string(index=False))
    if len(hard):
        hard_sheet(hard.reset_index(), out / "hard_images.jpg")
    print(f"\nSaved -> {out}/")


if __name__ == "__main__":
    main()
