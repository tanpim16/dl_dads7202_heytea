"""
สร้าง group split 70/10/20 ครั้งเดียว แล้วเซฟเป็น tea_dataset/split.csv (commit ขึ้น git)
ทุกคน / ทุก architecture / ทุก seed ใช้ split เดียวกันจากไฟล์นี้

group (รูปใน group เดียวกันต้องอยู่ split เดียวกัน กัน data leakage):
    รูปที่หามาเอง (source == manual) -> shop_id      (รูปเดี่ยว = 1 group, โฟลเดอร์ร้าน = 1 group)
    รูปที่ scrape                    -> search_class | keyword_id | engine
                                        (ผลค้นหาเดียวกันมักมีรูปร้าน/ชุด stock เดียวกัน)

ทำไมไม่ใช้ StratifiedGroupKFold: group ของรูป scrape ใหญ่และไม่เท่ากัน (1-61 รูป)
    -> ชาดำเย็นได้ val 3 / test 3 รูป; เลยสุ่มลำดับ group แล้วใส่ group ลง split ที่ยังขาดคลาสนั้นมากสุด
       ทำซ้ำ N_TRIALS รอบ เลือกรอบที่สัดส่วน "ต่อคลาส" และ "ต่อแหล่งรูป" (bing/baidu/manual) ใกล้ 70/10/20 ที่สุด
       (ไม่คุมแหล่งรูป -> val เกือบไม่มีรูป bing = val/test หน้าตาต่างจาก train)

รันใหม่เมื่อ metadata_clean.csv เปลี่ยน (ingest / sync):
    python make_split.py
"""
import numpy as np
import pandas as pd

from config import METADATA, CLASSES, SPLIT_SEED, SPLIT_FILE
TARGET = {"train": 0.7, "val": 0.1, "test": 0.2}
N_TRIALS = 20000


def group_key(df: pd.DataFrame) -> pd.Series:
    scraped = df["search_class"] + "|" + df["keyword_id"].astype(str) + "|" + df["engine"]
    return pd.Series(np.where(df["source"] == "manual", df["shop_id"], scraped), index=df.index)


def search_split(df: pd.DataFrame):
    names = list(TARGET)
    target = np.array([TARGET[n] for n in names])
    G = (df.groupby("group")["class"].value_counts().unstack(fill_value=0)
           .reindex(columns=CLASSES, fill_value=0))
    E = df.groupby("group")["engine"].value_counts().unstack(fill_value=0).reindex(G.index)
    M = G.to_numpy(float)
    ME = E.to_numpy(float)
    tot, totE = M.sum(0), ME.sum(0)
    rng = np.random.default_rng(SPLIT_SEED)

    best = None
    for _ in range(N_TRIALS):
        have = np.zeros((len(names), len(CLASSES)))
        haveE = np.zeros((len(names), ME.shape[1]))
        assign = np.empty(len(M), int)
        order = rng.permutation(len(M))
        if rng.random() < 0.5:   # ครึ่งหนึ่งของรอบ: ใส่ group ใหญ่ก่อน
            order = order[np.argsort(-M[order].sum(1), kind="stable")]
        for g in order:
            need = (target[:, None] * tot - have) / tot
            needE = (target[:, None] * totE - haveE) / totE
            k = int(np.argmax((need * M[g]).sum(1) / M[g].sum() + 0.5 * (needE * ME[g]).sum(1) / ME[g].sum()))
            assign[g] = k
            have[k] += M[g]
            haveE[k] += ME[g]
        dev = max(np.abs(have / tot - target[:, None]).max(),
                  0.5 * np.abs(haveE / totE - target[:, None]).max())   # แหล่งรูปคุมหลวมกว่าคลาส
        if best is None or dev < best[0]:
            best = (dev, assign.copy())

    dev, assign = best
    return dict(zip(G.index, np.array(names)[assign])), dev


def main():
    df = pd.read_csv(METADATA)
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    df["group"] = group_key(df)

    g2s, dev = search_split(df)
    df["split"] = df["group"].map(g2s)
    df[["filename", "class", "group", "split"]].to_csv(SPLIT_FILE, index=False)

    assert df.groupby("group")["split"].nunique().max() == 1   # ไม่มี group ไหนข้าม split
    t = pd.crosstab(df["class"], df["split"])[list(TARGET)]
    print(f"max deviation from 70/10/20 (class + source): {dev:.3f}  | groups: {df['group'].nunique()}")
    print(t.assign(**{f"{s}%": (t[s] / t.sum(1) * 100).round(1) for s in TARGET}).to_string())
    print("\nsource ต่อ split (ควรกระจายพอ ๆ กัน)")
    print(pd.crosstab(df["engine"], df["split"])[list(TARGET)].to_string())
    print(f"\nเซฟ -> {SPLIT_FILE}")


if __name__ == "__main__":
    main()
