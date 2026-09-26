"""
ChaNet — Tea Image Scraper (Hey Tea group)
==========================================
ดึงรูปเครื่องดื่มชา 5 คลาส จากหลาย search engine พร้อมกัน
สร้าง metadata.csv ให้พร้อมทำ group split + EDA

ติดตั้ง:
    pip install icrawler ddgs requests pillow imagehash pandas tqdm

รัน:
    python chanet_scraper.py                  # ดึงรูป + รวม + ตัดซ้ำ
    python chanet_scraper.py --skip-download  # รวม + ตัดซ้ำใหม่จาก _raw อย่างเดียว

ผลลัพธ์:
    tea_dataset/
        cha_thai/ cha_dam_yen/ cha_khiao_nom/ cha_nom_khai_muk/ cha_phonlamai/
        metadata.csv
        _duplicates/     <- รูปซ้ำย้ายมาที่นี่ (ไม่ลบ)
"""

import os
import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import requests
from ddgs import DDGS
from icrawler.builtin import BaiduImageCrawler

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BASE = Path("tea_dataset")
PER_KEYWORD = {"bing": 100, "baidu": 60}   # ต่อ 1 keyword ต่อ 1 engine
MIN_SIZE = (350, 350)
PHASH_THRESHOLD = 6        # ยิ่งน้อยยิ่งเข้มงวด
MAX_WORKERS = 3            # จำนวน keyword ที่ดึงพร้อมกัน (อย่าเกิน 4 เดี๋ยวโดนบล็อก)

# Google: icrawler 0.6.10 parser พัง (ได้ 0 รูปทุก keyword) -> ตัดออก
# Bing ผ่าน icrawler: เปลี่ยนหน้าไม่ได้ (ได้ ~20 รูป/keyword) + รับแค่ .jpg -> ใช้ ddgs(backend="bing") แทน
# Baidu: keyword ไทยได้แต่ขยะ (นาข้าว, แว่นตา, ไพ่นกกระจอก) -> ใช้เฉพาะ keyword อังกฤษ
#        ยังสมมาตรทุกคลาส เพราะทุกคลาสใช้กติกาเดียวกัน
ENGINES = {
    "bing": ("ddgs", {"th", "en"}),
    "baidu": (BaiduImageCrawler, {"en"}),
}
REGION = {"th": "th-th", "en": "us-en"}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

# keyword ไทยสำคัญมาก -> ได้รูปจากร้านไทยจริง
# keyword อังกฤษ -> ได้รูปสตูดิโอ/ต่างชาติ
# ต้องมีทั้งสองแบบทุกคลาส ไม่งั้นโมเดลเรียน "สไตล์ภาพ" แทน "ชนิดชา"
CONFIG = {
    "cha_thai": [
        ("ชาไทยเย็น แก้ว", "th"),
        ("ชาเย็น ร้านชา", "th"),
        ("ชานมไทย แก้วพลาสติก", "th"),
        ("thai milk tea iced glass", "en"),
        ("thai tea orange drink cup", "en"),
        # รอบ 2 (หลังคัดมือ keyword ไทยเหลือน้อย -> สัดส่วน th ไม่เท่ากันทุกคลาส)
        ("ชาเย็น แก้วพลาสติก ใส่น้ำแข็ง", "th"),
        ("ชาไทยเย็น ร้านชาริมทาง", "th"),
        ("ชานมเย็น สีส้ม แก้ว", "th"),
    ],
    "cha_dam_yen": [
        ("ชาดำเย็น แก้ว", "th"),
        ("ชาดำเย็น ร้านกาแฟ", "th"),
        ("ชาดำเย็นไม่ใส่นม", "th"),
        ("thai iced black tea glass", "en"),
        ("iced black tea no milk cup", "en"),
        ("ชาดำเย็น ใส่น้ำแข็ง แก้วพลาสติก", "th"),
        ("ชาดำเย็น ร้านชาริมทาง", "th"),
        ("ชาดำใส่น้ำแข็ง ชาดำหวาน", "th"),
    ],
    "cha_khiao_nom": [
        ("ชาเขียวนมเย็น แก้ว", "th"),
        ("ชาเขียวนม ร้านชา", "th"),
        ("ชาเขียวมัทฉะลาเต้เย็น", "th"),
        ("green milk tea iced cup", "en"),
        ("matcha latte iced plastic cup", "en"),
        ("ชาเขียวนม แก้วพลาสติก ใส่น้ำแข็ง", "th"),
        ("ชาเขียวนมเย็น ร้านชาริมทาง", "th"),
        ("ชาเขียวนมสด เย็น", "th"),
    ],
    "cha_nom_khai_muk": [
        ("ชานมไข่มุก แก้ว", "th"),
        ("ชาไข่มุก บราวน์ชูการ์", "th"),
        ("ชานมไข่มุกร้านดัง", "th"),
        ("bubble milk tea boba cup", "en"),
        ("brown sugar boba milk tea", "en"),
        ("ชานมไข่มุก แก้วพลาสติก ใส่น้ำแข็ง", "th"),
        ("ชานมไข่มุก ร้านชาริมทาง", "th"),
        ("ชานมไต้หวัน ไข่มุก เย็น", "th"),
    ],
    "cha_phonlamai": [
        ("ชาผลไม้ แก้ว", "th"),
        ("ชาพีชเย็น แก้ว", "th"),
        ("ชาลิ้นจี่ ชาเสาวรส เย็น", "th"),
        ("fruit tea iced cup slices", "en"),
        ("peach iced tea with fruit pieces", "en"),
        ("ชาผลไม้ แก้วพลาสติก ใส่น้ำแข็ง", "th"),
        ("ชาผลไม้ ร้านชาริมทาง", "th"),
        ("ชาผลไม้รวม ชาสตรอว์เบอร์รี่ ชาส้ม เย็น", "th"),
    ],
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# keyword index ที่เริ่มรอบเก็บรูปใหม่ — dedup ทำรอบเก่าก่อนเสมอ
# ไม่งั้นรูปรอบใหม่ของคลาสที่มาก่อน จะ "แย่ง" hash แล้วรูปรอบเก่า (ที่คัดมือแล้ว) หายไป
ROUND_STARTS = [0, 5]


# ---------------------------------------------------------
# STEP 1: ดาวน์โหลด (ขนานกัน)
# ---------------------------------------------------------
def fetch_image(url, dst_stem):
    try:
        r = requests.get(url, headers=UA, timeout=10)
        r.raise_for_status()
        from io import BytesIO
        with Image.open(BytesIO(r.content)) as im:
            fmt = (im.format or "").lower()
            if im.size[0] < MIN_SIZE[0] or im.size[1] < MIN_SIZE[1]:
                return False
        ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "bmp": ".bmp"}.get(fmt)
        if ext is None:   # gif, svg ฯลฯ
            return False
        Path(f"{dst_stem}{ext}").write_bytes(r.content)
        return True
    except Exception:
        return False


def crawl_ddgs(keyword, lang, tmp, max_num):
    urls, page, stale = [], 1, 0
    while len(urls) < max_num * 1.6 and page <= 15 and stale < 2:   # เผื่อลิงก์เสีย/รูปเล็ก
        try:
            res = DDGS().images(keyword, region=REGION[lang], backend="bing",
                                type_image="photo", max_results=100, page=page)
        except Exception:
            res = []
        new = [x["image"] for x in res if x["image"] not in urls]
        stale = stale + 1 if not new else 0
        urls += new
        page += 1

    saved = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, ok in enumerate(ex.map(lambda a: fetch_image(*a),
                                      [(u, tmp / f"{i:06d}") for i, u in enumerate(urls)])):
            saved += ok
            if saved >= max_num:
                break
    # ลบส่วนเกินที่โหลดเกินมา (thread ยังวิ่งอยู่ตอน break)
    files = sorted(p for p in tmp.iterdir() if p.suffix.lower() in IMG_EXT)
    for p in files[max_num:]:
        p.unlink()


def crawl_one(args):
    cls, kw_idx, keyword, lang, engine_name = args
    tmp = BASE / "_raw" / cls / f"kw{kw_idx:02d}_{engine_name}"
    tmp.mkdir(parents=True, exist_ok=True)

    # ถ้ามีรูปพอแล้วให้ข้าม -> รันซ้ำได้ไม่เสียเวลา
    existing = [p for p in tmp.iterdir() if p.suffix.lower() in IMG_EXT]
    if len(existing) >= PER_KEYWORD[engine_name] * 0.5:
        return f"  ข้าม [{cls}] kw{kw_idx:02d}/{engine_name} (มี {len(existing)} รูป)"

    try:
        if ENGINES[engine_name][0] == "ddgs":
            # ddgs เริ่มจากหน้า 1 ใหม่ทุกครั้ง -> ล้างของเก่าก่อน ไม่งั้นไฟล์เก่า/ใหม่ปนกัน
            shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            crawl_ddgs(keyword, lang, tmp, PER_KEYWORD[engine_name])
            n = len([p for p in tmp.iterdir() if p.suffix.lower() in IMG_EXT])
            return f"  [{cls}] kw{kw_idx:02d}/{engine_name}: {n} รูป — {keyword}"
        crawler = ENGINES[engine_name][0](
            storage={"root_dir": str(tmp)},
            downloader_threads=8,
            parser_threads=2,
            log_level=40,
        )
        crawler.crawl(
            keyword=keyword,
            max_num=PER_KEYWORD[engine_name],
            min_size=MIN_SIZE,
            file_idx_offset="auto",
        )
        n = len([p for p in tmp.iterdir() if p.suffix.lower() in IMG_EXT])
        return f"  [{cls}] kw{kw_idx:02d}/{engine_name}: {n} รูป — {keyword}"
    except Exception as e:
        return f"  ! [{cls}] kw{kw_idx:02d}/{engine_name} ล้มเหลว: {type(e).__name__}"


def download_all():
    jobs = []
    for cls, kws in CONFIG.items():
        for i, (kw, lang) in enumerate(kws):
            for eng, (_, langs) in ENGINES.items():
                if lang in langs:
                    jobs.append((cls, i, kw, lang, eng))

    print(f"เริ่มดึงรูป: {len(jobs)} งาน ({MAX_WORKERS} พร้อมกัน)\n")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for msg in ex.map(crawl_one, jobs):
            print(msg)
    print("\nดาวน์โหลดเสร็จ\n")


# ---------------------------------------------------------
# STEP 2: รวมไฟล์ + metadata
# ---------------------------------------------------------
def looks_grayscale(im, tol=8):
    # รูปขาวดำจากเว็บส่วนใหญ่เซฟเป็น RGB -> ดู mode อย่างเดียวไม่พอ
    if im.mode in ("L", "1", "LA"):
        return True
    a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.int16)
    return bool(np.abs(a - a.mean(axis=2, keepdims=True)).max() <= tol)


def read_meta(p):
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            return im.size[0], im.size[1], im.mode, looks_grayscale(im)
    except Exception:
        return None


def consolidate():
    rows = []
    for cls, kws in CONFIG.items():
        out = BASE / cls
        # สร้างใหม่จาก _raw ทุกครั้ง ไม่งั้นไฟล์เก่าค้าง / รูปที่ย้ายไป _duplicates ถูก copy กลับมา
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        cnt = 0

        for i, (kw, lang) in enumerate(kws):
            for eng in ENGINES:
                tmp = BASE / "_raw" / cls / f"kw{i:02d}_{eng}"
                if not tmp.exists():
                    continue

                for src in sorted(tmp.iterdir()):
                    if src.suffix.lower() not in IMG_EXT:
                        continue
                    m = read_meta(src)
                    if m is None:
                        continue
                    w, h, mode, gray = m
                    if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
                        continue

                    # ชื่อผูกกับไฟล์ต้นทางใน _raw (ไม่ใช้ตัวนับ) -> เพิ่ม/ลดรูปแล้วชื่อไฟล์อื่นไม่เลื่อน
                    # ผลคัดมือ (curation/manual.csv) อ้างชื่อไฟล์นี้ ห้ามเปลี่ยนรูปแบบ
                    name = f"{cls}_kw{i:02d}_{eng}_{src.stem}{src.suffix.lower()}"
                    os.link(src, out / name)   # hardlink: ไม่กินพื้นที่ดิสก์ซ้ำ
                    cnt += 1

                    rows.append({
                        "filename": name,
                        "relpath": f"{cls}/{name}",
                        "class": cls,
                        "keyword_id": f"kw{i:02d}",
                        "round": sum(i >= r for r in ROUND_STARTS),
                        "keyword": kw,
                        "lang": lang,
                        "engine": eng,
                        "source": "web_scrape",
                        "shop_id": "",        # <- เติมเองตอนคัดรูป ถ้าระบุร้านได้
                        "width": w,
                        "height": h,
                        "aspect_ratio": round(w / h, 4),
                        "mode": mode,
                        "is_grayscale": gray,
                    })
        print(f"  {cls}: {cnt} รูป")

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# STEP 3: ตัดรูปซ้ำ
# ---------------------------------------------------------
def dedup(df):
    dup_dir = BASE / "_duplicates"
    shutil.rmtree(dup_dir, ignore_errors=True)
    dup_dir.mkdir()

    hashes = []
    for _, r in df.iterrows():
        try:
            with Image.open(BASE / r["relpath"]) as im:
                hashes.append(imagehash.phash(im.convert("RGB")).hash.flatten())
        except Exception:
            hashes.append(None)

    ok = np.array([h is not None for h in hashes])
    H = np.stack([h if h is not None else np.zeros(64, bool) for h in hashes])
    classes = df["class"].to_numpy()

    keep = ok.copy()
    dup_of = [""] * len(df)
    xclass = np.zeros(len(df), bool)   # มีรูปซ้ำอยู่ในคลาสอื่น -> label น่าสงสัย ต้องดูมือ
    kept_idx = []
    for i in df.sort_values("round", kind="stable").index:
        if not ok[i]:
            continue
        if kept_idx:
            K = np.array(kept_idx)
            d = (H[K] != H[i]).sum(axis=1)
            hit = K[d <= PHASH_THRESHOLD]
            if len(hit):
                j = hit[0]
                keep[i] = False
                dup_of[i] = df.at[j, "filename"]
                if classes[j] != classes[i]:
                    xclass[j] = True
                continue
        kept_idx.append(i)

    for i in np.where(~keep & ok)[0]:
        shutil.move(str(BASE / df.at[i, "relpath"]), str(dup_dir / df.at[i, "filename"]))

    dups = df.assign(dup_of=dup_of)[~keep & ok]
    dups.to_csv(dup_dir / "duplicates.csv", index=False)
    n_x = (dups["class"] != dups["dup_of"].str.extract(r"^(cha_[a-z_]+?)_kw")[0]).sum()
    print(f"\n  ตัดซ้ำ: {int((~keep & ok).sum())} รูป (ข้ามคลาส {n_x}), อ่านไม่ได้: {int((~ok).sum())} รูป")

    df = df.assign(xclass_dup=xclass)
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------
# STEP 4: สรุป
# ---------------------------------------------------------
def summary(df):
    print("\n" + "=" * 60)
    print("จำนวนรูปต่อคลาส (ก่อนคัดมือ)")
    print("=" * 60)
    print(df["class"].value_counts().sort_index().to_string())

    print("\nสัดส่วน keyword ไทย vs อังกฤษ (ต้องใกล้เคียงกันทุกคลาส)")
    ct = pd.crosstab(df["class"], df["lang"])
    print(ct.assign(th_pct=(ct["th"] / ct.sum(axis=1) * 100).round(1)).to_string())

    print("\nแยกตาม engine")
    print(pd.crosstab(df["class"], df["engine"]).to_string())

    n = df["class"].value_counts()
    print(f"\nImbalance ratio: {n.max() / n.min():.2f}")
    print(f"รวม: {len(df)} รูป  | grayscale: {int(df['is_grayscale'].sum())}"
          f"  | ซ้ำข้ามคลาส (ต้องดูมือ): {int(df['xclass_dup'].sum())}")
    print("\nถัดไป: คัดมือครั้งเดียว — เป้าหมายใช้ได้จริงคลาสละ 120-200 รูป")


if __name__ == "__main__":
    if "--skip-download" not in sys.argv:
        download_all()
    print("รวมไฟล์...")
    df = consolidate()
    print("\nตัดรูปซ้ำ...")
    df = dedup(df)
    df.to_csv(BASE / "metadata.csv", index=False)
    summary(df)
    print(f"\nเซฟ metadata -> {BASE / 'metadata.csv'}")