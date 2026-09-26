"""
ChaNet — คัดรูป (Hey Tea group)
================================
ขั้นที่ 1  score : CLIP zero-shot ให้คะแนนทุกรูป -> curation/clip_scores.csv
ขั้นที่ 2  sheets: ทำ contact sheet (รูปมีเลขกำกับ) ไว้ดูด้วยตา -> curation/sheets/
ขั้นที่ 3  decide: แปลงผลคัดมือจาก sheet (curation/sheet_decisions.txt) -> curation/manual.csv
ขั้นที่ 4  apply : รวม auto-reject + คัดมือ -> ย้ายรูปทิ้งไป tea_dataset/_rejected/,
                   ย้ายรูปผิดคลาสไปคลาสที่ถูก (relabel) และเขียน tea_dataset/metadata_clean.csv

รัน:
    python chanet_curate.py score
    python chanet_curate.py sheets
    python chanet_curate.py decide
    python chanet_curate.py apply

รูปแบบ sheet_decisions.txt (1 บรรทัด / 1 sheet / 1 การกระทำ):
    <sheet>|keep_only|<idx,...>|<reason>          # นอกจาก idx ที่ระบุ = reject
    <sheet>|reject_only|<idx,...>|<reason>        # นอกจาก idx ที่ระบุ = keep
    <sheet>|relabel:<class>|<idx,...>|            # ย้ายไปคลาสที่ถูก (override keep/reject)

กติกาคัด (ใช้เหมือนกันทุกคลาส):
    เก็บ : ภาพถ่ายจริง, เครื่องดื่มของคลาสนั้นเป็นจุดเด่น, เห็นแก้ว/แก้วพลาสติกอย่างน้อย 1 ใบ
    ทิ้ง : ไม่ใช่เครื่องดื่ม, ภาพวาด/การ์ตูน/3D render, โลโก้, เมนู/โปสเตอร์ที่ตัวหนังสือเด่นกว่าแก้ว,
           ชาร้อนในถ้วยเซรามิก, ใบชา/ผงชาอย่างเดียว, ขวด/กล่องสำเร็จรูป, คอลลาจหลายชนิด, ผิดคลาส
ข้อสำคัญ: CLIP class prediction ใช้แค่ "ชี้ให้คนดู" ห้ามใช้ลบอัตโนมัติ
          ไม่งั้น dataset เหลือแต่รูปที่ CLIP ทายง่าย -> accuracy สูงเกินจริง
เคสกำกวม (ตัดออกจากทุกคลาส ให้ขอบเขตคลาสชัด):
    ชาไทย/ชาเขียว + ไข่มุก, ชามะนาว (ผลไม้มีแค่มะนาว), นมชมพู/เผือก + ไข่มุก,
    กาแฟดำเย็น (หน้าตาเหมือนชาดำเย็น), ภาพ AI render ชัด ๆ (กัน shortcut เรื่องสไตล์ภาพ)
"""

import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE = Path("tea_dataset")
OUT = Path("curation")
META = BASE / "metadata.csv"
SCORES = OUT / "clip_scores.csv"
MANUAL = OUT / "manual.csv"          # filename,decision(keep/reject/relabel:<class>),reason
SHEET_DECISIONS = OUT / "sheet_decisions.txt"

AUTO_REJECT = 0.15   # p_drink ต่ำกว่านี้ = ขยะแน่ ๆ (แต่ยังทำ sheet ให้คนตรวจซ้ำ)
AUTO_KEEP = 0.60     # p_drink สูงกว่านี้ + CLIP class ตรง = ไม่ต้องดูทีละรูป (สุ่มตรวจ)

DRINK_PROMPTS = [
    "a photo of an iced tea drink in a glass",
    "a photo of an iced milk tea in a plastic cup",
    "a photo of a cold drink with ice in a cup",
    "a photo of bubble tea in a plastic cup",
]
JUNK_PROMPTS = {
    "logo": "a logo or brand graphic",
    "illustration": "a cartoon, drawing or illustration",
    "text": "a menu, poster or advertisement with lots of text",
    "person": "a photo of a person",
    "food": "a photo of a plate of food",
    "leaves": "a photo of dry tea leaves or tea powder",
    "hot_tea": "a hot cup of tea in a ceramic teacup",
    "bottle": "a bottled or packaged drink product",
    "vehicle": "a photo of an airplane or vehicle",
    "scenery": "a photo of a landscape, building or temple",
    "object": "a photo of an unrelated object",
}
CLASS_PROMPTS = {
    "cha_thai": "a photo of Thai iced milk tea, bright orange colour",
    "cha_dam_yen": "a photo of Thai iced black tea without milk, dark clear brown-orange",
    "cha_khiao_nom": "a photo of iced green milk tea or matcha latte, green colour",
    "cha_nom_khai_muk": "a photo of bubble milk tea with black tapioca pearls",
    "cha_phonlamai": "a photo of iced fruit tea with fruit slices",
}


def load_clip():
    import open_clip
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    return model.eval().to(dev), prep, tok, dev


@torch.no_grad()
def encode_text(model, tok, dev, prompts):
    t = model.encode_text(tok(prompts).to(dev))
    return t / t.norm(dim=-1, keepdim=True)


@torch.no_grad()
def score():
    df = pd.read_csv(META)
    model, prep, tok, dev = load_clip()

    gate_prompts = DRINK_PROMPTS + list(JUNK_PROMPTS.values())
    T_gate = encode_text(model, tok, dev, gate_prompts)
    T_cls = encode_text(model, tok, dev, list(CLASS_PROMPTS.values()))
    n_pos = len(DRINK_PROMPTS)
    junk_names = list(JUNK_PROMPTS)
    cls_names = list(CLASS_PROMPTS)

    rows = []
    B = 64
    for s in range(0, len(df), B):
        chunk = df.iloc[s:s + B]
        ims = torch.stack([prep(Image.open(BASE / p).convert("RGB")) for p in chunk["relpath"]]).to(dev)
        f = model.encode_image(ims)
        f = f / f.norm(dim=-1, keepdim=True)

        pg = (100 * f @ T_gate.T).softmax(-1).cpu().numpy()
        pc = (100 * f @ T_cls.T).softmax(-1).cpu().numpy()
        for k, fn in enumerate(chunk["filename"]):
            j = pg[k, n_pos:].argmax()
            rows.append({
                "filename": fn,
                "p_drink": round(float(pg[k, :n_pos].sum()), 4),
                "top_junk": junk_names[j],
                "p_top_junk": round(float(pg[k, n_pos + j]), 4),
                "clip_class": cls_names[pc[k].argmax()],
                "clip_class_p": round(float(pc[k].max()), 4),
            })
        print(f"\r  {min(s + B, len(df))}/{len(df)}", end="", flush=True)

    sc = df[["filename", "class", "keyword_id", "lang", "engine", "xclass_dup"]].merge(pd.DataFrame(rows))
    sc["clip_agree"] = sc["class"] == sc["clip_class"]
    sc["bucket"] = np.select(
        [sc["p_drink"] < AUTO_REJECT, (sc["p_drink"] >= AUTO_KEEP) & sc["clip_agree"] & ~sc["xclass_dup"]],
        ["auto_reject", "auto_keep"], "review")
    OUT.mkdir(exist_ok=True)
    sc.to_csv(SCORES, index=False)

    print("\n\nbucket ต่อคลาส")
    print(pd.crosstab(sc["class"], sc["bucket"]).to_string())
    print("\nสัดส่วนที่เป็นเครื่องดื่ม (p_drink >= AUTO_REJECT) ต่อ keyword  <- keyword ไหนต่ำ ควรเปลี่ยน")
    y = sc.assign(ok=sc["p_drink"] >= AUTO_REJECT).groupby(["class", "keyword_id", "lang", "engine"])["ok"]
    print(y.agg(["sum", "count", "mean"]).round(2).to_string())


def contact_sheet(files, labels, path, cols=8, S=220):
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * S, rows * S), "white")
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for i, (f, lab) in enumerate(zip(files, labels)):
        x, y = (i % cols) * S, (i // cols) * S
        try:
            sheet.paste(ImageOps.fit(Image.open(f).convert("RGB"), (S - 4, S - 4)), (x + 2, y + 2))
        except Exception:
            pass
        d.rectangle([x + 2, y + 2, x + 62, y + 28], fill="black")
        d.text((x + 6, y + 3), str(lab), fill="yellow", font=font)
    sheet.save(path, quality=75)


def sheets(per_sheet=48):
    sc = pd.read_csv(SCORES)
    # --new: ทำ sheet เฉพาะรูปที่ยังไม่เคยคัดมือ (รอบถัดไป) โดยไม่ทับ sheet/index เดิม
    new = "--new" in sys.argv
    tag = ""
    if new:
        done = set(pd.read_csv(MANUAL)["filename"])
        sc = sc[~sc["filename"].isin(done) & (sc["bucket"] != "auto_reject")]
        n = len(list(OUT.glob("sheets_r*"))) + 2
        tag = f"r{n}__"
    sd = OUT / (f"sheets_{tag[:-2]}" if new else "sheets")
    shutil.rmtree(sd, ignore_errors=True)
    sd.mkdir(parents=True)
    index = []
    for cls in sc["class"].unique():
        for bucket in ["auto_reject", "review", "auto_keep"]:
            sub = sc[(sc["class"] == cls) & (sc["bucket"] == bucket)].sort_values("p_drink")
            for n, s in enumerate(range(0, len(sub), per_sheet)):
                part = sub.iloc[s:s + per_sheet]
                name = f"{tag}{cls}__{bucket}__{n:02d}.jpg"
                files = [BASE / cls / fn for fn in part["filename"]]
                contact_sheet(files, range(len(part)), sd / name)
                index += [{"sheet": name, "idx": k, "filename": fn} for k, fn in enumerate(part["filename"])]
    pd.DataFrame(index).to_csv(sd / "index.csv", index=False)
    print(f"  ทำ {len(set(r['sheet'] for r in index))} sheet -> {sd}")


def decide():
    idx = pd.concat([pd.read_csv(f) for f in OUT.glob("sheets*/index.csv")])
    by_sheet = {k: g.set_index("idx")["filename"] for k, g in idx.groupby("sheet")}
    dec = {}
    for line in SHEET_DECISIONS.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sheet, mode, ids, reason = (line.split("|") + [""])[:4]
        fns = by_sheet[sheet]
        ids = {int(i) for i in ids.split(",") if i.strip()}
        gone = ids - set(fns.index)   # รูปที่ไฟล์ต้นทางหายไปแล้ว (เช่น _raw ถูกโหลดทับ)
        if gone:
            print(f"  ! {sheet}: ข้าม idx {sorted(gone)} (ไม่มีไฟล์แล้ว)")
            ids -= gone
        if mode == "keep_only":
            for i, fn in fns.items():
                dec.setdefault(fn, ("keep", "") if i in ids else ("reject", reason))
        elif mode == "reject_only":
            for i, fn in fns.items():
                dec.setdefault(fn, ("reject", reason) if i in ids else ("keep", ""))
        elif mode.startswith("relabel:"):
            assert mode.split(":", 1)[1] in CLASS_PROMPTS, mode
            for i in ids:
                dec[fns[i]] = (mode, "wrong_class")   # relabel ชนะ keep/reject
        else:
            raise ValueError(mode)
    out = pd.DataFrame([(fn, d, r) for fn, (d, r) in dec.items()], columns=["filename", "decision", "reason"])
    out.to_csv(MANUAL, index=False)
    print(out["decision"].value_counts().to_string())


def apply():
    df = pd.read_csv(META)
    df["search_class"] = df["class"]          # คลาสของ keyword ที่ค้นมา (ใช้ทำ group split)
    sc = pd.read_csv(SCORES).set_index("filename")
    dec = sc["bucket"].map({"auto_reject": "reject"}).fillna("keep")
    reason = sc["top_junk"].where(dec == "reject", "")
    if MANUAL.exists():   # คัดมือ override ทุกอย่าง
        man = pd.read_csv(MANUAL).drop_duplicates("filename", keep="last").set_index("filename")
        dec.update(man["decision"])
        reason.update(man["reason"].fillna(""))
    df["decision"] = df["filename"].map(dec)
    df["reject_reason"] = df["filename"].map(reason).fillna("")

    relab = df["decision"].str.startswith("relabel:")
    df.loc[relab, "class"] = df.loc[relab, "decision"].str.split(":").str[1]
    df.loc[relab, "decision"] = "keep"
    new_rel = df["class"] + "/" + df["filename"]

    # หาไฟล์จริงไม่ว่าจะอยู่ที่ไหน (รันซ้ำได้): โฟลเดอร์คลาสไหนก็ได้ หรือ _rejected
    rej_dir = BASE / "_rejected"
    rej_dir.mkdir(exist_ok=True)
    where = {p.name: p for d in [*[BASE / c for c in CLASS_PROMPTS], rej_dir] if d.exists() for p in d.iterdir()}
    for fn, d, rel in zip(df["filename"], df["decision"], new_rel):
        dst = BASE / rel if d == "keep" else rej_dir / fn
        src = where.get(fn)
        if src is not None and src != dst:
            shutil.move(str(src), str(dst))
    df["relpath"] = new_rel

    df[df["decision"] == "reject"].to_csv(rej_dir / "rejected.csv", index=False)
    clean = df[df["decision"] == "keep"].drop(columns=["decision", "reject_reason"])
    clean.to_csv(BASE / "metadata_clean.csv", index=False)

    print(f"relabel: {int(relab.sum())} รูป")
    print(pd.crosstab(df.loc[relab, "search_class"], df.loc[relab, "class"]).to_string())
    print("\nหลังคัด: จำนวนต่อคลาส")
    print(clean["class"].value_counts().sort_index().to_string())
    ct = pd.crosstab(clean["class"], clean["lang"])
    print("\nlang ต่อคลาส")
    print(ct.assign(th_pct=(ct["th"] / ct.sum(axis=1) * 100).round(1)).to_string())
    ct = pd.crosstab(clean["class"], clean["engine"])
    print("\nengine ต่อคลาส (baidu = รูป stock มีลายน้ำเยอะ ต้องใกล้กันทุกคลาส)")
    print(ct.assign(baidu_pct=(ct["baidu"] / ct.sum(axis=1) * 100).round(1)).to_string())
    print("\nจำนวนรูปต่อ group (search_class, keyword_id) แยกตามคลาสจริง — ใช้ทำ group split")
    print(clean.groupby(["class", "search_class", "keyword_id"]).size().unstack(fill_value=0).to_string())
    n = clean["class"].value_counts()
    print(f"\nImbalance ratio: {n.max() / n.min():.2f}  | รวม {len(clean)} รูป")


if __name__ == "__main__":
    {"score": score, "sheets": sheets, "decide": decide, "apply": apply}[sys.argv[1]]()
