"""
Add new photos to the dataset and update metadata_clean.csv.

Usage:
    python ingest_new_photos.py <folder_of_photos> <class_name>

Example (after team collects more cha_dam_yen photos):
    python ingest_new_photos.py ~/Downloads/new_cha_dam_yen cha_dam_yen

Valid class names:
    cha_dam_yen  cha_khiao_nom  cha_nom_khai_muk  cha_phonlamai  cha_thai
"""
import sys
import shutil
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "tea_dataset"
METADATA = DATA_DIR / "metadata_clean.csv"

VALID_CLASSES = {
    "cha_dam_yen", "cha_khiao_nom",
    "cha_nom_khai_muk", "cha_phonlamai", "cha_thai",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


def compute_meta(path: Path) -> dict:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    arr  = np.array(img)
    gray = np.array(img.convert("L"))
    is_grayscale = bool(np.allclose(arr[:,:,0], arr[:,:,1], atol=5) and
                        np.allclose(arr[:,:,1], arr[:,:,2], atol=5))
    return {
        "width":        w,
        "height":       h,
        "aspect_ratio": round(w / h, 4),
        "mode":         img.mode,
        "is_grayscale": is_grayscale,
    }


def main(src_folder: str, class_name: str):
    if class_name not in VALID_CLASSES:
        print(f"ERROR: '{class_name}' is not a valid class.")
        print(f"Valid: {sorted(VALID_CLASSES)}")
        sys.exit(1)

    src  = Path(src_folder)
    dest = DATA_DIR / class_name
    dest.mkdir(parents=True, exist_ok=True)

    photos = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not photos:
        print(f"No images found in {src}")
        sys.exit(1)

    df = pd.read_csv(METADATA)
    existing_files = set(df["filename"].tolist())

    # Find next index for naming
    existing_manual = [f for f in existing_files
                       if f.startswith(class_name) and "manual" in f]
    next_idx = len(existing_manual)

    new_rows = []
    skipped  = 0

    for photo in sorted(photos):
        new_name = f"{class_name}_manual_{next_idx:06d}{photo.suffix.lower()}"

        # Skip if exact filename already exists
        if new_name in existing_files:
            skipped += 1
            continue

        out_path = dest / new_name
        try:
            meta = compute_meta(photo)
        except Exception as e:
            print(f"  SKIP {photo.name}: {e}")
            continue

        shutil.copy2(photo, out_path)

        new_rows.append({
            "filename":     new_name,
            "relpath":      f"{class_name}/{new_name}",
            "class":        class_name,
            "keyword_id":   "kw_manual",
            "round":        3,
            "keyword":      "",
            "lang":         "",
            "engine":       "manual",
            "source":       "photo",
            "shop_id":      "",
            "width":        meta["width"],
            "height":       meta["height"],
            "aspect_ratio": meta["aspect_ratio"],
            "mode":         meta["mode"],
            "is_grayscale": meta["is_grayscale"],
            "xclass_dup":   False,
            "search_class": class_name,
        })
        next_idx += 1

    if not new_rows:
        print(f"No new photos added (all {skipped} skipped as duplicates).")
        return

    new_df  = pd.DataFrame(new_rows)
    updated = pd.concat([df, new_df], ignore_index=True)
    updated.to_csv(METADATA, index=False)

    print(f"\nDone! Added {len(new_rows)} photos to '{class_name}'.")
    if skipped:
        print(f"  Skipped {skipped} duplicates.")

    # Show new class counts
    print("\nUpdated class counts:")
    for cls, cnt in updated.groupby("class").size().items():
        marker = " <-- updated" if cls == class_name else ""
        print(f"  {cls:25s}: {cnt}{marker}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
