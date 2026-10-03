# Team notes (internal — ไม่ใช่รายงาน)

รายงานหลักอยู่ที่ [README.md](../README.md)

---

## Status

### Done
- [x] Web scraper (`chanet_scraper.py`) — 2 rounds, Bing + Baidu
- [x] Curation pipeline (`chanet_curate.py`) — CLIP scoring + manual review
- [x] Metadata (`tea_dataset/metadata_clean.csv`) — 913 images with EDA fields (incl. ~110 from Grab/LINE MAN/FB)
- [x] Group split (`training/make_split.py` → `tea_dataset/split.csv`)
- [x] Pilot ResNet-50 (`training/pilot.py`) — test acc 0.918 / macro F1 0.909 (group split), no leakage signal
- [x] Full training pipeline (`training/`)
- [x] EDA notebook (`notebooks/01_eda.ipynb`)
- [x] Pre-trained baseline demo (`notebooks/02_baseline_demo.ipynb`)
- [x] Kaggle training notebook (`notebooks/03_kaggle_train.ipynb`)

### To Do
- [x] ~~Collect more `cha_dam_yen` photos~~ — ตกลงใช้ 74 รูป + class weight และระบุในรายงานว่าคลาสนี้ข้อมูลน้อย
- [ ] Run EDA notebook → export plots for slides
- [ ] Run baseline demo → screenshot wrong predictions for slides
- [x] Upload images to Kaggle dataset `heytea-images`
- [x] Run `03_kaggle_train.ipynb` on Kaggle GPU (final 5-seed) — ผลอยู่ใน `results_kaggle/`
  - macro F1: ResNet-50 0.891±0.017 ≈ EfficientNet-B3 0.883±0.017 (p=0.48) > MobileNet-V3 0.839±0.016 > VGG-16 0.795±0.023
  - error analysis: `results_kaggle/analysis/` (confusion matrix, คู่ที่สับสน, acc แยกแหล่งรูป, รูปยาก 16 รูป)
- [ ] Compare imbalance methods — `training/run_imbalance.py` พร้อมแล้ว (none / class_weight / sampler / focal, ResNet-50 × 5 seeds) รันบน Kaggle รอบที่ 2 (notebook Option C)
- [ ] Error analysis — `training/error_analysis.py` (รันในเครื่องด้วย checkpoint จาก Kaggle)
- [ ] Build presentation slides (7 sections)

---

## Dataset

**Current class counts (metadata_clean.csv):**

| Class | Count | Status |
|---|---|---|
| cha_nom_khai_muk | 232 | OK |
| cha_thai | 223 | OK |
| cha_khiao_nom | 214 | OK |
| cha_phonlamai | 170 | OK |
| **cha_dam_yen** | **74** | **Small — report as limitation** |

**Imbalance:** `config.IMBALANCE = "class_weight"` (CrossEntropy weight) — ใช้อย่างเดียว
(pilot ที่ใช้ทั้ง sampler + class weight ทายชาดำเย็นเกิน: recall 1.0 แต่ precision 0.74)

**Split:** Group split 70/10/20 จาก `tea_dataset/split.csv` (ทุก arch / seed ใช้ไฟล์เดียวกัน)
group = `shop_id` (รูปที่หาเอง) หรือ keyword + engine (รูป scrape) → รูปกลุ่มเดียวกันไม่ข้าม train/test

### เพิ่ม / ลบรูป (อ่านก่อนแก้ dataset)

- **เพิ่มรูป:** ใส่ใน `manual_add/<คลาส>/` (ร้านละ 1 รูป) แล้วรัน `python chanet_curate.py ingest`
- **ลบรูป:** ลบใน `tea_dataset/<คลาส>/` แล้วรัน `python chanet_curate.py sync`
- **ทุกครั้งที่รูปเปลี่ยน ต้องรัน `python training/make_split.py` ใหม่ แล้ว push**
- 🙏 **ไม่ใช้ `training/ingest_new_photos.py` แล้วนะ** เพราะชนกับระบบ sync
  (รูปที่เพิ่มด้วยสคริปต์นี้จะหายจาก metadata) และไม่เช็ครูปซ้ำ
- รูปเปลี่ยนแล้วต้องอัป zip รูปขึ้น Kaggle dataset `heytea-images` ใหม่ด้วย (metadata + split อ่านจาก git อยู่แล้ว)

---

## Repo Structure

```
tea_dataset/
  metadata.csv          # raw metadata
  metadata_clean.csv    # curated metadata (images on Drive / Kaggle)
  split.csv             # group split train/val/test (สร้างด้วย training/make_split.py)

chanet_curate.py        # คัดรูป: score / sheets / decide / apply / sync / ingest
manual_add/<class>/     # รูปที่หาเอง (ไม่อยู่ใน git) -> ingest

training/
  config.py             # paths, classes, seeds, BEST_HPARAMS
  dataset.py            # TeaDataset, group split (split.csv), imbalance handling
  models.py             # build_model(), freeze/unfreeze, GradCAM target layers
  trainer.py            # two-stage training loop + W&B logging
  evaluate.py           # metrics, mean±SD, Welch's t-test, plots
  gradcam.py            # GradCAM + correct/wrong visualization
  utils.py              # set_seed(), get_device()
  run_sweep.py          # W&B Bayesian sweep (run first)
  make_split.py         # สร้าง group split -> tea_dataset/split.csv
  pilot.py              # ResNet-50 seed เดียว: group vs random split (leak check)
  run_final.py          # 5-seed final evaluation (resumable, results/runs.csv, results/preds/)
  run_imbalance.py      # เทียบวิธีจัดการ imbalance: none / class_weight / sampler / focal
  losses.py             # FocalLoss
  error_analysis.py     # คู่คลาสที่สับสน, acc แยกแหล่งรูป, GradCAM รูปที่ทายผิดทั้งหมด
  ingest_new_photos.py  # ⚠️ เลิกใช้ — ใช้ chanet_curate.py ingest แทน
  requirements.txt

notebooks/
  01_eda.ipynb           # EDA plots for slides
  02_baseline_demo.ipynb # pre-trained CNN baseline (CPU, no training)
  03_kaggle_train.ipynb  # full training on Kaggle GPU
```

---

## Teacher Requirements Checklist

### 1. Dataset Description
- [ ] Data sources (web scraping: Bing, Baidu, keywords used)
- [ ] Sample images per class
- [ ] EDA: image count per class, image sizes, aspect ratios, brightness, color vs grayscale
- [ ] **Imbalanced dataset — must state clearly and explain how it is handled** *(mandatory)*

### 2. Data Preparation
- [ ] Preprocessing steps: resize (224×224), normalize (ImageNet mean/std)
- [ ] Train/val/test split method (group split by shop/keyword), ratios, per-class counts after split
- [ ] **Data augmentation — must be done, online or offline both OK** *(mandatory)*
  - RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter

### 3. Model Architecture
- [ ] **3–5 significantly different CNN backbones** (VGG+ResNet = 1, not 2)
- [ ] Brief reason for choosing each architecture
- [ ] Pre-trained CNN demo on our images **before any fine-tuning** — show wrong predictions
- [ ] Which layers were removed from the original backbone
- [ ] New classifier head details: number of layers, connections, parameter values

### 4. Training Method
- [ ] Which layers are frozen/unfrozen per stage
- [ ] Optimizer, loss function, initial LR, LR scheduler
- [ ] Hyperparameter tuning technique (W&B Sweep + Bayesian Optimization)

### 5. Evaluation Metrics
- [ ] Overall: Accuracy, Precision, Recall, F1 (specify: weighted/macro/micro)
- [ ] Per-class: Precision, Recall, F1 for each class
- [ ] Confusion matrix

### 6. Experimental Results
- [ ] Learning curves (train/val loss + metric per epoch) — state which seed is shown
- [ ] **All numbers reported as mean±SD** *(mandatory)* — 3–10 runs per architecture
- [ ] Comparison table across all architectures
- [ ] Welch's t-test p-values for all pairwise architecture comparisons
- [ ] Clearly state all results are on **test set**

### 7. Discussion & Conclusions
- [ ] Which architecture is best/worst and is the difference statistically significant?
- [ ] Any anomalous results and probable causes
- [ ] **GradCAM analysis** *(mandatory)* — show both correct AND misclassified cases
- [ ] Eyeball analysis: pre-trained baseline vs. fine-tuned model A vs. model B
- [ ] Error analysis for underperforming classes — what pattern do misclassified images share?
