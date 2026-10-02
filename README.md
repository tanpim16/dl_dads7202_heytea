# DADS7202 Deep Learning — HeyTea CNN Classifier

**Course:** 1/2569 Deep Learning, GSAS NIDA  
**Task:** Build an image classifier with CNN on a self-collected dataset  
**Topic:** Thai tea drink classification (5 classes)

---

## Project Summary

We classify 5 types of Thai tea drinks that are **not** in ImageNet's 1,000 classes:

| Class | Description |
|---|---|
| `cha_dam_yen` | Iced black tea |
| `cha_khiao_nom` | Green milk tea |
| `cha_nom_khai_muk` | Bubble milk tea |
| `cha_phonlamai` | Fruit tea |
| `cha_thai` | Thai iced tea (orange) |

Images collected via **web scraping** (Bing + Baidu, 2 rounds) and curated with CLIP-score filtering + manual review. Images are shared via Google Drive (not committed to git due to copyright).

---

## Status

### Done
- [x] Web scraper (`chanet_scraper.py`) — 2 rounds, Bing + Baidu
- [x] Curation pipeline (`chanet_curate.py`) — CLIP scoring + manual review
- [x] Metadata (`tea_dataset/metadata_clean.csv`) — 878 images with EDA fields
- [x] Full training pipeline (`training/`)
- [x] EDA notebook (`notebooks/01_eda.ipynb`)
- [x] Pre-trained baseline demo (`notebooks/02_baseline_demo.ipynb`)
- [x] Kaggle training notebook (`notebooks/03_kaggle_train.ipynb`)

### To Do
- [ ] Collect more `cha_dam_yen` photos (currently 82, target 150–200)
- [ ] Ingest new photos: `python training/ingest_new_photos.py <folder> cha_dam_yen`
- [ ] Run EDA notebook → export plots for slides
- [ ] Run baseline demo → screenshot wrong predictions for slides
- [ ] Upload images to Kaggle dataset `heytea-images`
- [ ] Run `03_kaggle_train.ipynb` on Kaggle GPU
  - [ ] Sweep first (`RUN_SWEEP=True`, ~4–6 hr)
  - [ ] Update `BEST_HPARAMS` in config cell
  - [ ] Final 5-seed run (`RUN_SWEEP=False`, ~5 hr)
- [ ] Build presentation slides (7 sections)

---

## Dataset

**Current class counts (metadata_clean.csv):**

| Class | Count | Status |
|---|---|---|
| cha_nom_khai_muk | 221 | OK |
| cha_thai | 220 | OK |
| cha_khiao_nom | 202 | OK |
| cha_phonlamai | 153 | OK |
| **cha_dam_yen** | **82** | **Need more** |

**Imbalance handled by:**
1. `WeightedRandomSampler` — oversamples minority during training
2. `class_weight` in `CrossEntropyLoss` — penalizes minority errors more

**Split:** Stratified 70/10/20 (train/val/test), fixed seed=42

---

## Model Architectures (4 chosen)

| Architecture | Params | Reason |
|---|---|---|
| VGG-16 | ~138M | Simple baseline |
| ResNet-50 | ~25M | Skip connections, strong standard |
| EfficientNet-B3 | ~12M | Best for fine-grained tasks |
| MobileNet-V3-Large | ~5.5M | Lightweight comparison |

Training: **Two-stage fine-tuning** — Stage 1 freeze backbone, Stage 2 unfreeze top layers.

---

## Repo Structure

```
tea_dataset/
  metadata.csv          # raw metadata
  metadata_clean.csv    # curated metadata (images on Drive)

training/
  config.py             # paths, classes, seeds, BEST_HPARAMS
  dataset.py            # TeaDataset, stratified split, WeightedRandomSampler
  models.py             # build_model(), freeze/unfreeze, GradCAM target layers
  trainer.py            # two-stage training loop + W&B logging
  evaluate.py           # metrics, mean±SD, Welch's t-test, plots
  gradcam.py            # GradCAM + correct/wrong visualization
  utils.py              # set_seed(), get_device()
  run_sweep.py          # W&B Bayesian sweep (run first)
  run_final.py          # 5-seed final evaluation (run after sweep)
  ingest_new_photos.py  # add new photos to dataset + update metadata
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
- [ ] Train/val/test split method (stratified), ratios, per-class counts after split
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
