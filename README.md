# ChaNet: Classifying Thai Tea Drinks with Fine-tuned CNNs

DADS7202 Deep Learning (1/2569), GSAS NIDA. Midterm programming homework.

**Group:** Hey Tea

| Name | Student ID |
|---|---|
| กฤตณัฐ ทับทิมแก้ว | 6810422006 |
| พิมกนิษฐ์ ทองศรีแก้ว | 6810422011 |
| สมฤดี การภักดี | 6810422023 |

In this project we collected our own image dataset of five Thai tea drinks and used it to fine-tune four CNN backbones pretrained on
ImageNet: VGG-16, ResNet-50, EfficientNet-B3 and MobileNet-V3-Large. Every model was trained five times with different random seeds
on the same train/validation/test split. Unless stated otherwise, all numbers in this report are measured on the held-out test set and
reported as mean ± SD over the five seeds.

In short:

| | |
|---|---|
| Best models | ResNet-50 (macro F1 0.891 ± 0.017) and EfficientNet-B3 (0.883 ± 0.017). The difference between them is not significant (Welch p = 0.48). |
| Worst model | VGG-16 (macro F1 0.795 ± 0.023), significantly lower than the other three (p ≤ 0.011, \|Hedges' g\| ≥ 1.96). |
| Most common error | Drinks with similar colours: orange fruit teas predicted as Thai tea, dark fruit teas predicted as black tea. |
| Hyperparameter tuning | 78 trials with Optuna and W&B Sweep. Re-training with the tuned values did not change any model significantly (§6.7). |

---

## Contents
1. [Dataset](#1-dataset)
2. [Data preparation](#2-data-preparation)
3. [Model architectures](#3-model-architectures)
4. [Training method](#4-training-method)
5. [Evaluation metrics](#5-evaluation-metrics)
6. [Experimental results](#6-experimental-results)
7. [Discussion and conclusions](#7-discussion-and-conclusions)
8. [Reproducibility](#8-reproducibility)

---

## 1. Dataset

### 1.1 Classes
We chose five iced tea drinks that are common in Thailand. ImageNet-1k does not have a class for any of them; the closest classes are
generic ones such as *eggnog*, *espresso* or *beer glass*.

| Class | Thai name | Description |
|---|---|---|
| `cha_thai` | ชาไทย / ชาเย็น | Thai iced milk tea, bright orange |
| `cha_dam_yen` | ชาดำเย็น | Thai iced black tea without milk, clear dark brown-orange |
| `cha_khiao_nom` | ชาเขียวนม | Iced green milk tea or matcha latte |
| `cha_nom_khai_muk` | ชานมไข่มุก | Bubble milk tea with tapioca pearls |
| `cha_phonlamai` | ชาผลไม้ | Iced fruit tea (peach, lychee, passion fruit, mixed fruit) |

![samples](report/samples.jpg)
*Five random images per class. The image source is shown in brackets.*

### 1.2 Data sources
Most images were scraped from image search engines. We later added photos from food-delivery apps and Facebook shop pages, which we
collected by hand.

| Source | How we collected it | Images in the final set |
|---|---|---|
| Bing image search | `ddgs` (Bing backend), 8 keywords per class (6 Thai, 2 English), up to 100 images per keyword | 669 |
| Baidu image search | `icrawler`, English keywords only, because Thai keywords returned unrelated images | 132 |
| Delivery apps and Facebook (Grab, LINE MAN, shop pages) | collected manually, one image per shop per class | 112 |

We used the same keyword templates, the same Thai/English mix and the same search engines for every class. The reason is that if one
class came mostly from English stock photos and another from Thai street-stall photos, a model could learn the photo style instead of
the drink. The full keyword list is in [`chanet_scraper.py`](chanet_scraper.py).

### 1.3 Cleaning
| Step | Images |
|---|---|
| Downloaded in two scraping rounds | 4,373 |
| Near-duplicates removed with perceptual hashing (pHash distance ≤ 6, also across classes) | − 1,192 → 3,181 |
| Every image checked by eye on numbered contact sheets, using CLIP only to sort them first ([`chanet_curate.py`](chanet_curate.py)) | 878 kept, 73 moved to the correct class |
| Second review by the team | − 77 → 801 |
| Delivery-app and Facebook photos added (checked against existing images with pHash and dHash) | + 112 → **913** |

We kept real photos in which the drink of that class is the main subject and is served in a glass or cup. We removed drawings and 3D
renders, logos, menus or posters where the text is more prominent than the drink, hot tea in ceramic cups, packaged drinks and
collages. We also removed drinks whose class is ambiguous, for example Thai or green tea with pearls, lemon tea, iced black coffee and
obvious AI-generated images. CLIP was only used to order the images for review. We did not let it remove images automatically,
because then the dataset would only keep the images that CLIP already finds easy.

### 1.4 Exploratory data analysis

![counts](report/eda_counts.png)

| | dam_yen | khiao_nom | nom_khai_muk | phonlamai | thai |
|---|---|---|---|---|---|
| Images | **74** | 214 | 232 | 170 | 223 |
| Median width × height (px) | 980 × 1088 | 1024 × 1400 | 1117 × 1133 | 1024 × 1059 | 1001 × 1024 |
| Median aspect ratio (w/h) | 1.00 | 0.75 | 0.90 | 1.00 | 1.00 |
| Mean brightness (0–255) | 137 | 156 | 151 | 151 | 145 |

![image stats](report/eda_image_stats.png)

All 913 images are in colour (no grayscale images) and the shorter side of every image is at least 350 px. Image size and brightness
are similar across classes. `cha_khiao_nom` has more portrait photos (median aspect ratio 0.75), but this difference disappears once
the images are resized and cropped to 224 × 224.

### 1.5 Class imbalance
The dataset is imbalanced. `cha_dam_yen` has only 74 images, while `cha_nom_khai_muk` has 232, a ratio of about 3.1 : 1. Iced black tea
is rarely photographed on its own, and most search results for it turned out to be Thai milk tea, lemon tea or iced coffee. We added
21 hand-collected photos, but decided not to fill the class with low-quality images just to balance the numbers.

We handled the imbalance in three ways. During training we used class-weighted cross-entropy, with weights inversely proportional to
class frequency (about 2.5× for `cha_dam_yen`). For evaluation we use macro F1 as the main metric, so that every class counts equally,
and we also report per-class precision and recall. Finally, we compared four imbalance strategies in a separate experiment
([§6.6](#66-class-imbalance-strategies)). None of them was significantly better than the others; class weighting mainly raised the
recall of `cha_dam_yen` at the cost of its precision.

---

## 2. Data preparation

### 2.1 Train / validation / test split
A purely random split would leak information between the sets. Images returned by the same search query often come from the same shop
or the same series of stock photos, and if they end up in both train and test, the test score becomes too optimistic. We therefore
split the data by group:

| Image source | Group |
|---|---|
| Scraped images | search class + keyword + search engine (for example `cha_thai | kw03 | bing`) |
| Hand-collected images | shop (one image is one group, unless several photos came from the same shop) |

The 160 groups were assigned to train, validation and test so that each class and each image source is split close to 70 / 10 / 20.
The largest deviation is 2.2 % for a class and 3.9 % for a source. No group appears in more than one set. The split is saved in
[`tea_dataset/split.csv`](tea_dataset/split.csv) and every model and seed uses the same file ([`training/make_split.py`](training/make_split.py)).

| | dam_yen | khiao_nom | nom_khai_muk | phonlamai | thai | total |
|---|---|---|---|---|---|---|
| train | 51 | 152 | 160 | 120 | 155 | **638** |
| val | 9 | 23 | 24 | 15 | 22 | **93** |
| test | 14 | 39 | 48 | 35 | 46 | **182** |

To check that this was enough, we trained a pilot ResNet-50 (one seed) on both the group split and an ordinary stratified random
split. The group split gave test accuracy 0.918 and macro F1 0.909, and the random split gave 0.902 and 0.904. Since the random split
was not higher, we found no sign of leakage from the scraped groups. An accuracy around 0.9, rather than 0.99, also looks plausible
for this task.

### 2.2 Preprocessing
1. Every image is decoded as RGB (palette and RGBA images are converted).
2. Training images go through online augmentation (§2.3), which ends with a random crop to 224 × 224. Validation and test images are
   resized so that the shorter side is 256 px and then centre-cropped to 224 × 224.
3. `ToTensor` scales pixel values from 0–255 to 0–1, and the images are normalised with the ImageNet mean and standard deviation.
4. We use a batch size of 32. The training set is reshuffled at every epoch; validation and test keep a fixed order.

### 2.3 Data augmentation
Augmentation is applied online to the training set only:
`RandomResizedCrop(224, scale 0.8–1.0)`, `RandomHorizontalFlip(0.5)`, `RandomRotation(15°)`,
`ColorJitter(brightness 0.3, contrast 0.3, saturation 0.3, hue 0.1)` and `RandomGrayscale(0.1)`.

![augmentation](report/augmentation.png)

---

## 3. Model architectures

### 3.1 Backbones
We picked four backbones from different families so that the comparison covers clearly different designs.

| Model | Parameters | Reason for choosing it |
|---|---|---|
| VGG-16 | 134.3 M | A plain stack of 3×3 convolutions without skip connections; a classic baseline |
| ResNet-50 | 23.5 M | Residual connections and bottleneck blocks; a common strong backbone |
| EfficientNet-B3 | 10.7 M | Compound scaling of depth, width and resolution, with MBConv and squeeze-and-excitation blocks |
| MobileNet-V3-Large | 4.2 M | Depthwise-separable convolutions and NAS-designed blocks; built for mobile devices |

All four use torchvision ImageNet weights (`VGG16_IMAGENET1K_V1`, `ResNet50_IMAGENET1K_V2`, `EfficientNet_B3_IMAGENET1K_V1`,
`MobileNet_V3_Large_IMAGENET1K_V2`).

### 3.2 Pretrained models before fine-tuning
Before changing or training anything, we ran the four original ImageNet models on the same five test images, one per class:

![baseline all models](report/baseline_all_models.png)

Because ImageNet has no tea-drink classes, the models fall back on containers and other drinks: *cocktail shaker*, *pitcher*,
*water jug*, *beaker*, *pop bottle*, *strainer*, *ice cream*, and most often *eggnog*, which is predicted for green milk tea and bubble tea
alike. None of these labels separates our five drinks, which is why fine-tuning is needed. More examples from ResNet-50 are in
[`report/baseline_imagenet.png`](report/baseline_imagenet.png).

### 3.3 Removed layers and new classification head
We only replaced the last 1000-class layer of each model and kept everything before it with its pretrained weights.

| Model | Layer removed | New head |
|---|---|---|
| VGG-16 | `classifier[6]`: Linear(4096 → 1000) | Dropout(0.5) → Linear(4096 → 5). The two pretrained FC-4096 layers (with ReLU and Dropout 0.5) are kept. |
| ResNet-50 | `fc`: Linear(2048 → 1000) | Dropout(0.3) → Linear(2048 → 5), after global average pooling |
| EfficientNet-B3 | `classifier[1]`: Linear(1536 → 1000) | Dropout(0.3) → Linear(1536 → 5). The original Dropout(0.3) in front of it is kept. |
| MobileNet-V3-L | `classifier[3]`: Linear(1280 → 1000) | Dropout(0.3) → Linear(1280 → 5). Linear(960 → 1280), Hardswish and Dropout are kept. |

---

## 4. Training method

### 4.1 Two-stage fine-tuning
Training is done in two stages. In the first stage the backbone is frozen and only the new head is trained. In the second stage we
unfreeze the top part of the backbone and fine-tune it with a smaller learning rate.

| | Stage 1: head only | Stage 2: top layers unfrozen |
|---|---|---|
| Trainable layers | classifier head only | VGG: `features[24:]` (last conv block) + classifier. ResNet: `layer4` + `fc`. EfficientNet and MobileNet: last 3 feature blocks + classifier |
| Trainable parameters | VGG 119.6 M, ResNet 10 k, EfficientNet 7.7 k, MobileNet 1.24 M | VGG 126.6 M, ResNet 15.0 M, EfficientNet 8.5 M, MobileNet 3.0 M |
| Optimizer | Adam, lr 1e-3 | AdamW, lr 1e-4 (MobileNet 5e-5) |
| Maximum epochs | 10 (MobileNet 5) | 15 (MobileNet 20) |
| LR scheduler | cosine annealing within the stage | cosine annealing within the stage |

The loss is class-weighted cross-entropy, with label smoothing 0.05 for VGG-16 and none for the other models. Images are 224 × 224 and
the batch size is 32. We use early stopping with a patience of 5 epochs on the validation weighted F1, keep the best validation
checkpoint of each stage, and start stage 2 from the best stage-1 weights.

All models use the same five seeds: 11, 22, 33, 44 and 55. The seed changes the initialisation of the head, dropout, augmentation and
batch order, while the data split stays the same. Training ran on Kaggle with an NVIDIA Tesla T4 GPU.

The hyperparameters above are the defaults used for the main comparison in §6.1–6.6. We later tuned them with Optuna and W&B Sweep
(§4.3) and re-trained all four models with the tuned values (§6.7).

### 4.2 Learning curves
![learning curves](report/learning_curves_all.png)
*Curves taken from the Kaggle log of the final run ([`results_kaggle/logs/final_v2.log`](results_kaggle/logs/final_v2.log)). The bold
lines show the median seed of each model by test macro F1 (VGG-16 seed 11, ResNet-50 seed 44, EfficientNet-B3 seed 11, MobileNet-V3
seed 11). The faint lines are the validation curves of the other four seeds. The dashed line marks the start of stage 2.*

| At the last epoch (mean ± SD over 5 seeds) | train F1 | val F1 | train − val F1 | epochs run (min–max) |
|---|---|---|---|---|
| VGG-16 | 0.958 ± 0.011 | 0.902 ± 0.021 | 0.056 | 16–18 |
| ResNet-50 | 0.982 ± 0.007 | 0.933 ± 0.018 | 0.049 | 18–25 |
| EfficientNet-B3 | 0.929 ± 0.010 | 0.915 ± 0.017 | 0.014 | 18–25 |
| MobileNet-V3-L | 0.934 ± 0.016 | 0.895 ± 0.009 | 0.039 | 11–15 |

With the backbone frozen, the models underfit. For ResNet-50 and EfficientNet-B3 the validation F1 stays around 0.72–0.80 during stage 1.
Most of the improvement (about +0.15 to +0.20 validation F1) comes right after the top blocks are unfrozen in stage 2, which supports
the two-stage approach.

We do not see strong overfitting. At the end of training the gap between train and validation F1 is between 0.01 and 0.06.
ResNet-50 fits the training set most closely (train F1 0.98), but its validation loss keeps decreasing and then levels off rather
than going back up. VGG-16 has by far the highest validation loss (0.51, compared with 0.21–0.28 for the other models) and the most
unstable validation curve, which is consistent with its lower test score. Early stopping ended most runs before the maximum number of
epochs; VGG-16, for example, stopped after 16–18 of its 25 epochs.

In the first epochs the training F1 is sometimes lower than the validation F1. This is expected, because training metrics are computed
on augmented images with dropout switched on, while the validation images are clean centre crops.

### 4.3 Hyperparameter tuning with Optuna and W&B Sweep
We tuned each architecture separately with both tools, using the same search space and the same fixed seed (42). Each trial was scored
by its best validation weighted F1, and the test set was not used at any point. The code is in
[`run_tune_optuna.py`](training/run_tune_optuna.py), [`run_sweep.py`](training/run_sweep.py) and [`tuning.py`](training/tuning.py),
and every trial is listed in [`trials.csv`](results_kaggle/tuning/trials.csv).

| Hyperparameter | Values searched (108 combinations per model) |
|---|---|
| stage-1 learning rate | 5e-4, 1e-3 |
| stage-1 epochs | 5, 10 |
| stage-2 learning rate | 1e-5, 5e-5, 1e-4 |
| stage-2 epochs | 10, 15, 20 |
| label smoothing | 0, 0.05, 0.1 |

| | Optuna | W&B Sweep |
|---|---|---|
| Search method | TPE sampler (4 random start-up trials) with median pruning during stage 2 | Bayesian optimisation |
| Budget per model | 10 trials | 10 runs |
| Trials run, all models | 38 (6 pruned early; repeated suggestions reused rather than retrained) | 40 |
| GPU time on Kaggle T4 | 2.6 h | 2.9 h |
| Best val F1 found (VGG-16 / ResNet-50 / EfficientNet-B3 / MobileNet-V3) | 0.936 / 0.957 / 0.948 / 0.914 | 0.957 / 0.957 / 0.949 / 0.915 |

![tuning convergence](report/tuning_convergence.png)

For each model we kept the configuration with the highest validation F1 across both tools
([`best_hparams.json`](results_kaggle/tuning/best_hparams.json)):

| Model | stage-1 lr | stage-1 epochs | stage-2 lr | stage-2 epochs | label smoothing | found by |
|---|---|---|---|---|---|---|
| VGG-16 | 1e-3 | 5 | 1e-4 | 20 | 0.1 | W&B Sweep |
| ResNet-50 | 5e-4 | 10 | 1e-4 | 15 | 0 | Optuna (tied with Sweep) |
| EfficientNet-B3 | 5e-4 | 10 | 1e-4 | 20 | 0.1 | W&B Sweep |
| MobileNet-V3-L | 5e-4 | 10 | 5e-5 | 10 | 0.05 | W&B Sweep |

In the end the two tools found almost the same best values. They differ by at most 0.002, except for VGG-16 where W&B Sweep found 0.957
and Optuna 0.936. Note that with only 93 validation images, a difference of 0.01 F1 is roughly one image, and several configurations
tie for first place (three ResNet-50 trials reached 0.957). Optuna used less GPU time because pruning stopped poor runs early, but its
first trials, which are random, were weaker, so it needed more trials to catch up on ResNet-50 and EfficientNet-B3. Both searches
settled on a stage-2 learning rate of 1e-4 for the three larger models, and a small amount of label smoothing helped VGG-16 and
EfficientNet-B3.

Below are screenshots of two W&B Sweep dashboards. The project belongs to the NIDA organisation account, which does not allow public
projects, so we include screenshots instead of a link. Each dashboard shows the best validation F1 of every run over time, W&B's
parameter importance (random-forest importance and correlation with `best_val_f1`) and a parallel-coordinates plot.

| ResNet-50 sweep | VGG-16 sweep |
|---|---|
| ![W&B sweep ResNet-50](report/wandb_sweep_resnet50.png) | ![W&B sweep VGG-16](report/wandb_sweep_vgg16.png) |

For ResNet-50, the stage-2 learning rate is clearly the most important parameter and is positively correlated with the result: the
higher value (1e-4) works better, and the runs with 1e-5 are the low points at about 0.77–0.82. For VGG-16, the number of stage-1
epochs matters most and is negatively correlated, so a short head-only stage (5 epochs) followed by a longer stage 2 works better; label
smoothing is positively correlated. Both observations agree with the values that were selected. Since each sweep only has 10 runs, we
treat these importances as a rough indication rather than a firm conclusion.

---

## 5. Evaluation metrics
We report accuracy, precision, recall and F1. Precision, recall and F1 are given both as macro averages (the unweighted mean over the
five classes, which is our main metric because of the imbalance) and as weighted averages (weighted by the number of test images per
class). We also report precision, recall and F1 for each class, and a confusion matrix that pools the five seeds and is normalised by
row, so each row shows the recall of that class.

Every number is the mean ± SD (ddof = 1) over the five seeds. To compare two models we use Welch's t-test, which does not assume equal
variances, together with Hedges' g as the effect size (roughly 0.2 is small, 0.5 medium and 0.8 or more large).

---

## 6. Experimental results
All results in this section are on the test set (182 images), as mean ± SD over five seeds. The raw numbers for every run are in
[`results_kaggle/runs.csv`](results_kaggle/runs.csv).

### 6.1 Model comparison
| Model | Macro F1 | Macro precision | Macro recall | Weighted F1 | Accuracy |
|---|---|---|---|---|---|
| VGG-16 | 0.795 ± 0.023 | 0.825 ± 0.030 | 0.780 ± 0.021 | 0.810 ± 0.018 | 0.811 ± 0.019 |
| ResNet-50 | **0.891 ± 0.017** | **0.883 ± 0.018** | **0.910 ± 0.012** | 0.899 ± 0.014 | 0.898 ± 0.014 |
| EfficientNet-B3 | 0.883 ± 0.017 | 0.878 ± 0.017 | 0.898 ± 0.021 | **0.908 ± 0.011** | **0.906 ± 0.011** |
| MobileNet-V3-L | 0.839 ± 0.016 | 0.853 ± 0.012 | 0.845 ± 0.023 | 0.859 ± 0.016 | 0.855 ± 0.016 |

![macro F1](results_kaggle/comparison_f1_macro.png)
*Each dot is one seed; the horizontal bar is the mean and the vertical bar is ± 1 SD.*

### 6.2 Statistical tests
Pairwise Welch's t-tests, five seeds per model:

| Pair | Δ macro F1 | p (macro F1) | Hedges' g | p (weighted F1) |
|---|---|---|---|---|
| ResNet-50 vs EfficientNet-B3 | +0.008 | 0.485 (not significant) | 0.42 | 0.270 (not significant) |
| ResNet-50 vs MobileNet-V3 | +0.053 | 0.001 | 2.88 | 0.003 |
| EfficientNet-B3 vs MobileNet-V3 | +0.045 | 0.003 | 2.41 | 0.001 |
| MobileNet-V3 vs VGG-16 | +0.044 | 0.011 | 1.96 | 0.002 |
| ResNet-50 vs VGG-16 | +0.096 | < 0.001 | 4.26 | < 0.001 |
| EfficientNet-B3 vs VGG-16 | +0.088 | < 0.001 | 3.87 | < 0.001 |

The full table is in [`results_kaggle/ttest.csv`](results_kaggle/ttest.csv).

### 6.3 Per-class results
| F1 | dam_yen | khiao_nom | nom_khai_muk | phonlamai | thai |
|---|---|---|---|---|---|
| VGG-16 | 0.69 ± 0.06 | 0.81 ± 0.04 | 0.88 ± 0.03 | 0.84 ± 0.06 | 0.75 ± 0.03 |
| ResNet-50 | **0.85 ± 0.05** | 0.87 ± 0.03 | **0.96 ± 0.02** | **0.93 ± 0.02** | 0.86 ± 0.02 |
| EfficientNet-B3 | 0.73 ± 0.07 | **0.95 ± 0.01** | 0.95 ± 0.01 | 0.90 ± 0.02 | **0.88 ± 0.01** |
| MobileNet-V3-L | 0.71 ± 0.04 | 0.87 ± 0.03 | 0.93 ± 0.01 | 0.86 ± 0.01 | 0.82 ± 0.03 |

| `cha_dam_yen` | Recall | Precision |
|---|---|---|
| VGG-16 | 0.61 ± 0.06 | 0.80 ± 0.08 |
| ResNet-50 | 1.00 ± 0.00 | 0.74 ± 0.08 |
| EfficientNet-B3 | 0.86 ± 0.10 | 0.63 ± 0.06 |
| MobileNet-V3-L | 0.83 ± 0.08 | 0.63 ± 0.03 |

### 6.4 Confusion matrices
To avoid picking one favourable seed, each matrix pools the predictions of all five seeds and is normalised by row.

![confusion matrices](results_kaggle/analysis/cm_all.png)

Averaged over all models and seeds, the most frequent mistakes per run are green milk tea predicted as Thai tea (3.8 images), fruit tea
as Thai tea (3.5), Thai tea as bubble tea (2.6) and fruit tea as black tea (2.3). See
[`confusion_pairs.csv`](results_kaggle/analysis/confusion_pairs.csv).

### 6.5 Accuracy by image source
| Accuracy | Bing (n = 131) | Baidu (n = 28) | Delivery apps / FB (n = 23) |
|---|---|---|---|
| VGG-16 | 0.824 ± 0.017 | 0.736 ± 0.065 | 0.826 ± 0.043 |
| ResNet-50 | 0.907 ± 0.017 | 0.871 ± 0.032 | 0.878 ± 0.036 |
| EfficientNet-B3 | 0.915 ± 0.013 | 0.850 ± 0.030 | 0.922 ± 0.019 |
| MobileNet-V3-L | 0.855 ± 0.013 | 0.857 ± 0.044 | 0.852 ± 0.024 |

The hand-collected delivery-app photos are classified about as well as the web images, so we do not think the models learned to
recognise the source rather than the drink. Baidu images, which are mostly watermarked stock photos, have the lowest accuracy, although
there are only 28 of them in the test set.

### 6.6 Class-imbalance strategies
In this experiment we trained ResNet-50 with the same split, seeds and default hyperparameters, and changed only how the imbalance is
handled ([`training/run_imbalance.py`](training/run_imbalance.py); raw results in [`imbalance_runs.csv`](results_kaggle/imbalance_runs.csv)).

| Strategy | Macro F1 | Weighted F1 | Accuracy | `cha_dam_yen` recall | `cha_dam_yen` precision | `cha_dam_yen` F1 |
|---|---|---|---|---|---|---|
| No compensation (plain cross-entropy) | **0.902 ± 0.012** | **0.906 ± 0.012** | **0.906 ± 0.012** | 0.96 ± 0.06 | 0.83 ± 0.04 | **0.89 ± 0.05** |
| Class-weighted cross-entropy (used in §6.1) | 0.891 ± 0.017 | 0.899 ± 0.014 | 0.898 ± 0.014 | **1.00 ± 0.00** | 0.74 ± 0.08 | 0.85 ± 0.05 |
| WeightedRandomSampler | 0.889 ± 0.009 | 0.894 ± 0.014 | 0.893 ± 0.013 | 0.99 ± 0.03 | 0.77 ± 0.04 | 0.86 ± 0.03 |
| Focal loss (γ = 2) | 0.881 ± 0.020 | 0.883 ± 0.025 | 0.882 ± 0.024 | 0.91 ± 0.06 | **0.85 ± 0.04** | 0.88 ± 0.02 |

![imbalance macro F1](results_kaggle/imbalance_f1_macro_resnet50.png)

None of the strategies is significantly better than another in macro F1 (all six pairs have p ≥ 0.09). Training without any
compensation gave the highest mean, but its lead over class weighting is small (p = 0.27, g = 0.68). What the strategies really change
is the decision boundary of `cha_dam_yen`. Class weighting and oversampling push its recall up to about 1.00, but they also let other
dark drinks into the class, so precision drops to 0.74–0.77. Focal loss and plain cross-entropy keep precision at 0.83–0.85 with
slightly lower recall. The only significant difference is in recall between class weighting and focal loss (1.00 vs 0.91, p = 0.03,
g = 1.83).

Our interpretation is that with an imbalance of about 3 : 1, the pretrained features already separate `cha_dam_yen` reasonably well, so
re-weighting is not necessary and mostly costs precision. It would probably matter more with a stronger imbalance. As a check, the
class-weighted results reproduce the ResNet-50 runs of §6.1 exactly, which shows that our runs are deterministic.

We still use class-weighted cross-entropy for the main comparison, because that choice was made before we saw any test results.
Switching to the best strategy now would mean selecting a training setting based on the test set.

### 6.7 Default vs tuned hyperparameters
We re-trained all four models with the tuned values from §4.3, keeping the same split and seeds
([`results_kaggle/tuned/`](results_kaggle/tuned/)).

| Model | Macro F1, default | Macro F1, tuned | Δ | Welch p | Hedges' g | Weighted F1, tuned |
|---|---|---|---|---|---|---|
| VGG-16 | 0.795 ± 0.023 | **0.818 ± 0.027** | +0.023 | 0.20 | 0.81 | 0.832 ± 0.028 |
| ResNet-50 | **0.891 ± 0.017** | 0.883 ± 0.016 | −0.008 | 0.46 | −0.45 | 0.891 ± 0.014 |
| EfficientNet-B3 | **0.883 ± 0.017** | 0.879 ± 0.013 | −0.005 | 0.66 | −0.26 | 0.903 ± 0.010 |
| MobileNet-V3-L | 0.839 ± 0.016 | **0.842 ± 0.021** | +0.003 | 0.79 | 0.16 | 0.863 ± 0.020 |

![default vs tuned](report/default_vs_tuned.png)

Tuning did not change any model significantly (all p ≥ 0.20). Only VGG-16 improved noticeably (+0.023, g = 0.81). It was the weakest
model and the one furthest from its best settings; its tuned version uses a shorter stage 1, a longer stage 2 and label smoothing 0.1.

We see two reasons why tuning helped so little. First, the configurations were chosen from a single seed and 93 validation images, and
many of them were within one image of each other, so part of the apparent improvement during tuning was noise. Second, the default
settings were already reasonable for two-stage fine-tuning.

The ranking of the models did not change: ResNet-50 and EfficientNet-B3 are still tied (p = 0.64) and ahead of MobileNet-V3 and VGG-16.
The only difference is that after tuning, MobileNet-V3 and VGG-16 are no longer significantly different (p = 0.16). Because tuning did
not change any of our conclusions, we keep the default runs as the main results in §6.1–6.6 and in the discussion, and present the
tuned runs as a robustness check.

---

## 7. Discussion and conclusions

### 7.1 Best and worst architecture
ResNet-50 and EfficientNet-B3 performed best, and statistically we cannot separate them (macro F1 0.891 vs 0.883, p = 0.48, small
effect g = 0.42). ResNet-50 is slightly ahead on macro F1, mostly because it never misses a `cha_dam_yen` image. EfficientNet-B3 is
slightly ahead on weighted F1 and accuracy because it does better on the larger classes, especially `cha_khiao_nom` (F1 0.95), and it
reaches this with 2.2 times fewer parameters. MobileNet-V3-Large is about five points lower, which is still reasonable for a model with
only 4.2 M parameters.

For our use case, we would choose ResNet-50 if the goal is to recognise every drink equally well, since it has the best macro F1 and the
best `cha_dam_yen` F1 (0.85). EfficientNet-B3 is a good alternative when model size matters, and MobileNet-V3-Large would make sense for
something like a menu app running on a phone, at the cost of about five F1 points.

VGG-16 is clearly the weakest model (p ≤ 0.011 and |g| ≥ 1.96 against every other model), even though it has the most parameters (134 M).
Overfitting alone does not explain this: its train-validation gap (0.056) is close to that of ResNet-50 (0.049, §4.2). The difference is
that VGG-16 has a much higher validation loss (0.51 compared with 0.21–0.28) and a noisier validation curve, meaning its predictions are
less confident and less stable. We think this is because VGG-16 has neither skip connections nor batch normalisation, and because stage 2
fine-tunes about 127 M parameters, mostly in the two FC-4096 layers, on only 638 images. VGG-16 also has the largest spread between seeds,
with one seed reaching only 0.754.

### 7.2 Unexpected results
The clearest anomaly is that `cha_dam_yen` is predicted too often. ResNet-50 reaches a recall of 1.00 for this class but a precision of
only 0.74, and EfficientNet-B3 and MobileNet-V3 have a precision of 0.63. The class weight of about 2.5× on a class with only 51 training
images pushes borderline dark drinks, such as dark fruit teas or green teas in dark cups, into `cha_dam_yen`. VGG-16 behaves the
opposite way (recall 0.61) and tends to confuse black tea with Thai tea. The experiment in §6.6 supports this explanation: without class
weights, the precision of `cha_dam_yen` for ResNet-50 rises from 0.74 to 0.83 and macro F1 does not drop (0.902 vs 0.891, not significant).

A second pattern is that `cha_thai` collects most of the errors from other classes, because orange appears not only in Thai tea but also
in peach and passion-fruit teas.

### 7.3 GradCAM: correct vs misclassified predictions
We applied GradCAM to the last convolutional block of the best seed of the two strongest models (ResNet-50 seed 33 and EfficientNet-B3
seed 22). We generated it for every misclassified test image (15 per model) and for two correctly classified images per class. We ran it
locally from the checkpoints saved on Kaggle, and the predictions matched the Kaggle results exactly (accuracy 0.918 for both). The code
is in [`training/error_analysis.py`](training/error_analysis.py).

Correct predictions, ResNet-50 (two per class):
![ResNet-50 correct](results_kaggle/gradcam_full/resnet50_seed33_correct.jpg)

All misclassified test images, ResNet-50 (each title shows the true class, the predicted class and its confidence; most confident first):
![ResNet-50 wrong](results_kaggle/gradcam_full/resnet50_seed33_wrong.jpg)

All misclassified test images, EfficientNet-B3:
![EfficientNet-B3 wrong](results_kaggle/gradcam_full/efficientnet_b3_seed22_wrong.jpg)

When the prediction is correct, the models mostly look at the drink itself: the middle of the cup for Thai tea and green milk tea, the
layer of pearls at the bottom of bubble tea, and the fruit pieces inside fruit tea. These are the same cues a person would use.

The most common mistake, fruit tea predicted as black tea or Thai tea, is explained well by the heatmaps. On the dark iced fruit teas,
the model looks at the clear brown liquid and not at the orange slice on the rim, so it sees "dark clear tea" and predicts `cha_dam_yen`.
On the passion-fruit tea it focuses on the orange-yellow middle layer and ignores the seeds and the fruit next to the cup, so it predicts
`cha_thai`. In other words, colour outweighs the fruit.

Logos and printed cups cause another group of errors. The Thai tea in a ChaTraMue cup (predicted as fruit tea) and the green tea with a
large red shop logo (predicted as Thai tea) are wrong because the heatmap sits on the red-orange logo. For the printed paper cup there is
no liquid visible at all, so the model focuses on the print. On Shutterstock images, both models give part of their attention to the
watermark or the table and tend to predict a lighter class, for example Thai tea as green milk tea. The bubble tea photo with hand-drawn
hearts and Thai text is wrong in almost every run; the heatmap is on the milky body of the drink, while the pearls are small and partly
covered by the drawing.

The two models do not always make the same mistakes. EfficientNet-B3 classifies the ChaTraMue cup correctly by looking below the logo,
but it misreads a dark black tea in a stock photo as Thai tea, which ResNet-50 gets right. This fits with the two models being tied
overall while differing on individual images.

### 7.4 Eyeball comparison: ImageNet baseline, VGG-16 and ResNet-50
![eyeball](report/eyeball.png)
*The same test images for all three models. A red title means VGG-16 is wrong. Best seed of each model (VGG-16 seed 22, ResNet-50 seed 33).*

The original ImageNet model only gives object labels such as *beaker*, *eggnog* or *broccoli*, which say nothing about the type of drink.
After fine-tuning, both VGG-16 and ResNet-50 handle the easy images. ResNet-50 also gets several images right that VGG-16 misses: a black
tea in a watermarked stock photo, a bright green tea next to coffee beans, and Thai tea in a branded cup.

### 7.5 Error analysis
Sixteen test images are misclassified in at least half of the 20 runs (all models and seeds). They are listed in
[`hard_images.csv`](results_kaggle/analysis/hard_images.csv):

![hard images](results_kaggle/analysis/hard_images.jpg)

Looking at these images, we found four recurring patterns:
1. Colours that overlap between classes. Passion-fruit and orange teas are predicted as Thai tea, and a dark iced tea with an orange
   slice is predicted as black tea in all 20 runs. In this case even the label is debatable, since it could be seen as either a fruit
   tea or a black tea with a garnish.
2. The drink is hidden, for example in opaque paper cups or behind large shop logos and stickers.
3. Overlays such as text, hand-drawn hearts or Shutterstock watermarks.
4. The key feature is not visible, such as bubble tea whose pearls are hidden at the bottom, or Thai tea so pale that it looks pink or beige.

Overall, the models rely mainly on colour. Our augmentation only weakens the colour cue slightly (`ColorJitter` with hue 0.1 and
`RandomGrayscale` with p = 0.1), so texture cues such as pearls and fruit pieces are learned less reliably. We did not relabel or remove
any of these test images after seeing the results, since that would mean adjusting the test set to the model.

### 7.6 A closer look at `cha_dam_yen`
Because `cha_dam_yen` is the smallest and most problematic class, we collected every test image that was wrongly predicted as
`cha_dam_yen` (false positives, red) and every `cha_dam_yen` image that was predicted as something else (false negatives, blue) in at
least 3 of the 20 runs ([`cha_dam_yen_errors.csv`](results_kaggle/analysis/cha_dam_yen_errors.csv)):

![cha_dam_yen errors](results_kaggle/analysis/cha_dam_yen_errors.jpg)

| Average per run | VGG-16 | ResNet-50 | EfficientNet-B3 | MobileNet-V3-L |
|---|---|---|---|---|
| False positives (other class → dam_yen) | 2.2 | 5.2 | 7.0 | 6.8 |
| False negatives (dam_yen → other class) | 5.4 | 0.0 | 2.0 | 2.4 |

Of the 106 false-positive predictions, 46 come from fruit tea, 29 from green milk tea, 26 from Thai tea and 5 from bubble tea. The images
that appear again and again are dark iced fruit teas (clear brown-red tea with an orange or peach slice, wrong in up to 19 of 20 runs),
Thai tea photographed in dim light so that it looks dark orange-brown, and drinks in opaque or printed cups. What they have in common is
a dark, clear-looking liquid, which seems to be the main cue the models use for `cha_dam_yen`.

The false negatives are black teas in which the liquid is hard to see: drinks in branded cups (ICE CUP, MICHA), a watermarked stock photo,
a wide bowl photographed from above, and a lighter, orange-tinted black tea that looks like Thai tea or fruit tea.

The models balance these two kinds of error differently. ResNet-50 never misses a black tea but accepts about five wrong ones per run,
while VGG-16 does the opposite. This is consistent with §6.6, where removing the class weight reduced ResNet-50's false positives
(precision 0.74 → 0.83).

To improve this class, we would collect more black-tea photos in branded or opaque cups, and more dark fruit teas labelled as fruit tea,
so that the model has to look for fruit pieces instead of relying on the colour of the liquid. A stricter definition of fruit tea, where
the fruit has to be inside the drink and not only used as a garnish, would also remove most of the ambiguous cases.

### 7.7 Limitations
- `cha_dam_yen` has only 14 test images, so one image changes its recall by about 7 %, and its per-class numbers are noisy.
- The standard deviations only reflect randomness in training. All five seeds use the same split, so variation due to the choice of
  data is not included.
- Hyperparameters were chosen from single-seed runs on 93 validation images, which is a noisy basis. Averaging over several seeds or
  using cross-validation would be more reliable but would need three to five times more GPU time.
- Web images lean towards styled marketing and stock photos. Real photos from street stalls are a minority (112 hand-collected images).

### 7.8 Conclusions
With fewer than 1,000 curated images, fine-tuned ImageNet CNNs can separate five visually similar tea drinks with a macro F1 of about 0.89.
The choice of architecture matters: ResNet-50 and EfficientNet-B3 outperform VGG-16 by about nine F1 points with large and significant
effects, while the two of them are statistically tied. Tuning with Optuna and W&B Sweep confirmed this ranking but did not significantly
improve any model. Most of the remaining errors come from drinks whose colours overlap between classes and from photos in which the drink
is partly hidden. We think the most useful next steps are clearer class definitions (for example, that fruit tea must show fruit in the
drink) and more images of `cha_dam_yen`.

---

## 8. Reproducibility

```
chanet_scraper.py        # scraping (Bing via ddgs, Baidu via icrawler) and pHash de-duplication -> tea_dataset/metadata.csv
chanet_curate.py         # CLIP sorting, contact sheets, manual decisions, sync/ingest -> metadata_clean.csv
tea_dataset/
  metadata_clean.csv     # the 913 curated images (the images themselves are not in git because of copyright)
  split.csv              # fixed group split
training/
  make_split.py          # builds split.csv
  config.py dataset.py models.py trainer.py losses.py utils.py
  pilot.py               # ResNet-50 on group vs random split (leakage check)
  run_final.py           # 4 architectures x 5 seeds -> runs.csv, preds/, plots, t-tests
  run_imbalance.py       # imbalance-strategy experiment
  run_tune_optuna.py     # Optuna tuning (TPE + median pruning), per architecture
  run_sweep.py           # W&B Sweep tuning (Bayesian), per architecture
  tuning.py              # shared search space, trial runner, selection of best hyperparameters
  analyze_preds.py       # confusion matrices, error pairs, accuracy by source, hard images
  error_analysis.py      # GradCAM for all misclassified images from a checkpoint
  report_figures.py      # EDA, baseline, eyeball, learning-curve and tuning figures in report/
notebooks/03_kaggle_train.ipynb   # runs the experiments on a Kaggle GPU
results_kaggle/          # results used in this report (default runs, imbalance experiment, logs)
  tuning/                # all tuning trials and the selected hyperparameters
  tuned/                 # 4 architectures x 5 seeds re-trained with the tuned hyperparameters
  gradcam_full/          # GradCAM for every misclassified test image
report/                  # figures used in this README
```

To reproduce the experiments, upload the 913 images as a Kaggle dataset and run `notebooks/03_kaggle_train.ipynb` on a T4 GPU. Locally,
`python training/run_final.py` trains the models, and `python training/analyze_preds.py --results <dir>` and
`python training/report_figures.py` produce the analysis and figures. Internal team notes are in [`docs/TEAM_NOTES.md`](docs/TEAM_NOTES.md).
