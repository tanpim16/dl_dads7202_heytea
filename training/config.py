from pathlib import Path

ROOT         = Path(__file__).parent.parent
DATA_DIR     = ROOT / "tea_dataset"
METADATA     = DATA_DIR / "metadata_clean.csv"
SPLIT_FILE   = DATA_DIR / "split.csv"          # สร้างด้วย make_split.py (commit ขึ้น git)
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR  = ROOT / "results"

CLASSES    = ["cha_dam_yen", "cha_khiao_nom", "cha_nom_khai_muk", "cha_phonlamai", "cha_thai"]
NUM_CLASSES = len(CLASSES)
CLASS2IDX  = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS  = {i: c for i, c in enumerate(CLASSES)}

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 4

# วิธีจัดการ class imbalance (เลือกอย่างเดียว ไม่งั้นชดเชยซ้ำ 2 ชั้น):
#   "class_weight" = CrossEntropy weight ตามความถี่ | "sampler" = WeightedRandomSampler
#   "both" = ทั้งคู่ (พฤติกรรมเดิม) | "none"
IMBALANCE = "class_weight"   # pilot (both): dam_yen recall 1.0 แต่ precision 0.74 = ชดเชยเกิน

SPLIT_SEED = 42          # fixed — same data split for every run
SEEDS      = [11, 22, 33, 44, 55]   # model init seeds → gives mean±SD

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ARCHS = ["vgg16", "resnet50", "efficientnet_b3", "mobilenet_v3_large"]

# Fill in after running run_sweep.py — defaults shown below
BEST_HPARAMS = {
    "vgg16": {
        "stage1_lr": 0.001,  "stage1_epochs": 10,
        "stage2_lr": 0.0001, "stage2_epochs": 15,
        "label_smoothing": 0.05,
    },
    "resnet50": {
        "stage1_lr": 0.001,  "stage1_epochs": 10,
        "stage2_lr": 0.0001, "stage2_epochs": 15,
        "label_smoothing": 0.0,
    },
    "efficientnet_b3": {
        "stage1_lr": 0.001,  "stage1_epochs": 10,
        "stage2_lr": 0.0001, "stage2_epochs": 15,
        "label_smoothing": 0.0,
    },
    "mobilenet_v3_large": {
        "stage1_lr": 0.001,  "stage1_epochs": 5,
        "stage2_lr": 0.00005,"stage2_epochs": 20,
        "label_smoothing": 0.0,
    },
}
