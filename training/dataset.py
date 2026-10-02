import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import (
    DATA_DIR, METADATA, CLASSES, CLASS2IDX,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD, SPLIT_FILE,
)
import config


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TeaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(DATA_DIR / row["relpath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, CLASS2IDX[row["class"]]


def make_splits():
    """
    Group split 70/10/20 (train/val/test) อ่านจาก tea_dataset/split.csv (สร้างด้วย make_split.py)
    group = shop_id (รูปที่หาเอง) หรือ keyword+engine (รูป scrape) -> รูปกลุ่มเดียวกันไม่ข้าม split
    ใช้ split เดียวกันทุก arch / ทุก seed
    """
    df = pd.read_csv(METADATA)
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)

    split_file = SPLIT_FILE
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} not found -> run: python make_split.py")
    sp = pd.read_csv(split_file)[["filename", "group", "split"]]
    df = df.merge(sp, on="filename", how="left")
    if df["split"].isna().any():
        raise RuntimeError(f"{int(df['split'].isna().sum())} images in metadata_clean.csv have no split "
                           "(metadata changed after split) -> run: python make_split.py")

    return tuple(df[df["split"] == s].drop(columns="split").reset_index(drop=True)
                 for s in ("train", "val", "test"))


def make_loaders(train_df, val_df, test_df, batch_size=BATCH_SIZE):
    train_ds = TeaDataset(train_df, get_transforms("train"))
    val_ds   = TeaDataset(val_df,   get_transforms("val"))
    test_ds  = TeaDataset(test_df,  get_transforms("test"))

    # WeightedRandomSampler balances cha_dam_yen (only ~74 images) — ใช้เมื่อ IMBALANCE เป็น sampler/both
    if config.IMBALANCE in ("sampler", "both"):
        labels  = [CLASS2IDX[c] for c in train_df["class"]]
        counts  = np.bincount(labels, minlength=len(CLASSES)).astype(float)
        w       = 1.0 / counts
        sampler = WeightedRandomSampler(
            [w[l] for l in labels], num_samples=len(labels), replacement=True
        )
        train_kw = {"sampler": sampler}
    else:
        train_kw = {"shuffle": True}

    return (
        DataLoader(train_ds, batch_size=batch_size, **train_kw,
                   num_workers=NUM_WORKERS, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=NUM_WORKERS, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                   num_workers=NUM_WORKERS, pin_memory=True),
    )


def get_class_weights(train_df):
    """CrossEntropyLoss class weights (inverse frequency); ทุกคลาส = 1 ถ้า IMBALANCE ไม่ใช่ class_weight/both."""
    if config.IMBALANCE not in ("class_weight", "both"):
        return torch.ones(len(CLASSES), dtype=torch.float32)
    labels = [CLASS2IDX[c] for c in train_df["class"]]
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    weights = torch.tensor(len(labels) / (len(CLASSES) * counts),
                           dtype=torch.float32)
    return weights
