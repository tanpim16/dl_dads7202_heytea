import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import (
    DATA_DIR, METADATA, CLASSES, CLASS2IDX,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD, SPLIT_SEED,
)


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
    Stratified 70/10/20 split (train/val/test).
    Uses a fixed SPLIT_SEED so every model sees the same test set.
    """
    df = pd.read_csv(METADATA)
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)

    # Step 1: carve out 20 % test
    train_val, test = train_test_split(
        df, test_size=0.20, stratify=df["class"], random_state=SPLIT_SEED
    )
    # Step 2: 12.5 % of remainder = 10 % of total → val
    train, val = train_test_split(
        train_val, test_size=0.125, stratify=train_val["class"],
        random_state=SPLIT_SEED
    )
    return train, val, test


def make_loaders(train_df, val_df, test_df, batch_size=BATCH_SIZE):
    train_ds = TeaDataset(train_df, get_transforms("train"))
    val_ds   = TeaDataset(val_df,   get_transforms("val"))
    test_ds  = TeaDataset(test_df,  get_transforms("test"))

    # WeightedRandomSampler balances cha_dam_yen (only ~82 images)
    labels  = [CLASS2IDX[c] for c in train_df["class"]]
    counts  = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w       = 1.0 / counts
    sampler = WeightedRandomSampler(
        [w[l] for l in labels], num_samples=len(labels), replacement=True
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                   num_workers=NUM_WORKERS, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=NUM_WORKERS, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                   num_workers=NUM_WORKERS, pin_memory=True),
    )


def get_class_weights(train_df):
    """CrossEntropyLoss class weights (inverse frequency)."""
    labels = [CLASS2IDX[c] for c in train_df["class"]]
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    weights = torch.tensor(len(labels) / (len(CLASSES) * counts),
                           dtype=torch.float32)
    return weights
