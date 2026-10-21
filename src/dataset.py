"""眼底图像数据集：MuReD 20 类多标签。

与单标签版本的差异：
    标签是 20 维 0/1 向量（一张图可同时具有多个标签）；
    id 使用 CSV 的 path 列（文件名），与 sample_submission 对齐。
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import LABELS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def crop_black_border(img: Image.Image, tol: int = 7) -> Image.Image:
    """裁掉眼底图四周的黑色背景，只保留圆形视场。"""
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def apply_clahe(img: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """在 LAB 空间的亮度通道上做限制对比度自适应直方图均衡。"""
    arr = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def preprocess(img: Image.Image, size: int, use_clahe: bool = True) -> Image.Image:
    """确定性预处理：裁剪黑边 -> CLAHE -> 缩放。"""
    img = crop_black_border(img)
    if use_clahe:
        img = apply_clahe(img)
    return img.resize((size, size), Image.BILINEAR)


def build_transforms(size: int, train: bool):
    """训练集做增强；验证/测试集只做张量化与归一化。"""
    if train:
        return transforms.Compose([
            # 随机缩放裁剪：让模型不能依赖「病灶固定在画面某处」这一捷径
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.11)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FundusMultiLabelDataset(Dataset):
    """返回 (图像张量, 20 维标签向量, 文件名 path)。"""

    def __init__(self, csv_path, image_dir, size=300, train=False,
                 use_clahe=True, labels=None):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.labels = labels or LABELS
        self.image_dir = image_dir
        self.size = size
        self.use_clahe = use_clahe
        self.tf = build_transforms(size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, str(row["path"]))

        img = Image.open(path).convert("RGB")
        img = preprocess(img, self.size, self.use_clahe)
        tensor = self.tf(img)

        if all(c in self.df.columns for c in self.labels):
            y = torch.tensor([float(row[c]) for c in self.labels], dtype=torch.float32)
        else:
            # 测试集只有 path 列、没有标签列：返回占位零向量
            y = torch.zeros(len(self.labels), dtype=torch.float32)
        return tensor, y, str(row["path"])
