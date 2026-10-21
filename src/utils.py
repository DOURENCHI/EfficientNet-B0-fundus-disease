"""通用工具：20 类标签定义、随机种子、设备、多标签指标计算。

多标签任务说明：
    每张图像可以同时具有多个标签，标签向量为 20 维 0/1。
    主指标为宏平均 ROC-AUC（对 20 个标签分别计算 one-vs-rest AUC 再取平均），
    与单标签任务中的「宏平均 AUC」同名，但计算对象是标签而非类别。
"""
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

# 20 个标签缩写（与 CSV 列顺序一致）
LABELS = ["DR", "NORMAL", "MH", "ODC", "TSLN", "ARMD", "DN", "MYA", "BRVO",
          "ODP", "CRVO", "CNV", "RS", "ODE", "LS", "CSR", "HTR", "ASR", "CRS", "OTHER"]

# 各标签完整英文名（来源：MuReD 数据集说明，Rodriguez Rivera et al., Mendeley Data）
LABEL_FULL = {
    "DR": "Diabetic Retinopathy", "NORMAL": "Normal Retina", "MH": "Media Haze",
    "ODC": "Optic Disc Cupping", "TSLN": "Tessellation",
    "ARMD": "Age-Related Macular Degeneration", "DN": "Drusen", "MYA": "Myopia",
    "BRVO": "Branch Retinal Vein Occlusion", "ODP": "Optic Disc Pallor",
    "CRVO": "Central Retinal Vein Occlusion", "CNV": "Choroidal Neovascularization",
    "RS": "Retinitis", "ODE": "Optic Disc Edema", "LS": "Laser Scars",
    "CSR": "Central Serous Retinopathy", "HTR": "Hypertensive Retinopathy",
    "ASR": "Arteriosclerotic Retinopathy", "CRS": "Chorioretinitis",
    "OTHER": "Other Diseases",
}


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_pos_weight(y, cap: float = 20.0) -> torch.Tensor:
    """BCE 的正类权重 = 负样本数 / 正样本数，并设置上限。

    极稀有标签（如 CRS 仅 21 张正例）的原始权重会超过 70，
    不加限制会让梯度被个别稀有标签主导。
    """
    y = np.asarray(y)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = neg / np.maximum(pos, 1.0)
    w = np.minimum(w, cap)
    return torch.tensor(w, dtype=torch.float32)


def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    """多标签整体指标。y_true / y_prob 形状均为 (N, 20)。

    返回 (metrics dict, 每标签 AUC 数组)。
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # 每标签 AUC；某标签在当前集合中全 0 或全 1 时无法计算，记为 NaN 并跳过
    per_label_auc = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == y_true.shape[0]:
            per_label_auc.append(float("nan"))
        else:
            per_label_auc.append(roc_auc_score(col, y_prob[:, c]))
    per_label_auc = np.array(per_label_auc)

    pred = (y_prob >= threshold).astype(int)
    metrics = {
        "macro_auc": float(np.nanmean(per_label_auc)),
        "micro_auc": float(roc_auc_score(y_true, y_prob, average="micro")),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0) * 100),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0) * 100),
        "hamming_acc": float((y_true == pred).mean() * 100),
    }
    return metrics, per_label_auc


def per_label_table(y_true, y_prob, threshold: float = 0.5):
    """逐标签指标：正例数、精确率、召回率、F1、AUC。"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)

    rows = []
    for c, name in enumerate(LABELS):
        tp = int(((y_true[:, c] == 1) & (pred[:, c] == 1)).sum())
        fp = int(((y_true[:, c] == 0) & (pred[:, c] == 1)).sum())
        fn = int(((y_true[:, c] == 1) & (pred[:, c] == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == y_true.shape[0]:
            auc = float("nan")
        else:
            auc = roc_auc_score(col, y_prob[:, c])

        rows.append({
            "label": name,
            "full_name": LABEL_FULL[name],
            "pos": int(col.sum()),
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            "f1": f1 * 100.0,
            "auc": auc,
        })
    return rows
