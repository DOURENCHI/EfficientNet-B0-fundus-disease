"""评估脚本：多标签整体指标 + 逐标签指标表 + 每标签 AUC 条形图 + 训练曲线。

多标签任务没有混淆矩阵（一张图可同时属于多个标签），
因此用「逐标签 AUC / F1 条形图」替代单标签任务中的混淆矩阵。

运行：
    python src/evaluate.py                # 默认在验证集上评估
输出：
    outputs/logs/metrics_val.json
    outputs/logs/per_label_val.csv
    outputs/figures/label_auc_val.png
    outputs/figures/training_curve.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import torch                                          # noqa: E402
import yaml                                           # noqa: E402
from torch.utils.data import DataLoader               # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import FundusMultiLabelDataset       # noqa: E402
from src.model import build_model                     # noqa: E402
from src.utils import (LABEL_FULL, LABELS, compute_metrics,  # noqa: E402
                       get_device, per_label_table)


def setup_cjk_font():
    """尽量使用系统中文字体；找不到就退回默认字体。"""
    from matplotlib import font_manager
    for name in ["Microsoft YaHei", "SimHei", "PingFang SC",
                 "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    return False


def plot_label_auc(rows, save_path):
    """逐标签 AUC 横向条形图（按 AUC 升序）。"""
    items = sorted([r for r in rows if not np.isnan(r["auc"])],
                   key=lambda r: r["auc"])
    names = ["%s (%s)" % (r["label"], r["full_name"]) for r in items]
    vals = [r["auc"] for r in items]
    posn = [r["pos"] for r in items]

    fig, ax = plt.subplots(figsize=(10.0, 6.8))
    colors = ["#1A4E8A" if v >= 0.9 else ("#4A8FD4" if v >= 0.8 else "#A9C6E4")
              for v in vals]
    ax.barh(names, vals, color=colors, height=0.72)
    # 右侧额外留出 0.17 的空间，保证 AUC=1.000 的标注也能完整显示
    for i, (v, p) in enumerate(zip(vals, posn)):
        ax.text(v + 0.008, i, "%.3f  (n=%d)" % (v, p),
                va="center", fontsize=8.5, color="#333333", clip_on=False)
    ax.set_xlim(0.5, 1.19)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("One-vs-rest ROC-AUC")
    ax.axvline(0.5, linestyle="--", linewidth=1.0, color="#999999")
    ax.margins(y=0.015)
    fig.tight_layout(pad=0.9)
    # bbox_inches="tight" + pad_inches 保证四周留白，避免 x 轴标签贴边被裁
    fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  逐标签 AUC 图 -> {save_path}")


def plot_training_curve(log_csv, save_path):
    if not os.path.exists(log_csv):
        return
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    axes[0].plot(df["epoch"], df["train_loss"], marker="o", ms=3, label="train loss")
    axes[0].plot(df["epoch"], df["val_loss"], marker="s", ms=3, label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[0].set_title("Loss curves")

    axes[1].plot(df["epoch"], df["macro_auc"], marker="o", ms=3, color="#1A4E8A",
                 label="val macro AUC")
    best_idx = int(df["macro_auc"].idxmax())
    axes[1].scatter(df.loc[best_idx, "epoch"], df.loc[best_idx, "macro_auc"],
                    color="#D85A30", zorder=5,
                    label=f"best epoch {int(df.loc[best_idx, 'epoch'])}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro AUC")
    axes[1].legend()
    axes[1].set_title("Validation macro AUC")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  训练曲线 -> {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="val", choices=["val"],
                    help="test.csv 无标签，只能在验证集上评估")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_cjk_font()
    device = get_device()
    d, out = cfg["data"], cfg["output"]
    for key in ("log_dir", "fig_dir"):
        os.makedirs(out[key], exist_ok=True)

    ckpt_path = args.ckpt or os.path.join(out["ckpt_dir"], "best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"已加载权重：{ckpt_path}（第 {ckpt.get('epoch', '?')} 轮）")

    ds = FundusMultiLabelDataset(d["val_csv"], d["image_dir"],
                                 d["image_size"], train=False, use_clahe=d["clahe"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"])

    probs, labels = [], []
    # 推理：可选水平翻转 TTA（原图 + 翻转图的概率取平均）
    tta = bool(cfg.get("infer", {}).get("tta", False))
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    print(f"推理增强（TTA）：{'开启（水平翻转平均）' if tta else '关闭'}")
    with torch.no_grad():
        for imgs, y, _ in loader:
            imgs = imgs.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
                logits = model(imgs)
                if tta:
                    logits_f = model(torch.flip(imgs, dims=[3]))
            p = torch.sigmoid(logits.float())
            if tta:
                p = (p + torch.sigmoid(logits_f.float())) / 2.0
            probs.append(p.cpu().numpy())
            labels.append(y.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)

    metrics, per_auc = compute_metrics(y_true, y_prob)
    print(f"\n===== {args.split} 集整体指标（n={len(y_true)}，20 标签）=====")
    print(f"宏平均 AUC    {metrics['macro_auc']:.4f}")
    print(f"微平均 AUC    {metrics['micro_auc']:.4f}")
    print(f"微平均 F1     {metrics['micro_f1']:.2f} %")
    print(f"宏平均 F1     {metrics['macro_f1']:.2f} %")
    print(f"Hamming 正确率 {metrics['hamming_acc']:.2f} %")

    with open(os.path.join(out["log_dir"], f"metrics_{args.split}.json"),
              "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "n": int(len(y_true)),
                   "checkpoint": ckpt_path, **metrics}, f,
                  ensure_ascii=False, indent=2)

    rows = per_label_table(y_true, y_prob)
    print("\n===== 逐标签指标（阈值 0.5）=====")
    print(f"{'标签':<8}{'正例':>6}{'精确率':>10}{'召回率':>10}{'F1':>10}{'AUC':>10}")
    for r in rows:
        print(f"{r['label']:<8}{r['pos']:>6}{r['precision']:>10.2f}"
              f"{r['recall']:>10.2f}{r['f1']:>10.2f}{r['auc']:>10.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(out["log_dir"], f"per_label_{args.split}.csv"),
                              index=False, encoding="utf-8-sig")

    plot_label_auc(rows, os.path.join(out["fig_dir"], f"label_auc_{args.split}.png"))
    plot_training_curve(os.path.join(out["log_dir"], "train_log.csv"),
                        os.path.join(out["fig_dir"], "training_curve.png"))

    np.savez(os.path.join(out["log_dir"], f"probs_{args.split}.npz"),
             y_true=y_true, y_prob=y_prob)
    print(f"概率矩阵 -> {os.path.join(out['log_dir'], f'probs_{args.split}.npz')}")
    print(f"\n标签缩写说明见 src/utils.py 的 LABEL_FULL；完整对应关系已写入 CSV。")


if __name__ == "__main__":
    main()
