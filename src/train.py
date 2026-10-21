"""训练脚本：多标签 BCE + 正类权重 + 混合精度 + 按验证宏平均 AUC 早停。

与单标签版本的差异：
    损失为 BCEWithLogitsLoss（每标签独立），正类权重取「负/正样本比」
    并设上限，缓解 20 个标签间最大 16 倍的正例频率差异；
    验证指标为宏平均 AUC（20 个标签的 one-vs-rest AUC 取平均）。

关键约定：
    1) 只使用课程固定的 train.csv / val.csv 划分，不重新划分；
    2) 增强只作用于训练集；
    3) 验证集只用于监控收敛与选择权重，不参与反向传播。

运行：
    python src/train.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import FundusMultiLabelDataset            # noqa: E402
from src.model import build_model                           # noqa: E402
from src.utils import (compute_metrics, compute_pos_weight,  # noqa: E402
                       get_device, set_seed)


def run_epoch(model, loader, criterion, device, use_amp=False,
              amp_dtype=torch.float16, optimizer=None, scaler=None):
    """跑一轮。传入 optimizer 即为训练模式，否则为评估模式。"""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, probs, labels = 0.0, [], []
    desc = "train" if train_mode else "eval"

    for imgs, y, _ in tqdm(loader, leave=False, desc=desc):
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        probs.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    return total_loss / len(loader.dataset), labels, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    d, t, out = cfg["data"], cfg["train"], cfg["output"]
    print(f"使用设备：{device}")

    for key in ("ckpt_dir", "log_dir", "fig_dir"):
        os.makedirs(out[key], exist_ok=True)

    # ---------------- 数据（课程固定划分，直接使用） ----------------
    train_ds = FundusMultiLabelDataset(d["train_csv"], d["image_dir"],
                                       d["image_size"], train=True, use_clahe=d["clahe"])
    val_ds = FundusMultiLabelDataset(d["val_csv"], d["image_dir"],
                                     d["image_size"], train=False, use_clahe=d["clahe"])
    print(f"训练集 {len(train_ds)} 张，验证集 {len(val_ds)} 张（课程固定划分）")

    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=t["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            num_workers=t["num_workers"], pin_memory=True)

    # ---------------- 模型与损失 ----------------
    model = build_model(cfg).to(device)

    if t["pos_weight"]:
        tr_labels = train_ds.df[[c for c in train_ds.df.columns if c != "path"]].values
        w = compute_pos_weight(tr_labels, cap=t["pos_weight_cap"]).to(device)
        print("BCE 正类权重（截断到 %.0f）：" % t["pos_weight_cap"],
              np.round(w.detach().cpu().numpy(), 2))
    else:
        w = None
    criterion = nn.BCEWithLogitsLoss(weight=None, pos_weight=w)

    # 分层学习率：骨干用较小的 lr 保护预训练特征不被小数据破坏，新分类头用完整 lr
    lr = t["lr"]
    bb_scale = float(t.get("backbone_lr_scale", 1.0))
    bb_params, head_params = [], []
    for name, p in model.named_parameters():
        (bb_params if name.startswith("backbone.") else head_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": bb_params, "lr": lr * bb_scale},
         {"params": head_params, "lr": lr}],
        weight_decay=t["weight_decay"])
    print("分层学习率：骨干 %.2e ／ 分类头 %.2e"
          % (lr * bb_scale, lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])

    use_amp = bool(t["amp"]) and device.type == "cuda"
    # 优先 bf16：动态范围与 fp32 相同，不会因 BN 层权重过大而溢出；
    # （本项目换用 ra4 强配方权重后，BN 的 γ 最大达 69，fp16 会在第一步就产生 inf 梯度）
    if use_amp and torch.cuda.is_bf16_supported() and str(t.get("amp_dtype", "bf16")).lower() == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    # GradScaler 只对 fp16 有意义；bf16 不需要 loss scaling
    scaler_enabled = use_amp and amp_dtype == torch.float16
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    print("混合精度：%s" % (str(amp_dtype).replace("torch.", "") if use_amp else "关闭"))

    # ---------------- 训练循环 ----------------
    best_auc, best_epoch, patience = -1.0, -1, 0
    ckpt_path = os.path.join(out["ckpt_dir"], "best.pth")
    history = []

    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_loader, criterion, device,
                                  use_amp, amp_dtype, optimizer, scaler)
        va_loss, y_true, y_prob = run_epoch(model, val_loader, criterion, device,
                                            use_amp, amp_dtype)
        metrics, _ = compute_metrics(y_true, y_prob)
        scheduler.step()

        print(f"[{epoch:02d}/{t['epochs']:02d}] train_loss={tr_loss:.4f} "
              f"val_loss={va_loss:.4f} val_macro_auc={metrics['macro_auc']:.4f} "
              f"micro_auc={metrics['micro_auc']:.4f} micro_f1={metrics['micro_f1']:.2f} "
              f"({time.time() - t0:.1f}s)")

        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_loss": va_loss, **metrics})

        if metrics["macro_auc"] > best_auc:
            best_auc, best_epoch, patience = metrics["macro_auc"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "metrics": metrics, "config": cfg}, ckpt_path)
            print(f"    -> 保存最佳权重（验证宏平均 AUC = {best_auc:.4f}）")
        else:
            patience += 1
            if patience >= t["early_stop_patience"]:
                print(f"验证宏平均 AUC 连续 {patience} 轮未提升，提前停止训练。")
                break

    pd.DataFrame(history).to_csv(os.path.join(out["log_dir"], "train_log.csv"), index=False)
    with open(os.path.join(out["log_dir"], "train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_macro_auc": best_auc,
                   "history": history}, f, ensure_ascii=False, indent=2)

    print(f"\n训练结束：最佳轮次 {best_epoch}，验证宏平均 AUC {best_auc:.4f}")
    print(f"权重文件：{ckpt_path}")
    print("下一步：python src/evaluate.py")


if __name__ == "__main__":
    main()
