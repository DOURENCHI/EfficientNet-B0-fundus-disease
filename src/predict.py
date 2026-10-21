"""推理脚本：对课程测试集（442 张，无标签）推理，生成提交文件 test.csv。

按 sample_submission.csv 的行顺序与列名输出，每格填该标签的预测概率
（数据集 README 允许填「预测标签或预测概率」；填概率可让 AUC 的计算
不依赖具体阈值，因此默认填概率）。

文件命名：课程 PPT 规定的提交文件名为 test.csv，本脚本以此为默认输出名。

运行：
    python src/predict.py                    # 输出 outputs/test.csv
    python src/predict.py --out ../test.csv  # 直接输出到提交目录
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import FundusMultiLabelDataset   # noqa: E402
from src.model import build_model                  # noqa: E402
from src.utils import LABELS, get_device, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default=None, help="输出 CSV 路径")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    d, out_dir = cfg["data"], cfg["output"]
    out_path = args.out or out_dir["pred_csv"]
    ckpt_path = args.ckpt or os.path.join(out_dir["ckpt_dir"], "best.pth")

    # 读取测试集与提交模板，保证行列顺序与模板一致
    test_df = pd.read_csv(d["test_csv"], encoding="utf-8-sig")
    tmpl = pd.read_csv(d["sample_submission"], encoding="utf-8-sig")
    assert list(test_df["path"]) == list(tmpl["path"]), "test.csv 与 sample_submission 顺序不一致"

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"已加载权重：{ckpt_path}（第 {ckpt.get('epoch', '?')} 轮）")
    print(f"待预测图像：{len(test_df)} 张")

    ds = FundusMultiLabelDataset(d["test_csv"], d["image_dir"],
                                 d["image_size"], train=False, use_clahe=d["clahe"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"])

    # 推理：可选水平翻转 TTA（与原图概率取平均）
    tta = bool(cfg.get("infer", {}).get("tta", False))
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    print(f"推理增强（TTA）：{'开启（水平翻转平均）' if tta else '关闭'}")

    probs, paths = [], []
    with torch.no_grad():
        for imgs, _, pths in loader:
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
            paths.extend(pths)

    prob_mat = np.concatenate(probs)

    # 按 sample_submission 的列顺序组装（第 0 列为 path，其余为标签名）
    sub = tmpl.copy()
    for j, lab in enumerate(LABELS):
        sub[lab] = [round(float(v), 4) for v in prob_mat[:, j]]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n提交文件已生成：{out_path}")
    print(f"共 {len(sub)} 行 × {len(sub.columns)} 列（path + 20 个标签概率）")

    # 附带 0/1 阈值版本，便于直接核对
    bin_path = os.path.join(out_dir["log_dir"], "test_binary.csv")
    binv = tmpl.copy()
    for j, lab in enumerate(LABELS):
        binv[lab] = (prob_mat[:, j] >= 0.5).astype(int)
    binv.to_csv(bin_path, index=False, encoding="utf-8-sig")
    print(f"（0/1 阈值版本：{bin_path}）")

    # 预测的阳性数概览
    print("\n各标签预测阳性数（概率>=0.5）：")
    for j, lab in enumerate(LABELS):
        print(f"  {lab:<7}{int((prob_mat[:, j] >= 0.5).sum()):>5}  平均概率 {prob_mat[:, j].mean():.3f}")


if __name__ == "__main__":
    main()
