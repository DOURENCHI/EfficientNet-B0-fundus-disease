"""Grad-CAM：多标签任务中可视化某个特定标签的激活区域。

与单标签版本的差异：
    反向传播的目标从「预测类别分数」改为「某一个标签的 logit」，
    其余计算流程不变。

运行：
    python src/gradcam.py                       # 对若干代表标签各生成 1 张
    python src/gradcam.py --labels DR,MH,CNV    # 指定标签
输出：
    outputs/figures/gradcam_labels.png
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402
import yaml                                           # noqa: E402
from torch.utils.data import DataLoader               # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, FundusMultiLabelDataset  # noqa: E402
from src.model import build_model                                      # noqa: E402
from src.utils import LABELS, get_device                               # noqa: E402

DEFAULT_LABELS = ["DR", "NORMAL", "ODC", "MYA", "CNV"]


class GradCAM:
    """最小实现的 Grad-CAM，只依赖 PyTorch。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, label_idx):
        """x 形状 (1, 3, H, W)；label_idx 为要解释的标签下标。"""
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, label_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def denormalize(tensor):
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = tensor.detach().cpu().numpy() * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def overlay_heatmap(rgb_img, cam, alpha=0.45):
    base = (rgb_img * 255).astype(np.uint8)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (base * (1 - alpha) + heat * alpha).astype(np.uint8)


def find_positive_sample(dataset, label_idx, max_scan=400):
    """在数据集里找到该标签的一个正例，返回其下标。"""
    for i in range(min(len(dataset), max_scan)):
        _, y, _ = dataset[i]
        if y[label_idx] == 1:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                    help="要可视化的标签缩写，逗号分隔")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    d, out = cfg["data"], cfg["output"]
    os.makedirs(out["fig_dir"], exist_ok=True)

    ckpt_path = args.ckpt or os.path.join(out["ckpt_dir"], "best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"已加载权重：{ckpt_path}")

    target_layer = model.backbone.conv_head
    gradcam = GradCAM(model, target_layer)

    ds = FundusMultiLabelDataset(d["val_csv"], d["image_dir"],
                                 d["image_size"], train=False, use_clahe=d["clahe"])

    want = [s.strip() for s in args.labels.split(",") if s.strip()]
    picked = []
    for name in want:
        if name not in LABELS:
            print(f"  未知标签 {name}，跳过")
            continue
        idx = LABELS.index(name)
        i = find_positive_sample(ds, idx)
        if i is None:
            print(f"  标签 {name} 没有找到正例，跳过")
            continue
        picked.append((name, i))

    if not picked:
        print("没有可用的样本。")
        return

    n = len(picked)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (name, i) in enumerate(picked):
        tensor, y, path = ds[i]
        rgb = denormalize(tensor)
        cam = gradcam.generate(tensor.unsqueeze(0).to(device), LABELS.index(name))
        overlaid = overlay_heatmap(rgb, cam)

        pos_labels = [LABELS[k] for k in range(len(LABELS)) if y[k] == 1]
        axes[0, col].imshow(rgb)
        axes[0, col].set_title("%s\n%s" % (name, "+".join(pos_labels)[:18]), fontsize=8.5)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlaid)
        axes[1, col].set_title(str(path)[:10], fontsize=7.5)
        axes[1, col].axis("off")

    fig.suptitle("Grad-CAM for selected labels (top: original, bottom: heatmap)", fontsize=11)
    fig.tight_layout()
    save_path = os.path.join(out["fig_dir"], "gradcam_labels.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Grad-CAM 对照图 -> {save_path}")


if __name__ == "__main__":
    main()
