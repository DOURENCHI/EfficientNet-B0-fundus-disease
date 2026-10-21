"""Gradio 演示界面：上传一张眼底图 -> 输出各标签概率 + Top 命中标签 + Grad-CAM。

运行：
    python app/app.py
然后在浏览器打开提示的地址（默认 http://127.0.0.1:7860）。
"""
import os
import sys

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                          # noqa: E402
import torch                                # noqa: E402
import yaml                                 # noqa: E402
from PIL import Image                       # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import build_transforms, preprocess       # noqa: E402
from src.gradcam import GradCAM, overlay_heatmap           # noqa: E402
from src.model import build_model                           # noqa: E402
from src.utils import LABEL_FULL, LABELS, get_device        # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "config.yaml")


def load_everything():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = get_device()
    ckpt_path = os.path.join(cfg["output"]["ckpt_dir"], "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"没有找到权重 {ckpt_path}，请先运行 python src/train.py")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    gradcam = GradCAM(model, model.backbone.conv_head)
    transform = build_transforms(cfg["data"]["image_size"], train=False)
    return cfg, model, gradcam, transform, device


CFG, MODEL, GRADCAM, TRANSFORM, DEVICE = load_everything()


def probability_plot(probs):
    """20 个标签的概率条形图，按概率降序取前 10 个展示。"""
    order = np.argsort(probs)[::-1][:10][::-1]
    names = ["%s\n(%s)" % (l, LABEL_FULL[l]) for l in order]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = ["#1A4E8A" if probs[i] >= 0.5 else "#AEC6DE" for i in order]
    ax.barh(names, probs[order] * 100, color=colors)
    for y, i in enumerate(order):
        ax.text(probs[i] * 100 + 1.2, y, "%.1f%%" % (probs[i] * 100), va="center", fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Probability (%)")
    ax.axvline(50, linestyle="--", linewidth=1, color="#D85A30")
    fig.tight_layout()
    out = os.path.join(CFG["output"]["fig_dir"], "_gradio_prob.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def predict(image):
    if image is None:
        return "-", None, None
    pil = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")
    prepared = preprocess(pil, CFG["data"]["image_size"], CFG["data"]["clahe"])
    tensor = TRANSFORM(prepared)

    with torch.no_grad():
        logits = MODEL(tensor.unsqueeze(0).to(DEVICE))
        probs = torch.sigmoid(logits.float())[0].cpu().numpy()

    hit = [(LABELS[i], probs[i]) for i in np.argsort(probs)[::-1] if probs[i] >= 0.5]
    if hit:
        label = "命中标签：" + "、".join("%s (%.1f%%)" % (l, p * 100) for l, p in hit)
    else:
        top = int(np.argmax(probs))
        label = "无标签超过 0.5；最高为 %s（%.1f%%）" % (LABELS[top], probs[top] * 100)

    cam = GRADCAM.generate(tensor.unsqueeze(0).to(DEVICE), int(np.argmax(probs)))
    rgb = np.array(prepared).astype(np.float32) / 255.0
    overlay = overlay_heatmap(rgb, cam)

    return label, probability_plot(probs), overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传眼底图像"),
    outputs=[
        gr.Textbox(label="预测结果"),
        gr.Image(type="filepath", label="各标签概率（Top 10）"),
        gr.Image(type="numpy", label="Grad-CAM 关注区域"),
    ],
    title="眼底多标签疾病识别演示（MuReD 20 类）",
    description=("上传一张彩色眼底照片，模型会给出 20 个标签各自的概率、"
                 "概率最高的标签列表，以及 Grad-CAM 热力图。"
                 "本页面仅用于课程实验展示，不构成任何医学诊断依据。"),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
