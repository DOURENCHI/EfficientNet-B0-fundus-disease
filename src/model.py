"""模型定义：EfficientNet-B0 骨干 + 多标签分类头。

与单标签版本的差异：
    输出 20 个独立 logit（每标签一个），配合 BCEWithLogitsLoss 使用；
    推理时经 sigmoid 得到每个标签的概率。
"""
import timm
import torch.nn as nn


class MultiLabelNet(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", num_classes: int = 20,
                 dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features          # efficientnet_b0 -> 1280

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))             # raw logits


def build_model(cfg) -> MultiLabelNet:
    m, d = cfg["model"], cfg["data"]
    return MultiLabelNet(
        backbone=m["backbone"],
        num_classes=d["num_classes"],
        dropout=m["dropout"],
        pretrained=m["pretrained"],
    )
