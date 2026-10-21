# 代码笔记

关键函数、参数、数据结构与原理。

## 1. 模型（`src/model.py`）

### `MultiLabelNet`

```python
self.backbone = timm.create_model(
    backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
)
feat_dim = self.backbone.num_features        # efficientnet_b0 → 1280

self.head = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(feat_dim, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, num_classes),             # 20
)
```

三个细节：

| 细节 | 作用 |
|---|---|
| `num_classes=0` | 让 timm 不加原生的 ImageNet 分类头，只留骨干 |
| `global_pool="avg"` | 全局平均池化，输出 `(B, 1280)` 而不是特征图 |
| **返回 raw logits** | 配合 `BCEWithLogitsLoss`；推理时再 `sigmoid` |

参数量约 **4.34 M**（EfficientNet-B0 4.0 M + 头部 0.33 M）。

## 2. 预处理（`src/dataset.py`）

### `crop_black_border(img, tol=7)`

```python
gray = cv2.cvtColor(arr, RGB2GRAY)
mask = gray > tol                 # 认为"非黑"的像素
rows = np.where(mask.any(axis=1))[0]
cols = np.where(mask.any(axis=0))[0]
return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
```

`tol=7` 而不是 0：JPEG 压缩会在纯黑区域留下 1~5 的噪声值，
用 0 会框到整张图，裁边等于没做。

`mask.sum() == 0` 时直接返回原图——避免全黑图导致 `rows/cols` 为空、索引越界。

### `apply_clahe(img, clip_limit=2.0)`

```python
lab = cv2.cvtColor(arr, RGB2LAB)
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])     # 只动 L 通道
```

**只动 L（亮度）通道**是关键：眼底病变（出血、渗出、微血管瘤）靠**颜色**区分，
在 RGB 三个通道上分别做 CLAHE 会改变色相，破坏这些线索。

`clipLimit=2.0` 限制对比度放大倍数，防止噪声被一起放大；
`tileGridSize=8×8` 把图分成 64 块各做各的直方图均衡。

### `build_transforms(size, train)`

| 增强 | 参数 | 为什么合理 |
|---|---|---|
| `RandomResizedCrop` | `scale=(0.8, 1.0)`、`ratio=(0.9, 1.11)` | **打破"病灶固定在画面某处"的捷径** |
| `RandomHorizontalFlip` | — | 眼底图左右无固定方向 |
| `RandomVerticalFlip` | — | 同上 |
| `RandomRotation` | 15° | 拍摄角度有轻微差异 |
| `ColorJitter` | 0.3 / 0.3 / 0.3 / 0.03 | 模拟不同相机与光照；hue 只给 0.03，避免破坏病变颜色 |

验证/测试集**只做 `ToTensor + Normalize`**，不做任何几何变换。

### `FundusMultiLabelDataset`

返回三元组 `(tensor, y, path)`：

- `y` 是 20 维 `float32`
- 测试集 CSV 只有 `path` 列时，`y` 返回**全零占位**（判断依据是"标签列是否都在 CSV 里"）
- `path` 回传是为了写提交文件时能对上文件名

## 3. 指标与权重（`src/utils.py`）

### `compute_pos_weight(y, cap=20.0)`

```python
pos = y.sum(axis=0)
neg = y.shape[0] - pos
w = neg / np.maximum(pos, 1.0)
w = np.minimum(w, cap)
```

`np.maximum(pos, 1.0)` 防除零（某标签在训练集里一个正例都没有时）。

**`cap=20` 是必须的**：最稀有标签（CRS 21 张）的原始权重约 `(1545-21)/21 ≈ 72`，
不加限制会让梯度被这一个标签主导，其他 19 个标签学不动。

### `compute_metrics(y_true, y_prob, threshold=0.5)`

```python
for c in range(20):
    col = y_true[:, c]
    if col.sum() == 0 or col.sum() == N:
        per_label_auc.append(float("nan"))     # 无定义
    else:
        per_label_auc.append(roc_auc_score(col, y_prob[:, c]))

macro_auc = float(np.nanmean(per_label_auc))
```

**为什么记 NaN 而不是 0**：某标签在验证集里全 0 时，AUC 数学上无定义。
记 0 会让它变成"最差标签"，把宏平均拉低——这是**错误归因**，不是模型真的差。
`np.nanmean` 会自动跳过这些项。

### `per_label_table`

手算 tp/fp/fn 而不是调 `classification_report`，因为需要**同时输出正例数**——
读者必须看到"这一类的正例只有 3 个"，才能判断 F1 = 0 是模型问题还是样本问题。

## 4. Grad-CAM（`src/gradcam.py`）

### 核心：两个 hook + 一次反传

```python
class GradCAM:
    def __init__(self, model, target_layer):
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
```

| hook | 存什么 |
|---|---|
| forward | `self.activations = output.detach()` —— 该层输出的特征图 `(1, C, h, w)` |
| backward | `self.gradients = grad_output[0].detach()` —— 该层的梯度 |

生成过程：

```python
logits = self.model(x)
logits[0, label_idx].backward()                    # ① 只对某个标签反传

weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # ② 空间维度求平均 → (1,C,1,1)
cam = (weights * self.activations).sum(dim=1, keepdim=True)  # ③ 加权求和
cam = F.relu(cam)                                  # ④ 只保留正贡献
cam = F.interpolate(cam, size=x.shape[2:], ...)    # ⑤ 上采样回原尺寸
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # ⑥ 归一化
```

**每一步的理由**：

| 步 | 理由 |
|---|---|
| ① 只反传一个标签 | 多标签下没有"预测类别"，必须先指定要解释谁 |
| ② 对空间求平均 | 得到"每个通道对这类有多重要"的权重 |
| ③ 加权求和 | 把 C 个通道压成 1 张热力图 |
| ④ **ReLU** | 只保留**促进**该标签的特征；抑制它的（负值）不显示 |
| ⑤ 上采样 | 特征图分辨率低（如 10×10），要放大到 300×300 |
| ⑥ min-max | 让热力图在不同样本间可比 |

`+1e-8` 防止全零时除零。

### `target_layer = model.backbone.conv_head`

选最后一层卷积而不是更靠前的层：

| 层 | 热力图特点 |
|---|---|
| 浅层 | 分辨率高但语义弱，高亮的是边缘与纹理 |
| **`conv_head`** | 语义最强、定位较准，是 Grad-CAM 的常规选择 |

### `overlay_heatmap(rgb, cam, alpha=0.45)`

```python
heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)        # 必须转，cv2 默认 BGR
return (base * (1 - alpha) + heat * alpha)
```

`alpha=0.45` 是经验值：太低看不清热区，太高遮住解剖结构。

## 5. 参数速查

### 数据

| 项 | 值 |
|---|---|
| 数据集 | MuReD 多标签眼底（20 类） |
| 图像 | 2208 张 300×300 RGB |
| 划分 | train 1545 / val 221 / test 442（课程固定） |
| 标签顺序 | DR, NORMAL, MH, ODC, TSLN, ARMD, DN, MYA, BRVO, ODP, CRVO, CNV, RS, ODE, LS, CSR, HTR, ASR, CRS, OTHER |
| 多标签比例 | 约 22.7% 训练图有 ≥2 个标签 |
| 正例频率 | 22.4%（DR/NORMAL）~ 1.4%（CRS），最大差约 16 倍 |

### 训练

| 参数 | 值 | 说明 |
|---|---|---|
| `backbone` | `efficientnet_b0.ra_in1k` | ImageNet 预训练 |
| `image_size` | 300 | 与数据集一致 |
| `batch_size` | 16 | |
| `epochs` | 20 | 实际第 16 轮早停 |
| `lr` | 1e-4 | |
| `weight_decay` | 0.01 | |
| `amp_dtype` | **bf16** | fp16 会在 BN 权重极端时溢出 |
| `early_stop_patience` | 5 | 按验证宏 AUC |
| `pos_weight_cap` | 20.0 | |
| `dropout` | 0.3 | |
| TTA | 水平翻转平均 | 宏 AUC +0.008 |

### 结果（221 张验证集）

| 指标 | 数值 |
|---|---|
| **宏平均 AUC（主指标）** | **0.9268** |
| 微平均 AUC | 0.9494 |
| 微平均 F1（0.5 阈值） | 54.64% |
| 宏平均 F1（0.5 阈值） | 50.94% |
| Hamming 正确率 | 91.70% |

最佳第 11 轮，第 16 轮早停；单轮约 80 秒，峰值显存 1.33 GiB，总训练 21 分 13 秒。

## 6. 原理补充

### 为什么多标签不能用 softmax

softmax 的输出**强制和为 1**，意味着"20 个标签互斥"。
但临床上糖网患者常常合并黄斑水肿，两个标签必须能同时为真。
sigmoid 对每个标签独立判断，才是正确的建模方式。

### 为什么主指标选宏平均 AUC 而不是 F1

- **AUC 与阈值无关**：正例只有 21 张的标签，在 0.5 阈值下 F1 几乎必然很低（模型偏保守），但这不代表它"分不开"。AUC 衡量的是**排序能力**，更公平。
- **宏平均对稀有类公平**：微平均会被高频标签主导（DR/NORMAL 各 346 张，CRS 只有 21 张），宏平均让每个标签权重相同。

### 为什么「AUC 高、F1 低」是稀有标签的常态

正例只有 3~6 张的标签，模型即使**排序完全正确**（AUC 接近 1），
在固定 0.5 阈值下也很可能一个都不报——因为训练时正负样本差太悬殊，
模型学到的偏置就是"偏向预测负类"。

这类问题的解法是**逐标签阈值搜索**，而不是继续调模型。

### 为什么 TTA 只做水平翻转

TTA 的收益来自"多次扰动取平均降低方差"，但每次扰动必须是**模型训练时见过的合理变换**。
训练时的几何增强里，水平翻转是确定的（眼底图左右无固定方向），
而旋转、裁剪会引入插值误差与边界效应，收益不稳定。所以只用最简单的一种。
