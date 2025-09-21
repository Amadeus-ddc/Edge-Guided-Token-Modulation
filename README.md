# Edge-Guided Token Modulation (EGTM) · 双语 README

> **EN ⇄ 中文**：每个小节同时提供英文与中文说明；命令保持一致。

---

## 1) Project Overview / 项目简介

**EN** — This repository implements **Edge‑Guided Token Modulation (EGTM)**: features from **ResNet‑18** and edges from a **Sobel** operator are fused and passed through a lightweight conv module to learn a **spatial weight map**. The map modulates fused tokens position‑wise and feeds a **ViT** head for classification. The design steers attention toward boundaries and salient regions, improving accuracy and robustness on CIFAR‑100 with modest parameters.

> Core class name: `CCT` (*Contour‑guided CNN + Transformer*).

**中文** — 本仓库实现 **边缘引导的 Token 调制（EGTM）**：将 **ResNet‑18** 特征与 **Sobel** 边缘融合，经轻量卷积得到**空间权重图**，对 token 逐位置调制后送入 **ViT** 分类头。该设计使注意力聚焦于边界与关键信息区域，在 CIFAR‑100 上以较小参数量获得更好的精度与鲁棒性。

---

## 2) Repository Layout / 目录结构

```
.
├─ models/
│  ├─ cct.py           # EGTM backbone: Sobel + ResNet18 fusion + weight map + ViT
│  ├─ resnet18.py      # ResNet‑18 feature extractor
│  ├─ vit.py           # Minimal ViT classifier head
│  └─ sobel.py         # Sobel edge operator
├─ train.py            # Single‑GPU training (TensorBoard, EMA, warmup+cosine)
├─ train_ddp.py        # Multi‑GPU / mixed precision via 🤗 Accelerate
├─ compare_gating.py   # Compare sigmoid / relu / hard gating
├─ robustness.py       # Robustness evaluation (e.g., corruptions)
└─ visualize.py        # Weight‑map & Grad‑CAM visualization
```

**EN** — The above layout groups model components, training/evaluation scripts, and visualization utilities.

**中文** — 上述结构包含模型组件、训练/评测脚本与可视化工具。

---

## 3) Environment / 环境依赖

**EN**

* Python ≥ 3.9
* PyTorch + torchvision (GPU recommended)
* Others: `accelerate`, `tensorboard`, `matplotlib`, `numpy`, `Pillow`

Install example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # choose per CUDA
pip install accelerate tensorboard matplotlib numpy pillow
```

**中文**

* Python ≥ 3.9
* PyTorch + torchvision（建议使用 GPU）
* 其他：`accelerate`、`tensorboard`、`matplotlib`、`numpy`、`Pillow`

安装示例同上；国内可配置镜像源以提升下载速度。

---

## 4) Dataset / 数据准备（CIFAR‑100）

**EN** — Scripts read **CIFAR‑100** from `./data` by default and **do not auto‑download**. Prepare the dataset at:

```
./data/
  └─ cifar-100-python/
     ├─ train
     ├─ test
     └─ meta
```

(If you prefer auto‑download, set `download=True` in the script.)

**中文** — 脚本默认从 `./data` 读取 **CIFAR‑100** 且**不自动下载**，请按上方结构放置数据；或在代码中将 `download=False` 改为 `True`。

---

## 5) Quickstart: Training / 快速开始：训练

### 5.1 Single GPU (`train.py`) / 单机单卡

**EN**

```bash
python train.py
```

* Logs in `runs/` (use TensorBoard)
* Checkpoints in `./checkpoints/` (`best_acc.pth`, `last.pth`, ...)

**中文**

* 日志目录 `runs/`（可用 TensorBoard 查看）
* 权重保存在 `./checkpoints/`（如 `best_acc.pth`、`last.pth`）

Open TensorBoard / 启动 TensorBoard：

```bash
tensorboard --logdir runs
```

### 5.2 Multi‑GPU & Mixed Precision (`train_ddp.py` + 🤗 Accelerate) / 多卡与混精

**EN** — Configure once:

```bash
accelerate config
```

Launch examples:

```bash
# 4 GPUs
accelerate launch --num_processes 4 train_ddp.py
# Or single process (still supports AMP)
python train_ddp.py
```

Checkpoints are saved to `./checkpoints_ddp/`.

**中文** — 先运行：`accelerate config` 完成多卡/混精配置；随后使用上面的命令启动。模型权重默认存到 `./checkpoints_ddp/`。

---

## 6) Visualization / 可视化（权重图 & Grad‑CAM）

**EN** — `visualize.py` samples test images and plots: original, EGTM weight‑map overlay, EGTM Grad‑CAM, and baseline Grad‑CAM.

```bash
python visualize.py \
  --data-dir ./data \
  --num-images 15 \
  --cct-checkpoint checkpoints_ddp/best_acc.pth \
  --baseline-checkpoint path/to/baseline_resnet18.pth \
  --output-dir feature_cam_outputs
```

**中文** — `visualize.py` 会输出原图、EGTM 权重图叠加、EGTM Grad‑CAM 与基线 Grad‑CAM；注意把 `--cct-checkpoint` 指向你训练得到的权重。

---

## 7) Experiment Scripts / 实验脚本

### A) Gating comparison / 不同 gating 对比

**EN**

```bash
python compare_gating.py \
  --data-dir ./data \
  --gatings sigmoid,relu,hard \
  --epochs 50 \
  --batch-size 128 \
  --output gating_results.json
```

**中文** — 对比 `sigmoid / relu / hard`，输出训练/验证/测试指标并汇总为 JSON。

### B) Robustness evaluation / 鲁棒性评测（CCT vs ResNet‑18）

**EN**

```bash
python robustness.py \
  --models cct,resnet18(baseline) \
  --data-dir ./data \
  --epochs 50 \
  --batch-size 128 \
  --corruption-files defocus_blur.npy,fog.npy \
  --labels-file labels.npy
```

**中文** — 在标准测试集与多种扰动（模糊、雾等）上评测；对应的 `.npy` 文件请放在项目根目录或提供绝对路径。

---

## 8) Training Defaults / 训练细节（默认）

**EN**

* Aug: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `RandAugment(2,7)`, Normalize, `RandomErasing(p=0.10)`
* Optim: SGD (momentum=0.9, nesterov=True), `weight_decay=5e‑4`
* LR: warmup + cosine (base\_lr=0.05 for single GPU; scaled by global batch for DDP)
* Label smoothing: 0.1; EMA enabled; 5k validation split from training

**中文**

* 数据增强：随机裁剪/翻转、RandAugment、标准化、随机擦除
* 优化设置：SGD（momentum=0.9, nesterov=True），`weight_decay=5e‑4`
* 学习率：warmup + cosine（单卡 `base_lr=0.05`，多卡按全局 batch 缩放）
* 其他：label smoothing=0.1，默认启用 EMA；从官方训练集划分 5k 作为验证集

> All hyper‑parameters are editable in scripts / 所有超参可在脚本中直接修改。

---

## 9) FAQ / 常见问题

**EN**

* *Missing CIFAR‑100*: verify folder layout or set `download=True`.
* *Out of memory*: reduce `--batch-size`, image size (if supported), or disable AMP.
* *Checkpoint not found in visualization*: ensure `--cct-checkpoint` path matches your saved file.

**中文**

* *找不到 CIFAR‑100*：检查目录结构；或启用自动下载。
* *显存不足*：减小 `--batch-size`，或降低图像尺寸（若脚本支持），或关闭混精。
* *可视化加载权重失败*：确认 `--cct-checkpoint` 与实际保存的文件名一致。

---

## 10) Citation & Acknowledgements / 引用与致谢

**EN** — This work builds on ResNet‑18, ViT, and Sobel‑based priors. Multi‑GPU training is powered by 🤗 Accelerate. Thanks to the open‑source community.

**中文** — 本实现基于 ResNet‑18、ViT 与 Sobel 先验；多卡训练依赖 🤗 Accelerate。感谢开源社区的相关工作与工具链。
