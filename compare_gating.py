from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.resnet18 import ResNet18
from models.sobel import Sobel
from models.vit import ViT


@dataclass
class TrainConfig:
    data_dir: str
    batch_size: int
    num_workers: int
    epochs: int
    warmup_epochs: int
    base_lr: float
    weight_decay: float
    label_smoothing: float
    seed: int
    val_split: int
    amp: bool
    device: torch.device
    img_size: int
    gatings: List[str]
    output: Optional[str]


GATING_CHOICES = {"sigmoid", "relu", "hard"}


class CCTWithGating(nn.Module):
    """EGTM variant with configurable gating function."""

    def __init__(self, gating: str, num_classes: int = 100) -> None:
        super().__init__()
        gating = gating.lower()
        if gating not in GATING_CHOICES:
            raise ValueError(f"Unsupported gating \"{gating}\". Choose from {sorted(GATING_CHOICES)}")
        self.gating = gating
        self.sobel = Sobel()
        self.resnet = ResNet18()
        self.vit = ViT(num_classes=num_classes)
        self.weight_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def _apply_gating(self, raw: torch.Tensor) -> torch.Tensor:
        if self.gating == "sigmoid":
            return torch.sigmoid(raw)
        if self.gating == "relu":
            return F.relu(raw)
        # Hard thresholding with straight-through estimator for gradients
        prob = torch.sigmoid(raw)
        binary = (prob > 0.5).float()
        return binary + (prob - prob.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.resnet(x)
        fusion = torch.cat([features, features], dim=1)
        weight_raw = self.weight_conv(fusion)
        weight_map = self._apply_gating(weight_raw)
        token_weight = fusion * weight_map
        logits = self.vit(token_weight)
        return logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: TrainConfig) -> Dict[str, DataLoader]:
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.10),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    base_train = datasets.CIFAR100(root=cfg.data_dir, train=True, transform=train_tf, download=False)
    base_val = datasets.CIFAR100(root=cfg.data_dir, train=True, transform=test_tf, download=False)
    base_test = datasets.CIFAR100(root=cfg.data_dir, train=False, transform=test_tf, download=False)

    n_total = len(base_train)
    indices = list(range(n_total))
    random.Random(cfg.seed).shuffle(indices)
    val_indices = indices[: cfg.val_split]
    train_indices = indices[cfg.val_split :]

    train_set = Subset(base_train, train_indices)
    val_set = Subset(base_val, val_indices)

    pin_memory = cfg.device.type == "cuda"
    persistent = cfg.num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        base_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def create_optimizer(model: nn.Module, base_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.SGD(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=base_lr,
        momentum=0.9,
        nesterov=True,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    iters_per_epoch: int,
) -> Optional[LambdaLR]:
    total_steps = total_epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch
    if total_steps == 0:
        return None

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps == warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start = time.perf_counter()
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    avg_loss = total_loss / max(1, total_samples)
    acc = 100.0 * total_correct / max(1, total_samples)
    lr = optimizer.param_groups[0]["lr"]
    return {"loss": avg_loss, "accuracy": acc, "time": elapsed, "lr": lr, "samples": float(total_samples)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start = time.perf_counter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    avg_loss = total_loss / max(1, total_samples)
    acc = 100.0 * total_correct / max(1, total_samples)
    throughput = total_samples / elapsed if elapsed > 0 else 0.0
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "time": elapsed,
        "samples": float(total_samples),
        "throughput": throughput,
    }


def run_experiment(gating: str, cfg: TrainConfig, loaders: Dict[str, DataLoader]) -> Dict[str, object]:
    device = cfg.device
    set_seed(cfg.seed)
    model = CCTWithGating(gating=gating, num_classes=100).to(device)
    params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = create_optimizer(model, cfg.base_lr, cfg.weight_decay)
    scheduler = create_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs, len(loaders["train"]))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    history: List[Dict[str, float]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "train_time": train_metrics["time"],
            }
        )
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[{gating}] epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.2f}% "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.2f}% "
            f"lr={train_metrics['lr']:.5f} time={train_metrics['time']:.2f}s"
        )
    total_train_time = time.perf_counter() - train_start
    max_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], criterion, device)

    results: Dict[str, object] = {
        "gating": gating,
        "parameters": float(params),
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "train_time_sec": total_train_time,
        "avg_epoch_time_sec": total_train_time / cfg.epochs if cfg.epochs > 0 else 0.0,
        "inference_time_sec": test_metrics["time"],
        "inference_throughput": test_metrics["throughput"],
        "max_memory_bytes": float(max_memory) if max_memory is not None else None,
        "history": history,
    }
    return results


def format_summary(results: List[Dict[str, object]]) -> str:
    headers = [
        "Gating",
        "Params (M)",
        "Best Val (%)",
        "Test (%)",
        "Train Time (s)",
        "Infer Time (s)",
    ]
    lines = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for res in results:
        params_m = res["parameters"] / 1e6
        line = " | ".join(
            [
                res["gating"],
                f"{params_m:.2f}",
                f"{res['best_val_accuracy']:.2f}",
                f"{res['test_accuracy']:.2f}",
                f"{res['train_time_sec']:.1f}",
                f"{res['inference_time_sec']:.3f}",
            ]
        )
        lines.append(line)
    return "\n".join(lines)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Compare gating functions within EGTM model")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=int, default=5000)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto/cpu/cuda)")
    parser.add_argument("--img-size", type=int, default=32, help="Input image size")
    parser.add_argument(
        "--gatings",
        type=str,
        default="sigmoid,relu,hard",
        help="Comma separated gating names to compare (choices: sigmoid,relu,hard)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gating_results.json",
        help="Where to store detailed metrics",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    gating_list = [g.strip().lower() for g in args.gatings.split(",") if g.strip()]
    for g in gating_list:
        if g not in GATING_CHOICES:
            raise ValueError(f"Invalid gating \"{g}\". Allowed: {sorted(GATING_CHOICES)}")

    cfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        val_split=args.val_split,
        amp=not args.no_amp,
        device=device,
        img_size=args.img_size,
        gatings=gating_list,
        output=args.output,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    loaders = build_dataloaders(cfg)

    results: List[Dict[str, object]] = []
    for gating in cfg.gatings:
        print(f"\n==== Training gating: {gating} ====")
        res = run_experiment(gating, cfg, loaders)
        results.append(res)

    print("\n" + format_summary(results))
    if cfg.output:
        with open(cfg.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed metrics saved to {os.path.abspath(cfg.output)}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
