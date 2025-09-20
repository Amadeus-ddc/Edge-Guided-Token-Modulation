from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cct import CCT
from models.resnet18 import ResNet18
from models.vit import ViT


MEAN = [0.5071, 0.4867, 0.4408]
STD = [0.2675, 0.2565, 0.2761]


@dataclass
class SampleResult:
    index: int
    true_label: str
    cct_pred: str
    baseline_pred: str
    cct_confidence: float
    baseline_confidence: float
    figure_path: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "true_label": self.true_label,
            "cct_prediction": self.cct_pred,
            "baseline_prediction": self.baseline_pred,
            "cct_confidence": self.cct_confidence,
            "baseline_confidence": self.baseline_confidence,
            "figure": self.figure_path,
        }


class EdgeFreeModel(nn.Module):
    """ResNet-18 backbone followed by ViT without edge-guided weighting."""

    def __init__(self) -> None:
        super().__init__()
        self.resnet = ResNet18()
        self.vit = ViT()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.resnet(x)
        tokens = torch.cat([features, features], dim=1)
        return self.vit(tokens)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(MEAN)[:, None, None]
    std = image.new_tensor(STD)[:, None, None]
    return torch.clamp(image * std + mean, 0.0, 1.0)


def compute_grad_cam(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
    cam = cam.squeeze(0).squeeze(0)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def upsample_to_image(map_2d: torch.Tensor) -> torch.Tensor:
    map_2d = map_2d.unsqueeze(0).unsqueeze(0)
    map_2d = F.interpolate(map_2d, size=(32, 32), mode="bilinear", align_corners=False)
    map_2d = map_2d.squeeze(0).squeeze(0)
    map_2d -= map_2d.min()
    if map_2d.max() > 0:
        map_2d /= map_2d.max()
    return map_2d


def save_comparison_figure(
    out_path: str,
    original: torch.Tensor,
    cct_weight_overlay: torch.Tensor,
    cct_cam: torch.Tensor,
    baseline_cam: torch.Tensor,
    titles: Dict[str, str],
) -> None:
    image_np = original.permute(1, 2, 0).cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    axes[0].imshow(image_np)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(image_np)
    axes[1].imshow(cct_weight_overlay.cpu().numpy(), cmap="viridis", alpha=0.6)
    axes[1].set_title(titles.get("weight", "EGTM Weight"), fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(image_np)
    axes[2].imshow(cct_cam.cpu().numpy(), cmap="jet", alpha=0.6)
    axes[2].set_title(titles.get("cct_cam", "EGTM Grad-CAM"), fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(image_np)
    axes[3].imshow(baseline_cam.cpu().numpy(), cmap="jet", alpha=0.6)
    axes[3].set_title(titles.get("baseline_cam", "Baseline Grad-CAM"), fontsize=10)
    axes[3].axis("off")

    fig.suptitle(titles.get("header", ""), fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def forward_cct(model: CCT, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    features = model.resnet(x)
    fusion = torch.cat([features, features], dim=1)
    weight_map = model.weight(fusion)
    token = fusion * weight_map
    token.retain_grad()
    weight_map.retain_grad()
    logits = model.vit(token)
    return {
        "logits": logits,
        "token": token,
        "fusion": fusion,
        "weight": weight_map,
    }


def forward_baseline(model: EdgeFreeModel, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    features = model.resnet(x)
    fusion = torch.cat([features, features], dim=1)
    token = fusion
    token.retain_grad()
    logits = model.vit(token)
    return {
        "logits": logits,
        "token": token,
        "fusion": fusion,
    }


def zero_module_grads(module: nn.Module) -> None:
    for param in module.parameters():
        if param.grad is not None:
            param.grad.zero_()


def maybe_load_checkpoint(module: nn.Module, path: Optional[str]) -> None:
    if not path:
        return
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    module.load_state_dict(state, strict=False)
    print(f"Loaded weights from {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EGTM attention modulation with Grad-CAM")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-100 data directory")
    parser.add_argument("--num-images", type=int, default=15, help="Number of images to visualize")
    parser.add_argument("--indices", type=str, default="", help="Comma separated image indices to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when sampling images")
    parser.add_argument("--cct-checkpoint", type=str, default="CCT/checkpoints_ddp/cct.pth", help="Optional path to trained EGTM checkpoint")
    parser.add_argument("--baseline-checkpoint", type=str, default="resnet-DDP/checkpoints_baseline/best.pth", help="Optional path to baseline checkpoint")
    parser.add_argument("--output-dir", type=str, default="feature_cam_outputs", help="Directory to store figures")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto/cpu/cuda)")
    parser.add_argument("--save-json", type=str, default="", help="Optional path for experiment metadata JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    dataset = datasets.CIFAR100(root=args.data_dir, train=False, transform=normalize, download=False)
    class_names = dataset.classes

    if args.indices:
        indices = [int(idx.strip()) for idx in args.indices.split(",") if idx.strip()]
    else:
        random.seed(args.seed)
        indices = random.sample(range(len(dataset)), k=min(args.num_images, len(dataset)))

    cct_model = CCT().to(device)
    baseline_model = EdgeFreeModel().to(device)
    maybe_load_checkpoint(cct_model, args.cct_checkpoint)
    maybe_load_checkpoint(baseline_model, args.baseline_checkpoint)
    cct_model.eval()
    baseline_model.eval()

    results: List[SampleResult] = []

    for idx in indices:
        image_norm, label = dataset[idx]
        image_batch = image_norm.unsqueeze(0).to(device)
        original = denormalize(image_norm)

        for module in (cct_model, baseline_model):
            zero_module_grads(module)

        cct_out = forward_cct(cct_model, image_batch)
        cct_logits = cct_out["logits"]
        cct_prob = torch.softmax(cct_logits, dim=1)
        cct_pred_idx = int(cct_prob.argmax(dim=1).item())
        cct_score = cct_logits[0, cct_pred_idx]
        if cct_out["token"].grad is not None:
            cct_out["token"].grad.zero_()
        if cct_out["weight"].grad is not None:
            cct_out["weight"].grad.zero_()
        cct_score.backward()
        cct_cam = compute_grad_cam(cct_out["token"].detach(), cct_out["token"].grad)
        weight_map = cct_out["weight"].detach().squeeze(0).squeeze(0)
        weight_overlay = upsample_to_image(weight_map)

        zero_module_grads(cct_model)

        baseline_out = forward_baseline(baseline_model, image_batch)
        baseline_logits = baseline_out["logits"]
        baseline_prob = torch.softmax(baseline_logits, dim=1)
        baseline_pred_idx = int(baseline_prob.argmax(dim=1).item())
        baseline_score = baseline_logits[0, baseline_pred_idx]
        if baseline_out["token"].grad is not None:
            baseline_out["token"].grad.zero_()
        baseline_score.backward()
        baseline_cam = compute_grad_cam(baseline_out["token"].detach(), baseline_out["token"].grad)
        zero_module_grads(baseline_model)

        titles = {
            "header": f"True: {class_names[label]} | EGTM: {class_names[cct_pred_idx]} ({cct_prob[0, cct_pred_idx]:.2f}) | Baseline: {class_names[baseline_pred_idx]} ({baseline_prob[0, baseline_pred_idx]:.2f})",
            "weight": "EGTM Weight Map",
            "cct_cam": "EGTM Grad-CAM",
            "baseline_cam": "Baseline Grad-CAM",
        }
        fig_name = f"sample_{idx:05d}.png"
        fig_path = os.path.join(args.output_dir, fig_name)
        save_comparison_figure(fig_path, original, weight_overlay, cct_cam, baseline_cam, titles)

        results.append(
            SampleResult(
                index=idx,
                true_label=class_names[label],
                cct_pred=class_names[cct_pred_idx],
                baseline_pred=class_names[baseline_pred_idx],
                cct_confidence=float(cct_prob[0, cct_pred_idx].item()),
                baseline_confidence=float(baseline_prob[0, baseline_pred_idx].item()),
                figure_path=fig_path,
            )
        )

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"samples": [res.to_dict() for res in results]}, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {os.path.abspath(args.save_json)}")

    print(f"Generated {len(results)} visualization figures in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
