import os
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from accelerate import Accelerator, DistributedDataParallelKwargs
from typing import List, cast
from models.cct import CCT


def main():
    # ---- Accelerate: init & device ----
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])  # or "fp16"
    device = accelerator.device

    # Use an absolute checkpoints directory next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(script_dir, "checkpoints_ddp")
    os.makedirs(ckpt_dir, exist_ok=True)

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # Slightly gentler augmentation to reduce variance
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.10),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Datasets: train/val split from training set; separate test set
    data_root = os.path.join(script_dir, "data")

    # Download the dataset only once on the main process, then all ranks wait
    if accelerator.is_main_process:
        datasets.CIFAR100(root=data_root, train=True, download=False)
        datasets.CIFAR100(root=data_root, train=False, download=False)
    accelerator.wait_for_everyone()

    base_train = datasets.CIFAR100(root=data_root, train=True, transform=train_tf, download=False)
    base_val   = datasets.CIFAR100(root=data_root, train=True, transform=test_tf, download=False)
    base_test  = datasets.CIFAR100(root=data_root, train=False, transform=test_tf, download=False)

    n_total = len(base_train)
    n_val = 5000
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_set = Subset(base_train, train_idx)
    val_set = Subset(base_val, val_idx)

    batch_size = 128  # per-process batch size
    num_workers = 8
    pin = True if torch.cuda.is_available() else False
    persistent = True if num_workers > 0 else False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        base_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # ---- Model & Optimizer ----
    model = CCT()  # let Accelerate move it to device

    def make_param_groups(model, wd):
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    epochs = 300
    warm_epochs = 5
    weight_decay = 5e-4

    # Scale LR by GLOBAL batch size (per-device × #processes × grad_accum)
    grad_accum = accelerator.gradient_accumulation_steps
    world_size = accelerator.num_processes
    global_batch = batch_size * world_size * grad_accum
    base_lr = 0.1 * (global_batch / 256)

    optimizer = torch.optim.SGD(
        make_param_groups(model, weight_decay),
        lr=base_lr,
        momentum=0.9,
        nesterov=True,
    )

    # ---- Prepare for DDP/AMP ----
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    # ---- Schedulers (create AFTER .prepare so they track wrapped optimizer) ----
    iters_per_epoch = len(train_loader)
    warmup_total_steps = warm_epochs * iters_per_epoch
    total_steps = epochs * iters_per_epoch

    def warmup_lambda(step: int):
        if step < warmup_total_steps:
            return float(step + 1) / float(max(1, warmup_total_steps))
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_total_steps),
        eta_min=base_lr * 1e-2,
    )

    # ---- Simple EMA of model weights for smoother validation ----
    class ModelEma:
        def __init__(self, model, decay=0.999):
            self.ema = copy.deepcopy(model).eval()
            self.decay = decay
            for p in self.ema.parameters():
                p.requires_grad_(False)

        @torch.no_grad()
        def update(self, src_model):
            msd = src_model.state_dict()
            for k, v in self.ema.state_dict().items():
                src = msd[k]
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + src * (1.0 - self.decay))
                else:
                    v.copy_(src)

        def to(self, device):
            self.ema.to(device)
    def unwrap_ddp(m):
        return m.module if hasattr(m, "module") else m

    use_ema_eval = True
    ema = ModelEma(unwrap_ddp(model), decay=0.999) if use_ema_eval else None
    if ema is not None:
        ema.to(device)

    @torch.no_grad()
    def evaluate(loader, net=None):
        # prefer EMA if available; otherwise unwrap DDP model for clean state_dict
        if net is None:
            net = ema.ema if (ema is not None and use_ema_eval) else unwrap_ddp(model)
        net.eval()
        all_pred: List[torch.Tensor] = []
        all_label: List[torch.Tensor] = []
        loss_sum_local, count_local = 0.0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = net(x)
            loss = criterion(logits, y)
            loss_sum_local += loss.item() * x.size(0)
            count_local += x.size(0)
            pred = logits.argmax(1)
            # gather predictions/labels across processes for correct metrics
            pred_g = cast(torch.Tensor, accelerator.gather_for_metrics(pred))
            y_g = cast(torch.Tensor, accelerator.gather_for_metrics(y))
            # Pylance sometimes infers list; coerce to Tensor for type checker & robustness
            if isinstance(pred_g, list):
                pred_g = torch.cat([p.detach() for p in pred_g], dim=0)
            if isinstance(y_g, list):
                y_g = torch.cat([t.detach() for t in y_g], dim=0)
            all_pred.append(pred_g.detach().cpu())
            all_label.append(y_g.detach().cpu())
        pred_t = torch.cat(all_pred, dim=0)
        label_t = torch.cat(all_label, dim=0)
        acc = (pred_t == label_t).to(torch.float32).mean().item() * 100.0
        # gather loss sums and counts for a true global average
        lc = torch.tensor([loss_sum_local, count_local], device=device, dtype=torch.float64).unsqueeze(0)
        glc_any = accelerator.gather_for_metrics(lc)
        if isinstance(glc_any, list):
            glc = torch.cat(glc_any, dim=0)
        else:
            glc = cast(torch.Tensor, glc_any)
        loss_col = glc.select(dim=1, index=0)
        count_col = glc.select(dim=1, index=1)
        loss_sum = loss_col.sum().item()
        count_sum = count_col.sum().item()
        avg_loss = loss_sum / max(1.0, count_sum)
        return avg_loss, acc

    # ---- Training loop ----
    global_step = 0
    best_val_acc = 0.0
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience = 10
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with accelerator.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Update EMA with the UNWRAPPED model
            if ema is not None:
                ema.update(unwrap_ddp(model))

            # per-step LR schedule
            if global_step < warmup_total_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            global_step += 1

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / max(1, total)
        cur_lr = optimizer.param_groups[0]['lr']

        # Validation each epoch for best checkpointing
        val_loss, val_acc = evaluate(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_loss.pth'))
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = unwrap_ddp(model)
                state_dict = (ema.ema.state_dict() if (ema is not None and use_ema_eval) else unwrapped.state_dict())
                accelerator.save(
                    {
                        'epoch': epoch,
                        'model': state_dict,
                        'optimizer': optimizer.state_dict(),
                        'val_acc': best_val_acc,
                    },
                    os.path.join(ckpt_dir, 'best_acc.pth'),
                )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = unwrap_ddp(model)
            state_dict = (ema.ema.state_dict() if (ema is not None and use_ema_eval) else unwrapped.state_dict())
            accelerator.save(
                {
                    'epoch': epoch,
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'val_acc': val_acc,
                },
                os.path.join(ckpt_dir, 'last.pth'),
            )

        accelerator.print(
            f"[train] epoch {epoch:03d} | lr={cur_lr:.5f} | loss={epoch_loss:.4f} | acc={epoch_acc:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%"
        )

    # Final evaluation on val and test sets
    val_loss, val_acc = evaluate(val_loader)
    test_loss, test_acc = evaluate(test_loader)
    accelerator.print(f"[final] val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%")
    accelerator.print(f"[final] test_loss={test_loss:.4f} | test_acc={test_acc:.2f}%")


if __name__ == "__main__":
    main()
