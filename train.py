import os
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from models.cct import CCT
from torch.utils.tensorboard import SummaryWriter

def main():
    writer = SummaryWriter('runs/cct_experiment2')  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCT().to(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.10),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    base_train = datasets.CIFAR100(root="./data", train=True, transform=train_tf, download=False)
    base_val = datasets.CIFAR100(root="./data", train=True, transform=test_tf, download=False)
    base_test = datasets.CIFAR100(root="./data", train=False, transform=test_tf, download=False)

    n_total = len(base_train)
    n_val = 5000
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_set = Subset(base_train, train_idx)
    val_set = Subset(base_val, val_idx)

    batch_size = 128
    num_workers = 2
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

    epochs = 200
    warm_epochs = 5
    weight_decay = 5e-4

    base_lr = 0.05

    optimizer = torch.optim.SGD(
        make_param_groups(model, weight_decay),
        lr=base_lr,
        momentum=0.9,
        nesterov=True,
    )

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

    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16
    scaler = torch.GradScaler(enabled=use_amp)

    class ModelEma:
        def __init__(self, model, decay=0.999):
            self.ema = copy.deepcopy(model).eval()
            self.decay = decay
            for p in self.ema.parameters():
                p.requires_grad_(False)

        @torch.no_grad()
        def update(self, model):
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                src = msd[k]
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + src * (1.0 - self.decay))
                else:
                    v.copy_(src)

        def to(self, device):
            self.ema.to(device)

    use_ema_eval = True
    ema = ModelEma(model, decay=0.999) if use_ema_eval else None
    if ema is not None:
        ema.to(device)

    @torch.no_grad()
    def evaluate(loader, net=None):
        if net is None:
            net = ema.ema if (ema is not None and use_ema_eval) else model
        net.eval()
        tot, cor, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = net(x)
            loss_sum += criterion(logits, y).item() * x.size(0)
            cor += (logits.argmax(1) == y).sum().item()
            tot += x.size(0)
        return loss_sum / tot, 100.0 * cor / tot

    global_step = 0
    best_val_acc = 0.0
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience = 5
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

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if ema is not None:
                ema.update(model)

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
        epoch_acc = 100.0 * correct / total
        cur_lr = optimizer.param_groups[0]['lr']

        val_loss, val_acc = evaluate(val_loader)
        writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

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
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    'epoch': epoch,
                    'model': (ema.ema.state_dict() if (ema is not None and use_ema_eval) else model.state_dict()),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if use_amp else None,
                    'val_acc': best_val_acc,
                },
                os.path.join(ckpt_dir, 'best_acc.pth'),
            )

        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                'epoch': epoch,
                'model': (ema.ema.state_dict() if (ema is not None and use_ema_eval) else model.state_dict()),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if use_amp else None,
                'val_acc': val_acc,
            },
            os.path.join(ckpt_dir, 'last.pth'),
        )

        print(
            f"[train] epoch {epoch:03d} | lr={cur_lr:.5f} | loss={epoch_loss:.4f} | acc={epoch_acc:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%"
        )
    val_loss, val_acc = evaluate(val_loader)
    test_loss, test_acc = evaluate(test_loader)
    print(f"[final] val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%")
    print(f"[final] test_loss={test_loss:.4f} | test_acc={test_acc:.2f}%")
    writer.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
