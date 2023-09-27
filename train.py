from typing import Tuple, Callable, Mapping, Text, Any
import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    Normalize,
)
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.hub import tqdm


from model import VisionTransformer
from schedule import CosineDecayWithWarmupLRScheduler
from config import TrainConfig as cfg


torch.autograd.set_detect_anomaly(True)


def get_grad_norm(model: VisionTransformer) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            local_norm = torch.linalg.vector_norm(p.grad, dtype=p.dtype)
            total_norm += local_norm**2
    return (total_norm**0.5).item()


def train_one_epoch(
    model: VisionTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineDecayWithWarmupLRScheduler,
    train_loader: DataLoader,
    device: torch.device,
    current_epoch: int,
    loss_fn: Callable = torch.nn.CrossEntropyLoss(),
    grad_clip: float = 0.0,
) -> dict:
    model = model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_grad_norms = 0.0
    max_grad_norms = 0.0

    tk = tqdm(train_loader, desc=f'EPOCH [TRAIN] {current_epoch}')

    for t, data in enumerate(tk):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()

        grad_norm = get_grad_norm(model)
        total_grad_norms += grad_norm
        if grad_norm > max_grad_norms:
            max_grad_norms = grad_norm

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += len(predicted)

        tk.set_postfix({'Loss': f'{float(total_loss / (t + 1)):.6f}'})

    loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    stats = {
        'loss': loss,
        'accuracy': accuracy,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'grad_norm': total_grad_norms / len(train_loader),
        'grad_norm_max': max_grad_norms,
    }

    return stats


@torch.no_grad()
def eval_one_epoch(
    model: VisionTransformer,
    val_loader: DataLoader,
    device: torch.device,
    current_epoch: int,
    loss_fn: Callable = torch.nn.CrossEntropyLoss(),
) -> dict:
    model = model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    tk = tqdm(val_loader, desc=f'EPOCH [VALIDATION] {current_epoch}')

    for t, data in enumerate(tk):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += len(predicted)

        tk.set_postfix({'Loss': f'{float(total_loss / (t + 1)):.6f}'})

    loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    stats = {'loss': loss, 'accuracy': accuracy}
    return stats


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Calculate the mean and standard deviation of the CIFAR-10 dataset
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=ToTensor())

    mean = cifar10_dataset.data.mean(axis=(0, 1, 2)) / 255.0
    std = cifar10_dataset.data.std(axis=(0, 1, 2)) / 255.0

    transforms_train = Compose(
        [
            Resize((cfg.image_size, cfg.image_size)),
            ToTensor(),
            Normalize(mean, std),
        ]
    )
    transforms_val = Compose(
        [
            Resize((cfg.image_size, cfg.image_size)),
            ToTensor(),
            Normalize(mean, std),
        ]
    )

    train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transforms_train)
    val_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transforms_val)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    model = VisionTransformer(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        num_classes=cfg.num_classes,
    )

    model_prefix = f'vit_base_{cfg.patch_size}'  # base model with patch size

    if cfg.pretrained_ckpt is not None and os.path.exists(cfg.pretrained_ckpt):
        print(f'Loading pretrained weights from {cfg.pretrained_ckpt} ...')
        state = torch.load(cfg.pretrained_ckpt)
        # delete head since number of classes are not the same
        del state['heads.head.weight']
        del state['heads.head.bias']
        model.load_state_dict(state, strict=False)
        del state

        model_prefix += '_pretrained'

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.init_lr,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        fused=cfg.adam_fused,
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        init_lr=cfg.init_lr,
        max_lr=cfg.max_lr,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_epochs * len(train_loader),
        max_decay_steps=cfg.decay_epochs * len(train_loader),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    if cfg.use_tensorboard:
        tb_writer = SummaryWriter(os.path.join(cfg.log_dir, f'{model_prefix}'))

    best_valid_loss = np.inf
    best_train_loss = np.inf
    for i in range(1, cfg.train_epochs + 1):
        train_stats = train_one_epoch(model, optimizer, scheduler, train_loader, device, i, loss_fn, cfg.grad_clip)

        print(f"Train epoch {i}, loss={train_stats['loss']:.6f}, accuracy={train_stats['accuracy']:.2f}")

        if tb_writer is not None:
            for k, v in train_stats.items():
                tb_writer.add_scalar(f'train_epoch/{k}', v, i)

        val_stats = eval_one_epoch(model, val_loader, device, i, loss_fn)

        print(f"Validation epoch {i}, loss={val_stats['loss']:.6f}, accuracy={val_stats['accuracy']:.2f}")

        if tb_writer is not None:
            for k, v in val_stats.items():
                tb_writer.add_scalar(f'validation_epoch/{k}', v, i)

        if val_stats['loss'] < best_valid_loss:
            torch.save(
                model.state_dict(),
                os.path.join(cfg.ckpt_dir, f'{model_prefix}_best_weights.pt'),
            )
            print('Saved Best Weights')
            best_valid_loss = val_stats['loss']
            best_train_loss = train_stats['loss']

    print(f'Best training loss : {best_train_loss}')
    print(f'Best validation loss : {best_valid_loss}')


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')
    main()
