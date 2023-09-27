from typing import Tuple
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # model definition
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_dim: int = 3072

    pretrained_ckpt: str = None  # "./checkpoints/pretrained/vit_b_16-c867db91.pth"

    # CIFAR10 datasets
    num_classes: int = 10
    num_channels: int = 3
    image_size: int = 224  # resize original image same as pretrained weights
    patch_size: int = 16  # same as pretrained weights

    # training
    train_epochs: int = 20
    batch_size: int = 150

    # learning rate scheduler, note learning rate too big might cause NaN during training
    init_lr: float = 1e-5  # initial learning rate
    max_lr: float = 5e-5  # maximum learning rate after warm up
    min_lr: float = 5e-5  # minimum learning rate after decay
    warmup_epochs: int = 2
    decay_epochs: int = 20

    # Adam optimizer
    weight_decay: float = 0.03
    adam_betas: Tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_fused: bool = True

    # regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    grad_clip: float = 10.0

    # others
    seed: int = 127
    log_dir: str = './logs/'  # save logs and traces
    ckpt_dir: str = './checkpoints/'
    use_tensorboard: bool = True
