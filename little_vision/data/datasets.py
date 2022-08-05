from typing import Any
from typing import List
from typing import Union
from typing import Iterator
from typing import Callable
from typing import Sequence

from functools import partial

from itertools import cycle

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as tud

import torchvision
import torchvision.transforms as transforms

from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform


import jax
import jax.numpy as jnp

from flax import jax_utils

import ml_collections as mlc

import einops


ConfigDict = Union[mlc.ConfigDict, mlc.FrozenConfigDict]


def mnist(
    train: bool,
    config: ConfigDict
) -> Iterator:
    """Return a MNIST loader.

    Example:
        >>> dataset_config = mlc.ConfigDict()
        >>> dataset_config.batch_size = 32
        >>> dataset_config.num_workers = 0
        >>> loader = mnist(train=True, config=dataset_config)
        >>> images, labels = next(iter(loader))
        >>> print(images.shape)
        torch.Size([32, 28, 28, 1])
        >>> print(labels.shape)
        torch.Size([32])
        >>> print(images.min(), images.max())
        tensor(0.) tensor(1.)
    """
    dataset = torchvision.datasets.MNIST(
        root=config.dataset.root, 
        train=train, 
        transform=transform(config),
        download=config.dataset.download)
    
    loader = tud.DataLoader(
        dataset=dataset,
        batch_size=config.dataset.batch_size,
        shuffle=train,
        num_workers=config.dataset.num_workers,
        drop_last=train)

    return cycle(loader) if train else loader


def cifar100(
    train: bool,
    config: ConfigDict
) -> Iterator:
    dataset = torchvision.datasets.CIFAR100(
        root=config.dataset.root, 
        train=train, 
        transform=transform(config),
        download=config.dataset.download)
    
    loader = tud.DataLoader(
        dataset=dataset,
        batch_size=config.dataset.batch_size,
        shuffle=train,
        num_workers=config.dataset.num_workers,
        drop_last=train)

    return cycle(loader) if train else loader



