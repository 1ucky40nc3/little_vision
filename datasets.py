from typing import List
from typing import Iterator

from functools import partial

from itertools import cycle

import torch
import torch.utils.data as tud

import torchvision
import torchvision.transforms as transforms

import jax
import jax.numpy as jnp

from flax import jax_utils

import ml_collections as mlc

import einops



def to_jax_img(image: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(image, "c h w -> h w c")

transforms.ToJax = partial(transforms.Lambda, lambd=to_jax_img)



def transform(
    config: mlc.ConfigDict
) -> transforms.Compose:
    t = [] # TODO: parse transform from config
    return transforms.Compose([
        transforms.ToTensor(),
        *t,
        transforms.ToJax(),
    ])


def mnist(
    train: bool,
    config: mlc.ConfigDict
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

    # TODO: implement prefetching
    return cycle(loader) if train else loader


def shard(array: jnp.ndarray):
  return einops.rearrange(
      array, 
      "(d n) ... -> d n ...", 
      d=jax.local_device_count())


def prepare(
    iterator: Iterator, 
    config: mlc.ConfigDict
) -> Iterator:
    iterator = (
        jax.tree_map(
            lambda t: t.numpy(), b
        ) for b in iterator)
    iterator = (
        jax.tree_map(
            shard, b
        ) for b in iterator)
    iterator = jax_utils.prefetch_to_device(
        iterator, config.dataset.prefetch_size)

    return iterator
