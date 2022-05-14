from typing import List

from functools import partial

import torch
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

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
) -> data.DataLoader:
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
    
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=config.dataset.batch_size,
        shuffle=train,
        num_workers=config.dataset.num_workers)

    # TODO: implement prefetching
    return loader


