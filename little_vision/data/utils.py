from typing import Union
from typing import Iterator

import torch

from timm.data.mixup import Mixup

import jax
import jax.numpy as jnp

from flax import jax_utils

import ml_collections as mlc

import einops


ConfigDict = Union[mlc.ConfigDict, mlc.FrozenConfigDict]


def to_jax_img(image: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(image, "n c h w -> n h w c")


def shard(array: jnp.ndarray):
  return einops.rearrange(
      array, 
      "(d n) ... -> d n ...", 
      d=jax.local_device_count())


def prepare(
    iterator: Iterator, 
    config: ConfigDict
) -> Iterator:
    is_list = lambda x: isinstance(x, list)

    if config.transform.mixup:
        mixup = Mixup(
            **config.transform.mixup_config)
        mixup_fn = lambda t: mixup(*t)

        iterator = (
             jax.tree_map(
                mixup_fn,
                b,
                is_leaf=is_list)
             for b in iterator)

    iterator = (
        jax.tree_map(
            lambda t: t.numpy(), b)
        for b in iterator)

    to_jax = (
        lambda t: t 
        if t.ndim != 4 
        else to_jax_img(t))
    iterator = (
        jax.tree_map(
            to_jax, b) 
        for b in iterator)

    iterator = (
        jax.tree_map(
            shard, b) 
        for b in iterator)
    
    iterator = jax_utils.prefetch_to_device(
        iterator, config.dataset.prefetch_size)

    return iterator