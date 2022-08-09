from typing import Union
from typing import Iterator

import torch


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