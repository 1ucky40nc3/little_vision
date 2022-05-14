import torch

import jax
import jax.numpy as jnp

import einops


def shard(array: jnp.ndarray) -> jnp.array:
    return einops.rearrange(
        array, 
        "(d n) ... -> d n ...", 
        d=jax.local_device_count())


# TODO: maybe shard with flax.jax_utils.replicate
def jaxify(tensor: torch.Tensor) -> jnp.ndarray:
    return shard(tensor.numpy())


class Writer:
    pass