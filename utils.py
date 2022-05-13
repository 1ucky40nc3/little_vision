
import jax
import jax.numpy as jnp

import einops


def shard(array: jnp.ndarray) -> jnp.array:
    return einops.rearrange(
        array, 
        "(d n) ... -> d n ...", 
        d=jax.local_device_count())


class Writer:
    pass