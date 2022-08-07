from functools import partial

import jax
import jax.numpy as jnp

import flax.linen as nn


class DropPath(nn.Module):
    """Drop path implementation.
    
    For reference see:
    https://github.com/rwightman/efficientnet-jax/blob/a65811fbf63cb90b9ad0724792040ce93b749303/jeffnet/linen/layers/stochastic.py
    """
    rate: float = 0.

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        deterministic: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        if deterministic or self.rate == 0.:
            return x

        rng = self.make_rng('dropout')

        keep_prob = 1. - self.rate
        mask = jax.random.bernoulli(
            key=rng, 
            p=keep_prob, 
            shape=(x.shape[0], 1, 1, 1))
        mask = jnp.broadcast_to(
            mask, x.shape)
        return jax.lax.select(
            mask, 
            x / keep_prob, 
            jnp.zeros_like(x))