from functools import partial

import jax
import jax.numpy as jnp

import flax.linen as nn


class DropPath(nn.Module):
    rate: float = 0.

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        deterministic: bool = False
    ) -> jnp.ndarray:
        if deterministic or self.rate == 0.:
            return x

        rng = self.make_rng('dropout')
        print("drop_path", x.shape)

        keep_prob = 1. - self.rate
        mask = jax.random.bernoulli(
            key=rng, 
            p=keep_prob, 
            shape=(x.shape[0], 1, 1))
        mask = jnp.broadcast_to(
            mask, x.shape)
        return jax.lax.select(
            mask, 
            x / keep_prob, 
            jnp.zeros_like(x))