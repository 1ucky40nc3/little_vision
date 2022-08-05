from functools import partial

import jax.numpy as jnp

import flax.linen as nn

import einops


class CNN(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        **kwargs
    ) -> jnp.ndarray:
        conv = partial(
            nn.Conv, 
            kernel_size=(3, 3))
        pool = partial(
            nn.avg_pool, 
            window_shape=(2, 2), 
            strides=(2, 2))

        x = conv(features=32)(x)
        x = nn.relu(x)
        x = pool(x)
        x = conv(features=64)(x)
        x = nn.relu(x)
        x = pool(x)
        x = einops.rearrange(
            x, "n h w c -> n (h w c)")
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x
