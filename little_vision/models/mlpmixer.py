from typing import Any
from typing import Union
from typing import Callable

from functools import partial

import jax.numpy as jnp

import flax.linen as nn

import einops

from little_vision.models import layers


DType = Any
Module = Union[partial, nn.Module]


class MixingMLP(nn.Module):
    mlp_dim: int
    dense: Module = nn.Dense
    act: Callable = nn.gelu

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        *_, d = x.shape
        x = self.dense(self.mlp_dim)(x)
        x = self.act(x)
        x = self.dense(d)(x)
        return x


class MixerBlock(nn.Module):
    tokens_mlp_dim: int
    channels_mlp_dim: int
    norm: Module = nn.LayerNorm
    mlp: Module = MixingMLP
    drop_path: float = 0.

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        y = self.norm()(x)
        y = einops.rearrange(
            y, "n l d -> n d l")
        y = self.mlp(
            self.tokens_mlp_dim,
            name="token_mixing")(y)
        y = einops.rearrange(
            y, "n d l -> n l d")
        y = layers.DropPath(
            rate=self.drop_path
        )(y, deterministic=deterministic)
        x += y
        y = self.norm()(x)
        y = self.mlp(
            self.channels_mlp_dim,
            name="channel_mixing")(y)
        y = layers.DropPath(
            rate=self.drop_path
        )(y, deterministic=deterministic)
        return x + y


class MLPMixer(nn.Module):
    num_classes: int = 10
    num_blocks: int = 3
    patch_size: int = 4
    hidden_dim: int = 64
    tokens_mlp_dim: int = 256
    channels_mlp_dim: int = 256
    drop_path: float = 0.
    block: Module = MixerBlock
    norm: Module = nn.LayerNorm

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        patches = (self.patch_size, self.patch_size)

        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=patches,
            strides=patches,
            name="stem")(x)
        x = einops.rearrange(
            x, "n h w d -> n (h w) d")
        
        for i in range(self.num_blocks):
            x = self.block(
                self.tokens_mlp_dim,
                self.channels_mlp_dim,
                drop_path=self.drop_path,
                norm=self.norm,
                name=f"mixer_block_{i}"
            )(x, deterministic=deterministic)
        x = self.norm(name="pre_head_norm")(x)

        if self.num_classes:
            x = einops.reduce(
                x, "n l d -> n d", "mean")
            x = nn.Dense(
                features=self.num_classes,
                kernel_init=nn.initializers.zeros,
                name="head")(x)
        
        return x


# size: 18.066.564
MLPMixerS = partial(
    MLPMixer,
    num_blocks=8,
    hidden_dim=512,
    patch_size=16,
    tokens_mlp_dim=256,
    channels_mlp_dim=2048)

# size: 59.188.372
MLPMixerB = partial(
    MLPMixer,
    num_blocks=12,
    hidden_dim=768,
    patch_size=16,
    tokens_mlp_dim=384,
    channels_mlp_dim=3072)