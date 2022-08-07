from typing import Any
from typing import Union

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import flax.linen as nn

import einops

from little_vision.models import resnet
from little_vision.models import vit


DType = Any
Module = Union[partial, nn.Module]


class Identity(nn.Module):
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        return x


class TransformerFFN(nn.Module):
    ffm: Module = partial(
        nn.Conv,
        kernel_size=(1, 1))
    act: Module = nn.gelu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        dim = x.shape[-1]
        x = self.ffm(
            features=dim * 4)(x)
        x = self.act(x)
        x = self.ffm(
            features=dim)(x)

        return x


class TransformerBlock(nn.Module):
    features: int = 64
    strides: int = 2

    norm: Module = nn.LayerNorm
    attn: Module = partial(
        nn.SelfAttention,
        num_heads=32)
    pool: Module = partial(
        nn.max_pool,
        window_shape=(2, 2),
        strides=(2, 2))
    proj: Module = partial(
        nn.Conv,
        kernel_size=(1, 1)
    )
    ffn: Module = TransformerFFN

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        h = x.shape[1]
        y = x


        y = self.norm()(y)
        y = self.pool()(y) if self.strides == 2 else y
        y = einops.rearrange(
            y, "n h w c -> n (h w) c")
        y = self.attn(
            out_features=self.features)(y)
        y = einops.rearrange(
            y, "n (h w) c -> n h w c", h=h)

        x = self.pool()(x) if self.strides == 2 else x
        x = self.proj(
            features=self.features)(x)
        
        x += y

        x = self.ffn()(x)

        return x


class CoAtNetish(nn.Module):
    num_classes: int = 10
    dtype: DType = jnp.float32

    norm: Module = partial(
        nn.BatchNorm,
        momentum=0.9)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        norm = partial(
            self.norm,
            use_running_average=deterministic,
            dtype=self.dtype)

        conv_block = partial(
            resnet.BottleneckResNetBlock,
            norm=norm
        )

        # s0
        for i in range(2):
            strides = (2, 2) if i == 0 else (1, 1)
            x = conv_block(
                features=64,
                strides=strides)(x)
        
        # s1
        for i in range(2):
            strides = (2, 2) if i == 0 else (1, 1)
            x = conv_block(
                features=96,
                strides=strides)(x)

        # s2
        for i in range(3):
            strides = (2, 2) if i == 0 else (1, 1)
            x = conv_block(
                features=192,
                strides=strides)(x)

        # s3
        for i in range(5):
            strides = (2, 2) if i == 0 else (1, 1)
            x = TransformerBlock(
                features=384,
                strides=strides)(x)

        # s4
        for i in range(2):
            strides = (2, 2) if i == 0 else (1, 1)
            x = TransformerBlock(
                features=768,
                strides=strides)(x)

        x = einops.reduce(
            x, "n h w c -> n c", "mean")
        x = nn.Dense(
            self.num_classes,
            dtype=self.dtype)(x)

        return x


if __name__ == "__main__":
    num_classes = 1000
    dims = (224, 224, 3)
    rng = jax.random.PRNGKey(42)

    model = CoAtNetish(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    num = sum(p.size for p in jax.tree_leaves(variables["params"]))
    print(num)