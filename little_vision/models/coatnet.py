from typing import Any
from typing import Tuple
from typing import Union
from typing import Callable

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import flax.linen as nn

import einops

from little_vision.models import resnet
from little_vision.models import vit
from little_vision.models import layers


DType = Any
Module = Union[partial, nn.Module]


def global_avg_pool(
    x: jnp.ndarray
) -> jnp.ndarray:
    return jnp.mean(
        x, axis=(1, 2), keepdims=True)


class SqueezeExcitationBlock(nn.Module):
    se_rate: int = 0.25
    pool: Callable = global_avg_pool
    conv: Module = partial(
        nn.Conv,
        kernel_size=(1, 1),
        strides=1,
        use_bias=True)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        reduced = int(x.shape[-1] * self.se_rate)
        
        y = self.pool(x)
        y = self.conv(
            features=reduced)(y)
        y = nn.relu(y)
        y = self.conv(
            features=x.shape[-1])(y)
        y = nn.sigmoid(y)

        return x * y 


class MBConvBlock(nn.Module):
    features: int = 64
    exp_rate: int = 4
    kernel_size: Tuple[int] = (3, 3)
    strides: int = 1
    se_rate: float = 0.25
    norm: Module = nn.BatchNorm
    act: Callable = nn.swish
    drop_path_rate: float = 0.

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        expansion = x.shape[-1] * self.exp_rate
        conv = partial(
            nn.Conv,
            kernel_size=(1, 1),
            use_bias=False)
        
        y = self.norm(
            use_running_average=deterministic)(x)
        y = nn.swish(y)
        y = conv(
            features=expansion,
            strides=self.strides,
            name="expansion")(y)

        y = self.norm(
            use_running_average=deterministic)(y)
        y = nn.swish(y)
        y = nn.Conv(
            features=expansion,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            feature_group_count=expansion,
            use_bias=False,
            name="depthwise")(y)

        if self.se_rate > 0.:
            y = SqueezeExcitationBlock(
                se_rate=self.se_rate)(y)

        y = self.norm(
            use_running_average=deterministic)(y)
        y = conv(
            features=self.features,
            strides=1,
            name="projection")(y)

        if self.features == x.shape[-1]:
            y = layers.DropPath(
                rate=self.drop_path_rate
            )(y, deterministic=deterministic)
            y += x

        return y


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


class CoAtNet(nn.Module):
    num_classes: int = 10
    c_block: Module = partial(
        MBConvBlock,
        se_rate=0.)
    t_block: Module = partial(
        TransformerBlock)
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        # s0
        for i in range(2):
            strides = (2, 2) if i == 0 else (1, 1)
            x = MBConvBlock(
                features=64,
                strides=strides)(x)
        
        # s1
        for i in range(2):
            strides = (2, 2) if i == 0 else (1, 1)
            x = self.c_block(
                features=96,
                strides=strides)(x)

        # s2
        for i in range(3):
            strides = (2, 2) if i == 0 else (1, 1)
            x = self.c_block(
                features=192,
                strides=strides)(x)

        # s3
        for i in range(5):
            strides = (2, 2) if i == 0 else (1, 1)
            x = self.t_block(
                features=384,
                strides=strides)(x)

        # s4
        for i in range(2):
            strides = (2, 2) if i == 0 else (1, 1)
            x = self.t_block(
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

    model = CoAtNet(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    num = sum(p.size for p in jax.tree_leaves(variables["params"]))
    print(num)
