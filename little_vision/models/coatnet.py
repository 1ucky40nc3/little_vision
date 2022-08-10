from typing import Any
from typing import Tuple
from typing import Union
from typing import Optional
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

"""CoAtNet implementation inspired by:
https://github.com/blakechi/ComVEX/tree/master/comvex/coatnet
"""


class ConvStem(nn.Module):
    features: int = 64
    kernel_size: Tuple[int] = (3, 3)
    strides: Optional[int] = None
    
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        conv = partial(
            nn.Conv,
            features=self.features,
            kernel_size=self.kernel_size,
            padding=1)

        x = conv(strides=2)(x)
        x = conv(strides=1)(x)

        return x


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
        deterministic: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        expansion = x.shape[-1] * self.exp_rate
        conv = partial(
            nn.Conv,
            kernel_size=(1, 1),
            use_bias=False)
        norm = partial(
            self.norm,
            use_running_average=deterministic)
        
        y = norm()(x)
        y = nn.swish(y)
        y = conv(
            features=expansion,
            strides=self.strides,
            name="expansion")(y)

        y = norm()(y)
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

        y = norm()(y)
        y = conv(
            features=self.features,
            strides=1,
            name="projection")(y)

        if self.features == x.shape[-1] and self.strides == 1:
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
    strides: int = 1
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
    drop_path_rate: float = 0.

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        h = x.shape[1] // 2 if self.strides == 2 else x.shape[1]

        y = self.norm()(x)
        y = self.pool(y) if self.strides == 2 else y
        y = einops.rearrange(
            y, "n h w c -> n (h w) c")
        y = self.attn(
            out_features=self.features)(y)
        y = einops.rearrange(
            y, "n (h w) c -> n h w c", h=h)

        x = self.pool(x) if self.strides == 2 else x
        x = self.proj(
            features=self.features)(x)
        
        x += y

        y = self.ffn()(x)
        y = layers.DropPath(
            rate=self.drop_path_rate
        )(y, deterministic=deterministic)

        return x + y


class CoAtNet(nn.Module):
    num_classes: int = 10
    layers: Tuple[int] = (2, 2, 3, 5, 2)
    features: Tuple[int] = (64, 96, 192, 384, 768)
    layout: Tuple[str] = ("S", "C", "C", "T", "T")
    stem: Module = ConvStem
    conv: Module = MBConvBlock
    trans: Module = TransformerBlock
    dtype: DType = jnp.float32

    def get_layer(
        self,
        name: str
    ) -> Module:
        return {
            "S": self.stem,
            "C": self.conv,
            "T": self.trans
        }[name]

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        assert len(self.layers) == len(self.features) == len(self.layout) == 5

        for i, (n, f, l) in enumerate(
            zip(self.layers, self.features, self.layout)):
            for j in range(n):
                strides = (2, 2) if j == 0 else (1, 1)
                x = self.get_layer(l)(
                    features=f,
                    strides=strides,
                    name=f"{l}{i}-{j}"
                )(x, deterministic=deterministic)

        x = einops.reduce(
            x, "n h w c -> n c", "mean")
        x = nn.Dense(
            self.num_classes,
            dtype=self.dtype)(x)

        return x


CoAtNet0 = CoAtNet

CoAtNet1 = partial(
    CoAtNet,
    layers=(2, 2, 6, 14, 2))

CoAtNet2 = partial(
    CoAtNet,
    layers=(2, 2, 6, 14, 2),
    features=(128, 128, 256, 512, 1024))