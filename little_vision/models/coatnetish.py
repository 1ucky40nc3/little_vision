"""
A major part of this code is translated from https://github.com/microsoft/Swin-Transformer
"""

import jax.numpy as jnp
import flax.linen as nn



from typing import Iterable, Optional


class WindowAttention(nn.Module):
    dim: int
    num_heads: int
    window_size: Iterable[int]
    use_bias: bool = True
    att_drop: float = 0.0
    proj_drop: float = 0.0
    deterministic: Optional[bool] = None

    def get_rel_pos_index(self):
        h_indices = jnp.arange(0, self.window_size[0])
        w_indices = jnp.arange(0, self.window_size[1])
        indices = jnp.stack(jnp.meshgrid(w_indices, h_indices, indexing="ij"))
        flatten_indices = jnp.reshape(indices, (2, -1))
        relative_indices = flatten_indices[:, :, None] - flatten_indices[:, None, :]
        relative_indices = jnp.transpose(relative_indices, (1, 2, 0))
        relative_indices = relative_indices.at[:, :, 0].add(self.window_size[0] - 1)
        relative_indices = relative_indices.at[:, :, 1].add(self.window_size[1] - 1)
        relative_indices = relative_indices.at[:, :, 0].multiply(
            2 * self.window_size[1] - 1
        )
        relative_pos_index = jnp.sum(relative_indices, -1)
        return relative_pos_index

    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        rpbt = self.param(
            "relative_position_bias_table",
            nn.initializers.normal(0.02),
            (
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ),
        )

        relative_pos_index = self.variable(
            "relative_position_index", "relative_position_index", self.get_rel_pos_index
        )

        batch, n, channels = inputs.shape
        qkv = nn.Dense(self.dim * 3, use_bias=self.use_bias, name="qkv")(inputs)
        qkv = qkv.reshape(batch, n, 3, self.num_heads, channels // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * ((self.dim // self.num_heads) ** -0.5)
        att = q @ jnp.swapaxes(k, -2, -1)

        rel_pos_bias = jnp.reshape(
            rpbt[jnp.reshape(relative_pos_index.value, (-1))],
            (
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ),
        )
        rel_pos_bias = jnp.transpose(rel_pos_bias, (2, 0, 1))
        att += jnp.expand_dims(rel_pos_bias, 0)

        if mask is not None:
            att = jnp.reshape(
                att, (batch // mask.shape[0], mask.shape[0], self.num_heads, n, n)
            )
            att = att + jnp.expand_dims(jnp.expand_dims(mask, 1), 0)
            att = jnp.reshape(att, (-1, self.num_heads, n, n))
            att = nn.softmax(att)

        else:
            att = nn.softmax(att)

        att = nn.Dropout(self.att_drop)(att, deterministic)

        x = jnp.reshape(jnp.swapaxes(att @ v, 1, 2), (batch, n, channels))
        x = nn.Dense(self.dim, name="proj")(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic)
        return x


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
        WindowAttention,
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
    drop_path_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        h = x.shape[1] // 2 if self.strides == 2 else x.shape[1]

        attn = partial(
            self.attn,
            dim=self.features,
            window_size=(h, h)
        )

        if self.strides > 1:
            x = nn.Conv(
                features=self.features,
                kernel_size=(7, 7),
                strides=self.strides,
            )(x)
            x = nn.LayerNorm()(x)
        
        y = einops.rearrange(
            x, "n h w c -> n (h w) c")
        y = attn()(y, deterministic=deterministic)
        y = einops.rearrange(
            y, "n (h w) c -> n h w c", h=h)
        
        x += y

        y = self.ffn()(x)
        y = layers.DropPath(
            rate=self.drop_path_rate
        )(y, deterministic=deterministic)

        return x + y


class CoAtNetish(nn.Module):
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


CoAtNetish0 = CoAtNetish

CoAtNetish1 = partial(
    CoAtNetish,
    layers=(2, 2, 6, 14, 2))

CoAtNetish2 = partial(
    CoAtNetish,
    layers=(2, 2, 6, 14, 2),
    features=(128, 128, 256, 512, 1024))