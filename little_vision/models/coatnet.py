from typing import Any
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional

from functools import partial

import math

import jax.numpy as jnp

import flax.linen as nn

import einops

import models.layers


DType = Any
Module = Union[partial, nn.Module]


class CoAtNetStemBlock(nn.Module):
    features: int = 64
    kernel_size: Tuple[int] = (3, 3)
    strides: Tuple[int] = (2, 2)
    norm: Module = nn.LayerNorm
    act: Callable = nn.gelu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        conv = partial(
            nn.Conv,
            features=self.features,
            kernel_size=self.kernel_size)
        
        x = self.norm()(x)
        x = conv(
            strides=self.strides)(x)
        x = self.act(x)
        x = self.norm()(x)
        x = conv()(x)
        
        return x


class SqueezeExcite(nn.Module):
    squeeze_ratio: float = 0.25
    act: Module = nn.gelu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        conv = partial(
            nn.Conv,
            kernel_size=(1, 1),
            strides=1)
        c_current = x.shape[-1]
        c_squeeze = int(
            max(1, c_current * self.squeeze_ratio))
        
        x = jnp.mean(
            x, 
            axis=(1, 2),
            dtype=jnp.float32,
            keepdims=True)
        x = conv(
            features=c_squeeze,
            name="squeeze_reduce")(x)
        x = self.act(x)
        x = conv(
            features=c_current,
            name="squeeze_expand")(x)
        x = nn.sigmoid(x)

        return x


class CoAtNetConvBlock(nn.Module):
    features: int
    strides: int = 2
    expand_factor: int = 4
    squeeze_ratio: float = 0.25
    drop_path: float = 0.
    norm: Module = nn.LayerNorm
    act: Module = nn.gelu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        conv = partial(
            nn.Conv,
            use_bias=False)
        expand = int(
            self.features * self.expand_factor)

        y = conv(
            features=expand,
            kernel_size=(1, 1),
            strides=self.strides,
            name="conv_expand")(x)
        y = self.norm()(y)
        y = self.act(y)
        y = conv(
            features=expand,
            kernel_size=(3, 3),
            strides=1,
            feature_group_count=expand,
            name="conv_depthwise")(y)
        y = self.norm()(y)
        y = self.act(y)
        y = SqueezeExcite(
            squeeze_ratio=self.squeeze_ratio,
            name="squeeze")(y)
        y = conv(
            features=x.shape[-1],
            kernel_size=(1, 1),
            strides=1,
            name="conv_reduce")(y)
        y = self.norm()(y)
        y = layers.DropPath(
            rate=self.drop_path
        )(y, deterministic=deterministic)

        if x.shape != y.shape:
            x = self.norm()(x)
            x = conv(
                features=y.shape[-1],
                kernel_size=(1, 1),
                strides=self.strides,
                name="conv_proj")(x)
        
        return x + y


class PositionWiseMLP(nn.Module):
    mlp_dim: Optional[int] = None
    dropout: float = 0.
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    act: Callable = nn.gelu
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        deterministic: bool = False
    ) -> jnp.ndarray:
        dense = partial(
            nn.Dense,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype)
        dropout = partial(
            nn.Dropout,
            rate=self.dropout)
        
        *_, d = x.shape
        x = dense(features=self.mlp_dim or 4 * d)(x)
        x = self.act(x)
        x = dropout()(x, deterministic=deterministic)
        x = dense(features=d)(x)
        x = dropout()(x, deterministic=deterministic)
        
        return x

class CoAtNetTransformerBlock(nn.Module):
    features: int
    strides: int = 2
    num_heads: int = 32
    dropout: float = 0.
    drop_path: float = 0.
    norm: Module = nn.LayerNorm
    attn: Module = nn.SelfAttention
    mlp: Module = PositionWiseMLP
    kernel_init: Callable = nn.initializers.xavier_uniform()
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        deterministic: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        norm = partial(
            self.norm,
            dtype=self.dtype)
        attn = partial(
            self.attn,
            num_heads=self.num_heads,
            dtype=self.dtype,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            kernel_init=self.kernel_init)
        drop_path = partial(
            layers.DropPath,
            rate=self.dropout)
        mlp = partial(
            self.mlp,
            mlp_dim=self.features,
            dropout=self.dropout,
            dtype=self.dtype)
        pool = partial(
            nn.max_pool,
            window_shape=(3, 3),
            strides=(2, 2),
            padding="SAME")
        flatten = partial(
            einops.rearrange,
            pattern="n h w c -> n (h w) c")

        y = norm()(x)
        if self.strides != 1:
            y = pool(y)
            y = flatten(y)

            x = pool(x)
            x = flatten(x)

        y = attn()(y, deterministic=deterministic)
        y = drop_path()(y, deterministic=deterministic)
        x = x + y
        y = norm()(x)
        y = mlp()(y, deterministic=deterministic)
        y = drop_path()(y, deterministic=deterministic)

        return x + y


class CoAtNet(nn.Module):
    num_classes: int
    num_s0: int = 2
    dim_s0: int = 64
    num_s1: int = 2
    dim_s1: int = 96
    num_s2: int = 3
    dim_s2: int = 192
    num_s3: int = 5
    dim_s3: int = 384
    num_s4: int = 2
    dim_s4: int = 768

    head_bias: float = 0.

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        for i in range(self.num_s0):
            strides = 1 if i else 2
            x = CoAtNetStemBlock(
                features=self.dim_s0,
                strides=strides,
                name=f"s0_l{i}"
            )(x, deterministic=deterministic)
        for i in range(self.num_s1):
            strides = 1 if i else 2
            x = CoAtNetConvBlock(
                features=self.dim_s1,
                strides=strides,
                name=f"s1_l{i}"
            )(x, deterministic=deterministic)
        for i in range(self.num_s2):
            strides = 1 if i else 2
            x = CoAtNetConvBlock(
                features=self.dim_s2,
                strides=strides,
                name=f"s2_l{i}"
            )(x, deterministic=deterministic)
        print(x.shape)
        for i in range(self.num_s3):
            strides = 1 if i else 2
            x = CoAtNetTransformerBlock(
                features=self.dim_s3,
                strides=strides,
                name=f"s3_l{i}"
            )(x, deterministic=deterministic)
        print(x.shape)
        x = einops.rearrange(
            x, 
            "n (h w) d -> n h w d",
            h=int(math.sqrt(x.shape[-2])),
            w=int(math.sqrt(x.shape[-2])))
        print(x.shape)
        for i in range(self.num_s3):
            strides = 1 if i else 2
            x = CoAtNetTransformerBlock(
                features=self.dim_s3,
                strides=strides,
                name=f"s4_l{i}"
            )(x, deterministic=deterministic)

        x = einops.reduce(
            x, "n l d -> n d", "mean")
        x = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(
                self.head_bias
            ), name="head")(x)

        return x


"""
class CoAtNet(nn.Module):
    num_classes: int
    num_stages: Tuple[int] = (1, 1, 1, 1, 1)
    strides: int = 2
    hidden_dims: Tuple[int] = (64, 96, 192, 384, 768)
    dropout: float = 0.
    drop_path: float = 0.
    num_heads: int = 32
    head_bias: float = 0.

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        _, height, width, _ = x.shape

        for i, (l, h) in enumerate(
            zip(self.num_stages, self.hidden_dims)):
            print(l, h)
            for j in range(l):
                s = 1 if j else self.strides
                print(i, j, h, s)
                
                if i == 0:
                    print("stem")
                    x = CoAtNetStemBlock(
                        features=h,
                        strides=s,
                        name=f"S{i}_L{j}"
                    )(x, deterministic=deterministic)
                elif i in (1, 2):
                    print("conv")
                    x = CoAtNetConvBlock(
                        features=h,
                        strides=s,
                        drop_path=self.drop_path,
                        name=f"S{i}_L{j}"
                    )(x, deterministic=deterministic)
                elif i in (3, 4):
                    print("trans")
                    if i > 3 and j == 0:
                        downscale = 2**i
                        x = einops.rearrange(
                            x, 
                            "n (h w) d -> n h w d",
                            h=height // downscale,
                            w=width // downscale)
                    x = CoAtNetTransformerBlock(
                        features=h,
                        strides=s,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                        drop_path=self.drop_path,
                        name=f"S{i}_L{j}"
                    )(x, deterministic=deterministic)

        x = einops.reduce(
            x, "n l d -> n d", "mean")  
        x = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(
                self.head_bias
            ), name="head")(x)
        
        return x
"""

"""
    num_classes: int
    num_stages: Tuple[int] = (1, 1, 1, 1, 1)
    strides: int = 2
    hidden_dims: Tuple[int] = (64, 96, 192, 384, 768)
    dropout: float = 0.
    drop_path: float = 0.
    num_heads: int = 4
    head_bias: float = 0.
"""
"""
# paper: 25 m
# 2.360.910
# 2.333.870
CoAtNet0 = partial(
    CoAtNet,
    num_stages=(2, 2, 3, 5, 2),
    hidden_dims=(64, 96, 192, 384, 768),
    num_heads=32
)
# paper: 42 m
# 4.394.862
# 4.394.862
# 4.057.454
CoAtNet1 = partial(
    CoAtNet,
    num_stages=(2, 2, 6, 14, 2),
    hidden_dims=(64, 96, 192, 384, 768)
)
"""