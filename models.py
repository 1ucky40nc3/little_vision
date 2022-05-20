from ast import Call
from base64 import encode
from this import d
from typing import Any
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional

from functools import partial
from click import ParamType

import jax
import jax.numpy as jnp

import flax.linen as nn

import einops
from numpy import block
from torch import dropout


DType = Any
Module = Union[partial, nn.Module]


class CNN(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
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


class ResNetBlock(nn.Module):
    features: int
    conv: Module = nn.Conv
    norm: Module = nn.BatchNorm
    act: Callable = nn.relu
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Union[int, Tuple[int, int]] = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        conv = partial(
            self.conv, 
            features=self.features,
            kernel_size=self.kernel_size)
        norm, act = self.norm, self.act

        y = conv(strides=self.strides)(x)
        y = norm()(y)
        y = act(y)
        y = conv()(y)
        y = norm(scale_init=nn.initializers.zeros)(y)

        if x.shape != y.shape:
            x = conv(
                kernel_size=(1, 1),
                strides=self.strides,
                name="conv_proj")(y)
            x = norm(
                name="norm_proj")(y)

        return act(x + y)


class BottleneckResNetBlock(nn.Module):
    features: int
    conv: Module = nn.Conv
    norm: Module = nn.BatchNorm
    act: Callable = nn.relu
    strides: Union[int, Tuple[int, int]] = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.conv(
            features=self.features, 
            kernel_size=(1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(
            features=self.features, 
            kernel_size=(3, 3), 
            strides=self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(
            features=self.features * 4,
            kernel_size=(1, 1))(y)
        y = self.norm(
            scale_init=nn.initializers.zeros)(y)

        if x.shape != y.shape:
            x = self.conv(
                features=self.features * 4,
                kernel_size=(1, 1),
                strides=self.strides,
                name="conv_proj")(x)
            x = self.norm(
                name="norm_proj")(x)
        
        return self.act(x + y)


class ResNet(nn.Module):
    stage_sizes: Tuple[int] = (2, 2, 2, 2)
    block_class: nn.Module = ResNetBlock
    num_classes: int = 10
    features: int = 64
    dtype: DType = jnp.float32
    act: Callable = nn.relu
    pool: Module = partial(
        nn.max_pool,
        window_shape=(3, 3),
        strides=(2, 2),
        padding="SAME")
    norm: Module = partial(
        nn.BatchNorm,
        momentum=0.9)
    conv: Module = partial(
        nn.Conv,
        use_bias=False)
    state_strides: Tuple[int] = (1, 2, 2, 2)

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        train: bool = True
    ) -> jnp.ndarray:
        conv = partial(
            self.conv,
            dtype=self.dtype)
        norm = partial(
            self.norm,
            use_running_average=not train,
            dtype=self.dtype)

        x = conv(
            features=self.features,
            kernel_size=(7, 7),
            strides=(2, 2),
            name="conv_root")(x)
        x = norm(
            name="norm_root")(x)
        x = self.act(x)
        x = self.pool(x)

        for i, (sizes, strides) in enumerate(
            zip(self.stage_sizes, self.state_strides)):
            for j in range(sizes):
                x = self.block_class(
                    self.features * 2**i,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                    strides=1 if j else strides)(x)
        
        x = einops.reduce(
            x, "n h w c -> n c", "mean")
        x = nn.Dense(
            self.num_classes,
            dtype=self.dtype)(x)
        
        return x

ResNet18 = ResNet
ResNet34 = partial(
    ResNet,
    stage_sizes=(3, 4, 6, 3))

ResNet50 = partial(
    ResNet,
    stage_sizes=(3, 4, 6, 3),
    block_class=BottleneckResNetBlock)
ResNet101 = partial(
    ResNet,
    stage_sizes=(3, 4, 23, 3),
    block_class=BottleneckResNetBlock)


class LearnablePositionalEmbedding(nn.Module):
    init_fn: Callable = nn.initializers.normal(stddev=0.02) 
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, l, d  = x.shape
        pos_emb = self.param(
            name="pos_emb", 
            init_fn=self.init_fn, 
            shape=(1, l, d),
            dtype=self.dtype)

        return x + pos_emb


class PointwiseMlpBlock(nn.Module):
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
        deterministic: bool = True
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
        x = dropout()(x, deterministic)
        x = dense(features=d)(x)
        x = dropout()(x, deterministic)
        
        return x



class MultiHeadScaledDotProductAttention(nn.Module):
    pass


class TransformerEncoderBlock(nn.Module):
    mlp_dim: Optional[int] = None
    num_heads: int = 4
    dropout: float = 0.
    attn_dropout: float = 0.
    norm: Module = nn.LayerNorm
    attn: Module = nn.SelfAttention
    mlp: Module = PointwiseMlpBlock
    kernel_init: Callable = nn.initializers.xavier_uniform()
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        deterministic: bool = True
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
        dropout = partial(
            nn.Dropout,
            dropout=self.dropout)
        mlp = partial(
            mlp_dim=self.mlp_dim,
            dropout=dropout,
            dtype=self.dtype)
        
        y = norm()(x)
        y = attn()(x)
        y = dropout()(y, deterministic)
        x += y
        y = norm()(x)
        y = mlp()(y, deterministic)
        x += y

        return x



class TransformerEncoder(nn.Module):
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.
    attn_dropout: float = 0.
    emb: Module = LearnablePositionalEmbedding
    norm: Module = nn.LayerNorm
    attn: Module = nn.SelfAttention
    mlp: Module = PointwiseMlpBlock

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        train: bool = True
    ) -> jnp.ndarray:
        dropout = partial(
            nn.Dropout,
            rate=self.dropout)

        x = self.emb()(x)
        x = dropout()(x, not train)

        for i in range(self.num_layers):
            x = TransformerEncoderBlock(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                name=f"encoderblock_{i}",
            )(x, not train)

        x = self.norm(name="encoder_norm")(x)

        return x


class VisionTransformer(nn.Module):
    num_classes: int
    patch_size: Tuple[int, int]
    hidden_size: int
    encoder: Module = TransformerEncoder
    classifier: str = "token"
    head_bias: float = 0.

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        train: bool = True
    ) -> jnp.ndarray:
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="vit_stem")(x)
        
        x = einops.rearrange(
            x, "n h w c -> n (h w) c")
        
        if self.classifier == "token":
            n, _, c = x.shape
            token = self.param(
                name="cls", 
                init_fn=nn.initializers.zeros,
                shape=(1, 1, c))
            token = einops.repeat(
                token, "n l d -> (i n) l d", i=n)
            x = jnp.concatenate([token, x], axis=1)

        x = self.encoder()(x, train)

        if self.classifier == "token":
            x = x[:, 0]
        else:
            x = einops.reduce(
                x, "n l d -> n d", "mean")
        
        x = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(
                self.head_bias
            ))(x)

        return x
            
