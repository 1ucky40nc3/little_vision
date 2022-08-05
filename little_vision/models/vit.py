from typing import Any
from typing import Union
from typing import Optional
from typing import Callable

from functools import partial

import jax.numpy as jnp

import flax.linen as nn

import einops


DType = Any
Module = Union[partial, nn.Module]


class LearnablePositionalEmbedding(nn.Module):
    init_fn: Callable = nn.initializers.normal(stddev=0.02) 

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, l, d  = x.shape
        pos_emb = self.param(
            "pos_emb", 
            self.init_fn, 
            (1, l, d))

        return x + pos_emb


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


class TransformerEncoderBlock(nn.Module):
    mlp_dim: Optional[int] = None
    num_heads: int = 4
    dropout: float = 0.
    attn_dropout: float = 0.
    norm: Module = nn.LayerNorm
    attn: Module = nn.SelfAttention
    mlp: Module = PositionWiseMLP
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
            kernel_init=self.kernel_init)
        dropout = partial(
            nn.Dropout,
            rate=self.dropout)
        mlp = partial(
            self.mlp,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype=self.dtype)
        
        y = norm()(x)
        y = attn()(y, deterministic=deterministic)
        y = dropout()(y, deterministic=deterministic)
        x += y
        y = norm()(x)
        y = mlp()(y, deterministic=deterministic)
        x += y

        return x


class TransformerEncoder(nn.Module):
    num_blocks: int = 1
    num_heads: int = 8
    mlp_dim: int = 64
    dropout: float = 0.
    attn_dropout: float = 0.
    emb: Module = LearnablePositionalEmbedding
    norm: Module = nn.LayerNorm
    attn: Module = nn.SelfAttention
    mlp: Module = PositionWiseMLP

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        deterministic: bool = False
    ) -> jnp.ndarray:
        dropout = partial(
            nn.Dropout,
            rate=self.dropout)

        x = self.emb()(x)
        x = dropout()(x, deterministic=deterministic)

        for i in range(self.num_blocks):
            x = TransformerEncoderBlock(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                name=f"encoderblock_{i}",
            )(x, deterministic=deterministic)
        x = self.norm(name="encoder_norm")(x)

        return x


class ViT(nn.Module):
    num_classes: int = 10
    patch_size: int = 4
    hidden_size: int = 64
    encoder: Module = TransformerEncoder
    num_blocks: int = 1
    num_heads: int = 8
    mlp_dim: int = 256
    dropout: float = 0.
    attn_dropout: float = 0.
    classifier: str = "token"
    head_bias: float = 0.

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        deterministic: bool = False
    ) -> jnp.ndarray:
        patches = (self.patch_size, self.patch_size)
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=patches,
            strides=patches,
            padding="VALID",
            name="vit_stem")(x)
        x = einops.rearrange(
            x, "n h w c -> n (h w) c")
        
        if self.classifier == "token":
            n, _, c = x.shape
            token = self.param(
                "cls",
                nn.initializers.zeros,
                (1, 1, c))
            token = einops.repeat(
                token, "n l d -> (i n) l d", i=n)
            x = jnp.concatenate([token, x], axis=1)

        x = self.encoder(
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout
        )(x, deterministic=deterministic)

        if self.num_classes:
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
                ), name="head")(x)

        return x


ViTSmall = partial(
    ViT,
    patch_size=16,
    hidden_size=512,
    num_blocks=8,
    mlp_dim=2048,
    num_heads=8)


ViTBase = partial(
    ViT,
    patch_size=16,
    hidden_size=768,
    num_blocks=12,
    mlp_dim=3072,
    num_heads=12)