from ast import Call
from curses import window
import math
from re import S
from typing import Any
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import flax.linen as nn

import einops
from numpy import expand_dims, indices, squeeze
from torch import dropout


DType = Any
Module = Union[partial, nn.Module]























class MBConvBlock(nn.Module):
    expand_factor: int = 4

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        *_, channels = x.shape
        x = nn.BatchNorm()(x)
        x = nn.Conv(
            features=channels,
            kernel=(1, 1),
            strides=(2, 2))(x)
        x = nn.Conv(
            features=channels * self.expand_dims,
            kernel=(3, 3),
            strides=(1, 1))(x)
        x = nn.Conv(
            features=channels,
            kernel=(1, 1),
            strides=(1, 1))(x)

        return x




        




def relative_dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[jnp.ndarray] = None,
    dropout_rate: float = 0.,
    deterministic: bool = False,
    dtype: Optional[DType] = None,
    precision: nn.linear.PrecisionLike = None
) -> jnp.ndarray:
    query, key, value = nn.dtypes.promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        'q, k, v batch dims must match.')
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        'q, k, v num_heads must match.')
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

    attn_weights = jnp.einsum(
        '...qhd,...khd->...hqk', 
        query, 
        key,
        precision=precision)

    if bias is not None:
        attn_weights = attn_weights + bias

    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(
            mask, attn_weights, big_neg)

    # normalize the attention weights
    attn_weights = jax.nn.softmax(
        attn_weights).astype(dtype)

    # apply attention dropout
    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        multiplier = (keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier

    # return weighted sum over values for each query position
    return jnp.einsum(
        '...hqk,...khd->...qhd', 
        attn_weights, 
        value,
        precision=precision)


def relative_positions(
    image_size: Tuple[int] = (32, 32),
) -> np.ndarray:
    h, w = image_size
    y = np.arange(h)
    x = np.arange(w)

    coords = np.meshgrid(y, x, indexing="ij")
    coords = np.stack(coords)
    coords = einops.rearrange(
        coords, "n h w -> n (h w)")

    relative_coords = coords[:, :, None] - coords[:, None, :]
    relative_coords = einops.rearrange(
        relative_coords, "n a b -> a b n")
    relative_coords[:, :, 0] += h - 1
    relative_coords[:, :, 1] += w - 1
    relative_coords[:, :, 0] *= 2 * w - 1

    positions = einops.reduce(
        relative_coords, "a b n -> a b", "sum")
    
    return np.int32(positions)


class RelativeMultiHeadDotProductAttention(nn.Module):
    num_heads: int
    image_size: Tuple[int] = (32, 32)
    dtype: Optional[DType] = None
    param_dtype: DType = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.
    deterministic: bool = False
    precision: nn.linear.PrecisionLike = None
    kernel_init: Callable = nn.linear.default_kernel_init
    bias_init: Callable = nn.initializers.zeros
    use_bias: bool = True
    attention_fn: Callable = relative_dot_product_attention
    decode: bool = False

    @nn.compact
    def __call__(
        self,
        inputs_q: jnp.ndarray,
        inputs_kv: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        dense = partial(
            nn.DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision)
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (
            dense(name='query')(inputs_q),
            dense(name='key')(inputs_kv),
            dense(name='value')(inputs_kv))

        dropout_rng = None
        if self.dropout_rate > 0.:
            m_deterministic = nn.module.merge_param(
                'deterministic', 
                self.deterministic,
                deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng('dropout')
        else:
            m_deterministic = True

        size = int(math.sqrt(
            inputs_q.shape[-2]))
        bias_shape = ((
                (2 * size - 1)
                 * (2 * size - 1)
            ), self.num_heads)

        relative_position_bias = self.param(
            "relative_position_bias", 
            nn.initializers.normal(stddev=.02), 
            bias_shape)
        pos = relative_positions(
            (size, size))
        bias = relative_position_bias[pos]
        bias = einops.rearrange(
            bias, "a b n -> n a b")
        bias = bias[None, :, :, :]

        x = self.attention_fn(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision)
        # back to the original inputs dimensions
        out = nn.DenseGeneral(
            features=features,
            axis=(-2, -1),
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name='out')(x)
        return out

class RelativeSelfAttention(RelativeMultiHeadDotProductAttention):
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        return super().__call__(
            x, 
            x, 
            mask,
            deterministic=deterministic
        )





