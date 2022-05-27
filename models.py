from curses import window
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
    conv: Module = partial(nn.Conv, use_bias=False)
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
            deterministic=deterministic,
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
        y = attn()(x)
        y = dropout()(y, deterministic)
        x += y
        y = norm()(x)
        y = mlp()(y, deterministic)
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
        train: bool = True
    ) -> jnp.ndarray:
        dropout = partial(
            nn.Dropout,
            rate=self.dropout)

        x = self.emb()(x)
        x = dropout()(x, not train)

        for i in range(self.num_blocks):
            x = TransformerEncoderBlock(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                name=f"encoderblock_{i}",
            )(x, not train)
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
        train: bool = True
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
            attn_dropout=self.attn_dropout)(x, train)

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


def drop_path(
    x: jnp.ndarray, 
    drop_rate: float = 0., 
    rng=None
) -> jnp.ndarray:
    """This code has been taken from:
    https://github.com/rwightman/efficientnet-jax/blob/master/jeffnet/linen/layers/stochastic.py
    """
    if drop_rate == 0.:
        return x

    keep_prob = 1. - drop_rate
    if rng is None:
        rng = nn.make_rng()

    mask = jax.random.bernoulli(
        key=rng, 
        p=keep_prob, 
        shape=(x.shape[0], 1, 1, 1))
    mask = jnp.broadcast_to(
        mask, x.shape)
    return jax.lax.select(
        mask, 
        x / keep_prob, 
        jnp.zeros_like(x))


class DropPath(nn.Module):
    """This code has been taken from:
    https://github.com/rwightman/efficientnet-jax/blob/master/jeffnet/linen/layers/stochastic.py
    """
    rate: float = 0.

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        training: bool = False,
        rng: jax.random.PRNGKey = None
    ) -> jnp.ndarray:
        if not training or self.rate == 0.:
            return x
        if rng is None:
            rng = self.make_rng('dropout')
        return drop_path(x, self.rate, rng)


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
        training: bool = False,
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
        y = DropPath(
            rate=self.drop_path
        )(y, training)
        x += y
        y = self.norm()(x)
        y = self.mlp(
            self.channels_mlp_dim,
            name="channel_mixing")(y)
        y = DropPath(
            rate=self.drop_path
        )(y, training)
        return x + y


class MLPMixer(nn.Module):
    num_classes: int = 10
    num_blocks: int = 3
    patch_size: int = 4
    hidden_dim: int = 64
    tokens_mlp_dim: int = 256
    channels_mlp_dim: int = 256
    block: Module = MixerBlock
    norm: Module = nn.LayerNorm

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
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
                norm=self.norm,
                name=f"mixer_block_{i}")(x)
        x = self.norm(name="pre_head_norm")(x)

        if self.num_classes:
            x = einops.reduce(
                x, "n l d -> n d", "mean")
            x = nn.Dense(
                features=self.num_classes,
                kernel_init=nn.initializers.zeros,
                name="head")(x)
        
        return x


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
            stride=(2, 2))(x)
        x = nn.Conv(
            features=channels * self.expand_dims,
            kernel=(3, 3),
            stride=(1, 1))(x)
        x = nn.Conv(
            features=channels,
            kernel=(1, 1),
            stride=(1, 1))(x)

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
            stride=1)
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
    norm: Module = nn.LayerNorm
    act: Module = nn.gelu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
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

        if x.shape != y.shape:
            x = conv(
                features=y.shape[-1],
                kernel_size=(1, 1),
                strides=self.strides,
                name="conv_proj")(x)
            x = self.norm()(x)
        
        # TODO: add stochastic depth
        return x + y


"""
The code for the relative attention components was adapted from:
[1] https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html
[2] https://github.com/dqshuai/MetaFormer/blob/master/models/MHSA.py
"""
def relative_dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    relative_position_bias: jnp.ndarray,
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

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum(
        '...qhd,...khd->...hqk', 
        query, 
        key,
        precision=precision)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(
            mask, attn_weights, big_neg)

    attn_weights += relative_position_bias

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
    deterministic: Optional[bool] = None
    precision: nn.linear.PrecisionLike = None
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
        deterministic: Optional[bool] = None
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
        if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
            m_deterministic = nn.module.merge_param(
                'deterministic', 
                self.deterministic,
                deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng('dropout')
        else:
            m_deterministic = True

        bias_shape = ((
                (2 * self.image_size[0] - 1)
                 * (2 * self.image_size[1] - 1)
            ), self.num_heads)

        relative_position_bias = self.param(
            "relative_position_bias", 
            nn.initializers.normal(stddev=.02), 
            bias_shape)
        pos = relative_positions(
            self.image_size)
        bias = relative_position_bias[pos]
        bias = einops.rearrange(
            bias, "a b n -> n a b")
        bias = bias[None, :, :, :]

        x = self.attention_fn(
            query,
            key,
            value,
            bias,
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
        deterministic: Optional[bool] = None
    ) -> jnp.ndarray:
        return super().__call__(
            x, 
            x, 
            mask,
            deterministic=deterministic
        )


class CoAtNetTransformerBlock(nn.Module):
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        pass

"""
class CoAtConvBlock(nn.Module):
    expand_factor: int = 4

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        y = MBConvBlock(
            expand_factor=self.expand_factor)(x)
        
        x = nn.avg_pool(
            inputs=x,
            window_shape=(3, 3),
            strides=(2, 2),
            name="conv_pool")
        x = nn.Conv(
            features=y.shape[-1],
            kernel_size=(1, 1),
            stride=(1, 1),
            name="conv_proj")
        
        return x + y


class CoAtTransformerBlock(nn.Module):
    @nn.compact
    def __init__(
        self,
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        pool = partial(
            nn.max_pool,
            window_shape=(3, 3),
            strides=(2, 2))
        
        y = nn.LayerNorm()(x)
        y = pool(y)
        y = nn.SelfAttention(
            # pass args
        )
        
        x = pool(x)
        #x = nn.
        
"""