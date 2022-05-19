from typing import Any
from typing import Tuple
from typing import Union
from typing import Callable

from functools import partial

import jax
import jax.numpy as jnp

import flax.linen as nn

import einops
from numpy import block


DType = Any
Module = Union[partial, nn.Module]


class CNN(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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

        r = x

        x = conv(strides=self.strides)(x)
        x = norm()(x)
        x = act(x)
        x = conv()(x)
        x = norm(scale_init=nn.initializers.zeros)(x)

        if r.shape != x.shape:
            r = conv(
                kernel_size=(1, 1),
                strides=self.strides,
                name="conv_proj")(r)
            r = norm(
                name="norm_proj")(r)

        return act(r + x)


class BottleneckResNetBlock(nn.Module):
    features: int
    conv: Module = nn.Conv
    norm: Module = nn.BatchNorm
    act: Callable = nn.relu
    strides: Union[int, Tuple[int, int]] = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        r = x
        
        x = self.conv(
            features=self.features, 
            kernel_size=(1, 1))(x)
        x = self.norm()(x)
        x = self.act(x)
        x = self.conv(
            features=self.features, 
            kernel_size=(3, 3), 
            strides=self.strides)(x)
        x = self.norm()(x)
        x = self.act(x)
        x = self.conv(
            features=self.features * 4,
            kernel_size=(1, 1))(x)
        x = self.norm(
            scale_init=nn.initializers.zeros)(x)

        if r.shape != x.shape:
            r = self.conv(
                features=self.features * 4,
                kernel_size=(1, 1),
                strides=self.strides,
                name="conv_proj")(r)
            r = self.norm(
                name="conv_proj")(r)
        
        return self.act(r + x)


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
        stride=(2, 2),
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
            stride=(2, 2),
            name="conv_root")(x)
        x = norm(
            name="norm_root")(x)
        x = self.act(x)
        x = self.pool(x)

        for i, (sizes, strides) in enumerate(
            zip(self.stage_sizes, self.state_strides)):
            for j in range(sizes):
                x = self.block_cls(
                    self.num_filters * 2**i,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                    strides=1 if j else strides)(x)
        
        x = jnp.mean(x, axis=(1, 2))
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
    block_cls=BottleneckResNetBlock)
ResNet101 = partial(
    ResNet,
    stage_sizes=(3, 4, 23, 3),
    block_cls=BottleneckResNetBlock)