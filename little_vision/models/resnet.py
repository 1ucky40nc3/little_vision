from typing import Any
from typing import Tuple
from typing import Union
from typing import Callable

from functools import partial

import jax.numpy as jnp

import flax.linen as nn

import einops


DType = Any
Module = Union[partial, nn.Module]


class ResNetBlock(nn.Module):
    features: int
    conv: Module = nn.Conv
    norm: Module = nn.BatchNorm
    act: Callable = nn.relu
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Union[int, Tuple[int, int]] = 1

    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
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
    def __call__(
        self, 
        x: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
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
        deterministic: bool = False, 
        **kwargs
    ) -> jnp.ndarray:
        conv = partial(
            self.conv,
            dtype=self.dtype)
        norm = partial(
            self.norm,
            use_running_average=deterministic,
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