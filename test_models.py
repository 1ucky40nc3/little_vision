from typing import Tuple

from functools import partial

import jax
import jax.numpy as jnp

import flax.linen as nn

import models

from typing import Any
from typing import Callable


def tree_equal(a: Any, b: Any) -> bool:
    a_values, a_treedef = jax.tree_util.tree_flatten(a)
    b_values, b_treedef = jax.tree_util.tree_flatten(b)
    
    return (
        a_treedef == b_treedef
        and all([
            (i == j).all() 
            for i, j in zip(a_values, b_values)
        ])
    )


def test_resnetblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    block = models.ResNetBlock(
        features=features,
        norm=partial(
            nn.BatchNorm, 
            use_running_average=True))
    variables = block.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    out = block.apply(variables, jnp.ones([10, *dims]))
    assert out.shape == (10, *dims[:-1], features)
    
    
test_resnetblock()


def test_bottleneckresnetblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    block = models.BottleneckResNetBlock(
        features=features,
        norm=partial(
            nn.BatchNorm, 
            use_running_average=True))
    variables = block.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    out = block.apply(variables, jnp.ones([10, *dims]))
    assert out.shape == (10, *dims[:-1], features * 4)

    
test_bottleneckresnetblock()


def test_resnet(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    resnet = models.ResNet(
        features=features,
        norm=partial(
            nn.BatchNorm, 
            use_running_average=True))
    variables = resnet.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    sample = jnp.ones([10, *dims])
    out = resnet.apply(
        variables, 
        sample, 
        train=False)
    assert out.shape == (10, resnet.num_classes)

    out, mutated_vars = resnet.apply(
        variables, 
        sample, 
        train=True, 
        mutable=["batch_stats"])
    assert out.shape == (10, resnet.num_classes)
    assert "batch_stats" in mutated_vars.keys()


test_resnet()


def test_conv(
    features: int = 64,
    kernel_size: Tuple[int, int] = (3, 3), 
    strides: Tuple[int, int] = (1, 1),
    dims: Tuple[int, int, int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    conv0 = nn.Conv(
        features=features, 
        kernel_size=kernel_size)
    conv1 = nn.Conv(
        features=features,
        kernel_size=kernel_size,
        strides=strides)
    
    sample = jnp.ones([1, *dims])
    variables0 = conv0.init(rng, sample)
    variables1 = conv1.init(rng, sample)

    out0 = conv0.apply(variables0, sample)
    out1 = conv1.apply(variables1, sample)

    assert (out0 == out1).all()

test_conv()

    