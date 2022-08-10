from typing import Tuple

from functools import partial

import jax
import jax.numpy as jnp

import flax.linen as nn

from little_vision.models import resnet
from little_vision.models import utils


def test_resnetblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    block = resnet.ResNetBlock(
        features=features,
        norm=partial(
            nn.BatchNorm, 
            use_running_average=True))
    variables = block.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    out = block.apply(
        variables, 
        jnp.ones([10, *dims]), 
        mutable=["batch_stats"])
    assert out[0].shape == (10, *dims[:-1], features)
    assert "batch_stats" in out[1]


def test_bottleneckresnetblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    block = resnet.BottleneckResNetBlock(
        features=features,
        norm=partial(
            nn.BatchNorm,
            use_running_average=False))
    variables = block.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    out, mutvars = block.apply(
        variables, 
        jnp.ones([10, *dims]),
        mutable=["batch_stats"])
    assert out.shape == (10, *dims[:-1], features * 4)
    assert "batch_stats" in mutvars

    block = resnet.BottleneckResNetBlock(
        features=features,
        norm=partial(
            nn.BatchNorm,
            use_running_average=True))
    variables = block.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    out = block.apply(
        variables, 
        jnp.ones([10, *dims]),
        mutable=False,
        deterministic=True)
    assert out.shape == (10, *dims[:-1], features * 4)


def test_resnet(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = resnet.ResNet(
        features=features)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    sample = jnp.ones([10, *dims])
    out, mutvars = model.apply(
        variables, 
        sample, 
        mutable=["batch_stats"])
    assert out.shape == (10, model.num_classes)
    assert "batch_stats" in mutvars

    out = model.apply(
        variables, 
        sample, 
        mutable=False,
        deterministic=True)
    assert out.shape == (10, model.num_classes)


def test_resnet_sizes(
    num_classes: int = 1000,
    dims: Tuple[int] = (224, 224, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = resnet.ResNet18(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 11_000_000

    model = resnet.ResNet34(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 21_000_000

    model = resnet.ResNet50(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 25_000_000