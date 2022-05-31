from typing import Tuple

from functools import partial

import jax
import jax.numpy as jnp

import flax.linen as nn

import resnet


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

    out = block.apply(variables, jnp.ones([10, *dims]))
    assert out.shape == (10, *dims[:-1], features)

test_resnetblock()


def test_bottleneckresnetblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    block = resnet.BottleneckResNetBlock(
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

    model = resnet.ResNet(
        features=features,
        norm=partial(
            nn.BatchNorm, 
            use_running_average=True))
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    sample = jnp.ones([10, *dims])
    out = model.apply(
        variables, 
        sample, 
        train=False)
    assert out.shape == (10, model.num_classes)

    out, mutated_vars = model.apply(
        variables, 
        sample, 
        train=True, 
        mutable=["batch_stats"])
    assert out.shape == (10, model.num_classes)
    assert "batch_stats" in mutated_vars.keys()

test_resnet()


def test_resnet34_size(
    num_classes: int = 1000,
    dims: Tuple[int] = (224, 224, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = resnet.ResNet34(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))

    model = resnet.ResNet50(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))

test_resnet34_size()