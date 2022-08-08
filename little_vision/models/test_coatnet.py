from typing import Tuple

import jax
import jax.numpy as jnp

from little_vision.models import coatnet
from little_vision.models import utils


def test_coatnetstemblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.ConvStem(
        features=features)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, 28//2, 28//2, features)


def test_squeezeexcite(
    dims: Tuple[int] = (28, 28, 64)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.SqueezeExcitationBlock()
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, *dims)


def test_mbconvblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 64)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.MBConvBlock(
        features=features)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, 28, 28, features)

    model = coatnet.MBConvBlock(
        features=features,
        strides=2)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, 28//2, 28//2, features)


def test_ffn(
    l: int = 32,
    d: int = 64,
    mlp_dim: int = 128,
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.TransformerFFN()
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)


def test_transformerblock(
    l: int = 32,
    d: int = 64,
    dims: Tuple[int] = (16, 16, 64)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.TransformerBlock(
        features=d)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, *dims)

    model = coatnet.TransformerBlock(
        features=d,
        strides=2)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, 8, 8, 64)


def test_coatnet(
    num_classes: int = 10,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.CoAtNet(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, 10)


def test_coatnet_size(
    num_classes: int = 1000,
    dims: Tuple[int] = (224, 224, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.CoAtNet(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 24_000_000

    model = coatnet.CoAtNet1(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 41_000_000

    model = coatnet.CoAtNet2(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 74_000_000