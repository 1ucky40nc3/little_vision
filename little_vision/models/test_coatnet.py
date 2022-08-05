from typing import Tuple

import jax
import jax.numpy as jnp

from little_vision.models import coatnet


def test_coatnetstemblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.CoAtNetStemBlock(
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

    model = coatnet.SqueezeExcite()
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, *dims)


def test_coatnetconvblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 64)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.CoAtNetConvBlock(
        features=features)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, 28//2, 28//2, features)


def test_mlp(
    l: int = 32,
    d: int = 64,
    mlp_dim: int = 128,
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.PositionWiseMLP()
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)


def test_coatnettransformerblock(
    l: int = 32,
    d: int = 64,
    dims: Tuple[int] = (16, 16, 64)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.CoAtNetTransformerBlock(
        features=d)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, d, d)

    model = coatnet.CoAtNetTransformerBlock(
        features=d,
        strides=1)
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)


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
    num = sum(p.size for p in jax.tree_leaves(variables["params"]))
    print(num)

    assert False