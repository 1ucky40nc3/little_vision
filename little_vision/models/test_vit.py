from typing import Tuple

import jax
import jax.numpy as jnp

from little_vision.models import vit
from little_vision.models import utils


def test_embedding(
    l: int = 32,
    d: int = 64
) -> None:
    rng = jax.random.PRNGKey(42)

    model = vit.LearnablePositionalEmbedding()
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)


def test_mlp(
    l: int = 32,
    d: int = 64,
    mlp_dim: int = 256
) -> None:
    rng = jax.random.PRNGKey(42)

    model = vit.PositionWiseMLP()
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)

    model = vit.PositionWiseMLP(
        mlp_dim=mlp_dim)
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)


def test_encoderblock(
    l: int = 32,
    d: int = 64,
) -> None:
    rng = jax.random.PRNGKey(42)

    model = vit.TransformerEncoderBlock()
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)


def test_encoder(
    l: int = 32,
    d: int = 64,
) -> None:
    rng = jax.random.PRNGKey(42)

    model = vit.TransformerEncoder()
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)


def test_vit(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = vit.ViT()
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, 10)


def test_vit_size(
    num_classes: int = 1000,
    dims: Tuple[int] = (224, 224, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = vit.ViTSmall(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 25_000_000

    model = vit.ViTBase(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 85_000_000