from typing import Tuple

import jax
import jax.numpy as jnp

from little_vision.models import mlpmixer
from little_vision.models import utils


def test_mixingmlp(
    l: int = 32,
    d: int = 64,
    mlp_dim: int = 128,
) -> None:
    rng = jax.random.PRNGKey(42)

    model = mlpmixer.MixingMLP(
        mlp_dim=mlp_dim)
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)


def test_mixerblock(
    l: int = 32,
    d: int = 64,
    mlp_dim: int = 128
) -> None:
    rng = jax.random.PRNGKey(42)

    model = mlpmixer.MixerBlock(
        tokens_mlp_dim=mlp_dim,
        channels_mlp_dim=mlp_dim)
    variables = model.init(rng, jnp.ones([1, l, d]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]))
    assert out.shape == (10, l, d)

    out = model.apply(
        variables, 
        jnp.ones([10, l, d]),
        deterministic=False)
    assert out.shape == (10, l, d)


def test_mlpmixer(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = mlpmixer.MLPMixer()
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables

    out = model.apply(
        variables, 
        jnp.ones([10, *dims]))
    assert out.shape == (10, 10)


def test_mlpmixer_size(
    num_classes: int = 1000,
    dims: Tuple[int] = (224, 224, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    # mixer S/16
    model = mlpmixer.MLPMixerS(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 18_000_000

    # mixer B/16
    model = mlpmixer.MLPMixerB(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    assert utils.count_params(variables) > 59_000_000