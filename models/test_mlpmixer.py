from typing import Tuple

import jax
import jax.numpy as jnp

import mlpmixer


def test_mlpmixer_size(
    num_classes: int = 100,
    dims: Tuple[int] = (224, 224, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = mlpmixer.MLPMixerS(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))

    model = mlpmixer.MLPMixerB(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))

test_mlpmixer_size()