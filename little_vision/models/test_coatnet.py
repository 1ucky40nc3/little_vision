from typing import Tuple

import jax
import jax.numpy as jnp

import coatnet


def test_coatnet_size(
    num_classes: int = 1000,
    dims: Tuple[int] = (224, 224, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = coatnet.CoAtNet(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))

test_coatnet_size()