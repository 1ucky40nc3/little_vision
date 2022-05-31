from typing import Tuple

import jax
import jax.numpy as jnp

import vit


def test_vit_size(
    num_classes: int = 1000,
    dims: Tuple[int] = (224, 224, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    model = vit.ViTSmall(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))

    model = vit.ViTBase(
        num_classes=num_classes)
    variables = model.init(rng, jnp.ones([1, *dims]))
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))

test_vit_size()