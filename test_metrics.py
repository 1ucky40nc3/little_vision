import jax
import jax.numpy as jnp

import metrics as little_metrics


def test_precision_recall_f1(
    num_classes: int = 10
) -> None:
    rng = jax.random.PRNGKey(42)

    rng, key = jax.random.split(rng)
    x = jax.random.normal(key, (512, num_classes,))
    
    rng, key = jax.random.split(rng)
    y = jax.random.normal(key, (512, num_classes))
    y = jnp.argmax(y, axis=-1).reshape(-1)

    metrics = little_metrics.precision_recall_f1(
        x, jnp.argmax(x, axis=-1).reshape(-1))
    assert all(m == 1. for m in metrics)

    metrics = little_metrics.precision_recall_f1(
        x, jnp.argmax(x, axis=-1).reshape(-1) + 1)
    assert all(m == 0. for m in metrics)

    metrics = little_metrics.precision_recall_f1(x, y)
    assert len(metrics) == len(set(metrics))

test_precision_recall_f1()