from typing import Any

import jax
import jax.numpy as jnp

import einops

from little_vision import metrics as little_metrics


def test_top1_acc(
    num_classes: int = 10
) -> None:
    hot = lambda a: jax.nn.one_hot(a, num_classes)

    y = jnp.array([1, 2, 3])
    x = hot(y)

    top1_acc = lambda a, b: jnp.mean(little_metrics.top1_acc(a, b))

    assert top1_acc(x, y) == 1.
    assert top1_acc(x, hot(y)) == 1.

    assert top1_acc(x, jnp.zeros(3)) == 0.
    assert top1_acc(x, hot(jnp.zeros(3))) == 0.

    assert top1_acc(x, jnp.ones(3)) == 1/3
    assert top1_acc(x, hot(jnp.ones(3))) == 1/3


def test_top5_acc(
    num_classes: int = 10
) -> None:
    hot = lambda a: jax.nn.one_hot(a, num_classes)

    y = jnp.array([1, 2, 3])
    x = hot(y)

    top5_acc = lambda a, b: jnp.mean(little_metrics.top5_acc(a, b))

    assert top5_acc(x, y) == 1.
    assert top5_acc(x, hot(y)) == 1.

    assert top5_acc(-x + 1, y) == 0.
    assert top5_acc(-x + 1, hot(y)) == 0.

    assert top5_acc(x, jnp.ones(3)) == 1/3
    assert top5_acc(x, hot(jnp.ones(3))) == 1/3

    x = jnp.array([1, 1, 1] + [0]*5 + [1, 1])
    x = einops.repeat(x, "k -> n k", n=3)
    assert top5_acc(x, y) == 2/3
    assert top5_acc(x, hot(y)) == 2/3


def test_precision_recall_f1(
    num_classes: int = 10
) -> None:
    hot = lambda a: jax.nn.one_hot(a, num_classes)

    rng = jax.random.PRNGKey(42)

    rng, key = jax.random.split(rng)
    x = jax.random.normal(key, (512, num_classes,))
    y = jnp.argmax(x, axis=-1).reshape(-1)

    metrics = little_metrics.precision_recall_f1(x, y)
    assert all(m == 1. for m in metrics)

    metrics = little_metrics.precision_recall_f1(x, hot(y))
    assert all(m == 1. for m in metrics)

    metrics = little_metrics.precision_recall_f1(x, y + 1)
    assert all(m == 0. for m in metrics)

    metrics = little_metrics.precision_recall_f1(x, y)
    assert len(metrics) == 3


def test_calc(
    num_classes: int = 10,
    devices: int = 8,
) -> None:
    hot = lambda a: jax.nn.one_hot(a, num_classes)
    def assert_value(m: Any, v: float) -> None:
        assert m["top1_acc"] == v
        assert m["top5_acc"] == v
        assert m["precision"] == v
        assert m["recall"] == v
        assert m["f1"] == v

    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)
    x = jax.random.normal(key, (devices, 512, num_classes,))
    y = jnp.argmax(x, axis=-1)
    l = jnp.mean(x, axis=-1)

    metrics = little_metrics.calc(
        [l]*10, [x]*10, [y]*10)
    assert_value(metrics, 1.)

    metrics = little_metrics.calc(
        [l]*10 + [l[:,:-10]], 
        [x]*10 + [x[:,:-10,:]], 
        [y]*10 + [y[:,:-10]])
    assert_value(metrics, 1.)

    y = hot(jnp.argmax(x, axis=-1))
    metrics = little_metrics.calc(
        [l]*10, [x]*10, [y]*10)
    assert_value(metrics, 1.)
    metrics = little_metrics.calc(
        [l]*10 + [l[:,:-10]], 
        [x]*10 + [x[:,:-10,:]], 
        [y]*10 + [y[:,:-10]])
    assert_value(metrics, 1.)

    x = -x + 1
    metrics = little_metrics.calc(
        [l]*10, [x]*10, [y]*10)
    assert_value(metrics, 0.)