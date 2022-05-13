from typing import Any

from functools import partial

import jax
import jax.numpy as jnp

import flax

import optax


def softmax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    **kwargs
) -> jnp.ndarray:
    labels = jax.nn.one_hot(labels, num_classes)
    loss = optax.softmax_cross_entropy(logits, labels)
    return loss.mean()