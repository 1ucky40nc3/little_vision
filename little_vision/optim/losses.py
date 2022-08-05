import jax
import jax.numpy as jnp

import optax


def softmax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    **kwargs
) -> jnp.ndarray:
    if labels.ndim == 1:
        labels = jax.nn.one_hot(labels, num_classes)
    print(logits.shape, labels.shape)
    loss = optax.softmax_cross_entropy(logits, labels)
    return loss.mean()