from typing import Callable

from functools import partial

import jax
import jax.numpy as jnp


def topk_acc(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    k: int,
) -> jnp.ndarray:
    preds = jnp.argsort(logits)
    k_preds = preds[:, -k:]
    v_isin = jax.vmap(jnp.isin)
    return v_isin(k_preds, labels).any(axis=-1)

top1_acc: Callable = partial(topk_acc, k=1)
top5_acc: Callable = partial(topk_acc, k=5)