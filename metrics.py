from typing import Tuple
from typing import Callable
from typing import NamedTuple

from functools import partial

import importlib

import jax
import jax.numpy as jnp

import ml_collections as mlc


MetricFn = Callable[[Tuple[jnp.ndarray]], jnp.ndarray]


def loss(loss: jnp.ndarray, *args):
    return loss


def topk_acc(
    loss: jnp.ndarray,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    k: int
) -> jnp.ndarray:
    preds = jnp.argsort(logits)
    k_preds = preds[:, -k:]
    v_isin = jax.vmap(jnp.isin)
    return v_isin(k_preds, labels).any(axis=-1)

top1_acc: MetricFn = partial(topk_acc, k=1)
top5_acc: MetricFn = partial(topk_acc, k=5)