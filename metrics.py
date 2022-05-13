from typing import Tuple
from typing import Callable
from typing import NamedTuple

from functools import partial

import jax
import jax.numpy as jnp


def loss_metric(loss: jnp.ndarray, *args):
    return loss


def topk_err_metric(
    loss: jnp.ndarray,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    k: int
) -> jnp.ndarray:
    preds = jnp.argsort(logits)
    k_preds = preds[:, -k:]
    v_isin = jax.vmap(jnp.isin)
    return v_isin(k_preds, labels).any(axis=-1)


MetricFn = Callable[[Tuple[jnp.ndarray]], jnp.ndarray]

class Metrics(NamedTuple):
    loss: MetricFn = loss_metric
    top1_err: MetricFn = partial(topk_err_metric, k=1)
    top5_err: MetricFn = partial(topk_err_metric, k=5)