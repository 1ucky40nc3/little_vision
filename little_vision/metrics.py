from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Callable

from functools import partial

import jax
import jax.numpy as jnp

from sklearn import metrics

import einops


def topk_acc(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    k: int,
) -> jnp.ndarray:
    preds = jnp.argsort(logits)
    k_preds = preds[:, -k:]
    
    if labels.ndim > 1:
        labels = jnp.argmax(
            labels, axis=-1)

    v_isin = jax.vmap(jnp.isin)
    return v_isin(k_preds, labels).any(axis=-1)

top1_acc: Callable = partial(topk_acc, k=1)
top5_acc: Callable = partial(topk_acc, k=5)


def precision_recall_f1(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> Tuple[jnp.ndarray]:
    preds = jnp.argmax(
        logits, axis=-1
    ).reshape(-1)

    if labels.ndim > 1:
        labels = jnp.argmax(
            labels, axis=-1)

    classes = list(range(logits.shape[-1]))

    precision, recall, f1, _ = (
        metrics.precision_recall_fscore_support(
        labels, preds, labels=classes, average="macro"))

    return precision, recall, f1



def calc(
    loss: List[jnp.ndarray],
    logits: List[jnp.ndarray],
    labels: List[jnp.ndarray]
) -> Dict[str, float]:    
    def rearrange(a: Any):
        if a.ndim < 2:
            return a

        pattern = "d n ... -> (d n) ..."
        return jax.tree_util.tree_map(
            partial(einops.rearrange, pattern=pattern), a)

    loss, logits, labels = jax.tree_util.tree_map(
        rearrange, (loss, logits, labels))
    loss = jnp.concatenate(loss)
    logits = jnp.concatenate(logits)
    labels = jnp.concatenate(labels)

    top1 = top1_acc(logits, labels)
    top5 = top5_acc(logits, labels)

    loss, top1, top5 = jax.tree_util.tree_map(
        jnp.mean, (loss, top1, top5))

    prec, rec, f1 = precision_recall_f1(
        logits, labels)

    return dict(
        loss=loss,
        top1_acc=top1,
        top5_acc=top5,
        precision=prec,
        recall=rec,
        f1=f1
    )