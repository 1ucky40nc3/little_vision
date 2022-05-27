from typing import Dict
from typing import Tuple
from typing import Callable

from functools import partial

import jax
import jax.numpy as jnp

from sklearn import metrics


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


def precision_recall_f1(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> Tuple[jnp.ndarray]:
    preds = jnp.argmax(
        logits, axis=-1
    ).reshape(-1)

    precision, recall, f1, _ = (
        metrics.precision_recall_fscore_support(
        labels, preds, average="macro"))

    return precision, recall, f1


def calc(
    loss: jnp.ndarray,
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> Dict[str, float]:
    loss = jnp.stack(loss)
    logits, labels = map(
        jnp.column_stack, 
        (logits, labels))
    loss, logits, labels = map(
        jnp.squeeze,
        (loss, logits, labels))

    top1 = top1_acc(logits, labels)
    top5 = top5_acc(logits, labels)

    loss, top1, top5 = jax.tree_map(
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
