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

    print("labels m", labels.shape)
    if labels.ndim > 1:
        labels = jnp.argmax(
            labels, axis=-1)

    classes = list(range(logits.shape[-1]))
    print(labels.shape, preds.shape, classes)
    print(labels)
    print(preds)
    precision, recall, f1, _ = (
        metrics.precision_recall_fscore_support(
        labels, preds, labels=classes, average="macro"))

    return precision, recall, f1


def calc(
    loss: jnp.ndarray,
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> Dict[str, float]:
    print(logits[0].shape, labels[0].shape)
    loss = jnp.stack(loss)
    logits = jnp.column_stack(logits)
    labels = jnp.column_stack(labels)
    print("stack", loss.shape, logits.shape, labels.shape)

    loss, logits, labels = jax.tree_map(
        partial(einops.rearrange, pattern="d n ... -> (d n) ..."),
        (loss, logits, labels))

    print("metrics", logits.shape, labels.shape)

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


a = jnp.ones((1, 512, 10))
b = jnp.ones((1, 200, 10))

print(jnp.column_stack([a, b]).shape)



x = jnp.array([0, 0, 1])
x_ = jnp.array(2)
y = jnp.array([0, 1, 0])
n = 2
d = 1
x = jnp.stack([x]*n)
x = jnp.stack([x]*d)
print("x", x.shape)

x_ = jnp.stack([x_]*n)
x_ = jnp.stack([x_]*d)
print("x_", x_.shape)

y = jnp.stack([y]*n)
y = jnp.stack([y]*d)

l = jnp.ones([1])
b = 10
print("true")
print(calc([l]*b, [x]*b, [x]*b))
print("eieieieieiei")
print(calc([l]*b, [x]*b, [x_]*b))
print("false")
print(calc([l]*b, [x]*b, [y]*b))
