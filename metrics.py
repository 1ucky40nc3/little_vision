from typing import List
from typing import Dict
from typing import Tuple
from typing import Callable
from typing import Optional

from functools import partial

import importlib

import jax
import jax.numpy as jnp

import flax
from flax import jax_utils

import ml_collections as mlc


MetricFn = Callable[[Tuple[jnp.ndarray]], jnp.ndarray]


def loss(loss: jnp.ndarray, *args, **kwargs):
    return loss


def topk_acc(
    loss: jnp.ndarray,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    k: int,
    *args,
    **kwargs
) -> jnp.ndarray:
    preds = jnp.argsort(logits)
    k_preds = preds[:, -k:]
    v_isin = jax.vmap(jnp.isin)
    return v_isin(k_preds, labels).any(axis=-1)

top1_acc: MetricFn = partial(topk_acc, k=1)
top5_acc: MetricFn = partial(topk_acc, k=5)

"""
class MetricsState(flax.struct.PyTreeNode):
    buffer: List[List[jnp.ndarray]]
    fns: Tuple[Callable] = flax.struct.field(pytree_node=False)
    names: Tuple[str]
    loss: Callable = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, *, 
        #fns: Tuple[Callable], 
        #names: Tuple[str], 
        buffer: List[List[jnp.ndarray]] = [],
        **kwargs
    ) -> "MetricsState":
        return cls(
            buffer=buffer,
            #fns=fns,
            #names=names,
            **kwargs
        )

    def update(
        self, *,
        loss: jnp.ndarray,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        grads: Optional[jnp.ndarray] = None,
        **kwargs
    ) -> "MetricsState":
        metrics = [
            fn(
                loss=loss, 
                logits=logits,
                labels=labels,
                grads=grads
            ) for fn in self.fns]
        # simulate pmean
        metrics = jax.tree_map(jnp.mean, metrics)
        print(metrics)
        self.buffer.append(metrics)
        
        return self.replace(
            buffer=self.buffer,
            **kwargs
        )

    def compute(self) -> Dict[str, jnp.ndarray]:
        print("buffer", self.buffer)
        buffer = jnp.array(self.buffer)
        print(buffer.ndim, buffer)
        buffer = buffer.T
        buffer = jnp.mean(buffer, axis=-1)
        #buffer = jax_utils.unreplicate(buffer)

        print(self.names, self.buffer)
        return {k: v for k, v in zip(self.names, buffer)}

    def reset(self, **kwargs) -> "MetricsState":
        return self.replace(
            buffer=[],
            **kwargs
        )
"""