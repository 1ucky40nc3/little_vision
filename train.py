from typing import Tuple

from functools import partial

import jax
import jax.numpy as jnp

import flax
from flax.training.train_state import TrainState

import ml_collections as mlc

import models
import optimizers
import metrics
import losses


@partial(jax.pmap, static_broadcasted_argnums=(1,))
def train_state(
    rng: jnp.ndarray, 
    config: mlc.ConfigDict
) -> TrainState:
    cls = getattr(models, config.model.name)
    model = cls(**config.model.config)
    images = jnp.ones([1, *config.dataset.image_dims])
    params = model.init(rng, images)["params"]

    tx = optimizers.tx(config.optimizer)

    return TrainState(apply_fn=model.apply, params=params, tx=tx)


@jax.pmap
def update_step(
    state: TrainState,
    grads: jnp.ndarray
) -> TrainState:
    return state.apply_gradients(grads=grads)


@partial(jax.pmap, axis_name="i", static_broadcasted_argnums=(3, 4))
def train_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    metric_fns: metrics.Metrics,
    config: mlc.ConfigDict
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = getattr(losses, config.loss.name)(
            logits, labels, **config.loss.config)
        return loss, logits
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    metrics = [fn(loss, logits, labels) for fn in metric_fns]
    grads, metrics = jax.lax.pmean((grads, metrics), axis_name="i")
    metrics = jax.tree_map(jnp.mean, metrics)

    return grads, metrics


def training(
    config: mlc.ConfigDict
) -> None:
    pass
    
