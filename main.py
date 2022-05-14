from typing import Tuple

import time

from functools import partial


from absl import app
from absl import flags
from absl import logging

import torch.utils.data as data

import jax
import jax.numpy as jnp

import flax
from flax.training.train_state import TrainState
from flax import jax_utils as jutils

import ml_collections as mlc

import models as little_models
import optimizers as little_optimizers
import metrics as little_metrics
import losses as little_losses
import utils as little_utils
import datasets as little_datasets


# TODO: implement for cpu and later shard
@partial(jax.pmap, static_broadcasted_argnums=(1,))
def train_state(
    rng: jnp.ndarray, 
    config: mlc.ConfigDict
) -> TrainState:
    cls = getattr(little_models, config.model.name)
    model = cls(**config.model.config)
    images = jnp.ones([1, *config.dataset.image_dims])
    params = model.init(rng, images)["params"]

    tx = little_optimizers.tx(config.optimizer)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.pmap
def update(
    state: TrainState,
    grads: jnp.ndarray
) -> TrainState:
    return state.apply_gradients(grads=grads)


@partial(jax.pmap, axis_name="i", static_broadcasted_argnums=(3, 4))
def train_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    metric_fns: little_metrics.Metrics,
    config: mlc.ConfigDict
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = getattr(little_losses, config.loss.name)(
            logits, labels, **config.loss.config)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    metrics = [fn(loss, logits, labels) for fn in metric_fns]
    grads, metrics = jax.lax.pmean((grads, metrics), axis_name="i")
    metrics = jax.tree_map(jnp.mean, metrics)

    return grads, metrics


def train(
    config: mlc.ConfigDict
) -> None:
    num_devices = jax.local_device_count()
    key = jax.random.PRNGKey(config.random_seed)
    key, subkey = jax.random.split(key)

    ds_cls = getattr(little_datasets, config.dataset.name)
    train_ds = ds_cls(train=True, config=config)
    test_ds = ds_cls(train=False, config=config)

    subkeys = jax.random.split(subkey, num_devices)
    state = train_state(subkeys, config)

    metric_fns = little_metrics.Metrics()

    for epoch in range(config.num_epochs):
        epoch_metrics = []
        t = time.time()

        for batch in train_ds:
            images, labels = jax.tree_map(little_utils.jaxify, batch)

            grads, metrics = train_step(state, images, labels, metric_fns, config)
            state = update(state, grads)

            metrics = jutils.unreplicate(metrics)
            epoch_metrics.append(metrics)

        epoch_metrics = jnp.array(epoch_metrics).T
        epoch_metrics = jnp.mean(epoch_metrics, axis=-1)
        epoch_string = ": {:.4f}; ".join([name for name in metric_fns._fields]) + ": {:.4f}"
        logging.info(f"Epoch: {epoch + 1}/{config.num_epochs} ~ {time.time() - t:.3f}s | "
                     f"{epoch_string.format(*epoch_metrics)}")


@partial(jax.pmap, axis_name="i", static_broadcasted_argnums=(3, 4))
def eval_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    metric_fns: little_metrics.Metrics,
    config: mlc.ConfigDict
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = getattr(little_losses, config.loss.name)(
            logits, labels, **config.loss.config)
        return loss, logits

    loss, logits = loss_fn(state.params)

    metrics = [fn(loss, logits, labels) for fn in metric_fns]
    grads, metrics = jax.lax.pmean((grads, metrics), axis_name="i")
    metrics = jax.tree_map(jnp.mean, metrics)

    return metrics


def evaluate(
    config: mlc.ConfigDict,
    dataset: data.DataLoader,
    logger: little_utils.Writer
) -> None:
    pass
    

def main(_):
    from configs import default
    train(default.get_config())


if __name__ == "__main__":
    app.run(main)