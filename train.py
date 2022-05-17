from typing import Tuple
from typing import Optional


from functools import partial

from dataclasses import dataclass

import torch
import torch.utils.data as tud

import jax
import jax.numpy as jnp

import einops

import flax
from flax import struct
from flax import jax_utils
from flax.optim import dynamic_scale
from flax.training.train_state import TrainState

import ml_collections as mlc

import models as little_models
import optimizers as little_optimizers
import metrics as little_metrics
import losses as little_losses
import actions as little_actions


def collect_metrics(config: mlc.ConfigDict) -> Tuple[little_metrics.MetricFn]:
    return tuple(getattr(little_metrics, name) for name in config.metrics.names)


def shard(array: jnp.ndarray) -> jnp.array:
    return einops.rearrange(
        array, 
        "(d n) ... -> d n ...", 
        d=jax.local_device_count())


# TODO: maybe shard with flax.jax_utils.replicate
def jaxify(tensor: torch.Tensor) -> jnp.ndarray:
    return shard(tensor.numpy())


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


@partial(
    jax.pmap, 
    axis_name="i", 
    donate_argnums=(0,), 
    static_broadcasted_argnums=(3, 4)
)
def train_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    metric_fns: Tuple[little_metrics.MetricFn],
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
    state = state.apply_gradients(grads=grads)
    metrics = jax.tree_map(jnp.mean, metrics)

    return state, metrics


@partial(
    jax.pmap, 
    axis_name="i", 
    static_broadcasted_argnums=(3, 4)
)
def eval_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    metric_fns: Tuple[little_metrics.MetricFn],
    config: mlc.ConfigDict
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = getattr(little_losses, config.loss.name)(
            logits, labels, **config.loss.config)
        return loss, logits

    loss, logits = loss_fn(state.params)
    metrics = [fn(loss, logits, labels) for fn in metric_fns]

    metrics = jax.lax.pmean(metrics, axis_name="i")
    metrics = jax.tree_map(jnp.mean, metrics)

    return metrics


def train(
    config: mlc.ConfigDict,
    dataset: tud.DataLoader,
    actions: Tuple[little_actions.Action] = (),
    **kwargs
) -> None:
    num_devices = jax.local_device_count()
    key = jax.random.PRNGKey(config.random_seed)
    key, subkey = jax.random.split(key)

    subkeys = jax.random.split(subkey, num_devices)
    state = train_state(subkeys, config)

    metric_fns = collect_metrics(config)

    num = config.num_steps_per_epoch
    for i in range(config.num_epochs):
        for j, batch in enumerate(dataset):
            images, labels = jax.tree_map(jaxify, batch)

            state, metrics = train_step(state, images, labels, metric_fns, config)
            metrics, step = jax.tree_map(jax_utils.unreplicate, (metrics, state.step))

            for action in actions:
                action(
                    step=step,
                    index=i*num + j,
                    update=metrics, 
                    state=state)


def evaluate(
    state: TrainState,
    config: mlc.ConfigDict,
    dataset: tud.DataLoader,
    actions: Tuple[little_actions.Action],
    **kwargs
) -> None:
    step = jax_utils.unreplicate(state.step)
    metric_fns = collect_metrics(config)
    print("WOWOWOWOWO")

    for index, batch in enumerate(dataset):
        #images, labels = jax.tree_map(jaxify, batch)
        imags, labels = batch

        metrics = eval_step(state, images, labels, metric_fns, config)
        metrics = jax_utils.unreplicate(metrics)
        
        print("index", index)
        for action in actions:
            last_batch = index == config.max_valid_steps
            action(
                step=step,
                index=index,
                update=metrics, 
                only_std=not last_batch, 
                clear_buffer=last_batch,
                reset_index=last_batch)
            # TODO: put global index into action for (wandb logging)