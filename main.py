from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import NamedTuple

import time

from functools import partial

from dataclasses import dataclass

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

import wandb

import models as little_models
import optimizers as little_optimizers
import metrics as little_metrics
import losses as little_losses
import utils as little_utils
import datasets as little_datasets


"""class Action(NamedTuple):
    description: str
    fn: Callable[[Any], Any]
    trigger: Callable[[Union[TrainState, Any]], Any]
    data: List[Any]
"""






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

    train_metrics = little_utils.Action(
        fn=little_utils.log_metrics,
        kwargs=dict(
            metric_fns=metric_fns,
            desc="Training ({} / %d) | " % config.max_train_steps,
            prefix="train_",
        ),
        int_t=config.log_every,
        t_type=config.log_t_type,
        max_steps=config.max_train_steps
    )
    print({**config})

    evaluate_ = little_utils.Action(
        fn=evaluate,
        kwargs=dict(
            dataset=test_ds,
            config=dict(config=config),
            metric_fns=metric_fns,
        ),
        int_t=config.eval_every,
        t_type=config.eval_t_type,
        max_steps=config.max_valid_steps,
        save_data=False
    )

    for i in range(config.num_epochs):
        for j, batch in enumerate(train_ds):
            images, labels = jax.tree_map(little_utils.jaxify, batch)

            grads, metrics = train_step(state, images, labels, metric_fns, config)
            state = update(state, grads)

            metrics = jutils.unreplicate(metrics)

            step = i * config.num_steps_per_epoch + j
            
            train_metrics(step, metrics)
            evaluate_(step, state)



        #eval_metrics = evaluate(state, test_ds, config, metric_fns)

        #metrics_string = ": {:.4f}; ".join([name for name in metric_fns._fields]) + ": {:.4f}"
        """
        logging.info(f"Epoch: {i + 1}/{config.num_epochs} ~ {time.time() - t:.3f}s | "
                     f"{metrics_string.format(*epoch_metrics)} / {metrics_string.format(*eval_metrics)}")
        """


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
    metrics = jax.lax.pmean(metrics, axis_name="i")
    metrics = jax.tree_map(jnp.mean, metrics)

    return metrics


def evaluate(
    state: TrainState,
    dataset: data.DataLoader,
    config: mlc.ConfigDict,
    metric_fns: little_metrics.Metrics
) -> None:
    eval_metrics = little_utils.Action(
        fn=little_utils.log_metrics,
        kwargs=dict(
            metric_fns=metric_fns,
            desc="Evaluation ({} / %d) | " % config.max_valid_steps,
            prefix="train_",
        ),
        int_t=config.log_every,
        t_type=config.log_t_type,
        max_steps=config.max_valid_steps,
        flush_data=False
    )

    for i, batch in enumerate(dataset):
        images, labels = jax.tree_map(little_utils.jaxify, batch)

        metrics = eval_step(state, images, labels, metric_fns, config)
        metrics = jutils.unreplicate(metrics)
        #eval_metrics.append(metrics)
        eval_metrics(i, metrics, only_last=True, log_in_board=False)
    eval_metrics(i + 1, metrics)

    #eval_metrics = jnp.array(eval_metrics).T
    #eval_metrics = jnp.mean(eval_metrics, axis=-1)

    #return eval_metrics

    

def main(_):
    #wandb.init()
    from configs import default
    train(default.get_config())


if __name__ == "__main__":
    app.run(main)