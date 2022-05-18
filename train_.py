from typing import Any
from typing import Dict
from typing import Tuple
from typing import Iterator

from functools import partial
from itertools import cycle

from absl import app
from absl import logging

import wandb

import torch
import torch.utils.data as tud

import jax
import jax.numpy as jnp

from flax import jax_utils
from flax.optim import dynamic_scale
from flax.training.train_state import TrainState

import ml_collections as mlc

import models as little_models
import optimizers as little_optimizers
import losses as little_losses
import metrics as little_metrics
import datasets as little_datasets


def do_logging(step: int, config: mlc.ConfigDict, host: int = 0) -> bool:
    return jax.process_index() == host and step > 0 and step % config.log_every == 0

def do_eval(step: int, config: mlc.ConfigDict) -> bool:
    return step > 0 and (step % config.eval_every == 0 or step == config.max_train_steps - 1)

def do_checkpoint(step: int, config: mlc.ConfigDict, host: int = 0) -> bool:
    return jax.process_index() == host and step > 0 and step % config.log_every == 0


def jaxify(tensor: torch.Tensor, config: mlc.ConfigDict) -> jnp.ndarray:
    return jax_utils.prefetch_to_device(tensor.numpy(), config.prefetch_size)


def load_ds(train: bool, config: mlc.ConfigDict) -> tud.DataLoader:
    return getattr(little_datasets, config.dataset.name)(train=train, config=config)


def log(
    data: Dict[str, Any], 
    desc: str, 
    prefix: str, 
    only_std: bool = False
) -> None:
    data_str = []
    for k, v in data.items():
        if isinstance(v, int):
            data_str.append(f"{k}: {v}")
        else:
            data_str.append(f"{k}: {v:.4f}")
    data_str =  "; ".join(data_str)
    logging.info(f"{desc}{data_str}")

    if not only_std:
        data = {
            f"{prefix}{k}": v
            for k, v in data.items()
        }
        #wandb.log(data)


# TODO: implement for cpu and later shard
@partial(jax.jit, static_argnums=(1,), backend="cpu")
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
    static_broadcasted_argnums=(3,)
)
def eval_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    config: mlc.ConfigDict
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = getattr(little_losses, config.loss.name)(
            logits, labels, **config.loss.config)
        return loss, logits

    loss, logits = loss_fn(state.params)
    top1 = little_metrics.top1_acc(logits, labels)
    top5 = little_metrics.top5_acc(logits, labels)

    metrics = jax.lax.pmean((loss, top1, top5), axis_name="i")
    metrics = jax.tree_map(jnp.mean, metrics)

    return metrics


def evaluate(
    state: TrainState,
    dataset: Iterator,
    config: mlc.ConfigDict,
    **kwargs
) -> None:
    metrics = jnp.zeros((3,))
    for step, (images, labels) in enumerate(
        little_datasets.prepare(dataset, config)):

        bmetrics = eval_step(state, images, labels, config)
        bmetrics = jax_utils.unreplicate(bmetrics)
        metrics += jnp.array(bmetrics, dtype=jnp.float32)

        if step and step % config.log_every == 0:
            loss, top1, top5 = metrics / (step + 1)
            data = dict(step=step, loss=loss, top1=top1, top5=top5)
            log(data, "Valid | ", prefix="valid", only_std=True)

    return metrics


@partial(
    jax.pmap, 
    axis_name="i", 
    donate_argnums=(0,), 
    static_broadcasted_argnums=(3,)
)
def train_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    config: mlc.ConfigDict
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = getattr(little_losses, config.loss.name)(
            logits, labels, **config.loss.config)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    top1 = little_metrics.top1_acc(logits, labels)
    top5 = little_metrics.top5_acc(logits, labels)

    grads, loss, top1, top5 = jax.lax.pmean(
        (grads, loss, top1, top5), axis_name="i")
    state = state.apply_gradients(grads=grads)
    metrics = jax.tree_map(jnp.mean, (loss, top1, top5))

    return state, metrics


def train(
    config: mlc.ConfigDict,
    **kwargs
) -> None:
    key = jax.random.PRNGKey(config.random_seed)
    key, subkey = jax.random.split(key)

    state = train_state(subkey, config)
    # TODO: implement resuming from checkpoint
    start_step = state.step
    state = jax.tree_map(jax_utils.replicate, state)

    train_ds = load_ds(train=True, config=config)
    valid_ds = load_ds(train=False, config=config)

    tmetrics, counter = jnp.zeros((3,)), 0
    for step, (images, labels) in zip(
        range(start_step, config.max_train_steps), 
        little_datasets.prepare(train_ds, config)):

        state, metrics = train_step(state, images, labels, config)
        metrics = jax_utils.unreplicate(metrics)
        tmetrics += jnp.array(metrics, dtype=jnp.float32)
        counter += 1

        if do_logging(step, config):
            tmetrics /= counter
            loss, top1, top5 = tmetrics
            data = dict(step=step, loss=loss, top1=top1, top5=top5)
            log(data, "Train | ", prefix="train")
            tmetrics, counter = jnp.zeros((3,)), 0

        if do_eval(step, config):
            vmetrics = evaluate(state, valid_ds, config)
            print("max", config.max_valid_steps)
            vmetrics /= config.max_valid_steps + 1
            loss, top1, top5 = vmetrics
            data = dict(step=step, loss=loss, top1=top1, top5=top5)
            log(data, "Valid | ", prefix="valid")


def main(_):
    from configs import default
    train(default.get_config())



if __name__ == "__main__":
    app.run(main)


