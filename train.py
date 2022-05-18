from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Iterator

from functools import partial

from absl import app
from absl import logging

import ml_collections as mlc
from ml_collections import config_flags

import wandb

import torch
import torch.utils.data as tud

import jax
import jax.numpy as jnp

from flax import jax_utils
from flax.optim import dynamic_scale
from flax.training import checkpoints
from flax.training import train_state

import models as little_models
import optimizers as little_optimizers
import losses as little_losses
import metrics as little_metrics
import datasets as little_datasets


ConfigDict = Union[mlc.ConfigDict, mlc.FrozenConfigDict]


CONFIG_FLAG = config_flags.DEFINE_config_file(
    'config', default="configs/default.py")


def do_logging(step: int, config: ConfigDict, host: int = 0) -> bool:
    return jax.process_index() == host and step > 0 and step % config.log_every == 0

def do_eval(step: int, config: ConfigDict) -> bool:
    return step > 0 and (step % config.eval_every == 0 or step == config.max_train_steps - 1)

def do_save(step: int, config: ConfigDict, host: int = 0) -> bool:
    return jax.process_index() == host and step > 0 and step % config.log_every == 0


def jaxify(tensor: torch.Tensor, config: ConfigDict) -> jnp.ndarray:
    return jax_utils.prefetch_to_device(tensor.numpy(), config.prefetch_size)


def load_ds(train: bool, config: ConfigDict) -> tud.DataLoader:
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
        wandb.log(data)


def save(
    state: train_state.TrainState, 
    config: ConfigDict
) -> None:
    state = jax_utils.unreplicate(state)
    checkpoints.save_checkpoint(
        config.save_dir, 
        state, 
        int(state.step), 
        keep=config.keep_num_saves
    )


def resume(
    state: train_state.TrainState,
    config: ConfigDict
) -> None:
    return checkpoints.restore_checkpoint(
        config.save_dir,
        state
    )

# TODO: implement for cpu and later shard
@partial(jax.jit, static_argnums=(1,), backend="cpu")
def initialize(
    rng: jnp.ndarray, 
    config: ConfigDict
) -> train_state.TrainState:
    cls = getattr(little_models, config.model.name)
    model = cls(**config.model.config)
    images = jnp.ones([1, *config.dataset.image_dims])
    params = model.init(rng, images)["params"]

    tx = little_optimizers.tx(config.optimizer)

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


@partial(
    jax.pmap, 
    axis_name="i", 
    static_broadcasted_argnums=(3,)
)
def eval_step(
    state: train_state.TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    config: ConfigDict
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
    state: train_state.TrainState,
    dataset: Iterator,
    config: ConfigDict,
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
            log(data, "Valid | ", prefix="valid_", only_std=True)

    return metrics


@partial(
    jax.pmap, 
    axis_name="i", 
    donate_argnums=(0,), 
    static_broadcasted_argnums=(3,)
)
def train_step(
    state: train_state.TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    config: ConfigDict
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
    config: ConfigDict,
    **kwargs
) -> None:
    key = jax.random.PRNGKey(config.random_seed)
    key, subkey = jax.random.split(key)

    state = initialize(subkey, config)
    if config.resume:
        state = resume(state, config)
    
    start_step = state.step
    #state = jax.tree_map(jax_utils.replicate, state)
    state = jax_utils.replicate(state)

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
            log(data, "Train | ", prefix="train_")
            tmetrics, counter = jnp.zeros((3,)), 0

        if do_eval(step, config):
            vmetrics = evaluate(state, valid_ds, config)
            vmetrics /= config.max_valid_steps + 1
            loss, top1, top5 = vmetrics
            data = dict(step=step, loss=loss, top1=top1, top5=top5)
            log(data, "Valid | ", prefix="valid_")


def main(_):
    config = mlc.FrozenConfigDict(CONFIG_FLAG.value)
    logging.info(f"New run with config: \n{config}")

    wandb.init(
        project=config.project,
        id=config.run_id or None,
        resume=config.resume or None,
        config=config.as_configdict().to_dict())
    
    train(config)


if __name__ == "__main__":
    app.run(main)


