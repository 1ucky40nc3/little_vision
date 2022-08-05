from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Iterator
from typing import Optional

import os

from functools import partial

import multiprocessing

from absl import app
from absl import logging

import ml_collections as mlc
from ml_collections import config_flags

import wandb

import torch
import torch.utils.data as tud

import jax
import jax.numpy as jnp

import flax
from flax import jax_utils
from flax.optim import dynamic_scale
from flax.training import checkpoints
from flax.training import train_state

import einops

import models as little_models
import optimizers as little_optimizers
import losses as little_losses
import metrics as little_metrics
import data.datasets as little_datasets


ConfigDict = Union[mlc.ConfigDict, mlc.FrozenConfigDict]


CONFIG_FLAG = config_flags.DEFINE_config_file(
    'config', default="configs/default.py")


def do_logging(
    step: int, 
    config: ConfigDict, 
    host: int = 0
) -> bool:
    return (
        jax.process_index() == host 
        and step > 0 
        and step % config.log_every == 0
    )

def do_eval(
    step: int, 
    config: ConfigDict
) -> bool:
    return (
        step > 0 
        and (
            step % config.eval_every == 0 
            or step == config.max_train_steps - 1
        )
    )

def do_save(
    step: int, 
    config: ConfigDict, 
    host: int = 0
) -> bool:
    return (
        jax.process_index() == host 
        and step > 0 
        and (
            (step + 1) % config.save_every == 0
            or step == config.max_train_steps - 1
        )
    )


def jaxify(
    tensor: torch.Tensor, 
    config: ConfigDict
) -> jnp.ndarray:
    return jax_utils.prefetch_to_device(
        tensor.numpy(), config.prefetch_size)


def load_ds(
    train: bool, 
    config: ConfigDict
) -> tud.DataLoader:
    return getattr(
        little_datasets, 
        config.dataset.name
    )(train=train, config=config)


def log(
    step: int,
    data: Dict[str, Any], 
    desc: str, 
    prefix: str, 
    only_std: bool = False
) -> None:
    data_str = [f"{k}: {v:.5f}" for k, v in data.items()]
    data_str =  "; ".join((f"step: {step:7d}", *data_str))
    logging.info(f"{desc}{data_str}")

    if not only_std:
        data = {
            f"{prefix}{k}": v
            for k, v in data.items()
        }
        wandb.log(data, step=step)


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




class TrainState(train_state.TrainState):
    batch_stats: flax.core.frozen_dict.FrozenDict


@partial(jax.jit, static_argnums=(1,), backend="cpu")
def initialize(
    rng: jnp.ndarray, 
    config: ConfigDict
) -> TrainState:
    cls = getattr(little_models, config.model.name)
    model = cls(**config.model.config)
    images = jnp.ones([1, *config.dataset.image_dims])
    variables = model.init(rng, images, deterministic=True)

    tx = little_optimizers.tx(config.optimizer)

    return TrainState.create(
        tx=tx,
        apply_fn=model.apply, 
        params=variables.get("params"),
        batch_stats=variables.get("batch_stats") or dict()
    )


@partial(
    jax.pmap, 
    axis_name="i", 
    static_broadcasted_argnums=(3,)
)
def eval_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    config: ConfigDict
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    def loss_fn(params):
        logits = state.apply_fn({
            "params": params,
            "batch_stats": state.batch_stats
        }, images, deterministic=True)
        loss = getattr(little_losses, config.loss.name)(
            logits, labels, **config.loss.config)
        return loss, logits

    loss, logits = loss_fn(state.params)
    return loss, logits


def evaluate(
    state: TrainState,
    dataset: Iterator,
    config: ConfigDict,
    **kwargs
) -> None:
    vloss, vlogits, vlabels = [], [], []
    for step, (images, labels) in enumerate(
        little_datasets.prepare(dataset, config)):

        loss, logits = eval_step(state, images, labels, config)
        vloss.append(loss)
        vlogits.append(logits)
        vlabels.append(labels)

        if step and step % config.log_every == 0:
            data = little_metrics.calc(vloss, vlogits, vlabels)
            log(step, data, "Valid | ", prefix="valid_", only_std=True)

    return vloss, vlogits, vlabels


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
    config: ConfigDict,
    rngs: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    rngs = jax.random.fold_in(rngs, state.step)

    def loss_fn(params):
        logits, mutvars = state.apply_fn({
                "params": params,
                "batch_stats": state.batch_stats
            }, 
            images, 
            mutable=["batch_stats"], 
            rngs=dict(dropout=rngs),
            deterministic=False)
        loss = getattr(little_losses, config.loss.name)(
            logits, labels, **config.loss.config)
        return loss, (logits, mutvars)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, mutvars)), grads = grad_fn(state.params)

    grads  = jax.lax.pmean(grads, axis_name="i")
    state = state.apply_gradients(
        grads=grads, batch_stats=mutvars["batch_stats"])

    return state, (loss, logits)


def train(
    config: ConfigDict,
    **kwargs
) -> None:
    pool = multiprocessing.pool.ThreadPool()

    key = jax.random.PRNGKey(config.random_seed)
    key, subkey = jax.random.split(key)
    print("i")

    state = initialize(subkey, config)
    #print(state.opt_state)
    if config.resume:
        state = resume(state, config)
    
    start_step = state.step
    state = jax_utils.replicate(state)

    train_ds = load_ds(train=True, config=config)
    valid_ds = load_ds(train=False, config=config)

    keys = jax.random.split(key, jax.local_device_count())
    print("wow")

    tloss, tlogits, tlabels, = [], [], []
    for step, (images, labels) in zip(
        range(start_step, config.max_train_steps), 
        little_datasets.prepare(train_ds, config)):

        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            state, (loss, logits) = train_step(state, images, labels, config, keys)

        tloss.append(loss)
        tlogits.append(logits)
        tlabels.append(labels)

        if do_logging(step, config):
            data = little_metrics.calc(tloss, tlogits, tlabels)
            log(step, data, "Train | ", "train_")
            tloss, tlogits, tlabels = [], [], []

        if do_eval(step, config):
            vmetrics = evaluate(state, valid_ds, config)
            data = little_metrics.calc(*vmetrics)
            log(step, data, "Valid | ", "valid_")

        if do_save(step, config):
            pool.apply_async(save, (state, config))

    pool.close()
    pool.join()


def main(_):
    config = mlc.FrozenConfigDict(CONFIG_FLAG.value)
    logging.info(f"New run with config: \n{config}")
    # TODO: create subdir for run based on wandb run_id
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)

    wandb.init(
        dir=config.log_dir,
        project=config.project,
        tags=config.tags,
        notes=config.notes,
        id=config.run_id or None,
        resume=config.resume or None,
        config=config.as_configdict().to_dict())
    
    train(config)


if __name__ == "__main__":
    app.run(main)


