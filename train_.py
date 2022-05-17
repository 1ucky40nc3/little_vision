from typing import Tuple
from typing import Callable

from functools import partial
from itertools import cycle

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
    return jax.process_index() == host and step > 0 and step == config.log_every

def do_eval(step: int, config: mlc.ConfigDict) -> bool:
    return step > 0 and step in (config.eval_every, config.max_train_steps)

def do_checkpoint(step: int, config: mlc.ConfigDict, host: int = 0) -> bool:
    return jax.process_index() == host and step > 0 and step == config.save_every
    

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


def jaxify(tensor: torch.Tensor) -> jnp.ndarray:
    return jax_utils.replicate(tensor.numpy())

def evaluate(
    state: TrainState,
    dataset: tud.DataLoader,
    config: mlc.ConfigDict,
    **kwargs
) -> None:
    metrics = jnp.zeros((3,))
    for step, batch in enumerate(dataset):
        images, labels = jax.tree_map(jaxify, batch)

        bmetrics = eval_step(state, images, labels, config)
        bmetrics = jax_utils.unreplicate(bmetrics)
        metrics += jnp.array(bmetrics)

        if step and step % config.log_every == 0:
            loss, top1, top5 = metrics / step
            print(f"Eval sub {step}, {loss:.4f}, {top1:.4f}, {top5:.4f}")

    return metrics





def train(
    config: mlc.ConfigDict,
    **kwargs
) -> None:
    num_devices = jax.local_device_count()
    key = jax.random.PRNGKey(config.random_seed)
    key, subkey = jax.random.split(key)

    subkeys = jax.random.split(subkey, num_devices)
    state = train_state(subkeys, config)
    start_step = jax_utils.unreplicate(state.step)

    train_ds = getattr(little_datasets, config.dataset.name)(train=True, config=config)
    valid_ds = getattr(little_datasets, config.dataset.name)(train=False, config=config)

    tmetrics = jnp.zeros((3,))
    for step, batch in zip(
        range(start_step, config.max_train_steps), 
        cycle(train_ds)):
        images, labels = jax.tree_map(jaxify, batch) # TODO try to jit

        state, metrics = train_step(state, images, labels, config)
        metrics = jax_utils.unreplicate(metrics)
        tmetrics += jnp.array(metrics)

        if do_logging(step, config):
            tmetrics /= config.log_every
            loss, top1, top5 = tmetrics
            print(f"Train {step}, {loss:.4f}, {top1:.4f}, {top5:.4f}")
            tmetrics = jnp.zeros((3,))

        if do_eval(step, config):
            vmetrics = evaluate(state, valid_ds, config)
            vmetrics /= config.max_valid_steps
            loss, top1, top5 = vmetrics
            print(f"Eval {step}, {loss:.4f}, {top1:.4f}, {top5:.4f}")


from configs import default
train(default.get_config())



