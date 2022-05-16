from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Tuple
from typing import Optional
from typing import Callable
from typing import OrderedDict

import time

from dataclasses import dataclass

from absl import logging

import torch
import torch.utils.data as tud

import jax
import jax.numpy as jnp

import einops

import ml_collections as mlc

import wandb

import train as little_train
import actions as little_actions
import metrics as little_metrics


def log(
    update: Union[List[jnp.ndarray], List[List[jnp.ndarray]]],
    index: int,
    config: mlc.ConfigDict,
    desc: str = "",
    prefix: str = "",
    format: str = ": {:.4f}%s ",
    delimiter: str = ";",
    only_std: bool = False,
    **kwargs
) -> None:
    update = jnp.array(update)
    print(update.shape)
    #print(update)
    if update.ndim == 1:
        update = update[None, ...]
    update = update.T
    update = jnp.mean(update, axis=-1)

    string = (format % delimiter).join(config.metrics.names) + format % ""
    logging.info(f"{desc.format(index)}{string.format(*update)}")

    if not only_std:
        data = {
            f"{prefix}{k}": v 
            for k, v in zip(
                config.metrics.names, 
                update
            )
        }
        #wandb.log(data)
    

def log_train_action(
    config: mlc.ConfigDict
) -> little_actions.Action:
    return little_actions.Action(
        fn=log,
        fn_kwargs=dict(
            config=config,
            desc="Training ({} / %d) | " % config.max_train_steps,
            prefix="train_",
        ),
        interval=config.log_every,
        interval_type=config.log_interval_type,
        max_index=config.max_train_steps,
        clear_upates=True
    )


def log_valid_action(
    config: mlc.ConfigDict
) -> little_actions.Action:
    return little_actions.Action(
        fn=log,
        fn_kwargs=dict(
            config=config,
            desc="Evaluation ({} / %d) | " % config.max_valid_steps,
            prefix="train_",
        ),
        interval=config.log_every,
        interval_type=config.log_interval_type,
        max_index=config.max_valid_steps
    )


def valid_action(
    config: mlc.ConfigDict,
    dataset: tud.DataLoader
) -> little_actions.Action:
    return little_actions.Action(
        fn=little_train.evaluate,
        fn_kwargs=dict(
            dataset=dataset,
            config=config,
            actions=(log_valid_action(config),)
        ),
        interval=config.eval_every,
        interval_type=config.eval_interval_type,
        max_index=config.max_train_steps,
        use_latest=True,
        save_updates=False,
    )