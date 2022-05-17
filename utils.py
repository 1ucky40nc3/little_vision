from typing import List
from typing import Union
from typing import Optional

from absl import logging

import torch
import torch.utils.data as tud

import jax
import jax.numpy as jnp

import ml_collections as mlc

import wandb

import train as little_train
import actions as little_actions


def log(
    update: Union[List[jnp.ndarray], List[List[jnp.ndarray]]],
    config: mlc.ConfigDict,
    step: Optional[int] = None,
    index: Optional[int] = None,
    max_index: Optional[int] = None,
    counter: Optional[int] = None,
    interval: Optional[int] = None,
    desc: str = "Debug [{step}/{index}/{max_index}]*({counter}/{interval}) ",
    prefix: str = "debug_",
    format: str = ": {:.4f}%s ",
    delimiter: str = ";",
    only_std: bool = False,
    **kwargs
) -> None:
    update = jnp.array(update)
    print(prefix, update.shape)
    if only_std:
        print(update)
    if update.ndim == 1:
        update = update[None, ...]
    update = update.T
    update = jnp.mean(update, axis=-1)

    progress = dict(
        step=step, 
        index=index, 
        max_index=max_index, 
        counter=counter, 
        interval=interval)
    string = (format % delimiter).join(config.metrics.names) + format % ""
    logging.info(f"{desc.format(**progress)}{string.format(*update)}")

    if not only_std:
        data = {
            f"{prefix}{k}": v 
            for k, v in zip(
                config.metrics.names, 
                update
            )
        }
        data["step"] = step
        # TODO: add step to log (global step for logging)
        # maybe get step from state
        # real step for train = step * interval
        # real step for eval = -> introduce from train during eval action call
        #wandb.log(data)
    

def log_train_action(
    config: mlc.ConfigDict
) -> little_actions.Action:
    return little_actions.Action(
        fn=log,
        fn_kwargs=dict(
            config=config,
            desc="Training ({step} / %d) | " % config.max_train_steps,
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
            desc="Evaluation ({index} / %d) | " % config.max_valid_steps,
            prefix="valid_",
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