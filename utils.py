from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Callable
from typing import OrderedDict

import time

from dataclasses import dataclass

from absl import logging

import torch

import jax
import jax.numpy as jnp

import einops

import wandb

import metrics as little_metrics


def shard(array: jnp.ndarray) -> jnp.array:
    return einops.rearrange(
        array, 
        "(d n) ... -> d n ...", 
        d=jax.local_device_count())


# TODO: maybe shard with flax.jax_utils.replicate
def jaxify(tensor: torch.Tensor) -> jnp.ndarray:
    return shard(tensor.numpy())

"""
@dataclass
class Action:
    desc: str
    fn: Callable[[Any], Any]
    data: List[Any] = []

    last: float = 0.
    max: float = 0.
    type: str = "step"

    @property
    def cond(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> None:
        raise NotImplementedError()
"""

T_TYPES = ("steps", "time")

@dataclass
class Action:
    def __init__(
        self,
        fn: Callable[[Any], Any],
        kwargs: Dict[str, Any],
        int_t: float,
        t_type: str,
        max_steps: float,
        data: OrderedDict = OrderedDict(),
        last_t: float = 0.,
        save_data: bool = True,
        flush_data: bool = True
    ) -> None:
        assert t_type in T_TYPES, (
            f"The `t_type` has to be in {T_TYPES}! "
            f"But {t_type} was provided!")

        self.fn = fn
        self.kwargs = kwargs
        self.int_t = int_t
        self.max_steps = max_steps
        self.t_type = t_type
        self.data = data # TODO: make data as ordered dict -> to be able to call func on local and global data
        self.last_t = last_t
        self.save_data = save_data
        self.flush_data = flush_data

    def __call__(
        self,
        step: Optional[float] = None,
        value: Optional[Any] = None,
        only_last: bool = True,
        **kwargs
    ) -> None:
        if self.save_data:
            self.data[step] = value 

        is_time, t = self.eval(step)
        if is_time:
            if only_last:
                data = value
            else:
                data = []
                for value in self.data.values():
                    data = [*data, *value]

            print({**self.kwargs, **kwargs})
            self.fn(
                step,
                data, 
                **{**self.kwargs, **kwargs}
            )
            self.last_t = t

            if self.flush_data:
                self.data = []

    def eval(self, step: float) -> bool:
        t = self.type_t(step)
        return (
            t - self.last_t >= self.int_t 
            or step == self.max_steps
        ), t

    def type_t(
        self,
        t: Optional[float] = None
    ) -> float:
        return time.time() if self.t_type == "time" else t


def log_metrics(
    step: float,
    metrics: List[jnp.ndarray],
    metric_fns: little_metrics.Metrics,
    desc: str = "",
    prefix: str = "",
    format: str = ": {:.4f}%s ",
    delimiter: str = ";",
    log_in_board: bool = True
) -> None:
    metrics = jnp.array(metrics).T
    metrics = jnp.mean(metrics, axis=-1)

    string = (format % delimiter).join(metric_fns._fields) + format % ""
    logging.info(f"{desc.format(step)}{string.format(*metrics)}")

    if log_in_board:
        dictionary = {
            f"{prefix}{k}": v 
            for k, v in zip(metric_fns._fields, metrics)}
        print(dictionary)
    #wandb.log(dictionary)
    


class Writer:
    pass