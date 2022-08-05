from typing import Union

import optax

import ml_collections as mlc

import schedules


ConfigDict = Union[
    mlc.ConfigDict, 
    mlc.FrozenConfigDict
]


def scale_by_sgd(
    momentum: float = 0.9,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    return optax.trace(
        decay=momentum,
        nesterov=nesterov
    )

optax.scale_by_sgd = scale_by_sgd


def scale_by_adamw(
    b1: float = 0.9,
    b2: float = 0.999,
    wd: float = 1e-4
) -> optax.GradientTransformation:
    return optax.chain(
        optax.scale_by_adam(
            b1=b1, b2=b2),
        optax.add_decayed_weights(wd)
    )

optax.scale_by_adamw = scale_by_adamw


def tx(
    config: ConfigDict
) -> optax.GradientTransformation:
    gc_tx = optax.identity()
    if config.gc_norm:
        gc_tx = optax.clip_by_global_norm(
            config.gc_norm)
    tx_fn = getattr(optax, config.tx_name)(
        **config.tx_config)
    wd_tx = optax.additive_weight_decay(
        config.wd)
    lr_tx = optax.scale_by_schedule(
        getattr(
            schedules, 
            config.schedule_name)(
        **config.schedule_config))

    return optax.chain(
        gc_tx,
        tx_fn,
        wd_tx,
        lr_tx,
        optax.scale(-1)
    )