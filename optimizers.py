import optax

import ml_collections as mlc

import schedules


def tx(config: mlc.ConfigDict) -> optax.GradientTransformation:
    gc_tx = optax.clip_by_global_norm(config.gc_norm)
    tx_fn = getattr(optax, config.tx_name, **config.tx_config)
    wd_tx = optax.additive_weight_decay(config.wd)
    lr_tx = optax.scale_by_schedule(
        getattr(schedules, config.schedule_name)(
            **config.schedule_config))

    return optax.chain(
        gc_tx,
        tx_fn,
        wd_tx,
        lr_tx,
        optax.scale(-1)
    )