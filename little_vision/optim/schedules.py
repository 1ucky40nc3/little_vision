import optax


def linear(
    base_lr: float,
    min_lr: float,
    num_steps: float,
    **kwargs
) -> optax.Schedule:
    return optax.linear_schedule(
        init_value=base_lr,
        end_value=min_lr,
        transition_steps=num_steps
    )


def warmup_linear(
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
    **kwargs
) -> optax.Schedule:
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=base_lr,
        transition_steps=warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=base_lr,
        end_value=min_lr,
        transition_steps=decay_steps)

    return optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_steps]
    )


def warmup_cosine(
    base_lr: float, 
    warmup_steps: int, 
    decay_steps: int,
    **kwargs
) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps
    )