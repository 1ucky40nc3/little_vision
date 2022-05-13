import optax


def warmup_cosine(
    num_epochs: int, 
    warmup_epochs: int, 
    base_lr: float, 
    num_steps_per_epoch: int
) -> optax.Schedule:
    warmup_steps = warmup_epochs * num_steps_per_epoch
    decay_epochs = max(num_epochs - warmup_epochs, 1)
    decay_steps = decay_epochs * num_steps_per_epoch
    assert warmup_steps < decay_steps, (
        "The number of warmup steps has to be smaller than the number of decay steps!")

    return optax.warmup_cosine_decay_schedule(
        init_value=0.,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps)