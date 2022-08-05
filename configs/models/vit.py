import ml_collections as mlc
from torch import dropout


def preprocessing(
    config: mlc.ConfigDict
) -> mlc.ConfigDict:
    cfg = mlc.ConfigDict()
    cfg.name = "vit_transform"
    cfg.crop_padding = 3
    cfg.mixup = False

    return cfg


def optimizer(
    config: mlc.ConfigDict
) -> mlc.ConfigDict:
    cfg = mlc.ConfigDict()
    cfg.gc_norm = 1.
    cfg.tx_name = "scale_by_adam"
    cfg.tx_config = {}
    cfg.wd = .3
    cfg.schedule_name = "warmup_cosine"
    cfg.schedule_config = dict(
        base_lr=1e-3,
        min_lr=.0,
        warmup_steps=config.warmup_steps, # TODO: add fields
        decay_steps=config.decay_steps)

    return cfg


def model(
    config: mlc.ConfigDict
) -> mlc.ConfigDict:
    cfg = mlc.ConfigDict()
    cfg.name = config.model_name
    cfg.config = dict(
        num_classes=config.dataset.num_classes,
        dropout=0.1)

    return cfg