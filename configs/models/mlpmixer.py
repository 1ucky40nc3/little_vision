import ml_collections as mlc
from torch import dropout


def preprocessing(
    config: mlc.ConfigDict
) -> mlc.ConfigDict:
    cfg = mlc.ConfigDict()
    cfg.name = "mlpmixer_transform"
    cfg.crop_padding = 3
    cfg.mixup = True
    cfg.mixup_config = dict(
        mixup_alpha=0.5,
        cutmix_alpha=0.,
        cutmix_minmax=None,
        prob=1.,
        switch_prob=0.,
        mode="batch",
        label_smoothing=0.,
        num_classes=config.dataset.num_classes
    )
    cfg.randaugment_config = dict(
        m=15, n=2)

    return cfg


def optimizer(
    config: mlc.ConfigDict
) -> mlc.ConfigDict:
    cfg = mlc.ConfigDict()
    cfg.gc_norm = 1.
    cfg.tx_name = "scale_by_adam"
    cfg.tx_config = dict()
    cfg.wd = 0.1
    cfg.schedule_name = "warmup_linear"
    cfg.schedule_config = dict(
        base_lr=1e-3,
        min_lr=0.,
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
        drop_path=0.1) 

    return cfg