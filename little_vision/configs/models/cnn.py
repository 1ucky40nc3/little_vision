import ml_collections as mlc


def preprocessing(
    config: mlc.ConfigDict
) -> mlc.ConfigDict:
    cfg = mlc.ConfigDict()
    cfg.name = "resnet_transform"
    cfg.crop_padding = 3
    cfg.mixup = False

    return cfg


def optimizer(
    config: mlc.ConfigDict
) -> mlc.ConfigDict:
    cfg = mlc.ConfigDict()
    cfg.gc_norm = None
    cfg.tx_name = "scale_by_sgd"
    cfg.tx_config = dict(
        momentum=0.9)
    cfg.wd = 1e-4
    cfg.schedule_name = "linear"
    cfg.schedule_config = dict(
        base_lr=0.1,
        min_lr=1e-5,
        num_steps=config.epoch_steps)

    return cfg


def model(
    config: mlc.ConfigDict
) -> mlc.ConfigDict:
    cfg = mlc.ConfigDict()
    cfg.name = "CNN"
    cfg.config = dict(
        num_classes=config.dataset.num_classes)

    return cfg
