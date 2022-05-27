import ml_collections as mlc


def get_config() -> mlc.ConfigDict:
    config = mlc.ConfigDict()
    config.name = "mnist"
    config.image_dims = (28, 28, 1)
    config.train_size = 60_000
    config.valid_size = 10_000
    config.num_classes = 10
    config.mean = (0.13066047,)
    config.std = (0.30810955,)

    return config