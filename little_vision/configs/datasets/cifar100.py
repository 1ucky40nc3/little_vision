import ml_collections as mlc


def get_config() -> mlc.ConfigDict:
    config = mlc.ConfigDict()
    config.name = "cifar100"
    config.image_dims = (32, 32, 3)
    config.train_size = 50_000
    config.valid_size = 10_000
    config.num_classes = 100
    config.mean = (0.5071, 0.4865, 0.4409)
    config.std = (0.2673, 0.2564, 0.2762)

    return config