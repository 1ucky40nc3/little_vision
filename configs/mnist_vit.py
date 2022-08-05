import ml_collections as mlc

from configs import default


def get_config() -> mlc.FrozenConfigDict:
    """Basic config for the MNIST example."""
    config = default.get_config()

    config.name = "MNIST example with a ViT"
    config.notes = "Train a ViT classifier on the MNIST dataset."

    config.model.name = "ViT"

    return config