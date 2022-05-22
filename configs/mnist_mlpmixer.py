import ml_collections as mlc

from configs import default


def get_config() -> mlc.FrozenConfigDict:
    """Basic config for the MNIST example."""
    config = default.get_config()

    config.name = "MNIST example with a MLP-Mixer"
    config.notes = "Train a MLP-Mixer classifier on the MNIST dataset."

    config.model.name = "MLPMixer"

    return config