import numpy as np

import jax

import torch

from little_vision import configs
from little_vision.data import utils


def test_to_jax_img():
    img = np.ones((32, 3, 224, 224))
    img = utils.to_jax_img(img)
    assert img.shape == (32, 224, 224, 3)


def test_shard():
    d = jax.local_device_count()
    img = np.ones((32*d, 224, 224, 3))
    img = utils.shard(img)
    assert img.shape == (d, 32, 224, 224, 3)