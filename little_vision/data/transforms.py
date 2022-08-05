from typing import List
from typing import Union
from typing import Callable

from functools import partial


import torch.nn as nn

import torchvision.transforms as transforms

from timm.data.auto_augment import rand_augment_transform


import ml_collections as mlc



ConfigDict = Union[mlc.ConfigDict, mlc.FrozenConfigDict]


def randaugment(
    config: ConfigDict
) -> Callable:
    config_str = "rand-m{m}-n{n}".format(
        **config.transform.randaugment)
    return transforms.Lambda(
        partial(
            rand_augment_transform, 
            config_str=config_str
    ))


def transform(
    config: ConfigDict,
    t: List[nn.Module] = []
) -> transforms.Compose:
    t = []
    return transforms.Compose([
        transforms.RandomCrop(
            size=config.dataset.image_dims[:-1],
            padding=config.transform.crop_padding),
        transforms.ToTensor(),
        *t
    ])


def resnet_transform(
    config: ConfigDict
) -> transforms.Compose:
    return transform(
        config, [
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(
            mean=config.dataset.mean,
            std=config.dataset.std)
    ])



def vit_transform(
    config: ConfigDict
) -> transforms.Compose:
    return transform(
        config, [
        transforms.RandomHorizontalFlip(),
    ])


def coatnet_transform(
    config: ConfigDict
) -> transforms.Compose:
    return transform(
        config, [
        randaugment(config)
    ])


def mlpmixer_transform(
    config: ConfigDict
) -> transforms.Compose:
    return transform(
        config, [
        randaugment(config)
    ])