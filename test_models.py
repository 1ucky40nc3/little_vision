from typing import Tuple

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import flax.linen as nn

import models

import torch
import torch.nn.functional as F

from typing import Any
from typing import Callable


def tree_equal(a: Any, b: Any) -> bool:
    a_values, a_treedef = jax.tree_util.tree_flatten(a)
    b_values, b_treedef = jax.tree_util.tree_flatten(b)
    
    return (
        a_treedef == b_treedef
        and all([
            (i == j).all() 
            for i, j in zip(a_values, b_values)
        ])
    )


def test_resnetblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    block = models.ResNetBlock(
        features=features,
        norm=partial(
            nn.BatchNorm, 
            use_running_average=True))
    variables = block.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    out = block.apply(variables, jnp.ones([10, *dims]))
    assert out.shape == (10, *dims[:-1], features)
       
#test_resnetblock()


def test_bottleneckresnetblock(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    block = models.BottleneckResNetBlock(
        features=features,
        norm=partial(
            nn.BatchNorm, 
            use_running_average=True))
    variables = block.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    out = block.apply(variables, jnp.ones([10, *dims]))
    assert out.shape == (10, *dims[:-1], features * 4)
   
#test_bottleneckresnetblock()


def test_resnet(
    features: int = 64,
    dims: Tuple[int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    resnet = models.ResNet(
        features=features,
        norm=partial(
            nn.BatchNorm, 
            use_running_average=True))
    variables = resnet.init(rng, jnp.ones([1, *dims]))
    assert "params" in variables and "batch_stats" in variables

    sample = jnp.ones([10, *dims])
    out = resnet.apply(
        variables, 
        sample, 
        train=False)
    assert out.shape == (10, resnet.num_classes)

    out, mutated_vars = resnet.apply(
        variables, 
        sample, 
        train=True, 
        mutable=["batch_stats"])
    assert out.shape == (10, resnet.num_classes)
    assert "batch_stats" in mutated_vars.keys()

#test_resnet()


def test_conv(
    features: int = 64,
    kernel_size: Tuple[int, int] = (3, 3), 
    strides: Tuple[int, int] = (1, 1),
    dims: Tuple[int, int, int] = (28, 28, 1)
) -> None:
    rng = jax.random.PRNGKey(42)

    conv0 = nn.Conv(
        features=features, 
        kernel_size=kernel_size)
    conv1 = nn.Conv(
        features=features,
        kernel_size=kernel_size,
        strides=strides)
    
    sample = jnp.ones([1, *dims])
    variables0 = conv0.init(rng, sample)
    variables1 = conv1.init(rng, sample)

    out0 = conv0.apply(variables0, sample)
    out1 = conv1.apply(variables1, sample)

    assert (out0 == out1).all()

#test_conv()

    
def test_relative_positions(
    img_size: Tuple[int] = (32, 32)
) -> None:
    """Reference implemenation from:
    https://github.com/dqshuai/MetaFormer/blob/master/models/MHSA.py
    """
    extra_token_num = 0
    coords_h = torch.arange(img_size[0])
    coords_w = torch.arange(img_size[1])
    coords = torch.stack(
        torch.meshgrid([coords_h, coords_w]))  # 2, h, w
    coords_flatten = torch.flatten(coords, 1)  # 2, h*w
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
    relative_coords[:, :, 0] += img_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += img_size[1] - 1
    relative_coords[:, :, 0] *= 2 * img_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # h*w, h*w
    relative_position_index = F.pad(
        relative_position_index, 
        (extra_token_num, 0, extra_token_num, 0))
    relative_position_index = relative_position_index.long()
    relative_position_index = relative_position_index.numpy().astype(np.int32)

    rel_pos = models.relative_positions(img_size)
    assert (rel_pos == relative_position_index).all()

#test_relative_positions()


def test_relativeselfattention(
    num_heads: int = 4,
    img_size: Tuple[int] = (24, 24)
) -> None:
    rng = jax.random.PRNGKey(42)

    attn = models.RelativeSelfAttention(
        num_heads, img_size)
    variables = attn.init(rng, jnp.ones([1, 576, 384]))
    assert "params" in variables

    out = attn.apply(variables, jnp.ones([32, 576, 384]))
    assert out.shape == (32, 576, 384)

    
test_relativeselfattention()