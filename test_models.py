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






       
#test_resnetblock()



   
#test_bottleneckresnetblock()




#test_resnet()





#test_resnet34_size()





#test_vit_size()





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

def test_coatnetconv(
    features: int = 64,
    image_dims: Tuple[int] = (32, 32, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    block = models.CoAtNetConvBlock(
        features=features)

    x = jnp.ones((1, *image_dims))
    variables = block.init(rng, x)
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))

#test_coatnetconv()

    
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

    print(sum(p.size for p in jax.tree_leaves(variables["params"])))

#test_relativeselfattention()


def test_coatnet_flow(
    image_dims: Tuple[int] = (32, 32, 3)
) -> None:
    pool = partial(
        nn.max_pool,
        window_shape=(3, 3),
        strides=(2, 2),
        padding="SAME")

    x = jnp.ones([1, *image_dims])
    print("x", x.shape)
    x = pool(x)
    print("s0", x.shape)
    x = pool(x)
    print("s1", x.shape)
    x = pool(x)
    print("s2", x.shape)
    x = pool(x)
    print("s3", x.shape)
    x = pool(x)
    print("s4", x.shape)

#test_coatnet_flow()


def test_coatnet(
    image_dims: Tuple[int] = (32, 32, 3),
    num_classes: int = 100,
) -> None:
    rng = jax.random.PRNGKey(42)

    coatnet = models.CoAtNet(
        num_classes=num_classes)
    variables = coatnet.init(rng, jnp.ones([1, *image_dims]))

    out = coatnet.apply(variables, jnp.ones([32, *image_dims]))
    assert out.shape == (32, num_classes)

#test_coatnet()


def test_coatnet_size(
    num_classes: int = 1000,
    dims: Tuple[int] = (224, 224, 3)
) -> None:
    rng = jax.random.PRNGKey(42)

    coatnet = models.CoAtNet0(
        num_classes=num_classes)
    variables = coatnet.init(rng, jnp.ones([1, *dims]))
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))
    """
    coatnet = models.CoAtNet1(
        num_classes=num_classes)
    variables = coatnet.init(rng, jnp.ones([1, *dims]))
    print(sum(p.size for p in jax.tree_leaves(variables["params"])))
    """

test_coatnet_size()


def test_head_init(
    features: int = 10,
    bias: float = 0.
) -> None:
    rng = jax.random.PRNGKey(42)

    dense = nn.Dense(
        features=features,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.constant(
            bias
        ), name="head")
    
    x = jnp.ones([1, 10, 1])
    variables = dense.init(rng, x)

    assert (jnp.zeros(x.shape) == dense.apply(variables, x)).all()

#test_head_init()