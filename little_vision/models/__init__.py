from .cnn import *
from .resnet import *
from .vit import *
from .mlpmixer import *
from .coatnet import *

"""Note:

Parts of this code have been taken from other authors.
This is common a practice but shall be highlighted nevertheless.

For the sources regarding ResNet look here:
[1] https://github.com/google-research/big_vision/blob/main/big_vision/models/bit.py
[2] https://github.com/google/flax/blob/main/examples/imagenet/models.py
[3] https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_resnet.py

References for the ViT Implementation may also be found in [1, 3].

The CoAtNet implementation borrows code from:
[4] https://github.com/dqshuai/MetaFormer/tree/master/models
[5] https://github.com/rwightman/efficientnet-jax/tree/master/jeffnet/linen
[6] https://github.com/google/flax/blob/main/flax/linen/attention.py

The MLP-Mixer implementation is implemented as described in the paper:
[7] https://arxiv.org/abs/2105.01601v4
[8] https://github.com/rwightman/efficientnet-jax/tree/master/jeffnet/linen (DropPath)
"""