<p align="center">
    <br>
      <img src="https://raw.githubusercontent.com/1ucky40nc3/little_vision/main/assets/logo.png" width="600"/>
    <br>
</p>
<h2 align="center">
  <p>little experiments with deep neural networks</p>
</h2>

---

This repository is a learning resource for deep neural networks in a computer vision context. Specifically, we are trying to solve image classification tasks. We build neural networks following popular architectures and train with openly available datasets. 

Because of current resource restrictions experiments make use of small models and little data. Maybe we can change this in the future... If you are interested in contributing or collaborating in any way, you are welcome. Just contact me by creating an issue or e-mail.


# Content
This repository supports the training of a number of deep neural networks on image classification datasets.
Supported architectures are:
- [ ] ResNet
- [ ] ViT
- [ ] MLP-Mixer
- [ ] CoAtNet

We train with the following datasets:
- [ ] MNIST
- [ ] CIFAR-10
- [ ] CIFAR-100

We construct theese networks and train them using [jax](https://github.com/google/jax) und [flax](https://github.com/google/flax). Datasets are loaded via [PyTorch](https://github.com/pytorch/pytorch), in the style of [torchvision](https://github.com/pytorch/vision). Specific image augmentations are implemented using torchvision and [timm](https://github.com/rwightman/pytorch-image-models). The evaluation is done using [scikit-learn](https://github.com/scikit-learn/scikit-learn) and reported to [wandb](https://wandb.ai/site).

## Models
### ResNet
- status: implemented
- paper: K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv:1512.03385 [cs], Dec. 2015, Accessed: May 10, 2022. [Online]. Available: http://arxiv.org/abs/1512.03385
- models:
ResNet models can be split into stages (s) that repeat blocks of the same size. At the start of each but the first layer a stride of 2 is applied. A ResNet block combines convolution layers and uses ReLU activations and BatchNorm normalization.
There are two different kinds of blocks. The a default (ResNetBlock) one and a version that introduces an additional bottleneck layer (BottleneckResNetBlock).

Name | Structure | Parameters
ResNet18 | s=(2, 2, 2, 2) | 11,689,512
ResNet34 | s=(3, 4, 6, 3) | 21,797,672
ResNet50 | s=(3, 4, 6, 3)* | 25,557,032
*The ResNet50 makes use of BottleneckResNetBlock layers
- implementation: [resnet.py](little_vision/models/resnet.py)

### ViT
- status: implemented
- paper: A. Dosovitskiy u. a., „An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale“, arXiv:2010.11929 [cs], Juni 2021, Zugegriffen: 10. Mai 2022. [Online]. Verfügbar unter: http://arxiv.org/abs/2010.11929

- models:
Name | Structure | Parameters
ResNet18 | s=(2, 2, 2, 2) | 11 M
ResNet34 | s=(3, 4, 6, 3) | 21 M
ResNet50 | s=(3, 4, 6, 3)* | 25 M
### MLP-Mixer
### CoAtNet



# Training
Working example of training runs can be found as
iPython notebooks that may be run in Google Colaboratory.


# Example of how it should have been: 
To run the MNIST example with Weights and Biases logging:
```
python3 train.py
```
To start a run with a different config execute:
```
python3 train.py --config=/path/to/config.py
```

To resume a run:
```
python3 train.py --config.run_id="$run_id" --config.resume="must"
```

# Testing
Tests are included in parallel to the implementation of most components.
To run the tests install all dependencies and execute
```
python3 -m pip install pytest
```
next you can execute all tests via
```
python3 -m pytest
```
individual files are functions can be tested in the following ways:
```
python3 -m pytest /path/to/test_file.py
python3 -m pytest /path/to/test_file.py::test_function
```
