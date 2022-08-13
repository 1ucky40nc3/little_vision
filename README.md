# little_vision
experiments with deep neural networks

---

This repository is a learning resource for deep neural networks in a computer vision context. Specifically we are trying to solve image classification tasks. To accomplish this goal we build neural networks following popular architectures. Their training is currently restricted by my resources - with basically is [Google Colab ;)](https://research.google.com/colaboratory/faq.html). Therefore presented training examples will be restricted to small versions of each architecture and smaller image classification datasets. 

If you are interested to contribute or collaborate in any way, you are very welcome. Just contact me via an issue or e-mail.


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

# Training
Working example of training runs can be found as
iPython notebooks that may be run in Google Colaboratory.

# Warning!
Training  using the ```train.py``` script is sadly not working.
To fix this crisis traing notebooks (as described above) were implemented.

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
