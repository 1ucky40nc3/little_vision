<table align="center" style="border:none">
  <tr style="border:none">
    <td style="border:none">
      <h1> little <br/> vision </h1>
    </td>
    <td style="border:none">
      <img src="https://raw.githubusercontent.com/1ucky40nc3/little_vision/main/assets/logo.png" width="400"/>
    </td>
  </tr>
</table>


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
