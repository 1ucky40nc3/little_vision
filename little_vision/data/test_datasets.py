from little_vision.configs import default
from little_vision.data import datasets


def test_mnist():
    config = default.get_config()
    print(config)
    ds = datasets.mnist(train=True, config=config)
    it = datasets.prepare(ds, config)
    x, y = next(iter(it))
    print(x.shape)
    assert x.shape == (1, 512, 28, 28, 1)
    assert y.shape == (1, 512)


def test_mnist_mlpmixer():
    config = configs.default.get_config()
    config.transform = configs.models.mlpmixer.preprocessing(config)
    ds = datasets.mnist(train=True, config=config)
    it = datasets.prepare(ds, config)
    x, y = next(iter(it))
    assert x.shape == (1, 512, 28, 28, 1)
    assert y.shape == (1, 512, 10)


def test_cifar100():
    config = configs.default.get_config(
        dataset="cifar100"
    )
    ds = datasets.cifar100(train=True, config=config)
    it = datasets.prepare(ds, config)
    x, y = next(iter(it))
    print(x.shape)
    assert x.shape == (1, 512, 32, 32, 3)
    assert y.shape == (1, 512)