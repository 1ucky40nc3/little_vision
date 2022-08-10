from little_vision import configs
from little_vision.data import datasets
from little_vision.data import utils


def test_prepare():
    # TODO: Implement test.
    pass


def test_mnist():
    config = configs.default.get_config()

    ds = datasets.mnist(train=True, config=config)
    it = datasets.prepare(ds, config)

    x, y = next(iter(it))
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
        dataset="cifar100")
    
    ds = datasets.cifar100(train=True, config=config)
    it = datasets.prepare(ds, config)

    x, y = next(iter(it))
    assert x.shape == (1, 512, 32, 32, 3)
    assert y.shape == (1, 512)