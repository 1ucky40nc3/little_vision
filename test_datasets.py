
import configs
import datasets


def test_mnist():
    config = configs.default.get_config()
    print(config)
    ds = datasets.mnist(train=True, config=config)
    it = datasets.prepare(ds, config)
    x, y = next(iter(it))
    print(x.shape)
    assert x.shape == (1, 512, 28, 28, 1)
    assert y.shape == (1, 512)

test_mnist()


def test_mnist_mlpmixer():
    config = configs.default.get_config()
    config.transform = configs.models.mlpmixer.preprocessing(config)
    ds = datasets.mnist(train=True, config=config)
    it = datasets.prepare(ds, config)
    x, y = next(iter(it))
    assert x.shape == (1, 512, 28, 28, 1)
    assert y.shape == (1, 512, 10)

test_mnist_mlpmixer()


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

test_cifar100()