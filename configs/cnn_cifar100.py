import ml_collections as mlc

from configs import models
from configs import datasets

from configs.utils import set


def get_config() -> mlc.FrozenConfigDict:
    """Basic config for the MNIST example."""
    config = mlc.ConfigDict()

    config.project = "little_vision"
    config.name = "MNIST example"
    config.tags = ("computer-vision", "classification")
    config.notes = "Train a classifier on the MNIST dataset."
    config.do_train = True

    config.resume = ""
    config.run_id = ""

    config.dataset = datasets.cifar100.get_config()
    config.dataset.batch_size = 512
    config.dataset.num_workers = 0
    config.dataset.root = "/tmp"
    config.dataset.download = True
    config.dataset.prefetch_size = 4

    config.num_epochs = 10
    config.num_steps = None
    config.epoch_steps = config.dataset.train_size // config.dataset.batch_size
    config.max_train_steps = set(config.num_steps, config.num_epochs * config.epoch_steps)
    config.warmup_steps = 0
    config.decay_steps = config.max_train_steps - config.warmup_steps
    config.max_valid_steps = config.dataset.valid_size // config.dataset.batch_size

    config.loss = mlc.ConfigDict()
    config.loss.name = "softmax_cross_entropy"
    config.loss.config = dict(
        num_classes=config.dataset.num_classes)

    config.model = models.cnn.model(config)
    config.optimizer = models.cnn.optimizer(config)
    config.transform = models.cnn.preprocessing(config)

    config.metrics = mlc.ConfigDict()
    config.metrics.names = ("loss", "top1_acc", "top5_acc")
    
    config.random_seed = 42
    config.log_every = 10
    config.eval_every = 50
    config.save_every = 300
    config.log_interval_type = "steps"
    config.eval_interval_type = "steps"
    config.save_interval_type = "steps"
    config.log_dir = "./logs"
    config.save_dir = "./checkpoints"
    config.keep_num_saves = 3

    return config