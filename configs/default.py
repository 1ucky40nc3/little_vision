import ml_collections as mlc


def get_config():
    """Basic config for the MNIST example."""
    config = mlc.ConfigDict()

    config.do_train = True
    config.epochs = 10


    config.model = mlc.ConfigDict()
    config.model.name = "CNN"
    config.model.config = {}
    
    config.loss = mlc.ConfigDict()
    config.loss.name = "softmax_cross_entropy"
    config.loss.config = {}

    config.dataset = mlc.ConfigDict()
    config.dataset.name = "mnist"
    config.dataset.image_dims = (28, 28, 1)
    config.dataset.train_size = 60_000
    config.dataset.valid_size = 10_000
    config.dataset.num_classes = 10
    config.dataset.batch_size = 512
    config.dataset.num_workers = 0
    config.dataset.root = "/tmp"
    config.dataset.download = True

    config.optimizer = mlc.ConfigDict()
    config.optimizer.tx_name = "scale_by_adam"
    config.optimizer.tx_config = {}
    config.optimizer.gc_norm = 1.
    config.optimizer.wd = 1e-4
    config.optimizer.schedule_name = "warmup_cosine"
    config.optimizer.schedule_config = dict(
        num_epochs=config.epochs,
        warmup_epochs=2,
        base_lr=1e-3,
        num_steps_per_epoch=config.epochs // config.dataset.batch_size
    )

    config.metrics = mlc.ConfigDict()
    config.metrics.names = ("loss", "top1_err", "top5_err")
    
    config.log_every = 100
    config.save_every = 10_000
    config.log_dir = "./logs"
    config.save_dir = "./checkpoints"

    return config