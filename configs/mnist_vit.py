import ml_collections as mlc


def get_config() -> mlc.FrozenConfigDict:
    """Basic config for the MNIST example."""
    config = mlc.ConfigDict()

    config.project = "little_vision"
    config.name = "MNIST example with a ViT"
    config.tags = ("computer-vision", "classification")
    config.notes = "Train a ViT classifier on the MNIST dataset."
    config.do_train = True

    config.resume = ""
    config.run_id = ""

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
    config.dataset.prefetch_size = 4

    config.num_epochs = 10
    config.num_steps_per_epoch = config.dataset.train_size // config.dataset.batch_size
    config.max_train_steps = config.num_steps_per_epoch * config.num_epochs
    config.max_valid_steps = config.dataset.valid_size // config.dataset.batch_size

    config.model = mlc.ConfigDict()
    config.model.name = "ViT"
    config.model.config = dict(
        num_classes=config.dataset.num_classes)
    
    config.loss = mlc.ConfigDict()
    config.loss.name = "softmax_cross_entropy"
    config.loss.config = dict(
        num_classes=config.dataset.num_classes)

    config.optimizer = mlc.ConfigDict()
    config.optimizer.tx_name = "scale_by_adam"
    config.optimizer.tx_config = {}
    config.optimizer.gc_norm = 1.
    config.optimizer.wd = 1e-4
    config.optimizer.schedule_name = "warmup_cosine"
    config.optimizer.schedule_config = dict(
        num_epochs=config.num_epochs,
        warmup_epochs=2,
        base_lr=1e-3,
        num_steps_per_epoch=config.num_steps_per_epoch
    )

    config.metrics = mlc.ConfigDict()
    config.metrics.names = ("loss", "top1_acc", "top5_acc")
    
    config.random_seed = 42
    config.log_every = 10
    config.eval_every = 300
    config.save_every = 300
    config.log_interval_type = "steps"
    config.eval_interval_type = "steps"
    config.save_interval_type = "steps"
    config.log_dir = "./logs"
    config.save_dir = "./checkpoints"
    config.keep_num_saves = 3

    return config