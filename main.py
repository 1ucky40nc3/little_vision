from absl import app
from absl import flags
from absl import logging


import train as little_train
import utils as little_utils
import datasets as little_datasets


def main(_):
    #wandb.init()
    from configs import default
    config = default.get_config()

    ds_cls = getattr(little_datasets, config.dataset.name)
    train_ds = ds_cls(train=True, config=config)
    valid_ds = ds_cls(train=False, config=config)

    actions = (
        little_utils.log_train_action(config),
        little_utils.valid_action(config, valid_ds)
    )

    little_train.train(config, train_ds, actions)


if __name__ == "__main__":
    app.run(main)