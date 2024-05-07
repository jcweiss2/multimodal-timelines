import os
import logging

import hydra
from omegaconf import OmegaConf, DictConfig

from atp.data import load_dataset
from atp.models import build_model
from atp.trainer import Trainer
from atp.utils.config_utils import add_custom_resolvers
from atp.utils.logger import init_logger
from atp.utils.misc import seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs")
def main(config: DictConfig):
    # Create logger
    OmegaConf.resolve(config)  # resolve the config first
    os.makedirs(config.output_dir, exist_ok=True)
    # logger = init_logger(config.output_dir, "train" if not config.test else "test")
    logger.info(OmegaConf.to_yaml(config))  # log the config

    if not config.test:  # Training
        # Save the training conf
        OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))

        # Seed
        seed(config.seed)

        # Dataset (train/dev)
        train_dataset = load_dataset(config.data, "train")
        dev_dataset = load_dataset(config.data, "dev")

        # Model
        model = build_model(config.model)

        # Trainer
        trainer = Trainer(config.trainer.params)
        trainer.train(model, train_dataset, dev_dataset)

        print("done")
    else:  # Testing
        # Dataset
        if config.test_all_split:
            train_dataset = load_dataset(config.data, "train")
            dev_dataset = load_dataset(config.data, "dev")
        test_dataset = load_dataset(config.data, "test")

        # Model
        model = build_model(config.model)

        # Trainer
        trainer = Trainer(config.trainer.params)
        if config.test_all_split:
            trainer.test(model, train_dataset)
            trainer.test(model, dev_dataset)
        trainer.test(model, test_dataset)

if __name__ == "__main__":
    add_custom_resolvers()
    main()
