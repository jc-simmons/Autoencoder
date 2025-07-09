from pathlib import Path
import numpy as np
import random

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from tracking import Experiment
import custom_logging as cl
import model as ml
from dataset import create_datasets
from stage import Stage
from evaluation import create_evaluator
from runner import create_runner


from config import Config


def main(cfg: Config):

    set_random_state(cfg.random_state)
    torch.set_num_threads(cfg.threads)

    model = cfg.model_cls(**cfg.model_kwargs)
    optimizer = cfg.optimizer_cls(model.parameters(), **cfg.optimizer_kwargs)
    scheduler = cfg.scheduler_cls(optimizer, **cfg.scheduler_kwargs)
    logger = cfg.logger_cls(**cfg.logger_kwargs)

    experiment = Experiment()
    experiment.add_logger(logger)

    for param, val in cfg.to_simple_dict().items():
        experiment.add_param({param: val})
    
    dataset = cfg.dataset_factory(**cfg.dataset_kwargs)

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset.train, range(cfg.samples)), 
        batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset.val, range(int(cfg.samples * 0.1))), 
        batch_size=cfg.batch_size, shuffle=True
    )

    metric_evaluator = create_evaluator(cfg.metrics)
    epoch_runner = create_runner(model, metric_evaluator, optimizer, loss=cfg.loss)

    for epoch in range(1, cfg.epochs + 1):
        with Stage.TRAIN(model) as stage:
            train_metrics = epoch_runner(train_loader)
            experiment.add_metrics(tag=stage.name, step=epoch, metrics=train_metrics)

        scheduler.step()

        with Stage.VAL(model) as stage:
            val_metrics = epoch_runner(val_loader)
            experiment.add_metrics(tag=stage.name, step=epoch, metrics=val_metrics)

    return experiment


def set_random_state(seed=None):
    """ Sets the global random state. """
    if not isinstance(seed, int):
        seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # seems necessary to work on HPC clusters having their own MKL libraries
    # torch.multiprocessing.set_start_method("spawn")
    cfg = Config(

        batch_size = 8,
        epochs = 100,
        samples = 50000,

        model_cls=ml.CAE,
        model_kwargs={"latent_channels": 12, "hidden_channels": [64, 128]},

        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 5e-4},

        scheduler_cls=torch.optim.lr_scheduler.ExponentialLR,
        scheduler_kwargs={"gamma": 0.99},

        logger_cls=cl.ConsoleLogger,
        logger_kwargs={'log_dir' : Path('output/test')},

        dataset_factory= create_datasets,
        dataset_kwargs= {"data_path": Path('data/CIFAR10/'), 
                        "feature_transform": transforms.ToTensor(),
                        "target_transform": transforms.ToTensor()}
    )

    cfg.loss = 'MSE'

    main(cfg)


