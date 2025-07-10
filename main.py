from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.tracking import Experiment
import src.custom_logging as cl
import src.model as ml
from src.dataset import create_datasets
from src.stage import Stage
from src.evaluation import create_evaluator
from src.runner import create_runner
from src.config import Config
from src.training_utils import set_random_state, EarlyStopping


def main(cfg: Config, early_stopper):

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

        if scheduler:
            scheduler.step()

        with Stage.VAL(model) as stage:
            val_metrics = epoch_runner(val_loader)
            experiment.add_metrics(tag=stage.name, step=epoch, metrics=val_metrics)

        val_loss = val_metrics.get('MSE')
        if val_loss is not None:
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return experiment


if __name__ == "__main__":

    cfg = Config(
        batch_size = 8,
        epochs = 100,
        samples = 1000,
        loss = 'MSE',
        metrics = ['MSE', 'PSNR', 'SSIM'],

        model_cls=ml.CAE,
        model_kwargs={"latent_channels": 12, "hidden_channels": [64, 128]},

        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 5e-4},

        scheduler_cls=torch.optim.lr_scheduler.ExponentialLR,
        scheduler_kwargs={"gamma": 0.99},

        logger_cls=cl.ConsoleLogger,
        logger_kwargs={'log_dir' : Path('output/test')},

        dataset_factory= create_datasets,
        dataset_kwargs= {"data_path": Path('data/CIFAR10/')},

        early_stopper_cls=EarlyStopping,
        early_stopper_kwargs= {'patience': 5, 'delta': 1e-5}
    )

    main(cfg)


