from dataclasses import dataclass, field
from typing import Any


@dataclass
class Experiment:
    """A container for tracking experiment parameters, metrics, and artifacts, 
    with optional logger callbacks for each update. """
    params : dict = field(default_factory=dict)
    metrics : dict = field(default_factory=dict)   
    artifacts : dict = field(default_factory=dict)
    log_registry : list = field(default_factory=list) 
    best_step : int = None
    best_loss : float = float('inf')
    model: Any = None 
    

    def add_logger(self, logger):
        self.log_registry.append(logger)

    def add_param(self, param):
        self.params.update(param)

        for logger in self.log_registry:
            if hasattr(logger, 'log_param'):
                logger.log_param(param)

    def add_metrics(self, tag, step, metrics):
        if step not in self.metrics:
            self.metrics[step] = {}      
        self.metrics[step][tag] = metrics

        for logger in self.log_registry:
            if hasattr(logger, 'log_metrics'):
                entry = {
                    'STEP' : step,
                    'TAG' : tag,
                    **metrics
                    }
                logger.log_metrics(entry)

    def add_artifact(self, name, artifact):
        self.artifacts.update({str(name): artifact})

        for logger in self.log_registry:
            if hasattr(logger, 'log_artifact'):
                logger.log_artifact(name, artifact)


    def add_model(self, model, name=None):
        self.model = model

        for logger in self.log_registry:
            if hasattr(logger, 'log_model'):
                logger.log_model(model, name)

    def log_info(self, name: str, content, **kwargs):
        """ Dispatches basic values to log entry. """
        for logger in self.log_registry:
            if hasattr(logger, 'log_info'):
                logger.log_info(name, content, **kwargs)


