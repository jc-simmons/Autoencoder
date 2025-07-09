from dataclasses import dataclass, field
from typing import Any

@dataclass
class Experiment:
    """A container for tracking experiment parameters, metrics, and artifacts, 
    with optional logger callbacks for each update. """
    params : dict = field(default_factory=dict)
    metrics : list =  field(default_factory=list)     
    artifacts : dict = field(default_factory=dict)
    log_registry : list = field(default_factory=list) 
    model: Any = None 

    def add_logger(self, logger):
        self.log_registry.append(logger)

    def add_param(self, param):
        self.params.update(param)

        for logger in self.log_registry:
            if hasattr(logger, 'log_param'):
                logger.log_param(param)

    def add_metrics(self, tag, step, metrics):
        experiment_entry = {
            'TAG' : tag,
            'STEP' : step,
            **metrics
            }
        
        self.metrics.append(experiment_entry)

        for logger in self.log_registry:
            if hasattr(logger, 'log_metrics'):
                logger.log_metrics(experiment_entry)

    def add_artifact(self, name, artifact):
        self.artifacts.update({str(name): artifact})

        for logger in self.log_registry:
            if hasattr(logger, 'log_artifact'):
                logger.log_artifact(name, artifact)


    def add_model(self, model):
        self.model = model

        for logger in self.log_registry:
            if hasattr(logger, 'log_model'):
                logger.log_model(model)


