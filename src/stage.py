import torch
from contextlib import contextmanager
from enum import Enum, auto

class Stage(Enum):
    """Enum representing different stages of model evaluation with context management for each stage."""
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

    @contextmanager
    def context(self, model):
        if self == Stage.TRAIN:
            model.train()
            with torch.set_grad_enabled(True):
                yield self
        else:
            model.eval()
            with torch.inference_mode():
                yield self

    def __call__(self, model): 
        return self.context(model)
