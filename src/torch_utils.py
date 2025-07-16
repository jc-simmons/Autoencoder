import torch
from contextlib import contextmanager
from enum import Enum, auto

# ---------- Model Stage Management ----------
class Stage(Enum):
    """Enum representing model stages with context management."""
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


# ---------- Training Runner ----------
def create_runner(model, evaluator, optimizer, device='cpu', loss=None):
    """ Returns a runner function that processes a data loader, performs evaluation, 
    optional training, and computes metrics. """
    def run(loader):
        running_totals = {}

    
        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            batch_metrics = evaluator(model, X, y)

            if model.training:
                batch_loss = batch_metrics[loss]
                optimizer.zero_grad() 
                batch_loss.backward()
                optimizer.step()

            for metric, value in batch_metrics.items():
                detached_value = value.detach()
                if metric not in running_totals:
                    running_totals[metric] = detached_value.clone()
                else:
                    running_totals[metric] += detached_value


        averages = {metric: value.item() / len(loader) for metric, value in running_totals.items()}
        return averages
    return run


# ---------- Model Saving ----------
def save_torch_model(model, file_name):
    """Saves a PyTorch model's state dict to file."""
    torch.save(model.state_dict(), file_name)