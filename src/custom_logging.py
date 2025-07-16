import json
import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class BaseLogger(ABC):
    def log_param(self, param):
        self._validate_dict(param, "param")
        self._log_param(param)

    def log_metrics(self, metrics):
        self._validate_dict(metrics, "metrics")
        self._log_metrics(metrics)

    def log_artifact(self, name, artifact):
        self._validate_figure(artifact)
        self._log_artifact(name, artifact)

    @staticmethod
    def _validate_dict(obj, name):
        if not isinstance(obj, dict):
            raise ValueError(f"{name} must be a dictionary.")

    @staticmethod
    def _validate_figure(artifact):
        if not isinstance(artifact, plt.Figure):
            raise ValueError("Artifact must be a matplotlib.figure.Figure.")

    @abstractmethod
    def _log_param(self, param): ...
    
    @abstractmethod
    def _log_metrics(self, metrics): ...
    
    @abstractmethod
    def _log_artifact(self, name, artifact): ...


class ConsoleLogger(BaseLogger):
    """Logger that prints parameters, metrics, and artifacts to the console."""
    def __init__(self, log_dir, experiment_name=None):
        pass

    def _log_param(self, param):
        """Displays static parameters to the console."""
        for k, v in param.items():
            print(f"PARAM:     {k}: {v}")

    def _log_metrics(self, metrics):
        """Displays training or evaluation metrics to the console."""
        #for k, v in metrics.items():
        print(f"METRICS:    {metrics}")

    def _log_artifact(self, name, artifact, max_windows=10):
        """Displays the artifact if fewer than max_windows figures are open."""
        #plt.figure(artifact)
        #plt.show()
        open_figs = plt.get_fignums()
        if len(open_figs) >= max_windows:
            print('Artifact output limit exceeded')
            return
        artifact.show()


class SaveLogger(BaseLogger):
    """Logger for saving parameters, metrics, and artifacts during an experiment."""
    def __init__(self, log_dir, experiment_name=None, model_save_fn=None):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or generate_timestamp()
        self.log_path = self.log_dir / self.experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.model_save_fn = model_save_fn

    def _log_param(self, param, filename='params.json'):
        """Logs static parameters to a JSON file."""
        file_path = self.log_path / filename
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}

        data.update(param)

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _log_metrics(self, metrics, filename='metrics.jsonl'):
        """Logs metrics by appending to a JSONL file."""
        with open(self.log_path / filename, 'a') as f:
            f.write(json.dumps(metrics)+ "\n")

    def _log_artifact(self, name, artifact):
        """Saves a plot of the given artifact as a PNG image."""
        #plt.figure(artifact)
        #plt.savefig(self.log_path / f"{name}.png")
        artifact.savefig(self.log_path / f"{name}.png")
        plt.close(artifact)

    def log_model(self, model, filename=None):
        """Saves a model object given a defined model-save callable."""
        if filename is None:
            filename = "model.pt"
        file_path = self.log_path / filename
        self.model_save_fn(model, file_path)

    def log_info(self, filename, content, append= False):
        """Writes simple content to a dat file."""
        file_path = self.log_path / f"{filename}.dat"
        mode = 'a' if append else 'w'
        with open(file_path, mode) as f:
            f.write(f"{content}\n")

        


def generate_timestamp() -> str:
    """ Returns a timestamp in format 'HH-MM-DD-mm-YYYY'. """
    return datetime.datetime.now().strftime("%H-%M-%d-%m-%Y")