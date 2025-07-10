import numpy as np
import random


def set_random_state(seed=None):
    """ Sets the global random state. """
    if not isinstance(seed, int):
        seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)


class EarlyStopping:
    "Provides a Counter for halting training when no sufficient improvement is observed over patience period"
    def __init__(self, patience: int = 5, delta: float = 1e-5, minimize: bool = True):
        self.patience = patience
        self.delta = delta
        self.minimize = minimize
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score: float) -> None:
        if self.best_score is None:
            self.best_score = current_score
            return

        improvement = (
            self.best_score - current_score
            if self.minimize else
            current_score - self.best_score
        )

        if improvement > self.delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True