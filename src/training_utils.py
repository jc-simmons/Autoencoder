import numpy as np
import random


def set_random_state(seed=None):
    """ Sets the global random state. """
    if not isinstance(seed, int):
        seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)


class StoppingPatience:
    "Provides a Counter for halting training when no sufficient improvement is observed over patience period"
    def __init__(self, patience = 5):
        self.patience = patience
        self.counter = 0

    def check_stop(self, improved) :
        self.counter = 0 if improved else self.counter + 1
        return self.counter >= self.patience
    

def check_improvement(best_score, current_score, delta = 1e-5, minimize = True):
    improvement = best_score - current_score if minimize else current_score - best_score
    return improvement > delta

    