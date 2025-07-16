from dataclasses import dataclass, asdict, field
from typing import List, Callable,  Dict, Any, Optional
from pathlib import Path

@dataclass
class Config:
    """Configuration container for training pipeline components, hyperparameters, and runtime options."""
    batch_size: int
    epochs: int
    samples: int
    loss: str

    dataset_factory: Callable
    model_cls: Callable
    optimizer_cls: Callable
    logger_cls: Callable

    metrics: List[str] = field(default_factory=list)
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    logger_kwargs: Dict[str, Any] = field(default_factory=dict)

    scheduler_cls: Optional[Callable] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    early_stopper_cls: Optional[Callable] = None
    early_stopper_kwargs: Dict[str, Any] = field(default_factory=dict)

    random_state: int = 42
    threads: int = 1

    def to_simple_dict(self) -> Dict[str, Any]:
        """Return a dictionary of config values with non-basic types simplified to readable strings."""
        SIMPLE_TYPES = (int, float, str, bool, type(None))

        def simplify(value):
            if isinstance(value, SIMPLE_TYPES):
                return value
            if callable(value):
                return value.__name__
            if isinstance(value, dict):
                return {k: simplify(v) for k, v in value.items()}
            if isinstance(value, list):
                return [simplify(v) for v in value]
            return str(value)
    
        object_dict = asdict(self)
        return {k: simplify(v) for k, v in object_dict.items()}