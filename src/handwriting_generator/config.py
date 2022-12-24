from dataclasses import dataclass

__all__ = ["Parameters"]


@dataclass
class Parameters:
    train_model: bool = True
    use_gpu: bool = True
    # Model Parameters
    input_size: int = 3
    hidden_size: int = 200
    n_window_components: int = 10
    n_mixture_components: int = 20
    probability_bias: float = 1.0
    # Training Parameters
    n_epochs: int = 2
    batch_size: int = 32
    max_norm: int = 400
    # Optimizer Parameters
    learning_rate: float = 1.0e-4
