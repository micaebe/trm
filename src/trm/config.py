from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration class for TRM training."""
    device: str = "cpu"
    dim: int = 128
    seq_len: int = 81
    vocab_size: int = 10
    n_layers: int = 2
    n_recursions: int = 6
    t_recursions: int = 3
    use_attention: bool = False
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    eval_every: int = 10
    use_ema: bool = True
    n_supervision: int = 16
    ema_decay: float = 0.999
    eval_batch_size: int = 32
    print_interval: int = 1
