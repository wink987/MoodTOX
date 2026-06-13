from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DataConfig:
    csv_path: str = "data/example.csv"
    smiles_column: str = "SMILES"
    label_column: str = "label"
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    decomposition: str = "brics"
    max_smiles_length: int = 300
    max_atoms: int = 200
    min_fragment_atoms: int = 2
    max_fallback_fragments: int = 10


@dataclass
class ModelRuntimeConfig:
    num_layers: int = 2
    num_timesteps: int = 2
    auxiliary_size_ratio: float = 0.5


@dataclass
class TrainingRuntimeConfig:
    seed: int = 1
    environment_k: int = 10
    assistant_epochs: int = 20
    focal_gamma: float = 2.0
    prior_type: str = "uniform"
    kl_weight: float = 1.0
    lambda_loss: float = 1.0
    deviation_loss_weight: float = 1.0
    early_stopping_patience: int = 50
    selection_metric: str = "auc"
    device: str = "auto"
    num_workers: int = 0
    output_directory: str = "outputs"
    save_predictions: bool = True


@dataclass
class GridSearchConfig:
    max_trials: int | None = None
    learning_rate: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    weight_decay: tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3)
    batch_size: tuple[int, ...] = (32, 64, 128, 256)
    graph_feat_size: tuple[int, ...] = (32, 64, 128, 256)
    dropout: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)
    epochs: tuple[int, ...] = (100, 200, 300, 500)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model_runtime: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    training_runtime: TrainingRuntimeConfig = field(
        default_factory=TrainingRuntimeConfig
    )
    grid_search: GridSearchConfig = field(default_factory=GridSearchConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONFIG = ExperimentConfig()


def get_config() -> ExperimentConfig:
    return CONFIG
