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


@dataclass
class RuntimeConfig:
    seed: int = 1
    environment_k: int = 10
    early_stopping_patience: int = 50
    device: str = "auto"
    output_directory: str = "outputs"
    save_predictions: bool = True


@dataclass
class GridSearchConfig:
    enabled: bool = False
    max_trials: int | None = None
    learning_rate: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    weight_decay: tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3)
    batch_size: tuple[int, ...] = (32, 64, 128, 256)
    num_layers: tuple[int, ...] = (2, 3, 4, 5)
    graph_feat_size: tuple[int, ...] = (32, 64, 128, 256)
    dropout: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)
    epochs: tuple[int, ...] = (100, 200, 300, 500)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    grid_search: GridSearchConfig = field(default_factory=GridSearchConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONFIG = ExperimentConfig()


def get_config() -> ExperimentConfig:
    return CONFIG
