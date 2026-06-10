from __future__ import annotations

import random
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.loader import DataLoader

from .config import DataConfig
from .features import molecule_to_graph
from .substructures import extract_substructures


SUBSTRUCTURE_MAX_SMILES_LENGTH = 300
SUBSTRUCTURE_MAX_ATOMS = 200
DATA_LOADER_WORKERS = 0


@dataclass
class DatasetSplits:
    train: list
    valid: list
    test: list


def load_dataset(config: DataConfig) -> list:
    frame = pd.read_csv(Path(config.csv_path))
    required = {config.smiles_column, config.label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    records = []
    for row_id, row in frame.iterrows():
        smiles = str(row[config.smiles_column]).strip()
        try:
            graph = molecule_to_graph(smiles, float(row[config.label_column]))
            graph.row_id = int(row_id)
            graph.substructures_json = json.dumps(
                extract_substructures(
                    smiles,
                    method=config.decomposition,
                    max_smiles_length=SUBSTRUCTURE_MAX_SMILES_LENGTH,
                    max_atoms=SUBSTRUCTURE_MAX_ATOMS,
                )
            )
            records.append(graph)
        except (TypeError, ValueError):
            continue
    if not records:
        raise ValueError("No valid molecules were loaded")
    return records


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"invalid:{smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold or "[NO_SCAFFOLD]"


def scaffold_split(records: list, ratios=(0.8, 0.1, 0.1), seed: int = 1) -> DatasetSplits:
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1")

    groups: dict[str, list] = {}
    for graph in records:
        groups.setdefault(scaffold_key(graph.smiles), []).append(graph)

    rng = random.Random(seed)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)
    names = ("train", "valid", "test")
    targets = dict(zip(names, [ratio * len(records) for ratio in ratios]))
    result = {name: [] for name in names}
    for group in grouped:
        destination = min(
            names,
            key=lambda name: (
                max(0.0, len(result[name]) + len(group) - targets[name]),
                -(targets[name] - len(result[name])),
                len(result[name]),
            ),
        )
        result[destination].extend(group)
    splits = DatasetSplits(**result)
    assert_scaffold_disjoint(splits)
    return splits


def assert_scaffold_disjoint(splits: DatasetSplits) -> None:
    scaffold_sets = {
        split_name: {
            scaffold_key(graph.smiles)
            for graph in getattr(splits, split_name)
        }
        for split_name in ("train", "valid", "test")
    }
    comparisons = (("train", "valid"), ("train", "test"), ("valid", "test"))
    for left, right in comparisons:
        overlap = scaffold_sets[left].intersection(scaffold_sets[right])
        if overlap:
            raise RuntimeError(
                f"Scaffold leakage between {left} and {right}: {sorted(overlap)[:5]}"
            )


def make_loader(
    records: list,
    batch_size: int,
    shuffle: bool,
    num_workers: int = DATA_LOADER_WORKERS,
):
    return DataLoader(
        records,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
