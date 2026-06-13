from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data


ATOM_SYMBOLS = [
    "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br",
    "Te", "I", "At", "other",
]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices, unknown: bool = False) -> list[float]:
    if unknown and value not in choices:
        value = choices[-1]
    return [float(value == choice) for choice in choices]


def atom_features(atom: Chem.Atom) -> list[float]:
    symbol = atom.GetSymbol() if atom.GetSymbol() in ATOM_SYMBOLS[:-1] else "other"
    values = (
        one_hot(symbol, ATOM_SYMBOLS)
        + one_hot(atom.GetDegree(), list(range(14)))
        + [float(atom.GetFormalCharge()), float(atom.GetNumRadicalElectrons())]
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS, unknown=True)
        + [float(atom.GetIsAromatic())]
        + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4], unknown=True)
    )
    try:
        values += one_hot(atom.GetProp("_CIPCode"), ["R", "S"], unknown=True)
    except KeyError:
        values += [0.0, 0.0]
    values += [float(atom.HasProp("_ChiralityPossible"))]
    return values


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        one_hot(bond.GetBondType(), BOND_TYPES, unknown=True)
        + [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]
        + one_hot(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
            unknown=True,
        )
    )


def molecule_to_graph(smiles: str, label: float | None = None) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edges: list[list[int]] = []
    attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edges.extend([[i, j], [j, i]])
        attrs.extend([feat, feat])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_feature_dim()), dtype=torch.float32)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if label is not None:
        graph.y = torch.tensor([float(label)], dtype=torch.float32)
    return graph


def atom_feature_dim() -> int:
    return len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


def bond_feature_dim() -> int:
    return len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))
