from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Batch

from .features import molecule_to_graph
from .models import MoodTOXModel
from .substructures import extract_substructures, match_fragment_atoms


def load_checkpoint(path: str | Path, device: str = "cpu"):
    target_device = torch.device(device)
    payload = torch.load(path, map_location=target_device, weights_only=False)
    required = {
        "model_state",
        "hyperparameters",
        "environment_k",
        "config",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"Checkpoint is not a compatible MoodTOX checkpoint; missing {sorted(missing)}"
        )
    hyperparameters = payload["hyperparameters"]
    config = payload["config"]
    model_runtime = config["model_runtime"]
    model = MoodTOXModel(
        graph_feat_size=hyperparameters["graph_feat_size"],
        auxiliary_feat_size=max(
            1,
            round(
                hyperparameters["graph_feat_size"]
                * model_runtime["auxiliary_size_ratio"]
            ),
        ),
        num_layers=model_runtime["num_layers"],
        num_timesteps=model_runtime["num_timesteps"],
        dropout=hyperparameters["dropout"],
    ).to(target_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def predict(model, smiles: str, data_config: dict, device: torch.device):
    graph = molecule_to_graph(smiles)
    fragments = extract_substructures(
        smiles,
        method=data_config["decomposition"],
        max_smiles_length=data_config["max_smiles_length"],
        max_atoms=data_config["max_atoms"],
        min_fragment_atoms=data_config["min_fragment_atoms"],
        max_fallback_fragments=data_config["max_fallback_fragments"],
    )
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(batch, [fragments]))[0].item()
    atom_weights = model.molecular_backbone.last_atom_attention.view(-1).cpu().numpy()
    fragment_weights = model.last_substructure_attention[0].cpu().numpy()
    bond_attention = build_chemical_bond_attention_matrix(model, batch)
    return probability, fragments, atom_weights, fragment_weights, bond_attention


def build_chemical_bond_attention_matrix(model, graph_batch):
    num_atoms = int(graph_batch.num_nodes)
    edge_index = graph_batch.edge_index.detach().cpu().numpy()
    pair_sums = {}
    pair_counts = {}
    layer_attentions = [
        scores
        for scores in model.molecular_backbone.last_bond_attentions
        if scores is not None
    ]
    if not layer_attentions:
        return np.zeros((num_atoms, num_atoms), dtype=np.float32)

    scores = layer_attentions[-1].detach().view(-1).cpu().numpy()
    for edge_id, score in enumerate(scores):
        source = int(edge_index[0, edge_id])
        target = int(edge_index[1, edge_id])
        if source == target:
            continue
        pair = tuple(sorted((source, target)))
        pair_sums[pair] = pair_sums.get(pair, 0.0) + float(score)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)
    for (atom_i, atom_j), score_sum in pair_sums.items():
        mean_score = score_sum / pair_counts[(atom_i, atom_j)]
        matrix[atom_i, atom_j] = mean_score
        matrix[atom_j, atom_i] = mean_score
    return matrix


def _color(value: float):
    return (1.0, 1.0 - 0.65 * value, 1.0 - 0.9 * value)


def draw_atom_attention(smiles: str, weights, output_path: Path):
    mol = Chem.MolFromSmiles(smiles)
    normalized = np.asarray(weights, dtype=float)
    normalized /= max(float(normalized.max()), 1e-12)
    colors = {idx: _color(float(value)) for idx, value in enumerate(normalized)}
    drawer = rdMolDraw2D.MolDraw2DSVG(800, 520)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(colors),
        highlightAtomColors=colors,
        highlightAtomRadii={idx: 0.35 for idx in colors},
    )
    drawer.FinishDrawing()
    output_path.write_text(drawer.GetDrawingText(), encoding="utf-8")


def draw_substructure_attention(smiles, fragments, weights, output_path: Path):
    mol = Chem.MolFromSmiles(smiles)
    atom_scores = np.zeros(mol.GetNumAtoms(), dtype=float)
    for fragment, weight in zip(fragments, weights):
        for atom_idx in match_fragment_atoms(mol, fragment):
            atom_scores[atom_idx] = max(atom_scores[atom_idx], float(weight))
    atom_scores /= max(float(atom_scores.max()), 1e-12)
    colors = {idx: _color(float(value)) for idx, value in enumerate(atom_scores)}
    drawer = rdMolDraw2D.MolDraw2DSVG(800, 520)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(colors),
        highlightAtomColors=colors,
        highlightAtomRadii={idx: 0.4 for idx in colors},
    )
    drawer.FinishDrawing()
    output_path.write_text(drawer.GetDrawingText(), encoding="utf-8")


def draw_fragment_chart(fragments, weights, output_path: Path):
    order = np.argsort(weights)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(fragments))))
    ax.barh(np.asarray(fragments)[order], np.asarray(weights)[order], color="#6f86d6")
    ax.set_xlabel("Normalized attention")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_chemical_bond_visualization_heatmap(
    smiles: str,
    matrix,
    output_path: Path,
):
    mol = Chem.MolFromSmiles(smiles)
    matrix = np.asarray(matrix, dtype=np.float32)
    labels = [
        f"{atom.GetIdx()}:{atom.GetSymbol()}"
        for atom in mol.GetAtoms()
    ]
    size = max(6.0, 0.52 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size), dpi=180)
    vmax = max(float(matrix.max()), 1e-12)
    image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target atom")
    ax.set_ylabel("Source atom")
    ax.set_title("Chemical Bond Visualization Heatmap")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    if len(labels) <= 18:
        for source in range(len(labels)):
            for target in range(len(labels)):
                value = float(matrix[source, target])
                if value > 0:
                    ax.text(
                        target,
                        source,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if value >= 0.55 * vmax else "black",
                    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Chemical bond attention")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def visualize_checkpoint(checkpoint, smiles, output_dir, device="cpu"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, payload = load_checkpoint(checkpoint, device)
    data_config = payload["config"]["data"]
    probability, fragments, atom_weights, fragment_weights, bond_attention = predict(
        model, smiles, data_config, torch.device(device)
    )
    draw_atom_attention(smiles, atom_weights, output_dir / "atom_attention.svg")
    draw_substructure_attention(
        smiles, fragments, fragment_weights, output_dir / "substructure_attention.svg"
    )
    draw_fragment_chart(fragments, fragment_weights, output_dir / "fragment_attention.png")
    heatmap_path = output_dir / "chemical_bond_visualization_heatmap.svg"
    draw_chemical_bond_visualization_heatmap(
        smiles,
        bond_attention,
        heatmap_path,
    )
    summary = {
        "smiles": smiles,
        "probability": probability,
        "fragments": [
            {"smiles": fragment, "attention": float(weight)}
            for fragment, weight in zip(fragments, fragment_weights)
        ],
        "chemical_bond_visualization_heatmap": str(heatmap_path),
        "chemical_bond_attention_matrix": bond_attention.tolist(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
