from __future__ import annotations

import csv
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import ExperimentConfig
from .data import load_dataset, make_loader, scaffold_key, scaffold_split
from .metrics import binary_metrics
from .models import (
    ConditionalEnvironmentPredictor,
    EnvironmentClassifier,
    MoodTOXModel,
)

ASSISTANT_EPOCHS = 20
FOCAL_GAMMA = 2.0
PRIOR_TYPE = "uniform"
KL_WEIGHT = 1.0
GROUP_LOSS_WEIGHT = 1.0
DEVIATION_LOSS_WEIGHT = 1.0
MOLECULAR_BACKBONE_RADIUS = 20
AUXILIARY_BACKBONE_RADIUS = 2
SELECTION_METRIC = "auc"


@dataclass(frozen=True)
class HyperParameters:
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_layers: int
    graph_feat_size: int
    dropout: float
    epochs: int


def default_hyperparameters(config: ExperimentConfig) -> HyperParameters:
    grid = config.grid_search

    def choose(values, preferred):
        return preferred if preferred in values else values[0]

    return HyperParameters(
        learning_rate=choose(grid.learning_rate, 1e-3),
        weight_decay=choose(grid.weight_decay, 1e-5),
        batch_size=choose(grid.batch_size, 128),
        num_layers=choose(grid.num_layers, 2),
        graph_feat_size=choose(grid.graph_feat_size, 256),
        dropout=choose(grid.dropout, 0.2),
        epochs=choose(grid.epochs, 100),
    )


def seed_everything(seed: int) -> None:
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def batch_substructures(batch) -> list[list[str]]:
    values = batch.substructures_json
    if isinstance(values, str):
        values = [values]
    return [json.loads(value) for value in values]


def evaluate(model, loader, device, return_rows=False):
    model.eval()
    labels, probabilities, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch, batch_substructures(batch))
            probs = torch.sigmoid(logits)
            labels.extend(batch.y.view(-1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            if return_rows:
                for smiles, label, prob in zip(batch.smiles, batch.y.view(-1), probs):
                    rows.append({
                        "SMILES": smiles,
                        "label": float(label.cpu()),
                        "probability": float(prob.cpu()),
                    })
    return binary_metrics(labels, probabilities), rows


def _selection_value(metrics: dict[str, float], name: str) -> float:
    value = float(metrics[name])
    return value if math.isfinite(value) else float("-inf")


def focal_bce_with_logits(logits, labels, pos_weight, reduction="mean"):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    probabilities = torch.sigmoid(logits)
    pt = probabilities * labels + (1.0 - probabilities) * (1.0 - labels)
    loss = torch.pow(1.0 - pt, FOCAL_GAMMA) * bce
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def _environment_prior(k: int, device: torch.device):
    if k < 1:
        raise ValueError("Environment parameter K must be at least 1")
    if PRIOR_TYPE != "uniform":
        raise ValueError("Only a uniform environment prior is currently supported")
    return torch.full((k,), 1.0 / k, device=device)


def _conditional_losses(
    predictor,
    graph_batch,
    k: int,
    pos_weight,
):
    losses = []
    batch_size = graph_batch.num_graphs
    for environment_id in range(k):
        environment_ids = torch.full(
            (batch_size,),
            environment_id,
            dtype=torch.long,
            device=graph_batch.x.device,
        )
        logits = predictor(graph_batch, environment_ids)
        losses.append(
            focal_bce_with_logits(
                logits,
                graph_batch.y.view(-1),
                pos_weight=pos_weight,
                reduction="none",
            )
        )
    return torch.stack(losses, dim=1)


def train_environment_models(
    config,
    hyperparameters,
    train_loader,
    valid_loader,
    device,
    pos_weight,
):
    model_args = {
        "graph_feat_size": hyperparameters.graph_feat_size,
        "num_layers": hyperparameters.num_layers,
        "radius": AUXILIARY_BACKBONE_RADIUS,
        "dropout": hyperparameters.dropout,
    }
    classifier = EnvironmentClassifier(
        **model_args,
        k=config.runtime.environment_k,
    ).to(device)
    conditional = ConditionalEnvironmentPredictor(
        **model_args,
        k=config.runtime.environment_k,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(classifier.parameters()) + list(conditional.parameters()),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    prior = _environment_prior(config.runtime.environment_k, device)
    best_state = None
    best_valid_loss = float("inf")

    def run_epoch(loader, training):
        classifier.train(training)
        conditional.train(training)
        epoch_losses = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for graph_batch in loader:
                graph_batch = graph_batch.to(device)
                posterior = torch.softmax(classifier(graph_batch), dim=-1)
                conditional_losses = _conditional_losses(
                    conditional,
                    graph_batch,
                    config.runtime.environment_k,
                    pos_weight,
                )
                expected_risk = torch.sum(
                    posterior * conditional_losses, dim=1
                ).mean()
                kl = torch.sum(
                    posterior
                    * (
                        torch.log(posterior.clamp_min(1e-8))
                        - torch.log(prior.clamp_min(1e-8))
                    )
                )
                loss = expected_risk + KL_WEIGHT * kl
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
        return float(np.mean(epoch_losses))

    for epoch in range(ASSISTANT_EPOCHS):
        train_loss = run_epoch(train_loader, training=True)
        valid_loss = run_epoch(valid_loader, training=False)
        print(
            f"environment_epoch={epoch + 1:03d} "
            f"k={config.runtime.environment_k} "
            f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}"
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {
                "classifier": deepcopy(classifier.state_dict()),
                "conditional": deepcopy(conditional.state_dict()),
            }

    classifier.load_state_dict(best_state["classifier"])
    conditional.load_state_dict(best_state["conditional"])
    classifier.eval()
    conditional.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    for parameter in conditional.parameters():
        parameter.requires_grad_(False)
    return classifier, conditional, prior


def _grouped_classification_loss(logits, labels, groups, pos_weight):
    losses = []
    for group_id in torch.unique(groups):
        mask = groups == group_id
        losses.append(
            focal_bce_with_logits(
                logits[mask],
                labels[mask],
                pos_weight=pos_weight,
            )
        )
    return torch.stack(losses).mean()


def train_model(
    config: ExperimentConfig,
    hyperparameters: HyperParameters,
    splits,
    run_directory: Path,
    evaluate_test: bool = True,
) -> dict:
    seed = config.runtime.seed
    seed_everything(seed)
    device = resolve_device(config.runtime.device)
    loaders = {
        "train": make_loader(
            splits.train, hyperparameters.batch_size, True
        ),
        "valid": make_loader(
            splits.valid, hyperparameters.batch_size, False
        ),
        "test": make_loader(
            splits.test, hyperparameters.batch_size, False
        ),
    }
    if not splits.train or not splits.valid or not splits.test:
        raise ValueError(
            "Bemis-Murcko scaffold split produced an empty train, valid, or test set"
        )
    model = MoodTOXModel(
        graph_feat_size=hyperparameters.graph_feat_size,
        num_layers=hyperparameters.num_layers,
        molecular_radius=MOLECULAR_BACKBONE_RADIUS,
        fragment_radius=AUXILIARY_BACKBONE_RADIUS,
        dropout=hyperparameters.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    labels = [float(graph.y.item()) for graph in splits.train]
    positives = sum(labels)
    negatives = len(labels) - positives
    pos_weight = torch.tensor(
        [negatives / positives if positives and negatives else 1.0], device=device
    )
    environment_classifier, conditional_predictor, environment_prior = (
        train_environment_models(
            config,
            hyperparameters,
            loaders["train"],
            loaders["valid"],
            device,
            pos_weight,
        )
    )

    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "best_validation.pt"
    history_path = run_directory / "history.csv"
    best_score = float("-inf")
    best_epoch = -1
    stale_epochs = 0

    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["epoch", "train_loss", "valid_auc", "valid_mcc", "valid_f1"],
        )
        writer.writeheader()
        for epoch in range(hyperparameters.epochs):
            model.train()
            losses = []
            for batch in loaders["train"]:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch, batch_substructures(batch))
                labels_batch = batch.y.view(-1)
                with torch.no_grad():
                    environment_posterior = environment_classifier(batch)
                    environment_groups = torch.argmax(
                        environment_posterior, dim=-1
                    )
                    conditional_losses = _conditional_losses(
                        conditional_predictor,
                        batch,
                        config.runtime.environment_k,
                        pos_weight,
                    )
                    prior_risk = torch.matmul(
                        conditional_losses,
                        environment_prior,
                    )
                main_losses = focal_bce_with_logits(
                    logits,
                    labels_batch,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                group_loss = _grouped_classification_loss(
                    logits,
                    labels_batch,
                    environment_groups,
                    pos_weight,
                )
                deviation_loss = torch.mean(
                    torch.abs(main_losses - prior_risk)
                )
                loss = (
                    GROUP_LOSS_WEIGHT * group_loss
                    + DEVIATION_LOSS_WEIGHT * deviation_loss
                )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            valid_metrics, _ = evaluate(model, loaders["valid"], device)
            train_loss = float(np.mean(losses))
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_auc": valid_metrics["auc"],
                "valid_mcc": valid_metrics["mcc"],
                "valid_f1": valid_metrics["f1"],
            })
            stream.flush()
            print(
                f"seed={seed} epoch={epoch + 1:03d} train_loss={train_loss:.4f} "
                f"valid_auc={valid_metrics['auc']:.4f} "
                f"valid_mcc={valid_metrics['mcc']:.4f} "
                f"valid_f1={valid_metrics['f1']:.4f}"
            )

            score = _selection_value(valid_metrics, SELECTION_METRIC)
            if best_epoch < 0 or score > best_score:
                best_score, best_epoch, stale_epochs = score, epoch + 1, 0
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "hyperparameters": asdict(hyperparameters),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "selection_split": "valid",
                    "selection_metric": SELECTION_METRIC,
                    "selection_score": best_score,
                    "environment_k": config.runtime.environment_k,
                    "environment_classifier_state": (
                        environment_classifier.state_dict()
                    ),
                    "conditional_environment_state": (
                        conditional_predictor.state_dict()
                    ),
                }, checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= config.runtime.early_stopping_patience:
                    print(
                        f"early_stopping epoch={epoch + 1} "
                        f"reason=validation_{SELECTION_METRIC}_not_improved_for_"
                        f"{config.runtime.early_stopping_patience}_epochs"
                    )
                    break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    train_metrics, _ = evaluate(model, loaders["train"], device)
    valid_metrics, _ = evaluate(model, loaders["valid"], device)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_split": "valid",
        "environment_k": config.runtime.environment_k,
        "hyperparameters": asdict(hyperparameters),
        "train": train_metrics,
        "valid": valid_metrics,
        "checkpoint": str(checkpoint_path),
        "split_sizes": {
            "train": len(splits.train),
            "valid": len(splits.valid),
            "test": len(splits.test),
        },
        "unique_scaffolds": {
            split_name: len({
                scaffold_key(graph.smiles)
                for graph in getattr(splits, split_name)
            })
            for split_name in ("train", "valid", "test")
        },
    }
    if evaluate_test:
        test_metrics, test_rows = evaluate(
            model, loaders["test"], device, return_rows=True
        )
        result["test"] = test_metrics
        if config.runtime.save_predictions:
            with (run_directory / "test_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["SMILES", "label", "probability"]
                )
                writer.writeheader()
                writer.writerows(test_rows)
        print(f"final_test={json.dumps(test_metrics)}")

    (run_directory / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    return result


def _write_split_manifest(splits, output_path: Path) -> None:
    rows = []
    for split_name in ("train", "valid", "test"):
        for graph in getattr(splits, split_name):
            rows.append({
                "split": split_name,
                "SMILES": graph.smiles,
                "row_id": int(graph.row_id),
                "bemis_murcko_scaffold": scaffold_key(graph.smiles),
            })
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["split", "SMILES", "row_id", "bemis_murcko_scaffold"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_and_evaluate_test(
    config,
    hyperparameters,
    splits,
    checkpoint_path,
    output_directory,
):
    device = resolve_device(config.runtime.device)
    model = MoodTOXModel(
        graph_feat_size=hyperparameters.graph_feat_size,
        num_layers=hyperparameters.num_layers,
        molecular_radius=MOLECULAR_BACKBONE_RADIUS,
        fragment_radius=AUXILIARY_BACKBONE_RADIUS,
        dropout=hyperparameters.dropout,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loader = make_loader(
        splits.test,
        hyperparameters.batch_size,
        False,
    )
    test_metrics, test_rows = evaluate(model, test_loader, device, return_rows=True)
    if config.runtime.save_predictions:
        with (output_directory / "test_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["SMILES", "label", "probability"]
            )
            writer.writeheader()
            writer.writerows(test_rows)
    return test_metrics


def run_experiment(config: ExperimentConfig) -> dict:
    records = load_dataset(config.data)
    output_directory = Path(config.runtime.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    splits = scaffold_split(records, config.data.split, config.runtime.seed)
    _write_split_manifest(splits, output_directory / "scaffold_split.csv")
    hyperparameters = default_hyperparameters(config)
    result = train_model(
        config,
        hyperparameters,
        splits,
        output_directory / "single_run",
        evaluate_test=True,
    )
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    return result


def run_grid_search(config: ExperimentConfig) -> dict:
    records = load_dataset(config.data)
    output_directory = Path(config.runtime.output_directory) / "grid_search"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Every trial uses this exact Bemis-Murcko scaffold split.
    splits = scaffold_split(records, config.data.split, config.runtime.seed)
    _write_split_manifest(splits, output_directory / "scaffold_split.csv")

    grid = config.grid_search
    combinations = product(
        grid.learning_rate,
        grid.weight_decay,
        grid.batch_size,
        grid.num_layers,
        grid.graph_feat_size,
        grid.dropout,
        grid.epochs,
    )
    best_result = None
    best_hyperparameters = None
    trial_results = []
    for trial_index, values in enumerate(combinations, start=1):
        if grid.max_trials is not None and trial_index > grid.max_trials:
            break
        (
            learning_rate,
            weight_decay,
            batch_size,
            num_layers,
            graph_feat_size,
            dropout,
            epochs,
        ) = values
        hyperparameters = HyperParameters(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
            epochs=epochs,
        )
        parameters = asdict(hyperparameters)
        print(f"grid_trial={trial_index} parameters={json.dumps(parameters)}")
        trial_directory = output_directory / f"trial_{trial_index:05d}"
        result = train_model(
            config,
            hyperparameters,
            splits,
            trial_directory,
            evaluate_test=False,
        )
        result["trial"] = trial_index
        result["parameters"] = parameters
        trial_results.append(result)

        score = _selection_value(result["valid"], SELECTION_METRIC)
        if best_result is None or score > _selection_value(
            best_result["valid"], SELECTION_METRIC
        ):
            best_result = result
            best_hyperparameters = hyperparameters

    if best_result is None:
        raise ValueError("Grid search produced no trials")

    test_metrics = _load_and_evaluate_test(
        config,
        best_hyperparameters,
        splits,
        best_result["checkpoint"],
        output_directory,
    )
    summary = {
        "seed": config.runtime.seed,
        "environment_k": config.runtime.environment_k,
        "split_method": "Bemis-Murcko scaffold split",
        "selection_split": "valid",
        "selection_metric": SELECTION_METRIC,
        "best_trial": best_result["trial"],
        "best_parameters": best_result["parameters"],
        "train": best_result["train"],
        "valid": best_result["valid"],
        "test": test_metrics,
        "checkpoint": best_result["checkpoint"],
        "num_trials": len(trial_results),
    }
    (output_directory / "grid_results.json").write_text(
        json.dumps(trial_results, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output_directory / "best_result.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(f"best_grid_result={json.dumps(summary)}")
    return summary
