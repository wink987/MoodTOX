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

@dataclass(frozen=True)
class HyperParameters:
    learning_rate: float
    weight_decay: float
    batch_size: int
    graph_feat_size: int
    dropout: float
    epochs: int


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


def focal_bce_with_logits(
    logits,
    labels,
    class_weights,
    gamma: float,
    reduction="mean",
):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    negative_weight, positive_weight = class_weights
    sample_weights = (
        positive_weight * labels
        + negative_weight * (1.0 - labels)
    )
    probabilities = torch.sigmoid(logits)
    pt = probabilities * labels + (1.0 - probabilities) * (1.0 - labels)
    loss = sample_weights * torch.pow(1.0 - pt, gamma) * bce
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def balanced_class_weights(labels, device):
    labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
    positives = float(labels.sum())
    negatives = float(labels.numel() - labels.sum())
    if positives == 0.0 or negatives == 0.0:
        return torch.ones(2, dtype=torch.float32, device=device)
    total = positives + negatives
    return torch.tensor(
        [
            total / (2.0 * negatives),
            total / (2.0 * positives),
        ],
        dtype=torch.float32,
        device=device,
    )


def _environment_prior(k: int, prior_type: str, device: torch.device):
    if k < 1:
        raise ValueError("Environment parameter K must be at least 1")
    if prior_type == "uniform":
        return torch.full((k,), 1.0 / k, device=device)
    if prior_type == "gaussian":
        normal = torch.distributions.Normal(0.0, 1.0)
        width = 6.0 / k
        centers = torch.arange(
            -3.0 + width / 2.0,
            3.0,
            width,
            device=device,
        )
        prior = normal.cdf(centers + width / 2.0) - normal.cdf(
            centers - width / 2.0
        )
        return prior / prior.sum()
    raise ValueError("prior_type must be 'uniform' or 'gaussian'")


def _conditional_losses(
    predictor,
    graph_batch,
    k: int,
    class_weights,
    focal_gamma: float,
):
    logits = predictor.forward_all_environments(graph_batch)
    labels = graph_batch.y.view(-1, 1).expand(-1, k)
    return focal_bce_with_logits(
        logits,
        labels,
        class_weights=class_weights,
        gamma=focal_gamma,
        reduction="none",
    )


def train_environment_models(
    config,
    hyperparameters,
    train_loader,
    valid_loader,
    device,
    class_weights,
):
    training_runtime = config.training_runtime
    model_runtime = config.model_runtime
    auxiliary_feat_size = max(
        1,
        round(
            hyperparameters.graph_feat_size
            * model_runtime.auxiliary_size_ratio
        ),
    )
    model_args = {
        "graph_feat_size": auxiliary_feat_size,
        "num_layers": model_runtime.num_layers,
        "num_timesteps": model_runtime.num_timesteps,
        "dropout": hyperparameters.dropout,
    }
    classifier = EnvironmentClassifier(
        **model_args,
        k=training_runtime.environment_k,
    ).to(device)
    conditional = ConditionalEnvironmentPredictor(
        **model_args,
        k=training_runtime.environment_k,
    ).to(device)
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    conditional_optimizer = torch.optim.Adam(
        conditional.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    prior = _environment_prior(
        training_runtime.environment_k,
        training_runtime.prior_type,
        device,
    )
    best_state = None
    best_training_loss = float("inf")

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
                    training_runtime.environment_k,
                    class_weights,
                    training_runtime.focal_gamma,
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
                loss = expected_risk + training_runtime.kl_weight * kl
                if training:
                    loss.backward()
                    classifier_optimizer.step()
                    conditional_optimizer.step()
                    classifier_optimizer.zero_grad()
                    conditional_optimizer.zero_grad()
                epoch_losses.append(float(loss.detach().cpu()))
        return float(np.mean(epoch_losses))

    for epoch in range(training_runtime.assistant_epochs):
        train_loss = run_epoch(train_loader, training=True)
        valid_loss = run_epoch(valid_loader, training=False)
        print(
            f"environment_epoch={epoch + 1:03d} "
            f"k={training_runtime.environment_k} "
            f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}"
        )
        if train_loss < best_training_loss:
            best_training_loss = train_loss
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


def _grouped_classification_loss(
    logits,
    labels,
    groups,
    class_weights,
    focal_gamma,
):
    losses = []
    for group_id in torch.unique(groups):
        mask = groups == group_id
        losses.append(
            focal_bce_with_logits(
                logits[mask],
                labels[mask],
                class_weights=class_weights,
                gamma=focal_gamma,
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
    training_runtime = config.training_runtime
    model_runtime = config.model_runtime
    seed = training_runtime.seed
    seed_everything(seed)
    device = resolve_device(training_runtime.device)
    loaders = {
        "train": make_loader(
            splits.train,
            hyperparameters.batch_size,
            True,
            training_runtime.num_workers,
        ),
        "valid": make_loader(
            splits.valid,
            hyperparameters.batch_size,
            False,
            training_runtime.num_workers,
        ),
        "test": make_loader(
            splits.test,
            hyperparameters.batch_size,
            False,
            training_runtime.num_workers,
        ),
    }
    if not splits.train or not splits.valid or not splits.test:
        raise ValueError(
            "Bemis-Murcko scaffold split produced an empty train, valid, or test set"
        )
    model = MoodTOXModel(
        graph_feat_size=hyperparameters.graph_feat_size,
        auxiliary_feat_size=max(
            1,
            round(
                hyperparameters.graph_feat_size
                * model_runtime.auxiliary_size_ratio
            ),
        ),
        num_layers=model_runtime.num_layers,
        num_timesteps=model_runtime.num_timesteps,
        dropout=hyperparameters.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    labels = [float(graph.y.item()) for graph in splits.train]
    class_weights = balanced_class_weights(labels, device)
    environment_classifier, conditional_predictor, environment_prior = (
        train_environment_models(
            config,
            hyperparameters,
            loaders["train"],
            loaders["valid"],
            device,
            class_weights,
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
                        training_runtime.environment_k,
                        class_weights,
                        training_runtime.focal_gamma,
                    )
                    prior_risk = torch.matmul(
                        conditional_losses,
                        environment_prior,
                    )
                main_losses = focal_bce_with_logits(
                    logits,
                    labels_batch,
                    class_weights=class_weights,
                    gamma=training_runtime.focal_gamma,
                    reduction="none",
                )
                group_loss = _grouped_classification_loss(
                    logits,
                    labels_batch,
                    environment_groups,
                    class_weights,
                    training_runtime.focal_gamma,
                )
                deviation_loss = torch.mean(
                    torch.abs(main_losses - prior_risk)
                )
                loss = (
                    training_runtime.lambda_loss * group_loss
                    + training_runtime.deviation_loss_weight
                    * deviation_loss
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

            score = _selection_value(
                valid_metrics, training_runtime.selection_metric
            )
            if best_epoch < 0 or score > best_score:
                best_score, best_epoch, stale_epochs = score, epoch + 1, 0
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "hyperparameters": asdict(hyperparameters),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "selection_split": "valid",
                    "selection_metric": training_runtime.selection_metric,
                    "selection_score": best_score,
                    "environment_k": training_runtime.environment_k,
                    "environment_classifier_state": (
                        environment_classifier.state_dict()
                    ),
                    "conditional_environment_state": (
                        conditional_predictor.state_dict()
                    ),
                }, checkpoint_path)
            else:
                stale_epochs += 1
                if (
                    stale_epochs
                    >= training_runtime.early_stopping_patience
                ):
                    print(
                        f"early_stopping epoch={epoch + 1} "
                        f"reason=validation_"
                        f"{training_runtime.selection_metric}_"
                        f"not_improved_for_"
                        f"{training_runtime.early_stopping_patience}_epochs"
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
        "environment_k": training_runtime.environment_k,
        "hyperparameters": asdict(hyperparameters),
        "model_dimensions": {
            "molecular_backbone": hyperparameters.graph_feat_size,
            "fragment_backbone": max(
                1,
                round(
                    hyperparameters.graph_feat_size
                    * model_runtime.auxiliary_size_ratio
                ),
            ),
            "environment_backbone": max(
                1,
                round(
                    hyperparameters.graph_feat_size
                    * model_runtime.auxiliary_size_ratio
                ),
            ),
            "conditional_backbone": max(
                1,
                round(
                    hyperparameters.graph_feat_size
                    * model_runtime.auxiliary_size_ratio
                ),
            ),
        },
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
        if training_runtime.save_predictions:
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
    training_runtime = config.training_runtime
    model_runtime = config.model_runtime
    device = resolve_device(training_runtime.device)
    model = MoodTOXModel(
        graph_feat_size=hyperparameters.graph_feat_size,
        auxiliary_feat_size=max(
            1,
            round(
                hyperparameters.graph_feat_size
                * model_runtime.auxiliary_size_ratio
            ),
        ),
        num_layers=model_runtime.num_layers,
        num_timesteps=model_runtime.num_timesteps,
        dropout=hyperparameters.dropout,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loader = make_loader(
        splits.test,
        hyperparameters.batch_size,
        False,
        training_runtime.num_workers,
    )
    test_metrics, test_rows = evaluate(model, test_loader, device, return_rows=True)
    if training_runtime.save_predictions:
        with (output_directory / "test_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["SMILES", "label", "probability"]
            )
            writer.writeheader()
            writer.writerows(test_rows)
    return test_metrics


def run_grid_search(config: ExperimentConfig) -> dict:
    records = load_dataset(config.data)
    training_runtime = config.training_runtime
    output_directory = Path(training_runtime.output_directory) / "grid_search"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Every trial uses this exact Bemis-Murcko scaffold split.
    splits = scaffold_split(records, config.data.split, training_runtime.seed)
    _write_split_manifest(splits, output_directory / "scaffold_split.csv")

    grid = config.grid_search
    combinations = product(
        grid.learning_rate,
        grid.weight_decay,
        grid.batch_size,
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
            graph_feat_size,
            dropout,
            epochs,
        ) = values
        hyperparameters = HyperParameters(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
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

        score = _selection_value(
            result["valid"], training_runtime.selection_metric
        )
        if best_result is None or score > _selection_value(
            best_result["valid"], training_runtime.selection_metric
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
        "seed": training_runtime.seed,
        "environment_k": training_runtime.environment_k,
        "split_method": "Bemis-Murcko scaffold split",
        "selection_split": "valid",
        "selection_metric": training_runtime.selection_metric,
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
