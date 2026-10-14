"""Predeclared additional neural initializations; no test-based model selection."""

import hashlib
import json
from pathlib import Path

from .config import SimConfig
from .deep_experiment import evaluate, write_csv
from .neural import NeuralPredictor, dataset


def main():
    destination = Path("results/training_variability")
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path("results/deep/training_metadata.json").read_text())
    training = dataset(metadata["training_seeds"])
    validation = dataset(metadata["validation_seeds"])
    cfg = SimConfig(common_radio_stream=True, research_metrics=True, prediction_mode="kinematic")
    test_seeds = list(range(9000, 9020))
    initializations = [74, 75, 76]
    summaries = []
    runs = []
    models = []
    for initialization in initializations:
        print(f"Training predeclared initialization {initialization}", flush=True)
        model = NeuralPredictor(initialization)
        history, accuracy = model.fit(*training, validation, epochs=30, seed=initialization)
        checkpoint = destination / f"predictor_seed_{initialization}.npz"
        model.save(
            checkpoint, {**metadata, "training_seed": initialization, "validation": accuracy}
        )
        write_csv(destination / f"training_seed_{initialization}.csv", history)
        print(
            f"Evaluating all 20 test trajectories for initialization {initialization}", flush=True
        )
        aggregate, individual, _ = evaluate(
            "QoS-aware / neural reservation", cfg, test_seeds, 320, model
        )
        summaries.append({"training_seed": initialization, **accuracy, **aggregate})
        runs.extend({"training_seed": initialization, **row} for row in individual)
        models.append(
            {
                "training_seed": initialization,
                "checkpoint": str(checkpoint),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
        write_csv(destination / "initialization_metrics.csv", summaries)
        write_csv(destination / "initialization_metrics_by_test_seed.csv", runs)
    report = {
        "status": "passed",
        "initializations": initializations,
        "test_seeds": test_seeds,
        "training_seeds": metadata["training_seeds"],
        "validation_seeds": metadata["validation_seeds"],
        "epochs": 30,
        "model_selection": "Best validation epoch within each run; no selection among initializations",
        "models": models,
        "metrics": summaries,
    }
    (destination / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
