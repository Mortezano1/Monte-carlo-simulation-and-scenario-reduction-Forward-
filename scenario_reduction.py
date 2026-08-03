from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_scenarios(count: int = 1000, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.array([90.0, 30.0, 15.0])
    stds = np.array([10.0, 4.0, 2.0])
    scenarios = rng.normal(loc=means, scale=stds, size=(count, 3))
    probabilities = np.full(count, 1.0 / count, dtype=float)
    return scenarios, probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and reduce a set of weighted scenarios.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    if "probability" in df.columns:
        probabilities = df["probability"].astype(float).to_numpy()
    else:
        probabilities = np.full(len(df), 1.0 / len(df), dtype=float)

    if {"scenario_1", "scenario_2", "scenario_3"}.issubset(df.columns):
        columns = ["scenario_1", "scenario_2", "scenario_3"]
    elif {"x", "y", "z"}.issubset(df.columns):
        columns = ["x", "y", "z"]
    else:
        raise ValueError("Input must contain scenario_1/scenario_2/scenario_3 or x/y/z columns.")

    probabilities = probabilities / probabilities.sum()
    scenarios = df[columns].to_numpy(dtype=float)
    return scenarios, probabilities, df


def compute_distance_matrix(scenarios: np.ndarray) -> np.ndarray:
    diffs = scenarios[:, None, :] - scenarios[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def reduce_scenarios(scenarios: np.ndarray, probabilities: np.ndarray, target_count: int) -> tuple[np.ndarray, np.ndarray]:
    if target_count >= len(scenarios):
        return scenarios, probabilities

    active_scenarios = scenarios.copy()
    active_probabilities = probabilities.copy()
    active_indices = list(range(len(scenarios)))

    while len(active_scenarios) > target_count:
        distances = compute_distance_matrix(active_scenarios)
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        nearest_indices = np.argmin(distances, axis=1)
        removal_cost = active_probabilities[nearest_indices] * nearest_distances
        remove_idx = int(np.argmin(removal_cost))

        active_indices.pop(remove_idx)
        active_scenarios = np.delete(active_scenarios, remove_idx, axis=0)
        active_probabilities = np.delete(active_probabilities, remove_idx)

    retained = scenarios[active_indices]
    retained_probabilities = np.zeros(len(active_indices), dtype=float)
    index_map = {idx: pos for pos, idx in enumerate(active_indices)}

    for original_idx, original_probability in enumerate(probabilities):
        if original_idx in index_map:
            retained_probabilities[index_map[original_idx]] += original_probability
        else:
            distances = np.linalg.norm(retained - scenarios[original_idx], axis=1)
            nearest_pos = int(np.argmin(distances))
            retained_probabilities[nearest_pos] += original_probability

    return retained, retained_probabilities


def save_outputs(output_dir: Path, original_scenarios: np.ndarray, reduced_scenarios: np.ndarray, reduced_probabilities: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    reduced_frame = pd.DataFrame(reduced_scenarios, columns=["scenario_1", "scenario_2", "scenario_3"])
    reduced_frame["probability"] = reduced_probabilities
    reduced_frame.to_csv(output_dir / "reduced_scenarios.csv", index=False)

    summary = {
        "original_count": len(original_scenarios),
        "reduced_count": len(reduced_scenarios),
        "probability_sum": float(reduced_probabilities.sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(original_scenarios[:, 0], original_scenarios[:, 1], original_scenarios[:, 2], s=12, c="lightgray", label="Original")
    ax.scatter(reduced_scenarios[:, 0], reduced_scenarios[:, 1], reduced_scenarios[:, 2], s=80, c="tab:blue", label="Reduced")
    ax.set_xlabel("Scenario 1")
    ax.set_ylabel("Scenario 2")
    ax.set_zlabel("Scenario 3")
    ax.set_title("Scenario reduction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scenario_reduction_plot.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.input is not None:
        scenarios, probabilities, _ = load_data(args.input)
    else:
        scenarios, probabilities = generate_scenarios(count=args.count, seed=args.seed)
    reduced_scenarios, reduced_probabilities = reduce_scenarios(scenarios, probabilities, args.target_count)
    save_outputs(args.output_dir, scenarios, reduced_scenarios, reduced_probabilities)


if __name__ == "__main__":
    main()
