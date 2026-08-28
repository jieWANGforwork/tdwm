#!/usr/bin/env python3
"""Render a document-ready PNG from the normalized Actor-Free loss CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "reports/artifacts/actor_free_td_lewm_cube_seed3072/training_loss_curves.csv"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "reports/artifacts/actor_free_td_lewm_cube_seed3072/training_loss_curves.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the validated 7-method training/validation total-loss curves."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    curves: dict[str, list[dict[str, str]]] = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            curves[row["variant"]].append(row)
    if not curves:
        raise SystemExit(f"No loss rows found in {source}.")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), dpi=180)
    for variant, rows in curves.items():
        epochs = [int(row["epoch"]) for row in rows]
        label = rows[0]["display_name"]
        axes[0].plot(
            epochs,
            [float(row["train_loss"]) for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=label,
        )
        axes[1].plot(
            epochs,
            [float(row["validation_loss"]) for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=label,
        )
    for axis, title in zip(axes, ("Training total loss", "Validation total loss")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Method total loss")
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Actor-Free TD-LeWM loss curves", fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Auxiliary-loss definitions differ; compare convergence within a method only.",
        ha="center",
        color="#5C6975",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"Software": "tdwm"})
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
