#!/usr/bin/env python3
"""Evaluate a saved DSPy program against a dataset.

Loads a serialised DSPy program JSON and runs it against a CSV dataset,
reporting accuracy and other metrics.

Usage::

    python scripts/evaluate_program.py \
        --program my_program.json \
        --module my_pkg.programs --attr MyProgram \
        --dataset eval_set.csv \
        --input-col question \
        --label-col answer
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable


def load_dataset(csv_path: Path, input_col: str, label_col: str) -> list[dict[str, str]]:
    """Load a CSV evaluation dataset.

    Args:
        csv_path: Path to the CSV file.
        input_col: Column name containing input text.
        label_col: Column name containing ground-truth labels.

    Returns:
        List of dicts with ``input`` and ``label`` keys.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({"input": row[input_col], "label": row[label_col]})
    return rows


def exact_match(prediction: str, label: str) -> bool:
    """Case-insensitive exact match metric.

    Args:
        prediction: The program's output string.
        label: The ground-truth label string.

    Returns:
        ``True`` when prediction and label match after normalisation.
    """
    return prediction.strip().lower() == label.strip().lower()


def evaluate(
    program: object,
    dataset: list[dict[str, str]],
    metric: Callable[[str, str], bool] = exact_match,
) -> dict[str, float]:
    """Run a DSPy program against a dataset and compute metrics.

    Args:
        program: A DSPy ``Module`` instance with a ``__call__`` method.
        dataset: List of ``{input, label}`` dicts.
        metric: Callable accepting ``(prediction, label)`` and returning bool.

    Returns:
        Dict with ``accuracy`` and ``total`` keys.
    """
    correct = 0
    for ex in dataset:
        try:
            pred = program(ex["input"])
            pred_str = str(getattr(pred, "answer", pred))
            if metric(pred_str, ex["label"]):
                correct += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  Error on example: {exc}")
    total = len(dataset)
    return {"accuracy": correct / total if total else 0.0, "total": total}


def main() -> None:
    """Entry point for the program evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a DSPy program.")
    parser.add_argument("--program", required=True, type=Path, help="Path to saved program JSON.")
    parser.add_argument("--module", required=True, help="Module containing the program class.")
    parser.add_argument("--attr", required=True, help="Class name of the DSPy program.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to CSV dataset.")
    parser.add_argument("--input-col", default="question", help="Input column name.")
    parser.add_argument("--label-col", default="answer", help="Label column name.")
    args = parser.parse_args()

    try:
        import importlib
        mod = importlib.import_module(args.module)
        cls = getattr(mod, args.attr)
    except (ImportError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    program = cls()
    try:
        program.load(str(args.program))
    except Exception as exc:
        print(f"ERROR: Failed to load program from '{args.program}': {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        dataset = load_dataset(args.dataset, args.input_col, args.label_col)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating on {len(dataset)} examples...")
    metrics = evaluate(program, dataset)
    print(f"Accuracy: {metrics['accuracy']:.1%} ({int(metrics['accuracy'] * metrics['total'])}/{int(metrics['total'])})")


if __name__ == "__main__":
    main()
