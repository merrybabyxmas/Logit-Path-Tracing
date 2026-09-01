from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpt_core import certify_cells, pairwise_gap_epsilon, partition_interval, partition_simplex, robust_subregions


def contains_interval(cell, point: float, tol: float = 1e-9) -> bool:
    left, right = cell.vertices[0][0], cell.vertices[1][0]
    return left - tol <= point <= right + tol


def contains_simplex_cell(cell, point: np.ndarray, tol: float = 1e-9) -> bool:
    poly = np.array(cell.vertices, dtype=np.float64)
    signs = []
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        edge = b - a
        rel = point - a
        signs.append(edge[0] * rel[1] - edge[1] * rel[0])
    return all(s >= -tol for s in signs) or all(s <= tol for s in signs)


def smoke_interval() -> dict:
    a = np.array([0.15, -0.35, 0.55, -0.10, 0.25])
    b = np.array([0.10, 1.80, -1.10, 0.35, -0.55])
    cells = partition_interval(a, b)
    mismatches = 0
    checked = 0
    for s in np.linspace(0.001, 0.999, 500):
        direct = int(np.argmax(a + s * b))
        recovered = next((cell.token for cell in cells if contains_interval(cell, float(s))), None)
        checked += 1
        mismatches += int(direct != recovered)
    eps = pairwise_gap_epsilon(a + b, a + b + np.array([0.001, -0.002, 0.0005, 0.0, 0.003]))
    return {
        "dimension": 1,
        "num_cells": len(cells),
        "num_certified_at_eps_1e-3": len(certify_cells(cells, 1e-3)),
        "num_robust_subregions_at_eps_1e-3": len(robust_subregions(cells, a, b, 1e-3)),
        "pairwise_gap_epsilon_example": eps,
        "grid_points_checked": checked,
        "grid_mismatches": mismatches,
    }


def smoke_simplex() -> dict:
    a = np.array([0.30, -0.20, 0.10, 0.45, -0.10, 0.00])
    b = np.array(
        [
            [0.25, -0.85],
            [1.10, 0.05],
            [-0.70, 0.95],
            [-0.25, -0.15],
            [0.50, 0.55],
            [-0.40, 0.35],
        ],
        dtype=np.float64,
    )
    cells = partition_simplex(a, b)
    mismatches = 0
    checked = 0
    for x in np.linspace(0.02, 0.96, 32):
        for y in np.linspace(0.02, 0.96, 32):
            point = np.array([x, y])
            if x + y >= 0.98:
                continue
            direct = int(np.argmax(a + b @ point))
            recovered = next((cell.token for cell in cells if contains_simplex_cell(cell, point)), None)
            checked += 1
            mismatches += int(direct != recovered)
    return {
        "dimension": 2,
        "num_cells": len(cells),
        "num_certified_at_eps_1e-3": len(certify_cells(cells, 1e-3)),
        "num_robust_subregions_at_eps_1e-3": len(robust_subregions(cells, a, b, 1e-3)),
        "grid_points_checked": checked,
        "grid_mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"interval": smoke_interval(), "simplex": smoke_simplex()}
    summary["status"] = "passed" if all(v["grid_mismatches"] == 0 for v in summary.values()) else "failed"
    (args.out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
