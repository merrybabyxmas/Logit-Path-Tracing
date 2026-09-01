from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_NAMES = {
    "qwen25-3b": "Qwen2.5-3B",
    "qwen25-1.5b": "Qwen2.5-1.5B",
    "smollm2-1.7b": "SmolLM2-1.7B",
}

MODEL_ORDER = ["qwen25-3b", "qwen25-1.5b", "smollm2-1.7b"]
METHOD_ORDER = ["lpt", "uniform", "adaptive"]

EXPECTED_TABLE1 = {
    ("A", 1, "qwen25-3b", "lpt"): (388.8, 100.0, 100.0, 71.2),
    ("A", 1, "qwen25-3b", "uniform"): (1515.7, 87.3, 96.7, 75.8),
    ("A", 1, "qwen25-3b", "adaptive"): (1518.3, 87.9, 97.1, 75.7),
    ("A", 1, "qwen25-1.5b", "lpt"): (648.5, 100.0, 100.0, 25.7),
    ("A", 1, "qwen25-1.5b", "uniform"): (750.3, 63.8, 79.6, 29.0),
    ("A", 1, "qwen25-1.5b", "adaptive"): (749.9, 63.7, 79.5, 28.9),
    ("A", 1, "smollm2-1.7b", "lpt"): (772.4, 100.0, 100.0, 38.9),
    ("A", 1, "smollm2-1.7b", "uniform"): (1296.3, 69.4, 87.7, 39.9),
    ("A", 1, "smollm2-1.7b", "adaptive"): (1296.1, 69.7, 87.8, 39.7),
    ("A", 2, "qwen25-3b", "lpt"): (4920.3, 100.0, 100.0, 171.2),
    ("A", 2, "qwen25-3b", "uniform"): (10104.9, 45.0, 86.1, 172.7),
    ("A", 2, "qwen25-3b", "adaptive"): (10184.5, 49.1, 85.8, 172.9),
    ("A", 2, "qwen25-1.5b", "lpt"): (1580.5, 100.0, 100.0, 70.8),
    ("A", 2, "qwen25-1.5b", "uniform"): (8076.9, 71.0, 95.4, 71.5),
    ("A", 2, "qwen25-1.5b", "adaptive"): (8119.4, 73.9, 94.5, 71.4),
    ("A", 2, "smollm2-1.7b", "lpt"): (2402.5, 100.0, 100.0, 44.4),
    ("A", 2, "smollm2-1.7b", "uniform"): (4631.8, 54.0, 87.3, 45.3),
    ("A", 2, "smollm2-1.7b", "adaptive"): (4666.1, 56.9, 86.3, 45.3),
    ("B", 1, "qwen25-3b", "lpt"): (388.8, 100.0, 100.0, 71.2),
    ("B", 1, "qwen25-3b", "uniform"): (367.5, 63.1, 73.6, 18.3),
    ("B", 1, "qwen25-3b", "adaptive"): (367.5, 63.1, 73.6, 18.3),
    ("B", 1, "qwen25-1.5b", "lpt"): (648.5, 100.0, 100.0, 25.7),
    ("B", 1, "qwen25-1.5b", "uniform"): (612.8, 59.7, 75.4, 24.1),
    ("B", 1, "qwen25-1.5b", "adaptive"): (613.4, 59.7, 75.6, 24.0),
    ("B", 1, "smollm2-1.7b", "lpt"): (772.4, 100.0, 100.0, 38.9),
    ("B", 1, "smollm2-1.7b", "uniform"): (754.5, 56.8, 74.2, 23.1),
    ("B", 1, "smollm2-1.7b", "adaptive"): (754.3, 56.7, 74.1, 22.9),
    ("B", 2, "qwen25-3b", "lpt"): (4920.3, 100.0, 100.0, 171.2),
    ("B", 2, "qwen25-3b", "uniform"): (4808.3, 33.9, 73.4, 83.1),
    ("B", 2, "qwen25-3b", "adaptive"): (4812.4, 36.6, 70.2, 82.8),
    ("B", 2, "qwen25-1.5b", "lpt"): (1580.5, 100.0, 100.0, 70.8),
    ("B", 2, "qwen25-1.5b", "uniform"): (1488.5, 50.2, 75.2, 14.6),
    ("B", 2, "qwen25-1.5b", "adaptive"): (1486.5, 52.6, 74.4, 14.7),
    ("B", 2, "smollm2-1.7b", "lpt"): (2402.5, 100.0, 100.0, 44.4),
    ("B", 2, "smollm2-1.7b", "uniform"): (2294.5, 43.9, 76.7, 22.8),
    ("B", 2, "smollm2-1.7b", "adaptive"): (2294.4, 45.8, 72.8, 22.6),
}

EXPECTED_TABLE1_REGIONS = {
    ("A", 1, "qwen25-3b", "lpt"): 5.23,
    ("A", 1, "qwen25-3b", "uniform"): 4.195,
    ("A", 1, "qwen25-3b", "adaptive"): 4.25,
    ("A", 1, "qwen25-1.5b", "lpt"): 8.975,
    ("A", 1, "qwen25-1.5b", "uniform"): 4.79,
    ("A", 1, "qwen25-1.5b", "adaptive"): 4.785,
    ("A", 1, "smollm2-1.7b", "lpt"): 10.36,
    ("A", 1, "smollm2-1.7b", "uniform"): 6.265,
    ("A", 1, "smollm2-1.7b", "adaptive"): 6.28,
    ("A", 2, "qwen25-3b", "lpt"): 68.3,
    ("A", 2, "qwen25-3b", "uniform"): 23.87,
    ("A", 2, "qwen25-3b", "adaptive"): 26.32,
    ("A", 2, "qwen25-1.5b", "lpt"): 18.815,
    ("A", 2, "qwen25-1.5b", "uniform"): 10.235,
    ("A", 2, "qwen25-1.5b", "adaptive"): 10.74,
    ("A", 2, "smollm2-1.7b", "lpt"): 26.71,
    ("A", 2, "smollm2-1.7b", "uniform"): 10.62,
    ("A", 2, "smollm2-1.7b", "adaptive"): 11.335,
    ("B", 1, "qwen25-3b", "lpt"): 5.23,
    ("B", 1, "qwen25-3b", "uniform"): 2.495,
    ("B", 1, "qwen25-3b", "adaptive"): 2.495,
    ("B", 1, "qwen25-1.5b", "lpt"): 8.975,
    ("B", 1, "qwen25-1.5b", "uniform"): 4.36,
    ("B", 1, "qwen25-1.5b", "adaptive"): 4.365,
    ("B", 1, "smollm2-1.7b", "lpt"): 10.36,
    ("B", 1, "smollm2-1.7b", "uniform"): 4.775,
    ("B", 1, "smollm2-1.7b", "adaptive"): 4.765,
    ("B", 2, "qwen25-3b", "lpt"): 68.3,
    ("B", 2, "qwen25-3b", "uniform"): 16.09,
    ("B", 2, "qwen25-3b", "adaptive"): 17.44,
    ("B", 2, "qwen25-1.5b", "lpt"): 18.815,
    ("B", 2, "qwen25-1.5b", "uniform"): 5.095,
    ("B", 2, "qwen25-1.5b", "adaptive"): 5.53,
    ("B", 2, "smollm2-1.7b", "lpt"): 26.71,
    ("B", 2, "smollm2-1.7b", "uniform"): 7.73,
    ("B", 2, "smollm2-1.7b", "adaptive"): 8.085,
}

EPS_COLUMNS = [0.0, 0.005, 0.01, 0.05, 0.25, 1.0]
EXPECTED_EPSILON = {
    (1, "qwen25-3b"): [100.0, 28.5, 13.9, 4.3, 1.0, 0.0],
    (1, "qwen25-1.5b"): [100.0, 64.6, 43.8, 7.6, 0.9, 0.5],
    (1, "smollm2-1.7b"): [100.0, 39.9, 19.1, 2.3, 0.0, 0.0],
    (2, "qwen25-3b"): [100.0, 24.6, 12.1, 4.0, 0.7, 0.0],
    (2, "qwen25-1.5b"): [100.0, 78.1, 63.0, 28.6, 14.3, 0.5],
    (2, "smollm2-1.7b"): [100.0, 63.4, 41.9, 8.0, 1.6, 0.0],
}

EXPECTED_SAMPLING_POINTS = {
    (1, "qwen25-3b"): (5.23, 16.44, 16.50, 3.76, 3.76),
    (1, "qwen25-1.5b"): (8.98, 7.30, 7.27, 5.94, 5.94),
    (1, "smollm2-1.7b"): (10.36, 11.60, 11.60, 6.67, 6.66),
    (2, "qwen25-3b"): (68.30, 77.345, 74.60, 34.94, 34.22),
    (2, "qwen25-1.5b"): (18.82, 94.99, 94.42, 15.61, 15.36),
    (2, "smollm2-1.7b"): (26.71, 35.57, 35.57, 16.59, 16.39),
}

EXPECTED_DENSE_GRID = {
    ("qwen25-3b", 2): (30, 1326, 218262.3, 3959.3, 55.1, 0, 89.3, 100.0),
    ("qwen25-1.5b", 2): (30, 1326, 221089.7, 1491.0, 148.3, 0, 94.9, 100.0),
    ("smollm2-1.7b", 2): (30, 1326, 225678.5, 1738.8, 129.8, 0, 95.3, 100.0),
}

EXPECTED_SYNTHETIC = {
    ("generic", 1): 3.72,
    ("generic", 2): 8.20,
    ("generic", 3): 14.60,
    ("generic", 4): 22.80,
    ("tie_stress", 1): 3.56,
    ("tie_stress", 2): 8.02,
    ("tie_stress", 3): 14.08,
    ("tie_stress", 4): 25.04,
}

EXPECTED_BATCH_EHAT = {
    ("qwen25-3b", 1): 0.0,
    ("qwen25-3b", 2): 0.0,
    ("qwen25-1.5b", 1): 0.5,
    ("qwen25-1.5b", 2): 0.5,
    ("smollm2-1.7b", 1): 0.0,
    ("smollm2-1.7b", 2): 0.0,
}

EXPECTED_PROPERTY_EXISTENCE = {
    (1, "qwen25-3b"): (68, 62, 91.2, 62, 91.2),
    (1, "qwen25-1.5b"): (68, 56, 82.4, 56, 82.4),
    (1, "smollm2-1.7b"): (55, 46, 83.6, 46, 83.6),
    (2, "qwen25-3b"): (94, 89, 94.7, 90, 95.7),
    (2, "qwen25-1.5b"): (62, 51, 82.3, 51, 82.3),
    (2, "smollm2-1.7b"): (74, 66, 89.2, 65, 87.8),
}


def parse_run_id(run_id: str) -> dict[str, Any]:
    model = next((key for key in MODEL_NAMES if f"__{key}__" in run_id), None)
    d_match = re.search(r"__d(\d)__", run_id)
    method_match = re.search(r"__method-(.*?)__budget-", run_id)
    k_match = re.search(r"__k-([^_]+)__", run_id)
    return {
        "model_key": model,
        "d": int(d_match.group(1)) if d_match else None,
        "method": method_match.group(1) if method_match else None,
        "k": k_match.group(1) if k_match else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_paths(root: Path, stage: str) -> list[Path]:
    return sorted((root / "runs" / stage).glob("*/metrics_summary.json"))


def pct(value: float) -> float:
    return 100.0 * float(value)


def r1(value: float) -> float:
    return round(float(value) + 1e-12, 1)


def verify_rows(name: str, rows: list[dict[str, Any]], expected_key: str = "expected") -> dict[str, Any]:
    failures = []
    for row in rows:
        for column in row.get(expected_key, {}):
            got = r1(row[column])
            exp = r1(row[expected_key][column])
            if got != exp:
                failures.append(
                    {
                        "table": name,
                        "id": row.get("id"),
                        "column": column,
                        "got": got,
                        "expected": exp,
                    }
                )
    return {"table": name, "rows": len(rows), "failures": failures}


def table1(root: Path) -> pd.DataFrame:
    lpt: dict[tuple[int, str], dict[str, float]] = {}
    for path in metric_paths(root, "main"):
        metric = load_json(path)
        meta = parse_run_id(metric["run_id"])
        if meta["method"] == "lpt" and "__eps-0__" in metric["run_id"]:
            lpt[(meta["d"], meta["model_key"])] = {
                "regions": metric["num_cells_mean"],
                "model_evals": metric["model_evals_per_prompt_mean"],
                "recall": pct(metric["output_region_recall"]),
                "coverage": pct(metric["decision_coverage"]),
                "time": metric["time_per_prompt_mean_sec"],
            }

    rows = []
    for panel, stage in [("A", "fixed_time_sampling"), ("B", "equal_eval_sampling")]:
        for d in [1, 2]:
            for model_key in MODEL_ORDER:
                base = lpt[(d, model_key)]
                rows.append(
                    {
                        "panel": panel,
                        "d": d,
                        "model_key": model_key,
                        "model": MODEL_NAMES[model_key],
                        "method": "lpt",
                        **base,
                    }
                )
        for path in metric_paths(root, stage):
            metric = load_json(path)
            prompt_metrics = pd.read_parquet(path.parent / "prompt_metrics.parquet")
            regions = (
                prompt_metrics["num_reference_cells"]
                * prompt_metrics["output_region_recall"]
            ).mean()
            rows.append(
                {
                    "panel": panel,
                    "d": int(metric["d"]),
                    "model_key": metric["model_key"],
                    "model": metric["model"],
                    "method": metric["method"],
                    "regions": regions,
                    "model_evals": metric["actual_model_evals_per_prompt_mean"],
                    "recall": pct(metric["output_region_recall"]),
                    "coverage": pct(metric["decision_coverage"]),
                    "time": metric["time_per_prompt_mean_sec"],
                }
            )
    df = pd.DataFrame(rows)
    df["method_rank"] = df["method"].map({method: i for i, method in enumerate(METHOD_ORDER)})
    df["model_rank"] = df["model_key"].map({model: i for i, model in enumerate(MODEL_ORDER)})
    return df.sort_values(["panel", "d", "model_rank", "method_rank"]).drop(columns=["model_rank", "method_rank"])


def epsilon_table(root: Path) -> pd.DataFrame:
    candidates = [
        root / "analysis" / "figures" / "epsilon_subregion" / "fig04_epsilon_tolerance.csv",
        root / "data" / "processed" / "fig04_epsilon_tolerance.csv",
        root / "analysis" / "figures" / "epsilon_tolerance_dense" / "epsilon_tolerance_dense_raw.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("could not locate epsilon tolerance summary CSV")
    df = pd.read_csv(source)
    rows = []
    for d in [1, 2]:
        for model_key in MODEL_ORDER:
            sub = df[(df["d"] == d) & (df["model_key"] == model_key)]
            row = {"d": d, "model_key": model_key, "model": MODEL_NAMES[model_key]}
            for alpha in EPS_COLUMNS:
                hit = sub[(sub["alpha_eps_over_epshat"] - alpha).abs() < 1e-12]
                if hit.empty:
                    raise ValueError(f"Missing epsilon alpha {alpha} for {model_key}, d={d}")
                row[f"coverage_alpha_{alpha:g}"] = pct(hit.iloc[0]["decision_coverage"])
            rows.append(row)
    return pd.DataFrame(rows)


def synthetic_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in metric_paths(root, "synthetic"):
        metric = load_json(path)
        meta = parse_run_id(metric["run_id"])
        method = meta["method"]
        if method == "tie":
            method = "tie_stress"
        rows.append(
            {
                "family": method,
                "d": meta["d"],
                "argmax_cells_mean": metric["argmax_cells_mean"],
                "trials": metric["trials"],
                "vocab_size": metric["vocab_size"],
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "d"])


def batch_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in metric_paths(root, "batch_precision"):
        metric = load_json(path)
        meta = parse_run_id(metric["run_id"])
        rows.append(
            {
                "model_key": meta["model_key"],
                "model": MODEL_NAMES[meta["model_key"]],
                "d": meta["d"],
                "method": meta["method"],
                "recall": pct(metric["output_region_recall"]),
                "coverage": pct(metric["decision_coverage"]),
            }
        )
    df = pd.DataFrame(rows)
    out = []
    for d in [1, 2]:
        for model_key in MODEL_ORDER:
            sub = df[(df["d"] == d) & (df["model_key"] == model_key)]
            nominal = sub[sub["method"] == "nominal"]
            robust = sub[sub["method"] == "robust"]
            out.append(
                {
                    "model_key": model_key,
                    "model": MODEL_NAMES[model_key],
                    "d": d,
                    "min_region_recall": nominal["recall"].min(),
                    "min_domain_coverage": nominal["coverage"].min(),
                    "coverage_at_ehat": robust["coverage"].max(),
                    "nominal_runs": int(len(nominal)),
                    "robust_runs": int(len(robust)),
                }
            )
    return pd.DataFrame(out)


def candidate_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in metric_paths(root, "candidate_invariance"):
        metric = load_json(path)
        meta = parse_run_id(metric["run_id"])
        rows.append(
            {
                "model_key": meta["model_key"],
                "model": MODEL_NAMES[meta["model_key"]],
                "d": meta["d"],
                "k": int(meta["k"]),
                "recall": pct(metric["output_region_recall"]),
                "coverage": pct(metric["decision_coverage"]),
            }
        )
    df = pd.DataFrame(rows)
    out = []
    for d in [1, 2]:
        for model_key in MODEL_ORDER:
            sub = df[(df["d"] == d) & (df["model_key"] == model_key)]
            out.append(
                {
                    "model_key": model_key,
                    "model": MODEL_NAMES[model_key],
                    "d": d,
                    "candidate_sizes": ", ".join(map(str, sorted(sub["k"].unique()))),
                    "min_recall": sub["recall"].min(),
                    "min_coverage": sub["coverage"].min(),
                    "runs": int(len(sub)),
                }
            )
    return pd.DataFrame(out)


def sampling_points_table(root: Path) -> pd.DataFrame:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for path in metric_paths(root, "main"):
        metric = load_json(path)
        meta = parse_run_id(metric["run_id"])
        if meta["method"] == "lpt" and "__eps-0__" in metric["run_id"]:
            key = (meta["d"], meta["model_key"])
            rows[key] = {
                "d": meta["d"],
                "model_key": meta["model_key"],
                "model": MODEL_NAMES[meta["model_key"]],
                "lpt_cells_per_prompt": metric["num_cells_mean"],
            }

    stage_columns = {
        "fixed_time_sampling": "panel_a",
        "equal_eval_sampling": "panel_b",
    }
    for stage, prefix in stage_columns.items():
        for path in metric_paths(root, stage):
            metric = load_json(path)
            key = (int(metric["d"]), metric["model_key"])
            method = metric["method"]
            rows[key][f"{prefix}_{method}_npts"] = metric[
                "sampled_parameter_points_per_prompt_mean"
            ]

    out = []
    for d in [1, 2]:
        for model_key in MODEL_ORDER:
            out.append(rows[(d, model_key)])
    return pd.DataFrame(out)


def dense_grid_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in metric_paths(root, "densegrid"):
        metric = load_json(path)
        meta = parse_run_id(metric["run_id"])
        prompt_metrics = pd.read_parquet(path.parent / "prompt_metrics.parquet")
        lpt_path = next(
            (root / "runs" / "main").glob(
                f"*{meta['model_key']}__d{meta['d']}__domain-simplex__method-lpt__budget-na__eps-0*"
            )
        )
        lpt_prompt_metrics = pd.read_parquet(lpt_path / "prompt_metrics.parquet")
        dense_prompts = set(prompt_metrics["prompt_id"])
        lpt_same_prompts = lpt_prompt_metrics[
            lpt_prompt_metrics["prompt_id"].isin(dense_prompts)
        ]
        dense_evals = prompt_metrics["model_evals"].mean()
        lpt_evals = lpt_same_prompts["model_evals"].mean()
        rows.append(
            {
                "model_key": meta["model_key"],
                "model": MODEL_NAMES[meta["model_key"]],
                "d": meta["d"],
                "prompts": metric["num_prompts"],
                "grid_points_per_prompt": prompt_metrics["dense_points"].mean(),
                "dense_model_evals_per_prompt": dense_evals,
                "lpt_model_evals_same_prompts": lpt_evals,
                "dense_to_lpt_eval_ratio": dense_evals / lpt_evals,
                "sample_mismatches": prompt_metrics["mismatches"].sum(),
                "output_region_recall": pct(metric["output_region_recall"]),
                "decision_coverage": pct(metric["decision_coverage"]),
            }
        )
    df = pd.DataFrame(rows)
    df["model_rank"] = df["model_key"].map({model: i for i, model in enumerate(MODEL_ORDER)})
    return df.sort_values(["d", "model_rank"]).drop(columns=["model_rank"])


def property_existence_table(root: Path) -> pd.DataFrame:
    candidates = [
        root / "analysis" / "property_existence" / "property_existence_summary.csv",
        root / "data" / "processed" / "table08_property_existence.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("could not locate property existence summary CSV")
    df = pd.read_csv(source)
    rows = []
    for d in [1, 2]:
        for model_key in MODEL_ORDER:
            sub = df[
                (df["panel"] == "B")
                & (df["d"] == d)
                & (df["model_key"] == model_key)
                & (df["prompt_family"] == "safety")
            ]
            row = {
                "d": d,
                "model_key": model_key,
                "model": MODEL_NAMES[model_key],
            }
            for method in ["uniform", "adaptive"]:
                hit = sub[sub["method"] == method]
                if len(hit) != 1:
                    raise ValueError(f"Missing property existence row for {model_key}, d={d}, {method}")
                metric = hit.iloc[0]
                row["lpt_positive_prompts"] = int(metric["lpt_positive_prompts"])
                row[f"{method}_detected"] = int(metric["detected_positive_prompts"])
                row[f"{method}_recall"] = pct(metric["existence_recall"])
            rows.append(row)
    return pd.DataFrame(rows)


def verify_table1(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in df.to_dict("records"):
        key = (row["panel"], int(row["d"]), row["model_key"], row["method"])
        expected = EXPECTED_TABLE1[key]
        rows.append(
            {
                "id": "|".join(map(str, key)),
                "regions": row["regions"],
                "model_evals": row["model_evals"],
                "recall": row["recall"],
                "coverage": row["coverage"],
                "time": row["time"],
                "expected": {
                    "regions": EXPECTED_TABLE1_REGIONS[key],
                    "model_evals": expected[0],
                    "recall": expected[1],
                    "coverage": expected[2],
                    "time": expected[3],
                },
            }
        )
    return verify_rows("table1_main", rows)


def verify_epsilon(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in df.to_dict("records"):
        key = (int(row["d"]), row["model_key"])
        expected_values = EXPECTED_EPSILON[key]
        expected = {f"coverage_alpha_{alpha:g}": value for alpha, value in zip(EPS_COLUMNS, expected_values)}
        rows.append({"id": "|".join(map(str, key)), **row, "expected": expected})
    return verify_rows("table2_robust_epsilon", rows)


def verify_synthetic(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in df.to_dict("records"):
        key = (row["family"], int(row["d"]))
        rows.append(
            {
                "id": "|".join(map(str, key)),
                "argmax_cells_mean": row["argmax_cells_mean"],
                "expected": {"argmax_cells_mean": EXPECTED_SYNTHETIC[key]},
            }
        )
    return verify_rows("appendix_synthetic", rows)


def verify_batch(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in df.to_dict("records"):
        key = (row["model_key"], int(row["d"]))
        rows.append(
            {
                "id": "|".join(map(str, key)),
                "min_region_recall": row["min_region_recall"],
                "min_domain_coverage": row["min_domain_coverage"],
                "coverage_at_ehat": row["coverage_at_ehat"],
                "expected": {
                    "min_region_recall": 100.0,
                    "min_domain_coverage": 100.0,
                    "coverage_at_ehat": EXPECTED_BATCH_EHAT[key],
                },
            }
        )
    return verify_rows("appendix_batch_precision", rows)


def verify_candidate(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in df.to_dict("records"):
        rows.append(
            {
                "id": f"{row['model_key']}|{row['d']}",
                "min_recall": row["min_recall"],
                "min_coverage": row["min_coverage"],
                "expected": {"min_recall": 100.0, "min_coverage": 100.0},
            }
        )
    return verify_rows("appendix_candidate_invariance", rows)


def verify_sampling_points(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in df.to_dict("records"):
        key = (int(row["d"]), row["model_key"])
        expected = EXPECTED_SAMPLING_POINTS[key]
        rows.append(
            {
                "id": "|".join(map(str, key)),
                "lpt_cells_per_prompt": row["lpt_cells_per_prompt"],
                "panel_a_uniform_npts": row["panel_a_uniform_npts"],
                "panel_a_adaptive_npts": row["panel_a_adaptive_npts"],
                "panel_b_uniform_npts": row["panel_b_uniform_npts"],
                "panel_b_adaptive_npts": row["panel_b_adaptive_npts"],
                "expected": {
                    "lpt_cells_per_prompt": expected[0],
                    "panel_a_uniform_npts": expected[1],
                    "panel_a_adaptive_npts": expected[2],
                    "panel_b_uniform_npts": expected[3],
                    "panel_b_adaptive_npts": expected[4],
                },
            }
        )
    return verify_rows("table_sampling_points", rows)


def verify_dense_grid(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in df.to_dict("records"):
        key = (row["model_key"], int(row["d"]))
        expected = EXPECTED_DENSE_GRID[key]
        rows.append(
            {
                "id": "|".join(map(str, key)),
                "prompts": row["prompts"],
                "grid_points_per_prompt": row["grid_points_per_prompt"],
                "dense_model_evals_per_prompt": row["dense_model_evals_per_prompt"],
                "lpt_model_evals_same_prompts": row["lpt_model_evals_same_prompts"],
                "dense_to_lpt_eval_ratio": row["dense_to_lpt_eval_ratio"],
                "sample_mismatches": row["sample_mismatches"],
                "output_region_recall": row["output_region_recall"],
                "decision_coverage": row["decision_coverage"],
                "expected": {
                    "prompts": expected[0],
                    "grid_points_per_prompt": expected[1],
                    "dense_model_evals_per_prompt": expected[2],
                    "lpt_model_evals_same_prompts": expected[3],
                    "dense_to_lpt_eval_ratio": expected[4],
                    "sample_mismatches": expected[5],
                    "output_region_recall": expected[6],
                    "decision_coverage": expected[7],
                },
            }
        )
    return verify_rows("appendix_dense_grid_spotcheck", rows)


def verify_property_existence(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in df.to_dict("records"):
        key = (int(row["d"]), row["model_key"])
        expected = EXPECTED_PROPERTY_EXISTENCE[key]
        rows.append(
            {
                "id": "|".join(map(str, key)),
                "lpt_positive_prompts": row["lpt_positive_prompts"],
                "uniform_detected": row["uniform_detected"],
                "uniform_recall": row["uniform_recall"],
                "adaptive_detected": row["adaptive_detected"],
                "adaptive_recall": row["adaptive_recall"],
                "expected": {
                    "lpt_positive_prompts": expected[0],
                    "uniform_detected": expected[1],
                    "uniform_recall": expected[2],
                    "adaptive_detected": expected[3],
                    "adaptive_recall": expected[4],
                },
            }
        )
    return verify_rows("appendix_property_existence", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/reproducibility"))
    parser.add_argument(
        "--include-local-paths",
        action="store_true",
        help="Record resolved local paths in verification_summary.json.",
    )
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "table01_main_recomputed.csv": table1(root),
        "table02_epsilon_recomputed.csv": epsilon_table(root),
        "table03_synthetic_recomputed.csv": synthetic_table(root),
        "table04_batch_precision_recomputed.csv": batch_table(root),
        "table05_candidate_invariance_recomputed.csv": candidate_table(root),
        "table06_sampling_points_recomputed.csv": sampling_points_table(root),
        "table08_dense_grid_spotcheck_recomputed.csv": dense_grid_table(root),
        "table09_property_existence_recomputed.csv": property_existence_table(root),
    }
    for name, df in tables.items():
        df.to_csv(out_dir / name, index=False)

    checks = [
        verify_table1(tables["table01_main_recomputed.csv"]),
        verify_epsilon(tables["table02_epsilon_recomputed.csv"]),
        verify_synthetic(tables["table03_synthetic_recomputed.csv"]),
        verify_batch(tables["table04_batch_precision_recomputed.csv"]),
        verify_candidate(tables["table05_candidate_invariance_recomputed.csv"]),
        verify_sampling_points(tables["table06_sampling_points_recomputed.csv"]),
        verify_dense_grid(tables["table08_dense_grid_spotcheck_recomputed.csv"]),
        verify_property_existence(tables["table09_property_existence_recomputed.csv"]),
    ]
    failures = [failure for check in checks for failure in check["failures"]]
    summary = {
        "artifact_root_name": root.name,
        "output_dir_name": out_dir.name,
        "checks": checks,
        "num_failures": len(failures),
        "status": "passed" if not failures else "failed",
    }
    if args.include_local_paths:
        summary["artifact_root"] = str(root)
        summary["output_dir"] = str(out_dir)
    (out_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
