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
    ("A", 1, "qwen25-3b", "uniform"): (1522.0, 32.5, 34.0, 75.3),
    ("A", 1, "qwen25-3b", "adaptive"): (1522.3, 32.5, 34.0, 75.3),
    ("A", 1, "qwen25-1.5b", "lpt"): (648.5, 100.0, 100.0, 25.7),
    ("A", 1, "qwen25-1.5b", "uniform"): (760.9, 18.1, 16.0, 29.7),
    ("A", 1, "qwen25-1.5b", "adaptive"): (761.0, 18.1, 16.0, 29.4),
    ("A", 1, "smollm2-1.7b", "lpt"): (772.4, 100.0, 100.0, 38.9),
    ("A", 1, "smollm2-1.7b", "uniform"): (1297.1, 15.1, 14.0, 39.8),
    ("A", 1, "smollm2-1.7b", "adaptive"): (1291.2, 15.1, 14.0, 39.6),
    ("A", 2, "qwen25-3b", "lpt"): (4920.3, 100.0, 100.0, 171.2),
    ("A", 2, "qwen25-3b", "uniform"): (9962.5, 15.3, 20.0, 172.7),
    ("A", 2, "qwen25-3b", "adaptive"): (10046.3, 14.9, 19.3, 172.8),
    ("A", 2, "qwen25-1.5b", "lpt"): (1580.5, 100.0, 100.0, 70.8),
    ("A", 2, "qwen25-1.5b", "uniform"): (8020.9, 35.2, 42.3, 71.5),
    ("A", 2, "qwen25-1.5b", "adaptive"): (8084.2, 33.3, 39.9, 71.5),
    ("A", 2, "smollm2-1.7b", "lpt"): (2402.5, 100.0, 100.0, 44.4),
    ("A", 2, "smollm2-1.7b", "uniform"): (4615.4, 18.6, 26.9, 45.3),
    ("A", 2, "smollm2-1.7b", "adaptive"): (4639.6, 16.4, 22.9, 45.2),
    ("B", 1, "qwen25-3b", "lpt"): (388.8, 100.0, 100.0, 71.2),
    ("B", 1, "qwen25-3b", "uniform"): (368.0, 32.7, 34.2, 18.3),
    ("B", 1, "qwen25-3b", "adaptive"): (368.0, 32.6, 34.0, 18.3),
    ("B", 1, "qwen25-1.5b", "lpt"): (648.5, 100.0, 100.0, 25.7),
    ("B", 1, "qwen25-1.5b", "uniform"): (631.6, 19.1, 16.8, 24.2),
    ("B", 1, "qwen25-1.5b", "adaptive"): (631.6, 18.7, 16.5, 24.1),
    ("B", 1, "smollm2-1.7b", "lpt"): (772.4, 100.0, 100.0, 38.9),
    ("B", 1, "smollm2-1.7b", "uniform"): (761.8, 16.2, 15.8, 28.0),
    ("B", 1, "smollm2-1.7b", "adaptive"): (761.3, 15.6, 14.9, 28.0),
    ("B", 2, "qwen25-3b", "lpt"): (4920.3, 100.0, 100.0, 171.2),
    ("B", 2, "qwen25-3b", "uniform"): (4798.5, 12.6, 15.9, 84.6),
    ("B", 2, "qwen25-3b", "adaptive"): (4796.2, 11.8, 14.4, 84.2),
    ("B", 2, "qwen25-1.5b", "lpt"): (1580.5, 100.0, 100.0, 70.8),
    ("B", 2, "qwen25-1.5b", "uniform"): (1500.5, 30.1, 35.3, 13.7),
    ("B", 2, "qwen25-1.5b", "adaptive"): (1499.7, 29.9, 34.9, 13.7),
    ("B", 2, "smollm2-1.7b", "lpt"): (2402.5, 100.0, 100.0, 44.4),
    ("B", 2, "smollm2-1.7b", "uniform"): (2296.7, 19.4, 27.0, 23.6),
    ("B", 2, "smollm2-1.7b", "adaptive"): (2296.3, 20.4, 29.9, 23.5),
}

EPS_COLUMNS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25]
EXPECTED_EPSILON = {
    (1, "qwen25-3b"): [100.0, 36.1, 16.0, 7.7, 4.5, 2.5, 1.0],
    (1, "qwen25-1.5b"): [100.0, 79.9, 57.0, 29.7, 9.5, 4.4, 1.1],
    (1, "smollm2-1.7b"): [100.0, 54.8, 26.5, 8.5, 2.8, 1.6, 0.0],
    (2, "qwen25-3b"): [100.0, 37.5, 15.9, 8.7, 4.2, 2.1, 1.0],
    (2, "qwen25-1.5b"): [100.0, 93.2, 79.8, 56.9, 32.2, 23.2, 15.2],
    (2, "smollm2-1.7b"): [100.0, 86.1, 61.1, 32.0, 9.2, 4.2, 1.5],
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
            rows.append(
                {
                    "panel": panel,
                    "d": int(metric["d"]),
                    "model_key": metric["model_key"],
                    "model": metric["model"],
                    "method": metric["method"],
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
    source = root / "analysis" / "figures" / "epsilon_tolerance_dense" / "epsilon_tolerance_dense_raw.csv"
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


def verify_table1(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in df.to_dict("records"):
        key = (row["panel"], int(row["d"]), row["model_key"], row["method"])
        expected = EXPECTED_TABLE1[key]
        rows.append(
            {
                "id": "|".join(map(str, key)),
                "model_evals": row["model_evals"],
                "recall": row["recall"],
                "coverage": row["coverage"],
                "time": row["time"],
                "expected": {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/reproducibility"))
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
    }
    for name, df in tables.items():
        df.to_csv(out_dir / name, index=False)

    checks = [
        verify_table1(tables["table01_main_recomputed.csv"]),
        verify_epsilon(tables["table02_epsilon_recomputed.csv"]),
        verify_synthetic(tables["table03_synthetic_recomputed.csv"]),
        verify_batch(tables["table04_batch_precision_recomputed.csv"]),
        verify_candidate(tables["table05_candidate_invariance_recomputed.csv"]),
    ]
    failures = [failure for check in checks for failure in check["failures"]]
    summary = {
        "artifact_root": str(root),
        "output_dir": str(out_dir),
        "checks": checks,
        "num_failures": len(failures),
        "status": "passed" if not failures else "failed",
    }
    (out_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
