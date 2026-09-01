from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


MODEL_NAMES = {
    "qwen25-3b": "Qwen2.5-3B",
    "qwen25-1.5b": "Qwen2.5-1.5B",
    "smollm2-1.7b": "SmolLM2-1.7B",
}
MODEL_ORDER = ["qwen25-3b", "qwen25-1.5b", "smollm2-1.7b"]
STAGES = {
    "A": "fixed_time_sampling",
    "B": "equal_eval_sampling",
}
SAFETY_TERM_RE = re.compile(
    r"\b(safe|safety|legal|ethical|harmful|dangerous|illegal|benign|constructive|instead)\b",
    re.IGNORECASE,
)


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {pattern}, found {len(matches)}")
    return matches[0]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def exact_cache_path(root: Path, model_key: str, d: int) -> Path:
    return find_one(
        root / "cache" / "exact",
        f"test200__{model_key}__d{d}__T64__k64__prec-fp32*",
    )


def load_completion_rows(root: Path) -> list[dict] | None:
    path = root / "data" / "processed" / "node_limit_completion.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    prompt_rows: list[dict] = []
    completion_rows: list[dict] = []

    existing_completion_rows = load_completion_rows(root)

    for d in [1, 2]:
        for model_key in MODEL_ORDER:
            lpt_dir = find_one(
                root / "runs" / "main",
                f"main__test200__{model_key}__d{d}__*method-lpt__budget-na__eps-0*",
            )
            cells = pd.read_parquet(lpt_dir / "cell_table.parquet")
            cells = cells[cells["measure"] > 0].copy()
            cells["prompt_family"] = cells["prompt_id"].map(
                lambda prompt_id: "safety" if "safety" in str(prompt_id) else "general"
            )
            cells["property_positive"] = cells["output_text"].fillna("").map(
                lambda text: bool(SAFETY_TERM_RE.search(str(text)))
            )
            property_cells = {
                (row.prompt_id, int(row.cell_id))
                for row in cells[cells["property_positive"]].itertuples()
            }
            positive_by_prompt = cells.groupby("prompt_id")["property_positive"].any()
            safety_prompts = [prompt_id for prompt_id in positive_by_prompt.index if "safety" in prompt_id]
            lpt_positive = {prompt_id for prompt_id in safety_prompts if bool(positive_by_prompt[prompt_id])}

            for panel, stage in STAGES.items():
                for method in ["uniform", "adaptive"]:
                    sample_dir = find_one(
                        root / "runs" / stage,
                        f"*{model_key}__d{d}__method-{method}*",
                    )
                    samples = pd.read_parquet(sample_dir / "sample_points.parquet")
                    detected = set()
                    for row in samples.itertuples():
                        if pd.isna(row.cell_id):
                            continue
                        if (row.prompt_id, int(row.cell_id)) in property_cells:
                            detected.add(row.prompt_id)

                    detected_positive = len(detected & lpt_positive)
                    total_positive = len(lpt_positive)
                    summary_rows.append(
                        {
                            "panel": panel,
                            "d": d,
                            "model_key": model_key,
                            "model": MODEL_NAMES[model_key],
                            "method": method,
                            "property": "safety_term_existence",
                            "prompt_family": "safety",
                            "lpt_positive_prompts": total_positive,
                            "detected_positive_prompts": detected_positive,
                            "missed_positive_prompts": total_positive - detected_positive,
                            "existence_recall": detected_positive / total_positive if total_positive else 1.0,
                        }
                    )
                    for prompt_id in sorted(lpt_positive):
                        prompt_rows.append(
                            {
                                "panel": panel,
                                "d": d,
                                "model_key": model_key,
                                "model": MODEL_NAMES[model_key],
                                "method": method,
                                "property": "safety_term_existence",
                                "prompt_family": "safety",
                                "prompt_id": prompt_id,
                                "lpt_positive": True,
                                "sampling_detected": prompt_id in detected,
                            }
                        )

            if existing_completion_rows is None:
                exact_records = load_jsonl(exact_cache_path(root, model_key, d))[:200]
                completion_rows.append(
                    {
                        "model_key": model_key,
                        "model": MODEL_NAMES[model_key],
                        "d": d,
                        "prompts": int(len(exact_records)),
                        "complete_prompts": int(sum(bool(row.get("certified_complete")) for row in exact_records)),
                        "node_limit_cells": int(
                            sum(
                                1
                                for row in exact_records
                                for response in row.get("responses", [])
                                if response.get("stop_reason") == "node_limit"
                            )
                        ),
                    }
                )

    if existing_completion_rows is not None:
        completion_rows = existing_completion_rows

    pd.DataFrame(summary_rows).sort_values(["panel", "d", "model_key", "method"]).to_csv(
        out_dir / "table08_property_existence.csv", index=False
    )
    pd.DataFrame(prompt_rows).sort_values(["panel", "d", "model_key", "method", "prompt_id"]).to_csv(
        out_dir / "table08_property_existence_prompt_metrics.csv", index=False
    )
    pd.DataFrame(completion_rows).sort_values(["d", "model_key"]).to_csv(
        out_dir / "node_limit_completion.csv", index=False
    )


if __name__ == "__main__":
    main()
